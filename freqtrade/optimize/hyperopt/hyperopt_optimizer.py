"""
此模块包含超参数优化器类，该类需要被序列化并发送到超参数优化工作进程。
"""

import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import optuna
from joblib import delayed, dump, load, wrap_non_picklable_objects
from joblib.externals import cloudpickle
from optuna.exceptions import ExperimentalWarning
from optuna.terminator import BestValueStagnationEvaluator, Terminator
from pandas import DataFrame

from freqtrade.constants import DATETIME_PRINT_FORMAT, Config
from freqtrade.data.converter import trim_dataframes
from freqtrade.data.history import get_timerange
from freqtrade.data.metrics import calculate_market_change
from freqtrade.enums import HyperoptState
from freqtrade.exceptions import OperationalException
from freqtrade.ft_types import BacktestContentType
from freqtrade.misc import deep_merge_dicts, round_dict
from freqtrade.optimize.backtesting import Backtesting

# 导入IHyperOptLoss以允许从这些模块反序列化类
from freqtrade.optimize.hyperopt.hyperopt_auto import HyperOptAuto
from freqtrade.optimize.hyperopt_loss.hyperopt_loss_interface import IHyperOptLoss
from freqtrade.optimize.hyperopt_tools import HyperoptStateContainer, HyperoptTools
from freqtrade.optimize.optimize_reports import generate_strategy_stats
from freqtrade.optimize.space import (
    DimensionProtocol,
    SKDecimal,
    ft_CategoricalDistribution,
    ft_FloatDistribution,
    ft_IntDistribution,
)
from freqtrade.resolvers.hyperopt_resolver import HyperOptLossResolver
from freqtrade.util.dry_run_wallet import get_dry_run_wallet


logger = logging.getLogger(__name__)

INITIAL_POINTS = 30

MAX_LOSS = 100000  # 足够大的数字，在损失优化中表示不良结果

optuna_samplers_dict = {
    "TPESampler": optuna.samplers.TPESampler,
    "GPSampler": optuna.samplers.GPSampler,
    "CmaEsSampler": optuna.samplers.CmaEsSampler,
    "NSGAIISampler": optuna.samplers.NSGAIISampler,
    "NSGAIIISampler": optuna.samplers.NSGAIIISampler,
    "QMCSampler": optuna.samplers.QMCSampler,
}


class HyperOptimizer:
    """
    超参数优化器类
    该类被发送到超参数优化工作进程。
    """

    def __init__(self, config: Config, data_pickle_file: Path) -> None:
        self.buy_space: list[DimensionProtocol] = []  # 买入空间
        self.sell_space: list[DimensionProtocol] = []  # 卖出空间
        self.protection_space: list[DimensionProtocol] = []  # 保护机制空间
        self.roi_space: list[DimensionProtocol] = []  # ROI空间
        self.stoploss_space: list[DimensionProtocol] = []  # 止损空间
        self.trailing_space: list[DimensionProtocol] = []  # 追踪止损空间
        self.max_open_trades_space: list[DimensionProtocol] = []  # 最大开仓交易空间
        self.dimensions: list[DimensionProtocol] = []  # 所有维度
        self.o_dimensions: dict = {}  # optuna维度

        self.config = config
        self.min_date: datetime  # 最小日期
        self.max_date: datetime  # 最大日期

        self.backtesting = Backtesting(self.config)  # 回测实例
        self.pairlist = self.backtesting.pairlists.whitelist  # 交易对列表
        self.custom_hyperopt: HyperOptAuto  # 自定义超参数优化实例
        self.analyze_per_epoch = self.config.get("analyze_per_epoch", False)  # 是否每轮分析

        if not self.config.get("hyperopt"):
            self.custom_hyperopt = HyperOptAuto(self.config)
        else:
            raise OperationalException(
                "在2021.9版本中已移除使用单独的Hyperopt文件。请将您现有的Hyperopt文件转换为新的可超参数优化策略接口"
            )

        self.backtesting._set_strategy(self.backtesting.strategylist[0])
        self.custom_hyperopt.strategy = self.backtesting.strategy

        self.hyperopt_pickle_magic(self.backtesting.strategy.__class__.__bases__)
        self.custom_hyperoptloss: IHyperOptLoss = HyperOptLossResolver.load_hyperoptloss(
            self.config
        )
        self.calculate_loss = self.custom_hyperoptloss.hyperopt_loss_function  # 损失计算函数

        self.data_pickle_file = data_pickle_file  # 数据序列化文件

        self.market_change = 0.0  # 市场变化率

        self.es_epochs = config.get("early_stop", 0)  # 早停轮数
        if self.es_epochs > 0 and self.es_epochs < 0.2 * config.get("epochs", 0):
            logger.warning(f"早停轮数 {self.es_epochs} 低于总轮数的20%")

        if HyperoptTools.has_space(self.config, "sell"):
            # 确保使用出场信号
            self.config["use_exit_signal"] = True

    def prepare_hyperopt(self) -> None:
        # 初始化空间...
        self.init_spaces()

        self.prepare_hyperopt_data()

        # 运行超参数优化时不再需要交易所实例
        self.backtesting.exchange.close()
        self.backtesting.exchange._api = None
        self.backtesting.exchange._api_async = None
        self.backtesting.exchange.loop = None  # type: ignore
        self.backtesting.exchange._loop_lock = None  # type: ignore
        self.backtesting.exchange._cache_lock = None  # type: ignore
        # self.backtesting.exchange = None  # type: ignore
        self.backtesting.pairlists = None  # type: ignore

    def get_strategy_name(self) -> str:
        return self.backtesting.strategy.get_strategy_name()

    def hyperopt_pickle_magic(self, bases: tuple[type, ...]) -> None:
        """
        超参数优化的序列化魔术，允许跨文件的策略继承。
        为了使其正常工作，我们需要将导入类的模块注册为可序列化的值。
        """
        for modules in bases:
            if modules.__name__ != "IStrategy":
                if mod := sys.modules.get(modules.__module__):
                    cloudpickle.register_pickle_by_value(mod)
                self.hyperopt_pickle_magic(modules.__bases__)

    def _get_params_details(self, params: dict) -> dict:
        """
        返回每个空间的参数
        """
        result: dict = {}

        if HyperoptTools.has_space(self.config, "buy"):
            result["buy"] = round_dict({p.name: params.get(p.name) for p in self.buy_space}, 13)
        if HyperoptTools.has_space(self.config, "sell"):
            result["sell"] = round_dict({p.name: params.get(p.name) for p in self.sell_space}, 13)
        if HyperoptTools.has_space(self.config, "protection"):
            result["protection"] = round_dict(
                {p.name: params.get(p.name) for p in self.protection_space}, 13
            )
        if HyperoptTools.has_space(self.config, "roi"):
            result["roi"] = round_dict(
                {str(k): v for k, v in self.custom_hyperopt.generate_roi_table(params).items()}, 13
            )
        if HyperoptTools.has_space(self.config, "stoploss"):
            result["stoploss"] = round_dict(
                {p.name: params.get(p.name) for p in self.stoploss_space}, 13
            )
        if HyperoptTools.has_space(self.config, "trailing"):
            result["trailing"] = round_dict(
                self.custom_hyperopt.generate_trailing_params(params), 13
            )
        if HyperoptTools.has_space(self.config, "trades"):
            result["max_open_trades"] = round_dict(
                {
                    "max_open_trades": (
                        self.backtesting.strategy.max_open_trades
                        if self.backtesting.strategy.max_open_trades != float("inf")
                        else -1
                    )
                },
                13,
            )

        return result

    def _get_no_optimize_details(self) -> dict[str, Any]:
        """
        获取未优化的参数
        """
        result: dict[str, Any] = {}
        strategy = self.backtesting.strategy
        if not HyperoptTools.has_space(self.config, "roi"):
            result["roi"] = {str(k): v for k, v in strategy.minimal_roi.items()}
        if not HyperoptTools.has_space(self.config, "stoploss"):
            result["stoploss"] = {"stoploss": strategy.stoploss}
        if not HyperoptTools.has_space(self.config, "trailing"):
            result["trailing"] = {
                "trailing_stop": strategy.trailing_stop,
                "trailing_stop_positive": strategy.trailing_stop_positive,
                "trailing_stop_positive_offset": strategy.trailing_stop_positive_offset,
                "trailing_only_offset_is_reached": strategy.trailing_only_offset_is_reached,
            }
        if not HyperoptTools.has_space(self.config, "trades"):
            result["max_open_trades"] = {"max_open_trades": strategy.max_open_trades}
        return result

    def init_spaces(self):
        """
        分配超参数优化空间中的维度。
        """
        if HyperoptTools.has_space(self.config, "protection"):
            # 只有使用Parameter接口时才能优化保护机制
            logger.debug("超参数优化包含'protection'空间")
            # 如果选择了保护空间，则启用保护机制
            self.config["enable_protections"] = True
            self.backtesting.enable_protections = True
            self.protection_space = self.custom_hyperopt.protection_space()

        if HyperoptTools.has_space(self.config, "buy"):
            logger.debug("超参数优化包含'buy'空间")
            self.buy_space = self.custom_hyperopt.buy_indicator_space()

        if HyperoptTools.has_space(self.config, "sell"):
            logger.debug("超参数优化包含'sell'空间")
            self.sell_space = self.custom_hyperopt.sell_indicator_space()

        if HyperoptTools.has_space(self.config, "roi"):
            logger.debug("超参数优化包含'roi'空间")
            self.roi_space = self.custom_hyperopt.roi_space()

        if HyperoptTools.has_space(self.config, "stoploss"):
            logger.debug("超参数优化包含'stoploss'空间")
            self.stoploss_space = self.custom_hyperopt.stoploss_space()

        if HyperoptTools.has_space(self.config, "trailing"):
            logger.debug("超参数优化包含'trailing'空间")
            self.trailing_space = self.custom_hyperopt.trailing_space()

        if HyperoptTools.has_space(self.config, "trades"):
            logger.debug("超参数优化包含'trades'空间")
            self.max_open_trades_space = self.custom_hyperopt.max_open_trades_space()

        self.dimensions = (
            self.buy_space
            + self.sell_space
            + self.protection_space
            + self.roi_space
            + self.stoploss_space
            + self.trailing_space
            + self.max_open_trades_space
        )

    def assign_params(self, params_dict: dict[str, Any], category: str) -> None:
        """
        分配可超参数优化的参数
        """
        for attr_name, attr in self.backtesting.strategy.enumerate_parameters(category):
            if attr.optimize:
                # noinspection PyProtectedMember
                attr.value = params_dict[attr_name]

    @delayed
    @wrap_non_picklable_objects
    def generate_optimizer_wrapped(self, params_dict: dict[str, Any]) -> dict[str, Any]:
        return self.generate_optimizer(params_dict)

    def generate_optimizer(self, params_dict: dict[str, Any]) -> dict[str, Any]:
        """
        用于优化的函数。
        每轮调用一次以优化配置的内容。
        尽量优化此函数！
        """
        HyperoptStateContainer.set_state(HyperoptState.OPTIMIZE)
        backtest_start_time = datetime.now(timezone.utc)

        # 应用参数
        if HyperoptTools.has_space(self.config, "buy"):
            self.assign_params(params_dict, "buy")

        if HyperoptTools.has_space(self.config, "sell"):
            self.assign_params(params_dict, "sell")

        if HyperoptTools.has_space(self.config, "protection"):
            self.assign_params(params_dict, "protection")

        if HyperoptTools.has_space(self.config, "roi"):
            self.backtesting.strategy.minimal_roi = self.custom_hyperopt.generate_roi_table(
                params_dict
            )

        if HyperoptTools.has_space(self.config, "stoploss"):
            self.backtesting.strategy.stoploss = params_dict["stoploss"]

        if HyperoptTools.has_space(self.config, "trailing"):
            d = self.custom_hyperopt.generate_trailing_params(params_dict)
            self.backtesting.strategy.trailing_stop = d["trailing_stop"]
            self.backtesting.strategy.trailing_stop_positive = d["trailing_stop_positive"]
            self.backtesting.strategy.trailing_stop_positive_offset = d[
                "trailing_stop_positive_offset"
            ]
            self.backtesting.strategy.trailing_only_offset_is_reached = d[
                "trailing_only_offset_is_reached"
            ]

        if HyperoptTools.has_space(self.config, "trades"):
            if self.config["stake_amount"] == "unlimited" and (
                params_dict["max_open_trades"] == -1 or params_dict["max_open_trades"] == 0
            ):
                # 如果赌注金额是无限的，则忽略无限的最大开仓交易数
                params_dict.update({"max_open_trades": self.config["max_open_trades"]})

            updated_max_open_trades = (
                int(params_dict["max_open_trades"])
                if (params_dict["max_open_trades"] != -1 and params_dict["max_open_trades"] != 0)
                else float("inf")
            )

            self.config.update({"max_open_trades": updated_max_open_trades})

            self.backtesting.strategy.max_open_trades = updated_max_open_trades

        with self.data_pickle_file.open("rb") as f:
            processed = load(f, mmap_mode="r")
        if self.analyze_per_epoch:
            # 数据尚未分析，重新运行populate_indicators
            processed = self.advise_and_trim(processed)

        bt_results = self.backtesting.backtest(
            processed=processed, start_date=self.min_date, end_date=self.max_date
        )
        backtest_end_time = datetime.now(timezone.utc)
        bt_results.update(
            {
                "backtest_start_time": int(backtest_start_time.timestamp()),
                "backtest_end_time": int(backtest_end_time.timestamp()),
            }
        )
        result = self._get_results_dict(
            bt_results, self.min_date, self.max_date, params_dict, processed=processed
        )
        return result

    def _get_results_dict(
        self,
        backtesting_results: BacktestContentType,
        min_date: datetime,
        max_date: datetime,
        params_dict: dict[str, Any],
        processed: dict[str, DataFrame],
    ) -> dict[str, Any]:
        params_details = self._get_params_details(params_dict)

        strat_stats = generate_strategy_stats(
            self.pairlist,
            self.backtesting.strategy.get_strategy_name(),
            backtesting_results,
            min_date,
            max_date,
            market_change=self.market_change,
            is_hyperopt=True,
        )
        results_explanation = HyperoptTools.format_results_explanation_string(
            strat_stats, self.config["stake_currency"]
        )

        not_optimized = self.backtesting.strategy.get_no_optimize_params()
        not_optimized = deep_merge_dicts(not_optimized, self._get_no_optimize_details())

        trade_count = strat_stats["total_trades"]
        total_profit = strat_stats["profit_total"]

        # 如果此评估包含的交易数量太少而不具参考价值
        # 则将其视为"不良"（分配最大损失值）
        # 以便将此超空间点从优化路径中排除
        # 我们不想优化"持有"策略
        loss: float = MAX_LOSS
        if trade_count >= self.config["hyperopt_min_trades"]:
            loss = self.calculate_loss(
                results=backtesting_results["results"],
                trade_count=trade_count,
                min_date=min_date,
                max_date=max_date,
                config=self.config,
                processed=processed,
                backtest_stats=strat_stats,
                starting_balance=get_dry_run_wallet(self.config),
            )
        return {
            "loss": loss,
            "params_dict": params_dict,
            "params_details": params_details,
            "params_not_optimized": not_optimized,
            "results_metrics": strat_stats,
            "results_explanation": results_explanation,
            "total_profit": total_profit,
        }

    def convert_dimensions_to_optuna_space(self, s_dimensions: list[DimensionProtocol]) -> dict:
        o_dimensions: dict[str, optuna.distributions.BaseDistribution] = {}
        for original_dim in s_dimensions:
            if isinstance(
                original_dim,
                ft_CategoricalDistribution | ft_IntDistribution | ft_FloatDistribution | SKDecimal,
            ):
                o_dimensions[original_dim.name] = original_dim
            else:
                raise OperationalException(
                    f"未知的搜索空间 {original_dim.name} - {original_dim} / \
                        {type(original_dim)}"
                )
        return o_dimensions

    def get_optimizer(
        self,
        random_state: int,
    ):
        o_sampler = self.custom_hyperopt.generate_estimator(
            dimensions=self.dimensions, random_state=random_state
        )
        self.o_dimensions = self.convert_dimensions_to_optuna_space(self.dimensions)

        if isinstance(o_sampler, str):
            if o_sampler not in optuna_samplers_dict.keys():
                raise OperationalException(f"Optuna采样器 {o_sampler} 不受支持。")
            with warnings.catch_warnings():
                warnings.filterwarnings(action="ignore", category=ExperimentalWarning)
                if o_sampler in ["NSGAIIISampler", "NSGAIISampler"]:
                    sampler = optuna_samplers_dict[o_sampler](
                        seed=random_state, population_size=INITIAL_POINTS
                    )
                elif o_sampler in ["GPSampler", "TPESampler", "CmaEsSampler"]:
                    sampler = optuna_samplers_dict[o_sampler](
                        seed=random_state, n_startup_trials=INITIAL_POINTS
                    )
                else:
                    sampler = optuna_samplers_dict[o_sampler](seed=random_state)
        else:
            sampler = o_sampler

        if self.es_epochs > 0:
            with warnings.catch_warnings():
                warnings.filterwarnings(action="ignore", category=ExperimentalWarning)
                self.es_terminator = Terminator(BestValueStagnationEvaluator(self.es_epochs))

        logger.info(f"使用optuna采样器 {o_sampler}。")
        return optuna.create_study(sampler=sampler, direction="minimize")

    def advise_and_trim(self, data: dict[str, DataFrame]) -> dict[str, DataFrame]:
        preprocessed = self.backtesting.strategy.advise_all_indicators(data)

        # 从分析的数据框中修剪启动期，以获得正确的输出日期
        # 这仅用于跟踪修剪后的最小/最大日期
        # 此方法不返回结果，实际修剪在回测中进行
        trimmed = trim_dataframes(preprocessed, self.timerange, self.backtesting.required_startup)
        self.min_date, self.max_date = get_timerange(trimmed)
        if not self.market_change:
            self.market_change = calculate_market_change(trimmed, "close")

        # 实际修剪将作为回测的一部分进行
        return preprocessed

    def prepare_hyperopt_data(self) -> None:
        HyperoptStateContainer.set_state(HyperoptState.DATALOAD)
        data, self.timerange = self.backtesting.load_bt_data()
        logger.info("数据加载完成。正在计算指标")

        if not self.analyze_per_epoch:
            HyperoptStateContainer.set_state(HyperoptState.INDICATORS)

            preprocessed = self.advise_and_trim(data)

            logger.info(
                f"超参数优化使用的数据从 "
                f"{self.min_date.strftime(DATETIME_PRINT_FORMAT)} "
                f"到 {self.max_date.strftime(DATETIME_PRINT_FORMAT)} "
                f"（{(self.max_date - self.min_date).days} 天）。"
            )
            # 存储未修剪的数据 - 将在信号生成后修剪
            dump(preprocessed, self.data_pickle_file)
        else:
            dump(data, self.data_pickle_file)