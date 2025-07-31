# pragma pylint: disable=attribute-defined-outside-init

"""
此模块用于加载自定义策略
"""

import logging
import tempfile
from base64 import urlsafe_b64decode
from inspect import getfullargspec
from os import walk
from pathlib import Path
from typing import Any

from freqtrade.configuration.config_validation import validate_migrated_strategy_settings
from freqtrade.constants import REQUIRED_ORDERTIF, REQUIRED_ORDERTYPES, USERPATH_STRATEGIES, Config
from freqtrade.enums import TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.resolvers.iresolver import IResolver
from freqtrade.strategy.interface import IStrategy


logger = logging.getLogger(__name__)


class StrategyResolver(IResolver):
    """
    此类包含加载自定义策略类的逻辑
    """

    object_type = IStrategy
    object_type_str = "Strategy"
    user_subdir = USERPATH_STRATEGIES
    initial_search_path = None
    extra_path = "strategy_path"

    @staticmethod
    def load_strategy(config: Config | None = None) -> IStrategy:
        """
        从配置参数加载自定义类
        :param config: 配置字典或None
        """
        config = config or {}

        if not config.get("strategy"):
            raise OperationalException(
                "未设置策略。请使用 `--strategy` 指定要使用的策略类。"
            )

        strategy_name = config["strategy"]
        strategy: IStrategy = StrategyResolver._load_strategy(
            strategy_name, config=config, extra_dir=config.get("strategy_path")
        )
        strategy.ft_load_params_from_file()
        # 设置属性
        # 检查是否需要覆盖配置
        #             （属性名，                          默认值，     子键）
        attributes = [
            ("minimal_roi", {"0": 10.0}),
            ("timeframe", None),
            ("stoploss", None),
            ("trailing_stop", None),
            ("trailing_stop_positive", None),
            ("trailing_stop_positive_offset", 0.0),
            ("trailing_only_offset_is_reached", None),
            ("use_custom_stoploss", None),
            ("process_only_new_candles", None),
            ("order_types", None),
            ("order_time_in_force", None),
            ("stake_currency", None),
            ("stake_amount", None),
            ("startup_candle_count", None),
            ("unfilledtimeout", None),
            ("use_exit_signal", True),
            ("exit_profit_only", False),
            ("ignore_roi_if_entry_signal", False),
            ("exit_profit_offset", 0.0),
            ("disable_dataframe_checks", False),
            ("ignore_buying_expired_candle_after", 0),
            ("position_adjustment_enable", False),
            ("max_entry_position_adjustment", -1),
            ("max_open_trades", float("inf")),
        ]
        for attribute, default in attributes:
            StrategyResolver._override_attribute_helper(strategy, config, attribute, default)

        # 再次循环此列表以合并输出
        for attribute, _ in attributes:
            if attribute in config:
                logger.info("策略使用 %s: %s", attribute, config[attribute])

        StrategyResolver._normalize_attributes(strategy)

        StrategyResolver._strategy_sanity_validations(strategy)
        return strategy

    @staticmethod
    def _override_attribute_helper(strategy, config: Config, attribute: str, default: Any):
        """
        覆盖策略中的属性。
        优先级：
        - 配置
        - 策略
        - 默认值（如果不为None）
        """
        if attribute in config and not isinstance(
            getattr(type(strategy), attribute, None), property
        ):
            # 确保不覆盖属性
            setattr(strategy, attribute, config[attribute])
            logger.info(
                "使用配置文件中的值覆盖策略 '%s': %s。",
                attribute,
                config[attribute],
            )
        elif hasattr(strategy, attribute):
            val = getattr(strategy, attribute)
            # None值不能存在于配置中，因此不复制
            if val is not None:
                # 策略中设置为-1的max_open_trades将在配置中复制为无穷大
                if attribute == "max_open_trades" and val == -1:
                    config[attribute] = float("inf")
                else:
                    config[attribute] = val
        # 显式检查None，因为可能存在其他"假"值
        elif default is not None:
            setattr(strategy, attribute, default)
            config[attribute] = default

    @staticmethod
    def _normalize_attributes(strategy: IStrategy) -> IStrategy:
        """
        标准化属性以确保正确的类型。
        """
        # 排序并应用类型转换
        if hasattr(strategy, "minimal_roi"):
            strategy.minimal_roi = dict(
                sorted(
                    {int(key): value for (key, value) in strategy.minimal_roi.items()}.items(),
                    key=lambda t: t[0],
                )
            )
        if hasattr(strategy, "stoploss"):
            strategy.stoploss = float(strategy.stoploss)
        if hasattr(strategy, "max_open_trades") and strategy.max_open_trades < 0:
            strategy.max_open_trades = float("inf")
        return strategy

    @staticmethod
    def _strategy_sanity_validations(strategy: IStrategy):
        # 确保首先执行必要的迁移。
        validate_migrated_strategy_settings(strategy.config)

        if not all(k in strategy.order_types for k in REQUIRED_ORDERTYPES):
            raise ImportError(
                f"无法加载策略 '{strategy.__class__.__name__}'。订单类型映射不完整。"
            )
        if not all(k in strategy.order_time_in_force for k in REQUIRED_ORDERTIF):
            raise ImportError(
                f"无法加载策略 '{strategy.__class__.__name__}'。订单有效期映射不完整。"
            )
        trading_mode = strategy.config.get("trading_mode", TradingMode.SPOT)

        if strategy.can_short and trading_mode == TradingMode.SPOT:
            raise ImportError(
                "做空策略不能在现货市场运行。请确保这是正确的策略，并且交易模式配置正确。"
                "可以通过在策略中设置 `can_short=False` 在现货市场运行此策略。请注意，做空信号在这种情况下将被忽略。"
            )

    @staticmethod
    def validate_strategy(strategy: IStrategy) -> IStrategy:
        if strategy.config.get("trading_mode", TradingMode.SPOT) != TradingMode.SPOT:
            # 需要新方法
            warn_deprecated_setting(strategy, "sell_profit_only", "exit_profit_only", True)
            warn_deprecated_setting(strategy, "sell_profit_offset", "exit_profit_offset", True)
            warn_deprecated_setting(strategy, "use_sell_signal", "use_exit_signal", True)
            warn_deprecated_setting(
                strategy, "ignore_roi_if_buy_signal", "ignore_roi_if_entry_signal", True
            )

            if not check_override(strategy, IStrategy, "populate_entry_trend"):
                raise OperationalException("必须实现 `populate_entry_trend`。")
            if not check_override(strategy, IStrategy, "populate_exit_trend"):
                raise OperationalException("必须实现 `populate_exit_trend`。")
            if check_override(strategy, IStrategy, "check_buy_timeout"):
                raise OperationalException(
                    "请将 `check_buy_timeout` 实现迁移到 `check_entry_timeout`。"
                )
            if check_override(strategy, IStrategy, "check_sell_timeout"):
                raise OperationalException(
                    "请将 `check_sell_timeout` 实现迁移到 `check_exit_timeout`。"
                )

            if check_override(strategy, IStrategy, "custom_sell"):
                raise OperationalException(
                    "请将 `custom_sell` 实现迁移到 `custom_exit`。"
                )

        else:
            # TODO: 实现以下方法之一时应显示弃用警告
            # buy_trend 和 sell_trend, custom_sell
            warn_deprecated_setting(strategy, "sell_profit_only", "exit_profit_only")
            warn_deprecated_setting(strategy, "sell_profit_offset", "exit_profit_offset")
            warn_deprecated_setting(strategy, "use_sell_signal", "use_exit_signal")
            warn_deprecated_setting(
                strategy, "ignore_roi_if_buy_signal", "ignore_roi_if_entry_signal"
            )

            if not check_override(strategy, IStrategy, "populate_buy_trend") and not check_override(
                strategy, IStrategy, "populate_entry_trend"
            ):
                raise OperationalException(
                    "必须实现 `populate_entry_trend` 或 `populate_buy_trend`。"
                )
            if not check_override(
                strategy, IStrategy, "populate_sell_trend"
            ) and not check_override(strategy, IStrategy, "populate_exit_trend"):
                raise OperationalException(
                    "必须实现 `populate_exit_trend` 或 `populate_sell_trend`。"
                )

            _populate_fun_len = len(getfullargspec(strategy.populate_indicators).args)
            _buy_fun_len = len(getfullargspec(strategy.populate_buy_trend).args)
            _sell_fun_len = len(getfullargspec(strategy.populate_sell_trend).args)
            if any(x == 2 for x in [_populate_fun_len, _buy_fun_len, _sell_fun_len]):
                raise OperationalException(
                    "策略接口 v1 不再支持。"
                    "请更新您的策略以实现"
                    "`populate_indicators`、`populate_entry_trend` 和 `populate_exit_trend`"
                    "并包含元数据参数。"
                )

        has_after_fill = "after_fill" in getfullargspec(
            strategy.custom_stoploss
        ).args and check_override(strategy, IStrategy, "custom_stoploss")
        if has_after_fill:
            strategy._ft_stop_uses_after_fill = True

        if check_override(strategy, IStrategy, "adjust_order_price") and (
            check_override(strategy, IStrategy, "adjust_entry_price")
            or check_override(strategy, IStrategy, "adjust_exit_price")
        ):
            raise OperationalException(
                "如果实现了 `adjust_order_price`，则 `adjust_entry_price` 和"
                "`adjust_exit_price` 将不会被使用。请为您的策略选择一种方法。"
            )
        return strategy

    @staticmethod
    def _load_strategy(
        strategy_name: str, config: Config, extra_dir: str | None = None
    ) -> IStrategy:
        """
        搜索并加载指定的策略。
        :param strategy_name: 要导入的模块名称
        :param config: 策略的配置
        :param extra_dir: 要搜索指定策略的附加目录
        :return: 策略实例或None
        """
        if config.get("recursive_strategy_search", False):
            extra_dirs: list[str] = [
                path[0] for path in walk(f"{config['user_data_dir']}/{USERPATH_STRATEGIES}")
            ]  # 子目录
        else:
            extra_dirs = []

        if extra_dir:
            extra_dirs.append(extra_dir)

        abs_paths = StrategyResolver.build_search_paths(
            config, user_subdir=USERPATH_STRATEGIES, extra_dirs=extra_dirs
        )

        if ":" in strategy_name:
            logger.info("正在加载base64编码的策略")
            strat = strategy_name.split(":")

            if len(strat) == 2:
                temp = Path(tempfile.mkdtemp("freq", "strategy"))
                name = strat[0] + ".py"

                temp.joinpath(name).write_text(urlsafe_b64decode(strat[1]).decode("utf-8"))
                temp.joinpath("__init__.py").touch()

                strategy_name = strat[0]

                # 向bot注册临时路径
                abs_paths.insert(0, temp.resolve())

        strategy = StrategyResolver._load_object(
            paths=abs_paths,
            object_name=strategy_name,
            add_source=True,
            kwargs={"config": config},
        )

        if strategy:
            return StrategyResolver.validate_strategy(strategy)

        raise OperationalException(
            f"无法加载策略 '{strategy_name}'。此类不存在"
            "或包含Python代码错误。"
        )


def warn_deprecated_setting(strategy: IStrategy, old: str, new: str, error=False):
    if hasattr(strategy, old):
        errormsg = f"已弃用: 使用 '{old}' 已移至 '{new}'。"
        if error:
            raise OperationalException(errormsg)
        logger.warning(errormsg)
        setattr(strategy, new, getattr(strategy, f"{old}"))


def check_override(obj, parentclass, attribute: str):
    """
    检查对象是否覆盖父类属性。
    :returns: 如果对象被覆盖则返回True。
    """
    return getattr(type(obj), attribute) != getattr(parentclass, attribute)