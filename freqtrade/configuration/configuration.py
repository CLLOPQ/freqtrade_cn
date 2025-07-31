"""
该模块包含配置类
"""

import logging
import warnings
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

from freqtrade import constants
from freqtrade.configuration.deprecated_settings import process_temporary_deprecated_settings
from freqtrade.configuration.directory_operations import create_datadir, create_userdata_dir
from freqtrade.configuration.environment_vars import enironment_vars_to_dict
from freqtrade.configuration.load_config import load_file, load_from_files
from freqtrade.constants import Config
from freqtrade.enums import (
    NON_UTIL_MODES,
    TRADE_MODES,
    CandleType,
    MarginMode,
    RunMode,
    TradingMode,
)
from freqtrade.exceptions import OperationalException
from freqtrade.loggers import setup_logging
from freqtrade.misc import deep_merge_dicts, parse_db_uri_for_logging, safe_value_fallback


logger = logging.getLogger(__name__)


class Configuration:
    """
    用于读取和初始化机器人配置的类
    机器人、回测、超参数优化以及所有需要配置的脚本都可复用此类
    """

    def __init__(self, args: dict[str, Any], runmode: RunMode | None = None) -> None:
        self.args = args
        self.config: Config | None = None
        self.runmode = runmode

    def get_config(self) -> Config:
        """
        返回配置。使用此方法获取机器人配置
        :return: 字典：机器人配置
        """
        if self.config is None:
            self.config = self.load_config()

        return self.config

    @staticmethod
    def from_files(files: list[str]) -> dict[str, Any]:
        """
        遍历传入的配置文件，加载所有文件并合并其内容。
        文件按顺序加载，后续配置文件中的参数会覆盖
        较早文件中的相同参数（最后定义的参数生效）。
        执行整个配置初始化过程，因此所有预期的配置项
        都可用于交互式环境。
        :param files: 文件路径列表
        :return: 配置字典
        """
        # 将此方法保留为静态方法，以便可从交互式环境中使用
        c = Configuration({"config": files}, RunMode.OTHER)
        return c.get_config()

    def load_config(self) -> dict[str, Any]:
        """
        从sys.argv提取信息并加载机器人配置
        :return: 配置字典
        """
        # 加载所有配置
        config: Config = load_from_files(self.args.get("config", []))

        # 加载环境变量
        from freqtrade.commands.arguments import NO_CONF_ALLOWED

        if self.args.get("command") not in NO_CONF_ALLOWED:
            env_data = enironment_vars_to_dict()
            config = deep_merge_dicts(env_data, config)

        # 标准化配置
        if "internals" not in config:
            config["internals"] = {}

        if "pairlists" not in config:
            config["pairlists"] = []

        # 保留原始配置文件的副本
        config["original_config"] = deepcopy(config)

        self._process_logging_options(config)

        self._process_runmode(config)

        self._process_common_options(config)

        self._process_trading_options(config)

        self._process_optimize_options(config)

        self._process_plot_options(config)

        self._process_data_options(config)

        self._process_analyze_options(config)

        self._process_freqai_options(config)

        # 在此处导入check_exchange以避免导入循环问题
        from freqtrade.exchange.check_exchange import check_exchange

        # 检查用户设置的交易所是否受支持
        check_exchange(config, config.get("experimental", {}).get("block_bad_exchanges", True))

        self._resolve_pairs_list(config)

        process_temporary_deprecated_settings(config)

        return config

    def _process_logging_options(self, config: Config) -> None:
        """
        从sys.argv提取信息并加载日志配置：
        -v/--verbose、--logfile选项
        """
        # 日志级别
        if "verbosity" not in config or self.args.get("verbosity") is not None:
            config.update(
                {"verbosity": safe_value_fallback(self.args, "verbosity", default_value=0)}
            )

        if self.args.get("logfile"):
            config.update({"logfile": self.args["logfile"]})

        if "print_colorized" in self.args and not self.args["print_colorized"]:
            logger.info("检测到参数--no-color...")
            config.update({"print_colorized": False})
        else:
            config.update({"print_colorized": True})

        setup_logging(config)

    def _process_trading_options(self, config: Config) -> None:
        if config["runmode"] not in TRADE_MODES:
            return

        if config.get("dry_run", False):
            logger.info("已启用模拟运行")
            if config.get("db_url") in [None, constants.DEFAULT_DB_PROD_URL]:
                # 如果未指定，模拟运行默认使用内存数据库
                config["db_url"] = constants.DEFAULT_DB_DRYRUN_URL
        else:
            if not config.get("db_url"):
                config["db_url"] = constants.DEFAULT_DB_PROD_URL
            logger.info("已禁用模拟运行")

        logger.info(f'使用数据库："{parse_db_uri_for_logging(config["db_url"])}"')

    def _process_common_options(self, config: Config) -> None:
        # 如果配置中未指定策略，或者是非默认策略，则设置策略
        if self.args.get("strategy") or not config.get("strategy"):
            config.update({"strategy": self.args.get("strategy")})

        self._args_to_config(
            config, argname="strategy_path", logstring="使用额外的策略查找路径：{}"
        )

        if (
            "db_url" in self.args
            and self.args["db_url"]
            and self.args["db_url"] != constants.DEFAULT_DB_PROD_URL
        ):
            config.update({"db_url": self.args["db_url"]})
            logger.info("检测到参数--db-url...")

        self._args_to_config(
            config, argname="db_url_from", logstring="检测到参数--db-url-from..."
        )

        if config.get("force_entry_enable", False):
            logger.warning("已启用`force_entry_enable` RPC消息。")

        # 支持sd_notify
        if self.args.get("sd_notify"):
            config["internals"].update({"sd_notify": True})

    def _process_datadir_options(self, config: Config) -> None:
        """
        从sys.argv提取信息并加载目录配置
        --user-data、--datadir
        """
        # 在此处检查交易所参数 - 否则`datadir`可能不正确。
        if self.args.get("exchange"):
            config["exchange"]["name"] = self.args["exchange"]
            logger.info(f"使用交易所 {config['exchange']['name']}")

        if "pair_whitelist" not in config["exchange"]:
            config["exchange"]["pair_whitelist"] = []

        if self.args.get("user_data_dir"):
            config.update({"user_data_dir": self.args["user_data_dir"]})
        elif "user_data_dir" not in config:
            # 默认使用当前工作目录/user_data（遗留选项...）
            config.update({"user_data_dir": str(Path.cwd() / "user_data")})

        # 重置为user_data_dir，使其包含绝对路径。
        config["user_data_dir"] = create_userdata_dir(config["user_data_dir"], create_dir=False)
        logger.info("使用用户数据目录：%s...", config["user_data_dir"])

        config.update({"datadir": create_datadir(config, self.args.get("datadir"))})
        logger.info("使用数据目录：%s...", config.get("datadir"))

        if self.args.get("exportfilename"):
            self._args_to_config(
                config, argname="exportfilename", logstring="将回测结果存储到{}..."
            )
            config["exportfilename"] = Path(config["exportfilename"])
        else:
            config["exportfilename"] = config["user_data_dir"] / "backtest_results"

        if self.args.get("show_sensitive"):
            logger.warning(
                "即将输出的内容中将显示敏感信息。"
                "请确保在分享此输出前自行编辑敏感信息。"
            )

    def _process_optimize_options(self, config: Config) -> None:
        # 这将覆盖策略配置
        self._args_to_config(
            config,
            argname="timeframe",
            logstring="检测到参数-i/--timeframe...使用时间框架：{}...",
        )

        self._args_to_config(
            config,
            argname="position_stacking",
            logstring="检测到参数--enable-position-stacking...",
        )

        self._args_to_config(
            config,
            argname="enable_protections",
            logstring="检测到参数--enable-protections，启用保护机制...",
        )

        if self.args.get("max_open_trades"):
            config.update({"max_open_trades": self.args["max_open_trades"]})
            logger.info(
                "检测到参数--max-open-trades，将max_open_trades覆盖为：%s...",
                config.get("max_open_trades"),
            )
        elif config["runmode"] in NON_UTIL_MODES:
            logger.info("使用max_open_trades：%s...", config.get("max_open_trades"))
        # 如果max_open_trades为-1，则设置为无限
        if config.get("max_open_trades") == -1:
            config["max_open_trades"] = float("inf")

        if self.args.get("stake_amount"):
            # 显式转换为float以支持CLI参数同时表示无限和具体值
            try:
                self.args["stake_amount"] = float(self.args["stake_amount"])
            except ValueError:
                pass

        configurations = [
            (
                "timeframe_detail",
                "检测到参数--timeframe-detail，使用{}进行烛台内回测...",
            ),
            ("backtest_show_pair_list", "检测到参数--show-pair-list。"),
            (
                "stake_amount",
                "检测到参数--stake-amount，将stake_amount覆盖为：{}...",
            ),
            (
                "dry_run_wallet",
                "检测到参数--dry-run-wallet，将dry_run_wallet覆盖为：{}...",
            ),
            ("fee", "检测到参数--fee，将费用设置为：{}..."),
            ("timerange", "检测到参数--timerange：{}..."),
        ]

        self._args_to_config_loop(config, configurations)

        self._process_datadir_options(config)

        self._args_to_config(
            config,
            argname="strategy_list",
            logstring="使用包含{}个策略的策略列表",
            logfun=len,
        )

        configurations = [
            (
                "recursive_strategy_search",
                "在策略文件夹中递归搜索策略。",
            ),
            ("timeframe", "使用命令行参数覆盖时间框架"),
            ("export", "检测到参数--export：{}..."),
            ("backtest_breakdown", "检测到参数--breakdown..."),
            ("backtest_cache", "检测到参数--cache={}..."),
            ("disableparamexport", "检测到参数--disableparamexport：{}..."),
            ("freqai_backtest_live_models", "检测到参数--freqai-backtest-live-models..."),
            ("backtest_notes", "检测到参数--notes：{}..."),
        ]
        self._args_to_config_loop(config, configurations)

        # 超参数优化部分

        configurations = [
            ("hyperopt", "使用超参数优化类名称：{}"),
            ("hyperopt_path", "使用额外的超参数优化查找路径：{}"),
            ("hyperoptexportfilename", "使用超参数优化文件：{}"),
            ("lookahead_analysis_exportfilename", "将前瞻分析结果保存到{}..."),
            ("epochs", "检测到参数--epochs...将以{}个周期运行超参数优化..."),
            ("spaces", "检测到参数-s/--spaces：{}"),
            ("analyze_per_epoch", "检测到参数--analyze-per-epoch。"),
            ("print_all", "检测到参数--print-all..."),
        ]
        self._args_to_config_loop(config, configurations)
        es_epochs = self.args.get("early_stop", 0)
        if es_epochs > 0:
            if es_epochs < 20:
                logger.warning(
                    f"提前停止周期{es_epochs}小于20。将替换为20。"
                )
                config.update({"early_stop": 20})
            else:
                config.update({"early_stop": self.args["early_stop"]})
            logger.info(
                f"检测到参数--early-stop...如果在{config.get('early_stop')}个周期内没有改进，将提前停止超参数优化..."
            )

        configurations = [
            ("print_json", "检测到参数--print-json..."),
            ("export_csv", "检测到参数--export-csv：{}"),
            ("hyperopt_jobs", "检测到参数-j/--job-workers：{}"),
            ("hyperopt_random_state", "检测到参数--random-state：{}"),
            ("hyperopt_min_trades", "检测到参数--min-trades：{}"),
            ("hyperopt_loss", "使用超参数优化损失类名称：{}"),
            ("hyperopt_show_index", "检测到参数-n/--index：{}"),
            ("hyperopt_list_best", "检测到参数--best：{}"),
            ("hyperopt_list_profitable", "检测到参数--profitable：{}"),
            ("hyperopt_list_min_trades", "检测到参数--min-trades：{}"),
            ("hyperopt_list_max_trades", "检测到参数--max-trades：{}"),
            ("hyperopt_list_min_avg_time", "检测到参数--min-avg-time：{}"),
            ("hyperopt_list_max_avg_time", "检测到参数--max-avg-time：{}"),
            ("hyperopt_list_min_avg_profit", "检测到参数--min-avg-profit：{}"),
            ("hyperopt_list_max_avg_profit", "检测到参数--max-avg-profit：{}"),
            ("hyperopt_list_min_total_profit", "检测到参数--min-total-profit：{}"),
            ("hyperopt_list_max_total_profit", "检测到参数--max-total-profit：{}"),
            ("hyperopt_list_min_objective", "检测到参数--min-objective：{}"),
            ("hyperopt_list_max_objective", "检测到参数--max-objective：{}"),
            ("hyperopt_list_no_details", "检测到参数--no-details：{}"),
            ("hyperopt_show_no_header", "检测到参数--no-header：{}"),
            ("hyperopt_ignore_missing_space", "检测到参数--ignore-missing-space：{}"),
        ]

        self._args_to_config_loop(config, configurations)

    def _process_plot_options(self, config: Config) -> None:
        configurations = [
            ("pairs", "使用交易对{}"),
            ("indicators1", "使用指标1：{}"),
            ("indicators2", "使用指标2：{}"),
            ("trade_ids", "根据trade_ids筛选：{}"),
            ("plot_limit", "将图表限制为：{}"),
            ("plot_auto_open", "检测到参数--auto-open。"),
            ("trade_source", "使用来自{}的交易数据"),
            ("prepend_data", "检测到Prepend。允许数据前置。"),
            ("erase", "检测到Erase。删除现有数据。"),
            ("no_trades", "检测到参数--no-trades。"),
            ("timeframes", "时间框架--timeframes：{}"),
            ("days", "检测到--days：{}"),
            ("include_inactive", "检测到--include-inactive-pairs：{}"),
            ("download_trades", "检测到--dl-trades：{}"),
            ("convert_trades", "检测到--convert：{} - 将交易数据转换为OHLCV {}"),
            ("dataformat_ohlcv", '使用"{}"存储OHLCV数据。'),
            ("dataformat_trades", '使用"{}"存储交易数据。'),
            ("show_timerange", "检测到--show-timerange"),
        ]
        self._args_to_config_loop(config, configurations)

    def _process_data_options(self, config: Config) -> None:
        self._args_to_config(
            config, argname="new_pairs_days", logstring="检测到--new-pairs-days：{}"
        )
        self._args_to_config(
            config, argname="trading_mode", logstring="检测到--trading-mode：{}"
        )
        config["candle_type_def"] = CandleType.get_default(
            config.get("trading_mode", "spot") or "spot"
        )
        config["trading_mode"] = TradingMode(config.get("trading_mode", "spot") or "spot")
        config["margin_mode"] = MarginMode(config.get("margin_mode", "") or "")
        self._args_to_config(
            config, argname="candle_types", logstring="检测到--candle-types：{}"
        )

    def _process_analyze_options(self, config: Config) -> None:
        configurations = [
            ("analysis_groups", "分析原因组：{}"),
            ("enter_reason_list", "分析入场标签列表：{}"),
            ("exit_reason_list", "分析出场标签列表：{}"),
            ("indicator_list", "分析指标列表：{}"),
            ("entry_only", "仅分析入场信号：{}"),
            ("exit_only", "仅分析出场信号：{}"),
            ("timerange", "按时间范围筛选交易：{}"),
            ("analysis_rejected", "分析被拒绝的信号：{}"),
            ("analysis_to_csv", "将分析表格存储到CSV：{}"),
            ("analysis_csv_path", "存储分析CSV的路径：{}"),
            # 前瞻分析结果
            ("targeted_trade_amount", "目标交易数量：{}"),
            ("minimum_trade_amount", "最小交易数量：{}"),
            ("lookahead_analysis_exportfilename", "存储前瞻分析结果的路径：{}"),
            ("startup_candle", "用于递归分析的起始烛台：{}"),
        ]
        self._args_to_config_loop(config, configurations)

    def _args_to_config_loop(self, config, configurations: list[tuple[str, str]]) -> None:
        for argname, logstring in configurations:
            self._args_to_config(config, argname=argname, logstring=logstring)

    def _process_runmode(self, config: Config) -> None:
        self._args_to_config(
            config,
            argname="dry_run",
            logstring="检测到参数--dry-run，将dry_run覆盖为：{}...",
        )

        if not self.runmode:
            # 处理实际模式，从配置推断模拟/实盘
            self.runmode = RunMode.DRY_RUN if config.get("dry_run", True) else RunMode.LIVE
            logger.info(f"运行模式设置为{self.runmode.value}。")

        config.update({"runmode": self.runmode})

    def _process_freqai_options(self, config: Config) -> None:
        self._args_to_config(
            config, argname="freqaimodel", logstring="使用freqaimodel类名称：{}"
        )

        self._args_to_config(
            config, argname="freqaimodel_path", logstring="使用freqaimodel路径：{}"
        )

        return

    def _args_to_config(
        self,
        config: Config,
        argname: str,
        logstring: str,
        logfun: Callable | None = None,
        deprecated_msg: str | None = None,
    ) -> None:
        """
        :param config: 配置字典
        :param argname: self.args中的参数名 - 将被复制到配置字典中。
        :param logstring: 日志字符串
        :param logfun: logfun应用于配置项，然后将该配置项
                       通过.format()传入日志字符串。
                       示例：logfun=len（打印找到的配置的长度
                       而不是内容）
        """
        if (
            argname in self.args
            and self.args[argname] is not None
            and self.args[argname] is not False
        ):
            config.update({argname: self.args[argname]})
            if logfun:
                logger.info(logstring.format(logfun(config[argname])))
            else:
                logger.info(logstring.format(config[argname]))
            if deprecated_msg:
                warnings.warn(f"已过时：{deprecated_msg}", DeprecationWarning, stacklevel=1)

    def _resolve_pairs_list(self, config: Config) -> None:
        """
        下载脚本的辅助函数。
        优先使用以下方式获取交易对：
        * -p（交易对参数）
        * --pairs-file
        * 配置中的白名单
        """

        if "pairs" in config:
            config["exchange"]["pair_whitelist"] = config["pairs"]
            return

        if self.args.get("pairs_file"):
            pairs_file = Path(self.args["pairs_file"])
            logger.info(f'读取交易对文件"{pairs_file}"。')
            # 如果未指定配置，或者显式指定了交易对文件，
            # 则从交易对文件下载交易对
            if not pairs_file.exists():
                raise OperationalException(f'未找到路径为"{pairs_file}"的交易对文件。')
            config["pairs"] = load_file(pairs_file)
            if isinstance(config["pairs"], list):
                config["pairs"].sort()
            return

        if self.args.get("config"):
            logger.info("使用配置中的交易对列表。")
            config["pairs"] = config.get("exchange", {}).get("pair_whitelist")
        else:
            # 回退到/dl_path/pairs.json
            pairs_file = config["datadir"] / "pairs.json"
            if pairs_file.exists():
                logger.info(f'读取交易对文件"{pairs_file}"。')
                config["pairs"] = load_file(pairs_file)
                if "pairs" in config and isinstance(config["pairs"], list):
                    config["pairs"].sort()