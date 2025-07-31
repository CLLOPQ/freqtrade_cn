"""
本模块包含参数管理器类
"""

from argparse import ArgumentParser, Namespace, _ArgumentGroup
from functools import partial
from pathlib import Path
from typing import Any

from freqtrade.commands.cli_options import AVAILABLE_CLI_OPTIONS
from freqtrade.constants import DEFAULT_CONFIG


# 通用参数列表
ARGS_COMMON = [
    "verbosity",          # 详细程度
    "print_colorized",    # 彩色打印
    "logfile",            # 日志文件
    "version",            # 版本信息
    "config",             # 配置文件
    "datadir",            # 数据目录
    "user_data_dir",      # 用户数据目录
]

# 主程序参数
ARGS_MAIN = ["version_main"]

# 策略相关参数
ARGS_STRATEGY = [
    "strategy",                       # 策略名称
    "strategy_path",                  # 策略路径
    "recursive_strategy_search",      # 递归搜索策略
    "freqaimodel",                    # FreqAI模型
    "freqaimodel_path",               # FreqAI模型路径
]

# 交易相关参数
ARGS_TRADE = ["db_url", "sd_notify", "dry_run", "dry_run_wallet", "fee"]
# 数据库URL、系统通知、模拟交易、模拟交易初始资金、手续费

# 网页服务器参数
ARGS_WEBSERVER: list[str] = []

# 通用优化参数
ARGS_COMMON_OPTIMIZE = [
    "timeframe",           # 时间周期
    "timerange",           # 时间范围
    "dataformat_ohlcv",    # OHLCV数据格式
    "max_open_trades",     # 最大同时持仓数
    "stake_amount",        # 每次交易金额
    "fee",                 # 手续费
    "pairs",               # 交易对
]

# 回测参数
ARGS_BACKTEST = [
    *ARGS_COMMON_OPTIMIZE,
    "position_stacking",              # 持仓叠加
    "enable_protections",             # 启用保护机制
    "dry_run_wallet",                 # 模拟交易初始资金
    "timeframe_detail",               # 详细时间周期
    "strategy_list",                  # 策略列表
    "export",                         # 导出结果
    "exportfilename",                 # 导出文件名
    "backtest_breakdown",             # 回测详情
    "backtest_cache",                 # 回测缓存
    "freqai_backtest_live_models",    # FreqAI回测实时模型
    "backtest_notes",                 # 回测备注
]

# 超参数优化参数
ARGS_HYPEROPT = [
    *ARGS_COMMON_OPTIMIZE,
    "hyperopt",                        # 超参数优化器
    "hyperopt_path",                   # 超参数优化路径
    "position_stacking",               # 持仓叠加
    "enable_protections",              # 启用保护机制
    "dry_run_wallet",                  # 模拟交易初始资金
    "timeframe_detail",                # 详细时间周期
    "epochs",                          # 迭代次数
    "spaces",                          # 优化空间
    "print_all",                       # 打印所有结果
    "print_json",                      # 以JSON格式打印
    "hyperopt_jobs",                   # 超参数优化并行任务数
    "hyperopt_random_state",           # 超参数优化随机种子
    "hyperopt_min_trades",             # 超参数优化最小交易数
    "hyperopt_loss",                   # 超参数优化损失函数
    "disableparamexport",              # 禁用参数导出
    "hyperopt_ignore_missing_space",   # 忽略缺失的优化空间
    "analyze_per_epoch",               # 每轮迭代分析
    "early_stop",                      # 早停机制
]

# Edge参数
ARGS_EDGE = [*ARGS_COMMON_OPTIMIZE]

# 策略列表参数
ARGS_LIST_STRATEGIES = [
    "strategy_path",                  # 策略路径
    "print_one_column",               # 单列打印
    "recursive_strategy_search",      # 递归搜索策略
]

# FreqAI模型列表参数
ARGS_LIST_FREQAIMODELS = ["freqaimodel_path", "print_one_column"]
# FreqAI模型路径、单列打印

# 超参数优化列表参数
ARGS_LIST_HYPEROPTS = ["hyperopt_path", "print_one_column"]
# 超参数优化路径、单列打印

# 回测展示参数
ARGS_BACKTEST_SHOW = ["exportfilename", "backtest_show_pair_list", "backtest_breakdown"]
# 导出文件名、回测展示交易对列表、回测详情

# 交易所列表参数
ARGS_LIST_EXCHANGES = ["print_one_column", "list_exchanges_all"]
# 单列打印、显示所有交易所

# 时间周期列表参数
ARGS_LIST_TIMEFRAMES = ["exchange", "print_one_column"]
# 交易所、单列打印

# 交易对列表参数
ARGS_LIST_PAIRS = [
    "exchange",                  # 交易所
    "print_list",                # 打印列表
    "list_pairs_print_json",     # 以JSON格式打印交易对
    "print_one_column",          # 单列打印
    "print_csv",                 # 以CSV格式打印
    "base_currencies",           # 基础货币
    "quote_currencies",          # 计价货币
    "list_pairs_all",            # 显示所有交易对
    "trading_mode",              # 交易模式
]

# 交易对列表测试参数
ARGS_TEST_PAIRLIST = [
    "user_data_dir",             # 用户数据目录
    "verbosity",                 # 详细程度
    "config",                    # 配置文件
    "quote_currencies",          # 计价货币
    "print_one_column",          # 单列打印
    "list_pairs_print_json",     # 以JSON格式打印交易对
    "exchange",                  # 交易所
]

# 创建用户目录参数
ARGS_CREATE_USERDIR = ["user_data_dir", "reset"]
# 用户数据目录、重置

# 构建配置参数
ARGS_BUILD_CONFIG = ["config"]  # 配置文件

# 显示配置参数
ARGS_SHOW_CONFIG = ["user_data_dir", "config", "show_sensitive"]
# 用户数据目录、配置文件、显示敏感信息

# 构建策略参数
ARGS_BUILD_STRATEGY = ["user_data_dir", "strategy", "strategy_path", "template"]
# 用户数据目录、策略名称、策略路径、模板

# 转换交易数据参数
ARGS_CONVERT_DATA_TRADES = ["pairs", "format_from_trades", "format_to", "erase", "exchange"]
# 交易对、源交易数据格式、目标格式、清除、交易所

# 转换数据参数
ARGS_CONVERT_DATA = ["pairs", "format_from", "format_to", "erase", "exchange"]
# 交易对、源格式、目标格式、清除、交易所

# 转换OHLCV数据参数
ARGS_CONVERT_DATA_OHLCV = [*ARGS_CONVERT_DATA, "timeframes", "trading_mode", "candle_types"]
# 继承转换数据参数，增加时间周期、交易模式、K线类型

# 转换交易数据为K线参数
ARGS_CONVERT_TRADES = [
    "pairs",                # 交易对
    "timeframes",           # 时间周期
    "exchange",             # 交易所
    "dataformat_ohlcv",     # OHLCV数据格式
    "dataformat_trades",    # 交易数据格式
    "trading_mode",         # 交易模式
]

# 数据列表参数
ARGS_LIST_DATA = [
    "exchange",             # 交易所
    "dataformat_ohlcv",     # OHLCV数据格式
    "dataformat_trades",    # 交易数据格式
    "trades",               # 交易数据
    "pairs",                # 交易对
    "trading_mode",         # 交易模式
    "show_timerange",       # 显示时间范围
]

# 下载数据参数
ARGS_DOWNLOAD_DATA = [
    "pairs",                # 交易对
    "pairs_file",           # 交易对文件
    "days",                 # 天数
    "new_pairs_days",       # 新交易对天数
    "include_inactive",     # 包含不活跃交易对
    "timerange",            # 时间范围
    "download_trades",      # 下载交易数据
    "convert_trades",       # 转换交易数据
    "exchange",             # 交易所
    "timeframes",           # 时间周期
    "erase",                # 清除现有数据
    "dataformat_ohlcv",     # OHLCV数据格式
    "dataformat_trades",    # 交易数据格式
    "trading_mode",         # 交易模式
    "prepend_data",         # 前置数据
]

# 绘制数据框参数
ARGS_PLOT_DATAFRAME = [
    "pairs",                # 交易对
    "indicators1",          # 指标1
    "indicators2",          # 指标2
    "plot_limit",           # 绘图限制
    "db_url",               # 数据库URL
    "trade_source",         # 交易数据源
    "export",               # 导出
    "exportfilename",       # 导出文件名
    "timerange",            # 时间范围
    "timeframe",            # 时间周期
    "no_trades",            # 不显示交易
]

# 绘制收益参数
ARGS_PLOT_PROFIT = [
    "pairs",                # 交易对
    "timerange",            # 时间范围
    "export",               # 导出
    "exportfilename",       # 导出文件名
    "db_url",               # 数据库URL
    "trade_source",         # 交易数据源
    "timeframe",            # 时间周期
    "plot_auto_open",       # 自动打开绘图
]

# 转换数据库参数
ARGS_CONVERT_DB = ["db_url", "db_url_from"]
# 目标数据库URL、源数据库URL

# 安装UI参数
ARGS_INSTALL_UI = ["erase_ui_only", "ui_prerelease", "ui_version"]
# 仅清除UI、UI预发布版本、UI版本

# 显示交易参数
ARGS_SHOW_TRADES = ["db_url", "trade_ids", "print_json"]
# 数据库URL、交易ID、以JSON格式打印

# 超参数优化列表参数
ARGS_HYPEROPT_LIST = [
    "hyperopt_list_best",              # 显示最佳结果
    "hyperopt_list_profitable",        # 显示盈利结果
    "hyperopt_list_min_trades",        # 最小交易数
    "hyperopt_list_max_trades",        # 最大交易数
    "hyperopt_list_min_avg_time",      # 最小平均交易时间
    "hyperopt_list_max_avg_time",      # 最大平均交易时间
    "hyperopt_list_min_avg_profit",    # 最小平均利润
    "hyperopt_list_max_avg_profit",    # 最大平均利润
    "hyperopt_list_min_total_profit",  # 最小总利润
    "hyperopt_list_max_total_profit",  # 最大总利润
    "hyperopt_list_min_objective",     # 最小目标值
    "hyperopt_list_max_objective",     # 最大目标值
    "print_json",                      # 以JSON格式打印
    "hyperopt_list_no_details",        # 不显示详情
    "hyperoptexportfilename",          # 超参数优化导出文件名
    "export_csv",                      # 导出为CSV
]

# 超参数优化展示参数
ARGS_HYPEROPT_SHOW = [
    "hyperopt_list_best",              # 显示最佳结果
    "hyperopt_list_profitable",        # 显示盈利结果
    "hyperopt_show_index",             # 显示索引
    "print_json",                      # 以JSON格式打印
    "hyperoptexportfilename",          # 超参数优化导出文件名
    "hyperopt_show_no_header",         # 不显示标题
    "disableparamexport",              # 禁用参数导出
    "backtest_breakdown",              # 回测详情
]

# 分析入场出场参数
ARGS_ANALYZE_ENTRIES_EXITS = [
    "exportfilename",                  # 导出文件名
    "analysis_groups",                 # 分析分组
    "enter_reason_list",               # 入场原因列表
    "exit_reason_list",                # 出场原因列表
    "indicator_list",                  # 指标列表
    "entry_only",                      # 仅分析入场
    "exit_only",                       # 仅分析出场
    "timerange",                       # 时间范围
    "analysis_rejected",               # 分析被拒绝的交易
    "analysis_to_csv",                 # 导出为CSV
    "analysis_csv_path",               # CSV导出路径
]


# 策略更新参数
ARGS_STRATEGY_UPDATER = ["strategy_list", "strategy_path", "recursive_strategy_search"]
# 策略列表、策略路径、递归搜索策略

# 前瞻分析参数
ARGS_LOOKAHEAD_ANALYSIS = [
    a
    for a in ARGS_BACKTEST
    if a not in ("position_stacking", "backtest_cache", "backtest_breakdown", "backtest_notes")
] + ["minimum_trade_amount", "targeted_trade_amount", "lookahead_analysis_exportfilename"]
# 从回测参数中排除部分参数，增加最小交易金额、目标交易金额、前瞻分析导出文件名

# 递归分析参数
ARGS_RECURSIVE_ANALYSIS = ["timeframe", "timerange", "dataformat_ohlcv", "pairs", "startup_candle"]
# 时间周期、时间范围、OHLCV数据格式、交易对、启动K线数

# 命令级配置 - 保持在上述定义的底部
# 不需要配置文件的命令
NO_CONF_REQURIED = [
    "convert-data",            # 转换数据
    "convert-trade-data",      # 转换交易数据
    "download-data",           # 下载数据
    "list-timeframes",         # 列出时间周期
    "list-markets",            # 列出市场
    "list-pairs",              # 列出交易对
    "list-strategies",         # 列出策略
    "list-freqaimodels",       # 列出FreqAI模型
    "list-hyperoptloss",       # 列出超参数优化损失函数
    "list-data",               # 列出数据
    "hyperopt-list",           # 超参数优化列表
    "hyperopt-show",           # 超参数优化展示
    "backtest-filter",         # 回测筛选
    "plot-dataframe",          # 绘制数据框
    "plot-profit",             # 绘制收益
    "show-trades",             # 显示交易
    "trades-to-ohlcv",         # 交易转OHLCV
    "strategy-updater",        # 策略更新器
]

# 允许没有配置文件的命令
NO_CONF_ALLOWED = ["create-userdir", "list-exchanges", "new-strategy"]
# 创建用户目录、列出交易所、新建策略


class Arguments:
    """
    参数类。管理命令行接收的参数
    """

    def __init__(self, args: list[str] | None) -> None:
        self.args = args
        self._parsed_arg: Namespace | None = None

    def get_parsed_arg(self) -> dict[str, Any]:
        """
        返回参数列表
        :return: 参数字典
        """
        if self._parsed_arg is None:
            self._build_subcommands()
            self._parsed_arg = self._parse_args()

        return vars(self._parsed_arg)

    def _parse_args(self) -> Namespace:
        """
        解析给定的参数并返回argparse Namespace实例。
        """
        parsed_arg = self.parser.parse_args(self.args)

        # 解决argparse中action='append'和默认值的问题
        # (参见 https://bugs.python.org/issue16399)
        # 允许某些命令不需要配置文件（如下载/绘图）
        if "config" in parsed_arg and parsed_arg.config is None:
            # 检查命令是否在不需要配置文件的列表中
            conf_required = "command" in parsed_arg and parsed_arg.command in NO_CONF_REQURIED

            if "user_data_dir" in parsed_arg and parsed_arg.user_data_dir is not None:
                user_dir = parsed_arg.user_data_dir
            else:
                # 默认情况
                user_dir = "user_data"
                # 尝试从"user_data/config.json"加载
            cfgfile = Path(user_dir) / DEFAULT_CONFIG
            if cfgfile.is_file():
                parsed_arg.config = [str(cfgfile)]
            else:
                # 否则使用"config.json"
                cfgfile = Path.cwd() / DEFAULT_CONFIG
                if cfgfile.is_file() or not conf_required:
                    parsed_arg.config = [DEFAULT_CONFIG]

        return parsed_arg

    def _build_args(self, optionlist: list[str], parser: ArgumentParser | _ArgumentGroup) -> None:
        """
        构建参数
        """
        for val in optionlist:
            opt = AVAILABLE_CLI_OPTIONS[val]
            parser.add_argument(*opt.cli, dest=val, **opt.kwargs)

    def _build_subcommands(self) -> None:
        """
        构建并附加所有子命令。
        :return: None
        """
        # 构建共享参数（作为"通用选项"组）
        _common_parser = ArgumentParser(add_help=False)
        group = _common_parser.add_argument_group("通用参数")
        self._build_args(optionlist=ARGS_COMMON, parser=group)

        _strategy_parser = ArgumentParser(add_help=False)
        strategy_group = _strategy_parser.add_argument_group("策略参数")
        self._build_args(optionlist=ARGS_STRATEGY, parser=strategy_group)

        # 构建主命令
        self.parser = ArgumentParser(
            prog="freqtrade", description="免费、开源的加密货币交易机器人"
        )
        self._build_args(optionlist=ARGS_MAIN, parser=self.parser)

        from freqtrade.commands import (
            start_analysis_entries_exits,
            start_backtesting,
            start_backtesting_show,
            start_convert_data,
            start_convert_db,
            start_convert_trades,
            start_create_userdir,
            start_download_data,
            start_edge,
            start_hyperopt,
            start_hyperopt_list,
            start_hyperopt_show,
            start_install_ui,
            start_list_data,
            start_list_exchanges,
            start_list_freqAI_models,
            start_list_hyperopt_loss_functions,
            start_list_markets,
            start_list_strategies,
            start_list_timeframes,
            start_lookahead_analysis,
            start_new_config,
            start_new_strategy,
            start_plot_dataframe,
            start_plot_profit,
            start_recursive_analysis,
            start_show_config,
            start_show_trades,
            start_strategy_update,
            start_test_pairlist,
            start_trading,
            start_webserver,
        )

        subparsers = self.parser.add_subparsers(
            dest="command",
            # 当没有添加子处理器时使用自定义消息
            # 从`main.py`显示
            # required=True
        )

        # 添加交易子命令
        trade_cmd = subparsers.add_parser(
            "trade", help="交易模块。", parents=[_common_parser, _strategy_parser]
        )
        trade_cmd.set_defaults(func=start_trading)
        self._build_args(optionlist=ARGS_TRADE, parser=trade_cmd)

        # 添加创建用户目录子命令
        create_userdir_cmd = subparsers.add_parser(
            "create-userdir",
            help="创建用户数据目录。",
        )
        create_userdir_cmd.set_defaults(func=start_create_userdir)
        self._build_args(optionlist=ARGS_CREATE_USERDIR, parser=create_userdir_cmd)

        # 添加新建配置子命令
        build_config_cmd = subparsers.add_parser(
            "new-config",
            help="创建新配置",
        )
        build_config_cmd.set_defaults(func=start_new_config)
        self._build_args(optionlist=ARGS_BUILD_CONFIG, parser=build_config_cmd)

        # 添加显示配置子命令
        show_config_cmd = subparsers.add_parser(
            "show-config",
            help="显示解析后的配置",
        )
        show_config_cmd.set_defaults(func=start_show_config)
        self._build_args(optionlist=ARGS_SHOW_CONFIG, parser=show_config_cmd)

        # 添加新建策略子命令
        build_strategy_cmd = subparsers.add_parser(
            "new-strategy",
            help="创建新策略",
        )
        build_strategy_cmd.set_defaults(func=start_new_strategy)
        self._build_args(optionlist=ARGS_BUILD_STRATEGY, parser=build_strategy_cmd)

        # 添加下载数据子命令
        download_data_cmd = subparsers.add_parser(
            "download-data",
            help="下载回测数据。",
            parents=[_common_parser],
        )
        download_data_cmd.set_defaults(func=start_download_data)
        self._build_args(optionlist=ARGS_DOWNLOAD_DATA, parser=download_data_cmd)

        # 添加转换数据子命令
        convert_data_cmd = subparsers.add_parser(
            "convert-data",
            help="将K线(OHLCV)数据从一种格式转换为另一种。",
            parents=[_common_parser],
        )
        convert_data_cmd.set_defaults(func=partial(start_convert_data, ohlcv=True))
        self._build_args(optionlist=ARGS_CONVERT_DATA_OHLCV, parser=convert_data_cmd)

        # 添加转换交易数据子命令
        convert_trade_data_cmd = subparsers.add_parser(
            "convert-trade-data",
            help="将交易数据从一种格式转换为另一种。",
            parents=[_common_parser],
        )
        convert_trade_data_cmd.set_defaults(func=partial(start_convert_data, ohlcv=False))
        self._build_args(optionlist=ARGS_CONVERT_DATA_TRADES, parser=convert_trade_data_cmd)

        # 添加交易转K线子命令
        convert_trade_data_cmd = subparsers.add_parser(
            "trades-to-ohlcv",
            help="将交易数据转换为OHLCV数据。",
            parents=[_common_parser],
        )
        convert_trade_data_cmd.set_defaults(func=start_convert_trades)
        self._build_args(optionlist=ARGS_CONVERT_TRADES, parser=convert_trade_data_cmd)

        # 添加数据列表子命令
        list_data_cmd = subparsers.add_parser(
            "list-data",
            help="列出已下载的数据。",
            parents=[_common_parser],
        )
        list_data_cmd.set_defaults(func=start_list_data)
        self._build_args(optionlist=ARGS_LIST_DATA, parser=list_data_cmd)

        # 添加回测子命令
        backtesting_cmd = subparsers.add_parser(
            "backtesting", help="回测模块。", parents=[_common_parser, _strategy_parser]
        )
        backtesting_cmd.set_defaults(func=start_backtesting)
        self._build_args(optionlist=ARGS_BACKTEST, parser=backtesting_cmd)

        # 添加回测展示子命令
        backtesting_show_cmd = subparsers.add_parser(
            "backtesting-show",
            help="显示过去的回测结果",
            parents=[_common_parser],
        )
        backtesting_show_cmd.set_defaults(func=start_backtesting_show)
        self._build_args(optionlist=ARGS_BACKTEST_SHOW, parser=backtesting_show_cmd)

        # 添加回测分析子命令
        analysis_cmd = subparsers.add_parser(
            "backtesting-analysis", help="回测分析模块。", parents=[_common_parser]
        )
        analysis_cmd.set_defaults(func=start_analysis_entries_exits)
        self._build_args(optionlist=ARGS_ANALYZE_ENTRIES_EXITS, parser=analysis_cmd)

        # 添加Edge子命令
        edge_cmd = subparsers.add_parser(
            "edge",
            help="Edge模块。不再是Freqtrade的一部分",
            parents=[_common_parser, _strategy_parser],
        )
        edge_cmd.set_defaults(func=start_edge)
        self._build_args(optionlist=ARGS_EDGE, parser=edge_cmd)

        # 添加超参数优化子命令
        hyperopt_cmd = subparsers.add_parser(
            "hyperopt",
            help="超参数优化模块。",
            parents=[_common_parser, _strategy_parser],
        )
        hyperopt_cmd.set_defaults(func=start_hyperopt)
        self._build_args(optionlist=ARGS_HYPEROPT, parser=hyperopt_cmd)

        # 添加超参数优化列表子命令
        hyperopt_list_cmd = subparsers.add_parser(
            "hyperopt-list",
            help="列出超参数优化结果",
            parents=[_common_parser],
        )
        hyperopt_list_cmd.set_defaults(func=start_hyperopt_list)
        self._build_args(optionlist=ARGS_HYPEROPT_LIST, parser=hyperopt_list_cmd)

        # 添加超参数优化展示子命令
        hyperopt_show_cmd = subparsers.add_parser(
            "hyperopt-show",
            help="显示超参数优化结果详情",
            parents=[_common_parser],
        )
        hyperopt_show_cmd.set_defaults(func=start_hyperopt_show)
        self._build_args(optionlist=ARGS_HYPEROPT_SHOW, parser=hyperopt_show_cmd)

        # 添加交易所列表子命令
        list_exchanges_cmd = subparsers.add_parser(
            "list-exchanges",
            help="打印可用的交易所。",
            parents=[_common_parser],
        )
        list_exchanges_cmd.set_defaults(func=start_list_exchanges)
        self._build_args(optionlist=ARGS_LIST_EXCHANGES, parser=list_exchanges_cmd)

        # 添加市场列表子命令
        list_markets_cmd = subparsers.add_parser(
            "list-markets",
            help="打印交易所的市场。",
            parents=[_common_parser],
        )
        list_markets_cmd.set_defaults(func=partial(start_list_markets, pairs_only=False))
        self._build_args(optionlist=ARGS_LIST_PAIRS, parser=list_markets_cmd)

        # 添加交易对列表子命令
        list_pairs_cmd = subparsers.add_parser(
            "list-pairs",
            help="打印交易所的交易对。",
            parents=[_common_parser],
        )
        list_pairs_cmd.set_defaults(func=partial(start_list_markets, pairs_only=True))
        self._build_args(optionlist=ARGS_LIST_PAIRS, parser=list_pairs_cmd)

        # 添加策略列表子命令
        list_strategies_cmd = subparsers.add_parser(
            "list-strategies",
            help="打印可用的策略。",
            parents=[_common_parser],
        )
        list_strategies_cmd.set_defaults(func=start_list_strategies)
        self._build_args(optionlist=ARGS_LIST_STRATEGIES, parser=list_strategies_cmd)

        # 添加超参数优化损失函数列表子命令
        list_hyperopt_loss_cmd = subparsers.add_parser(
            "list-hyperoptloss",
            help="打印可用的超参数优化损失函数。",
            parents=[_common_parser],
        )
        list_hyperopt_loss_cmd.set_defaults(func=start_list_hyperopt_loss_functions)
        self._build_args(optionlist=ARGS_LIST_HYPEROPTS, parser=list_hyperopt_loss_cmd)

        # 添加FreqAI模型列表子命令
        list_freqaimodels_cmd = subparsers.add_parser(
            "list-freqaimodels",
            help="打印可用的freqAI模型。",
            parents=[_common_parser],
        )
        list_freqaimodels_cmd.set_defaults(func=start_list_freqAI_models)
        self._build_args(optionlist=ARGS_LIST_FREQAIMODELS, parser=list_freqaimodels_cmd)

        # 添加时间周期列表子命令
        list_timeframes_cmd = subparsers.add_parser(
            "list-timeframes",
            help="打印交易所可用的时间周期。",
            parents=[_common_parser],
        )
        list_timeframes_cmd.set_defaults(func=start_list_timeframes)
        self._build_args(optionlist=ARGS_LIST_TIMEFRAMES, parser=list_timeframes_cmd)

        # 添加显示交易子命令
        show_trades = subparsers.add_parser(
            "show-trades",
            help="显示交易。",
            parents=[_common_parser],
        )
        show_trades.set_defaults(func=start_show_trades)
        self._build_args(optionlist=ARGS_SHOW_TRADES, parser=show_trades)

        # 添加测试交易对列表子命令
        test_pairlist_cmd = subparsers.add_parser(
            "test-pairlist",
            help="测试你的交易对列表配置。",
        )
        test_pairlist_cmd.set_defaults(func=start_test_pairlist)
        self._build_args(optionlist=ARGS_TEST_PAIRLIST, parser=test_pairlist_cmd)

        # 添加数据库转换子命令
        convert_db = subparsers.add_parser(
            "convert-db",
            help="迁移数据库到不同的系统",
        )
        convert_db.set_defaults(func=start_convert_db)
        self._build_args(optionlist=ARGS_CONVERT_DB, parser=convert_db)

        # 添加安装UI子命令
        install_ui_cmd = subparsers.add_parser(
            "install-ui",
            help="安装FreqUI",
        )
        install_ui_cmd.set_defaults(func=start_install_ui)
        self._build_args(optionlist=ARGS_INSTALL_UI, parser=install_ui_cmd)

        # 添加绘图子命令
        plot_dataframe_cmd = subparsers.add_parser(
            "plot-dataframe",
            help="绘制带有指标的K线。",
            parents=[_common_parser, _strategy_parser],
        )
        plot_dataframe_cmd.set_defaults(func=start_plot_dataframe)
        self._build_args(optionlist=ARGS_PLOT_DATAFRAME, parser=plot_dataframe_cmd)

        # 绘制收益
        plot_profit_cmd = subparsers.add_parser(
            "plot-profit",
            help="生成显示收益的图表。",
            parents=[_common_parser, _strategy_parser],
        )
        plot_profit_cmd.set_defaults(func=start_plot_profit)
        self._build_args(optionlist=ARGS_PLOT_PROFIT, parser=plot_profit_cmd)

        # 添加网页服务器子命令
        webserver_cmd = subparsers.add_parser(
            "webserver", help="网页服务器模块。", parents=[_common_parser]
        )
        webserver_cmd.set_defaults(func=start_webserver)
        self._build_args(optionlist=ARGS_WEBSERVER, parser=webserver_cmd)

        # 添加策略更新器子命令
        strategy_updater_cmd = subparsers.add_parser(
            "strategy-updater",
            help="将过时的策略文件更新到当前版本",
            parents=[_common_parser],
        )
        strategy_updater_cmd.set_defaults(func=start_strategy_update)
        self._build_args(optionlist=ARGS_STRATEGY_UPDATER, parser=strategy_updater_cmd)

        # 添加前瞻分析子命令
        lookahead_analayis_cmd = subparsers.add_parser(
            "lookahead-analysis",
            help="检查潜在的前瞻偏差。",
            parents=[_common_parser, _strategy_parser],
        )
        lookahead_analayis_cmd.set_defaults(func=start_lookahead_analysis)

        self._build_args(optionlist=ARGS_LOOKAHEAD_ANALYSIS, parser=lookahead_analayis_cmd)

        # 添加递归分析子命令
        recursive_analayis_cmd = subparsers.add_parser(
            "recursive-analysis",
            help="检查潜在的递归公式问题。",
            parents=[_common_parser, _strategy_parser],
        )
        recursive_analayis_cmd.set_defaults(func=start_recursive_analysis)

        self._build_args(optionlist=ARGS_RECURSIVE_ANALYSIS, parser=recursive_analayis_cmd)