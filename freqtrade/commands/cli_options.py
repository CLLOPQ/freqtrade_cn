"""
在 arguments.py 中使用的命令行参数定义
"""

from argparse import SUPPRESS, ArgumentTypeError

from freqtrade import constants
from freqtrade.constants import HYPEROPT_LOSS_BUILTIN
from freqtrade.enums import CandleType


def check_int_positive(value: str) -> int:
    """检查是否为正整数"""
    try:
        uint = int(value)
        if uint <= 0:
            raise ValueError
    except ValueError:
        raise ArgumentTypeError(
            f"{value} 对该参数无效，应为正整数值"
        )
    return uint


def check_int_nonzero(value: str) -> int:
    """检查是否为非零整数"""
    try:
        uint = int(value)
        if uint == 0:
            raise ValueError
    except ValueError:
        raise ArgumentTypeError(
            f"{value} 对该参数无效，应为非零整数值"
        )
    return uint


class Arg:
    """可选的命令行参数"""
    def __init__(self, *args, **kwargs):
        self.cli = args
        self.kwargs = kwargs


# 可用的命令行选项列表
AVAILABLE_CLI_OPTIONS = {
    # 通用选项
    "verbosity": Arg(
        "-v",
        "--verbose",
        help="详细模式（-vv 更详细，-vvv 显示所有消息）。",
        action="count",
    ),
    "logfile": Arg(
        "--logfile",
        "--log-file",
        help="将日志输出到指定文件。特殊值包括：'syslog'、'journald'。"
        "更多详情请参见文档。",
        metavar="文件",
    ),
    "version": Arg(
        "-V",
        "--version",
        help="显示程序版本号并退出",
        action="store_true",
    ),
    "version_main": Arg(
        # 版本参数的副本 - 用于在有或没有子命令时都能使用 -V
        "-V",
        "--version",
        help="显示程序版本号并退出",
        action="store_true",
    ),
    "config": Arg(
        "-c",
        "--config",
        help=f"指定配置文件（默认：`userdir/{constants.DEFAULT_CONFIG}` "
        f"或 `config.json`，取存在的那个）。"
        f"可以使用多个 --config 选项。"
        f"设置为 `-` 可从标准输入读取配置。",
        action="append",
        metavar="路径",
    ),
    "datadir": Arg(
        "-d",
        "--datadir",
        "--data-dir",
        help="包含交易所历史回测数据的基础目录路径。"
        "要查看期货数据，需额外使用 trading-mode 参数。",
        metavar="路径",
    ),
    "user_data_dir": Arg(
        "--userdir",
        "--user-data-dir",
        help="用户数据目录路径。",
        metavar="路径",
    ),
    "reset": Arg(
        "--reset",
        help="将示例文件重置为原始状态。",
        action="store_true",
    ),
    "recursive_strategy_search": Arg(
        "--recursive-strategy-search",
        help="在策略文件夹中递归搜索策略。",
        action="store_true",
    ),
    # 主要选项
    "strategy": Arg(
        "-s",
        "--strategy",
        help="指定机器人将使用的策略类名。",
        metavar="名称",
    ),
    "strategy_path": Arg(
        "--strategy-path",
        help="指定额外的策略查找路径。",
        metavar="路径",
    ),
    "db_url": Arg(
        "--db-url",
        help=f"覆盖交易数据库URL，这在自定义部署中很有用 "
        f"（默认：实盘模式使用 `{constants.DEFAULT_DB_PROD_URL}`，"
        f"模拟模式使用 `{constants.DEFAULT_DB_DRYRUN_URL}`）。",
        metavar="路径",
    ),
    "db_url_from": Arg(
        "--db-url-from",
        help="迁移数据库时使用的源数据库URL。",
        metavar="路径",
    ),
    "sd_notify": Arg(
        "--sd-notify",
        help="通知systemd服务管理器。",
        action="store_true",
    ),
    "dry_run": Arg(
        "--dry-run",
        help="强制以模拟模式交易（移除交易所密钥并模拟交易）。",
        action="store_true",
    ),
    "dry_run_wallet": Arg(
        "--dry-run-wallet",
        "--starting-balance",
        help="初始资金，用于回测/超参数优化和模拟交易。",
        type=float,
    ),
    # 通用优化选项
    "timeframe": Arg(
        "-i",
        "--timeframe",
        help="指定时间周期（`1m`、`5m`、`30m`、`1h`、`1d`）。",
    ),
    "timerange": Arg(
        "--timerange",
        help="指定要使用的数据时间范围。",
    ),
    "max_open_trades": Arg(
        "--max-open-trades",
        help="覆盖配置中的 `max_open_trades` 设置值。",
        type=int,
        metavar="整数",
    ),
    "stake_amount": Arg(
        "--stake-amount",
        help="覆盖配置中的 `stake_amount` 设置值。",
    ),
    # 回测选项
    "timeframe_detail": Arg(
        "--timeframe-detail",
        help="指定回测的详细时间周期（`1m`、`5m`、`30m`、`1h`、`1d`）。",
    ),
    "position_stacking": Arg(
        "--eps",
        "--enable-position-stacking",
        help="允许多次购买同一交易对（持仓叠加）。",
        action="store_true",
        default=False,
    ),
    "backtest_show_pair_list": Arg(
        "--show-pair-list",
        help="按利润排序显示回测交易对列表。",
        action="store_true",
        default=False,
    ),
    "enable_protections": Arg(
        "--enable-protections",
        "--enableprotections",
        help="为回测启用保护机制。"
        "这会显著减慢回测速度，但会包含配置的保护机制",
        action="store_true",
        default=False,
    ),
    "strategy_list": Arg(
        "--strategy-list",
        help="提供一个空格分隔的策略列表用于回测。"
        "请注意，时间周期需要在配置中设置或通过命令行指定。"
        "当与 `--export trades` 一起使用时，策略名将被注入到文件名中 "
        "（例如 `backtest-data.json` 变为 `backtest-data-SampleStrategy.json`）",
        nargs="+",
    ),
    "export": Arg(
        "--export",
        help="导出回测结果（默认：交易）。",
        choices=constants.EXPORT_OPTIONS,
    ),
    "backtest_notes": Arg(
        "--notes",
        help="向后测结果添加备注。",
        metavar="文本",
    ),
    "exportfilename": Arg(
        "--export-filename",
        "--backtest-filename",
        help="用于回测结果的文件名。"
        "需要同时设置 `--export`。"
        "示例：`--export-filename=user_data/backtest_results/backtest_today.json`",
        metavar="路径",
    ),
    "disableparamexport": Arg(
        "--disable-param-export",
        help="禁用超参数优化参数的自动导出。",
        action="store_true",
    ),
    "fee": Arg(
        "--fee",
        help="指定手续费比例。将在交易入场和出场时各应用一次。",
        type=float,
        metavar="浮点数",
    ),
    "backtest_breakdown": Arg(
        "--breakdown",
        help="按 [日、周、月、年] 显示回测详情。",
        nargs="+",
        choices=constants.BACKTEST_BREAKDOWNS,
    ),
    "backtest_cache": Arg(
        "--cache",
        help="加载不超过指定时长的缓存回测结果（默认：%(default)s）。",
        default=constants.BACKTEST_CACHE_DEFAULT,
        choices=constants.BACKTEST_CACHE_AGE,
    ),
    # 超参数优化选项
    "hyperopt": Arg(
        "--hyperopt",
        help=SUPPRESS,
        metavar="名称",
        required=False,
    ),
    "hyperopt_path": Arg(
        "--hyperopt-path",
        help="指定超参数优化损失函数的额外查找路径。",
        metavar="路径",
    ),
    "epochs": Arg(
        "-e",
        "--epochs",
        help="指定迭代次数（默认：%(default)d）。",
        type=check_int_positive,
        metavar="整数",
        default=constants.HYPEROPT_EPOCH,
    ),
    "early_stop": Arg(
        "--early-stop",
        help="如果在指定迭代次数（默认：%(default)d）后没有改进，则提前停止超参数优化。",
        type=check_int_positive,
        metavar="整数",
        default=0,  # 默认禁用
    ),
    "spaces": Arg(
        "--spaces",
        help="指定要进行超参数优化的参数。空格分隔的列表。",
        choices=[
            "all",
            "buy",
            "sell",
            "roi",
            "stoploss",
            "trailing",
            "protection",
            "trades",
            "default",
        ],
        nargs="+",
        default="default",
    ),
    "analyze_per_epoch": Arg(
        "--analyze-per-epoch",
        help="每个迭代轮次运行一次 populate_indicators。",
        action="store_true",
        default=False,
    ),
    "print_all": Arg(
        "--print-all",
        help="打印所有结果，而不仅仅是最佳结果。",
        action="store_true",
        default=False,
    ),
    "print_colorized": Arg(
        "--no-color",
        help="禁用超参数优化结果的彩色显示。如果将输出重定向到文件，这可能很有用。",
        action="store_false",
        default=True,
    ),
    "print_json": Arg(
        "--print-json",
        help="以JSON格式打印输出。",
        action="store_true",
        default=False,
    ),
    "export_csv": Arg(
        "--export-csv",
        help="导出到CSV文件。"
        "这将禁用表格打印。"
        "示例：--export-csv hyperopt.csv",
        metavar="文件",
    ),
    "hyperopt_jobs": Arg(
        "-j",
        "--job-workers",
        help="超参数优化的并发运行作业数量 "
        "（超参数优化工作进程）。"
        "如果为-1（默认），则使用所有CPU，-2则使用除一个外的所有CPU，依此类推。"
        "如果为1，则完全不使用并行计算代码。",
        type=int,
        metavar="作业数",
        default=-1,
    ),
    "hyperopt_random_state": Arg(
        "--random-state",
        help="设置随机状态为某个正整数，以获得可重现的超参数优化结果。",
        type=check_int_positive,
        metavar="整数",
    ),
    "hyperopt_min_trades": Arg(
        "--min-trades",
        help="设置超参数优化路径中评估的最小交易数量（默认：1）。",
        type=check_int_positive,
        metavar="整数",
        default=1,
    ),
    "hyperopt_loss": Arg(
        "--hyperopt-loss",
        "--hyperoptloss",
        help="指定超参数优化损失函数类（IHyperOptLoss）的类名。"
        "不同的函数可以产生完全不同的结果，因为优化目标不同。"
        "内置的超参数优化损失函数有："
        f"{', '.join(HYPEROPT_LOSS_BUILTIN)}",
        metavar="名称",
    ),
    "hyperoptexportfilename": Arg(
        "--hyperopt-filename",
        help="超参数优化结果文件名。"
        "示例：`--hyperopt-filename=hyperopt_results_2020-09-27_16-20-48.pickle`",
        metavar="文件名",
    ),
    # 列出交易所选项
    "print_one_column": Arg(
        "-1",
        "--one-column",
        help="以单列格式打印输出。",
        action="store_true",
    ),
    "list_exchanges_all": Arg(
        "-a",
        "--all",
        help="打印ccxt库已知的所有交易所。",
        action="store_true",
    ),
    # 列出交易对/市场选项
    "list_pairs_all": Arg(
        "-a",
        "--all",
        help="打印所有交易对或市场符号。默认只显示活跃的。",
        action="store_true",
    ),
    "print_list": Arg(
        "--print-list",
        help="打印交易对或市场符号列表。默认以表格格式打印数据。",
        action="store_true",
    ),
    "list_pairs_print_json": Arg(
        "--print-json",
        help="以JSON格式打印交易对或市场符号列表。",
        action="store_true",
        default=False,
    ),
    "print_csv": Arg(
        "--print-csv",
        help="以CSV格式打印交易所交易对或市场数据。",
        action="store_true",
    ),
    "quote_currencies": Arg(
        "--quote",
        help="指定计价货币。空格分隔的列表。",
        nargs="+",
        metavar="计价货币",
    ),
    "base_currencies": Arg(
        "--base",
        help="指定基础货币。空格分隔的列表。",
        nargs="+",
        metavar="基础货币",
    ),
    "trading_mode": Arg(
        "--trading-mode",
        "--tradingmode",
        help="选择交易模式",
        choices=constants.TRADING_MODES,
    ),
    "candle_types": Arg(
        "--candle-types",
        help="选择要转换的K线类型。默认为所有可用类型。",
        choices=[c.value for c in CandleType],
        nargs="+",
    ),
    # 脚本选项
    "pairs": Arg(
        "-p",
        "--pairs",
        help="将命令限制为这些交易对。交易对之间用空格分隔。",
        nargs="+",
    ),
    # 下载数据选项
    "pairs_file": Arg(
        "--pairs-file",
        help="包含交易对列表的文件。"
        "优先于 --pairs 或配置中设置的交易对。",
        metavar="文件",
    ),
    "days": Arg(
        "--days",
        help="下载指定天数的数据。",
        type=check_int_positive,
        metavar="整数",
    ),
    "include_inactive": Arg(
        "--include-inactive-pairs",
        help="也下载非活跃交易对的数据。",
        action="store_true",
    ),
    "new_pairs_days": Arg(
        "--new-pairs-days",
        help="下载新交易对指定天数的数据。默认：`%(default)s`。",
        type=check_int_positive,
        metavar="整数",
    ),
    "download_trades": Arg(
        "--dl-trades",
        help="下载交易数据而不是OHLCV数据。",
        action="store_true",
    ),
    "trades": Arg(
        "--trades",
        help="处理交易数据而不是OHLCV数据。",
        action="store_true",
    ),
    "convert_trades": Arg(
        "--convert",
        help="将下载的交易数据转换为OHLCV数据。仅在与 "
        "`--dl-trades` 结合使用时有效。"
        "对于没有历史OHLCV数据的交易所（如Kraken），这将自动进行。"
        "如果未提供此选项，请使用 `trades-to-ohlcv` 将交易数据转换为OHLCV数据。",
        action="store_true",
    ),
    "format_from_trades": Arg(
        "--format-from",
        help="数据转换的源格式。",
        choices=[*constants.AVAILABLE_DATAHANDLERS, "kraken_csv"],
        required=True,
    ),
    "format_from": Arg(
        "--format-from",
        help="数据转换的源格式。",
        choices=constants.AVAILABLE_DATAHANDLERS,
        required=True,
    ),
    "format_to": Arg(
        "--format-to",
        help="数据转换的目标格式。",
        choices=constants.AVAILABLE_DATAHANDLERS,
        required=True,
    ),
    "dataformat_ohlcv": Arg(
        "--data-format-ohlcv",
        help="下载的K线（OHLCV）数据的存储格式。（默认：`feather`）。",
        choices=constants.AVAILABLE_DATAHANDLERS,
    ),
    "dataformat_trades": Arg(
        "--data-format-trades",
        help="下载的交易数据的存储格式。（默认：`feather`）。",
        choices=constants.AVAILABLE_DATAHANDLERS,
    ),
    "show_timerange": Arg(
        "--show-timerange",
        help="显示可用数据的时间范围。（计算可能需要一段时间）。",
        action="store_true",
    ),
    "exchange": Arg(
        "--exchange",
        help="交易所名称。仅在未提供配置时有效。",
    ),
    "timeframes": Arg(
        "-t",
        "--timeframes",
        help="指定要下载的时间周期。空格分隔的列表。默认：`1m 5m`。",
        nargs="+",
    ),
    "prepend_data": Arg(
        "--prepend",
        help="允许数据前置。（数据追加被禁用）",
        action="store_true",
    ),
    "erase": Arg(
        "--erase",
        help="清除所选交易所/交易对/时间周期的所有现有数据。",
        action="store_true",
    ),
    "erase_ui_only": Arg(
        "--erase",
        help="清理UI文件夹，不下载新版本。",
        action="store_true",
        default=False,
    ),
    "ui_version": Arg(
        "--ui-version",
        help=(
            "指定要安装的FreqUI的特定版本。"
            "不指定则安装最新版本。"
        ),
        type=str,
    ),
    "ui_prerelease": Arg(
        "--prerelease",
        help=(
            "安装FreqUI的最新预发布版本。"
            "不推荐用于生产环境。"
        ),
        action="store_true",
        default=False,
    ),
    # 模板选项
    "template": Arg(
        "--template",
        help="使用模板，可以是 `minimal`（最小化）、"
        "`full`（包含多个示例指标）或 `advanced`（高级）。默认：`%(default)s`。",
        choices=["full", "minimal", "advanced"],
        default="full",
    ),
    # 绘制数据框选项
    "indicators1": Arg(
        "--indicators1",
        help="设置要在图表第一行显示的策略指标。"
        "空格分隔的列表。示例：`ema3 ema5`。默认：`['sma', 'ema3', 'ema5']`。",
        nargs="+",
    ),
    "indicators2": Arg(
        "--indicators2",
        help="设置要在图表第三行显示的策略指标。"
        "空格分隔的列表。示例：`fastd fastk`。默认：`['macd', 'macdsignal']`。",
        nargs="+",
    ),
    "plot_limit": Arg(
        "--plot-limit",
        help="指定绘图的K线限制。注意：过高的值会导致文件过大。"
        "默认：%(default)s。",
        type=check_int_positive,
        metavar="整数",
        default=750,
    ),
    "plot_auto_open": Arg(
        "--auto-open",
        help="自动打开生成的图表。",
        action="store_true",
    ),
    "no_trades": Arg(
        "--no-trades",
        help="跳过使用回测文件和数据库中的交易数据。",
        action="store_true",
    ),
    "trade_source": Arg(
        "--trade-source",
        help="指定交易数据的来源（可以是数据库或文件（回测文件））"
        "默认：%(default)s",
        choices=["DB", "file"],
        default="file",
    ),
    "trade_ids": Arg(
        "--trade-ids",
        help="指定交易ID列表。",
        nargs="+",
    ),
    # 超参数优化列表、超参数优化展示选项
    "hyperopt_list_profitable": Arg(
        "--profitable",
        help="只选择盈利的迭代轮次。",
        action="store_true",
    ),
    "hyperopt_list_best": Arg(
        "--best",
        help="只选择最佳的迭代轮次。",
        action="store_true",
    ),
    "hyperopt_list_min_trades": Arg(
        "--min-trades",
        help="选择交易数超过INT的迭代轮次。",
        type=check_int_positive,
        metavar="整数",
    ),
    "hyperopt_list_max_trades": Arg(
        "--max-trades",
        help="选择交易数少于INT的迭代轮次。",
        type=check_int_positive,
        metavar="整数",
    ),
    "hyperopt_list_min_avg_time": Arg(
        "--min-avg-time",
        help="选择平均时间以上的迭代轮次。",
        type=float,
        metavar="浮点数",
    ),
    "hyperopt_list_max_avg_time": Arg(
        "--max-avg-time",
        help="选择平均时间以下的迭代轮次。",
        type=float,
        metavar="浮点数",
    ),
    "hyperopt_list_min_avg_profit": Arg(
        "--min-avg-profit",
        help="选择平均利润以上的迭代轮次。",
        type=float,
        metavar="浮点数",
    ),
    "hyperopt_list_max_avg_profit": Arg(
        "--max-avg-profit",
        help="选择平均利润以下的迭代轮次。",
        type=float,
        metavar="浮点数",
    ),
    "hyperopt_list_min_total_profit": Arg(
        "--min-total-profit",
        help="选择总利润以上的迭代轮次。",
        type=float,
        metavar="浮点数",
    ),
    "hyperopt_list_max_total_profit": Arg(
        "--max-total-profit",
        help="选择总利润以下的迭代轮次。",
        type=float,
        metavar="浮点数",
    ),
    "hyperopt_list_min_objective": Arg(
        "--min-objective",
        help="选择目标值以上的迭代轮次。",
        type=float,
        metavar="浮点数",
    ),
    "hyperopt_list_max_objective": Arg(
        "--max-objective",
        help="选择目标值以下的迭代轮次。",
        type=float,
        metavar="浮点数",
    ),
    "hyperopt_list_no_details": Arg(
        "--no-details",
        help="不打印最佳迭代轮次的详情。",
        action="store_true",
    ),
    "hyperopt_show_index": Arg(
        "-n",
        "--index",
        help="指定要打印详情的迭代轮次索引。",
        type=check_int_nonzero,
        metavar="整数",
    ),
    "hyperopt_show_no_header": Arg(
        "--no-header",
        help="不打印迭代轮次详情的标题。",
        action="store_true",
    ),
    "hyperopt_ignore_missing_space": Arg(
        "--ignore-missing-spaces",
        "--ignore-unparameterized-spaces",
        help=(
            "对任何请求的不包含任何参数的超参数优化空间，抑制错误。"
        ),
        action="store_true",
    ),
    "analysis_groups": Arg(
        "--analysis-groups",
        help=(
            "分组输出 - "
            "0: 按入场标签的简单盈亏，"
            "1: 按入场标签，"
            "2: 按入场标签和出场标签，"
            "3: 按交易对和入场标签，"
            "4: 按交易对、入场和出场标签（可能会非常大），"
            "5: 按出场标签"
        ),
        nargs="+",
        default=[],
        choices=["0", "1", "2", "3", "4", "5"],
    ),
    "enter_reason_list": Arg(
        "--enter-reason-list",
        help=(
            "要分析的入场信号的空格分隔列表。默认：全部。"
            "例如：'entry_tag_a entry_tag_b'"
        ),
        nargs="+",
        default=["all"],
    ),
    "exit_reason_list": Arg(
        "--exit-reason-list",
        help=(
            "要分析的出场信号的空格分隔列表。默认：全部。"
            "例如：'exit_tag_a roi stop_loss trailing_stop_loss'"
        ),
        nargs="+",
        default=["all"],
    ),
    "indicator_list": Arg(
        "--indicator-list",
        help=(
            "要分析的指标的空格分隔列表。"
            "例如：'close rsi bb_lowerband profit_abs'"
        ),
        nargs="+",
        default=[],
    ),
    "entry_only": Arg(
        "--entry-only", help=("仅分析入场信号。"), action="store_true", default=False
    ),
    "exit_only": Arg(
        "--exit-only", help=("仅分析出场信号。"), action="store_true", default=False
    ),
    "analysis_rejected": Arg(
        "--rejected-signals",
        help="分析被拒绝的信号",
        action="store_true",
    ),
    "analysis_to_csv": Arg(
        "--analysis-to-csv",
        help="将所选分析表保存为单独的CSV文件",
        action="store_true",
    ),
    "analysis_csv_path": Arg(
        "--analysis-csv-path",
        help=(
            "如果启用了--analysis-to-csv，指定保存分析CSV的路径。"
            "默认：user_data/backtesting_results/"
        ),
    ),
    "freqaimodel": Arg(
        "--freqaimodel",
        help="指定自定义的freqai模型。",
        metavar="名称",
    ),
    "freqaimodel_path": Arg(
        "--freqaimodel-path",
        help="指定freqai模型的额外查找路径。",
        metavar="路径",
    ),
    "freqai_backtest_live_models": Arg(
        "--freqai-backtest-live-models", help="使用已准备好的模型运行回测。", action="store_true"
    ),
    "minimum_trade_amount": Arg(
        "--minimum-trade-amount",
        help="前瞻分析的最小交易数量",
        type=check_int_positive,
        metavar="整数",
    ),
    "targeted_trade_amount": Arg(
        "--targeted-trade-amount",
        help="前瞻分析的目标交易数量",
        type=check_int_positive,
        metavar="整数",
    ),
    "lookahead_analysis_exportfilename": Arg(
        "--lookahead-analysis-exportfilename",
        help="使用此csv文件名存储前瞻分析结果",
        type=str,
    ),
    "startup_candle": Arg(
        "--startup-candle",
        help="指定要检查的启动K线数量（`199`、`499`、`999`、`1999`）。",
        nargs="+",
    ),
    "show_sensitive": Arg(
        "--show-sensitive",
        help="在输出中显示敏感信息。",
        action="store_true",
        default=False,
    ),
}