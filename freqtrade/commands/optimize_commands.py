import logging
from typing import Any

from freqtrade import constants
from freqtrade.enums import RunMode
from freqtrade.exceptions import ConfigurationError, OperationalException


logger = logging.getLogger(__name__)


def setup_optimize_configuration(args: dict[str, Any], method: RunMode) -> dict[str, Any]:
    """
    准备超参数优化模块的配置
    :param args: 来自Arguments()的命令行参数
    :param method: 机器人运行模式
    :return: 配置字典
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.util import fmt_coin, get_dry_run_wallet

    config = setup_utils_configuration(args, method)

    # 不支持无限模式的运行模式
    no_unlimited_runmodes = {
        RunMode.BACKTEST: "回测",
        RunMode.HYPEROPT: "超参数优化",
    }
    if method in no_unlimited_runmodes.keys():
        # 计算可用钱包大小
        wallet_size = get_dry_run_wallet(config) * config["tradable_balance_ratio"]
        # 检查持仓金额是否合理
        if (
            config["stake_amount"] != constants.UNLIMITED_STAKE_AMOUNT
            and config["stake_amount"] > wallet_size
        ):
            wallet = fmt_coin(wallet_size, config["stake_currency"])
            stake = fmt_coin(config["stake_amount"], config["stake_currency"])
            raise ConfigurationError(
                f"初始资金 ({wallet}) 小于持仓金额 {stake}。"
                f"资金计算方式为 `dry_run_wallet * tradable_balance_ratio`。"
            )

    return config


def start_backtesting(args: dict[str, Any]) -> None:
    """
    启动回测脚本
    :param args: 来自Arguments()的命令行参数
    :return: None
    """
    # 此处导入以避免在不使用时加载回测模块
    from freqtrade.optimize.backtesting import Backtesting

    # 初始化配置
    config = setup_optimize_configuration(args, RunMode.BACKTEST)

    logger.info("以回测模式启动freqtrade")

    # 初始化回测对象
    backtesting = Backtesting(config)
    backtesting.start()


def start_backtesting_show(args: dict[str, Any]) -> None:
    """
    显示之前的回测结果
    """
    from freqtrade.configuration import setup_utils_configuration

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    from freqtrade.data.btanalysis import load_backtest_stats
    from freqtrade.optimize.optimize_reports import show_backtest_results, show_sorted_pairlist

    results = load_backtest_stats(config["exportfilename"])

    show_backtest_results(config, results)
    show_sorted_pairlist(config, results)


def start_hyperopt(args: dict[str, Any]) -> None:
    """
    启动超参数优化脚本
    :param args: 来自Arguments()的命令行参数
    :return: None
    """
    # 此处导入以避免在不使用时加载超参数优化模块
    try:
        from filelock import FileLock, Timeout

        from freqtrade.optimize.hyperopt import Hyperopt
    except ImportError as e:
        raise OperationalException(
            f"{e}。请确保已安装超参数优化所需的依赖。"
        ) from e
    # 初始化配置
    config = setup_optimize_configuration(args, RunMode.HYPEROPT)

    logger.info("以超参数优化模式启动freqtrade")

    lock = FileLock(Hyperopt.get_lock_filename(config))

    try:
        with lock.acquire(timeout=1):
            # 移除嘈杂的日志消息
            logging.getLogger("hyperopt.tpe").setLevel(logging.WARNING)
            logging.getLogger("filelock").setLevel(logging.WARNING)

            # 初始化超参数优化对象
            hyperopt = Hyperopt(config)
            hyperopt.start()

    except Timeout:
        logger.info("检测到另一个正在运行的freqtrade超参数优化实例。")
        logger.info(
            "不支持同时执行多个超参数优化命令。"
            "超参数优化模块消耗资源较大。请按顺序运行超参数优化"
            "或在不同的机器上运行。"
        )
        logger.info("现在退出。")


def start_edge(args: dict[str, Any]) -> None:
    """
    启动Edge脚本（已弃用）
    :param args: 来自Arguments()的命令行参数
    :return: None
    """
    raise ConfigurationError(
        "Edge模块已在2023.9版本中弃用，并在2025.6版本中移除。"
        "Edge的所有功能已被移除。"
    )


def start_lookahead_analysis(args: dict[str, Any]) -> None:
    """
    启动回测偏差测试脚本
    :param args: 来自Arguments()的命令行参数
    :return: None
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.optimize.analysis.lookahead_helpers import LookaheadAnalysisSubFunctions

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)
    LookaheadAnalysisSubFunctions.start(config)


def start_recursive_analysis(args: dict[str, Any]) -> None:
    """
    启动回测递归测试脚本
    :param args: 来自Arguments()的命令行参数
    :return: None
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.optimize.analysis.recursive_helpers import RecursiveAnalysisSubFunctions

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)
    RecursiveAnalysisSubFunctions.start(config)