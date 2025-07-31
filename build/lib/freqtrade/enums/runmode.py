from enum import Enum


class RunMode(str, Enum):
    """
    机器人运行模式（回测、超参数优化等）
    可选值："live"（实盘）、"dry-run"（模拟盘）、"backtest"（回测）、"hyperopt"（超参数优化）。
    """

    LIVE = "live"  # 实盘交易模式
    DRY_RUN = "dry_run"  # 模拟盘交易模式
    BACKTEST = "backtest"  # 回测模式
    HYPEROPT = "hyperopt"  # 超参数优化模式
    UTIL_EXCHANGE = "util_exchange"  # 带交易所连接的工具模式
    UTIL_NO_EXCHANGE = "util_no_exchange"  # 无交易所连接的工具模式
    PLOT = "plot"  # 绘图模式
    WEBSERVER = "webserver"  # 网页服务器模式
    OTHER = "other"  # 其他模式


# 交易相关模式
TRADE_MODES = [RunMode.LIVE, RunMode.DRY_RUN]
# 优化相关模式
OPTIMIZE_MODES = [RunMode.BACKTEST, RunMode.HYPEROPT]
# 非工具类模式
NON_UTIL_MODES = TRADE_MODES + OPTIMIZE_MODES