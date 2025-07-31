from typing import Any

from freqtrade.enums import RunMode
from freqtrade.exceptions import ConfigurationError


def validate_plot_args(args: dict[str, Any]) -> None:
    """验证绘图参数"""
    if not args.get("datadir") and not args.get("config"):
        raise ConfigurationError(
            "对于 plot-profit 和 plot-dataframe，您需要指定 `--datadir` 或 `--config`。"
        )


def start_plot_dataframe(args: dict[str, Any]) -> None:
    """
    数据框绘图的入口点
    """
    # 在此处导入以避免绘图依赖未安装时出错
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.plot.plotting import load_and_plot_trades

    validate_plot_args(args)
    config = setup_utils_configuration(args, RunMode.PLOT)

    load_and_plot_trades(config)


def start_plot_profit(args: dict[str, Any]) -> None:
    """
    利润绘图的入口点
    """
    # 在此处导入以避免绘图依赖未安装时出错
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.plot.plotting import plot_profit

    validate_plot_args(args)
    config = setup_utils_configuration(args, RunMode.PLOT)

    plot_profit(config)