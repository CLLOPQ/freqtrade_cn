import logging
from typing import Any

from freqtrade.enums import RunMode


logger = logging.getLogger(__name__)


def start_analysis_entries_exits(args: dict[str, Any]) -> None:
    """
    启动分析脚本
    :param args: 来自Arguments()的命令行参数
    :return: 无返回值
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.data.entryexitanalysis import process_entry_exit_reasons

    # 初始化配置
    config = setup_utils_configuration(args, RunMode.BACKTEST)

    logger.info("以分析模式启动freqtrade")

    process_entry_exit_reasons(config)