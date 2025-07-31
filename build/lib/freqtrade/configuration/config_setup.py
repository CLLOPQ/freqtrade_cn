import logging
from typing import Any

from freqtrade.enums import RunMode

from .config_validation import validate_config_consistency
from .configuration import Configuration


logger = logging.getLogger(__name__)


def setup_utils_configuration(
    args: dict[str, Any], method: RunMode, *, set_dry: bool = True
) -> dict[str, Any]:
    """
    为工具子命令准备配置
    :param args: 来自Arguments()的命令行参数
    :param method: 机器人运行模式
    :return: 配置字典
    """
    configuration = Configuration(args, method)
    config = configuration.get_config()

    # 确保这些模式使用模拟运行
    if set_dry:
        config["dry_run"] = True
    validate_config_consistency(config, preliminary=True)

    return config