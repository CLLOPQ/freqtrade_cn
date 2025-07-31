import logging
from pathlib import Path
from typing import Any

from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


def start_new_config(args: dict[str, Any]) -> None:
    """
    从模板创建新配置
    通过用户问题以相应方式填写入模板
    """

    from freqtrade.configuration.deploy_config import (
        ask_user_config,
        ask_user_overwrite,
        deploy_new_config,
    )
    from freqtrade.configuration.directory_operations import chown_user_directory

    config_path = Path(args["config"][0])
    chown_user_directory(config_path.parent)
    if config_path.exists():
        overwrite = ask_user_overwrite(config_path)
        if overwrite:
            config_path.unlink()
        else:
            raise OperationalException(
                f"配置文件 `{config_path}` 已存在。"
                "请删除它或使用不同的配置文件名。"
            )
    selections = ask_user_config()
    deploy_new_config(config_path, selections)


def start_show_config(args: dict[str, Any]) -> None:
    from freqtrade.configuration import sanitize_config
    from freqtrade.configuration.config_setup import setup_utils_configuration

    config = setup_utils_configuration(args, RunMode.UTIL_EXCHANGE, set_dry=False)

    print("合并的合并配置如下：")
    config_sanitized = sanitize_config(
        config["original_config"], show_sensitive=args.get("show_sensitive", False)
    )

    from rich import print_json

    print_json(data=config_sanitized)