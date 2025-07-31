"""
此模块包含加载配置文件的函数
"""

import logging
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import rapidjson

from freqtrade.constants import MINIMAL_CONFIG, Config
from freqtrade.exceptions import ConfigurationError, OperationalException
from freqtrade.misc import deep_merge_dicts


logger = logging.getLogger(__name__)


CONFIG_PARSE_MODE = rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS


def log_config_error_range(path: str, errmsg: str) -> str:
    """
    解析配置文件并打印错误周围的范围
    """
    if path != "-":
        offsetlist = re.findall(r"(?<=Parse\serror\sat\soffset\s)\d+", errmsg)
        if offsetlist:
            offset = int(offsetlist[0])
            text = Path(path).read_text()
            # 获取错误行周围80个字符的偏移量
            subtext = text[offset - min(80, offset) : offset + 80]
            segments = subtext.split("\n")
            if len(segments) > 3:
                # 移除第一行和最后一行，避免奇怪的截断
                return "\n".join(segments[1:-1])
            else:
                return subtext
    return ""


def load_file(path: Path) -> dict[str, Any]:
    try:
        with path.open("r") as file:
            config = rapidjson.load(file, parse_mode=CONFIG_PARSE_MODE)
    except FileNotFoundError:
        raise OperationalException(f'文件 "{path}" 未找到！') from None
    return config


def load_config_file(path: str) -> dict[str, Any]:
    """
    从给定路径加载配置文件
    :param path: 路径字符串
    :return: 配置字典
    """
    try:
        # 如果选项中要求，从标准输入读取配置
        with Path(path).open() if path != "-" else sys.stdin as file:
            config = rapidjson.load(file, parse_mode=CONFIG_PARSE_MODE)
    except FileNotFoundError:
        raise OperationalException(
            f'配置文件 "{path}" 未找到！'
            " 请创建配置文件或检查其是否存在。"
        ) from None
    except rapidjson.JSONDecodeError as e:
        err_range = log_config_error_range(path, str(e))
        raise ConfigurationError(
            f"{e}\n请验证配置的以下部分：\n{err_range}"
            if err_range
            else "请验证您的配置文件是否存在语法错误。"
        )

    return config


def load_from_files(
    files: list[str], base_path: Path | None = None, level: int = 0
) -> dict[str, Any]:
    """
    如果指定，递归加载配置文件。
    子文件被假定为相对于初始配置。
    """
    config: Config = {}
    if level > 5:
        raise ConfigurationError("检测到配置循环。")

    if not files:
        return deepcopy(MINIMAL_CONFIG)
    files_loaded = []
    # 这里我们期望一个配置文件名列表
    for filename in files:
        logger.info(f"使用配置：{filename} ...")
        if filename == "-":
            # 立即加载标准输入并返回
            return load_config_file(filename)
        file = Path(filename)
        if base_path:
            # 前置基础路径以允许相对路径赋值
            file = base_path / file

        config_tmp = load_config_file(str(file))
        if "add_config_files" in config_tmp:
            config_sub = load_from_files(
                config_tmp["add_config_files"], file.resolve().parent, level + 1
            )
            files_loaded.extend(config_sub.get("config_files", []))
            config_tmp = deep_merge_dicts(config_tmp, config_sub)

        files_loaded.insert(0, str(file))

        # 合并配置选项，覆盖先前的值
        config = deep_merge_dicts(config_tmp, config)

    config["config_files"] = files_loaded

    return config