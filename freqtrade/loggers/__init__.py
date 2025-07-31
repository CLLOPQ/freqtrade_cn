import logging
import logging.config
import os
from copy import deepcopy
from logging import Formatter
from pathlib import Path
from typing import Any

from freqtrade.constants import Config
from freqtrade.exceptions import OperationalException
from freqtrade.loggers.buffering_handler import FTBufferingHandler
from freqtrade.loggers.ft_rich_handler import FtRichHandler
from freqtrade.loggers.rich_console import get_rich_console


logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 初始化缓冲区处理器 - 将用于 /log 端点
bufferHandler = FTBufferingHandler(1000)
bufferHandler.setFormatter(Formatter(LOGFORMAT))


error_console = get_rich_console(stderr=True, color_system=None)


def get_existing_handlers(handlertype):
    """
    返回已存在的处理器或 None（如果该处理器尚未添加到根处理器中）。
    """
    return next((h for h in logging.root.handlers if isinstance(h, handlertype)), None)


def setup_logging_pre() -> None:
    """
    日志的早期设置。
    使用 INFO 日志级别且仅使用流处理器。
    因此，早期消息（在正确的日志设置之前）只会在真正初始化后发送到额外的日志处理器，
    因为我们事先不知道用户需要哪些处理器。
    """
    rh = FtRichHandler(console=error_console)
    rh.setFormatter(Formatter("%(message)s"))
    logging.basicConfig(
        level=logging.INFO,
        format=LOGFORMAT,
        handlers=[
            rh,
            bufferHandler,
        ],
    )


FT_LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "basic": {"format": "%(message)s"},
        "standard": {
            "format": LOGFORMAT,
        },
    },
    "handlers": {
        "console": {
            "class": "freqtrade.loggers.ft_rich_handler.FtRichHandler",
            "formatter": "basic",
        },
    },
    "root": {
        "handlers": [
            "console",
        ],
        "level": "INFO",
    },
}


def _set_log_levels(
    log_config: dict[str, Any], verbosity: int = 0, api_verbosity: str = "info"
) -> None:
    """
    为不同的日志记录器设置日志级别
    """
    if "loggers" not in log_config:
        log_config["loggers"] = {}

    # 为第三方库设置默认级别
    third_party_loggers = {
        "freqtrade": logging.INFO if verbosity <= 1 else logging.DEBUG,
        "requests": logging.INFO if verbosity <= 1 else logging.DEBUG,
        "urllib3": logging.INFO if verbosity <= 1 else logging.DEBUG,
        "httpcore": logging.INFO if verbosity <= 1 else logging.DEBUG,
        "ccxt.base.exchange": logging.INFO if verbosity <= 2 else logging.DEBUG,
        "telegram": logging.INFO,
        "httpx": logging.WARNING,
        "werkzeug": logging.ERROR if api_verbosity == "error" else logging.INFO,
    }

    # 将第三方日志记录器添加到配置中
    for logger_name, level in third_party_loggers.items():
        if logger_name not in log_config["loggers"]:
            log_config["loggers"][logger_name] = {
                "level": logging.getLevelName(level),
                "propagate": True,
            }


def _add_root_handler(log_config: dict[str, Any], handler_name: str):
    if handler_name not in log_config["root"]["handlers"]:
        log_config["root"]["handlers"].append(handler_name)


def _add_formatter(log_config: dict[str, Any], format_name: str, format_: str):
    if format_name not in log_config["formatters"]:
        log_config["formatters"][format_name] = {"format": format_}


def _create_log_config(config: Config) -> dict[str, Any]:
    # 从用户配置获取 log_config 或使用默认值
    log_config = deepcopy(config.get("log_config", FT_LOGGING_CONFIG))

    if logfile := config.get("logfile"):
        s = logfile.split(":")
        if s[0] == "syslog":
            logger.warning(
                "已弃用：通过命令行配置 syslog 日志记录已过时。"
                "请改用配置文件中的 log_config 选项。"
            )
            # 向配置添加 syslog 处理器
            log_config["handlers"]["syslog"] = {
                "class": "logging.handlers.SysLogHandler",
                "formatter": "syslog_format",
                "address": (s[1], int(s[2])) if len(s) > 2 else s[1] if len(s) > 1 else "/dev/log",
            }

            _add_formatter(log_config, "syslog_format", "%(name)s - %(levelname)s - %(message)s")
            _add_root_handler(log_config, "syslog")

        elif s[0] == "journald":  # pragma: no cover
            # 检查是否有可用的模块
            logger.warning(
                "已弃用：通过命令行配置 Journald 日志记录已过时。"
                "请改用配置文件中的 log_config 选项。"
            )
            try:
                from cysystemd.journal import JournaldLogHandler  # noqa: F401
            except ImportError:
                raise OperationalException(
                    "要使用 journald 日志记录，需要安装 cysystemd python 包。"
                )

            # 向配置添加 journald 处理器
            log_config["handlers"]["journald"] = {
                "class": "cysystemd.journal.JournaldLogHandler",
                "formatter": "journald_format",
            }

            _add_formatter(log_config, "journald_format", "%(name)s - %(levelname)s - %(message)s")
            _add_root_handler(log_config, "journald")

        else:
            # 常规文件日志记录
            # 更新现有文件处理器配置
            if "file" in log_config["handlers"]:
                log_config["handlers"]["file"]["filename"] = logfile
            else:
                log_config["handlers"]["file"] = {
                    "class": "logging.handlers.RotatingFileHandler",
                    "formatter": "standard",
                    "filename": logfile,
                    "maxBytes": 1024 * 1024 * 10,  # 10Mb
                    "backupCount": 10,
                }
            _add_root_handler(log_config, "file")

    # 动态更新一些处理器
    for handler_config in log_config.get("handlers", {}).values():
        if handler_config.get("class") == "freqtrade.loggers.ft_rich_handler.FtRichHandler":
            handler_config["console"] = error_console
        elif handler_config.get("class") == "logging.handlers.RotatingFileHandler":
            logfile_path = Path(handler_config["filename"])
            try:
                # 为文件处理器创建父目录
                logfile_path.parent.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                raise OperationalException(
                    f'无法创建或访问日志文件 "{logfile_path.absolute()}"。'
                    "请确保您对日志文件或其父目录具有写入权限。如果您使用 docker 运行 freqtrade，"
                    "看到此错误消息可能是因为您以 root 用户登录，请切换到非 root 用户，"
                    "删除并重新创建所需的目录，然后重试。"
                )
    return log_config


def setup_logging(config: Config) -> None:
    """
    处理 -v/--verbose、--logfile 选项
    """
    verbosity = config["verbosity"]
    if os.environ.get("PYTEST_VERSION") is None or config.get("ft_tests_force_logging"):
        log_config = _create_log_config(config)
        _set_log_levels(
            log_config, verbosity, config.get("api_server", {}).get("verbosity", "info")
        )

        logging.config.dictConfig(log_config)

    # 将缓冲区处理器添加到根日志记录器
    if bufferHandler not in logging.root.handlers:
        logging.root.addHandler(bufferHandler)

    # 设置控制台输出的颜色系统
    if config.get("print_colorized", True):
        logger.info("启用彩色输出。")
        error_console._color_system = error_console._detect_color_system()

    logging.info("日志文件已配置")

    # 设置详细级别
    logging.root.setLevel(logging.INFO if verbosity < 1 else logging.DEBUG)

    logger.info("详细级别设置为 %s", verbosity)