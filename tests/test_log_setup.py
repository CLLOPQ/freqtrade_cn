import logging
import re
import sys

import pytest

from freqtrade.exceptions import OperationalException
from freqtrade.loggers import (
    FTBufferingHandler,
    FtRichHandler,
    setup_logging,
    setup_logging_pre,
)
from freqtrade.loggers.set_log_levels import (
    reduce_verbosity_for_bias_tester,
    restore_verbosity_for_bias_tester,
)


@pytest.mark.usefixtures("keep_log_config_loggers")
def test_set_loggers() -> None:
    # 重置日志级别为DEBUG，否则由于全局设置会导致测试随机失败
    logging.getLogger("requests").setLevel(logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.DEBUG)
    logging.getLogger("ccxt.base.exchange").setLevel(logging.DEBUG)
    logging.getLogger("telegram").setLevel(logging.DEBUG)

    # 记录重置后的日志级别
    previous_value1 = logging.getLogger("requests").level
    previous_value2 = logging.getLogger("ccxt.base.exchange").level
    previous_value3 = logging.getLogger("telegram").level
    
    # 配置日志参数
    config = {
        "verbosity": 1,
        "ft_tests_force_logging": True,
    }
    setup_logging(config)

    # 验证verbosity=1时的日志级别设置
    value1 = logging.getLogger("requests").level
    assert previous_value1 is not value1
    assert value1 is logging.INFO

    value2 = logging.getLogger("ccxt.base.exchange").level
    assert previous_value2 is not value2
    assert value2 is logging.INFO

    value3 = logging.getLogger("telegram").level
    assert previous_value3 is not value3
    assert value3 is logging.INFO
    
    # 测试verbosity=2时的日志级别
    config["verbosity"] = 2
    setup_logging(config)

    assert logging.getLogger("requests").level is logging.DEBUG
    assert logging.getLogger("ccxt.base.exchange").level is logging.INFO
    assert logging.getLogger("telegram").level is logging.INFO
    assert logging.getLogger("werkzeug").level is logging.INFO

    # 测试verbosity=3时的日志级别，包含API服务器配置
    config["verbosity"] = 3
    config["api_server"] = {"verbosity": "error"}
    setup_logging(config)

    assert logging.getLogger("requests").level is logging.DEBUG
    assert logging.getLogger("ccxt.base.exchange").level is logging.DEBUG
    assert logging.getLogger("telegram").level is logging.INFO
    assert logging.getLogger("werkzeug").level is logging.ERROR


@pytest.mark.skipif(sys.platform == "win32", reason="在Windows系统上不运行此测试")
@pytest.mark.usefixtures("keep_log_config_loggers")
def test_set_loggers_syslog():
    logger = logging.getLogger()
    orig_handlers = logger.handlers  # 保存原始处理器
    logger.handlers = []  # 清空当前处理器

    # 配置系统日志
    config = {
        "ft_tests_force_logging": True,
        "verbosity": 2,
        "logfile": "syslog:/dev/log",
    }

    setup_logging_pre()
    setup_logging(config)
    # 验证是否正确添加了3个处理器
    assert len(logger.handlers) == 3
    # 验证是否包含系统日志处理器
    assert [x for x in logger.handlers if isinstance(x, logging.handlers.SysLogHandler)]
    # 验证是否包含富文本处理器
    assert [x for x in logger.handlers if isinstance(x, FtRichHandler)]
    # 验证是否包含缓冲处理器
    assert [x for x in logger.handlers if isinstance(x, FTBufferingHandler)]
    # 再次设置日志不应导致处理器重复添加
    setup_logging(config)
    assert len(logger.handlers) == 3
    
    # 恢复原始处理器，避免影响pytest
    logger.handlers = orig_handlers


@pytest.mark.skipif(sys.platform == "win32", reason="在Windows系统上不运行此测试")
@pytest.mark.usefixtures("keep_log_config_loggers")
def test_set_loggers_Filehandler(tmp_path):
    logger = logging.getLogger()
    orig_handlers = logger.handlers
    logger.handlers = []
    logfile = tmp_path / "logs/ft_logfile.log"
    
    # 配置文件日志
    config = {
        "ft_tests_force_logging": True,
        "verbosity": 2,
        "logfile": str(logfile),
    }

    setup_logging_pre()
    setup_logging(config)
    # 验证是否正确添加了3个处理器
    assert len(logger.handlers) == 3
    # 验证是否包含文件轮转处理器
    assert [x for x in logger.handlers if isinstance(x, logging.handlers.RotatingFileHandler)]
    # 验证是否包含富文本处理器
    assert [x for x in logger.handlers if isinstance(x, FtRichHandler)]
    # 验证是否包含缓冲处理器
    assert [x for x in logger.handlers if isinstance(x, FTBufferingHandler)]
    # 再次设置日志不应导致处理器重复添加
    setup_logging(config)
    assert len(logger.handlers) == 3
    
    # 清理并恢复原始处理器
    if logfile.exists:
        logfile.unlink()
    logger.handlers = orig_handlers


@pytest.mark.skipif(sys.platform == "win32", reason="在Windows系统上不运行此测试")
@pytest.mark.usefixtures("keep_log_config_loggers")
def test_set_loggers_Filehandler_without_permission(tmp_path):
    logger = logging.getLogger()
    orig_handlers = logger.handlers
    logger.handlers = []

    try:
        # 设置目录为只读权限
        tmp_path.chmod(0o400)
        logfile = tmp_path / "logs/ft_logfile.log"
        config = {
            "ft_tests_force_logging": True,
            "verbosity": 2,
            "logfile": str(logfile),
        }

        setup_logging_pre()
        # 验证在没有权限时是否抛出异常
        with pytest.raises(OperationalException):
            setup_logging(config)

        logger.handlers = orig_handlers
    finally:
        # 恢复目录权限
        tmp_path.chmod(0o700)


@pytest.mark.skip(reason="不是每个系统都安装了systemd，因此不测试此项")
@pytest.mark.usefixtures("keep_log_config_loggers")
def test_set_loggers_journald():
    logger = logging.getLogger()
    orig_handlers = logger.handlers
    logger.handlers = []

    # 配置journald日志
    config = {
        "ft_tests_force_logging": True,
        "verbosity": 2,
        "logfile": "journald",
    }

    setup_logging_pre()
    setup_logging(config)
    # 验证是否正确添加了3个处理器
    assert len(logger.handlers) == 3
    # 验证是否包含Journald日志处理器
    assert [x for x in logger.handlers if type(x).__name__ == "JournaldLogHandler"]
    # 验证是否包含富文本处理器
    assert [x for x in logger.handlers if isinstance(x, FtRichHandler)]
    
    # 恢复原始处理器
    logger.handlers = orig_handlers


@pytest.mark.usefixtures("keep_log_config_loggers")
def test_set_loggers_journald_importerror(import_fails):
    logger = logging.getLogger()
    orig_handlers = logger.handlers
    logger.handlers = []

    # 配置journald日志
    config = {
        "ft_tests_force_logging": True,
        "verbosity": 2,
        "logfile": "journald",
    }
    # 验证在缺少cysystemd包时是否抛出正确异常
    with pytest.raises(OperationalException, match=r"你需要安装cysystemd python包.*"):
        setup_logging(config)
    logger.handlers = orig_handlers


@pytest.mark.usefixtures("keep_log_config_loggers")
def test_set_loggers_json_format(capsys):
    logger = logging.getLogger()
    orig_handlers = logger.handlers
    logger.handlers = []

    # 配置JSON格式日志
    config = {
        "ft_tests_force_logging": True,
        "verbosity": 2,
        "log_config": {
            "version": 1,
            "formatters": {
                "json": {
                    "()": "freqtrade.loggers.json_formatter.JsonFormatter",
                    "fmt_dict": {
                        "timestamp": "asctime",
                        "level": "levelname",
                        "logger": "name",
                        "message": "message",
                    },
                }
            },
            "handlers": {
                "json": {
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                }
            },
            "root": {
                "handlers": ["json"],
                "level": "DEBUG",
            },
        },
    }

    setup_logging_pre()
    setup_logging(config)
    # 验证是否正确添加了2个处理器
    assert len(logger.handlers) == 2
    # 验证是否包含流处理器
    assert [x for x in logger.handlers if type(x).__name__ == "StreamHandler"]
    # 验证是否包含缓冲处理器
    assert [x for x in logger.handlers if isinstance(x, FTBufferingHandler)]

    # 测试JSON日志输出
    logger.info("测试消息")

    captured = capsys.readouterr()
    # 验证日志是否为JSON格式并包含测试消息
    assert re.search(r'{"timestamp": ".*"测试消息".*', captured.err)

    # 恢复原始处理器
    logger.handlers = orig_handlers


def test_reduce_verbosity():
    setup_logging_pre()
    # 降低日志详细程度
    reduce_verbosity_for_bias_tester()
    # 记录基础日志级别
    prior_level = logging.getLogger("freqtrade").getEffectiveLevel()

    # 验证特定模块的日志级别是否被降低
    assert logging.getLogger("freqtrade.resolvers").getEffectiveLevel() == logging.WARNING
    assert logging.getLogger("freqtrade.strategy.hyper").getEffectiveLevel() == logging.WARNING
    # 验证基础日志级别未改变
    assert logging.getLogger("freqtrade").getEffectiveLevel() == prior_level

    # 恢复日志详细程度
    restore_verbosity_for_bias_tester()

    # 验证日志级别已恢复
    assert logging.getLogger("freqtrade.resolvers").getEffectiveLevel() == prior_level
    assert logging.getLogger("freqtrade.strategy.hyper").getEffectiveLevel() == prior_level
    assert logging.getLogger("freqtrade").getEffectiveLevel() == prior_level
    # 基础日志级别仍未改变