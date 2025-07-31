import logging


logger = logging.getLogger(__name__)


__BIAS_TESTER_LOGGERS = [
    "freqtrade.resolvers",
    "freqtrade.strategy.hyper",
    "freqtrade.configuration.config_validation",
]


def reduce_verbosity_for_bias_tester() -> None:
    """
    降低偏差测试器的日志详细程度。
    它会多次次加载相同的策略，这会导致日志刷屏。
    """
    logger.info("为偏差测试器降低日志详细程度。")
    for logger_name in __BIAS_TESTER_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def restore_verbosity_for_bias_tester() -> None:
    """
    在偏差测试器运行后恢复日志详细程度。
    """
    logger.info("恢复日志详细程度。")
    log_level = logging.NOTSET
    for logger_name in __BIAS_TESTER_LOGGERS:
        logging.getLogger(logger_name).setLevel(log_level)