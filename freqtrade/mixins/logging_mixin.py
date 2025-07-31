from collections.abc import Callable

from cachetools import TTLCache, cached


class LoggingMixin:
    """
    日志混合类
    仅在 `refresh_period` 时间间隔内显示一次相似消息。
    """

    # 完全禁用输出
    show_output = True

    def __init__(self, logger, refresh_period: int = 3600):
        """
        :param refresh_period: 以秒为单位 - 在此时间间隔内显示相同的消息
        """
        self.logger = logger
        self.refresh_period = refresh_period
        self._log_cache: TTLCache = TTLCache(maxsize=1024, ttl=self.refresh_period)

    def log_once(self, message: str, logmethod: Callable, force_show: bool = False) -> None:
        """
        记录消息 - 不超过 "refresh_period" 频率，以避免日志刷屏
        同时将日志消息记录为 debug 级别，以便简化调试。
        :param message: 要发送到函数的消息字符串。
        :param logmethod: 将被调用的函数。很可能是 `logger.info`。
        :param force_show: 如果为 True，则无论 show_output 值如何都发送消息。
        :return: 无。
        """

        @cached(cache=self._log_cache)
        def _log_once(message: str):
            logmethod(message)

        # 首先记录为 debug 级别
        self.logger.debug(message)

        # 如果 show_output 为 True 或 force_show 为 True，则调用隐藏函数
        if self.show_output or force_show:
            _log_once(message)