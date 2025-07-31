import logging
import time
from collections.abc import Callable

from cachetools import TTLCache


logger = logging.getLogger(__name__)


class MeasureTime:
    """
    测量一段代码的执行时间，如果超过时间限制则调用回调函数。
    """

    def __init__(
        self, callback: Callable[[float, float], None], time_limit: float, ttl: int = 3600 * 4
    ):
        """
        :param callback: 如果超过时间限制，则调用此回调函数。此回调函数将每'ttl'秒调用一次，
            参数为'duration'（以秒为单位）和'time limit'——表示传入的时间限制。
        :param time_limit: 时间限制（以秒为单位）。
        :param ttl: 缓存的生存时间（以秒为单位）。默认为4小时。
        """
        self._callback = callback
        self._time_limit = time_limit
        self.__cache: TTLCache = TTLCache(maxsize=1, ttl=ttl)

    def __enter__(self):
        self._start = time.time()

    def __exit__(self, *args):
        end = time.time()
        if self.__cache.get("value"):
            return
        duration = end - self._start

        if duration < self._time_limit:
            return
        self._callback(duration, self._time_limit)

        self.__cache["value"] = True