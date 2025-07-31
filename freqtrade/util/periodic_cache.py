from datetime import datetime, timezone

from cachetools import TTLCache


class PeriodicCache(TTLCache):
    """
    在“整点”时间过期的特殊缓存。
    TTL为3600秒（1小时）的定时器将在每个整小时（:00）过期。
    """

    def __init__(self, maxsize, ttl, getsizeof=None):
        def local_timer():
            ts = datetime.now(timezone.utc).timestamp()
            offset = ts % ttl
            return ts - offset

        # 使用轻微的偏移量进行初始化
        super().__init__(maxsize=maxsize, ttl=ttl - 1e-5, timer=local_timer, getsizeof=getsizeof)