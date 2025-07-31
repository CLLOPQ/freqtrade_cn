from enum import Enum


class MarketDirection(Enum):
    """
    市场方向枚举类，用于表示各种市场走向。
    """

    LONG = "long"   # 多头（上涨方向）
    SHORT = "short" # 空头（下跌方向）
    EVEN = "even"   # 持平（无明显方向）
    NONE = "none"   # 无方向

    def __str__(self):
        # 转换为字符串
        return self.value