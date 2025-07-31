from enum import Enum


class MarginMode(str, Enum):
    """
    保证金模式枚举类，用于区分
    全仓保证金/期货保证金模式和
    逐仓保证金/期货保证金模式
    """

    CROSS = "cross"  # 全仓模式
    ISOLATED = "isolated"  # 逐仓模式
    NONE = ""  # 无保证金模式

    def __str__(self):
        return f"{self.value.lower()}"