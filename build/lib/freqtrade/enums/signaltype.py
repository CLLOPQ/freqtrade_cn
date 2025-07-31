from enum import Enum


class SignalType(Enum):
    """
    信号类型枚举类，用于区分入场和出场信号
    """

    ENTER_LONG = "enter_long"  # 多头入场信号
    EXIT_LONG = "exit_long"    # 多头出场信号
    ENTER_SHORT = "enter_short"  # 空头入场信号
    EXIT_SHORT = "exit_short"    # 空头出场信号

    def __str__(self):
        return f"{self.name.lower()}"


class SignalTagType(Enum):
    """
    信号标签类型枚举类，用于标识信号相关列
    """

    ENTER_TAG = "enter_tag"  # 入场标签列
    EXIT_TAG = "exit_tag"    # 出场标签列

    def __str__(self):
        return f"{self.name.lower()}"


class SignalDirection(str, Enum):
    """
    信号方向枚举类，标识交易方向
    """
    LONG = "long"   # 多头方向
    SHORT = "short"  # 空头方向

    def __str__(self):
        return f"{self.name.lower()}"