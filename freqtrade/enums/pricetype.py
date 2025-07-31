from enum import Enum


class PriceType(str, Enum):
    """价格类型枚举类，用于区分止损单的可能触发价格类型"""

    LAST = "last"  # 最新成交价
    MARK = "mark"  # 标记价格
    INDEX = "index"  # 指数价格