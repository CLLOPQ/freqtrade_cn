from enum import Enum


class CandleType(str, Enum):
    """K线类型枚举，用于区分不同类型的K线"""

    SPOT = "spot"  # 现货K线
    FUTURES = "futures"  # 期货K线
    MARK = "mark"  # 标记价格K线
    INDEX = "index"  # 指数K线
    PREMIUMINDEX = "premiumIndex"  # 溢价指数K线

    # 注意：这些严格来说不属于K线类型，但暂时放在这里
    FUNDING_RATE = "funding_rate"  # 资金费率
    # BORROW_RATE = "borrow_rate"  # * 未实现的借款费率

    def __str__(self):
        return f"{self.name.lower()}"

    @staticmethod
    def from_string(value: str) -> "CandleType":
        """从字符串转换为CandleType枚举值"""
        if not value:
            # 默认返回现货K线类型
            return CandleType.SPOT
        return CandleType(value)

    @staticmethod
    def get_default(trading_mode: str) -> "CandleType":
        """根据交易模式获取默认的K线类型"""
        if trading_mode == "futures":
            return CandleType.FUTURES
        return CandleType.SPOT