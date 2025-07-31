from enum import Enum


class TradingMode(str, Enum):
    """
    交易模式枚举类，用于区分
    现货、保证金、期货或其他交易方式
    """

    SPOT = "spot"      # 现货交易
    MARGIN = "margin"  # 保证金交易
    FUTURES = "futures"  # 期货交易

    def __str__(self):
        return f"{self.name.lower()}"