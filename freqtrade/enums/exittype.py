from enum import Enum


class ExitType(Enum):
    """
    出场原因枚举类，用于区分不同的出场原因
    """

    ROI = "roi"  # 基于回报率的出场
    STOP_LOSS = "stop_loss"  # 止损出场
    STOPLOSS_ON_EXCHANGE = "stoploss_on_exchange"  # 交易所止损出场
    TRAILING_STOP_LOSS = "trailing_stop_loss"  # 追踪止损出场
    LIQUIDATION = "liquidation"  # 爆仓出场
    EXIT_SIGNAL = "exit_signal"  # 出场信号触发
    FORCE_EXIT = "force_exit"  # 强制出场
    EMERGENCY_EXIT = "emergency_exit"  # 紧急出场
    CUSTOM_EXIT = "custom_exit"  # 自定义出场
    PARTIAL_EXIT = "partial_exit"  # 部分出场
    SOLD_ON_EXCHANGE = "sold_on_exchange"  # 在交易所手动卖出
    NONE = ""  # 无出场

    def __str__(self):
        # 显式转换为字符串，便于数据导出
        return self.value