from enum import Enum


class RPCMessageType(str, Enum):
    """RPC消息类型枚举类"""
    STATUS = "status"  # 状态消息
    WARNING = "warning"  # 警告消息
    EXCEPTION = "exception"  # 异常消息
    STARTUP = "startup"  # 启动消息

    ENTRY = "entry"  # 入场消息
    ENTRY_FILL = "entry_fill"  # 入场订单成交消息
    ENTRY_CANCEL = "entry_cancel"  # 入场订单取消消息

    EXIT = "exit"  # 出场消息
    EXIT_FILL = "exit_fill"  # 出场订单成交消息
    EXIT_CANCEL = "exit_cancel"  # 出场订单取消消息

    PROTECTION_TRIGGER = "protection_trigger"  # 保护机制触发消息
    PROTECTION_TRIGGER_GLOBAL = "protection_trigger_global"  # 全局保护机制触发消息

    STRATEGY_MSG = "strategy_msg"  # 策略消息

    WHITELIST = "whitelist"  # 白名单消息
    ANALYZED_DF = "analyzed_df"  # 分析数据帧消息
    NEW_CANDLE = "new_candle"  # 新K线消息

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value


# 用于解析来自WebSocket消费者的请求的枚举
class RPCRequestType(str, Enum):
    SUBSCRIBE = "subscribe"  # 订阅请求

    WHITELIST = "whitelist"  # 白名单请求
    ANALYZED_DF = "analyzed_df"  # 分析数据帧请求

    def __str__(self):
        return self.value


# 不需要回声的消息类型
NO_ECHO_MESSAGES = (RPCMessageType.ANALYZED_DF, RPCMessageType.WHITELIST, RPCMessageType.NEW_CANDLE)