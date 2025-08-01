from datetime import datetime
from typing import Any, Literal, TypedDict

from freqtrade.constants import PairWithTimeframe
from freqtrade.enums import RPCMessageType


ProfitLossStr = Literal["盈利", "亏损"]


class RPCSendMsgBase(TypedDict):
    pass
    # ty1pe: Literal[RPCMessageType]


class RPCStatusMsg(RPCSendMsgBase):
    """用于状态、启动和警告消息"""

    type: Literal[RPCMessageType.STATUS, RPCMessageType.STARTUP, RPCMessageType.WARNING]
    status: str


class RPCStrategyMsg(RPCSendMsgBase):
    """用于状态、启动和警告消息"""

    type: Literal[RPCMessageType.STRATEGY_MSG]
    msg: str


class RPCProtectionMsg(RPCSendMsgBase):
    type: Literal[RPCMessageType.PROTECTION_TRIGGER, RPCMessageType.PROTECTION_TRIGGER_GLOBAL]
    id: int
    pair: str
    base_currency: str | None
    lock_time: str
    lock_timestamp: int
    lock_end_time: str
    lock_end_timestamp: int
    reason: str
    side: str
    active: bool


class RPCWhitelistMsg(RPCSendMsgBase):
    type: Literal[RPCMessageType.WHITELIST]
    data: list[str]


class __RPCEntryExitMsgBase(RPCSendMsgBase):
    trade_id: int
    buy_tag: str | None
    enter_tag: str | None
    exchange: str
    pair: str
    base_currency: str
    quote_currency: str
    leverage: float | None
    direction: str
    limit: float
    open_rate: float
    order_type: str
    stake_amount: float
    stake_currency: str
    fiat_currency: str | None
    amount: float
    open_date: datetime
    current_rate: float | None
    sub_trade: bool


class RPCEntryMsg(__RPCEntryExitMsgBase):
    type: Literal[RPCMessageType.ENTRY, RPCMessageType.ENTRY_FILL]


class RPCCancelMsg(__RPCEntryExitMsgBase):
    type: Literal[RPCMessageType.ENTRY_CANCEL]
    reason: str


class RPCExitMsg(__RPCEntryExitMsgBase):
    type: Literal[RPCMessageType.EXIT, RPCMessageType.EXIT_FILL]
    cumulative_profit: float
    gain: ProfitLossStr
    close_rate: float
    profit_amount: float
    profit_ratio: float
    exit_reason: str | None
    close_date: datetime
    # current_rate: float | None
    order_rate: float | None
    final_profit_ratio: float | None
    is_final_exit: bool


class RPCExitCancelMsg(__RPCEntryExitMsgBase):
    type: Literal[RPCMessageType.EXIT_CANCEL]
    reason: str
    gain: ProfitLossStr
    profit_amount: float
    profit_ratio: float
    exit_reason: str | None
    close_date: datetime


class _AnalyzedDFData(TypedDict):
    key: PairWithTimeframe
    df: Any
    la: datetime


class RPCAnalyzedDFMsg(RPCSendMsgBase):
    """新的分析后数据框消息"""

    type: Literal[RPCMessageType.ANALYZED_DF]
    data: _AnalyzedDFData


class RPCNewCandleMsg(RPCSendMsgBase):
    """新K线通知消息，每个新K线/交易对发出一次"""

    type: Literal[RPCMessageType.NEW_CANDLE]
    data: PairWithTimeframe


RPCOrderMsg = RPCEntryMsg | RPCExitMsg | RPCExitCancelMsg | RPCCancelMsg


RPCSendMsg = (
    RPCStatusMsg
    | RPCStrategyMsg
    | RPCProtectionMsg
    | RPCWhitelistMsg
    | RPCEntryMsg
    | RPCCancelMsg
    | RPCExitMsg
    | RPCExitCancelMsg
    | RPCAnalyzedDFMsg
    | RPCNewCandleMsg
)