from datetime import datetime
from typing import Any, TypedDict

from pandas import DataFrame
from pydantic import BaseModel, ConfigDict

from freqtrade.constants import PairWithTimeframe
from freqtrade.enums import RPCMessageType, RPCRequestType


class BaseArbitraryModel(BaseModel):
    # 基础任意类型模型，允许模型包含任意类型的数据
    model_config = ConfigDict(arbitrary_types_allowed=True)


class WSRequestSchema(BaseArbitraryModel):
    # WebSocket请求模式基类
    type: RPCRequestType
    data: Any | None = None


class WSMessageSchemaType(TypedDict):
    # 用于类型检查以避免进行pydantic类型检查的类型。
    type: RPCMessageType
    data: dict[str, Any] | None


class WSMessageSchema(BaseArbitraryModel):
    # WebSocket消息模式基类
    type: RPCMessageType
    data: Any | None = None
    model_config = ConfigDict(extra="allow")


# ------------------------------ 请求模式 ----------------------------


class WSSubscribeRequest(WSRequestSchema):
    # 订阅请求模式
    type: RPCRequestType = RPCRequestType.SUBSCRIBE
    data: list[RPCMessageType]


class WSWhitelistRequest(WSRequestSchema):
    # 白名单请求模式
    type: RPCRequestType = RPCRequestType.WHITELIST
    data: None = None


class WSAnalyzedDFRequest(WSRequestSchema):
    # 分析数据框请求模式
    type: RPCRequestType = RPCRequestType.ANALYZED_DF
    data: dict[str, Any] = {"limit": 1500, "pair": None}


# ------------------------------ 消息模式 ----------------------------


class WSWhitelistMessage(WSMessageSchema):
    # 白名单消息模式
    type: RPCMessageType = RPCMessageType.WHITELIST
    data: list[str]


class WSAnalyzedDFMessage(WSMessageSchema):
    class AnalyzedDFData(BaseArbitraryModel):
        # 分析数据框消息的数据结构
        key: PairWithTimeframe
        df: DataFrame
        la: datetime

    type: RPCMessageType = RPCMessageType.ANALYZED_DF
    data: AnalyzedDFData


class WSErrorMessage(WSMessageSchema):
    # 错误消息模式
    type: RPCMessageType = RPCMessageType.EXCEPTION
    data: str


# --------------------------------------------------------------------------