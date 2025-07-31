import logging
import time
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.websockets import WebSocket
from pydantic import ValidationError

from freqtrade.enums import RPCMessageType, RPCRequestType
from freqtrade.exceptions import FreqtradeException
from freqtrade.rpc.api_server.api_auth import validate_ws_token
from freqtrade.rpc.api_server.deps import get_message_stream, get_rpc
from freqtrade.rpc.api_server.ws.channel import WebSocketChannel, create_channel
from freqtrade.rpc.api_server.ws.message_stream import MessageStream
from freqtrade.rpc.api_server.ws_schemas import (
    WSAnalyzedDFMessage,
    WSErrorMessage,
    WSMessageSchema,
    WSRequestSchema,
    WSWhitelistMessage,
)
from freqtrade.rpc.rpc import RPC


logger = logging.getLogger(__name__)

# 私有路由器，受API密钥认证保护
router = APIRouter()


async def channel_reader(channel: WebSocketChannel, rpc: RPC):
    """
    迭代通道中的消息并处理请求
    """
    async for message in channel:
        try:
            await _process_consumer_request(message, channel, rpc)
        except FreqtradeException:
            logger.exception(f"处理来自{channel}的请求时出错")
            response = WSErrorMessage(data="处理请求时出错")

            await channel.send(response.dict(exclude_none=True))


async def channel_broadcaster(channel: WebSocketChannel, message_stream: MessageStream):
    """
    迭代消息流中的消息并发送它们
    """
    async for message, ts in message_stream:
        if channel.subscribed_to(message.get("type")):
            # 如果此通道在消息流中落后太多，则记录警告
            # 这可能导致内存泄漏，如果经常看到此消息，请考虑减少交易对列表大小或消费者数量
            if (time.time() - ts) > 60:
                logger.warning(
                    f"通道{channel}落后于MessageStream达1分钟，"
                    "如果经常看到此消息，这可能导致内存泄漏，请考虑减少交易对列表大小或消费者数量。"
                )

            await channel.send(message, use_timeout=True)


async def _process_consumer_request(request: dict[str, Any], channel: WebSocketChannel, rpc: RPC):
    """
    验证并处理来自WebSocket消费者的请求
    """
    # 验证请求，确保其符合模式
    try:
        websocket_request = WSRequestSchema.model_validate(request)
    except ValidationError as e:
        logger.error(f"来自{channel}的无效请求：{e}")
        return

    type_, data = websocket_request.type, websocket_request.data
    response: WSMessageSchema

    logger.debug(f"来自{channel}的{type_}类型请求")

    # 如果请求类型为SUBSCRIBE，则设置此通道的主题
    if type_ == RPCRequestType.SUBSCRIBE:
        # 如果请求为空，则不执行任何操作
        if not data:
            return

        # 如果所有传递的主题都是有效的RPCMessageType，则设置通道的订阅
        if all([any(x.value == topic for x in RPCMessageType) for topic in data]):
            channel.set_subscriptions(data)

        # 订阅不发送响应
        return

    elif type_ == RPCRequestType.WHITELIST:
        # 获取交易对白名单
        whitelist = rpc._ws_request_whitelist()

        # 格式化响应
        response = WSWhitelistMessage(data=whitelist)
        await channel.send(response.model_dump(exclude_none=True))

    elif type_ == RPCRequestType.ANALYZED_DF:
        # 将每个数据帧的蜡烛数量限制为'limit'或1500
        limit = int(min(data.get("limit", 1500), 1500)) if data else None
        pair = data.get("pair", None) if data else None

        # 对于生成器中的每个消息，发送单独的消息
        for message in rpc._ws_request_analyzed_df(limit, pair):
            # 格式化响应
            response = WSAnalyzedDFMessage(data=message)
            await channel.send(response.model_dump(exclude_none=True))


@router.websocket("/message/ws")
async def message_endpoint(
    websocket: WebSocket,
    token: str = Depends(validate_ws_token),
    rpc: RPC = Depends(get_rpc),
    message_stream: MessageStream = Depends(get_message_stream),
):
    if token:
        async with create_channel(websocket) as channel:
            await channel.run_channel_tasks(
                channel_reader(channel, rpc), channel_broadcaster(channel, message_stream)
            )