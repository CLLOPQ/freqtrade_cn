import asyncio
import logging
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import WebSocketDisconnect
from websockets.exceptions import ConnectionClosed

from freqtrade.rpc.api_server.ws.proxy import WebSocketProxy
from freqtrade.rpc.api_server.ws.serializer import (
    HybridJSONWebSocketSerializer,
    WebSocketSerializer,
)
from freqtrade.rpc.api_server.ws.ws_types import WebSocketType
from freqtrade.rpc.api_server.ws_schemas import WSMessageSchemaType


logger = logging.getLogger(__name__)


class WebSocketChannel:
    """
    用于帮助管理WebSocket连接的对象
    """

    def __init__(
        self,
        websocket: WebSocketType,
        channel_id: str | None = None,
        serializer_cls: type[WebSocketSerializer] = HybridJSONWebSocketSerializer,
        send_throttle: float = 0.01,
    ):
        self.channel_id = channel_id if channel_id else uuid4().hex[:8]
        self._websocket = WebSocketProxy(websocket)

        # 用于表示WebSocket已关闭的内部事件
        self._closed = asyncio.Event()
        # 为该通道创建的异步任务
        self._channel_tasks: list[asyncio.Task] = []

        # 用于计算平均发送时间的双端队列
        self._send_times: deque[float] = deque([], maxlen=10)
        # 高限制默认初始值为3
        self._send_high_limit = 3
        self._send_throttle = send_throttle

        # 已订阅的消息类型
        self._subscriptions: list[str] = []

        # 使用序列化类包装WebSocket
        self._wrapped_ws = serializer_cls(self._websocket)

    def __repr__(self):
        return f"WebSocketChannel({self.channel_id}, {self.remote_addr})"

    @property
    def raw_websocket(self):
        return self._websocket.raw_websocket

    @property
    def remote_addr(self):
        return self._websocket.remote_addr

    @property
    def avg_send_time(self):
        return sum(self._send_times) / len(self._send_times)

    def _calc_send_limit(self):
        """
        计算此通道的发送高限制
        """

        # 只有当我们有足够的数据时才更新
        if len(self._send_times) == self._send_times.maxlen:
            # 至少1秒或发送时间平均值的两倍，每条消息最大3秒
            self._send_high_limit = min(max(self.avg_send_time * 2, 1), 3)

    async def send(self, message: WSMessageSchemaType | dict[str, Any], use_timeout: bool = False):
        """
        在包装的WebSocket上发送消息。如果发送耗时过长，将引发TimeoutError并断开连接。

        :param message: 要发送的消息
        :param use_timeout: 强制应用发送高限制，默认为False
        """
        try:
            _ = time.time()
            # 如果发送超时，将引发TimeoutError并向上传播到message_endpoint以关闭连接
            await asyncio.wait_for(
                self._wrapped_ws.send(message),
                timeout=self._send_high_limit if use_timeout else None,
            )
            total_time = time.time() - _
            self._send_times.append(total_time)

            self._calc_send_limit()
        except asyncio.TimeoutError:
            logger.info(f"{self}的连接超时，正在断开连接")
            raise

        # 显式将控制权交还给事件循环（因为websockets.send不会这样做）
        # 同时限制发送速度
        await asyncio.sleep(self._send_throttle)

    async def recv(self):
        """
        在包装的WebSocket上接收消息
        """
        return await self._wrapped_ws.recv()

    async def ping(self):
        """
        向WebSocket发送ping
        """
        return await self._websocket.ping()

    async def accept(self):
        """
        接受底层WebSocket连接，如果在接受前连接已关闭，则仅关闭通道。
        """
        try:
            return await self._websocket.accept()
        except RuntimeError:
            await self.close()

    async def close(self):
        """
        关闭WebSocketChannel
        """

        self._closed.set()

        try:
            await self._websocket.close()
        except RuntimeError:
            pass

    def is_closed(self) -> bool:
        """
        关闭标志
        """
        return self._closed.is_set()

    def set_subscriptions(self, subscriptions: list[str]) -> None:
        """
        设置此通道订阅的订阅项

        :param subscriptions: 订阅项列表，List[str]
        """
        self._subscriptions = subscriptions

    def subscribed_to(self, message_type: str) -> bool:
        """
        检查此通道是否订阅了message_type

        :param message_type: 要检查的消息类型
        """
        return message_type in self._subscriptions

    async def run_channel_tasks(self, *tasks, **kwargs):
        """
        创建并等待通道任务，除非引发异常，否则取消所有任务。

        :params *tasks: 所有要并发运行的协程或任务
        :param **kwargs: 要传递给gather的任何额外关键字参数
        """

        if not self.is_closed():
            # 如果任务不是已创建的Task实例，则包装为Task
            self._channel_tasks = [
                task if isinstance(task, asyncio.Task) else asyncio.create_task(task)
                for task in tasks
            ]

            try:
                return await asyncio.gather(*self._channel_tasks, **kwargs)
            except Exception:
                # 如果引发异常，取消其余任务
                await self.cancel_channel_tasks()

    async def cancel_channel_tasks(self):
        """
        取消并等待所有通道任务
        """
        for task in self._channel_tasks:
            task.cancel()

            # 等待任务完成取消
            try:
                await task
            except (
                asyncio.CancelledError,
                asyncio.TimeoutError,
                WebSocketDisconnect,
                ConnectionClosed,
                RuntimeError,
            ):
                pass
            except Exception as e:
                logger.info(f"遇到未知异常: {e}", exc_info=e)

        self._channel_tasks = []

    async def __aiter__(self):
        """
        接收消息的生成器
        """
        # 此处无法捕获任何错误，因为websocket.recv会首先捕获任何断开连接并向上传播，以便连接立即被垃圾回收
        while not self.is_closed():
            yield await self.recv()


@asynccontextmanager
async def create_channel(websocket: WebSocketType, **kwargs) -> AsyncIterator[WebSocketChannel]:
    """
    用于安全打开和关闭WebSocketChannel的上下文管理器
    """
    channel = WebSocketChannel(websocket, **kwargs)
    try:
        await channel.accept()
        logger.info(f"已连接到通道 - {channel}")

        yield channel
    finally:
        await channel.close()
        logger.info(f"已从通道断开连接 - {channel}")