from typing import Any

from fastapi import WebSocket as FastAPIWebSocket
from websockets.asyncio.client import ClientConnection as WebSocket

from freqtrade.rpc.api_server.ws.ws_types import WebSocketType


class WebSocketProxy:
    """
    WebSocket代理对象，用于将FastAPIWebSocket和websockets.WebSocketClientProtocol
    统一到相同的API下
    """

    def __init__(self, websocket: WebSocketType):
        self._websocket: FastAPIWebSocket | WebSocket = websocket

    @property
    def raw_websocket(self):
        return self._websocket

    @property
    def remote_addr(self) -> tuple[Any, ...]:
        if isinstance(self._websocket, WebSocket):
            return self._websocket.remote_address
        elif isinstance(self._websocket, FastAPIWebSocket):
            if self._websocket.client:
                client, port = self._websocket.client.host, self._websocket.client.port
                return (client, port)
        return ("unknown", 0)

    async def send(self, data):
        """
        在已包装的WebSocket上发送数据
        """
        if hasattr(self._websocket, "send_text"):
            await self._websocket.send_text(data)
        else:
            await self._websocket.send(data)

    async def recv(self):
        """
        在已包装的WebSocket上接收数据
        """
        if hasattr(self._websocket, "receive_text"):
            return await self._websocket.receive_text()
        else:
            return await self._websocket.recv()

    async def ping(self):
        """
        向WebSocket发送ping，FastAPI WebSocket不支持此操作
        """
        if hasattr(self._websocket, "ping"):
            return await self._websocket.ping()
        return False

    async def close(self, code: int = 1000):
        """
        关闭WebSocket连接，仅FastAPI WebSocket支持此操作
        """
        if hasattr(self._websocket, "close"):
            try:
                return await self._websocket.close(code)
            except RuntimeError:
                pass

    async def accept(self):
        """
        接受WebSocket连接，仅FastAPI WebSocket支持此操作
        """
        if hasattr(self._websocket, "accept"):
            return await self._websocket.accept()