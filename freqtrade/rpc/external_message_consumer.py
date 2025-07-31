"""
外部消息消费者模块

主要目的是连接到外部机器人的消息WebSocket以从中消费数据
"""

import asyncio
import logging
import socket
from collections.abc import Callable
from threading import Thread
from typing import Any, TypedDict

import websockets
from pydantic import ValidationError

from freqtrade.constants import FULL_DATAFRAME_THRESHOLD
from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import RPCMessageType
from freqtrade.misc import remove_entry_exit_signals
from freqtrade.rpc.api_server.ws.channel import WebSocketChannel, create_channel
from freqtrade.rpc.api_server.ws.message_stream import MessageStream
from freqtrade.rpc.api_server.ws_schemas import (
    WSAnalyzedDFMessage,
    WSAnalyzedDFRequest,
    WSMessageSchema,
    WSRequestSchema,
    WSSubscribeRequest,
    WSWhitelistMessage,
    WSWhitelistRequest,
)


class Producer(TypedDict):
    name: str
    host: str
    port: int
    secure: bool
    ws_token: str


logger = logging.getLogger(__name__)


def schema_to_dict(schema: WSMessageSchema | WSRequestSchema):
    """将模式转换为字典，排除None值"""
    return schema.model_dump(exclude_none=True)


class ExternalMessageConsumer:
    """
    用于从其他Freqtrade机器人消费外部消息的主控制器类
    """

    def __init__(self, config: dict[str, Any], dataprovider: DataProvider):
        self._config = config
        self._dp = dataprovider

        self._running = False
        self._thread = None
        self._loop = None
        self._main_task = None
        self._sub_tasks = None

        self._emc_config = self._config.get("external_message_consumer", {})

        self.enabled = self._emc_config.get("enabled", False)
        self.producers: list[Producer] = self._emc_config.get("producers", [])

        self.wait_timeout = self._emc_config.get("wait_timeout", 30)  # 单位：秒
        self.ping_timeout = self._emc_config.get("ping_timeout", 10)  # 单位：秒
        self.sleep_time = self._emc_config.get("sleep_time", 10)  # 单位：秒

        # 初始请求中的蜡烛数量
        self.initial_candle_limit = self._emc_config.get("initial_candle_limit", 1500)

        # 消息大小限制，单位：兆字节。默认8MB，转换为字节（WebSocket客户端需要字节）
        self.message_size_limit = self._emc_config.get("message_size_limit", 8) << 20

        # 显式设置这些值，因为用户可能不应该修改它们
        # 除非我们以某种方式与策略集成以允许创建消息回调
        self.topics = [RPCMessageType.WHITELIST, RPCMessageType.ANALYZED_DF]

        # 允许为每个初始请求设置数据
        self._initial_requests: list[WSRequestSchema] = [
            WSSubscribeRequest(data=self.topics),
            WSWhitelistRequest(),
            WSAnalyzedDFRequest(),
        ]

        # 指定哪种函数处理哪种RPC消息类型
        self._message_handlers: dict[str, Callable[[str, WSMessageSchema], None]] = {
            RPCMessageType.WHITELIST: self._consume_whitelist_message,
            RPCMessageType.ANALYZED_DF: self._consume_analyzed_df_message,
        }

        self._channel_streams: dict[str, MessageStream] = {}

        self.start()

    def start(self):
        """
        在另一个线程中启动主内部循环以运行协程
        """
        if self._thread and self._loop:
            return

        logger.info("启动外部消息消费者")

        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=self._loop.run_forever)
        self._running = True
        self._thread.start()

        self._main_task = asyncio.run_coroutine_threadsafe(self._main(), loop=self._loop)

    def shutdown(self):
        """
        关闭循环、线程和任务
        """
        if self._thread and self._loop:
            logger.info("停止外部消息消费者")
            self._running = False

            self._channel_streams = {}

            if self._sub_tasks:
                # 取消子任务
                for task in self._sub_tasks:
                    task.cancel()

            if self._main_task:
                # 取消主任务
                self._main_task.cancel()

            self._thread.join()

            self._thread = None
            self._loop = None
            self._sub_tasks = None
            self._main_task = None

    async def _main(self):
        """
        主任务协程
        """
        lock = asyncio.Lock()

        try:
            # 为每个生产者创建连接
            self._sub_tasks = [
                self._loop.create_task(self._handle_producer_connection(producer, lock))
                for producer in self.producers
            ]

            await asyncio.gather(*self._sub_tasks)
        except asyncio.CancelledError:
            pass
        finally:
            # 完成后停止循环
            self._loop.stop()

    async def _handle_producer_connection(self, producer: Producer, lock: asyncio.Lock):
        """
        消费者的主连接循环

        :param producer: 包含生产者信息的字典
        :param lock: 一个asyncio锁
        """
        try:
            await self._create_connection(producer, lock)
        except asyncio.CancelledError:
            # 静默退出
            pass

    async def _create_connection(self, producer: Producer, lock: asyncio.Lock):
        """
        实际创建和处理WebSocket连接，在超时进行ping操作并处理连接错误。

        :param producer: 包含生产者信息的字典
        :param lock: 一个asyncio锁
        """
        while self._running:
            try:
                host, port = producer["host"], producer["port"]
                token = producer["ws_token"]
                name = producer["name"]
                scheme = "wss" if producer.get("secure", False) else "ws"
                ws_url = f"{scheme}://{host}:{port}/api/v1/message/ws?token={token}"

                # 如果URL无效，这将引发InvalidURI
                async with websockets.connect(
                    ws_url, max_size=self.message_size_limit, ping_interval=None
                ) as ws:
                    async with create_channel(ws, channel_id=name, send_throttle=0.5) as channel:
                        # 为该通道创建消息流
                        self._channel_streams[name] = MessageStream()

                        # 连接时运行通道任务
                        await channel.run_channel_tasks(
                            self._receive_messages(channel, producer, lock),
                            self._send_requests(channel, self._channel_streams[name]),
                        )

            except (websockets.exceptions.InvalidURI, ValueError) as e:
                logger.error(f"{ws_url} 是无效的WebSocket URL - {e}")
                break

            except (
                socket.gaierror,
                ConnectionRefusedError,
                websockets.exceptions.InvalidHandshake,
            ) as e:
                logger.error(f"连接被拒绝 - {e}，{self.sleep_time}秒后重试")
                await asyncio.sleep(self.sleep_time)
                continue

            except (
                websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK,
            ):
                # 继续尝试无限期重连
                await asyncio.sleep(self.sleep_time)
                continue

            except Exception as e:
                # 发生不可预见的错误，记录并继续
                logger.error("发生意外错误:")
                logger.exception(e)
                await asyncio.sleep(self.sleep_time)
                continue

    async def _send_requests(self, channel: WebSocketChannel, channel_stream: MessageStream):
        # 发送初始请求
        for init_request in self._initial_requests:
            await channel.send(schema_to_dict(init_request))

        # 然后发送发布到该通道流中的任何后续请求
        async for request, _ in channel_stream:
            logger.debug(f"向通道发送请求 - {channel} - {request}")
            await channel.send(request)

    async def _receive_messages(
        self, channel: WebSocketChannel, producer: Producer, lock: asyncio.Lock
    ):
        """
        循环处理来自生产者的消息接收

        :param channel: WebSocket的WebSocketChannel对象
        :param producer: 包含生产者信息的字典
        :param lock: 一个asyncio锁
        """
        while self._running:
            try:
                message = await asyncio.wait_for(channel.recv(), timeout=self.wait_timeout)

                try:
                    async with lock:
                        # 处理消息
                        self.handle_producer_message(producer, message)
                except Exception as e:
                    logger.exception(f"处理生产者消息时出错: {e}")

            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                # 尚未收到数据。检查连接并继续。
                try:
                    # 发送ping
                    pong = await channel.ping()
                    latency = await asyncio.wait_for(pong, timeout=self.ping_timeout) * 1000

                    logger.info(f"连接到 {channel} 仍然活跃，延迟: {latency}ms")
                    continue

                except Exception as e:
                    # 仅记录错误并继续重连
                    logger.warning(f"Ping错误 {channel} - {e} - {self.sleep_time}秒后重试")
                    logger.debug(e, exc_info=e)
                    raise

    def send_producer_request(self, producer_name: str, request: WSRequestSchema | dict[str, Any]):
        """
        将消息发布到生产者的消息流，以便由通道任务发送。

        :param producer_name: 要发布消息的生产者名称
        :param request: 要发送给生产者的请求
        """
        if isinstance(request, WSRequestSchema):
            request = schema_to_dict(request)

        if channel_stream := self._channel_streams.get(producer_name):
            channel_stream.publish(request)

    def handle_producer_message(self, producer: Producer, message: dict[str, Any]):
        """
        处理来自生产者的外部消息
        """
        producer_name = producer.get("name", "default")

        try:
            producer_message = WSMessageSchema.model_validate(message)
        except ValidationError as e:
            logger.error(f"来自 `{producer_name}` 的消息无效: {e}")
            return

        if not producer_message.data:
            logger.error(f"从 `{producer_name}` 收到空消息")
            return

        logger.debug(f"从 `{producer_name}` 收到类型为 `{producer_message.type}` 的消息")

        message_handler = self._message_handlers.get(producer_message.type)

        if not message_handler:
            logger.info(f"收到未处理的消息: `{producer_message.data}`，已忽略...")
            return

        message_handler(producer_name, producer_message)

    def _consume_whitelist_message(self, producer_name: str, message: WSMessageSchema):
        try:
            # 验证消息
            whitelist_message = WSWhitelistMessage.model_validate(message.model_dump())
        except ValidationError as e:
            logger.error(f"来自 `{producer_name}` 的消息无效: {e}")
            return

        # 将交易对列表数据添加到数据提供器
        self._dp._set_producer_pairs(whitelist_message.data, producer_name=producer_name)

        logger.debug(f"从 `{producer_name}` 消费了类型为 `RPCMessageType.WHITELIST` 的消息")

    def _consume_analyzed_df_message(self, producer_name: str, message: WSMessageSchema):
        try:
            df_message = WSAnalyzedDFMessage.model_validate(message.model_dump())
        except ValidationError as e:
            logger.error(f"来自 `{producer_name}` 的消息无效: {e}")
            return

        key = df_message.data.key
        df = df_message.data.df
        la = df_message.data.la

        pair, timeframe, candle_type = key

        if df.empty:
            logger.debug(f"收到 {key} 的空数据帧")
            return

        # 如果设置，从生产者中移除入场和出场信号
        if self._emc_config.get("remove_entry_exit_signals", False):
            df = remove_entry_exit_signals(df)

        logger.debug(f"收到 {key} 的 {len(df)} 根K线")

        did_append, n_missing = self._dp._add_external_df(
            pair,
            df,
            last_analyzed=la,
            timeframe=timeframe,
            candle_type=candle_type,
            producer_name=producer_name,
        )

        if not did_append:
            # 我们希望K线有重叠，以防某些数据已更改
            n_missing += 1
            # 如果缺失数量超过完整数据帧阈值，则设置为1500
            n_missing = n_missing if n_missing < FULL_DATAFRAME_THRESHOLD else 1500

            logger.warning(
                f"数据存在缺口或没有现有数据帧，向 `{producer_name}` 请求 {n_missing} 根K线"
                f"用于 {key}"
            )

            self.send_producer_request(
                producer_name, WSAnalyzedDFRequest(data={"limit": n_missing, "pair": pair})
            )
            return

        logger.debug(
            f"从 `{producer_name}` 消费了类型为 `RPCMessageType.ANALYZED_DF` 的消息，用于 {key}"
        )