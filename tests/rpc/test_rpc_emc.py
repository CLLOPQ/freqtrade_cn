"""
外部消息消费者（rpc/external_message_consumer.py）的单元测试文件
"""

import asyncio
import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
import websockets

from freqtrade.data.dataprovider import DataProvider
from freqtrade.rpc.external_message_consumer import ExternalMessageConsumer
from tests.conftest import log_has, log_has_re, log_has_when


_TEST_WS_TOKEN = "secret_Ws_t0ken"
_TEST_WS_HOST = "127.0.0.1"
_TEST_WS_PORT = 9989


@pytest.fixture
def patched_emc(default_conf, mocker):
    default_conf.update(
        {
            "external_message_consumer": {
                "enabled": True,
                "producers": [
                    {"name": "default", "host": "null", "port": 9891, "ws_token": _TEST_WS_TOKEN}
                ],
            }
        }
    )
    dataprovider = DataProvider(default_conf, None, None, None)
    emc = ExternalMessageConsumer(default_conf, dataprovider)

    try:
        yield emc
    finally:
        emc.shutdown()


def test_emc_start(patched_emc, caplog):
    # 测试消息是否被打印
    assert log_has_when("启动外部消息消费者（ExternalMessageConsumer）", caplog, "setup")
    # 测试线程和事件循环对象是否被创建
    assert patched_emc._thread and patched_emc._loop

    # 测试再次调用start时不会有新的操作
    prev_thread = patched_emc._thread
    patched_emc.start()
    assert prev_thread == patched_emc._thread


def test_emc_shutdown(patched_emc, caplog):
    patched_emc.shutdown()

    assert log_has("停止外部消息消费者（ExternalMessageConsumer）", caplog)
    # 测试事件循环已停止
    assert patched_emc._loop is None
    # 测试线程已停止
    assert patched_emc._thread is None

    caplog.clear()
    patched_emc.shutdown()

    # 测试函数不会再次运行，因为已经调用过一次
    assert not log_has("停止外部消息消费者（ExternalMessageConsumer）", caplog)


def test_emc_init(patched_emc):
    # 测试设置是否正确
    assert patched_emc.initial_candle_limit <= 1500
    assert patched_emc.wait_timeout > 0
    assert patched_emc.sleep_time > 0


# 是否需要参数化？
def test_emc_handle_producer_message(patched_emc, caplog, ohlcv_history):
    test_producer = {"name": "test", "url": "ws://test", "ws_token": "test"}
    producer_name = test_producer["name"]
    invalid_msg = r"无效消息 .+"

    caplog.set_level(logging.DEBUG)

    # 测试处理白名单消息
    whitelist_message = {"type": "whitelist", "data": ["BTC/USDT"]}
    patched_emc.handle_producer_message(test_producer, whitelist_message)

    assert log_has(f"从 `{producer_name}` 收到类型为 `whitelist` 的消息", caplog)
    assert log_has(
        f"消费来自 `{producer_name}` 的类型为 `RPCMessageType.WHITELIST` 的消息", caplog
    )

    # 测试处理分析后的单根K线数据消息
    df_message = {
        "type": "analyzed_df",
        "data": {
            "key": ("BTC/USDT", "5m", "spot"),
            "df": ohlcv_history,
            "la": datetime.now(timezone.utc),
        },
    }
    patched_emc.handle_producer_message(test_producer, df_message)

    assert log_has(f"从 `{producer_name}` 收到类型为 `analyzed_df` 的消息", caplog)
    assert log_has_re(r"数据存在缺失或无现有数据框,.+", caplog)

    # 测试未处理的消息类型
    unhandled_message = {"type": "status", "data": "RUNNING"}
    patched_emc.handle_producer_message(test_producer, unhandled_message)

    assert log_has_re(r"收到未处理的消息\: .*", caplog)

    # 测试格式错误的消息
    caplog.clear()
    malformed_message = {"type": "whitelist", "data": {"pair": "BTC/USDT"}}
    patched_emc.handle_producer_message(test_producer, malformed_message)

    assert log_has_re(invalid_msg, caplog)
    caplog.clear()

    malformed_message = {
        "type": "analyzed_df",
        "data": {"key": "BTC/USDT", "df": ohlcv_history, "la": datetime.now(timezone.utc)},
    }
    patched_emc.handle_producer_message(test_producer, malformed_message)

    assert log_has(f"从 `{producer_name}` 收到类型为 `analyzed_df` 的消息", caplog)
    assert log_has_re(invalid_msg, caplog)
    caplog.clear()

    # 空数据框
    malformed_message = {
        "type": "analyzed_df",
        "data": {
            "key": ("BTC/USDT", "5m", "spot"),
            "df": ohlcv_history.loc[ohlcv_history["open"] < 0],
            "la": datetime.now(timezone.utc),
        },
    }
    patched_emc.handle_producer_message(test_producer, malformed_message)

    assert log_has(f"从 `{producer_name}` 收到类型为 `analyzed_df` 的消息", caplog)
    assert not log_has_re(invalid_msg, caplog)
    assert log_has_re(r"收到空数据框，针对.+", caplog)

    caplog.clear()
    malformed_message = {"some": "stuff"}
    patched_emc.handle_producer_message(test_producer, malformed_message)

    assert log_has_re(invalid_msg, caplog)
    caplog.clear()

    caplog.clear()
    malformed_message = {"type": "whitelist", "data": None}
    patched_emc.handle_producer_message(test_producer, malformed_message)

    assert log_has_re(r"空消息 .+", caplog)


async def test_emc_create_connection_success(default_conf, caplog, mocker):
    default_conf.update(
        {
            "external_message_consumer": {
                "enabled": True,
                "producers": [
                    {
                        "name": "default",
                        "host": _TEST_WS_HOST,
                        "port": _TEST_WS_PORT,
                        "ws_token": _TEST_WS_TOKEN,
                    }
                ],
                "wait_timeout": 60,
                "ping_timeout": 60,
                "sleep_timeout": 60,
            }
        }
    )

    mocker.patch(
        "freqtrade.rpc.external_message_consumer.ExternalMessageConsumer.start", MagicMock()
    )
    dp = DataProvider(default_conf, None, None, None)
    emc = ExternalMessageConsumer(default_conf, dp)

    test_producer = default_conf["external_message_consumer"]["producers"][0]
    lock = asyncio.Lock()

    emc._running = True

    async def eat(websocket):
        emc._running = False

    try:
        async with websockets.serve(eat, _TEST_WS_HOST, _TEST_WS_PORT):
            await emc._create_connection(test_producer, lock)

        assert log_has_re(r"已连接到通道.+", caplog)
    finally:
        emc.shutdown()


@pytest.mark.parametrize(
    "host,port",
    [
        (_TEST_WS_HOST, -1),
        ("10000.1241..2121/", _TEST_WS_PORT),
    ],
)
async def test_emc_create_connection_invalid_url(default_conf, caplog, mocker, host, port):
    default_conf.update(
        {
            "external_message_consumer": {
                "enabled": True,
                "producers": [
                    {"name": "default", "host": host, "port": port, "ws_token": _TEST_WS_TOKEN}
                ],
                "wait_timeout": 60,
                "ping_timeout": 60,
                "sleep_timeout": 60,
            }
        }
    )

    dp = DataProvider(default_conf, None, None, None)
    # 显式测试中显式处理start以避免线程问题
    mocker.patch("freqtrade.rpc.external_message_consumer.ExternalMessageConsumer.start")
    mocker.patch("freqtrade.rpc.api_server.ws.channel.create_channel")
    emc = ExternalMessageConsumer(default_conf, dp)

    try:
        emc._running = True
        await emc._create_connection(emc.producers[0], asyncio.Lock())
        assert log_has_re(r".+ 是无效的WebSocket URL .+", caplog)
    finally:
        emc.shutdown()


async def test_emc_create_connection_error(default_conf, caplog, mocker):
    default_conf.update(
        {
            "external_message_consumer": {
                "enabled": True,
                "producers": [
                    {
                        "name": "default",
                        "host": _TEST_WS_HOST,
                        "port": _TEST_WS_PORT,
                        "ws_token": _TEST_WS_TOKEN,
                    }
                ],
                "wait_timeout": 60,
                "ping_timeout": 60,
                "sleep_timeout": 60,
            }
        }
    )

    # 测试意外错误
    mocker.patch("websockets.connect", side_effect=RuntimeError)

    dp = DataProvider(default_conf, None, None, None)
    emc = ExternalMessageConsumer(default_conf, dp)

    try:
        await asyncio.sleep(0.05)
        assert log_has("发生意外错误：", caplog)
    finally:
        emc.shutdown()


async def test_emc_receive_messages_valid(default_conf, caplog, mocker):
    caplog.set_level(logging.DEBUG)

    default_conf.update(
        {
            "external_message_consumer": {
                "enabled": True,
                "producers": [
                    {
                        "name": "default",
                        "host": _TEST_WS_HOST,
                        "port": _TEST_WS_PORT,
                        "ws_token": _TEST_WS_TOKEN,
                    }
                ],
                "wait_timeout": 1,
                "ping_timeout": 60,
                "sleep_time": 60,
            }
        }
    )

    mocker.patch(
        "freqtrade.rpc.external_message_consumer.ExternalMessageConsumer.start", MagicMock()
    )

    lock = asyncio.Lock()
    test_producer = default_conf["external_message_consumer"]["producers"][0]

    dp = DataProvider(default_conf, None, None, None)
    emc = ExternalMessageConsumer(default_conf, dp)

    class TestChannel:
        async def recv(self, *args, **kwargs):
            emc._running = False
            return {"type": "whitelist", "data": ["BTC/USDT"]}

        async def ping(self, *args, **kwargs):
            return asyncio.Future()

    try:
        emc._running = True
        await emc._receive_messages(TestChannel(), test_producer, lock)

        assert log_has_re(r"收到类型为 `whitelist` 的消息.+", caplog)
    finally:
        emc.shutdown()


async def test_emc_receive_messages_invalid(default_conf, caplog, mocker):
    default_conf.update(
        {
            "external_message_consumer": {
                "enabled": True,
                "producers": [
                    {
                        "name": "default",
                        "host": _TEST_WS_HOST,
                        "port": _TEST_WS_PORT,
                        "ws_token": _TEST_WS_TOKEN,
                    }
                ],
                "wait_timeout": 1,
                "ping_timeout": 60,
                "sleep_time": 60,
            }
        }
    )

    mocker.patch(
        "freqtrade.rpc.external_message_consumer.ExternalMessageConsumer.start", MagicMock()
    )

    lock = asyncio.Lock()
    test_producer = default_conf["external_message_consumer"]["producers"][0]

    dp = DataProvider(default_conf, None, None, None)
    emc = ExternalMessageConsumer(default_conf, dp)

    class TestChannel:
        async def recv(self, *args, **kwargs):
            emc._running = False
            return {"type": ["BTC/USDT"]}

        async def ping(self, *args, **kwargs):
            return asyncio.Future()

    try:
        emc._running = True
        await emc._receive_messages(TestChannel(), test_producer, lock)

        assert log_has_re(r"来自.+的无效消息", caplog)
    finally:
        emc.shutdown()


async def test_emc_receive_messages_timeout(default_conf, caplog, mocker):
    default_conf.update(
        {
            "external_message_consumer": {
                "enabled": True,
                "producers": [
                    {
                        "name": "default",
                        "host": _TEST_WS_HOST,
                        "port": _TEST_WS_PORT,
                        "ws_token": _TEST_WS_TOKEN,
                    }
                ],
                "wait_timeout": 0.1,
                "ping_timeout": 1,
                "sleep_time": 1,
            }
        }
    )

    mocker.patch(
        "freqtrade.rpc.external_message_consumer.ExternalMessageConsumer.start", MagicMock()
    )

    lock = asyncio.Lock()
    test_producer = default_conf["external_message_consumer"]["producers"][0]

    dp = DataProvider(default_conf, None, None, None)
    emc = ExternalMessageConsumer(default_conf, dp)

    def change_running():
        emc._running = not emc._running

    class TestChannel:
        async def recv(self, *args, **kwargs):
            await asyncio.sleep(0.2)

        async def ping(self, *args, **kwargs):
            return asyncio.Future()

    try:
        change_running()

        with pytest.raises(asyncio.TimeoutError):
            await emc._receive_messages(TestChannel(), test_producer, lock)

        assert log_has_re(r"Ping错误.+", caplog)
    finally:
        emc.shutdown()


async def test_emc_receive_messages_handle_error(default_conf, caplog, mocker):
    default_conf.update(
        {
            "external_message_consumer": {
                "enabled": True,
                "producers": [
                    {
                        "name": "default",
                        "host": _TEST_WS_HOST,
                        "port": _TEST_WS_PORT,
                        "ws_token": _TEST_WS_TOKEN,
                    }
                ],
                "wait_timeout": 1,
                "ping_timeout": 1,
                "sleep_time": 1,
            }
        }
    )

    mocker.patch(
        "freqtrade.rpc.external_message_consumer.ExternalMessageConsumer.start", MagicMock()
    )

    lock = asyncio.Lock()
    test_producer = default_conf["external_message_consumer"]["producers"][0]

    dp = DataProvider(default_conf, None, None, None)
    emc = ExternalMessageConsumer(default_conf, dp)

    emc.handle_producer_message = MagicMock(side_effect=Exception)

    class TestChannel:
        async def recv(self, *args, **kwargs):
            emc._running = False
            return {"type": "whitelist", "data": ["BTC/USDT"]}

        async def ping(self, *args, **kwargs):
            return asyncio.Future()

    try:
        emc._running = True
        await emc._receive_messages_messages(TestChannel(), test_producer, lock)

        assert log_has_re(r"处理生产者消息时出错.+", caplog)
    finally:
        emc.shutdown()