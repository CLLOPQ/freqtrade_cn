import asyncio
import logging
import threading
from datetime import timedelta
from time import sleep
from unittest.mock import AsyncMock, MagicMock

from ccxt import NotSupported

from freqtrade.enums import CandleType
from freqtrade.exchange.exchange_ws import ExchangeWS
from ft_client.test_client.test_rest_client import log_has_re


def test_exchangews_init(mocker):
    """测试ExchangeWS初始化"""
    config = MagicMock()
    ccxt_object = MagicMock()
    mocker.patch("freqtrade.exchange.exchange_ws.ExchangeWS._start_forever", MagicMock())

    exchange_ws = ExchangeWS(config, ccxt_object)
    sleep(0.1)

    assert exchange_ws.config == config
    assert exchange_ws._ccxt_object == ccxt_object
    assert exchange_ws._thread.name == "ccxt_ws"
    assert exchange_ws._background_tasks == set()
    assert exchange_ws._klines_watching == set()
    assert exchange_ws._klines_scheduled == set()
    assert exchange_ws.klines_last_refresh == {}
    assert exchange_ws.klines_last_request == {}
    # 清理资源
    exchange_ws.cleanup()


def test_exchangews_cleanup_error(mocker, caplog):
    """测试ExchangeWS清理时的错误处理"""
    config = MagicMock()
    ccxt_object = MagicMock()
    ccxt_object.close = AsyncMock(side_effect=Exception("测试错误"))
    mocker.patch("freqtrade.exchange.exchange_ws.ExchangeWS._start_forever", MagicMock())

    exchange_ws = ExchangeWS(config, ccxt_object)
    patch_eventloop_threading(exchange_ws)

    sleep(0.1)
    exchange_ws.reset_connections()

    assert log_has_re("在_cleanup_async中发生异常", caplog)

    exchange_ws.cleanup()


def patch_eventloop_threading(exchange):
    """修补事件循环线程"""
    is_init = False

    def thread_fuck():
        nonlocal is_init
        exchange._loop = asyncio.new_event_loop()
        is_init = True
        exchange._loop.run_forever()

    x = threading.Thread(target=thread_fuck, daemon=True)
    x.start()
    while not is_init:
        pass


async def test_exchangews_ohlcv(mocker, time_machine, caplog):
    """测试ExchangeWS的OHLCV功能"""
    config = MagicMock()
    ccxt_object = MagicMock()
    caplog.set_level(logging.DEBUG)

    async def sleeper(*args, **kwargs):
        # 等待一段时间
        await asyncio.sleep(0.12)
        return MagicMock()

    ccxt_object.un_watch_ohlcv_for_symbols = AsyncMock(side_effect=NotSupported)

    ccxt_object.watch_ohlcv = AsyncMock(side_effect=sleeper)
    ccxt_object.close = AsyncMock()
    time_machine.move_to("2024-11-01 01:00:02 +00:00")

    mocker.patch("freqtrade.exchange.exchange_ws.ExchangeWS._start_forever", MagicMock())

    exchange_ws = ExchangeWS(config, ccxt_object)
    patch_eventloop_threading(exchange_ws)
    try:
        assert exchange_ws._klines_watching == set()
        assert exchange_ws._klines_scheduled == set()

        exchange_ws.schedule_ohlcv("ETH/BTC", "1m", CandleType.SPOT)
        exchange_ws.schedule_ohlcv("XRP/BTC", "1m", CandleType.SPOT)
        await asyncio.sleep(0.2)

        assert exchange_ws._klines_watching == {
            ("ETH/BTC", "1m", CandleType.SPOT),
            ("XRP/BTC", "1m", CandleType.SPOT),
        }
        assert exchange_ws._klines_scheduled == {
            ("ETH/BTC", "1m", CandleType.SPOT),
            ("XRP/BTC", "1m", CandleType.SPOT),
        }
        await asyncio.sleep(0.1)
        assert ccxt_object.watch_ohlcv.call_count == 6
        ccxt_object.watch_ohlcv.reset_mock()

        time_machine.shift(timedelta(minutes=5))
        exchange_ws.schedule_ohlcv("ETH/BTC", "1m", CandleType.SPOT)
        await asyncio.sleep(1)
        assert log_has_re("不支持un_watch_ohlcv_for_symbols: ", caplog)
        # XRP/BTC应该被清理
        assert exchange_ws._klines_watching == {
            ("ETH/BTC", "1m", CandleType.SPOT),
        }

        # 清理已发生
        ccxt_object.un_watch_ohlcv_for_symbols = AsyncMock(side_effect=ValueError)
        exchange_ws.schedule_ohlcv("ETH/BTC", "1m", CandleType.SPOT)
        assert exchange_ws._klines_watching == {
            ("ETH/BTC", "1m", CandleType.SPOT),
        }
        assert exchange_ws._klines_scheduled == {
            ("ETH/BTC", "1m", CandleType.SPOT),
        }

    finally:
        # 清理资源
        exchange_ws.cleanup()
    assert log_has_re("在_unwatch_ohlcv中发生异常", caplog)


async def test_exchangews_get_ohlcv(mocker, caplog):
    """测试ExchangeWS获取OHLCV数据"""
    config = MagicMock()
    ccxt_object = MagicMock()
    ccxt_object.ohlcvs = {
        "ETH/USDT": {
            "1m": [
                [1635840000000, 100, 200, 300, 400, 500],
                [1635840060000, 101, 201, 301, 401, 501],
                [1635840120000, 102, 202, 302, 402, 502],
            ],
            "5m": [
                [1635840000000, 100, 200, 300, 400, 500],
                [1635840300000, 105, 201, 301, 401, 501],
                [1635840600000, 102, 202, 302, 402, 502],
            ],
        }
    }
    mocker.patch("freqtrade.exchange.exchange_ws.ExchangeWS._start_forever", MagicMock())

    exchange_ws = ExchangeWS(config, ccxt_object)
    exchange_ws.klines_last_refresh = {
        ("ETH/USDT", "1m", CandleType.SPOT): 1635840120000,
        ("ETH/USDT", "5m", CandleType.SPOT): 1635840600000,
    }

    # 匹配最后一根K线时间 - drop hint为True
    resp = await exchange_ws.get_ohlcv("ETH/USDT", "1m", CandleType.SPOT, 1635840120000)
    assert resp[0] == "ETH/USDT"
    assert resp[1] == "1m"
    assert resp[3] == [
        [1635840000000, 100, 200, 300, 400, 500],
        [1635840060000, 101, 201, 301, 401, 501],
        [1635840120000, 102, 202, 302, 402, 502],
    ]
    assert resp[4] is True

    # 预期时间 > 最后一根K线时间 - drop hint为False
    resp = await exchange_ws.get_ohlcv("ETH/USDT", "1m", CandleType.SPOT, 1635840180000)
    assert resp[0] == "ETH/USDT"
    assert resp[1] == "1m"
    assert resp[3] == [
        [1635840000000, 100, 200, 300, 400, 500],
        [1635840060000, 101, 201, 301, 401, 501],
        [1635840120000, 102, 202, 302, 402, 502],
    ]
    assert resp[4] is False

    # 将"接收"时间更改为K线开始之前
    # 这应该触发"时间同步"警告
    exchange_ws.klines_last_refresh = {
        ("ETH/USDT", "1m", CandleType.SPOT): 1635840110000,
        ("ETH/USDT", "5m", CandleType.SPOT): 1635840600000,
    }
    msg = r".*K线日期 > 最后刷新时间.*"
    assert not log_has_re(msg, caplog)
    resp = await exchange_ws.get_ohlcv("ETH/USDT", "1m", CandleType.SPOT, 1635840120000)
    assert resp[0] == "ETH/USDT"
    assert resp[1] == "1m"
    assert resp[3] == [
        [1635840000000, 100, 200, 300, 400, 500],
        [1635840060000, 101, 201, 301, 401, 501],
        [1635840120000, 102, 202, 302, 402, 502],
    ]
    assert resp[4] is True

    assert log_has_re(msg, caplog)

    exchange_ws.cleanup()