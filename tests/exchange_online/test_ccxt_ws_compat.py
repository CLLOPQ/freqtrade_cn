import logging
from datetime import timedelta
from time import sleep

import pytest

from freqtrade.enums import CandleType
from freqtrade.exchange.exchange_utils import timeframe_to_prev_date
from freqtrade.util.datetime_helpers import dt_now
from tests.conftest import log_has_re
from tests.exchange_online.conftest import EXCHANGE_WS_FIXTURE_TYPE


@pytest.mark.longrun
@pytest.mark.timeout(3 * 60)
class TestCCXTExchangeWs:
    def test_ccxt_watch_ohlcv(self, exchange_ws: EXCHANGE_WS_FIXTURE_TYPE, caplog, mocker):
        exch, _exchangename, pair = exchange_ws

        assert exch._ws_async is not None
        timeframe = "1m"
        pair_tf = (pair, timeframe, CandleType.SPOT)
        m_hist = mocker.spy(exch, "_async_get_historic_ohlcv")
        m_cand = mocker.spy(exch, "_async_get_candle_history")

        while True:
            # 如果离分钟结束太近，则不开始测试
            if dt_now().second < 50 and dt_now().second > 1:
                break
            sleep(1)

        res = exch.refresh_latest_ohlcv([pair_tf])
        assert m_cand.call_count == 1

        # 当前未闭合的K线
        next_candle = timeframe_to_prev_date(timeframe, dt_now())
        now = next_candle - timedelta(seconds=1)
        # 当前已闭合的K线
        curr_candle = timeframe_to_prev_date(timeframe, now)

        assert pair_tf in exch._exchange_ws._klines_watching
        assert pair_tf in exch._exchange_ws._klines_scheduled
        assert res[pair_tf] is not None
        df1 = res[pair_tf]
        caplog.set_level(logging.DEBUG)
        assert df1.iloc[-1]["date"] == curr_candle

        # 等待下一根K线（最多可能需要1分钟）
        while True:
            caplog.clear()
            res = exch.refresh_latest_ohlcv([pair_tf])
            df2 = res[pair_tf]
            assert df2 is not None
            if df2.iloc[-1]["date"] == next_candle:
                break
            assert df2.iloc[-1]["date"] == curr_candle
            sleep(1)

        assert m_hist.call_count == 0
        # 不应尝试第二次调用fetch_ohlcv
        assert m_cand.call_count == 1
        assert log_has_re(r"watch result.*", caplog)