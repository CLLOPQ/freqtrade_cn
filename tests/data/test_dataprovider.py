from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from pandas import DataFrame, Timestamp

from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import CandleType, RunMode
from freqtrade.exceptions import ExchangeError, OperationalException
from freqtrade.plugins.pairlistmanager import PairListManager
from tests.conftest import EXMS, generate_test_data, get_patched_exchange


@pytest.mark.parametrize(
    "candle_type",
    [
        "mark",
        "",
    ],
)
def test_dp_ohlcv(mocker, default_conf, ohlcv_history, candle_type):
    """测试数据提供器的OHLCV数据获取功能"""
    default_conf["runmode"] = RunMode.DRY_RUN
    timeframe = default_conf["timeframe"]
    exchange = get_patched_exchange(mocker, default_conf)
    candletype = CandleType.from_string(candle_type)
    exchange._klines[("XRP/BTC", timeframe, candletype)] = ohlcv_history
    exchange._klines[("UNITTEST/BTC", timeframe, candletype)] = ohlcv_history

    dp = DataProvider(default_conf, exchange)
    assert dp.runmode == RunMode.DRY_RUN
    assert ohlcv_history.equals(dp.ohlcv("UNITTEST/BTC", timeframe, candle_type=candletype))
    assert isinstance(dp.ohlcv("UNITTEST/BTC", timeframe, candle_type=candletype), DataFrame)
    assert dp.ohlcv("UNITTEST/BTC", timeframe, candle_type=candletype) is not ohlcv_history
    assert dp.ohlcv("UNITTEST/BTC", timeframe, copy=False, candle_type=candletype) is ohlcv_history
    assert not dp.ohlcv("UNITTEST/BTC", timeframe, candle_type=candletype).empty
    assert dp.ohlcv("NONSENSE/AAA", timeframe, candle_type=candletype).empty

    # 测试带参数和不带参数的情况
    assert dp.ohlcv("UNITTEST/BTC", timeframe, candle_type=candletype).equals(
        dp.ohlcv("UNITTEST/BTC", candle_type=candle_type)
    )

    default_conf["runmode"] = RunMode.LIVE
    dp = DataProvider(default_conf, exchange)
    assert dp.runmode == RunMode.LIVE
    assert isinstance(dp.ohlcv("UNITTEST/BTC", timeframe, candle_type=candle_type), DataFrame)

    default_conf["runmode"] = RunMode.BACKTEST
    dp = DataProvider(default_conf, exchange)
    assert dp.runmode == RunMode.BACKTEST
    assert dp.ohlcv("UNITTEST/BTC", timeframe, candle_type=candle_type).empty


def test_historic_ohlcv(mocker, default_conf, ohlcv_history):
    """测试历史OHLCV数据获取功能"""
    historymock = MagicMock(return_value=ohlcv_history)
    mocker.patch("freqtrade.data.dataprovider.load_pair_history", historymock)

    dp = DataProvider(default_conf, None)
    data = dp.historic_ohlcv("UNITTEST/BTC", "5m")
    assert isinstance(data, DataFrame)
    assert historymock.call_count == 1
    assert historymock.call_args_list[0][1]["timeframe"] == "5m"


def test_historic_trades(mocker, default_conf, trades_history_df):
    """测试历史交易数据获取功能"""
    historymock = MagicMock(return_value=trades_history_df)
    mocker.patch(
        "freqtrade.data.history.datahandlers.featherdatahandler.FeatherDataHandler._trades_load",
        historymock,
    )

    dp = DataProvider(default_conf, None)
    # 实盘模式下测试
    with pytest.raises(OperationalException, match=r"数据提供器无法访问交易所。"):
        dp.trades("UNITTEST/BTC", "5m")

    exchange = get_patched_exchange(mocker, default_conf)
    dp = DataProvider(default_conf, exchange)
    data = dp.trades("UNITTEST/BTC", "5m")

    assert isinstance(data, DataFrame)
    assert len(data) == 0

    # 切换到回测模式
    default_conf["runmode"] = RunMode.BACKTEST
    default_conf["dataformat_trades"] = "feather"
    exchange = get_patched_exchange(mocker, default_conf)
    dp = DataProvider(default_conf, exchange)
    data = dp.trades("UNITTEST/BTC", "5m")
    assert isinstance(data, DataFrame)
    assert len(data) == len(trades_history_df)


def test_historic_ohlcv_dataformat(mocker, default_conf, ohlcv_history):
    """测试不同数据格式的历史OHLCV数据加载"""
    parquetloadmock = MagicMock(return_value=ohlcv_history)
    featherloadmock = MagicMock(return_value=ohlcv_history)
    mocker.patch(
        "freqtrade.data.history.datahandlers.parquetdatahandler.ParquetDataHandler._ohlcv_load",
        parquetloadmock,
    )
    mocker.patch(
        "freqtrade.data.history.datahandlers.featherdatahandler.FeatherDataHandler._ohlcv_load",
        featherloadmock,
    )

    default_conf["runmode"] = RunMode.BACKTEST
    exchange = get_patched_exchange(mocker, default_conf)
    dp = DataProvider(default_conf, exchange)
    data = dp.historic_ohlcv("UNITTEST/BTC", "5m")
    assert isinstance(data, DataFrame)
    parquetloadmock.assert_not_called()
    featherloadmock.assert_called_once()

    # 切换到parquet数据格式
    parquetloadmock.reset_mock()
    featherloadmock.reset_mock()
    default_conf["dataformat_ohlcv"] = "parquet"
    dp = DataProvider(default_conf, exchange)
    data = dp.historic_ohlcv("UNITTEST/BTC", "5m")
    assert isinstance(data, DataFrame)
    parquetloadmock.assert_called_once()
    featherloadmock.assert_not_called()


@pytest.mark.parametrize(
    "candle_type",
    [
        "mark",
        "futures",
        "",
    ],
)
def test_get_pair_dataframe(mocker, default_conf, ohlcv_history, candle_type):
    """测试获取交易对数据框的功能"""
    default_conf["runmode"] = RunMode.DRY_RUN
    timeframe = default_conf["timeframe"]
    exchange = get_patched_exchange(mocker, default_conf)
    candletype = CandleType.from_string(candle_type)
    exchange._klines[("XRP/BTC", timeframe, candletype)] = ohlcv_history
    exchange._klines[("UNITTEST/BTC", timeframe, candletype)] = ohlcv_history

    dp = DataProvider(default_conf, exchange)
    assert dp.runmode == RunMode.DRY_RUN
    assert ohlcv_history.equals(
        dp.get_pair_dataframe("UNITTEST/BTC", timeframe, candle_type=candle_type)
    )
    assert ohlcv_history.equals(
        dp.get_pair_dataframe("UNITTEST/BTC", timeframe, candle_type=candletype)
    )
    assert isinstance(
        dp.get_pair_dataframe("UNITTEST/BTC", timeframe, candle_type=candle_type), DataFrame
    )
    assert (
        dp.get_pair_dataframe("UNITTEST/BTC", timeframe, candle_type=candle_type)
        is not ohlcv_history
    )
    assert not dp.get_pair_dataframe("UNITTEST/BTC", timeframe, candle_type=candle_type).empty
    assert dp.get_pair_dataframe("NONSENSE/AAA", timeframe, candle_type=candle_type).empty

    # 测试带参数和不带参数的情况
    assert dp.get_pair_dataframe("UNITTEST/BTC", timeframe, candle_type=candle_type).equals(
        dp.get_pair_dataframe("UNITTEST/BTC", candle_type=candle_type)
    )

    default_conf["runmode"] = RunMode.LIVE
    dp = DataProvider(default_conf, exchange)
    assert dp.runmode == RunMode.LIVE
    assert isinstance(
        dp.get_pair_dataframe("UNITTEST/BTC", timeframe, candle_type=candle_type), DataFrame
    )
    assert dp.get_pair_dataframe("NONSENSE/AAA", timeframe, candle_type=candle_type).empty

    historymock = MagicMock(return_value=ohlcv_history)
    mocker.patch("freqtrade.data.dataprovider.load_pair_history", historymock)
    default_conf["runmode"] = RunMode.BACKTEST
    dp = DataProvider(default_conf, exchange)
    assert dp.runmode == RunMode.BACKTEST
    df = dp.get_pair_dataframe("UNITTEST/BTC", timeframe, candle_type=candle_type)
    assert isinstance(df, DataFrame)
    assert len(df) == 3  # ohlcv_history模拟数据只有3行

    dp._set_dataframe_max_date(ohlcv_history.iloc[-1]["date"])
    df = dp.get_pair_dataframe("UNITTEST/BTC", timeframe, candle_type=candle_type)
    assert isinstance(df, DataFrame)
    assert len(df) == 2  # 现在ohlcv_history限制为2行


def test_available_pairs(mocker, default_conf, ohlcv_history):
    """测试可用交易对的获取功能"""
    exchange = get_patched_exchange(mocker, default_conf)
    timeframe = default_conf["timeframe"]
    exchange._klines[("XRP/BTC", timeframe)] = ohlcv_history
    exchange._klines[("UNITTEST/BTC", timeframe)] = ohlcv_history

    dp = DataProvider(default_conf, exchange)
    assert len(dp.available_pairs) == 2
    assert dp.available_pairs == [
        ("XRP/BTC", timeframe),
        ("UNITTEST/BTC", timeframe),
    ]


def test_producer_pairs(default_conf):
    """测试生产者交易对的管理功能"""
    dataprovider = DataProvider(default_conf, None)

    producer = "default"
    whitelist = ["XRP/BTC", "ETH/BTC"]
    assert len(dataprovider.get_producer_pairs(producer)) == 0

    dataprovider._set_producer_pairs(whitelist, producer)
    assert len(dataprovider.get_producer_pairs(producer)) == 2

    new_whitelist = ["BTC/USDT"]
    dataprovider._set_producer_pairs(new_whitelist, producer)
    assert dataprovider.get_producer_pairs(producer) == new_whitelist

    assert dataprovider.get_producer_pairs("bad") == []


def test_get_producer_df(default_conf):
    """测试获取生产者数据框的功能"""
    dataprovider = DataProvider(default_conf, None)
    ohlcv_history = generate_test_data("5m", 150)
    pair = "BTC/USDT"
    timeframe = default_conf["timeframe"]
    candle_type = CandleType.SPOT

    empty_la = datetime.fromtimestamp(0, tz=timezone.utc)
    now = datetime.now(timezone.utc)

    # 尚未添加数据，任何请求都应返回空数据框
    dataframe, la = dataprovider.get_producer_df(pair, timeframe, candle_type)
    assert dataframe.empty
    assert la == empty_la

    # 已添加数据，应返回添加的数据框
    dataprovider._add_external_df(pair, ohlcv_history, now, timeframe, candle_type)
    dataframe, la = dataprovider.get_producer_df(pair, timeframe, candle_type)
    assert len(dataframe) > 0
    assert la > empty_la

    # 该生产者没有数据，应返回空数据框
    dataframe, la = dataprovider.get_producer_df(pair, producer_name="bad")
    assert dataframe.empty
    assert la == empty_la

    # 不存在的时间框架，返回空数据框
    _dataframe, la = dataprovider.get_producer_df(pair, timeframe="1h")
    assert dataframe.empty
    assert la == empty_la


def test_emit_df(mocker, default_conf, ohlcv_history):
    """测试数据框发送功能"""
    mocker.patch("freqtrade.rpc.rpc_manager.RPCManager.__init__", MagicMock())
    rpc_mock = mocker.patch("freqtrade.rpc.rpc_manager.RPCManager", MagicMock())
    send_mock = mocker.patch("freqtrade.rpc.rpc_manager.RPCManager.send_msg", MagicMock())

    dataprovider = DataProvider(default_conf, exchange=None, rpc=rpc_mock)
    dataprovider_no_rpc = DataProvider(default_conf, exchange=None)

    pair = "BTC/USDT"

    # 尚未发送任何内容
    assert send_mock.call_count == 0

    # 添加了RPC，调用emit，应该调用send_msg
    dataprovider._emit_df(pair, ohlcv_history, False)
    assert send_mock.call_count == 1

    send_mock.reset_mock()
    dataprovider._emit_df(pair, ohlcv_history, True)
    assert send_mock.call_count == 2

    send_mock.reset_mock()

    # 未添加RPC，调用emit，不应调用send_msg
    dataprovider_no_rpc._emit_df(pair, ohlcv_history, False)
    assert send_mock.call_count == 0


def test_refresh(mocker, default_conf):
    """测试数据刷新功能"""
    refresh_mock = mocker.patch(f"{EXMS}.refresh_latest_ohlcv")
    mock_refresh_trades = mocker.patch(f"{EXMS}.refresh_latest_trades")

    exchange = get_patched_exchange(mocker, default_conf, exchange="binance")
    timeframe = default_conf["timeframe"]
    pairs = [("XRP/BTC", timeframe), ("UNITTEST/BTC", timeframe)]

    pairs_non_trad = [("ETH/USDT", timeframe), ("BTC/TUSD", "1h")]

    dp = DataProvider(default_conf, exchange)
    dp.refresh(pairs)
    assert mock_refresh_trades.call_count == 0
    assert refresh_mock.call_count == 1
    assert len(refresh_mock.call_args[0]) == 1
    assert len(refresh_mock.call_args[0][0]) == len(pairs)
    assert refresh_mock.call_args[0][0] == pairs

    refresh_mock.reset_mock()
    dp.refresh(pairs, pairs_non_trad)
    assert mock_refresh_trades.call_count == 0
    assert refresh_mock.call_count == 1
    assert len(refresh_mock.call_args[0]) == 1
    assert len(refresh_mock.call_args[0][0]) == len(pairs) + len(pairs_non_trad)
    assert refresh_mock.call_args[0][0] == pairs + pairs_non_trad

    # 测试公共交易数据
    refresh_mock.reset_mock()
    refresh_mock.reset_mock()
    default_conf["exchange"]["use_public_trades"] = True
    dp.refresh(pairs, pairs_non_trad)
    assert mock_refresh_trades.call_count == 1
    assert refresh_mock.call_count == 1


def test_orderbook(mocker, default_conf, order_book_l2):
    """测试订单簿数据获取功能"""
    api_mock = MagicMock()
    api_mock.fetch_l2_order_book = order_book_l2
    exchange = get_patched_exchange(mocker, default_conf, api_mock=api_mock)

    dp = DataProvider(default_conf, exchange)
    res = dp.orderbook("ETH/BTC", 5)
    assert order_book_l2.call_count == 1
    assert order_book_l2.call_args_list[0][0][0] == "ETH/BTC"
    assert order_book_l2.call_args_list[0][0][1] >= 5

    assert isinstance(res, dict)
    assert "bids" in res
    assert "asks" in res


def test_market(mocker, default_conf, markets):
    """测试市场信息获取功能"""
    api_mock = MagicMock()
    api_mock.markets = markets
    exchange = get_patched_exchange(mocker, default_conf, api_mock=api_mock)

    dp = DataProvider(default_conf, exchange)
    res = dp.market("ETH/BTC")

    assert isinstance(res, dict)
    assert "symbol" in res
    assert res["symbol"] == "ETH/BTC"

    res = dp.market("UNITTEST/BTC")
    assert res is None


def test_ticker(mocker, default_conf, tickers):
    """测试行情数据获取功能"""
    ticker_mock = MagicMock(return_value=tickers()["ETH/BTC"])
    mocker.patch(f"{EXMS}.fetch_ticker", ticker_mock)
    exchange = get_patched_exchange(mocker, default_conf)
    dp = DataProvider(default_conf, exchange)
    res = dp.ticker("ETH/BTC")
    assert isinstance(res, dict)
    assert "symbol" in res
    assert res["symbol"] == "ETH/BTC"

    ticker_mock = MagicMock(side_effect=ExchangeError("交易对未找到"))
    mocker.patch(f"{EXMS}.fetch_ticker", ticker_mock)
    exchange = get_patched_exchange(mocker, default_conf)
    dp = DataProvider(default_conf, exchange)
    res = dp.ticker("UNITTEST/BTC")
    assert res == {}


def test_current_whitelist(mocker, default_conf, tickers):
    """测试当前白名单获取功能"""
    # 将默认配置修改为volumepairlist
    default_conf["pairlists"][0] = {"method": "VolumePairList", "number_assets": 5}

    mocker.patch.multiple(EXMS, exchange_has=MagicMock(return_value=True), get_tickers=tickers)
    exchange = get_patched_exchange(mocker, default_conf)

    pairlist = PairListManager(exchange, default_conf)
    dp = DataProvider(default_conf, exchange, pairlist)

    # 模拟从交易所获取的交易量交易对
    pairlist.refresh_pairlist()

    assert dp.current_whitelist() == pairlist._whitelist
    # 两个列表的标识不应相同，而应是副本
    assert dp.current_whitelist() is not pairlist._whitelist

    with pytest.raises(OperationalException):
        dp = DataProvider(default_conf, exchange)
        dp.current_whitelist()


def test_get_analyzed_dataframe(mocker, default_conf, ohlcv_history):
    """测试获取已分析的数据框功能"""
    default_conf["runmode"] = RunMode.DRY_RUN

    timeframe = default_conf["timeframe"]
    exchange = get_patched_exchange(mocker, default_conf)

    dp = DataProvider(default_conf, exchange)
    dp._set_cached_df("XRP/BTC", timeframe, ohlcv_history, CandleType.SPOT)
    dp._set_cached_df("UNITTEST/BTC", timeframe, ohlcv_history, CandleType.SPOT)

    assert dp.runmode == RunMode.DRY_RUN
    dataframe, time = dp.get_analyzed_dataframe("UNITTEST/BTC", timeframe)
    assert ohlcv_history.equals(dataframe)
    assert isinstance(time, datetime)

    dataframe, time = dp.get_analyzed_dataframe("XRP/BTC", timeframe)
    assert ohlcv_history.equals(dataframe)
    assert isinstance(time, datetime)

    dataframe, time = dp.get_analyzed_dataframe("NOTHING/BTC", timeframe)
    assert dataframe.empty
    assert isinstance(time, datetime)
    assert time == datetime(1970, 1, 1, tzinfo=timezone.utc)

    # 测试回测模式
    default_conf["runmode"] = RunMode.BACKTEST
    dp._set_dataframe_max_index("XRP/BTC", 1)
    dataframe, time = dp.get_analyzed_dataframe("XRP/BTC", timeframe)

    assert len(dataframe) == 1

    dp._set_dataframe_max_index("XRP/BTC", 2)
    dataframe, time = dp.get_analyzed_dataframe("XRP/BTC", timeframe)
    assert len(dataframe) == 2

    dp._set_dataframe_max_index("XRP/BTC", 3)
    dataframe, time = dp.get_analyzed_dataframe("XRP/BTC", timeframe)
    assert len(dataframe) == 3

    dp._set_dataframe_max_index("XRP/BTC", 500)
    dataframe, time = dp.get_analyzed_dataframe("XRP/BTC", timeframe)
    assert len(dataframe) == len(ohlcv_history)


def test_no_exchange_mode(default_conf):
    """测试无交易所模式下的功能限制"""
    dp = DataProvider(default_conf, None)

    message = "数据提供器无法访问交易所。"

    with pytest.raises(OperationalException, match=message):
        dp.refresh([()])

    with pytest.raises(OperationalException, match=message):
        dp.ohlcv("XRP/USDT", "5m", "")

    with pytest.raises(OperationalException, match=message):
        dp.market("XRP/USDT")

    with pytest.raises(OperationalException, match=message):
        dp.ticker("XRP/USDT")

    with pytest.raises(OperationalException, match=message):
        dp.orderbook("XRP/USDT", 20)

    with pytest.raises(OperationalException, match=message):
        dp.available_pairs()


def test_dp_send_msg(default_conf):
    """测试数据提供器的消息发送功能"""
    default_conf["runmode"] = RunMode.DRY_RUN

    default_conf["timeframe"] = "1h"
    dp = DataProvider(default_conf, None)
    msg = "测试消息"
    dp.send_msg(msg)

    assert msg in dp._msg_queue
    dp._msg_queue.pop()
    assert msg not in dp._msg_queue
    # 由于缓存，消息不会被重新发送
    dp.send_msg(msg)
    assert msg not in dp._msg_queue
    dp.send_msg(msg, always_send=True)
    assert msg in dp._msg_queue

    default_conf["runmode"] = RunMode.BACKTEST
    dp = DataProvider(default_conf, None)
    dp.send_msg(msg, always_send=True)
    assert msg not in dp._msg_queue


def test_dp__add_external_df(default_conf_usdt):
    """测试添加外部数据框的功能"""
    timeframe = "1h"
    default_conf_usdt["timeframe"] = timeframe
    dp = DataProvider(default_conf_usdt, None)
    df = generate_test_data(timeframe, 24, "2022-01-01 00:00:00+00:00")
    last_analyzed = datetime.now(timezone.utc)

    res = dp._add_external_df("ETH/USDT", df, last_analyzed, timeframe, CandleType.SPOT)
    assert res[0] is False
    # 为什么是1000？
    assert res[1] == 1000

    # 强制添加数据框
    dp._replace_external_df("ETH/USDT", df, last_analyzed, timeframe, CandleType.SPOT)
    # BTC尚未存储
    res = dp._add_external_df("BTC/USDT", df, last_analyzed, timeframe, CandleType.SPOT)
    assert res[0] is False
    df_res, _ = dp.get_producer_df("ETH/USDT", timeframe, CandleType.SPOT)
    assert len(df_res) == 24

    # 再次添加相同的数据框 - 数据框大小不应改变
    res = dp._add_external_df("ETH/USDT", df, last_analyzed, timeframe, CandleType.SPOT)
    assert res[0] is True
    assert isinstance(res[1], int)
    assert res[1] == 0
    df, _ = dp.get_producer_df("ETH/USDT", timeframe, CandleType.SPOT)
    assert len(df) == 24

    # 添加新的一天数据
    df2 = generate_test_data(timeframe, 24, "2022-01-02 00:00:00+00:00")

    res = dp._add_external_df("ETH/USDT", df2, last_analyzed, timeframe, CandleType.SPOT)
    assert res[0] is True
    assert isinstance(res[1], int)
    assert res[1] == 0
    df, _ = dp.get_producer_df("ETH/USDT", timeframe, CandleType.SPOT)
    assert len(df) == 48

    # 添加有12小时偏移的数据框 - 12个蜡烛图重叠，12个有效
    df3 = generate_test_data(timeframe, 24, "2022-01-02 12:00:00+00:00")

    res = dp._add_external_df("ETH/USDT", df3, last_analyzed, timeframe, CandleType.SPOT)
    assert res[0] is True
    assert isinstance(res[1], int)
    assert res[1] == 0
    df, _ = dp.get_producer_df("ETH/USDT", timeframe, CandleType.SPOT)
    # 新长度 = 48 + 12（因为有12小时偏移）
    assert len(df) == 60
    assert df.iloc[-1]["date"] == df3.iloc[-1]["date"]
    assert df.iloc[-1]["date"] == Timestamp("2022-01-03 11:00:00+00:00")

    # 生成1个新蜡烛图
    df4 = generate_test_data(timeframe, 1, "2022-01-03 12:00:00+00:00")
    res = dp._add_external_df("ETH/USDT", df4, last_analyzed, timeframe, CandleType.SPOT)
    # assert res[0] is True
    # assert res[1] == 0
    df, _ = dp.get_producer_df("ETH/USDT", timeframe, CandleType.SPOT)
    # 新长度 = 61 + 1
    assert len(df) == 61
    assert df.iloc[-2]["date"] == Timestamp("2022-01-03 11:00:00+00:00")
    assert df.iloc[-1]["date"] == Timestamp("2022-01-03 12:00:00+00:00")

    # 数据中存在缺口...
    df4 = generate_test_data(timeframe, 1, "2022-01-05 00:00:00+00:00")
    res = dp._add_external_df("ETH/USDT", df4, last_analyzed, timeframe, CandleType.SPOT)
    assert res[0] is False
    # 36小时 - 从2022-01-03 12:00:00+00:00到2022-01-05 00:00:00+00:00
    assert isinstance(res[1], int)
    assert res[1] == 36
    df, _ = dp.get_producer_df("ETH/USDT", timeframe, CandleType.SPOT)
    # 新长度 = 61 + 1
    assert len(df) == 61

    # 空数据框
    df4 = generate_test_data(timeframe, 0, "2022-01-05 00:00:00+00:00")
    res = dp._add_external_df("ETH/USDT", df4, last_analyzed, timeframe, CandleType.SPOT)
    assert res[0] is False
    # 36小时 - 从2022-01-03 12:00:00+00:00到2022-01-05 00:00:00+00:00
    assert isinstance(res[1], int)
    assert res[1] == 0


def test_dp_get_required_startup(default_conf_usdt):
    """测试获取所需的启动蜡烛图数量"""
    timeframe = "1h"
    default_conf_usdt["timeframe"] = timeframe
    dp = DataProvider(default_conf_usdt, None)

    # 没有FreqAI配置
    assert dp.get_required_startup("5m") == 0
    assert dp.get_required_startup("1h") == 0
    assert dp.get_required_startup("1d") == 0

    dp._config["startup_candle_count"] = 20
    assert dp.get_required_startup("5m") == 20
    assert dp.get_required_startup("1h") == 20
    assert dp.get_required_startup("1h") == 20

    # 有freqAI配置
    dp._config["freqai"] = {
        "enabled": True,
        "train_period_days": 20,
        "feature_parameters": {
            "indicator_periods_candles": [
                5,
                20,
            ]
        },
    }
    assert dp.get_required_startup("5m") == 5780
    assert dp.get_required_startup("1h") == 500
    assert dp.get_required_startup("1d") == 40

    # 如果startup_candle_count小于indicator_periods_candles，FreqAI会忽略它
    dp._config["startup_candle_count"] = 0
    assert dp.get_required_startup("5m") == 5780
    assert dp.get_required_startup("1h") == 500
    assert dp.get_required_startup("1d") == 40

    dp._config["freqai"]["feature_parameters"]["indicator_periods_candles"][1] = 50
    assert dp.get_required_startup("5m") == 5810
    assert dp.get_required_startup("1h") == 530
    assert dp.get_required_startup("1d") == 70

    # 来自issue https://github.com/freqtrade/freqtrade/issues/9432的场景
    dp._config["freqai"] = {
        "enabled": True,
        "train_period_days": 180,
        "feature_parameters": {
            "indicator_periods_candles": [
                10,
                20,
            ]
        },
    }
    dp._config["startup_candle_count"] = 40
    assert dp.get_required_startup("5m") == 51880
    assert dp.get_required_startup("1h") == 4360
    assert dp.get_required_startup("1d") == 220
