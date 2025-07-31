# pragma pylint: disable=missing-docstring, protected-access, C0103

import re
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pandas import DataFrame, Timestamp
from pandas.testing import assert_frame_equal

from freqtrade.configuration import TimeRange
from freqtrade.constants import AVAILABLE_DATAHANDLERS
from freqtrade.data.history.datahandlers.featherdatahandler import FeatherDataHandler
from freqtrade.data.history.datahandlers.idatahandler import (
    IDataHandler,
    get_datahandler,
    get_datahandlerclass,
)
from freqtrade.data.history.datahandlers.jsondatahandler import JsonDataHandler, JsonGzDataHandler
from freqtrade.data.history.datahandlers.parquetdatahandler import ParquetDataHandler
from freqtrade.enums import CandleType, TradingMode
from freqtrade.exceptions import OperationalException
from tests.conftest import log_has, log_has_re


def test_datahandler_ohlcv_get_pairs(testdatadir):
    # 测试获取OHLCV数据中的交易对
    pairs = FeatherDataHandler.ohlcv_get_pairs(testdatadir, "5m", candle_type=CandleType.SPOT)
    # 转换为集合以避免排序导致的失败
    assert set(pairs) == {
        "UNITTEST/BTC",
        "XLM/BTC",
        "ETH/BTC",
        "TRX/BTC",
        "LTC/BTC",
        "XMR/BTC",
        "ZEC/BTC",
        "ADA/BTC",
        "ETC/BTC",
        "NXT/BTC",
        "DASH/BTC",
        "XRP/ETH",
    }

    pairs = JsonGzDataHandler.ohlcv_get_pairs(testdatadir, "8m", candle_type=CandleType.SPOT)
    assert set(pairs) == {"UNITTEST/BTC"}

    pairs = FeatherDataHandler.ohlcv_get_pairs(testdatadir, "1h", candle_type=CandleType.MARK)
    assert set(pairs) == {"UNITTEST/USDT:USDT", "XRP/USDT:USDT"}

    pairs = JsonGzDataHandler.ohlcv_get_pairs(testdatadir, "1h", candle_type=CandleType.FUTURES)
    assert set(pairs) == {"XRP/USDT:USDT"}


@pytest.mark.parametrize(
    "filename,pair,timeframe,candletype",
    [
        ("XMR_BTC-5m.json", "XMR_BTC", "5m", ""),
        ("XMR_USDT-1h.h5", "XMR_USDT", "1h", ""),
        ("BTC-PERP-1h.h5", "BTC-PERP", "1h", ""),
        ("BTC_USDT-2h.jsongz", "BTC_USDT", "2h", ""),
        ("BTC_USDT-2h-mark.jsongz", "BTC_USDT", "2h", "mark"),
        ("XMR_USDT-1h-mark.h5", "XMR_USDT", "1h", "mark"),
        ("XMR_USDT-1h-random.h5", "XMR_USDT", "1h", "random"),
        ("BTC-PERP-1h-index.h5", "BTC-PERP", "1h", "index"),
        ("XMR_USDT_USDT-1h-mark.h5", "XMR_USDT_USDT", "1h", "mark"),
    ],
)
def test_datahandler_ohlcv_regex(filename, pair, timeframe, candletype):
    # 测试OHLCV数据文件的正则表达式匹配
    regex = JsonDataHandler._OHLCV_REGEX

    match = re.search(regex, filename)
    assert len(match.groups()) > 1
    assert match[1] == pair
    assert match[2] == timeframe
    assert match[3] == candletype


@pytest.mark.parametrize(
    "pair,expected",
    [
        ("XMR_USDT", "XMR/USDT"),
        ("BTC_USDT", "BTC/USDT"),
        ("USDT_BUSD", "USDT/BUSD"),
        ("BTC_USDT_USDT", "BTC/USDT:USDT"),  # 期货
        ("XRP_USDT_USDT", "XRP/USDT:USDT"),  # 期货
        ("BTC-PERP", "BTC-PERP"),
        ("BTC-PERP_USDT", "BTC-PERP:USDT"),
        ("UNITTEST_USDT", "UNITTEST/USDT"),
    ],
)
def test_rebuild_pair_from_filename(pair, expected):
    # 测试从文件名重建交易对名称
    assert IDataHandler.rebuild_pair_from_filename(pair) == expected


def test_datahandler_ohlcv_get_available_data(testdatadir):
    # 测试获取可用的OHLCV数据
    paircombs = FeatherDataHandler.ohlcv_get_available_data(testdatadir, TradingMode.SPOT)
    # 转换为集合以避免排序导致的失败
    assert set(paircombs) == {
        ("UNITTEST/BTC", "5m", CandleType.SPOT),
        ("ETH/BTC", "5m", CandleType.SPOT),
        ("XLM/BTC", "5m", CandleType.SPOT),
        ("TRX/BTC", "5m", CandleType.SPOT),
        ("LTC/BTC", "5m", CandleType.SPOT),
        ("XMR/BTC", "5m", CandleType.SPOT),
        ("ZEC/BTC", "5m", CandleType.SPOT),
        ("UNITTEST/BTC", "1m", CandleType.SPOT),
        ("ADA/BTC", "5m", CandleType.SPOT),
        ("ETC/BTC", "5m", CandleType.SPOT),
        ("NXT/BTC", "5m", CandleType.SPOT),
        ("DASH/BTC", "5m", CandleType.SPOT),
        ("XRP/ETH", "1m", CandleType.SPOT),
        ("XRP/ETH", "5m", CandleType.SPOT),
        ("UNITTEST/BTC", "30m", CandleType.SPOT),
        ("UNITTEST/BTC", "8m", CandleType.SPOT),
    }

    paircombs = FeatherDataHandler.ohlcv_get_available_data(testdatadir, TradingMode.FUTURES)
    # 转换为集合以避免排序导致的失败
    assert set(paircombs) == {
        ("UNITTEST/USDT:USDT", "1h", "mark"),
        ("XRP/USDT:USDT", "5m", "futures"),
        ("XRP/USDT:USDT", "1h", "futures"),
        ("XRP/USDT:USDT", "1h", "mark"),
        ("XRP/USDT:USDT", "8h", "mark"),
        ("XRP/USDT:USDT", "8h", "funding_rate"),
    }

    paircombs = JsonGzDataHandler.ohlcv_get_available_data(testdatadir, TradingMode.SPOT)
    assert set(paircombs) == {("UNITTEST/BTC", "8m", CandleType.SPOT)}


def test_jsondatahandler_ohlcv_purge(mocker, testdatadir):
    # 测试JSON数据处理器的OHLCV数据删除功能
    mocker.patch.object(Path, "exists", MagicMock(return_value=False))
    unlinkmock = mocker.patch.object(Path, "unlink", MagicMock())
    dh = JsonGzDataHandler(testdatadir)
    assert not dh.ohlcv_purge("UNITTEST/NONEXIST", "5m", "")
    assert not dh.ohlcv_purge("UNITTEST/NONEXIST", "5m", candle_type="mark")
    assert unlinkmock.call_count == 0

    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    assert dh.ohlcv_purge("UNITTEST/NONEXIST", "5m", "")
    assert dh.ohlcv_purge("UNITTEST/NONEXIST", "5m", candle_type="mark")
    assert unlinkmock.call_count == 2


def test_jsondatahandler_ohlcv_load(testdatadir, caplog):
    # 测试JSON数据处理器的OHLCV数据加载功能
    dh = JsonDataHandler(testdatadir)

    df = dh.ohlcv_load("UNITTEST/BTC", "1m", "spot")
    assert len(df) > 0

    # 测试失败情况（空数组）
    df1 = dh.ohlcv_load("NOPAIR/XXX", "4m", "spot")
    assert len(df1) == 0
    assert log_has("无法加载NOPAIR/XXX的数据。", caplog)
    assert df.columns.equals(df1.columns)


def test_datahandler_ohlcv_data_min_max(testdatadir):
    # 测试获取OHLCV数据的时间范围
    dh = JsonDataHandler(testdatadir)
    min_max = dh.ohlcv_data_min_max("UNITTEST/BTC", "5m", "spot")
    assert len(min_max) == 3

    # 测试不存在的交易对
    min_max = dh.ohlcv_data_min_max("UNITTEST/BTC", "8m", "spot")
    assert len(min_max) == 3
    assert min_max[0] == datetime.fromtimestamp(0, tz=timezone.utc)
    assert min_max[0] == min_max[1]
    # 测试另一个不存在的交易对
    min_max = dh.ohlcv_data_min_max("NOPAIR/XXX", "41m", "spot")
    assert len(min_max) == 3
    assert min_max[0] == datetime.fromtimestamp(0, tz=timezone.utc)
    assert min_max[0] == min_max[1]

    # 测试存在的交易对
    min_max = dh.ohlcv_data_min_max("UNITTEST/BTC", "1m", "spot")
    assert len(min_max) == 3
    assert min_max[0] == datetime(2017, 11, 4, 23, 2, tzinfo=timezone.utc)
    assert min_max[1] == datetime(2017, 11, 14, 22, 59, tzinfo=timezone.utc)


def test_datahandler__check_empty_df(testdatadir, caplog):
    # 测试检查空数据框的功能
    dh = JsonDataHandler(testdatadir)
    expected_text = r"UNITTEST/USDT，1h，spot之间的价格跳变"
    df = DataFrame(
        [
            [
                1511686200000,  # 8:50:00
                8.794,  # 开盘价
                8.948,  # 最高价
                8.794,  # 最低价
                8.88,  # 收盘价
                2255,  # 成交量（以计价货币计）
            ],
            [
                1511686500000,  # 8:55:00
                8.88,
                8.942,
                8.88,
                8.893,
                9911,
            ],
            [
                1511687100000,  # 9:05:00
                8.891,
                8.893,
                8.875,
                8.877,
                2251,
            ],
            [
                1511687400000,  # 9:10:00
                8.877,
                8.883,
                8.895,
                8.817,
                123551,
            ],
        ],
        columns=["date", "open", "high", "low", "close", "volume"],
    )

    dh._check_empty_df(df, "UNITTEST/USDT", "1h", CandleType.SPOT, True, True)
    assert not log_has_re(expected_text, caplog)
    df = DataFrame(
        [
            [
                1511686200000,  # 8:50:00
                8.794,  # 开盘价
                8.948,  # 最高价
                8.794,  # 最低价
                8.88,  # 收盘价
                2255,  # 成交量（以计价货币计）
            ],
            [
                1511686500000,  # 8:55:00
                8.88,
                8.942,
                8.88,
                8.893,
                9911,
            ],
            [
                1511687100000,  # 9:05:00
                889.1,  # 价格跳变了几个数量级
                889.3,
                887.5,
                887.7,
                2251,
            ],
            [
                1511687400000,  # 9:10:00
                8.877,
                8.883,
                8.895,
                8.817,
                123551,
            ],
        ],
        columns=["date", "open", "high", "low", "close", "volume"],
    )

    dh._check_empty_df(df, "UNITTEST/USDT", "1h", CandleType.SPOT, True, True)
    assert log_has_re(expected_text, caplog)


# @pytest.mark.parametrize('datahandler', [])
@pytest.mark.skip("目前所有数据处理器都支持交易数据。")
def test_datahandler_trades_not_supported(
    datahandler,
    testdatadir,
):
    # 目前禁用。如果有新的处理器不支持交易数据，应该重新启用。
    dh = get_datahandler(testdatadir, datahandler)
    with pytest.raises(NotImplementedError):
        dh.trades_load("UNITTEST/ETH")
    with pytest.raises(NotImplementedError):
        dh.trades_store("UNITTEST/ETH", MagicMock())


def test_jsondatahandler_trades_load(testdatadir, caplog):
    # 测试JSON数据处理器的交易数据加载功能
    dh = JsonGzDataHandler(testdatadir)
    logmsg = "检测到旧的交易数据格式 - 正在转换"
    dh.trades_load("XRP/ETH", TradingMode.SPOT)
    assert not log_has(logmsg, caplog)

    # 测试转换是否发生
    dh.trades_load("XRP/OLD", TradingMode.SPOT)
    assert log_has(logmsg, caplog)


@pytest.mark.parametrize("datahandler", AVAILABLE_DATAHANDLERS)
def test_datahandler_ohlcv_append(
    datahandler,
    testdatadir,
):
    # 测试OHLCV数据追加功能
    dh = get_datahandler(testdatadir, datahandler)
    with pytest.raises(NotImplementedError):
        dh.ohlcv_append("UNITTEST/ETH", "5m", DataFrame(), CandleType.SPOT)
    with pytest.raises(NotImplementedError):
        dh.ohlcv_append("UNITTEST/ETH", "5m", DataFrame(), CandleType.MARK)


@pytest.mark.parametrize("datahandler", AVAILABLE_DATAHANDLERS)
def test_datahandler_trades_append(datahandler, testdatadir):
    # 测试交易数据追加功能
    dh = get_datahandler(testdatadir, datahandler)
    with pytest.raises(NotImplementedError):
        dh.trades_append("UNITTEST/ETH", DataFrame())


@pytest.mark.parametrize(
    "datahandler,expected",
    [
        ("jsongz", {"XRP/ETH", "XRP/OLD"}),
        ("feather", {"XRP/ETH"}),
        ("parquet", {"XRP/ETH"}),
    ],
)
def test_datahandler_trades_get_pairs(testdatadir, datahandler, expected):
    # 测试获取交易数据中的交易对
    pairs = get_datahandlerclass(datahandler).trades_get_pairs(testdatadir)
    # 转换为集合以避免排序导致的失败
    assert set(pairs) == expected


def test_hdf5datahandler_deprecated(testdatadir):
    # 测试HDF5数据处理器已被弃用
    with pytest.raises(
        OperationalException, match=r"已弃用：hdf5数据格式已被弃用，并有.*"
    ):
        get_datahandler(testdatadir, "hdf5")


@pytest.mark.parametrize(
    "pair,timeframe,candle_type,candle_append,startdt,enddt",
    [
        # 数据范围：2018-01-10 至 2018-01-30
        ("UNITTEST/BTC", "5m", "spot", "", "2018-01-15", "2018-01-19"),
        # 标记数据范围：2021-11-15 至 2021-11-19
        ("UNITTEST/USDT:USDT", "1h", "mark", "-mark", "2021-11-16", "2021-11-18"),
    ],
)
@pytest.mark.parametrize("datahandler", ["feather", "parquet"])
def test_generic_datahandler_ohlcv_load_and_resave(
    datahandler,
    mocker,
    testdatadir,
    tmp_path,
    pair,
    timeframe,
    candle_type,
    candle_append,
    startdt,
    enddt,
    caplog,
):
    # 测试通用数据处理器的OHLCV数据加载和重新保存功能
    tmpdir2 = tmp_path
    if candle_type not in ("", "spot"):
        tmpdir2 = tmp_path / "futures"
        tmpdir2.mkdir()
    # 从一个通用文件加载数据
    dhbase = get_datahandler(testdatadir, "feather")
    ohlcv = dhbase._ohlcv_load(pair, timeframe, None, candle_type=candle_type)
    assert isinstance(ohlcv, DataFrame)
    assert len(ohlcv) > 0

    # 获取要测试的数据处理器
    dh = get_datahandler(testdatadir, datahandler)

    file = tmpdir2 / f"UNITTEST_NEW-{timeframe}{candle_append}.{dh._get_file_extension()}"
    assert not file.is_file()

    dh1 = get_datahandler(tmp_path, datahandler)
    dh1.ohlcv_store("UNITTEST/NEW", timeframe, ohlcv, candle_type=candle_type)
    assert file.is_file()

    assert not ohlcv[ohlcv["date"] < startdt].empty

    timerange = TimeRange.parse_timerange(f"{startdt.replace('-', '')}-{enddt.replace('-', '')}")

    ohlcv = dhbase.ohlcv_load(pair, timeframe, timerange=timerange, candle_type=candle_type)
    ohlcv1 = dh1.ohlcv_load("UNITTEST/NEW", timeframe, timerange=timerange, candle_type=candle_type)

    assert len(ohlcv) == len(ohlcv1)
    assert ohlcv.equals(ohlcv1)
    assert ohlcv[ohlcv["date"] < startdt].empty
    assert ohlcv[ohlcv["date"] > enddt].empty

    # 尝试加载不存在的文件
    ohlcv = dh1.ohlcv_load("UNITTEST/NONEXIST", timeframe, candle_type=candle_type)
    assert ohlcv.empty

    # 尝试加载存在但有错误的文件
    mocker.patch(
        "freqtrade.data.history.datahandlers.featherdatahandler.read_feather",
        side_effect=Exception("测试"),
    )
    mocker.patch(
        "freqtrade.data.history.datahandlers.parquetdatahandler.read_parquet",
        side_effect=Exception("测试"),
    )
    ohlcv_e = dh1.ohlcv_load("UNITTEST/NEW", timeframe, candle_type=candle_type)
    assert ohlcv_e.empty
    assert log_has_re("从以下位置加载数据时出错", caplog)


@pytest.mark.parametrize("datahandler", ["jsongz", "feather", "parquet"])
def test_datahandler_trades_load(testdatadir, datahandler):
    # 测试交易数据加载功能
    dh = get_datahandler(testdatadir, datahandler)
    trades = dh.trades_load("XRP/ETH", TradingMode.SPOT)
    assert isinstance(trades, DataFrame)
    assert trades.iloc[0]["timestamp"] == 1570752011620
    assert trades.iloc[0]["date"] == Timestamp("2019-10-11 00:00:11.620000+0000")
    assert trades.iloc[-1]["cost"] == 0.1986231

    trades1 = dh.trades_load("UNITTEST/NONEXIST", TradingMode.SPOT)
    assert isinstance(trades, DataFrame)
    assert trades1.empty


@pytest.mark.parametrize("datahandler", ["jsongz", "feather", "parquet"])
def test_datahandler_trades_store(testdatadir, tmp_path, datahandler):
    # 测试交易数据存储功能
    dh = get_datahandler(testdatadir, datahandler)
    trades = dh.trades_load("XRP/ETH", TradingMode.SPOT)

    dh1 = get_datahandler(tmp_path, datahandler)
    dh1.trades_store("XRP/NEW", trades, TradingMode.SPOT)

    file = tmp_path / f"XRP_NEW-trades.{dh1._get_file_extension()}"
    assert file.is_file()
    # 重新加载交易数据
    trades_new = dh1.trades_load("XRP/NEW", TradingMode.SPOT)
    assert_frame_equal(trades, trades_new, check_exact=True)
    assert len(trades_new) == len(trades)


@pytest.mark.parametrize("datahandler", ["jsongz", "feather", "parquet"])
def test_datahandler_trades_purge(mocker, testdatadir, datahandler):
    # 测试交易数据删除功能
    mocker.patch.object(Path, "exists", MagicMock(return_value=False))
    unlinkmock = mocker.patch.object(Path, "unlink", MagicMock())
    dh = get_datahandler(testdatadir, datahandler)
    assert not dh.trades_purge("UNITTEST/NONEXIST", TradingMode.SPOT)
    assert unlinkmock.call_count == 0

    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    assert dh.trades_purge("UNITTEST/NONEXIST", TradingMode.SPOT)
    assert unlinkmock.call_count == 1


def test_datahandler_trades_get_available_data(testdatadir):
    # 测试获取可用的交易数据
    paircombs = FeatherDataHandler.trades_get_available_data(testdatadir, TradingMode.SPOT)
    # 转换为集合以避免排序导致的失败
    assert set(paircombs) == {"XRP/ETH"}

    paircombs = FeatherDataHandler.trades_get_available_data(testdatadir, TradingMode.FUTURES)
    # 转换为集合以避免排序导致的失败
    assert set(paircombs) == set()

    paircombs = JsonGzDataHandler.trades_get_available_data(testdatadir, TradingMode.SPOT)
    assert set(paircombs) == {"XRP/ETH", "XRP/OLD"}


def test_datahandler_trades_data_min_max(testdatadir):
    # 测试获取交易数据的时间范围
    dh = FeatherDataHandler(testdatadir)
    min_max = dh.trades_data_min_max("XRP/ETH", TradingMode.SPOT)
    assert len(min_max) == 3

    # 测试不存在的交易对
    min_max = dh.trades_data_min_max("NADA/ETH", TradingMode.SPOT)
    assert len(min_max) == 3
    assert min_max[0] == datetime.fromtimestamp(0, tz=timezone.utc)
    assert min_max[0] == min_max[1]

    # 测试存在的交易对
    min_max = dh.trades_data_min_max("XRP/ETH", TradingMode.SPOT)
    assert len(min_max) == 3
    assert min_max[0] == datetime(2019, 10, 11, 0, 0, 11, 620000, tzinfo=timezone.utc)
    assert min_max[1] == datetime(2019, 10, 13, 11, 19, 28, 844000, tzinfo=timezone.utc)


def test_gethandlerclass():
    # 测试获取数据处理器类
    cl = get_datahandlerclass("json")
    assert cl == JsonDataHandler
    assert issubclass(cl, IDataHandler)

    cl = get_datahandlerclass("jsongz")
    assert cl == JsonGzDataHandler
    assert issubclass(cl, IDataHandler)
    assert issubclass(cl, JsonDataHandler)

    cl = get_datahandlerclass("feather")
    assert cl == FeatherDataHandler
    assert issubclass(cl, IDataHandler)

    cl = get_datahandlerclass("parquet")
    assert cl == ParquetDataHandler
    assert issubclass(cl, IDataHandler)

    with pytest.raises(ValueError, match=r"没有适用于.*的数据处理器"):
        get_datahandlerclass("DeadBeef")


def test_get_datahandler(testdatadir):
    # 测试获取数据处理器实例
    dh = get_datahandler(testdatadir, "json")
    assert isinstance(dh, JsonDataHandler)
    dh = get_datahandler(testdatadir, "jsongz")
    assert isinstance(dh, JsonGzDataHandler)
    dh1 = get_datahandler(testdatadir, "jsongz", dh)
    assert id(dh1) == id(dh)