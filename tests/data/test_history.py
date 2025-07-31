# pragma pylint: disable=missing-docstring, protected-access, C0103

import json
import logging
import uuid
from datetime import timedelta
from pathlib import Path
from shutil import copyfile
from unittest.mock import MagicMock, PropertyMock

import pytest
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from freqtrade.configuration import TimeRange
from freqtrade.constants import DATETIME_PRINT_FORMAT
from freqtrade.data.converter import ohlcv_to_dataframe
from freqtrade.data.history import get_datahandler
from freqtrade.data.history.datahandlers.jsondatahandler import JsonDataHandler, JsonGzDataHandler
from freqtrade.data.history.history_utils import (
    _download_pair_history,
    _download_trades_history,
    _load_cached_data_for_updating,
    get_timerange,
    load_data,
    load_pair_history,
    refresh_backtest_ohlcv_data,
    refresh_backtest_trades_data,
    refresh_data,
    validate_backtest_data,
)
from freqtrade.enums import CandleType, TradingMode
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.misc import file_dump_json
from freqtrade.resolvers import StrategyResolver
from freqtrade.util import dt_ts, dt_utc
from tests.conftest import (
    CURRENT_TEST_STRATEGY,
    EXMS,
    get_patched_exchange,
    log_has,
    log_has_re,
    patch_exchange,
)


def _clean_test_file(file: Path) -> None:
    """
    备份现有文件以避免删除用户文件
    :param file: 文件的完整路径
    :return: None
    """
    file_swp = Path(str(file) + ".swp")
    # 1. 删除测试生成的文件
    if file.is_file():
        file.unlink()

    # 2. 恢复初始文件
    if file_swp.is_file():
        file_swp.rename(file)


def test_load_data_30min_timeframe(caplog, testdatadir) -> None:
    ld = load_pair_history(pair="UNITTEST/BTC", timeframe="30m", datadir=testdatadir)
    assert isinstance(ld, DataFrame)
    assert not log_has(
        '下载交易对历史数据: "UNITTEST/BTC", 时间框架: 30m 并存储在 None。',
        caplog,
    )


def test_load_data_7min_timeframe(caplog, testdatadir) -> None:
    ld = load_pair_history(pair="UNITTEST/BTC", timeframe="7m", datadir=testdatadir)
    assert isinstance(ld, DataFrame)
    assert ld.empty
    assert log_has(
        "未找到 UNITTEST/BTC 的现货 7m 历史数据。"
        "使用 `freqtrade download-data` 下载数据",
        caplog,
    )


def test_load_data_1min_timeframe(ohlcv_history, mocker, caplog, testdatadir) -> None:
    mocker.patch(f"{EXMS}.get_historic_ohlcv", return_value=ohlcv_history)
    file = testdatadir / "UNITTEST_BTC-1m.feather"
    load_data(datadir=testdatadir, timeframe="1m", pairs=["UNITTEST/BTC"])
    assert file.is_file()
    assert not log_has(
        '下载交易对历史数据: "UNITTEST/BTC", 时间间隔: 1m 并存储在 None。', caplog
    )


def test_load_data_mark(ohlcv_history, mocker, caplog, testdatadir) -> None:
    mocker.patch(f"{EXMS}.get_historic_ohlcv", return_value=ohlcv_history)
    file = testdatadir / "futures/UNITTEST_USDT_USDT-1h-mark.feather"
    load_data(datadir=testdatadir, timeframe="1h", pairs=["UNITTEST/BTC"], candle_type="mark")
    assert file.is_file()
    assert not log_has(
        '下载交易对历史数据: "UNITTEST/USDT:USDT", 时间间隔: 1m 并存储在 None。',
        caplog,
    )


def test_load_data_startup_candles(mocker, testdatadir) -> None:
    ltfmock = mocker.patch(
        "freqtrade.data.history.datahandlers.featherdatahandler.FeatherDataHandler._ohlcv_load",
        MagicMock(return_value=DataFrame()),
    )
    timerange = TimeRange("date", None, 1510639620, 0)
    load_pair_history(
        pair="UNITTEST/BTC",
        timeframe="1m",
        datadir=testdatadir,
        timerange=timerange,
        startup_candles=20,
    )

    assert ltfmock.call_count == 1
    assert ltfmock.call_args_list[0][1]["timerange"] != timerange
    # 开始时间提前20分钟
    assert ltfmock.call_args_list[0][1]["timerange"].startts == timerange.startts - 20 * 60


@pytest.mark.parametrize("candle_type", ["mark", ""])
def test_load_data_with_new_pair_1min(
    ohlcv_history, mocker, caplog, default_conf, tmp_path, candle_type
) -> None:
    """
    测试使用1分钟时间框架的load_pair_history()
    """
    exchange = get_patched_exchange(mocker, default_conf)
    mocker.patch.object(exchange, "get_historic_ohlcv", return_value=ohlcv_history)
    file = tmp_path / "MEME_BTC-1m.feather"

    # 如果未设置refresh_pairs，则不下载新交易对
    load_pair_history(datadir=tmp_path, timeframe="1m", pair="MEME/BTC", candle_type=candle_type)
    assert not file.is_file()
    assert log_has(
        f"未找到 MEME/BTC 的 {candle_type} 1m 历史数据。"
        "使用 `freqtrade download-data` 下载数据",
        caplog,
    )

    # 如果设置refresh_pairs，则下载新交易对
    refresh_data(
        datadir=tmp_path,
        timeframe="1m",
        pairs=["MEME/BTC"],
        exchange=exchange,
        candle_type=CandleType.SPOT,
    )
    load_pair_history(datadir=tmp_path, timeframe="1m", pair="MEME/BTC", candle_type=candle_type)
    assert file.is_file()
    assert log_has_re(r'下载 "MEME/BTC" 的 1m 现货历史数据并存储在 .*', caplog)


def test_testdata_path(testdatadir) -> None:
    assert str(Path("tests") / "testdata") in str(testdatadir)


@pytest.mark.parametrize(
    "pair,timeframe,expected_result,candle_type",
    [
        ("ETH/BTC", "5m", "freqtrade/hello/world/ETH_BTC-5m.json", ""),
        ("ETH/USDT", "1M", "freqtrade/hello/world/ETH_USDT-1Mo.json", ""),
        ("Fabric Token/ETH", "5m", "freqtrade/hello/world/Fabric_Token_ETH-5m.json", ""),
        ("ETHH20", "5m", "freqtrade/hello/world/ETHH20-5m.json", ""),
        (".XBTBON2H", "5m", "freqtrade/hello/world/_XBTBON2H-5m.json", ""),
        ("ETHUSD.d", "5m", "freqtrade/hello/world/ETHUSD_d-5m.json", ""),
        ("ACC_OLD/BTC", "5m", "freqtrade/hello/world/ACC_OLD_BTC-5m.json", ""),
        ("ETH/BTC", "5m", "freqtrade/hello/world/futures/ETH_BTC-5m-mark.json", "mark"),
        ("ACC_OLD/BTC", "5m", "freqtrade/hello/world/futures/ACC_OLD_BTC-5m-index.json", "index"),
    ],
)
def test_json_pair_data_filename(pair, timeframe, expected_result, candle_type):
    fn = JsonDataHandler._pair_data_filename(
        Path("freqtrade/hello/world"), pair, timeframe, CandleType.from_string(candle_type)
    )
    assert isinstance(fn, Path)
    assert fn == Path(expected_result)
    fn = JsonGzDataHandler._pair_data_filename(
        Path("freqtrade/hello/world"),
        pair,
        timeframe,
        candle_type=CandleType.from_string(candle_type),
    )
    assert isinstance(fn, Path)
    assert fn == Path(expected_result + ".gz")


@pytest.mark.parametrize(
    "pair,trading_mode,expected_result",
    [
        ("ETH/BTC", "", "freqtrade/hello/world/ETH_BTC-trades.json"),
        ("ETH/USDT:USDT", "futures", "freqtrade/hello/world/futures/ETH_USDT_USDT-trades.json"),
        ("Fabric Token/ETH", "", "freqtrade/hello/world/Fabric_Token_ETH-trades.json"),
        ("ETHH20", "", "freqtrade/hello/world/ETHH20-trades.json"),
        (".XBTBON2H", "", "freqtrade/hello/world/_XBTBON2H-trades.json"),
        ("ETHUSD.d", "", "freqtrade/hello/world/ETHUSD_d-trades.json"),
        ("ACC_OLD_BTC", "", "freqtrade/hello/world/ACC_OLD_BTC-trades.json"),
    ],
)
def test_json_pair_trades_filename(pair, trading_mode, expected_result):
    fn = JsonDataHandler._pair_trades_filename(Path("freqtrade/hello/world"), pair, trading_mode)
    assert isinstance(fn, Path)
    assert fn == Path(expected_result)

    fn = JsonGzDataHandler._pair_trades_filename(Path("freqtrade/hello/world"), pair, trading_mode)
    assert isinstance(fn, Path)
    assert fn == Path(expected_result + ".gz")


def test_load_cached_data_for_updating(testdatadir) -> None:
    data_handler = get_datahandler(testdatadir, "json")

    test_data = None
    test_filename = testdatadir.joinpath("UNITTEST_BTC-1m.json")
    with test_filename.open("rt") as file:
        test_data = json.load(file)

    test_data_df = ohlcv_to_dataframe(
        test_data, "1m", "UNITTEST/BTC", fill_missing=False, drop_incomplete=False
    )
    # 当前时间 = 最后缓存项 + 1小时
    now_ts = test_data[-1][0] / 1000 + 60 * 60

    # 时间范围早于缓存数据
    # 更新时间戳为蜡烛结束日期
    timerange = TimeRange("date", None, test_data[0][0] / 1000 - 1, 0)
    data, start_ts, end_ts = _load_cached_data_for_updating(
        "UNITTEST/BTC", "1m", timerange, data_handler, CandleType.SPOT
    )
    assert not data.empty
    # 最后一根蜡烛被移除 - 因此有1根蜡烛重叠
    assert start_ts == test_data[-1][0] - 60 * 1000
    assert end_ts is None

    # 时间范围早于缓存数据 - 前置数据

    timerange = TimeRange("date", None, test_data[0][0] / 1000 - 1, 0)
    data, start_ts, end_ts = _load_cached_data_for_updating(
        "UNITTEST/BTC", "1m", timerange, data_handler, CandleType.SPOT, True
    )
    assert_frame_equal(data, test_data_df.iloc[:-1])
    assert start_ts == test_data[0][0] - 1000
    assert end_ts == test_data[0][0]

    # 时间范围开始于缓存数据的中间
    # 应返回没有最后一项的缓存数据
    timerange = TimeRange("date", None, test_data[0][0] / 1000 + 1, 0)
    data, start_ts, end_ts = _load_cached_data_for_updating(
        "UNITTEST/BTC", "1m", timerange, data_handler, CandleType.SPOT
    )

    assert_frame_equal(data, test_data_df.iloc[:-1])
    assert test_data[-2][0] <= start_ts < test_data[-1][0]
    assert end_ts is None

    # 时间范围开始于缓存数据之后
    # 应返回没有最后一项的缓存数据
    timerange = TimeRange("date", None, test_data[-1][0] / 1000 + 100, 0)
    data, start_ts, end_ts = _load_cached_data_for_updating(
        "UNITTEST/BTC", "1m", timerange, data_handler, CandleType.SPOT
    )
    assert_frame_equal(data, test_data_df.iloc[:-1])
    assert test_data[-2][0] <= start_ts < test_data[-1][0]
    assert end_ts is None

    # 不存在数据文件
    # 应返回时间戳开始时间
    timerange = TimeRange("date", None, now_ts - 10000, 0)
    data, start_ts, end_ts = _load_cached_data_for_updating(
        "NONEXIST/BTC", "1m", timerange, data_handler, CandleType.SPOT
    )
    assert data.empty
    assert start_ts == (now_ts - 10000) * 1000
    assert end_ts is None

    # 不存在数据文件
    # 应返回时间戳开始和结束时间
    timerange = TimeRange("date", "date", now_ts - 1000000, now_ts - 100000)
    data, start_ts, end_ts = _load_cached_data_for_updating(
        "NONEXIST/BTC", "1m", timerange, data_handler, CandleType.SPOT
    )
    assert data.empty
    assert start_ts == (now_ts - 1000000) * 1000
    assert end_ts == (now_ts - 100000) * 1000

    # 不存在数据文件，未设置时间范围
    # 应返回空数组和None
    data, start_ts, end_ts = _load_cached_data_for_updating(
        "NONEXIST/BTC", "1m", None, data_handler, CandleType.SPOT
    )
    assert data.empty
    assert start_ts is None
    assert end_ts is None


@pytest.mark.parametrize(
    "candle_type,subdir,file_tail",
    [
        ("mark", "futures/", "-mark"),
        ("spot", "", ""),
    ],
)
def test_download_pair_history(
    ohlcv_history, mocker, default_conf, tmp_path, candle_type, subdir, file_tail
) -> None:
    exchange = get_patched_exchange(mocker, default_conf)
    mocker.patch.object(exchange, "get_historic_ohlcv", return_value=ohlcv_history)
    file1_1 = tmp_path / f"{subdir}MEME_BTC-1m{file_tail}.feather"
    file1_5 = tmp_path / f"{subdir}MEME_BTC-5m{file_tail}.feather"
    file2_1 = tmp_path / f"{subdir}CFI_BTC-1m{file_tail}.feather"
    file2_5 = tmp_path / f"{subdir}CFI_BTC-5m{file_tail}.feather"

    assert not file1_1.is_file()
    assert not file2_1.is_file()

    assert _download_pair_history(
        datadir=tmp_path,
        exchange=exchange,
        pair="MEME/BTC",
        timeframe="1m",
        candle_type=candle_type,
    )
    assert _download_pair_history(
        datadir=tmp_path, exchange=exchange, pair="CFI/BTC", timeframe="1m", candle_type=candle_type
    )
    assert not exchange._pairs_last_refresh_time
    assert file1_1.is_file()
    assert file2_1.is_file()

    # 清理新下载的文件
    _clean_test_file(file1_1)
    _clean_test_file(file2_1)

    assert not file1_5.is_file()
    assert not file2_5.is_file()

    assert _download_pair_history(
        datadir=tmp_path,
        exchange=exchange,
        pair="MEME/BTC",
        timeframe="5m",
        candle_type=candle_type,
    )
    assert _download_pair_history(
        datadir=tmp_path, exchange=exchange, pair="CFI/BTC", timeframe="5m", candle_type=candle_type
    )
    assert not exchange._pairs_last_refresh_time
    assert file1_5.is_file()
    assert file2_5.is_file()


def test_download_pair_history2(mocker, default_conf, testdatadir, ohlcv_history) -> None:
    json_dump_mock = mocker.patch(
        "freqtrade.data.history.datahandlers.featherdatahandler.FeatherDataHandler.ohlcv_store",
        return_value=None,
    )
    exchange = get_patched_exchange(mocker, default_conf)
    mocker.patch.object(exchange, "get_historic_ohlcv", return_value=ohlcv_history)
    _download_pair_history(
        datadir=testdatadir,
        exchange=exchange,
        pair="UNITTEST/BTC",
        timeframe="1m",
        candle_type="spot",
    )
    _download_pair_history(
        datadir=testdatadir,
        exchange=exchange,
        pair="UNITTEST/BTC",
        timeframe="3m",
        candle_type="spot",
    )
    _download_pair_history(
        datadir=testdatadir,
        exchange=exchange,
        pair="UNITTEST/USDT",
        timeframe="1h",
        candle_type="mark",
    )
    assert json_dump_mock.call_count == 3


def test_download_backtesting_data_exception(mocker, caplog, default_conf, tmp_path) -> None:
    mocker.patch(f"{EXMS}.get_historic_ohlcv", side_effect=Exception("文件错误"))
    exchange = get_patched_exchange(mocker, default_conf)

    assert not _download_pair_history(
        datadir=tmp_path, exchange=exchange, pair="MEME/BTC", timeframe="1m", candle_type="spot"
    )
    assert log_has('下载交易对历史数据失败: "MEME/BTC", 时间框架: 1m。', caplog)


def test_load_partial_missing(testdatadir, caplog) -> None:
    # 确保我们从新开始 - 测试开始处缺少数据
    start = dt_utc(2018, 1, 1)
    end = dt_utc(2018, 1, 11)
    data = load_data(
        testdatadir,
        "5m",
        ["UNITTEST/BTC"],
        startup_candles=20,
        timerange=TimeRange("date", "date", start.timestamp(), end.timestamp()),
    )
    assert log_has("使用指标启动周期: 20 ...", caplog)
    # 5分钟的时间差
    td = ((end - start).total_seconds() // 60 // 5) + 1
    assert td != len(data["UNITTEST/BTC"])
    start_real = data["UNITTEST/BTC"].iloc[0, 0]
    assert log_has(
        f"UNITTEST/BTC, 现货, 5m, 数据开始于 {start_real.strftime(DATETIME_PRINT_FORMAT)}",
        caplog,
    )
    # 确保我们从新开始 - 测试结束处缺少数据
    caplog.clear()
    start = dt_utc(2018, 1, 10)
    end = dt_utc(2018, 2, 20)
    data = load_data(
        datadir=testdatadir,
        timeframe="5m",
        pairs=["UNITTEST/BTC"],
        timerange=TimeRange("date", "date", start.timestamp(), end.timestamp()),
    )
    # 5分钟的时间差
    td = ((end - start).total_seconds() // 60 // 5) + 1
    assert td != len(data["UNITTEST/BTC"])

    # 结束时间 +5
    end_real = data["UNITTEST/BTC"].iloc[-1, 0].to_pydatetime()
    assert log_has(
        f"UNITTEST/BTC, 现货, 5m, 数据结束于 {end_real.strftime(DATETIME_PRINT_FORMAT)}",
        caplog,
    )


def test_init(default_conf) -> None:
    assert {} == load_data(datadir=Path(), pairs=[], timeframe=default_conf["timeframe"])


def test_init_with_refresh(default_conf, mocker) -> None:
    exchange = get_patched_exchange(mocker, default_conf)
    refresh_data(
        datadir=Path(),
        pairs=[],
        timeframe=default_conf["timeframe"],
        exchange=exchange,
        candle_type=CandleType.SPOT,
    )
    assert {} == load_data(datadir=Path(), pairs=[], timeframe=default_conf["timeframe"])


def test_file_dump_json_tofile(testdatadir) -> None:
    file = testdatadir / f"test_{uuid.uuid4()}.json"
    data = {"bar": "foo"}

    # 检查将要创建的文件不存在
    assert not file.is_file()

    # 创建Json文件
    file_dump_json(file, data)

    # 检查文件已创建
    assert file.is_file()

    # 打开创建的Json文件并测试其中的数据
    with file.open() as data_file:
        json_from_file = json.load(data_file)

    assert "bar" in json_from_file
    assert json_from_file["bar"] == "foo"

    # 删除文件
    _clean_test_file(file)


def test_get_timerange(default_conf, mocker, testdatadir) -> None:
    patch_exchange(mocker)

    default_conf.update({"strategy": CURRENT_TEST_STRATEGY})
    strategy = StrategyResolver.load_strategy(default_conf)

    data = strategy.advise_all_indicators(
        load_data(datadir=testdatadir, timeframe="1m", pairs=["UNITTEST/BTC"])
    )
    min_date, max_date = get_timerange(data)
    assert min_date.isoformat() == "2017-11-04T23:02:00+00:00"
    assert max_date.isoformat() == "2017-11-14T22:59:00+00:00"


def test_validate_backtest_data_warn(default_conf, mocker, caplog, testdatadir) -> None:
    patch_exchange(mocker)

    default_conf.update({"strategy": CURRENT_TEST_STRATEGY})
    strategy = StrategyResolver.load_strategy(default_conf)

    data = strategy.advise_all_indicators(
        load_data(
            datadir=testdatadir, timeframe="1m", pairs=["UNITTEST/BTC"], fill_up_missing=False
        )
    )
    min_date, max_date = get_timerange(data)
    caplog.clear()
    assert validate_backtest_data(
        data["UNITTEST/BTC"], "UNITTEST/BTC", min_date, max_date, timeframe_to_minutes("1m")
    )
    assert len(caplog.record_tuples) == 1
    assert log_has(
        "UNITTEST/BTC 存在缺失的K线: 预期 14397 根, 实际 13681 根, 缺失 716 个值",
        caplog,
    )


def test_validate_backtest_data(default_conf, mocker, caplog, testdatadir) -> None:
    patch_exchange(mocker)

    default_conf.update({"strategy": CURRENT_TEST_STRATEGY})
    strategy = StrategyResolver.load_strategy(default_conf)

    timerange = TimeRange()
    data = strategy.advise_all_indicators(
        load_data(datadir=testdatadir, timeframe="5m", pairs=["UNITTEST/BTC"], timerange=timerange)
    )

    min_date, max_date = get_timerange(data)
    caplog.clear()
    assert not validate_backtest_data(
        data["UNITTEST/BTC"], "UNITTEST/BTC", min_date, max_date, timeframe_to_minutes("5m")
    )
    assert len(caplog.record_tuples) == 0


@pytest.mark.parametrize(
    "trademode,callcount",
    [
        ("spot", 4),
        ("margin", 4),
        ("futures", 8),  # 调用8次 - 4次正常, 2次资金费率和2次标记/指数调用
    ],
)
def test_refresh_backtest_ohlcv_data(
    mocker, default_conf, markets, caplog, testdatadir, trademode, callcount
):
    caplog.set_level(logging.DEBUG)
    dl_mock = mocker.patch("freqtrade.data.history.history_utils._download_pair_history")
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets))

    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    mocker.patch.object(Path, "unlink", MagicMock())
    default_conf["trading_mode"] = trademode

    ex = get_patched_exchange(mocker, default_conf, exchange="bybit")
    timerange = TimeRange.parse_timerange("20190101-20190102")
    refresh_backtest_ohlcv_data(
        exchange=ex,
        pairs=["ETH/BTC", "XRP/BTC"],
        timeframes=["1m", "5m"],
        datadir=testdatadir,
        timerange=timerange,
        erase=True,
        trading_mode=trademode,
    )

    assert dl_mock.call_count == callcount
    assert dl_mock.call_args[1]["timerange"].starttype == "date"

    assert log_has_re(r"下载交易对 ETH/BTC, .* 时间间隔 1m\.", caplog)
    if trademode == "futures":
        assert log_has_re(r"下载交易对 ETH/BTC, funding_rate, 时间间隔 8h\.", caplog)
        assert log_has_re(r"下载交易对 ETH/BTC, mark, 时间间隔 4h\.", caplog)


def test_download_data_no_markets(mocker, default_conf, caplog, testdatadir):
    dl_mock = mocker.patch(
        "freqtrade.data.history.history_utils._download_pair_history", MagicMock()
    )

    ex = get_patched_exchange(mocker, default_conf)
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value={}))
    timerange = TimeRange.parse_timerange("20190101-20190102")
    unav_pairs = refresh_backtest_ohlcv_data(
        exchange=ex,
        pairs=["BTT/BTC", "LTC/USDT"],
        timeframes=["1m", "5m"],
        datadir=testdatadir,
        timerange=timerange,
        erase=False,
        trading_mode="spot",
    )

    assert dl_mock.call_count == 0
    assert "BTT/BTC: 交易对在交易所不可用。" in unav_pairs
    assert "LTC/USDT: 交易对在交易所不可用。" in unav_pairs
    assert log_has("跳过交易对 BTT/BTC...", caplog)


def test_refresh_backtest_trades_data(mocker, default_conf, markets, caplog, testdatadir):
    dl_mock = mocker.patch(
        "freqtrade.data.history.history_utils._download_trades_history", MagicMock()
    )
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets))
    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    mocker.patch.object(Path, "unlink", MagicMock())

    ex = get_patched_exchange(mocker, default_conf)
    timerange = TimeRange.parse_timerange("20190101-20190102")
    unavailable_pairs = refresh_backtest_trades_data(
        exchange=ex,
        pairs=["ETH/BTC", "XRP/BTC", "XRP/ETH"],
        datadir=testdatadir,
        timerange=timerange,
        erase=True,
        trading_mode=TradingMode.SPOT,
    )

    assert dl_mock.call_count == 2
    assert dl_mock.call_args[1]["timerange"].starttype == "date"

    assert log_has("下载交易对 ETH/BTC 的交易数据。", caplog)
    assert [p for p in unavailable_pairs if "XRP/ETH" in p]
    assert log_has("跳过交易对 XRP/ETH...", caplog)


def test_download_trades_history(
    trades_history, mocker, default_conf, testdatadir, caplog, tmp_path, time_machine
) -> None:
    start_dt = dt_utc(2023, 1, 1)
    time_machine.move_to(start_dt, tick=False)

    ght_mock = MagicMock(side_effect=lambda pair, *args, **kwargs: (pair, trades_history))
    mocker.patch(f"{EXMS}.get_historic_trades", ght_mock)
    exchange = get_patched_exchange(mocker, default_conf)
    file1 = tmp_path / "ETH_BTC-trades.json.gz"
    data_handler = get_datahandler(tmp_path, data_format="jsongz")

    assert not file1.is_file()

    assert _download_trades_history(
        data_handler=data_handler, exchange=exchange, pair="ETH/BTC", trading_mode=TradingMode.SPOT
    )
    assert log_has("当前交易数量: 0", caplog)
    assert log_has("新交易数量: 6", caplog)
    assert ght_mock.call_count == 1
    # 默认 "since" - 当前日期前30天。
    assert ght_mock.call_args_list[0][1]["since"] == dt_ts(start_dt - timedelta(days=30))
    assert file1.is_file()
    caplog.clear()

    ght_mock.reset_mock()
    since_time = int(trades_history[-3][0] // 1000)
    since_time2 = int(trades_history[-1][0] // 1000)
    timerange = TimeRange("date", None, since_time, 0)
    assert _download_trades_history(
        data_handler=data_handler,
        exchange=exchange,
        pair="ETH/BTC",
        timerange=timerange,
        trading_mode=TradingMode.SPOT,
    )

    assert ght_mock.call_count == 1
    # 以秒为单位检查 - 因为我们也必须转换为秒
    assert int(ght_mock.call_args_list[0][1]["since"] // 1000) == since_time2 - 5
    assert ght_mock.call_args_list[0][1]["from_id"] is not None

    file1.unlink()

    mocker.patch(f"{EXMS}.get_historic_trades", MagicMock(side_effect=ValueError("出错了!")))
    caplog.clear()

    with pytest.raises(ValueError, match="出错了!"):
        _download_trades_history(
            data_handler=data_handler,
            exchange=exchange,
            pair="ETH/BTC",
            trading_mode=TradingMode.SPOT,
        )

    file2 = tmp_path / "XRP_ETH-trades.json.gz"
    copyfile(testdatadir / file2.name, file2)

    ght_mock.reset_mock()
    mocker.patch(f"{EXMS}.get_historic_trades", ght_mock)
    # 早于第一个开始日期
    since_time = int(trades_history[0][0] // 1000) - 500
    timerange = TimeRange("date", None, since_time, 0)

    with pytest.raises(ValueError, match=r"开始时间 .* 早于可用数据"):
        _download_trades_history(
            data_handler=data_handler,
            exchange=exchange,
            pair="XRP/ETH",
            timerange=timerange,
            trading_mode=TradingMode.SPOT,
        )

    assert ght_mock.call_count == 0

    _clean_test_file(file2)
    