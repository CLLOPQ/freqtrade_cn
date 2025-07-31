from unittest.mock import MagicMock, PropertyMock

import pytest

from freqtrade.configuration.config_setup import setup_utils_configuration
from freqtrade.data.history.history_utils import download_data_main
from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException
from tests.conftest import EXMS, log_has_re, patch_exchange


def test_download_data_main_no_markets(mocker, caplog):
    # 创建下载模拟函数
    dl_mock = mocker.patch(
        "freqtrade.data.history.history_utils.refresh_backtest_ohlcv_data",
        MagicMock(return_value=["ETH/BTC", "XRP/BTC"]),
    )
    # 模拟币安交易所
    patch_exchange(mocker, exchange="binance")
    # 模拟获取市场信息返回空字典
    mocker.patch(f"{EXMS}.get_markets", return_value={})
    # 设置配置
    config = setup_utils_configuration({"exchange": "binance"}, RunMode.UTIL_EXCHANGE)
    config.update({"days": 20, "pairs": ["ETH/BTC", "XRP/BTC"], "timeframes": ["5m", "1h"]})
    # 调用主函数
    download_data_main(config)
    # 断言检查：下载函数未被调用
    assert dl_mock.call_count == 0
    # 断言日志中包含预期信息
    assert log_has_re("没有可下载的交易对..*", caplog)


def test_download_data_main_all_pairs(mocker, markets):
    # 创建下载模拟函数
    dl_mock = mocker.patch(
        "freqtrade.data.history.history_utils.refresh_backtest_ohlcv_data",
        MagicMock(return_value=["ETH/BTC", "XRP/BTC"]),
    )
    # 模拟交易所
    patch_exchange(mocker)
    # 模拟市场信息属性
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets))

    # 设置配置
    config = setup_utils_configuration({"exchange": "binance"}, RunMode.UTIL_EXCHANGE)
    config.update({"pairs": [".*/USDT"], "timeframes": ["5m", "1h"]})
    download_data_main(config)
    # 预期的交易对集合
    expected = set(["BTC/USDT", "ETH/USDT", "XRP/USDT", "NEO/USDT", "TKN/USDT"])
    # 断言调用参数中的交易对与预期一致
    assert set(dl_mock.call_args_list[0][1]["pairs"]) == expected
    # 断言下载函数被调用一次
    assert dl_mock.call_count == 1

    # 重置模拟函数
    dl_mock.reset_mock()

    # 更新配置，包含非活跃交易对
    config.update({"pairs": [".*/USDT"], "timeframes": ["5m", "1h"], "include_inactive": True})
    download_data_main(config)
    # 预期的交易对集合（包含非活跃）
    expected = set(["BTC/USDT", "ETH/USDT", "LTC/USDT", "XRP/USDT", "NEO/USDT", "TKN/USDT"])
    assert set(dl_mock.call_args_list[0][1]["pairs"]) == expected


def test_download_data_main_trades(mocker):
    # 创建交易数据下载模拟函数
    dl_mock = mocker.patch(
        "freqtrade.data.history.history_utils.refresh_backtest_trades_data",
        MagicMock(return_value=[]),
    )
    # 创建交易数据转换模拟函数
    convert_mock = mocker.patch(
        "freqtrade.data.history.history_utils.convert_trades_to_ohlcv", MagicMock(return_value=[])
    )
    # 模拟交易所
    patch_exchange(mocker)
    # 模拟市场信息
    mocker.patch(f"{EXMS}.get_markets", return_value={"ETH/BTC": {}, "XRP/BTC": {}})
    # 设置配置
    config = setup_utils_configuration({"exchange": "binance"}, RunMode.UTIL_EXCHANGE)
    config.update(
        {
            "days": 20,
            "pairs": ["ETH/BTC", "XRP/BTC"],
            "timeframes": ["5m", "1h"],
            "download_trades": True,
        }
    )
    download_data_main(config)

    # 断言时间范围的起始类型是日期
    assert dl_mock.call_args[1]["timerange"].starttype == "date"
    # 断言下载函数被调用一次
    assert dl_mock.call_count == 1
    # 断言转换函数未被调用
    assert convert_mock.call_count == 0
    # 重置模拟函数
    dl_mock.reset_mock()

    # 更新配置，启用交易数据转换
    config.update(
        {
            "convert_trades": True,
        }
    )
    download_data_main(config)

    # 再次检查时间范围类型
    assert dl_mock.call_args[1]["timerange"].starttype == "date"
    assert dl_mock.call_count == 1
    # 断言转换函数被调用一次
    assert convert_mock.call_count == 1

    # 测试不支持历史数据下载的交易所
    config["exchange"]["name"] = "bybit"
    with pytest.raises(OperationalException, match=r".*交易所不支持交易历史下载"):
        download_data_main(config)


def test_download_data_main_data_invalid(mocker):
    # 模拟kraken交易所
    patch_exchange(mocker, exchange="kraken")
    mocker.patch(f"{EXMS}.get_markets", return_value={"ETH/BTC": {}})
    # 设置配置
    config = setup_utils_configuration({"exchange": "kraken"}, RunMode.UTIL_EXCHANGE)
    config.update(
        {
            "days": 20,
            "pairs": ["ETH/BTC", "XRP/BTC"],
            "timeframes": ["5m", "1h"],
        }
    )
    # 断言会抛出不支持K线历史数据的异常
    with pytest.raises(OperationalException, match=r".*不支持K线历史数据"):
        download_data_main(config)

    # 模拟hyperliquid交易所
    patch_exchange(mocker, exchange="hyperliquid")
    mocker.patch(f"{EXMS}.get_markets", return_value={"ETH/USDC": {}})
    config2 = setup_utils_configuration({"exchange": "hyperliquid"}, RunMode.UTIL_EXCHANGE)
    config2.update(
        {
            "days": 20,
            "pairs": ["ETH/USDC", "XRP/USDC"],
            "timeframes": ["5m", "1h"],
        }
    )
    # 断言会抛出不支持历史数据的异常
    with pytest.raises(OperationalException, match=r".*不支持历史数据"):
        download_data_main(config2)