from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from freqtrade.data.converter import ohlcv_to_dataframe
from freqtrade.enums import CandleType
from freqtrade.exceptions import OperationalException
from freqtrade.plugins.pairlist.PercentChangePairList import PercentChangePairList
from freqtrade.plugins.pairlistmanager import PairListManager
from tests.conftest import (
    EXMS,
    generate_test_data_raw,
    get_patched_exchange,
    get_patched_freqtradebot,
)


@pytest.fixture(scope="function")
def rpl_config(default_conf):
    default_conf["stake_currency"] = "USDT"

    default_conf["exchange"]["pair_whitelist"] = [
        "ETH/USDT",
        "XRP/USDT",
    ]
    default_conf["exchange"]["pair_blacklist"] = ["BLK/USDT"]

    return default_conf


def test_volume_change_pair_list_init_exchange_support(mocker, rpl_config):
    rpl_config["pairlists"] = [
        {
            "method": "PercentChangePairList",
            "number_assets": 2,
            "sort_key": "percentage",
            "min_value": 0,
            "refresh_period": 86400,
        }
    ]

    with pytest.raises(
        OperationalException,
        match=r"交易所在此配置下不支持动态白名单。请编辑您的配置，"
        r"要么移除PercentChangePairList，要么切换到使用蜡烛数据并重启机器人。",
    ):
        get_patched_freqtradebot(mocker, rpl_config)


def test_volume_change_pair_list_init_wrong_refresh_period(mocker, rpl_config):
    rpl_config["pairlists"] = [
        {
            "method": "PercentChangePairList",
            "number_assets": 2,
            "sort_key": "percentage",
            "min_value": 0,
            "refresh_period": 1800,
            "lookback_days": 4,
        }
    ]

    with pytest.raises(
        OperationalException,
        match=r"刷新周期1800秒小于一个时间框架1d。请调整refresh_period至少为86400并重启机器人。",
    ):
        get_patched_freqtradebot(mocker, rpl_config)


def test_volume_change_pair_list_init_wrong_lookback_period(mocker, rpl_config):
    rpl_config["pairlists"] = [
        {
            "method": "PercentChangePairList",
            "number_assets": 2,
            "sort_key": "percentage",
            "min_value": 0,
            "refresh_period": 86400,
            "lookback_days": 3,
            "lookback_period": 3,
        }
    ]

    with pytest.raises(
        OperationalException,
        match=r"配置冲突：在交易对列表配置中同时设置了lookback_days和lookback_period。"
        r"请只设置lookback_days，或者设置lookback_period和lookback_timeframe并重启机器人。",
    ):
        get_patched_freqtradebot(mocker, rpl_config)

    rpl_config["pairlists"] = [
        {
            "method": "PercentChangePairList",
            "number_assets": 2,
            "sort_key": "percentage",
            "min_value": 0,
            "refresh_period": 86400,
            "lookback_days": 1001,
        }
    ]

    with pytest.raises(
        OperationalException,
        match=r"ChangeFilter要求lookback_period不超过交易所最大请求大小\(\d+\)",
    ):
        get_patched_freqtradebot(mocker, rpl_config)


def test_volume_change_pair_list_init_wrong_config(mocker, rpl_config):
    rpl_config["pairlists"] = [
        {
            "method": "PercentChangePairList",
            "sort_key": "percentage",
            "min_value": 0,
            "refresh_period": 86400,
        }
    ]

    with pytest.raises(
        OperationalException,
        match=r"`number_assets`未指定。请检查您的配置中的\"pairlist.config.number_assets\"",
    ):
        get_patched_freqtradebot(mocker, rpl_config)


def test_gen_pairlist_with_valid_change_pair_list_config(mocker, rpl_config, tickers, time_machine):
    rpl_config["pairlists"] = [
        {
            "method": "PercentChangePairList",
            "number_assets": 2,
            "sort_key": "percentage",
            "min_value": 0,
            "refresh_period": 86400,
            "lookback_days": 4,
        }
    ]
    start = datetime(2024, 8, 1, 0, 0, 0, 0, tzinfo=timezone.utc)
    time_machine.move_to(start, tick=False)

    mock_ohlcv_data = {
        ("ETH/USDT", "1d", CandleType.SPOT): pd.DataFrame(
            ohlcv_to_dataframe(
                generate_test_data_raw("1d", 100, start.strftime("%Y-%m-%d"), random_seed=12),
                "1d",
                pair="ETH/USDT",
                fill_missing=True,
            )
        ),
        ("BTC/USDT", "1d", CandleType.SPOT): pd.DataFrame(
            ohlcv_to_dataframe(
                generate_test_data_raw("1d", 100, start.strftime("%Y-%m-%d"), random_seed=13),
                "1d",
                pair="BTC/USDT",
                fill_missing=True,
            )
        ),
        ("XRP/USDT", "1d", CandleType.SPOT): pd.DataFrame(
            ohlcv_to_dataframe(
                generate_test_data_raw("1d", 100, start.strftime("%Y-%m-%d"), random_seed=14),
                "1d",
                pair="XRP/USDT",
                fill_missing=True,
            )
        ),
        ("NEO/USDT", "1d", CandleType.SPOT): pd.DataFrame(
            ohlcv_to_dataframe(
                generate_test_data_raw("1d", 100, start.strftime("%Y-%m-%d"), random_seed=15),
                "1d",
                pair="NEO/USDT",
                fill_missing=True,
            )
        ),
        ("TKN/USDT", "1d", CandleType.SPOT): pd.DataFrame(
            # 确保始终有最高的百分比
            {
                "timestamp": [
                    "2024-07-01 00:00:00",
                    "2024-07-01 01:00:00",
                    "2024-07-01 02:00:00",
                    "2024-07-01 03:00:00",
                    "2024-07-01 04:00:00",
                    "2024-07-01 05:00:00",
                ],
                "open": [100, 102, 101, 103, 104, 105],
                "high": [102, 103, 102, 104, 105, 106],
                "low": [99, 101, 100, 102, 103, 104],
                "close": [101, 102, 103, 104, 105, 106],
                "volume": [1000, 1500, 2000, 2500, 3000, 3500],
            }
        ),
    }

    mocker.patch(f"{EXMS}.refresh_latest_ohlcv", MagicMock(return_value=mock_ohlcv_data))

    exchange = get_patched_exchange(mocker, rpl_config, exchange="binance")
    pairlistmanager = PairListManager(exchange, rpl_config)

    remote_pairlist = PercentChangePairList(
        exchange, pairlistmanager, rpl_config, rpl_config["pairlists"][0], 0
    )

    result = remote_pairlist.gen_pairlist(tickers)

    assert len(result) == 2
    assert result == ["NEO/USDT", "TKN/USDT"]


def test_filter_pairlist_with_empty_ticker(mocker, rpl_config, tickers, time_machine):
    rpl_config["pairlists"] = [
        {
            "method": "PercentChangePairList",
            "number_assets": 2,
            "sort_key": "percentage",
            "min_value": 0,
            "refresh_period": 86400,
            "sort_direction": "asc",
            "lookback_days": 4,
        }
    ]
    start = datetime(2024, 8, 1, 0, 0, 0, 0, tzinfo=timezone.utc)
    time_machine.move_to(start, tick=False)

    mock_ohlcv_data = {
        ("ETH/USDT", "1d", CandleType.SPOT): pd.DataFrame(
            {
                "timestamp": [
                    "2024-07-01 00:00:00",
                    "2024-07-01 01:00:00",
                    "2024-07-01 02:00:00",
                    "2024-07-01 03:00:00",
                    "2024-07-01 04:00:00",
                    "2024-07-01 05:00:00",
                ],
                "open": [100, 102, 101, 103, 104, 105],
                "high": [102, 103, 102, 104, 105, 106],
                "low": [99, 101, 100, 102, 103, 104],
                "close": [101, 102, 103, 104, 105, 105],
                "volume": [1000, 1500, 2000, 2500, 3000, 3500],
            }
        ),
        ("XRP/USDT", "1d", CandleType.SPOT): pd.DataFrame(
            {
                "timestamp": [
                    "2024-07-01 00:00:00",
                    "2024-07-01 01:00:00",
                    "2024-07-01 02:00:00",
                    "2024-07-01 03:00:00",
                    "2024-07-01 04:00:00",
                    "2024-07-01 05:00:00",
                ],
                "open": [100, 102, 101, 103, 104, 105],
                "high": [102, 103, 102, 104, 105, 106],
                "low": [99, 101, 100, 102, 103, 104],
                "close": [101, 102, 103, 104, 105, 104],
                "volume": [1000, 1500, 2000, 2500, 3000, 3400],
            }
        ),
    }

    mocker.patch(f"{EXMS}.refresh_latest_ohlcv", MagicMock(return_value=mock_ohlcv_data))
    exchange = get_patched_exchange(mocker, rpl_config, exchange="binance")
    pairlistmanager = PairListManager(exchange, rpl_config)

    remote_pairlist = PercentChangePairList(
        exchange, pairlistmanager, rpl_config, rpl_config["pairlists"][0], 0
    )

    result = remote_pairlist.filter_pairlist(rpl_config["exchange"]["pair_whitelist"], {})

    assert len(result) == 2
    assert result == ["XRP/USDT", "ETH/USDT"]


def test_filter_pairlist_with_max_value_set(mocker, rpl_config, tickers, time_machine):
    rpl_config["pairlists"] = [
        {
            "method": "PercentChangePairList",
            "number_assets": 2,
            "sort_key": "percentage",
            "min_value": 0,
            "max_value": 15,
            "refresh_period": 86400,
            "lookback_days": 4,
        }
    ]

    start = datetime(2024, 8, 1, 0, 0, 0, 0, tzinfo=timezone.utc)
    time_machine.move_to(start, tick=False)

    mock_ohlcv_data = {
        ("ETH/USDT", "1d", CandleType.SPOT): pd.DataFrame(
            {
                "timestamp": [
                    "2024-07-01 00:00:00",
                    "2024-07-01 01:00:00",
                    "2024-07-01 02:00:00",
                    "2024-07-01 03:00:00",
                    "2024-07-01 04:00:00",
                    "2024-07-01 05:00:00",
                ],
                "open": [100, 102, 101, 103, 104, 105],
                "high": [102, 103, 102, 104, 105, 106],
                "low": [99, 101, 100, 102, 103, 104],
                "close": [101, 102, 103, 104, 105, 106],
                "volume": [1000, 1500, 2000, 1800, 2400, 2500],
            }
        ),
        ("XRP/USDT", "1d", CandleType.SPOT): pd.DataFrame(
            {
                "timestamp": [
                    "2024-07-01 00:00:00",
                    "2024-07-01 01:00:00",
                    "2024-07-01 02:00:00",
                    "2024-07-01 03:00:00",
                    "2024-07-01 04:00:00",
                    "2024-07-01 05:00:00",
                ],
                "open": [100, 102, 101, 103, 104, 105],
                "high": [102, 103, 102, 104, 105, 106],
                "low": [99, 101, 100, 102, 103, 104],
                "close": [101, 102, 103, 104, 105, 101],
                "volume": [1000, 1500, 2000, 2500, 3000, 3500],
            }
        ),
    }

    mocker.patch(f"{EXMS}.refresh_latest_ohlcv", MagicMock(return_value=mock_ohlcv_data))
    exchange = get_patched_exchange(mocker, rpl_config, exchange="binance")
    pairlistmanager = PairListManager(exchange, rpl_config)

    remote_pairlist = PercentChangePairList(
        exchange, pairlistmanager, rpl_config, rpl_config["pairlists"][0], 0
    )

    result = remote_pairlist.filter_pairlist(rpl_config["exchange"]["pair_whitelist"], {})

    assert len(result) == 1
    assert result == ["ETH/USDT"]


def test_gen_pairlist_from_tickers(mocker, rpl_config, tickers):
    rpl_config["pairlists"] = [
        {
            "method": "PercentChangePairList",
            "number_assets": 2,
            "sort_key": "percentage",
            "min_value": 0,
        }
    ]

    mocker.patch(f"{EXMS}.exchange_has", MagicMock(return_value=True))

    exchange = get_patched_exchange(mocker, rpl_config, exchange="binance")
    pairlistmanager = PairListManager(exchange, rpl_config)

    remote_pairlist = pairlistmanager._pairlist_handlers[0]

    # 生成器返回BTC ETH和TKN - 过滤第一个确保在此步骤中移除交易对没有问题
    def _validate_pair(pair, ticker):
        if pair == "BTC/USDT":
            return False
        return True

    remote_pairlist._validate_pair = _validate_pair

    result = remote_pairlist.gen_pairlist(tickers.return_value)

    assert len(result) == 1
    assert result == ["ETH/USDT"]