# pragma pylint: disable=missing-docstring,C0103,protected-access

import logging
import time
from copy import deepcopy
from datetime import timedelta
from unittest.mock import MagicMock, PropertyMock

import pandas as pd
import pytest
import time_machine

from freqtrade.constants import AVAILABLE_PAIRLISTS
from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import CandleType, RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.persistence import LocalTrade, Trade
from freqtrade.plugins.pairlist.pairlist_helpers import dynamic_expand_pairlist, expand_pairlist
from freqtrade.plugins.pairlistmanager import PairListManager
from freqtrade.resolvers import PairListResolver
from freqtrade.util.datetime_helpers import dt_now
from tests.conftest import (
    EXMS,
    create_mock_trades_usdt,
    generate_test_data,
    get_patched_exchange,
    get_patched_freqtradebot,
    log_has,
    log_has_re,
    num_log_has,
)


# 从测试中排除 RemotePairList。
# 它有一个必需参数，需要特殊处理，这在 test_remotepairlist 中处理。
TESTABLE_PAIRLISTS = [p for p in AVAILABLE_PAIRLISTS if p not in ["RemotePairList"]]


@pytest.fixture(scope="function")
def whitelist_conf(default_conf):
    default_conf["runmode"] = "dry_run"
    default_conf["stake_currency"] = "BTC"
    default_conf["exchange"]["pair_whitelist"] = [
        "ETH/BTC",
        "TKN/BTC",
        "TRST/BTC",
        "SWT/BTC",
        "BCC/BTC",
        "HOT/BTC",
    ]
    default_conf["exchange"]["pair_blacklist"] = ["BLK/BTC"]
    default_conf["pairlists"] = [
        {
            "method": "VolumePairList",
            "number_assets": 5,
            "sort_key": "quoteVolume",
        },
    ]
    default_conf.update(
        {
            "external_message_consumer": {
                "enabled": True,
                "producers": [],
            }
        }
    )
    return default_conf


@pytest.fixture(scope="function")
def whitelist_conf_2(default_conf):
    default_conf["runmode"] = "dry_run"
    default_conf["stake_currency"] = "BTC"
    default_conf["exchange"]["pair_whitelist"] = [
        "ETH/BTC",
        "TKN/BTC",
        "BLK/BTC",
        "LTC/BTC",
        "BTT/BTC",
        "HOT/BTC",
        "FUEL/BTC",
        "XRP/BTC",
    ]
    default_conf["exchange"]["pair_blacklist"] = ["BLK/BTC"]
    default_conf["pairlists"] = [
        # {   "method": "StaticPairList"},
        {
            "method": "VolumePairList",
            "number_assets": 5,
            "sort_key": "quoteVolume",
            "refresh_period": 0,
        },
    ]
    return default_conf


@pytest.fixture(scope="function")
def whitelist_conf_agefilter(default_conf):
    default_conf["runmode"] = "dry_run"
    default_conf["stake_currency"] = "BTC"
    default_conf["exchange"]["pair_whitelist"] = [
        "ETH/BTC",
        "TKN/BTC",
        "BLK/BTC",
        "LTC/BTC",
        "BTT/BTC",
        "HOT/BTC",
        "FUEL/BTC",
        "XRP/BTC",
    ]
    default_conf["exchange"]["pair_blacklist"] = ["BLK/BTC"]
    default_conf["pairlists"] = [
        {
            "method": "VolumePairList",
            "number_assets": 5,
            "sort_key": "quoteVolume",
            "refresh_period": -1,
        },
        {"method": "AgeFilter", "min_days_listed": 2, "max_days_listed": 100},
    ]
    return default_conf


@pytest.fixture(scope="function")
def static_pl_conf(whitelist_conf):
    whitelist_conf["pairlists"] = [
        {
            "method": "StaticPairList",
        },
    ]
    return whitelist_conf


def test_log_cached(mocker, static_pl_conf, markets, tickers):
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True),
        get_tickers=tickers,
    )
    freqtrade = get_patched_freqtradebot(mocker, static_pl_conf)
    logmock = MagicMock()
    # 分配起始白名单
    pl = freqtrade.pairlists._pairlist_handlers[0]
    pl.log_once("Hello world", logmock)
    assert logmock.call_count == 1
    pl.log_once("Hello world", logmock)
    assert logmock.call_count == 1
    assert pl._log_cache.currsize == 1
    assert ("Hello world",) in pl._log_cache._Cache__data

    pl.log_once("Hello world2", logmock)
    assert logmock.call_count == 2
    assert pl._log_cache.currsize == 2


def test_load_pairlist_noexist(mocker, markets, default_conf):
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets))
    plm = PairListManager(freqtrade.exchange, default_conf, MagicMock())
    with pytest.raises(
        OperationalException,
        match=r"无法加载配对列表 'NonexistingPairList'。"
        r"此类不存在或包含Python代码错误。",
    ):
        PairListResolver.load_pairlist(
            "NonexistingPairList", freqtrade.exchange, plm, default_conf, {}, 1
        )


def test_load_pairlist_verify_multi(mocker, markets_static, default_conf):
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets_static))
    plm = PairListManager(freqtrade.exchange, default_conf, MagicMock())
    # 逐个调用不同版本，应该总是考虑传入的内容
    # 并且没有副作用（因此多次相同检查）
    assert plm.verify_whitelist(["ETH/BTC", "XRP/BTC"], print) == ["ETH/BTC", "XRP/BTC"]
    assert plm.verify_whitelist(["ETH/BTC", "XRP/BTC", "BUUU/BTC"], print) == ["ETH/BTC", "XRP/BTC"]
    assert plm.verify_whitelist(["XRP/BTC", "BUUU/BTC"], print) == ["XRP/BTC"]
    assert plm.verify_whitelist(["ETH/BTC", "XRP/BTC"], print) == ["ETH/BTC", "XRP/BTC"]
    assert plm.verify_whitelist(["ETH/USDT", "XRP/USDT"], print) == ["ETH/USDT"]
    assert plm.verify_whitelist(["ETH/BTC", "XRP/BTC"], print) == ["ETH/BTC", "XRP/BTC"]


def test_refresh_market_pair_not_in_whitelist(mocker, markets, static_pl_conf):
    freqtrade = get_patched_freqtradebot(mocker, static_pl_conf)

    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets))
    freqtrade.pairlists.refresh_pairlist()
    # 按基础交易量排序的列表
    whitelist = ["ETH/BTC", "TKN/BTC"]
    # 确保除白名单中的以外，所有配对都被移除
    assert set(whitelist) == set(freqtrade.pairlists.whitelist)
    # 确保配置字典没有被更改
    assert (
        static_pl_conf["exchange"]["pair_whitelist"]
        == freqtrade.config["exchange"]["pair_whitelist"]
    )


def test_refresh_static_pairlist(mocker, markets, static_pl_conf):
    freqtrade = get_patched_freqtradebot(mocker, static_pl_conf)
    mocker.patch.multiple(
        EXMS,
        exchange_has=MagicMock(return_value=True),
        markets=PropertyMock(return_value=markets),
    )
    freqtrade.pairlists.refresh_pairlist()
    # 按基础交易量排序的列表
    whitelist = ["ETH/BTC", "TKN/BTC"]
    # 确保除白名单中的以外，所有配对都被移除
    assert set(whitelist) == set(freqtrade.pairlists.whitelist)
    assert static_pl_conf["exchange"]["pair_blacklist"] == freqtrade.pairlists.blacklist


@pytest.mark.parametrize(
    "pairs,expected",
    [
        (
            ["NOEXIST/BTC", r"\+WHAT/BTC"],
            ["ETH/BTC", "TKN/BTC", "TRST/BTC", "NOEXIST/BTC", "SWT/BTC", "BCC/BTC", "HOT/BTC"],
        ),
        (
            ["NOEXIST/BTC", r"*/BTC"],  # 这是一个无效的正则表达式
            [],
        ),
    ],
)
def test_refresh_static_pairlist_noexist(mocker, markets, static_pl_conf, pairs, expected, caplog):
    static_pl_conf["pairlists"][0]["allow_inactive"] = True
    static_pl_conf["exchange"]["pair_whitelist"] += pairs
    freqtrade = get_patched_freqtradebot(mocker, static_pl_conf)
    mocker.patch.multiple(
        EXMS,
        exchange_has=MagicMock(return_value=True),
        markets=PropertyMock(return_value=markets),
    )
    freqtrade.pairlists.refresh_pairlist()

    # 确保除白名单中的以外，所有配对都被移除
    assert set(expected) == set(freqtrade.pairlists.whitelist)
    assert static_pl_conf["exchange"]["pair_blacklist"] == freqtrade.pairlists.blacklist
    if not expected:
        assert log_has_re(r"配对白名单包含无效通配符: 通配符错误.*", caplog)


def test_invalid_blacklist(mocker, markets, static_pl_conf, caplog):
    static_pl_conf["exchange"]["pair_blacklist"] = ["*/BTC"]
    freqtrade = get_patched_freqtradebot(mocker, static_pl_conf)
    mocker.patch.multiple(
        EXMS,
        exchange_has=MagicMock(return_value=True),
        markets=PropertyMock(return_value=markets),
    )
    freqtrade.pairlists.refresh_pairlist()
    whitelist = []
    # 确保除白名单中的以外，所有配对都被移除
    assert set(whitelist) == set(freqtrade.pairlists.whitelist)
    assert static_pl_conf["exchange"]["pair_blacklist"] == freqtrade.pairlists.blacklist
    log_has_re(r"配对黑名单包含无效通配符.*", caplog)


def test_remove_logs_for_pairs_already_in_blacklist(mocker, markets, static_pl_conf, caplog):
    logger = logging.getLogger(__name__)
    freqtrade = get_patched_freqtradebot(mocker, static_pl_conf)
    mocker.patch.multiple(
        EXMS,
        exchange_has=MagicMock(return_value=True),
        markets=PropertyMock(return_value=markets),
    )
    freqtrade.pairlists.refresh_pairlist()
    whitelist = ["ETH/BTC", "TKN/BTC"]
    caplog.clear()
    caplog.set_level(logging.INFO)

    # 确保除白名单中的以外，所有配对都被移除。
    assert set(whitelist) == set(freqtrade.pairlists.whitelist)
    assert static_pl_conf["exchange"]["pair_blacklist"] == freqtrade.pairlists.blacklist
    # 确保没有生成日志消息。
    assert not log_has("配对 BLK/BTC 在您的黑名单中。从白名单中移除它...", caplog)

    for _ in range(3):
        new_whitelist = freqtrade.pairlists.verify_blacklist(
            [*whitelist, "BLK/BTC"], logger.warning
        )
        # 确保配对从白名单中移除，并正确记录。
        assert set(whitelist) == set(new_whitelist)
    assert num_log_has("配对 BLK/BTC 在您的黑名单中。从白名单中移除它...", caplog) == 1


def test_refresh_pairlist_dynamic(mocker, shitcoinmarkets, tickers, whitelist_conf):
    mocker.patch.multiple(
        EXMS,
        get_tickers=tickers,
        exchange_has=MagicMock(return_value=True),
    )
    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
    # 使用 shitcoinmarkets 重新模拟市场，因为 get_patched_freqtradebot 使用 markets fixture
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=shitcoinmarkets),
    )
    # 参数：按交易所交易量动态使用白名单
    whitelist = ["ETH/BTC", "TKN/BTC", "LTC/BTC", "XRP/BTC", "HOT/BTC"]
    freqtrade.pairlists.refresh_pairlist()
    assert whitelist == freqtrade.pairlists.whitelist

    whitelist_conf["pairlists"] = [{"method": "VolumePairList"}]
    with pytest.raises(
        OperationalException,
        match=r"`number_assets` 未指定。请检查您的配置"
        r'中的 "pairlist.config.number_assets"',
    ):
        PairListManager(freqtrade.exchange, whitelist_conf, MagicMock())


def test_refresh_pairlist_dynamic_2(mocker, shitcoinmarkets, tickers, whitelist_conf_2):
    tickers_dict = tickers()

    mocker.patch.multiple(
        EXMS,
        exchange_has=MagicMock(return_value=True),
    )
    # 移除ticker数据的缓存以模拟第二次调用时交易量的变化
    mocker.patch.multiple(
        "freqtrade.plugins.pairlistmanager.PairListManager",
        _get_cached_tickers=MagicMock(return_value=tickers_dict),
    )
    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf_2)
    # 使用 shitcoinmarkets 重新模拟市场，因为 get_patched_freqtradebot 使用 markets fixture
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=shitcoinmarkets),
    )

    whitelist = ["ETH/BTC", "TKN/BTC", "LTC/BTC", "XRP/BTC", "HOT/BTC"]
    freqtrade.pairlists.refresh_pairlist()
    assert whitelist == freqtrade.pairlists.whitelist

    # 延迟以允许0 TTL缓存过期...
    time.sleep(1)
    whitelist = ["FUEL/BTC", "ETH/BTC", "TKN/BTC", "LTC/BTC", "XRP/BTC"]
    tickers_dict["FUEL/BTC"]["quoteVolume"] = 10000.0
    freqtrade.pairlists.refresh_pairlist()
    assert whitelist == freqtrade.pairlists.whitelist


def test_VolumePairList_refresh_empty(mocker, markets_empty, whitelist_conf):
    mocker.patch.multiple(
        EXMS,
        exchange_has=MagicMock(return_value=True),
    )
    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets_empty))

    # 参数：按交易所交易量动态使用白名单
    whitelist = []
    whitelist_conf["exchange"]["pair_whitelist"] = []
    freqtrade.pairlists.refresh_pairlist()
    pairslist = whitelist_conf["exchange"]["pair_whitelist"]

    assert set(whitelist) == set(pairslist)


@pytest.mark.parametrize(
    "pairlists,base_currency,whitelist_result",
    [
        # 仅 VolumePairList
        (
            [{"method": "VolumePairList", "number_assets": 5, "sort_key": "quoteVolume"}],
            "BTC",
            ["ETH/BTC", "TKN/BTC", "LTC/BTC", "XRP/BTC", "HOT/BTC"],
        ),
        (
            [{"method": "VolumePairList", "number_assets": 5, "sort_key": "quoteVolume"}],
            "USDT",
            ["ETH/USDT", "NANO/USDT", "ADAHALF/USDT", "ADADOUBLE/USDT"],
        ),
        # ETH 没有配对，VolumePairList
        ([{"method": "VolumePairList", "number_assets": 5, "sort_key": "quoteVolume"}], "ETH", []),
        # ETH 没有配对，StaticPairList
        ([{"method": "StaticPairList"}], "ETH", []),
        # ETH 没有配对，所有处理器
        (
            [
                {"method": "StaticPairList"},
                {"method": "VolumePairList", "number_assets": 5, "sort_key": "quoteVolume"},
                {"method": "AgeFilter", "min_days_listed": 2, "max_days_listed": None},
                {"method": "PrecisionFilter"},
                {"method": "PriceFilter", "low_price_ratio": 0.03},
                {"method": "SpreadFilter", "max_spread_ratio": 0.005},
                {"method": "ShuffleFilter"},
                {"method": "PerformanceFilter"},
            ],
            "ETH",
            [],
        ),
        # AgeFilter 和 VolumePairList（只需要2天，所有都应该通过年龄测试）
        (
            [
                {"method": "VolumePairList", "number_assets": 5, "sort_key": "quoteVolume"},
                {"method": "AgeFilter", "min_days_listed": 2, "max_days_listed": 100},
            ],
            "BTC",
            ["ETH/BTC", "TKN/BTC", "LTC/BTC", "XRP/BTC", "HOT/BTC"],
        ),
        # AgeFilter 和 VolumePairList（需要10天，所有都应该未通过年龄测试）
        (
            [
                {"method": "VolumePairList", "number_assets": 5, "sort_key": "quoteVolume"},
                {"method": "AgeFilter", "min_days_listed": 10, "max_days_listed": None},
            ],
            "BTC",
            [],
        ),
        # AgeFilter 和 VolumePairList（所有配对列出 > 2，所有都应该未通过年龄测试）
        (
            [
                {"method": "VolumePairList", "number_assets": 5, "sort_key": "quoteVolume"},
                {"method": "AgeFilter", "min_days_listed": 1, "max_days_listed": 2},
            ],
            "BTC",
            [],
        ),
        # AgeFilter 和 VolumePairList LTC/BTC 有6根K线 - 移除所有
        (
            [
                {"method": "VolumePairList", "number_assets": 5, "sort_key": "quoteVolume"},
                {"method": "AgeFilter", "min_days_listed": 4, "max_days_listed": 5},
            ],
            "BTC",
            [],
        ),
        # AgeFilter 和 VolumePairList LTC/BTC 有6根K线 - 通过
        (
            [
                {"method": "VolumePairList", "number_assets": 5, "sort_key": "quoteVolume"},
                {"method": "AgeFilter", "min_days_listed": 4, "max_days_listed": 10},
            ],
            "BTC",
            ["LTC/BTC"],
        ),
        # 精度过滤器和报价交易量
        (
            [
                {"method": "VolumePairList", "number_assets": 5, "sort_key": "quoteVolume"},
                {"method": "PrecisionFilter"},
            ],
            "BTC",
            ["ETH/BTC", "TKN/BTC", "LTC/BTC", "XRP/BTC"],
        ),
        (
            [
                {"method": "VolumePairList", "number_assets": 5, "sort_key": "quoteVolume"},
                {"method": "PrecisionFilter"},
            ],
            "USDT",
            ["ETH/USDT", "NANO/USDT"],
        ),
        # 价格过滤器和 VolumePairList
        (
            [
                {"method": "VolumePairList", "number_assets": 5, "sort_key": "quoteVolume"},
                {"method": "PriceFilter", "low_price_ratio": 0.03},
            ],
            "BTC",
            ["ETH/BTC", "TKN/BTC", "LTC/BTC", "XRP/BTC"],
        ),
        # 价格过滤器和 VolumePairList
        (
            [
                {"method": "VolumePairList", "number_assets": 5, "sort_key": "quoteVolume"},
                {"method": "PriceFilter", "low_price_ratio": 0.03},
            ],
            "USDT",
            ["ETH/USDT", "NANO/USDT"],
        ),
        # Hot 被精度过滤器移除，Fuel 被低价比率移除，Ripple 被最低价移除。
        (
            [
                {"method": "VolumePairList", "number_assets": 6, "sort_key": "quoteVolume"},
                {"method": "PrecisionFilter"},
                {"method": "PriceFilter", "low_price_ratio": 0.02, "min_price": 0.01},
            ],
            "BTC",
            ["ETH/BTC", "TKN/BTC", "LTC/BTC"],
        ),
        # Hot 被精度过滤器移除，Fuel 被低价比率移除，Ethereum 被最高价移除。
        (
            [
                {"method": "VolumePairList", "number_assets": 6, "sort_key": "quoteVolume"},
                {"method": "PrecisionFilter"},
                {"method": "PriceFilter", "low_price_ratio": 0.02, "max_price": 0.05},
            ],
            "BTC",
            ["TKN/BTC", "LTC/BTC", "XRP/BTC"],
        ),
        # HOT 和 XRP 被移除，因为低于1250报价交易量
        (
            [
                {
                    "method": "VolumePairList",
                    "number_assets": 5,
                    "sort_key": "quoteVolume",
                    "min_value": 1250,
                }
            ],
            "BTC",
            ["ETH/BTC", "TKN/BTC", "LTC/BTC"],
        ),
        # HOT、XRP 和 FUEL 被列入白名单，因为它们低于1300报价交易量。
        (
            [
                {
                    "method": "VolumePairList",
                    "number_assets": 5,
                    "sort_key": "quoteVolume",
                    "max_value": 1300,
                }
            ],
            "BTC",
            ["XRP/BTC", "HOT/BTC", "FUEL/BTC"],
        ),
        # HOT、XRP 被列入白名单，因为它们在100到1300报价交易量之间。
        (
            [
                {
                    "method": "VolumePairList",
                    "number_assets": 5,
                    "sort_key": "quoteVolume",
                    "min_value": 100,
                    "max_value": 1300,
                }
            ],
            "BTC",
            ["XRP/BTC", "HOT/BTC"],
        ),
        # 仅 StaticPairlist
        ([{"method": "StaticPairList"}], "BTC", ["ETH/BTC", "TKN/BTC", "HOT/BTC"]),
        # StaticPairlist 在 VolumePairList 之前 - 排序发生变化
        # SpreadFilter
        (
            [
                {"method": "VolumePairList", "number_assets": 5, "sort_key": "quoteVolume"},
                {"method": "SpreadFilter", "max_spread_ratio": 0.005},
            ],
            "USDT",
            ["ETH/USDT"],
        ),
        # ShuffleFilter
        (
            [
                {"method": "VolumePairList", "number_assets": 5, "sort_key": "quoteVolume"},
                {"method": "ShuffleFilter", "seed": 77},
            ],
            "USDT",
            ["ADADOUBLE/USDT", "ETH/USDT", "NANO/USDT", "ADAHALF/USDT"],
        ),
        # ShuffleFilter，其他种子
        (
            [
                {"method": "VolumePairList", "number_assets": 5, "sort_key": "quoteVolume"},
                {"method": "ShuffleFilter", "seed": 42},
            ],
            "USDT",
            ["ADAHALF/USDT", "NANO/USDT", "ADADOUBLE/USDT", "ETH/USDT"],
        ),
        # ShuffleFilter，无种子
        (
            [
                {"method": "VolumePairList", "number_assets": 5, "sort_key": "quoteVolume"},
                {"method": "ShuffleFilter"},
            ],
            "USDT",
            4,
        ),  # whitelist_result 是整数 -- 只检查随机配对列表的长度
        # 仅 AgeFilter
        (
            [{"method": "AgeFilter", "min_days_listed": 2}],
            "BTC",
            "filter_at_the_beginning",
        ),  # 预期 OperationalException
        # StaticPairList 后的 PrecisionFilter
        (
            [{"method": "StaticPairList"}, {"method": "PrecisionFilter"}],
            "BTC",
            ["ETH/BTC", "TKN/BTC"],
        ),
        # 仅 PrecisionFilter
        (
            [{"method": "PrecisionFilter"}],
            "BTC",
            "filter_at_the_beginning",
        ),  # 预期 OperationalException
        # StaticPairList 后的 PriceFilter
        (
            [
                {"method": "StaticPairList"},
                {
                    "method": "PriceFilter",
                    "low_price_ratio": 0.02,
                    "min_price": 0.000001,
                    "max_price": 0.1,
                },
            ],
            "BTC",
            ["ETH/BTC", "TKN/BTC"],
        ),
        # 仅 PriceFilter
        (
            [{"method": "PriceFilter", "low_price_ratio": 0.02}],
            "BTC",
            "filter_at_the_beginning",
        ),  # 预期 OperationalException
        # StaticPairList 后的 ShuffleFilter
        (
            [{"method": "StaticPairList"}, {"method": "ShuffleFilter", "seed": 42}],
            "BTC",
            ["TKN/BTC", "ETH/BTC", "HOT/BTC"],
        ),
        # 仅 ShuffleFilter
        (
            [{"method": "ShuffleFilter", "seed": 42}],
            "BTC",
            "filter_at_the_beginning",
        ),  # 预期 OperationalException
        # StaticPairList 后的 PerformanceFilter
        (
            [{"method": "StaticPairList"}, {"method": "PerformanceFilter"}],
            "BTC",
            ["ETH/BTC", "TKN/BTC", "HOT/BTC"],
        ),
        # 仅 PerformanceFilter
        (
            [{"method": "PerformanceFilter"}],
            "BTC",
            "filter_at_the_beginning",
        ),  # 预期 OperationalException
        # StaticPairList 后的 SpreadFilter
        (
            [{"method": "StaticPairList"}, {"method": "SpreadFilter", "max_spread_ratio": 0.005}],
            "BTC",
            ["ETH/BTC", "TKN/BTC"],
        ),
        # 仅 SpreadFilter
        (
            [{"method": "SpreadFilter", "max_spread_ratio": 0.005}],
            "BTC",
            "filter_at_the_beginning",
        ),  # 预期 OperationalException
        # VolumePairList 后的 StaticPairlist，在非第一位置（追加配对）
        (
            [
                {"method": "VolumePairList", "number_assets": 2, "sort_key": "quoteVolume"},
                {"method": "StaticPairList"},
            ],
            "BTC",
            ["ETH/BTC", "TKN/BTC", "TRST/BTC", "SWT/BTC", "BCC/BTC", "HOT/BTC"],
        ),
        (
            [
                {"method": "VolumePairList", "number_assets": 20, "sort_key": "quoteVolume"},
                {"method": "PriceFilter", "low_price_ratio": 0.02},
            ],
            "USDT",
            ["ETH/USDT", "NANO/USDT"],
        ),
        (
            [
                {"method": "VolumePairList", "number_assets": 20, "sort_key": "quoteVolume"},
                {"method": "PriceFilter", "max_value": 0.000001},
            ],
            "USDT",
            ["NANO/USDT"],
        ),
        (
            [
                {"method": "StaticPairList"},
                {
                    "method": "RangeStabilityFilter",
                    "lookback_days": 10,
                    "min_rate_of_change": 0.01,
                    "refresh_period": 1440,
                },
            ],
            "BTC",
            ["ETH/BTC", "TKN/BTC", "HOT/BTC"],
        ),
        (
            [
                {"method": "StaticPairList"},
                {
                    "method": "RangeStabilityFilter",
                    "lookback_days": 10,
                    "max_rate_of_change": 0.01,
                    "refresh_period": 1440,
                },
            ],
            "BTC",
            [],
        ),  # 全部移除，因为 max_rate_of_change 为 0.017
        (
            [
                {"method": "StaticPairList"},
                {
                    "method": "RangeStabilityFilter",
                    "lookback_days": 10,
                    "min_rate_of_change": 0.018,
                    "max_rate_of_change": 0.02,
                    "refresh_period": 1440,
                },
            ],
            "BTC",
            [],
        ),  # 全部移除 - 限制高于最高 change_rate
        (
            [
                {"method": "StaticPairList"},
                {
                    "method": "VolatilityFilter",
                    "lookback_days": 3,
                    "min_volatility": 0.002,
                    "max_volatility": 0.004,
                    "refresh_period": 1440,
                },
            ],
            "BTC",
            ["ETH/BTC", "TKN/BTC"],
        ),
        # 无偏移的 VolumePairList = 不变的配对列表
        (
            [
                {"method": "VolumePairList", "number_assets": 20, "sort_key": "quoteVolume"},
                {"method": "OffsetFilter", "offset": 0, "number_assets": 0},
            ],
            "USDT",
            ["ETH/USDT", "NANO/USDT", "ADAHALF/USDT", "ADADOUBLE/USDT"],
        ),
        # 偏移 = 2 的 VolumePairList
        (
            [
                {"method": "VolumePairList", "number_assets": 20, "sort_key": "quoteVolume"},
                {"method": "OffsetFilter", "offset": 2},
            ],
            "USDT",
            ["ADAHALF/USDT", "ADADOUBLE/USDT"],
        ),
        # 带偏移和限制的 VolumePairList
        (
            [
                {"method": "VolumePairList", "number_assets": 20, "sort_key": "quoteVolume"},
                {"method": "OffsetFilter", "offset": 1, "number_assets": 2},
            ],
            "USDT",
            ["NANO/USDT", "ADAHALF/USDT"],
        ),
        # VolumePairList 偏移高于总配对列表
        (
            [
                {"method": "VolumePairList", "number_assets": 20, "sort_key": "quoteVolume"},
                {"method": "OffsetFilter", "offset": 100},
            ],
            "USDT",
            [],
        ),
    ],
)
def test_VolumePairList_whitelist_gen(
    mocker,
    whitelist_conf,
    shitcoinmarkets,
    tickers,
    ohlcv_history,
    pairlists,
    base_currency,
    whitelist_result,
    caplog,
) -> None:
    whitelist_conf["runmode"] = "util_exchange"
    whitelist_conf["pairlists"] = pairlists
    whitelist_conf["stake_currency"] = base_currency

    ohlcv_history_high_vola = ohlcv_history.copy()
    ohlcv_history_high_vola.loc[ohlcv_history_high_vola.index == 1, "close"] = 0.00090

    ohlcv_data = {
        ("ETH/BTC", "1d", CandleType.SPOT): ohlcv_history,
        ("TKN/BTC", "1d", CandleType.SPOT): ohlcv_history,
        ("LTC/BTC", "1d", CandleType.SPOT): pd.concat([ohlcv_history, ohlcv_history]),
        ("XRP/BTC", "1d", CandleType.SPOT): ohlcv_history,
        ("HOT/BTC", "1d", CandleType.SPOT): ohlcv_history_high_vola,
    }

    mocker.patch(f"{EXMS}.exchange_has", MagicMock(return_value=True))

    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
    mocker.patch.multiple(
        EXMS, get_tickers=tickers, markets=PropertyMock(return_value=shitcoinmarkets)
    )
    mocker.patch.multiple(
        EXMS,
        refresh_latest_ohlcv=MagicMock(return_value=ohlcv_data),
    )

    # 为 PerformanceFilter 的依赖项提供支持
    mocker.patch.multiple(
        "freqtrade.persistence.Trade", get_overall_performance=MagicMock(return_value=[])
    )

    # 如果配对列表无效并应产生异常，则将 whitelist_result 设置为 None
    if whitelist_result == "filter_at_the_beginning":
        with pytest.raises(
            OperationalException,
            match=r"此配对列表处理器不应在配对列表处理器列表的第一位置使用。",
        ):
            freqtrade.pairlists.refresh_pairlist()
    else:
        freqtrade.pairlists.refresh_pairlist()
        whitelist = freqtrade.pairlists.whitelist

        assert isinstance(whitelist, list)

        # 验证配对列表的长度匹配（用于无种子的 ShuffleFilter）
        if isinstance(whitelist_result, list):
            assert whitelist == whitelist_result
        else:
            assert len(whitelist) == whitelist_result

        for pairlist in pairlists:
            if (
                pairlist["method"] == "AgeFilter"
                and pairlist["min_days_listed"]
                and len(ohlcv_history) < pairlist["min_days_listed"]
            ):
                assert log_has_re(
                    r"^从白名单中移除 .* ，因为年龄 .* 小于 " r".* 天.*", caplog
                )
            if (
                pairlist["method"] == "AgeFilter"
                and pairlist["max_days_listed"]
                and len(ohlcv_history) > pairlist["max_days_listed"]
            ):
                assert log_has_re(
                    r"^从白名单中移除 .* ，因为年龄 .* 小于 "
                    r".* 天.* 或超过 .* 天",
                    caplog,
                )
            if pairlist["method"] == "PrecisionFilter" and whitelist_result:
                assert log_has_re(
                    r"^从白名单中移除 .* ，因为止损价格 .* "
                    r"将 <= 止损限制.*",
                    caplog,
                )
            if pairlist["method"] == "PriceFilter" and whitelist_result:
                assert (
                    log_has_re(r"^从白名单中移除 .* ，因为1个单位为 .*%$", caplog)
                    or log_has_re(
                        r"^从白名单中移除 .* ，" r"因为最后价格 < .*%$", caplog
                    )
                    or log_has_re(
                        r"^从白名单中移除 .* ，" r"因为最后价格 > .*%$", caplog
                    )
                    or log_has_re(
                        r"^从白名单中移除 .* ，" r"因为最小值变化为 .*", caplog
                    )
                    or log_has_re(
                        r"^从白名单中移除 .* ，因为ticker\['last'\] " r"为空.*",
                        caplog,
                    )
                )
            if pairlist["method"] == "VolumePairList":
                logmsg = (
                    "已弃用：VolumePairList使用除quoteVolume以外的任何键已弃用。"
                )
                if pairlist["sort_key"] != "quoteVolume":
                    assert log_has(logmsg, caplog)
                else:
                    assert not log_has(logmsg, caplog)
            if pairlist["method"] == "VolatilityFilter":
                assert log_has_re(r"^从白名单中移除 .* ，因为波动性.*$", caplog)


@pytest.mark.parametrize(
    "pairlists,base_currency,exchange,volumefilter_result",
    [
        # 默认刷新1800对于每日K线回看来说太小
        (
            [
                {
                    "method": "VolumePairList",
                    "number_assets": 5,
                    "sort_key": "quoteVolume",
                    "lookback_days": 1,
                }
            ],
            "BTC",
            "binance",
            "default_refresh_too_short",
        ),  # 预期 OperationalException
        # 回看天数和周期的模糊配置
        (
            [
                {
                    "method": "VolumePairList",
                    "number_assets": 5,
                    "sort_key": "quoteVolume",
                    "lookback_days": 1,
                    "lookback_period": 1,
                }
            ],
            "BTC",
            "binance",
            "lookback_days_and_period",
        ),  # 预期 OperationalException
        # 负数回看周期
        (
            [
                {
                    "method": "VolumePairList",
                    "number_assets": 5,
                    "sort_key": "quoteVolume",
                    "lookback_timeframe": "1d",
                    "lookback_period": -1,
                }
            ],
            "BTC",
            "binance",
            "lookback_period_negative",
        ),  # 预期 OperationalException
        # 回看范围超过交易所限制
        (
            [
                {
                    "method": "VolumePairList",
                    "number_assets": 5,
                    "sort_key": "quoteVolume",
                    "lookback_timeframe": "1m",
                    "lookback_period": 2000,
                    "refresh_period": 3600,
                }
            ],
            "BTC",
            "binance",
            "lookback_exceeds_exchange_request_size",
        ),  # 预期 OperationalException
        # 预期配对如给定
        (
            [
                {
                    "method": "VolumePairList",
                    "number_assets": 5,
                    "sort_key": "quoteVolume",
                    "lookback_timeframe": "1d",
                    "lookback_period": 1,
                    "refresh_period": 86400,
                }
            ],
            "BTC",
            "binance",
            ["LTC/BTC", "ETH/BTC", "TKN/BTC", "XRP/BTC", "HOT/BTC"],
        ),
        # 预期配对作为输入，因为1h K线不可用
        (
            [
                {
                    "method": "VolumePairList",
                    "number_assets": 5,
                    "sort_key": "quoteVolume",
                    "lookback_timeframe": "1h",
                    "lookback_period": 2,
                    "refresh_period": 3600,
                }
            ],
            "BTC",
            "binance",
            ["ETH/BTC", "LTC/BTC", "NEO/BTC", "TKN/BTC", "XRP/BTC"],
        ),
        # TKN/BTC 被移除，因为它没有足够的K线
        (
            [
                {
                    "method": "VolumePairList",
                    "number_assets": 5,
                    "sort_key": "quoteVolume",
                    "lookback_timeframe": "1d",
                    "lookback_period": 6,
                    "refresh_period": 86400,
                }
            ],
            "BTC",
            "binance",
            ["LTC/BTC", "XRP/BTC", "ETH/BTC", "HOT/BTC", "NEO/BTC"],
        ),
        # 范围模式下的 VolumePairlist 作为过滤器。
        # TKN/BTC 被移除，因为它没有足够的K线
        (
            [
                {"method": "VolumePairList", "number_assets": 5},
                {
                    "method": "VolumePairList",
                    "number_assets": 5,
                    "sort_key": "quoteVolume",
                    "lookback_timeframe": "1d",
                    "lookback_period": 2,
                    "refresh_period": 86400,
                },
            ],
            "BTC",
            "binance",
            ["LTC/BTC", "XRP/BTC", "ETH/BTC", "TKN/BTC", "HOT/BTC"],
        ),
        # ftx 数据已经在报价货币中，因此不需要转换
        # ([{"method": "VolumePairList", "number_assets": 5, "sort_key": "quoteVolume",
        #    "lookback_timeframe": "1d", "lookback_period": 1, "refresh_period": 86400}],
        #  "BTC", "ftx", ['HOT/BTC', 'LTC/BTC', 'ETH/BTC', 'TKN/BTC', 'XRP/BTC']),
    ],
)
def test_VolumePairList_range(
    mocker,
    whitelist_conf,
    shitcoinmarkets,
    tickers,
    ohlcv_history,
    pairlists,
    base_currency,
    exchange,
    volumefilter_result,
    time_machine,
) -> None:
    whitelist_conf["pairlists"] = pairlists
    whitelist_conf["stake_currency"] = base_currency
    whitelist_conf["exchange"]["name"] = exchange
    # 确保我们有6根K线
    ohlcv_history_long = pd.concat([ohlcv_history, ohlcv_history])

    ohlcv_history_high_vola = ohlcv_history_long.copy()
    ohlcv_history_high_vola.loc[ohlcv_history_high_vola.index == 1, "close"] = 0.00090

    # 为中等总交易量创建K线，最后一根K线高交易量
    ohlcv_history_medium_volume = ohlcv_history_long.copy()
    ohlcv_history_medium_volume.loc[ohlcv_history_medium_volume.index == 2, "volume"] = 5

    # 为高交易量创建K线，所有K线都高交易量，但价格非常低。
    ohlcv_history_high_volume = ohlcv_history_long.copy()
    ohlcv_history_high_volume["volume"] = 10
    ohlcv_history_high_volume["low"] = ohlcv_history_high_volume.loc[:, "low"] * 0.01
    ohlcv_history_high_volume["high"] = ohlcv_history_high_volume.loc[:, "high"] * 0.01
    ohlcv_history_high_volume["close"] = ohlcv_history_high_volume.loc[:, "close"] * 0.01

    ohlcv_data = {
        ("ETH/BTC", "1d", CandleType.SPOT): ohlcv_history_long,
        ("TKN/BTC", "1d", CandleType.SPOT): ohlcv_history,
        ("LTC/BTC", "1d", CandleType.SPOT): ohlcv_history_medium_volume,
        ("XRP/BTC", "1d", CandleType.SPOT): ohlcv_history_high_vola,
        ("HOT/BTC", "1d", CandleType.SPOT): ohlcv_history_high_volume,
    }

    mocker.patch(f"{EXMS}.exchange_has", MagicMock(return_value=True))

    if volumefilter_result == "default_refresh_too_short":
        with pytest.raises(
            OperationalException,
            match=r"刷新周期 [0-9]+ 秒小于一个时间框架 "
            r"[0-9]+.*\. 请将 refresh_period 调整至至少 [0-9]+ "
            r"并重启机器人\.",
        ):
            freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
        return
    elif volumefilter_result == "lookback_days_and_period":
        with pytest.raises(
            OperationalException,
            match=r"模糊配置：在配对列表配置中同时设置了 lookback_days 和 lookback_period \..*",
        ):
            freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
    elif volumefilter_result == "lookback_period_negative":
        with pytest.raises(
            OperationalException, match=r"VolumeFilter 要求 lookback_period >= 0"
        ):
            freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
    elif volumefilter_result == "lookback_exceeds_exchange_request_size":
        with pytest.raises(
            OperationalException,
            match=r"VolumeFilter 要求 lookback_period 不超过 "
            r"交易所最大请求大小 \([0-9]+\)",
        ):
            freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
    else:
        freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
        mocker.patch.multiple(
            EXMS, get_tickers=tickers, markets=PropertyMock(return_value=shitcoinmarkets)
        )
        start_dt = dt_now()
        time_machine.move_to(start_dt)
        # 当 looback_timeframe != 1d 时移除 ohlcv
        # 强制回退到ticker数据
        if "lookback_timeframe" in pairlists[0]:
            if pairlists[0]["lookback_timeframe"] != "1d":
                ohlcv_data = {}

        ohclv_mock = mocker.patch(f"{EXMS}.refresh_latest_ohlcv", return_value=ohlcv_data)

        freqtrade.pairlists.refresh_pairlist()
        whitelist = freqtrade.pairlists.whitelist
        assert ohclv_mock.call_count == 1

        assert isinstance(whitelist, list)
        assert whitelist == volumefilter_result
        # 测试缓存
        ohclv_mock.reset_mock()
        freqtrade.pairlists.refresh_pairlist()
        # 在"过滤器"模式下，缓存被禁用。
        assert ohclv_mock.call_count == 0
        whitelist = freqtrade.pairlists.whitelist
        assert whitelist == volumefilter_result

        time_machine.move_to(start_dt + timedelta(days=2))
        ohclv_mock.reset_mock()
        freqtrade.pairlists.refresh_pairlist()
        assert ohclv_mock.call_count == 1
        whitelist = freqtrade.pairlists.whitelist
        assert whitelist == volumefilter_result


def test_PrecisionFilter_error(mocker, whitelist_conf) -> None:
    whitelist_conf["pairlists"] = [{"method": "StaticPairList"}, {"method": "PrecisionFilter"}]
    del whitelist_conf["stoploss"]

    mocker.patch(f"{EXMS}.exchange_has", MagicMock(return_value=True))

    with pytest.raises(
        OperationalException, match=r"PrecisionFilter 只能在定义了止损时工作\..*"
    ):
        PairListManager(MagicMock, whitelist_conf, MagicMock())


def test_PerformanceFilter_error(mocker, whitelist_conf, caplog) -> None:
    whitelist_conf["pairlists"] = [{"method": "StaticPairList"}, {"method": "PerformanceFilter"}]
    if hasattr(Trade, "session"):
        del Trade.session
    mocker.patch(f"{EXMS}.exchange_has", MagicMock(return_value=True))
    exchange = get_patched_exchange(mocker, whitelist_conf)
    pm = PairListManager(exchange, whitelist_conf, MagicMock())
    pm.refresh_pairlist()

    assert log_has("PerformanceFilter 在此模式下不可用。", caplog)


def test_VolatilityFilter_error(mocker, whitelist_conf) -> None:
    volatility_filter = {"method": "VolatilityFilter", "lookback_days": -1}
    whitelist_conf["pairlists"] = [{"method": "StaticPairList"}, volatility_filter]

    mocker.patch(f"{EXMS}.exchange_has", MagicMock(return_value=True))
    exchange_mock = MagicMock()
    exchange_mock.ohlcv_candle_limit = MagicMock(return_value=1000)

    with pytest.raises(
        OperationalException, match=r"VolatilityFilter 要求 lookback_days >= 1*"
    ):
        PairListManager(exchange_mock, whitelist_conf, MagicMock())

    volatility_filter = {"method": "VolatilityFilter", "lookback_days": 2000}
    whitelist_conf["pairlists"] = [{"method": "StaticPairList"}, volatility_filter]
    with pytest.raises(
        OperationalException,
        match=r"VolatilityFilter 要求 lookback_days 不超过交易所最大",
    ):
        PairListManager(exchange_mock, whitelist_conf, MagicMock())

    volatility_filter = {"method": "VolatilityFilter", "sort_direction": "Random"}
    whitelist_conf["pairlists"] = [{"method": "StaticPairList"}, volatility_filter]
    with pytest.raises(
        OperationalException,
        match=r"VolatilityFilter 要求 sort_direction 为 " r"None .*'asc'.*'desc' 之一",
    ):
        PairListManager(exchange_mock, whitelist_conf, MagicMock())


@pytest.mark.parametrize(
    "pairlist,expected_pairlist",
    [
        (
            {"method": "VolatilityFilter", "sort_direction": "asc"},
            ["XRP/BTC", "ETH/BTC", "LTC/BTC", "TKN/BTC"],
        ),
        (
            {"method": "VolatilityFilter", "sort_direction": "desc"},
            ["TKN/BTC", "LTC/BTC", "ETH/BTC", "XRP/BTC"],
        ),
        (
            {"method": "VolatilityFilter", "sort_direction": "desc", "min_volatility": 0.4},
            ["TKN/BTC", "LTC/BTC", "ETH/BTC"],
        ),
        (
            {"method": "VolatilityFilter", "sort_direction": "asc", "min_volatility": 0.4},
            ["ETH/BTC", "LTC/BTC", "TKN/BTC"],
        ),
        (
            {"method": "VolatilityFilter", "sort_direction": "desc", "max_volatility": 0.5},
            ["LTC/BTC", "ETH/BTC", "XRP/BTC"],
        ),
        (
            {"method": "VolatilityFilter", "sort_direction": "asc", "max_volatility": 0.5},
            ["XRP/BTC", "ETH/BTC", "LTC/BTC"],
        ),
        (
            {"method": "RangeStabilityFilter", "sort_direction": "asc"},
            ["ETH/BTC", "XRP/BTC", "LTC/BTC", "TKN/BTC"],
        ),
        (
            {"method": "RangeStabilityFilter", "sort_direction": "desc"},
            ["TKN/BTC", "LTC/BTC", "XRP/BTC", "ETH/BTC"],
        ),
        (
            {"method": "RangeStabilityFilter", "sort_direction": "asc", "min_rate_of_change": 0.4},
            ["XRP/BTC", "LTC/BTC", "TKN/BTC"],
        ),
        (
            {"method": "RangeStabilityFilter", "sort_direction": "desc", "min_rate_of_change": 0.4},
            ["TKN/BTC", "LTC/BTC", "XRP/BTC"],
        ),
    ],
)
def test_VolatilityFilter_RangeStabilityFilter_sort(
    mocker, whitelist_conf, tickers, time_machine, pairlist, expected_pairlist
) -> None:
    whitelist_conf["pairlists"] = [{"method": "VolumePairList", "number_assets": 10}, pairlist]

    df1 = generate_test_data("1d", 10, "2022-01-05 00:00:00+00:00", random_seed=42)
    df2 = generate_test_data("1d", 10, "2022-01-05 00:00:00+00:00", random_seed=2)
    df3 = generate_test_data("1d", 10, "2022-01-05 00:00:00+00:00", random_seed=3)
    df4 = generate_test_data("1d", 10, "2022-01-05 00:00:00+00:00", random_seed=4)
    df5 = generate_test_data("1d", 10, "2022-01-05 00:00:00+00:00", random_seed=5)
    df6 = generate_test_data("1d", 10, "2022-01-05 00:00:00+00:00", random_seed=6)

    assert not df1.equals(df2)
    time_machine.move_to("2022-01-15 00:00:00+00:00")

    ohlcv_data = {
        ("ETH/BTC", "1d", CandleType.SPOT): df1,
        ("TKN/BTC", "1d", CandleType.SPOT): df2,
        ("LTC/BTC", "1d", CandleType.SPOT): df3,
        ("XRP/BTC", "1d", CandleType.SPOT): df4,
        ("HOT/BTC", "1d", CandleType.SPOT): df5,
        ("BLK/BTC", "1d", CandleType.SPOT): df6,
    }
    ohlcv_mock = MagicMock(return_value=ohlcv_data)
    mocker.patch.multiple(
        EXMS,
        exchange_has=MagicMock(return_value=True),
        refresh_latest_ohlcv=ohlcv_mock,
        get_tickers=tickers,
    )

    exchange = get_patched_exchange(mocker, whitelist_conf)
    exchange.ohlcv_candle_limit = MagicMock(return_value=1000)
    plm = PairListManager(exchange, whitelist_conf, MagicMock())

    assert exchange.ohlcv_candle_limit.call_count == 2
    plm.refresh_pairlist()
    assert ohlcv_mock.call_count == 1
    assert exchange.ohlcv_candle_limit.call_count == 2
    assert plm.whitelist == expected_pairlist

    plm.refresh_pairlist()
    assert exchange.ohlcv_candle_limit.call_count == 2
    assert ohlcv_mock.call_count == 1


def test_ShuffleFilter_init(mocker, whitelist_conf, caplog) -> None:
    whitelist_conf["pairlists"] = [
        {"method": "StaticPairList"},
        {"method": "ShuffleFilter", "seed": 43},
    ]
    whitelist_conf["runmode"] = "backtest"

    exchange = get_patched_exchange(mocker, whitelist_conf)
    plm = PairListManager(exchange, whitelist_conf)
    assert log_has("检测到回测模式，应用种子值：43", caplog)

    with time_machine.travel("2021-09-01 05:01:00 +00:00") as t:
        plm.refresh_pairlist()
        pl1 = deepcopy(plm.whitelist)
        plm.refresh_pairlist()
        assert plm.whitelist == pl1

        t.shift(timedelta(minutes=10))
        plm.refresh_pairlist()
        assert plm.whitelist != pl1

    caplog.clear()
    whitelist_conf["runmode"] = RunMode.DRY_RUN
    plm = PairListManager(exchange, whitelist_conf)
    assert not log_has("检测到回测模式，应用种子值：42", caplog)
    assert log_has("检测到实时模式，不应用种子。", caplog)


@pytest.mark.usefixtures("init_persistence")
def test_PerformanceFilter_lookback(mocker, default_conf_usdt, fee, caplog) -> None:
    default_conf_usdt["exchange"]["pair_whitelist"].extend(["ADA/USDT", "XRP/USDT", "ETC/USDT"])
    default_conf_usdt["pairlists"] = [
        {"method": "StaticPairList"},
        {"method": "PerformanceFilter", "minutes": 60, "min_profit": 0.01},
    ]
    mocker.patch(f"{EXMS}.exchange_has", MagicMock(return_value=True))
    exchange = get_patched_exchange(mocker, default_conf_usdt)
    pm = PairListManager(exchange, default_conf_usdt)
    pm.refresh_pairlist()

    assert pm.whitelist == ["ETH/USDT", "XRP/USDT", "NEO/USDT", "TKN/USDT"]

    with time_machine.travel("2021-09-01 05:00:00 +00:00") as t:
        create_mock_trades_usdt(fee)
        pm.refresh_pairlist()
        assert pm.whitelist == ["XRP/USDT", "NEO/USDT"]
        assert log_has_re(r"移除配对 .* 因为 .* 低于 .*", caplog)

        # 移动到回看窗口"外部"，所以恢复原始排序。
        t.move_to("2021-09-01 07:00:00 +00:00")
        pm.refresh_pairlist()
        assert pm.whitelist == ["ETH/USDT", "XRP/USDT", "NEO/USDT", "TKN/USDT"]


@pytest.mark.usefixtures("init_persistence")
def test_PerformanceFilter_keep_mid_order(mocker, default_conf_usdt, fee, caplog) -> None:
    default_conf_usdt["exchange"]["pair_whitelist"].extend(["ADA/USDT", "ETC/USDT"])
    default_conf_usdt["pairlists"] = [
        {"method": "StaticPairList", "allow_inactive": True},
        {
            "method": "PerformanceFilter",
            "minutes": 60,
        },
    ]
    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    exchange = get_patched_exchange(mocker, default_conf_usdt)
    pm = PairListManager(exchange, default_conf_usdt)
    pm.refresh_pairlist()

    assert pm.whitelist == [
        "ETH/USDT",
        "LTC/USDT",
        "XRP/USDT",
        "NEO/USDT",
        "TKN/USDT",
        "ADA/USDT",
        "ETC/USDT",
    ]

    with time_machine.travel("2021-09-01 05:00:00 +00:00") as t:
        create_mock_trades_usdt(fee)
        pm.refresh_pairlist()
        assert pm.whitelist == [
            "XRP/USDT",
            "NEO/USDT",
            "ETH/USDT",
            "LTC/USDT",
            "TKN/USDT",
            "ADA/USDT",
            "ETC/USDT",
        ]
        # assert log_has_re(r'移除配对 .* 因为 .* 低于 .*', caplog)

        # 移动到回看窗口"外部"，所以恢复原始排序。
        t.move_to("2021-09-01 07:00:00 +00:00")
        pm.refresh_pairlist()
        assert pm.whitelist == [
            "ETH/USDT",
            "LTC/USDT",
            "XRP/USDT",
            "NEO/USDT",
            "TKN/USDT",
            "ADA/USDT",
            "ETC/USDT",
        ]


def test_gen_pair_whitelist_not_supported(mocker, default_conf, tickers) -> None:
    default_conf["pairlists"] = [{"method": "VolumePairList", "number_assets": 10}]

    mocker.patch.multiple(
        EXMS,
        get_tickers=tickers,
        exchange_has=MagicMock(return_value=False),
    )

    with pytest.raises(
        OperationalException, match=r"交易所不支持动态白名单.*"
    ):
        get_patched_freqtradebot(mocker, default_conf)


def test_pair_whitelist_not_supported_Spread(mocker, default_conf, tickers) -> None:
    default_conf["pairlists"] = [{"method": "StaticPairList"}, {"method": "SpreadFilter"}]

    mocker.patch.multiple(
        EXMS,
        get_tickers=tickers,
        exchange_has=MagicMock(return_value=False),
    )

    with pytest.raises(OperationalException, match=r"交易所不支持 fetchTickers， .*"):
        get_patched_freqtradebot(mocker, default_conf)

    mocker.patch(f"{EXMS}.exchange_has", MagicMock(return_value=True))
    mocker.patch(f"{EXMS}.get_option", MagicMock(return_value=False))
    with pytest.raises(OperationalException, match=r".*需要交易所有买/卖数据"):
        get_patched_freqtradebot(mocker, default_conf)


@pytest.mark.parametrize("pairlist", TESTABLE_PAIRLISTS)
def test_pairlist_class(mocker, whitelist_conf, markets, pairlist):
    whitelist_conf["pairlists"][0]["method"] = pairlist
    mocker.patch.multiple(
        EXMS, markets=PropertyMock(return_value=markets), exchange_has=MagicMock(return_value=True)
    )
    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)

    assert freqtrade.pairlists.name_list == [pairlist]
    assert pairlist in str(freqtrade.pairlists.short_desc())
    assert isinstance(freqtrade.pairlists.whitelist, list)
    assert isinstance(freqtrade.pairlists.blacklist, list)


@pytest.mark.parametrize("pairlist", TESTABLE_PAIRLISTS)
@pytest.mark.parametrize(
    "whitelist,log_message",
    [
        (["ETH/BTC", "TKN/BTC"], ""),
        # TRX/ETH 不在市场中
        (["ETH/BTC", "TKN/BTC", "TRX/ETH"], "与交易所不兼容"),
        # 错误的质押
        (["ETH/BTC", "TKN/BTC", "ETH/USDT"], "与您的质押货币不兼容"),
        # BCH/BTC 不可用
        (["ETH/BTC", "TKN/BTC", "BCH/BTC"], "与交易所不兼容"),
        # BTT/BTC 非活跃
        (["ETH/BTC", "TKN/BTC", "BTT/BTC"], "市场不活跃"),
        # XLTCUSDT 不是有效配对
        (["ETH/BTC", "TKN/BTC", "XLTCUSDT"], "无法与 Freqtrade 交易"),
    ],
)
def test__whitelist_for_active_markets(
    mocker, whitelist_conf, markets, pairlist, whitelist, caplog, log_message, tickers
):
    whitelist_conf["pairlists"][0]["method"] = pairlist
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True),
        get_tickers=tickers,
    )
    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
    caplog.clear()

    # 分配起始白名单
    pairlist_handler = freqtrade.pairlists._pairlist_handlers[0]
    new_whitelist = pairlist_handler._whitelist_for_active_markets(whitelist)

    assert set(new_whitelist) == set(["ETH/BTC", "TKN/BTC"])
    assert log_message in caplog.text


@pytest.mark.parametrize("pairlist", TESTABLE_PAIRLISTS)
def test__whitelist_for_active_markets_empty(mocker, whitelist_conf, pairlist, tickers):
    whitelist_conf["pairlists"][0]["method"] = pairlist

    mocker.patch(f"{EXMS}.exchange_has", return_value=True)

    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
    mocker.patch.multiple(EXMS, markets=PropertyMock(return_value=None), get_tickers=tickers)
    # 分配起始白名单
    pairlist_handler = freqtrade.pairlists._pairlist_handlers[0]
    with pytest.raises(OperationalException, match=r"市场未加载.*"):
        pairlist_handler._whitelist_for_active_markets(["ETH/BTC"])


def test_volumepairlist_invalid_sortvalue(mocker, whitelist_conf):
    whitelist_conf["pairlists"][0].update({"sort_key": "asdf"})

    mocker.patch(f"{EXMS}.exchange_has", MagicMock(return_value=True))
    with pytest.raises(OperationalException, match=r"键 asdf 不在 .* 中"):
        get_patched_freqtradebot(mocker, whitelist_conf)


def test_volumepairlist_caching(mocker, markets, whitelist_conf, tickers):
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True),
        get_tickers=tickers,
    )
    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
    assert len(freqtrade.pairlists._pairlist_handlers[0]._pair_cache) == 0
    assert tickers.call_count == 0
    freqtrade.pairlists.refresh_pairlist()
    assert tickers.call_count == 1

    assert len(freqtrade.pairlists._pairlist_handlers[0]._pair_cache) == 1
    freqtrade.pairlists.refresh_pairlist()
    assert tickers.call_count == 1


def test_agefilter_min_days_listed_too_small(mocker, default_conf, markets, tickers):
    default_conf["pairlists"] = [
        {"method": "VolumePairList", "number_assets": 10},
        {"method": "AgeFilter", "min_days_listed": -1},
    ]

    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True),
        get_tickers=tickers,
    )

    with pytest.raises(
        OperationalException, match=r"AgeFilter 要求 min_days_listed >= 1"
    ):
        get_patched_freqtradebot(mocker, default_conf)


def test_agefilter_max_days_lower_than_min_days(mocker, default_conf, markets, tickers):
    default_conf["pairlists"] = [
        {"method": "VolumePairList", "number_assets": 10},
        {"method": "AgeFilter", "min_days_listed": 3, "max_days_listed": 2},
    ]

    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True),
        get_tickers=tickers,
    )

    with pytest.raises(
        OperationalException, match=r"AgeFilter max_days_listed <= min_days_listed 不被允许"
    ):
        get_patched_freqtradebot(mocker, default_conf)


def test_agefilter_min_days_listed_too_large(mocker, default_conf, markets, tickers):
    default_conf["pairlists"] = [
        {"method": "VolumePairList", "number_assets": 10},
        {"method": "AgeFilter", "min_days_listed": 99999},
    ]

    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True),
        get_tickers=tickers,
    )

    with pytest.raises(
        OperationalException,
        match=r"AgeFilter 要求 min_days_listed 不超过 "
        r"交易所最大请求大小 \([0-9]+\)",
    ):
        get_patched_freqtradebot(mocker, default_conf)


def test_agefilter_caching(mocker, markets, whitelist_conf_agefilter, tickers, ohlcv_history):
    with time_machine.travel("2021-09-01 05:00:00 +00:00") as t:
        ohlcv_data = {
            ("ETH/BTC", "1d", CandleType.SPOT): ohlcv_history,
            ("TKN/BTC", "1d", CandleType.SPOT): ohlcv_history,
            ("LTC/BTC", "1d", CandleType.SPOT): ohlcv_history,
        }
        mocker.patch.multiple(
            EXMS,
            markets=PropertyMock(return_value=markets),
            exchange_has=MagicMock(return_value=True),
            get_tickers=tickers,
            refresh_latest_ohlcv=MagicMock(return_value=ohlcv_data),
        )

        freqtrade = get_patched_freqtradebot(mocker, whitelist_conf_agefilter)
        assert freqtrade.exchange.refresh_latest_ohlcv.call_count == 0
        freqtrade.pairlists.refresh_pairlist()
        assert len(freqtrade.pairlists.whitelist) == 3
        assert freqtrade.exchange.refresh_latest_ohlcv.call_count > 0

        freqtrade.pairlists.refresh_pairlist()
        assert len(freqtrade.pairlists.whitelist) == 3
        # 对 XRP/BTC 的调用被缓存
        assert freqtrade.exchange.refresh_latest_ohlcv.call_count == 2

        ohlcv_data = {
            ("ETH/BTC", "1d", CandleType.SPOT): ohlcv_history,
            ("TKN/BTC", "1d", CandleType.SPOT): ohlcv_history,
            ("LTC/BTC", "1d", CandleType.SPOT): ohlcv_history,
            ("XRP/BTC", "1d", CandleType.SPOT): ohlcv_history.iloc[[0]],
        }
        mocker.patch(f"{EXMS}.refresh_latest_ohlcv", return_value=ohlcv_data)
        freqtrade.pairlists.refresh_pairlist()
        assert len(freqtrade.pairlists.whitelist) == 3
        assert freqtrade.exchange.refresh_latest_ohlcv.call_count == 1

        # 移动到第二天
        t.move_to("2021-09-02 01:00:00 +00:00")
        mocker.patch(f"{EXMS}.refresh_latest_ohlcv", return_value=ohlcv_data)
        freqtrade.pairlists.refresh_pairlist()
        assert len(freqtrade.pairlists.whitelist) == 3
        assert freqtrade.exchange.refresh_latest_ohlcv.call_count == 1

        # 移动另一天使用新的模拟（现在配对足够老了）
        t.move_to("2021-09-03 01:00:00 +00:00")
        # 为 XRP/BTC 调用一次
        ohlcv_data = {
            ("ETH/BTC", "1d", CandleType.SPOT): ohlcv_history,
            ("TKN/BTC", "1d", CandleType.SPOT): ohlcv_history,
            ("LTC/BTC", "1d", CandleType.SPOT): ohlcv_history,
            ("XRP/BTC", "1d", CandleType.SPOT): ohlcv_history,
        }
        mocker.patch(f"{EXMS}.refresh_latest_ohlcv", return_value=ohlcv_data)
        freqtrade.pairlists.refresh_pairlist()
        assert len(freqtrade.pairlists.whitelist) == 4
        # 调用一次（仅为 XRP/BTC）
        assert freqtrade.exchange.refresh_latest_ohlcv.call_count == 1


def test_OffsetFilter_error(mocker, whitelist_conf) -> None:
    whitelist_conf["pairlists"] = [
        {"method": "StaticPairList"},
        {"method": "OffsetFilter", "offset": -1},
    ]

    mocker.patch(f"{EXMS}.exchange_has", MagicMock(return_value=True))

    with pytest.raises(OperationalException, match=r"OffsetFilter 要求 offset >= 0"):
        PairListManager(MagicMock, whitelist_conf)


def test_rangestabilityfilter_checks(mocker, default_conf, markets, tickers):
    default_conf["pairlists"] = [
        {"method": "VolumePairList", "number_assets": 10},
        {"method": "RangeStabilityFilter", "lookback_days": 99999},
    ]

    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True),
        get_tickers=tickers,
    )

    with pytest.raises(
        OperationalException,
        match=r"RangeStabilityFilter 要求 lookback_days 不超过 "
        r"交易所最大请求大小 \([0-9]+\)",
    ):
        get_patched_freqtradebot(mocker, default_conf)

    default_conf["pairlists"] = [
        {"method": "VolumePairList", "number_assets": 10},
        {"method": "RangeStabilityFilter", "lookback_days": 0},
    ]

    with pytest.raises(
        OperationalException, match="RangeStabilityFilter 要求 lookback_days >= 1"
    ):
        get_patched_freqtradebot(mocker, default_conf)

    default_conf["pairlists"] = [
        {"method": "VolumePairList", "number_assets": 10},
        {"method": "RangeStabilityFilter", "sort_direction": "something"},
    ]

    with pytest.raises(
        OperationalException,
        match="RangeStabilityFilter 要求 sort_direction 为 None.*之一",
    ):
        get_patched_freqtradebot(mocker, default_conf)


@pytest.mark.parametrize(
    "min_rate_of_change,max_rate_of_change,expected_length",
    [
        (0.01, 0.99, 5),
        (0.05, 0.0, 0),  # 将 min rate_of_change 设置为 5% 会从白名单中移除所有配对。
    ],
)
def test_rangestabilityfilter_caching(
    mocker,
    markets,
    default_conf,
    tickers,
    ohlcv_history,
    min_rate_of_change,
    max_rate_of_change,
    expected_length,
):
    default_conf["pairlists"] = [
        {"method": "VolumePairList", "number_assets": 10},
        {
            "method": "RangeStabilityFilter",
            "lookback_days": 2,
            "min_rate_of_change": min_rate_of_change,
            "max_rate_of_change": max_rate_of_change,
        },
    ]

    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True),
        get_tickers=tickers,
    )
    ohlcv_data = {
        ("ETH/BTC", "1d", CandleType.SPOT): ohlcv_history,
        ("TKN/BTC", "1d", CandleType.SPOT): ohlcv_history,
        ("LTC/BTC", "1d", CandleType.SPOT): ohlcv_history,
        ("XRP/BTC", "1d", CandleType.SPOT): ohlcv_history,
        ("HOT/BTC", "1d", CandleType.SPOT): ohlcv_history,
        ("BLK/BTC", "1d", CandleType.SPOT): ohlcv_history,
    }
    mocker.patch.multiple(
        EXMS,
        refresh_latest_ohlcv=MagicMock(return_value=ohlcv_data),
    )

    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    assert freqtrade.exchange.refresh_latest_ohlcv.call_count == 0
    freqtrade.pairlists.refresh_pairlist()
    assert len(freqtrade.pairlists.whitelist) == expected_length
    assert freqtrade.exchange.refresh_latest_ohlcv.call_count > 0

    previous_call_count = freqtrade.exchange.refresh_latest_ohlcv.call_count
    freqtrade.pairlists.refresh_pairlist()
    assert len(freqtrade.pairlists.whitelist) == expected_length
    # 自第一次调用以来不应该增加。
    assert freqtrade.exchange.refresh_latest_ohlcv.call_count == previous_call_count


def test_spreadfilter_invalid_data(mocker, default_conf, markets, tickers, caplog):
    default_conf["pairlists"] = [
        {"method": "VolumePairList", "number_assets": 10},
        {"method": "SpreadFilter", "max_spread_ratio": 0.1},
    ]

    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True),
        get_tickers=tickers,
    )

    ftbot = get_patched_freqtradebot(mocker, default_conf)
    ftbot.pairlists.refresh_pairlist()

    assert len(ftbot.pairlists.whitelist) == 5

    tickers.return_value["ETH/BTC"]["ask"] = 0.0
    del tickers.return_value["TKN/BTC"]
    del tickers.return_value["LTC/BTC"]
    mocker.patch.multiple(EXMS, get_tickers=tickers)

    ftbot.pairlists.refresh_pairlist()
    assert log_has_re(r"移除 .* 无效ticker数据.*", caplog)

    assert len(ftbot.pairlists.whitelist) == 2


@pytest.mark.parametrize(
    "pairlistconfig,desc_expected,exception_expected",
    [
        (
            {
                "method": "PriceFilter",
                "low_price_ratio": 0.001,
                "min_price": 0.00000010,
                "max_price": 1.0,
            },
            "[{'PriceFilter': 'PriceFilter - 过滤价格低于 "
            "0.1% 或低于 0.00000010 或高于 1.00000000 的配对。'}]",
            None,
        ),
        (
            {"method": "PriceFilter", "low_price_ratio": 0.001, "min_price": 0.00000010},
            "[{'PriceFilter': 'PriceFilter - 过滤价格低于 0.1% "
            "或低于 0.00000010 的配对。'}]",
            None,
        ),
        (
            {"method": "PriceFilter", "low_price_ratio": 0.001, "max_price": 1.00010000},
            "[{'PriceFilter': 'PriceFilter - 过滤价格低于 0.1% "
            "或高于 1.00010000 的配对。'}]",
            None,
        ),
        (
            {"method": "PriceFilter", "min_price": 0.00002000},
            "[{'PriceFilter': 'PriceFilter - 过滤价格低于 0.00002000 的配对。'}]",
            None,
        ),
        (
            {"method": "PriceFilter", "max_value": 0.00002000},
            "[{'PriceFilter': 'PriceFilter - 过滤价格值高于 0.00002000 的配对。'}]",
            None,
        ),
        (
            {"method": "PriceFilter"},
            "[{'PriceFilter': 'PriceFilter - 未配置价格过滤器。'}]",
            None,
        ),
        (
            {"method": "PriceFilter", "low_price_ratio": -0.001},
            None,
            "PriceFilter 要求 low_price_ratio >= 0",
        ),  # 预期 OperationalException
        (
            {"method": "PriceFilter", "min_price": -0.00000010},
            None,
            "PriceFilter 要求 min_price >= 0",
        ),  # 预期 OperationalException
        (
            {"method": "PriceFilter", "max_price": -1.00010000},
            None,
            "PriceFilter 要求 max_price >= 0",
        ),  # 预期 OperationalException
        (
            {"method": "PriceFilter", "max_value": -1.00010000},
            None,
            "PriceFilter 要求 max_value >= 0",
        ),  # 预期 OperationalException
        (
            {"method": "RangeStabilityFilter", "lookback_days": 10, "min_rate_of_change": 0.01},
            "[{'RangeStabilityFilter': 'RangeStabilityFilter - 过滤在过去几天中变化率 "
            "低于 0.01 的配对。'}]",
            None,
        ),
        (
            {
                "method": "RangeStabilityFilter",
                "lookback_days": 10,
                "min_rate_of_change": 0.01,
                "max_rate_of_change": 0.99,
            },
            "[{'RangeStabilityFilter': 'RangeStabilityFilter - 过滤在过去几天中变化率 "
            "低于 0.01 和高于 0.99 的配对。'}]",
            None,
        ),
        (
            {"method": "OffsetFilter", "offset": 5, "number_assets": 10},
            "[{'OffsetFilter': 'OffsetFilter - 获取 10 个配对，从第 5 个开始。'}]",
            None,
        ),
        (
            {"method": "ProducerPairList"},
            "[{'ProducerPairList': 'ProducerPairList - 默认'}]",
            None,
        ),
        (
            {
                "method": "RemotePairList",
                "number_assets": 10,
                "pairlist_url": "https://example.com",
            },
            "[{'RemotePairList': 'RemotePairList - 来自远程配对列表的 10 个配对。'}]",
            None,
        ),
    ],
)
def test_pricefilter_desc(
    mocker, whitelist_conf, markets, pairlistconfig, desc_expected, exception_expected
):
    mocker.patch.multiple(
        EXMS, markets=PropertyMock(return_value=markets), exchange_has=MagicMock(return_value=True)
    )
    whitelist_conf["pairlists"] = [pairlistconfig]

    if desc_expected is not None:
        freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
        short_desc = str(freqtrade.pairlists.short_desc())
        assert short_desc == desc_expected
    else:  # 预期 OperationalException
        with pytest.raises(OperationalException, match=exception_expected):
            freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)


def test_pairlistmanager_no_pairlist(mocker, whitelist_conf):
    mocker.patch(f"{EXMS}.exchange_has", MagicMock(return_value=True))

    whitelist_conf["pairlists"] = []

    with pytest.raises(OperationalException, match=r"未定义配对列表处理器"):
        get_patched_freqtradebot(mocker, whitelist_conf)


@pytest.mark.parametrize(
    "pairlists,pair_allowlist,overall_performance,allowlist_result",
    [
        # 还没有交易
        (
            [{"method": "StaticPairList"}, {"method": "PerformanceFilter"}],
            ["ETH/BTC", "TKN/BTC", "LTC/BTC"],
            [],
            ["ETH/BTC", "TKN/BTC", "LTC/BTC"],
        ),
        # 快乐路径：降序，所有值已填充
        (
            [{"method": "StaticPairList"}, {"method": "PerformanceFilter"}],
            ["ETH/BTC", "TKN/BTC"],
            [
                {"pair": "TKN/BTC", "profit_ratio": 0.05, "count": 3},
                {"pair": "ETH/BTC", "profit_ratio": 0.04, "count": 2},
            ],
            ["TKN/BTC", "ETH/BTC"],
        ),
        # 忽略允许列表外的性能数据
        (
            [{"method": "StaticPairList"}, {"method": "PerformanceFilter"}],
            ["ETH/BTC", "TKN/BTC"],
            [
                {"pair": "OTHER/BTC", "profit_ratio": 0.05, "count": 3},
                {"pair": "ETH/BTC", "profit_ratio": 0.04, "count": 2},
            ],
            ["ETH/BTC", "TKN/BTC"],
        ),
        # 部分性能数据缺失，在正负利润之间排序
        (
            [{"method": "StaticPairList"}, {"method": "PerformanceFilter"}],
            ["ETH/BTC", "TKN/BTC", "LTC/BTC"],
            [
                {"pair": "ETH/BTC", "profit_ratio": -0.05, "count": 100},
                {"pair": "TKN/BTC", "profit_ratio": 0.04, "count": 2},
            ],
            ["TKN/BTC", "LTC/BTC", "ETH/BTC"],
        ),
        # 性能数据中的平局由计数（升序）打破
        (
            [{"method": "StaticPairList"}, {"method": "PerformanceFilter"}],
            ["ETH/BTC", "TKN/BTC", "LTC/BTC"],
            [
                {"pair": "LTC/BTC", "profit_ratio": -0.0501, "count": 101},
                {"pair": "TKN/BTC", "profit_ratio": -0.0501, "count": 2},
                {"pair": "ETH/BTC", "profit_ratio": -0.0501, "count": 100},
            ],
            ["TKN/BTC", "ETH/BTC", "LTC/BTC"],
        ),
        # 性能和计数平局，由之前的排序顺序打破
        (
            [{"method": "StaticPairList"}, {"method": "PerformanceFilter"}],
            ["ETH/BTC", "TKN/BTC", "LTC/BTC"],
            [
                {"pair": "LTC/BTC", "profit_ratio": -0.0501, "count": 1},
                {"pair": "TKN/BTC", "profit_ratio": -0.0501, "count": 1},
                {"pair": "ETH/BTC", "profit_ratio": -0.0501, "count": 1},
            ],
            ["ETH/BTC", "TKN/BTC", "LTC/BTC"],
        ),
    ],
)
def test_performance_filter(
    mocker,
    whitelist_conf,
    pairlists,
    pair_allowlist,
    overall_performance,
    allowlist_result,
    tickers,
    markets,
    ohlcv_history_list,
):
    allowlist_conf = whitelist_conf
    allowlist_conf["pairlists"] = pairlists
    allowlist_conf["exchange"]["pair_whitelist"] = pair_allowlist

    mocker.patch(f"{EXMS}.exchange_has", MagicMock(return_value=True))

    freqtrade = get_patched_freqtradebot(mocker, allowlist_conf)
    mocker.patch.multiple(EXMS, get_tickers=tickers, markets=PropertyMock(return_value=markets))
    mocker.patch.multiple(
        EXMS,
        get_historic_ohlcv=MagicMock(return_value=ohlcv_history_list),
    )
    mocker.patch.multiple(
        "freqtrade.persistence.Trade",
        get_overall_performance=MagicMock(return_value=overall_performance),
    )
    freqtrade.pairlists.refresh_pairlist()
    allowlist = freqtrade.pairlists.whitelist
    assert allowlist == allowlist_result


@pytest.mark.parametrize(
    "wildcardlist,pairs,expected",
    [
        (["BTC/USDT"], ["BTC/USDT"], ["BTC/USDT"]),
        (["BTC/USDT", "ETH/USDT"], ["BTC/USDT", "ETH/USDT"], ["BTC/USDT", "ETH/USDT"]),
        (["BTC/USDT", "ETH/USDT"], ["BTC/USDT"], ["BTC/USDT"]),  # 测试一个太多
        ([".*/USDT"], ["BTC/USDT", "ETH/USDT"], ["BTC/USDT", "ETH/USDT"]),  # 简单通配符
        (
            [".*C/USDT"],
            ["BTC/USDT", "ETC/USDT", "ETH/USDT"],
            ["BTC/USDT", "ETC/USDT"],
        ),  # 通配符排除一个
        (
            [".*UP/USDT", "BTC/USDT", "ETH/USDT"],
            ["BTC/USDT", "ETC/USDT", "ETH/USDT", "BTCUP/USDT", "XRPUP/USDT", "XRPDOWN/USDT"],
            ["BTC/USDT", "ETH/USDT", "BTCUP/USDT", "XRPUP/USDT"],
        ),  # 通配符排除一个
        (
            ["BTC/.*", "ETH/.*"],
            ["BTC/USDT", "ETC/USDT", "ETH/USDT", "BTC/USD", "ETH/EUR", "BTC/GBP"],
            ["BTC/USDT", "ETH/USDT", "BTC/USD", "ETH/EUR", "BTC/GBP"],
        ),  # 通配符排除一个
        (
            ["*UP/USDT", "BTC/USDT", "ETH/USDT"],
            ["BTC/USDT", "ETC/USDT", "ETH/USDT", "BTCUP/USDT", "XRPUP/USDT", "XRPDOWN/USDT"],
            None,
        ),
        (["BTC/USD"], ["BTC/USD", "BTC/USDT"], ["BTC/USD"]),
    ],
)
def test_expand_pairlist(wildcardlist, pairs, expected):
    if expected is None:
        with pytest.raises(ValueError, match=r"通配符错误在 \*UP/USDT,"):
            expand_pairlist(wildcardlist, pairs)
    else:
        assert sorted(expand_pairlist(wildcardlist, pairs)) == sorted(expected)
        conf = {
            "pairs": wildcardlist,
            "freqai": {
                "enabled": True,
                "feature_parameters": {
                    "include_corr_pairlist": [
                        "BTC/USDT:USDT",
                        "XRP/BUSD",
                    ]
                },
            },
        }
        assert sorted(dynamic_expand_pairlist(conf, pairs)) == sorted(
            [*expected, "BTC/USDT:USDT", "XRP/BUSD"]
        )


@pytest.mark.parametrize(
    "wildcardlist,pairs,expected",
    [
        (["BTC/USDT"], ["BTC/USDT"], ["BTC/USDT"]),
        (["BTC/USDT", "ETH/USDT"], ["BTC/USDT", "ETH/USDT"], ["BTC/USDT", "ETH/USDT"]),
        (["BTC/USDT", "ETH/USDT"], ["BTC/USDT"], ["BTC/USDT", "ETH/USDT"]),  # 测试一个太多
        ([".*/USDT"], ["BTC/USDT", "ETH/USDT"], ["BTC/USDT", "ETH/USDT"]),  # 简单通配符
        (
            [".*C/USDT"],
            ["BTC/USDT", "ETC/USDT", "ETH/USDT"],
            ["BTC/USDT", "ETC/USDT"],
        ),  # 通配符排除一个
        (
            [".*UP/USDT", "BTC/USDT", "ETH/USDT"],
            ["BTC/USDT", "ETC/USDT", "ETH/USDT", "BTCUP/USDT", "XRPUP/USDT", "XRPDOWN/USDT"],
            ["BTC/USDT", "ETH/USDT", "BTCUP/USDT", "XRPUP/USDT"],
        ),  # 通配符排除一个
        (
            ["BTC/.*", "ETH/.*"],
            ["BTC/USDT", "ETC/USDT", "ETH/USDT", "BTC/USD", "ETH/EUR", "BTC/GBP"],
            ["BTC/USDT", "ETH/USDT", "BTC/USD", "ETH/EUR", "BTC/GBP"],
        ),  # 通配符排除一个
        (
            ["*UP/USDT", "BTC/USDT", "ETH/USDT"],
            ["BTC/USDT", "ETC/USDT", "ETH/USDT", "BTCUP/USDT", "XRPUP/USDT", "XRPDOWN/USDT"],
            None,
        ),
        (["HELLO/WORLD"], [], ["HELLO/WORLD"]),  # 保留无效配对
        (["BTC/USD"], ["BTC/USD", "BTC/USDT"], ["BTC/USD"]),
        (["BTC/USDT:USDT"], ["BTC/USDT:USDT", "BTC/USDT"], ["BTC/USDT:USDT"]),
        (
            ["BB_BTC/USDT", "CC_BTC/USDT", "AA_ETH/USDT", "XRP/USDT", "ETH/USDT", "XX_BTC/USDT"],
            ["BTC/USDT", "ETH/USDT"],
            ["XRP/USDT", "ETH/USDT"],
        ),
    ],
)
def test_expand_pairlist_keep_invalid(wildcardlist, pairs, expected):
    if expected is None:
        with pytest.raises(ValueError, match=r"通配符错误在 \*UP/USDT,"):
            expand_pairlist(wildcardlist, pairs, keep_invalid=True)
    else:
        assert sorted(expand_pairlist(wildcardlist, pairs, keep_invalid=True)) == sorted(expected)


def test_ProducerPairlist_no_emc(mocker, whitelist_conf):
    mocker.patch(f"{EXMS}.exchange_has", MagicMock(return_value=True))

    whitelist_conf["pairlists"] = [
        {
            "method": "ProducerPairList",
            "number_assets": 10,
            "producer_name": "hello_world",
        }
    ]
    del whitelist_conf["external_message_consumer"]

    with pytest.raises(
        OperationalException,
        match=r"ProducerPairList 需要启用 external_message_consumer。",
    ):
        get_patched_freqtradebot(mocker, whitelist_conf)


def test_ProducerPairlist(mocker, whitelist_conf, markets):
    mocker.patch(f"{EXMS}.exchange_has", MagicMock(return_value=True))
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True),
    )
    whitelist_conf["pairlists"] = [
        {
            "method": "ProducerPairList",
            "number_assets": 2,
            "producer_name": "hello_world",
        }
    ]
    whitelist_conf.update(
        {
            "external_message_consumer": {
                "enabled": True,
                "producers": [
                    {
                        "name": "hello_world",
                        "host": "null",
                        "port": 9891,
                        "ws_token": "dummy",
                    }
                ],
            }
        }
    )

    exchange = get_patched_exchange(mocker, whitelist_conf)
    dp = DataProvider(whitelist_conf, exchange, None)
    pairs = ["ETH/BTC", "LTC/BTC", "XRP/BTC"]
    # 不同的生产者
    dp._set_producer_pairs([*pairs, "MEEP/USDT"], "default")
    pm = PairListManager(exchange, whitelist_conf, dp)
    pm.refresh_pairlist()
    assert pm.whitelist == []
    # 正确的生产者
    dp._set_producer_pairs(pairs, "hello_world")
    pm.refresh_pairlist()

    # 配对列表减少到2个
    assert pm.whitelist == pairs[:2]
    assert len(pm.whitelist) == 2
    whitelist_conf["exchange"]["pair_whitelist"] = ["TKN/BTC"]

    whitelist_conf["pairlists"] = [
        {"method": "StaticPairList"},
        {
            "method": "ProducerPairList",
            "producer_name": "hello_world",
        },
    ]
    pm = PairListManager(exchange, whitelist_conf, dp)
    pm.refresh_pairlist()
    assert len(pm.whitelist) == 4
    assert pm.whitelist == ["TKN/BTC", *pairs]


@pytest.mark.usefixtures("init_persistence")
def test_FullTradesFilter(mocker, default_conf_usdt, fee, caplog) -> None:
    default_conf_usdt["exchange"]["pair_whitelist"].extend(["ADA/USDT", "XRP/USDT", "ETC/USDT"])
    default_conf_usdt["pairlists"] = [{"method": "StaticPairList"}, {"method": "FullTradesFilter"}]
    default_conf_usdt["max_open_trades"] = -1
    mocker.patch(f"{EXMS}.exchange_has", MagicMock(return_value=True))
    exchange = get_patched_exchange(mocker, default_conf_usdt)
    pm = PairListManager(exchange, default_conf_usdt)
    pm.refresh_pairlist()

    assert pm.whitelist == ["ETH/USDT", "XRP/USDT", "NEO/USDT", "TKN/USDT"]

    with time_machine.travel("2021-09-01 05:00:00 +00:00") as t:
        create_mock_trades_usdt(fee)
        pm.refresh_pairlist()

        # 无限最大开仓交易，所以白名单没有变化
        pm.refresh_pairlist()
        assert pm.whitelist == ["ETH/USDT", "XRP/USDT", "NEO/USDT", "TKN/USDT"]

        # 将最大开仓交易设置为4，过滤器应该清空白名单
        default_conf_usdt["max_open_trades"] = 4
        pm.refresh_pairlist()
        assert pm.whitelist == []
        assert log_has_re(r"白名单有 0 个配对: \[]", caplog)

        list_trades = LocalTrade.get_open_trades()
        assert len(list_trades) == 4

        # 移动到1小时后，关闭一个交易，所以恢复原始排序。
        t.move_to("2021-09-01 07:00:00 +00:00")
        list_trades[2].close(12)
        Trade.commit()

        # 开放交易数量低于最大开仓交易，白名单恢复
        list_trades = LocalTrade.get_open_trades()
        assert len(list_trades) == 3
        pm.refresh_pairlist()
        assert pm.whitelist == ["ETH/USDT", "XRP/USDT", "NEO/USDT", "TKN/USDT"]

        # 将最大开仓交易设置为3，过滤器应该清空白名单
        default_conf_usdt["max_open_trades"] = 3
        pm.refresh_pairlist()
        assert pm.whitelist == []
        assert log_has_re(r"白名单有 0 个配对: \[]", caplog)


@pytest.mark.parametrize(
    "pairlists,trade_mode,result,coin_market_calls",
    [
        (
            [
                # 获取2个配对
                {"method": "StaticPairList", "allow_inactive": True},
                {"method": "MarketCapPairList", "number_assets": 2},
            ],
            "spot",
            ["BTC/USDT", "ETH/USDT"],
            1,
        ),
        (
            [
                # 获取6个配对
                {"method": "StaticPairList", "allow_inactive": True},
                {"method": "MarketCapPairList", "number_assets": 6},
            ],
            "spot",
            ["BTC/USDT", "ETH/USDT", "XRP/USDT", "ADA/USDT"],
            1,
        ),
        (
            [
                # 获取前6名中的3个配对
                {"method": "StaticPairList", "allow_inactive": True},
                {"method": "MarketCapPairList", "max_rank": 6, "number_assets": 3},
            ],
            "spot",
            ["BTC/USDT", "ETH/USDT", "XRP/USDT"],
            1,
        ),
        (
            [
                # 获取前8名中的4个配对
                {"method": "StaticPairList", "allow_inactive": True},
                {"method": "MarketCapPairList", "max_rank": 8, "number_assets": 4},
            ],
            "spot",
            ["BTC/USDT", "ETH/USDT", "XRP/USDT"],
            1,
        ),
        (
            [
                # MarketCapPairList 作为生成器
                {"method": "MarketCapPairList", "number_assets": 5}
            ],
            "spot",
            ["BTC/USDT", "ETH/USDT", "XRP/USDT"],
            1,
        ),
        (
            [
                # MarketCapPairList 作为生成器 - 低 max_rank
                {"method": "MarketCapPairList", "max_rank": 2, "number_assets": 5}
            ],
            "spot",
            ["BTC/USDT", "ETH/USDT"],
            1,
        ),
        (
            [
                # MarketCapPairList 作为生成器 - 期货 - 低 max_rank
                {"method": "MarketCapPairList", "max_rank": 2, "number_assets": 5}
            ],
            "futures",
            ["ETH/USDT:USDT"],
            1,
        ),
        (
            [
                # MarketCapPairList 作为生成器 - 期货 - 低 number_assets
                {"method": "MarketCapPairList", "number_assets": 2}
            ],
            "futures",
            ["ETH/USDT:USDT", "ADA/USDT:USDT"],
            1,
        ),
        (
            [
                # MarketCapPairList 作为生成器 - 期货，1个类别
                {"method": "MarketCapPairList", "number_assets": 2, "categories": ["layer-1"]}
            ],
            "futures",
            ["ETH/USDT:USDT", "ADA/USDT:USDT"],
            ["layer-1"],
        ),
        (
            [
                # MarketCapPairList 作为生成器 - 期货，1个类别
                {
                    "method": "MarketCapPairList",
                    "number_assets": 2,
                    "categories": ["layer-1", "protocol"],
                }
            ],
            "futures",
            ["ETH/USDT:USDT", "ADA/USDT:USDT"],
            ["layer-1", "protocol"],
        ),
    ],
)
def test_MarketCapPairList_filter(
    mocker, default_conf_usdt, trade_mode, markets, pairlists, result, coin_market_calls
):
    test_value = [
        {"symbol": "btc"},
        {"symbol": "eth"},
        {"symbol": "usdt"},
        {"symbol": "bnb"},
        {"symbol": "sol"},
        {"symbol": "xrp"},
        {"symbol": "usdc"},
        {"symbol": "steth"},
        {"symbol": "ada"},
        {"symbol": "avax"},
    ]

    default_conf_usdt["trading_mode"] = trade_mode
    if trade_mode == "spot":
        default_conf_usdt["exchange"]["pair_whitelist"].extend(["BTC/USDT", "ETC/USDT", "ADA/USDT"])
    default_conf_usdt["pairlists"] = pairlists
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True),
    )
    mocker.patch(
        "freqtrade.plugins.pairlist.MarketCapPairList.FtCoinGeckoApi.get_coins_categories_list",
        return_value=[
            {"category_id": "layer-1"},
            {"category_id": "protocol"},
            {"category_id": "defi"},
        ],
    )

    gcm_mock = mocker.patch(
        "freqtrade.plugins.pairlist.MarketCapPairList.FtCoinGeckoApi.get_coins_markets",
        return_value=test_value,
    )

    exchange = get_patched_exchange(mocker, default_conf_usdt)

    pm = PairListManager(exchange, default_conf_usdt)
    pm.refresh_pairlist()
    if isinstance(coin_market_calls, int):
        assert gcm_mock.call_count == coin_market_calls
    else:
        assert gcm_mock.call_count == len(coin_market_calls)
        for call in coin_market_calls:
            assert any(
                "category" in c.kwargs and c.kwargs["category"] == call
                for c in gcm_mock.call_args_list
            )

    assert pm.whitelist == result


def test_MarketCapPairList_timing(mocker, default_conf_usdt, markets, time_machine):
    test_value = [
        {"symbol": "btc"},
        {"symbol": "eth"},
        {"symbol": "usdt"},
        {"symbol": "bnb"},
        {"symbol": "sol"},
        {"symbol": "xrp"},
        {"symbol": "usdc"},
        {"symbol": "steth"},
        {"symbol": "ada"},
        {"symbol": "avax"},
    ]

    default_conf_usdt["trading_mode"] = "spot"
    default_conf_usdt["exchange"]["pair_whitelist"].extend(["BTC/USDT", "ETC/USDT", "ADA/USDT"])
    default_conf_usdt["pairlists"] = [{"method": "MarketCapPairList", "number_assets": 2}]

    markets_mock = MagicMock(return_value=markets)
    mocker.patch.multiple(
        EXMS,
        get_markets=markets_mock,
        exchange_has=MagicMock(return_value=True),
    )

    mocker.patch(
        "freqtrade.plugins.pairlist.MarketCapPairList.FtCoinGeckoApi.get_coins_markets",
        return_value=test_value,
    )

    start_dt = dt_now()

    exchange = get_patched_exchange(mocker, default_conf_usdt)
    time_machine.move_to(start_dt)

    pm = PairListManager(exchange, default_conf_usdt)
    markets_mock.reset_mock()
    pm.refresh_pairlist()
    assert markets_mock.call_count == 3
    markets_mock.reset_mock()

    time_machine.move_to(start_dt + timedelta(hours=20))
    pm.refresh_pairlist()
    # 缓存的配对列表...
    assert markets_mock.call_count == 1

    markets_mock.reset_mock()
    time_machine.move_to(start_dt + timedelta(days=2))
    pm.refresh_pairlist()
    # 不再缓存的配对列表...
    assert markets_mock.call_count == 3


def test_MarketCapPairList_filter_special_no_pair_from_coingecko(
    mocker,
    default_conf_usdt,
    markets,
):
    default_conf_usdt["pairlists"] = [{"method": "MarketCapPairList", "number_assets": 2}]

    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True),
    )

    # 模拟从 coingecko 返回没有配对
    gcm_mock = mocker.patch(
        "freqtrade.plugins.pairlist.MarketCapPairList.FtCoinGeckoApi.get_coins_markets",
        return_value=[],
    )

    exchange = get_patched_exchange(mocker, default_conf_usdt)

    pm = PairListManager(exchange, default_conf_usdt)
    pm.refresh_pairlist()
    assert gcm_mock.call_count == 1
    assert pm.whitelist == []


def test_MarketCapPairList_exceptions(mocker, default_conf_usdt, caplog):
    exchange = get_patched_exchange(mocker, default_conf_usdt)
    default_conf_usdt["pairlists"] = [{"method": "MarketCapPairList"}]
    with pytest.raises(OperationalException, match=r"`number_assets` 未指定.*"):
        # 没有 number_assets
        PairListManager(exchange, default_conf_usdt)

    default_conf_usdt["pairlists"] = [
        {"method": "MarketCapPairList", "number_assets": 20, "max_rank": 500}
    ]
    with caplog.at_level(logging.WARNING):
        PairListManager(exchange, default_conf_usdt)
    assert log_has_re("您设置的最大排名 \\(500\\) 相当高", caplog)
    # 测试无效的硬币市场列表
    mocker.patch(
        "freqtrade.plugins.pairlist.MarketCapPairList.FtCoinGeckoApi.get_coins_categories_list",
        return_value=[
            {"category_id": "layer-1"},
            {"category_id": "protocol"},
            {"category_id": "defi"},
        ],
    )
    default_conf_usdt["pairlists"] = [
        {
            "method": "MarketCapPairList",
            "number_assets": 20,
            "categories": ["layer-1", "defi", "layer250"],
        }
    ]
    with pytest.raises(
        OperationalException, match="类别 layer250 不在 coingecko 类别列表中。"
    ):
        PairListManager(exchange, default_conf_usdt)


@pytest.mark.parametrize(
    "pairlists,expected_error,expected_warning",
    [
        (
            [{"method": "StaticPairList"}],
            None,  # 错误
            None,  # 警告
        ),
        (
            [{"method": "VolumePairList", "number_assets": 10}],
            "VolumePairList",  # 错误
            None,  # 警告
        ),
        (
            [{"method": "MarketCapPairList", "number_assets": 10}],
            None,  # 错误
            r"MarketCapPairList.*前瞻.*",  # 警告
        ),
        (
            [{"method": "StaticPairList"}, {"method": "FullTradesFilter"}],
            None,  # 错误
            r"FullTradesFilter 不生成.*",  # 警告
        ),
        (  # 组合，失败并警告
            [
                {"method": "VolumePairList", "number_assets": 10},
                {"method": "MarketCapPairList", "number_assets": 10},
            ],
            "VolumePairList",  # 错误
            r"MarketCapPairList.*前瞻.*",  # 警告
        ),
    ],
)
def test_backtesting_modes(
    mocker, default_conf_usdt, pairlists, expected_error, expected_warning, caplog, markets, tickers
):
    default_conf_usdt["runmode"] = "dry_run"
    default_conf_usdt["pairlists"] = pairlists

    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True),
        get_tickers=tickers,
    )
    exchange = get_patched_exchange(mocker, default_conf_usdt)

    # 模拟运行模式 - 始终工作
    PairListManager(exchange, default_conf_usdt)

    default_conf_usdt["runmode"] = "backtest"
    if expected_error:
        with pytest.raises(OperationalException, match=f"配对列表处理器 {expected_error}.*"):
            PairListManager(exchange, default_conf_usdt)

    if not expected_error:
        PairListManager(exchange, default_conf_usdt)

    if expected_warning:
        assert log_has_re(f"配对列表处理器 {expected_warning}", caplog)