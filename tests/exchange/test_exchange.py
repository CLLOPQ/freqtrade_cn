import copy
import logging
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from random import randint
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import ccxt
import pytest
from numpy import nan
from pandas import DataFrame, to_datetime

from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS
from freqtrade.enums import CandleType, MarginMode, RunMode, TradingMode
from freqtrade.exceptions import (
    ConfigurationError,
    DDosProtection,
    DependencyException,
    ExchangeError,
    InsufficientFundsError,
    InvalidOrderException,
    OperationalException,
    PricingError,
    TemporaryError,
)
from freqtrade.exchange import (
    Binance,
    Bybit,
    Exchange,
    Kraken,
    market_is_active,
    timeframe_to_prev_date,
)
from freqtrade.exchange.common import (
    API_FETCH_ORDER_RETRY_COUNT,
    API_RETRY_COUNT,
    calculate_backoff,
)
from freqtrade.resolvers.exchange_resolver import ExchangeResolver
from freqtrade.util import dt_now, dt_ts
from tests.conftest import (
    EXMS,
    generate_test_data_raw,
    get_mock_coro,
    get_patched_exchange,
    log_has,
    log_has_re,
    num_log_has_re,
)


# 确保这里始终保留一个未被子类化的交易所！
EXCHANGES = ["binance", "kraken", "gate", "kucoin", "bybit", "okx"]

# 入场价格测试数据
get_entry_rate_data = [
    ("other", 20, 19, 10, 0.0, 20),  # 完整卖单侧
    ("ask", 20, 19, 10, 0.0, 20),    # 完整卖单侧
    ("ask", 20, 19, 10, 1.0, 10),    # 完整最新价侧
    ("ask", 20, 19, 10, 0.5, 15),    # 卖单与最新价之间
    ("ask", 20, 19, 10, 0.7, 13),    # 卖单与最新价之间
    ("ask", 20, 19, 10, 0.3, 17),    # 卖单与最新价之间
    ("ask", 5, 6, 10, 1.0, 5),       # 最新价大于卖单价
    ("ask", 5, 6, 10, 0.5, 5),       # 最新价大于卖单价
    ("ask", 20, 19, 10, None, 20),   # 缺少价格最后平衡
    ("ask", 10, 20, None, 0.5, 10),  # 无最新价 - 使用卖单价
    ("ask", 4, 5, None, 0.5, 4),     # 无最新价 - 使用卖单价
    ("ask", 4, 5, None, 1, 4),       # 无最新价 - 使用卖单价
    ("ask", 4, 5, None, 0, 4),       # 无最新价 - 使用卖单价
    ("same", 21, 20, 10, 0.0, 20),   # 完整买单侧
    ("bid", 21, 20, 10, 0.0, 20),    # 完整买单侧
    ("bid", 21, 20, 10, 1.0, 10),    # 完整最新价侧
    ("bid", 21, 20, 10, 0.5, 15),    # 买单与最新价之间
    ("bid", 21, 20, 10, 0.7, 13),    # 买单与最新价之间
    ("bid", 21, 20, 10, 0.3, 17),    # 买单与最新价之间
    ("bid", 6, 5, 10, 1.0, 5),       # 最新价大于买单价
    ("bid", 21, 20, 10, None, 20),   # 缺少价格最后平衡
    ("bid", 6, 5, 10, 0.5, 5),       # 最新价大于买单价
    ("bid", 21, 20, None, 0.5, 20),  # 无最新价 - 使用买单价
    ("bid", 6, 5, None, 0.5, 5),     # 无最新价 - 使用买单价
    ("bid", 6, 5, None, 1, 5),       # 无最新价 - 使用买单价
    ("bid", 6, 5, None, 0, 5),       # 无最新价 - 使用买单价
]

# 出场价格测试数据
get_exit_rate_data = [
    ("bid", 12.0, 11.0, 11.5, 0.0, 11.0),   # 完整买单侧
    ("bid", 12.0, 11.0, 11.5, 1.0, 11.5),   # 完整最新价侧
    ("bid", 12.0, 11.0, 11.5, 0.5, 11.25),  # 买单与最新价之间
    ("bid", 12.0, 11.2, 10.5, 0.0, 11.2),   # 最新价小于买单价
    ("bid", 12.0, 11.2, 10.5, 1.0, 11.2),   # 最新价小于买单价 - 使用买单价
    ("bid", 12.0, 11.2, 10.5, 0.5, 11.2),   # 最新价小于买单价 - 使用买单价
    ("bid", 0.003, 0.002, 0.005, 0.0, 0.002),
    ("bid", 0.003, 0.002, 0.005, None, 0.002),
    ("ask", 12.0, 11.0, 12.5, 0.0, 12.0),   # 完整卖单侧
    ("ask", 12.0, 11.0, 12.5, 1.0, 12.5),   # 完整最新价侧
    ("ask", 12.0, 11.0, 12.5, 0.5, 12.25),  # 卖单与最新价之间
    ("ask", 12.2, 11.2, 10.5, 0.0, 12.2),   # 最新价小于卖单价
    ("ask", 12.0, 11.0, 10.5, 1.0, 12.0),   # 最新价小于卖单价 - 使用卖单价
    ("ask", 12.0, 11.2, 10.5, 0.5, 12.0),   # 最新价小于卖单价 - 使用卖单价
    ("ask", 10.0, 11.0, 11.0, 0.0, 10.0),
    ("ask", 10.11, 11.2, 11.0, 0.0, 10.11),
    ("ask", 0.001, 0.002, 11.0, 0.0, 0.001),
    ("ask", 0.006, 1.0, 11.0, 0.0, 0.006),
    ("ask", 0.006, 1.0, 11.0, None, 0.006),
]


def ccxt_exceptionhandlers(
    mocker,
    default_conf,
    api_mock,
    exchange_name,
    fun,
    mock_ccxt_fun,
    retries=API_RETRY_COUNT + 1,
    **kwargs,
):
    """测试CCXT异常处理"""
    with patch("freqtrade.exchange.common.time.sleep"):
        with pytest.raises(DDosProtection):
            api_mock.__dict__[mock_ccxt_fun] = MagicMock(side_effect=ccxt.DDoSProtection("DDos"))
            exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
            getattr(exchange, fun)(** kwargs)
        assert api_mock.__dict__[mock_ccxt_fun].call_count == retries

    with pytest.raises(TemporaryError):
        api_mock.__dict__[mock_ccxt_fun] = MagicMock(side_effect=ccxt.OperationFailed("DeaDBeef"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        getattr(exchange, fun)(**kwargs)
    assert api_mock.__dict__[mock_ccxt_fun].call_count == retries

    with pytest.raises(OperationalException):
        api_mock.__dict__[mock_ccxt_fun] = MagicMock(side_effect=ccxt.BaseError("DeadBeef"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        getattr(exchange, fun)(** kwargs)
    assert api_mock.__dict__[mock_ccxt_fun].call_count == 1


async def async_ccxt_exception(
    mocker, default_conf, api_mock, fun, mock_ccxt_fun, retries=API_RETRY_COUNT + 1, **kwargs
):
    """测试异步CCXT异常"""
    with patch("freqtrade.exchange.common.asyncio.sleep", get_mock_coro(None)):
        with pytest.raises(DDosProtection):
            api_mock.__dict__[mock_ccxt_fun] = MagicMock(side_effect=ccxt.DDoSProtection("Dooh"))
            exchange = get_patched_exchange(mocker, default_conf, api_mock)
            await getattr(exchange, fun)(** kwargs)
        assert api_mock.__dict__[mock_ccxt_fun].call_count == retries
    exchange.close()

    with pytest.raises(TemporaryError):
        api_mock.__dict__[mock_ccxt_fun] = MagicMock(side_effect=ccxt.NetworkError("DeadBeef"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        await getattr(exchange, fun)(**kwargs)
    assert api_mock.__dict__[mock_ccxt_fun].call_count == retries
    exchange.close()

    with pytest.raises(OperationalException):
        api_mock.__dict__[mock_ccxt_fun] = MagicMock(side_effect=ccxt.BaseError("DeadBeef"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock)
        await getattr(exchange, fun)(** kwargs)
    assert api_mock.__dict__[mock_ccxt_fun].call_count == 1
    exchange.close()


def test_init(default_conf, mocker, caplog):
    """测试初始化"""
    caplog.set_level(logging.INFO)
    get_patched_exchange(mocker, default_conf)
    assert log_has("实例运行在模拟交易模式下", caplog)


def test_init_ccxt_kwargs(default_conf, mocker, caplog):
    """测试CCXT参数初始化"""
    mocker.patch(f"{EXMS}.reload_markets")
    mocker.patch(f"{EXMS}.validate_stakecurrency")
    aei_mock = mocker.patch(f"{EXMS}.additional_exchange_init")

    caplog.set_level(logging.INFO)
    conf = copy.deepcopy(default_conf)
    conf["exchange"]["ccxt_async_config"] = {"aiohttp_trust_env": True, "asyncio_loop": True}
    ex = Exchange(conf)
    assert log_has(
        "应用额外的ccxt配置: {'aiohttp_trust_env': True, 'asyncio_loop': True}", caplog
    )
    assert ex._api_async.aiohttp_trust_env
    assert not ex._api.aiohttp_trust_env
    assert aei_mock.call_count == 1

    # 重置日志和配置
    caplog.clear()
    conf = copy.deepcopy(default_conf)
    conf["exchange"]["ccxt_config"] = {"TestKWARG": 11}
    conf["exchange"]["ccxt_sync_config"] = {"TestKWARG44": 11}
    conf["exchange"]["ccxt_async_config"] = {"asyncio_loop": True}
    asynclogmsg = "应用额外的ccxt配置: {'TestKWARG': 11, 'asyncio_loop': True}"
    ex = Exchange(conf)
    assert not ex._api_async.aiohttp_trust_env
    assert hasattr(ex._api, "TestKWARG")
    assert ex._api.TestKWARG == 11
    # ccxt_config同时分配给同步和异步
    assert not hasattr(ex._api_async, "TestKWARG44")

    assert hasattr(ex._api_async, "TestKWARG")
    assert log_has("应用额外的ccxt配置: {'TestKWARG': 11, 'TestKWARG44': 11}", caplog)
    assert log_has(asynclogmsg, caplog)
    # 测试额外的头信息情况
    Exchange._ccxt_params = {"hello": "world"}
    ex = Exchange(conf)

    assert log_has("应用额外的ccxt配置: {'TestKWARG': 11, 'TestKWARG44': 11}", caplog)
    assert ex._api.hello == "world"
    assert ex._ccxt_config == {}
    Exchange._headers = {}


def test_destroy(default_conf, mocker, caplog):
    """测试销毁方法"""
    caplog.set_level(logging.DEBUG)
    get_patched_exchange(mocker, default_conf)
    assert log_has("交易所对象已销毁，关闭异步循环", caplog)


def test_init_exception(default_conf, mocker):
    """测试初始化异常"""
    default_conf["exchange"]["name"] = "wrong_exchange_name"

    with pytest.raises(
        OperationalException, match=f"交易所 {default_conf['exchange']['name']} 不被支持"
    ):
        Exchange(default_conf)

    default_conf["exchange"]["name"] = "binance"
    with pytest.raises(
        OperationalException, match=f"交易所 {default_conf['exchange']['name']} 不被支持"
    ):
        mocker.patch("ccxt.binance", MagicMock(side_effect=AttributeError))
        Exchange(default_conf)

    with pytest.raises(
        OperationalException, match=r"CCXT初始化失败。原因: DeadBeef"
    ):
        mocker.patch("ccxt.binance", MagicMock(side_effect=ccxt.BaseError("DeadBeef")))
        Exchange(default_conf)


def test_exchange_resolver(default_conf, mocker, caplog):
    """测试交易所解析器"""
    mocker.patch(f"{EXMS}._init_ccxt", MagicMock(return_value=MagicMock()))
    mocker.patch(f"{EXMS}._load_async_markets")
    mocker.patch(f"{EXMS}.validate_timeframes")
    mocker.patch(f"{EXMS}.validate_stakecurrency")
    mocker.patch(f"{EXMS}.validate_pricing")
    default_conf["exchange"]["name"] = "zaif"
    exchange = ExchangeResolver.load_exchange(default_conf)
    assert isinstance(exchange, Exchange)
    assert log_has_re(r"未找到.*特定的子类。使用通用类代替。", caplog)
    caplog.clear()

    default_conf["exchange"]["name"] = "Bybit"
    exchange = ExchangeResolver.load_exchange(default_conf)
    assert isinstance(exchange, Exchange)
    assert isinstance(exchange, Bybit)
    assert not log_has_re(
        r"未找到.*特定的子类。使用通用类代替。", caplog
    )
    caplog.clear()

    default_conf["exchange"]["name"] = "kraken"
    exchange = ExchangeResolver.load_exchange(default_conf)
    assert isinstance(exchange, Exchange)
    assert isinstance(exchange, Kraken)
    assert not isinstance(exchange, Binance)
    assert not log_has_re(
        r"未找到.*特定的子类。使用通用类代替。", caplog
    )

    default_conf["exchange"]["name"] = "binance"
    exchange = ExchangeResolver.load_exchange(default_conf)
    assert isinstance(exchange, Exchange)
    assert isinstance(exchange, Binance)
    assert not isinstance(exchange, Kraken)

    assert not log_has_re(
        r"未找到.*特定的子类。使用通用类代替。", caplog
    )

    # 测试映射
    default_conf["exchange"]["name"] = "binanceus"
    exchange = ExchangeResolver.load_exchange(default_conf)
    assert isinstance(exchange, Exchange)
    assert isinstance(exchange, Binance)
    assert not isinstance(exchange, Kraken)


def test_validate_order_time_in_force(default_conf, mocker, caplog):
    """测试验证订单有效期"""
    caplog.set_level(logging.INFO)
    # 显式测试Bybit，实现其他策略的交易所需要单独测试
    ex = get_patched_exchange(mocker, default_conf, exchange="bybit")
    tif = {
        "buy": "gtc",
        "sell": "gtc",
    }

    ex.validate_order_time_in_force(tif)
    tif2 = {
        "buy": "fok",
        "sell": "ioc22",
    }
    with pytest.raises(OperationalException, match=r"订单有效期.*不支持.*"):
        ex.validate_order_time_in_force(tif2)
    tif2 = {
        "buy": "fok",
        "sell": "ioc",
    }
    # 修补以查看如果值在ft字典中是否会通过
    ex._ft_has.update({"order_time_in_force": ["GTC", "FOK", "IOC"]})
    ex.validate_order_time_in_force(tif2)


def test_validate_orderflow(default_conf, mocker, caplog):
    """测试验证订单流"""
    caplog.set_level(logging.INFO)
    # 测试Bybit - 因为它不支持历史交易数据
    ex = get_patched_exchange(mocker, default_conf, exchange="bybit")
    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    ex.validate_orderflow({"use_public_trades": False})

    with pytest.raises(ConfigurationError, match=r"交易数据在.*不可用"):
        ex.validate_orderflow({"use_public_trades": True})

    # Binance支持订单流
    ex = get_patched_exchange(mocker, default_conf, exchange="binance")
    ex.validate_orderflow({"use_public_trades": False})
    ex.validate_orderflow({"use_public_trades": True})


def test_validate_freqai_compat(default_conf, mocker, caplog):
    """测试验证FreqAI兼容性"""
    caplog.set_level(logging.INFO)
    # 测试Kraken - 因为它不支持历史交易数据
    ex = get_patched_exchange(mocker, default_conf, exchange="kraken")
    mocker.patch(f"{EXMS}.exchange_has", return_value=True)

    default_conf["freqai"] = {"enabled": False}
    ex.validate_freqai(default_conf)

    default_conf["freqai"] = {"enabled": True}
    with pytest.raises(ConfigurationError, match=r"历史K线数据在.*不可用"):
        ex.validate_freqai(default_conf)

    # Binance支持历史数据
    ex = get_patched_exchange(mocker, default_conf, exchange="binance")
    default_conf["freqai"] = {"enabled": True}
    ex.validate_freqai(default_conf)
    default_conf["freqai"] = {"enabled": False}
    ex.validate_freqai(default_conf)


@pytest.mark.parametrize(
    "price,precision_mode,precision,expected",
    [
        (2.34559, 2, 4, 0.0001),
        (2.34559, 2, 5, 0.00001),
        (2.34559, 2, 3, 0.001),
        (2.9999, 2, 3, 0.001),
        (200.0511, 2, 3, 0.001),
        # 点差测试
        (2.34559, 4, 0.0001, 0.0001),
        (2.34559, 4, 0.00001, 0.00001),
        (2.34559, 4, 0.0025, 0.0025),
        (2.9909, 4, 0.0025, 0.0025),
        (234.43, 4, 0.5, 0.5),
        (234.43, 4, 0.0025, 0.0025),
        (234.43, 4, 0.00013, 0.00013),
    ],
)
def test_price_get_one_pip(default_conf, mocker, price, precision_mode, precision, expected):
    """测试获取一个点差单位"""
    markets = PropertyMock(return_value={"ETH/BTC": {"precision": {"price": precision}}})
    exchange = get_patched_exchange(mocker, default_conf, exchange="binance")
    mocker.patch(f"{EXMS}.markets", markets)
    mocker.patch(f"{EXMS}.precisionMode", PropertyMock(return_value=precision_mode))
    mocker.patch(f"{EXMS}.precision_mode_price", PropertyMock(return_value=precision_mode))
    pair = "ETH/BTC"
    assert pytest.approx(exchange.price_get_one_pip(pair, price)) == expected


def test__get_stake_amount_limit(mocker, default_conf) -> None:
    """测试获取赌注金额限制"""
    exchange = get_patched_exchange(mocker, default_conf, exchange="binance")
    stoploss = -0.05
    markets = {"ETH/BTC": {"symbol": "ETH/BTC"}}

    # 未找到交易对
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets))
    with pytest.raises(ValueError, match=r".*获取市场信息.*"):
        exchange.get_min_pair_stake_amount("BNB/BTC", 1, stoploss)

    # 无成本/数量最小值
    markets["ETH/BTC"]["limits"] = {
        "cost": {"min": None, "max": None},
        "amount": {"min": None, "max": None},
    }
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets))
    result = exchange.get_min_pair_stake_amount("ETH/BTC", 1, stoploss)
    assert result is None
    result = exchange.get_max_pair_stake_amount("ETH/BTC", 1)
    assert result == float("inf")

    # 设置最小/最大成本
    markets["ETH/BTC"]["limits"] = {
        "cost": {"min": 2, "max": 10000},
        "amount": {"min": None, "max": None},
    }
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets))
    # 最小值
    result = exchange.get_min_pair_stake_amount("ETH/BTC", 1, stoploss)
    expected_result = 2 * (1 + 0.05) / (1 - abs(stoploss))
    assert pytest.approx(result) == expected_result
    # 带杠杆
    result = exchange.get_min_pair_stake_amount("ETH/BTC", 1, stoploss, 3.0)
    assert pytest.approx(result) == expected_result / 3
    # 最大值
    result = exchange.get_max_pair_stake_amount("ETH/BTC", 2)
    assert result == 10000

    # 设置最小数量
    markets["ETH/BTC"]["limits"] = {
        "cost": {"min": None, "max": None},
        "amount": {"min": 2, "max": 10000},
    }
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets))
    result = exchange.get_min_pair_stake_amount("ETH/BTC", 2, stoploss)
    expected_result = 2 * 2 * (1 + 0.05)
    assert pytest.approx(result) == expected_result
    # 带杠杆
    result = exchange.get_min_pair_stake_amount("ETH/BTC", 2, stoploss, 5.0)
    assert pytest.approx(result) == expected_result / 5
    # 最大值
    result = exchange.get_max_pair_stake_amount("ETH/BTC", 2)
    assert result == 20000

    # 同时设置最小数量和成本（成本是最小的，因此被忽略）
    markets["ETH/BTC"]["limits"] = {
        "cost": {"min": 2, "max": None},
        "amount": {"min": 2, "max": None},
    }
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets))
    result = exchange.get_min_pair_stake_amount("ETH/BTC", 2, stoploss)
    expected_result = max(2, 2 * 2) * (1 + 0.05)
    assert pytest.approx(result) == expected_result
    # 带杠杆
    result = exchange.get_min_pair_stake_amount("ETH/BTC", 2, stoploss, 10)
    assert pytest.approx(result) == expected_result / 10

    # 同时设置最小数量和成本（数量是最小的）
    markets["ETH/BTC"]["limits"] = {
        "cost": {"min": 8, "max": 10000},
        "amount": {"min": 2, "max": 500},
    }
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets))
    result = exchange.get_min_pair_stake_amount("ETH/BTC", 2, stoploss)
    expected_result = max(8, 2 * 2) * (1 + 0.05) / (1 - abs(stoploss))
    assert pytest.approx(result) == expected_result
    # 带杠杆
    result = exchange.get_min_pair_stake_amount("ETH/BTC", 2, stoploss, 7.0)
    assert pytest.approx(result) == expected_result / 7.0
    # 最大值
    result = exchange.get_max_pair_stake_amount("ETH/BTC", 2)
    assert result == 1000

    result = exchange.get_min_pair_stake_amount("ETH/BTC", 2, -0.4)
    expected_result = max(8, 2 * 2) * 1.5
    assert pytest.approx(result) == expected_result
    # 带杠杆
    result = exchange.get_min_pair_stake_amount("ETH/BTC", 2, -0.4, 8.0)
    assert pytest.approx(result) == expected_result / 8.0
    # 最大值
    result = exchange.get_max_pair_stake_amount("ETH/BTC", 2)
    assert result == 1000

    # 非常大的止损
    result = exchange.get_min_pair_stake_amount("ETH/BTC", 2, -1)
    expected_result = max(8, 2 * 2) * 1.5
    assert pytest.approx(result) == expected_result
    # 带杠杆
    result = exchange.get_min_pair_stake_amount("ETH/BTC", 2, -1, 12.0)
    assert pytest.approx(result) == expected_result / 12
    # 最大值
    result = exchange.get_max_pair_stake_amount("ETH/BTC", 2)
    assert result == 1000

    result = exchange.get_max_pair_stake_amount("ETH/BTC", 2, 12.0)
    assert result == 1000 / 12

    markets["ETH/BTC"]["contractSize"] = "0.01"
    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    exchange = get_patched_exchange(mocker, default_conf, exchange="binance")
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets))

    # 合约大小0.01
    result = exchange.get_min_pair_stake_amount("ETH/BTC", 2, -1)
    assert pytest.approx(result) == expected_result * 0.01
    # 最大值
    result = exchange.get_max_pair_stake_amount("ETH/BTC", 2)
    assert result == 10

    markets["ETH/BTC"]["contractSize"] = "10"
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets))
    # 带杠杆，合约大小10
    result = exchange.get_min_pair_stake_amount("ETH/BTC", 2, -1, 12.0)
    assert pytest.approx(result) == (expected_result / 12) * 10.0
    # 最大值
    result = exchange.get_max_pair_stake_amount("ETH/BTC", 2)
    assert result == 10000


def test_get_min_pair_stake_amount_real_data(mocker, default_conf) -> None:
    """使用真实数据测试获取最小交易对赌注金额"""
    exchange = get_patched_exchange(mocker, default_conf, exchange="binance")
    stoploss = -0.05
    markets = {"ETH/BTC": {"symbol": "ETH/BTC"}}

    # 近似真实的Binance数据
    markets["ETH/BTC"]["limits"] = {
        "cost": {"min": 0.0001, "max": 4000},
        "amount": {"min": 0.001, "max": 10000},
    }
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets))
    result = exchange.get_min_pair_stake_amount("ETH/BTC", 0.020405, stoploss)
    expected_result = max(0.0001, 0.001 * 0.020405) * (1 + 0.05) / (1 - abs(stoploss))
    assert round(result, 8) == round(expected_result, 8)
    # 最大值
    result = exchange.get_max_pair_stake_amount("ETH/BTC", 2.0)
    assert result == 4000

    # 杠杆
    result = exchange.get_min_pair_stake_amount("ETH/BTC", 0.020405, stoploss, 3.0)
    assert round(result, 8) == round(expected_result / 3, 8)

    # 合约大小
    markets["ETH/BTC"]["contractSize"] = 0.1
    result = exchange.get_min_pair_stake_amount("ETH/BTC", 0.020405, stoploss, 3.0)
    assert round(result, 8) == round((expected_result / 3), 8)

    # 最大值
    result = exchange.get_max_pair_stake_amount("ETH/BTC", 12.0)
    assert result == 4000


def test__load_async_markets(default_conf, mocker, caplog):
    """测试异步加载市场数据"""
    mocker.patch(f"{EXMS}._init_ccxt")
    mocker.patch(f"{EXMS}.validate_timeframes")
    mocker.patch(f"{EXMS}.reload_markets")
    mocker.patch(f"{EXMS}.validate_stakecurrency")
    mocker.patch(f"{EXMS}.validate_pricing")
    exchange = Exchange(default_conf)
    exchange._api_async.load_markets = get_mock_coro(None)
    exchange._load_async_markets()
    assert exchange._api_async.load_markets.call_count == 1
    caplog.set_level(logging.DEBUG)

    exchange._api_async.load_markets = get_mock_coro(side_effect=ccxt.BaseError("deadbeef"))
    with pytest.raises(TemporaryError, match="deadbeef"):
        exchange._load_async_markets()

    exchange._api_async.load_markets = get_mock_coro(side_effect=ccxt.DDoSProtection("deadbeef"))
    with pytest.raises(DDosProtection, match="deadbeef"):
        exchange._load_async_markets()

    exchange._api_async.load_markets = get_mock_coro(side_effect=ccxt.OperationFailed("deadbeef"))
    with pytest.raises(TemporaryError, match="deadbeef"):
        exchange._load_async_markets()


def test__load_markets(default_conf, mocker, caplog):
    """测试加载市场数据"""
    caplog.set_level(logging.INFO)
    api_mock = MagicMock()
    api_mock.load_markets = get_mock_coro(side_effect=ccxt.BaseError("SomeError"))
    mocker.patch(f"{EXMS}._init_ccxt", MagicMock(return_value=api_mock))
    mocker.patch(f"{EXMS}.validate_timeframes")
    mocker.patch(f"{EXMS}.validate_stakecurrency")
    mocker.patch(f"{EXMS}.validate_pricing")
    Exchange(default_conf)
    assert log_has("无法加载市场数据。", caplog)

    expected_return = {"ETH/BTC": "available"}
    api_mock = MagicMock()
    api_mock.load_markets = get_mock_coro()
    api_mock.markets = expected_return
    mocker.patch(f"{EXMS}._init_ccxt", MagicMock(return_value=api_mock))
    default_conf["exchange"]["pair_whitelist"] = ["ETH/BTC"]
    ex = Exchange(default_conf)

    assert ex.markets == expected_return


def test_reload_markets(default_conf, mocker, caplog, time_machine):
    """测试重新加载市场数据"""
    caplog.set_level(logging.DEBUG)
    initial_markets = {"ETH/BTC": {}}
    updated_markets = {"ETH/BTC": {}, "LTC/BTC": {}}
    start_dt = dt_now()
    time_machine.move_to(start_dt, tick=False)
    api_mock = MagicMock()
    api_mock.load_markets = get_mock_coro(return_value=initial_markets)
    api_mock.markets = initial_markets
    default_conf["exchange"]["markets_refresh_interval"] = 10
    exchange = get_patched_exchange(
        mocker, default_conf, api_mock, exchange="binance", mock_markets=False
    )
    lam_spy = mocker.spy(exchange, "_load_async_markets")
    assert exchange._last_markets_refresh == dt_ts()

    assert exchange.markets == initial_markets

    time_machine.move_to(start_dt + timedelta(minutes=8), tick=False)
    # 不到10分钟，不重新加载
    exchange.reload_markets()
    assert exchange.markets == initial_markets
    assert lam_spy.call_count == 0

    api_mock.load_markets = get_mock_coro(return_value=updated_markets)
    # 超过10分钟，执行重新加载
    time_machine.move_to(start_dt + timedelta(minutes=11), tick=False)
    api_mock.markets = updated_markets
    exchange.reload_markets()
    assert exchange.markets == updated_markets
    assert lam_spy.call_count == 1
    assert log_has("执行计划的市场重新加载..", caplog)

    # 不会再次调用
    lam_spy.reset_mock()

    exchange.reload_markets()
    assert lam_spy.call_count == 0

    # 另一次重新加载应该发生但失败
    time_machine.move_to(start_dt + timedelta(minutes=51), tick=False)
    api_mock.load_markets = get_mock_coro(side_effect=ccxt.NetworkError("LoadError"))

    exchange.reload_markets(force=False)
    assert exchange.markets == updated_markets
    assert lam_spy.call_count == 1
    # 尝试一次，失败

    lam_spy.reset_mock()
    # 强制刷新时（机器人启动），应该重试3次
    exchange.reload_markets(force=True)
    assert lam_spy.call_count == 4
    assert exchange.markets == updated_markets


def test_reload_markets_exception(default_conf, mocker, caplog):
    """测试重新加载市场数据时的异常"""
    caplog.set_level(logging.DEBUG)

    api_mock = MagicMock()
    api_mock.load_markets = get_mock_coro(side_effect=ccxt.NetworkError("LoadError"))
    default_conf["exchange"]["markets_refresh_interval"] = 10
    exchange = get_patched_exchange(
        mocker, default_conf, api_mock, exchange="binance", mock_markets=False
    )

    exchange._last_markets_refresh = 2
    # 不到10分钟，不重新加载
    exchange.reload_markets()
    assert exchange._last_markets_refresh == 2
    assert log_has_re(r"无法加载市场数据\..*", caplog)


@pytest.mark.parametrize("stake_currency", ["ETH", "BTC", "USDT"])
def test_validate_stakecurrency(default_conf, stake_currency, mocker):
    """测试验证赌注货币"""
    default_conf["stake_currency"] = stake_currency
    api_mock = MagicMock()
    api_mock.load_markets = get_mock_coro()
    api_mock.markets = {
        "ETH/BTC": {"quote": "BTC"},
        "LTC/BTC": {"quote": "BTC"},
        "XRP/ETH": {"quote": "ETH"},
        "NEO/USDT": {"quote": "USDT"},
    }
    mocker.patch(f"{EXMS}._init_ccxt", MagicMock(return_value=api_mock))
    mocker.patch(f"{EXMS}.validate_timeframes")
    mocker.patch(f"{EXMS}.validate_pricing")
    Exchange(default_conf)


def test_validate_stakecurrency_error(default_conf, mocker):
    """测试验证赌注货币时的错误"""
    default_conf["stake_currency"] = "XRP"
    api_mock = MagicMock()
    api_mock.load_markets = get_mock_coro()
    api_mock.markets = {
        "ETH/BTC": {"quote": "BTC"},
        "LTC/BTC": {"quote": "BTC"},
        "XRP/ETH": {"quote": "ETH"},
        "NEO/USDT": {"quote": "USDT"},
    }

    mocker.patch(f"{EXMS}._init_ccxt", MagicMock(return_value=api_mock))
    mocker.patch(f"{EXMS}.validate_timeframes")
    with pytest.raises(
        ConfigurationError,
        match=r"XRP不能作为此交易所的赌注货币。可用货币为：BTC, ETH, USDT",
    ):
        Exchange(default_conf)

    api_mock.load_markets = get_mock_coro(side_effect=ccxt.NetworkError("No connection."))
    mocker.patch(f"{EXMS}._init_ccxt", MagicMock(return_value=api_mock))

    with pytest.raises(
        OperationalException, match=r"无法加载市场数据，因此无法启动。请.*"
    ):
        Exchange(default_conf)


def test_get_quote_currencies(default_conf, mocker):
    """测试获取报价货币"""
    ex = get_patched_exchange(mocker, default_conf)

    assert set(ex.get_quote_currencies()) == set(["USD", "ETH", "BTC", "USDT", "BUSD"])


@pytest.mark.parametrize(
    "pair,expected",
    [
        ("XRP/BTC", "BTC"),
        ("LTC/USD", "USD"),
        ("ETH/USDT", "USDT"),
        ("XLTCUSDT", "USDT"),
        ("XRP/NOCURRENCY", ""),
    ],
)
def test_get_pair_quote_currency(default_conf, mocker, pair, expected):
    """测试获取交易对的报价货币"""
    ex = get_patched_exchange(mocker, default_conf)
    assert ex.get_pair_quote_currency(pair) == expected


@pytest.mark.parametrize(
    "pair,expected",
    [
        ("XRP/BTC", "XRP"),
        ("LTC/USD", "LTC"),
        ("ETH/USDT", "ETH"),
        ("XLTCUSDT", "LTC"),
        ("XRP/NOCURRENCY", ""),
    ],
)
def test_get_pair_base_currency(default_conf, mocker, pair, expected):
    """测试获取交易对的基础货币"""
    ex = get_patched_exchange(mocker, default_conf)
    assert ex.get_pair_base_currency(pair) == expected


@pytest.mark.parametrize("timeframe", [("5m"), ("1m"), ("15m"), ("1h")])
def test_validate_timeframes(default_conf, mocker, timeframe):
    """测试验证时间框架"""
    default_conf["timeframe"] = timeframe
    api_mock = MagicMock()
    id_mock = PropertyMock(return_value="test_exchange")
    type(api_mock).id = id_mock
    timeframes = PropertyMock(return_value={"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h"})
    type(api_mock).timeframes = timeframes

    mocker.patch(f"{EXMS}._init_ccxt", MagicMock(return_value=api_mock))
    mocker.patch(f"{EXMS}.reload_markets")
    mocker.patch(f"{EXMS}.validate_stakecurrency")
    mocker.patch(f"{EXMS}.validate_pricing")
    Exchange(default_conf)


def test_validate_timeframes_failed(default_conf, mocker):
    """测试验证时间框架失败的情况"""
    default_conf["timeframe"] = "3m"
    api_mock = MagicMock()
    id_mock = PropertyMock(return_value="test_exchange")
    type(api_mock).id = id_mock
    timeframes = PropertyMock(
        return_value={"15s": "15s", "1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h"}
    )
    type(api_mock).timeframes = timeframes

    mocker.patch(f"{EXMS}._init_ccxt", MagicMock(return_value=api_mock))
    mocker.patch(f"{EXMS}.reload_markets")
    mocker.patch(f"{EXMS}.validate_stakecurrency")
    mocker.patch(f"{EXMS}.validate_pricing")
    with pytest.raises(
        ConfigurationError, match=r"无效的时间框架 '3m'。此交易所支持.*"
    ):
        Exchange(default_conf)
    default_conf["timeframe"] = "15s"

    with pytest.raises(
        ConfigurationError, match=r"Freqtrade目前不支持小于1分钟的时间框架。"
    ):
        Exchange(default_conf)

    # 在工具模式下不会引发异常
    default_conf["runmode"] = RunMode.UTIL_EXCHANGE
    Exchange(default_conf)


def test_validate_timeframes_emulated_ohlcv_1(default_conf, mocker):
    """测试验证模拟K线的时间框架"""
    default_conf["timeframe"] = "3m"
    api_mock = MagicMock()
    id_mock = PropertyMock(return_value="test_exchange")
    type(api_mock).id = id_mock

    # 删除时间框架，使magicmock不会自动创建它
    del api_mock.timeframes

    mocker.patch(f"{EXMS}._init_ccxt", MagicMock(return_value=api_mock))
    mocker.patch(f"{EXMS}.reload_markets")
    mocker.patch(f"{EXMS}.validate_stakecurrency")
    with pytest.raises(
        OperationalException,
        match=r"ccxt库没有提供该交易所的时间框架列表，因此该交易所不受支持。*",
    ):
        Exchange(default_conf)


def test_validate_timeframes_emulated_ohlcvi_2(default_conf, mocker):
    """测试验证模拟K线（含成交量）的时间框架"""
    default_conf["timeframe"] = "3m"
    api_mock = MagicMock()
    id_mock = PropertyMock(return_value="test_exchange")
    type(api_mock).id = id_mock

    # 删除时间框架，使magicmock不会自动创建它
    del api_mock.timeframes

    mocker.patch(f"{EXMS}._init_ccxt", MagicMock(return_value=api_mock))
    mocker.patch(f"{EXMS}.reload_markets")
    mocker.patch(f"{EXMS}.validate_stakecurrency")
    with pytest.raises(
        OperationalException,
        match=r"ccxt库没有提供该交易所的时间框架列表，因此该交易所不受支持。*",
    ):
        Exchange(default_conf)


def test_validate_timeframes_not_in_config(default_conf, mocker):
    """测试配置中没有时间框架的情况"""
    # TODO: 此测试没有断言...
    del default_conf["timeframe"]
    api_mock = MagicMock()
    id_mock = PropertyMock(return_value="test_exchange")
    type(api_mock).id = id_mock
    timeframes = PropertyMock(return_value={"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h"})
    type(api_mock).timeframes = timeframes

    mocker.patch(f"{EXMS}._init_ccxt", MagicMock(return_value=api_mock))
    mocker.patch(f"{EXMS}.reload_markets")
    mocker.patch(f"{EXMS}.validate_stakecurrency")
    mocker.patch(f"{EXMS}.validate_pricing")
    mocker.patch(f"{EXMS}.validate_required_startup_candles")
    Exchange(default_conf)


def test_validate_pricing(default_conf, mocker):
    """测试验证定价"""
    api_mock = MagicMock()
    has = {
        "fetchL2OrderBook": True,
        "fetchTicker": True,
    }
    type(api_mock).has = PropertyMock(return_value=has)
    mocker.patch(f"{EXMS}._init_ccxt", MagicMock(return_value=api_mock))
    mocker.patch(f"{EXMS}.reload_markets")
    mocker.patch(f"{EXMS}.validate_trading_mode_and_margin_mode")
    mocker.patch(f"{EXMS}.validate_timeframes")
    mocker.patch(f"{EXMS}.validate_stakecurrency")
    mocker.patch(f"{EXMS}.name", "Binance")
    default_conf["exchange"]["name"] = "binance"
    ExchangeResolver.load_exchange(default_conf)
    has.update({"fetchTicker": False})
    with pytest.raises(OperationalException, match="此交易所不支持Ticker定价.*"):
        ExchangeResolver.load_exchange(default_conf)

    has.update({"fetchTicker": True})

    default_conf["exit_pricing"]["use_order_book"] = True
    ExchangeResolver.load_exchange(default_conf)
    has.update({"fetchL2OrderBook": False})

    with pytest.raises(OperationalException, match="此交易所不支持订单簿.*"):
        ExchangeResolver.load_exchange(default_conf)

    has.update({"fetchL2OrderBook": True})

    # Binance期货没有tickers
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED

    with pytest.raises(OperationalException, match="此交易所不支持Ticker定价.*"):
        ExchangeResolver.load_exchange(default_conf)


def test_validate_ordertypes(default_conf, mocker):
    """测试验证订单类型"""
    api_mock = MagicMock()

    type(api_mock).has = PropertyMock(return_value={"createMarketOrder": True})
    mocker.patch(f"{EXMS}._init_ccxt", MagicMock(return_value=api_mock))
    mocker.patch(f"{EXMS}.reload_markets")
    mocker.patch(f"{EXMS}.validate_timeframes")
    mocker.patch(f"{EXMS}.validate_stakecurrency")
    mocker.patch(f"{EXMS}.validate_pricing")

    default_conf["order_types"] = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    Exchange(default_conf)

    type(api_mock).has = PropertyMock(return_value={"createMarketOrder": False})
    mocker.patch(f"{EXMS}._init_ccxt", MagicMock(return_value=api_mock))

    default_conf["order_types"] = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    with pytest.raises(OperationalException, match=r"交易所.*不支持市价订单。"):
        Exchange(default_conf)

    default_conf["order_types"] = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "limit",
        "stoploss_on_exchange": True,
    }
    with pytest.raises(OperationalException, match=r"交易所不支持止损订单。"):
        Exchange(default_conf)


@pytest.mark.parametrize(
    "exchange_name,stopadv, expected",
    [
        ("binance", "last", True),
        ("binance", "mark", True),
        ("binance", "index", False),
        ("bybit", "last", True),
        ("bybit", "mark", True),
        ("bybit", "index", True),
        ("okx", "last", True),
        ("okx", "mark", True),
        ("okx", "index", True),
        ("gate", "last", True),
        ("gate", "mark", True),
        ("gate", "index", True),
    ],
)
def test_validate_ordertypes_stop_advanced(default_conf, mocker, exchange_name, stopadv, expected):
    """测试验证高级止损订单类型"""
    api_mock = MagicMock()
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    type(api_mock).has = PropertyMock(return_value={"createMarketOrder": True})
    mocker.patch(f"{EXMS}._init_ccxt", MagicMock(return_value=api_mock))
    mocker.patch(f"{EXMS}.reload_markets")
    mocker.patch(f"{EXMS}.validate_timeframes")
    mocker.patch(f"{EXMS}.validate_stakecurrency")
    mocker.patch(f"{EXMS}.validate_pricing")
    default_conf["order_types"] = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "limit",
        "stoploss_on_exchange": True,
        "stoploss_price_type": stopadv,
    }
    default_conf["exchange"]["name"] = exchange_name
    if expected:
        ExchangeResolver.load_exchange(default_conf)
    else:
        with pytest.raises(
            OperationalException, match=r"交易所不支持止损价格类型.*"
        ):
            ExchangeResolver.load_exchange(default_conf)


def test_validate_order_types_not_in_config(default_conf, mocker):
    """测试配置中没有订单类型的情况"""
    api_mock = MagicMock()
    mocker.patch(f"{EXMS}._init_ccxt", MagicMock(return_value=api_mock))
    mocker.patch(f"{EXMS}.reload_markets")
    mocker.patch(f"{EXMS}.validate_timeframes")
    mocker.patch(f"{EXMS}.validate_timeframes")
    mocker.patch(f"{EXMS}.validate_pricing")
    mocker.patch(f"{EXMS}.validate_stakecurrency")

    conf = copy.deepcopy(default_conf)
    Exchange(conf)


def test_validate_required_startup_candles(default_conf, mocker, caplog):
    """测试验证所需的启动K线数量"""
    api_mock = MagicMock()
    mocker.patch(f"{EXMS}.name", PropertyMock(return_value="Binance"))

    mocker.patch(f"{EXMS}._init_ccxt", api_mock)
    mocker.patch(f"{EXMS}.validate_timeframes")
    mocker.patch(f"{EXMS}._load_async_markets")
    mocker.patch(f"{EXMS}.validate_pricing")
    mocker.patch(f"{EXMS}.validate_stakecurrency")

    default_conf["startup_candle_count"] = 20
    ex = Exchange(default_conf)
    assert ex
    # 假设交易所每次调用提供500根K线
    assert ex.validate_required_startup_candles(200, "5m") == 1
    assert ex.validate_required_startup_candles(499, "5m") == 1
    assert ex.validate_required_startup_candles(600, "5m") == 2
    assert ex.validate_required_startup_candles(501, "5m") == 2
    assert ex.validate_required_startup_candles(499, "5m") == 1
    assert ex.validate_required_startup_candles(1000, "5m") == 3
    assert ex.validate_required_startup_candles(2499, "5m") == 5
    assert log_has_re(r"使用5次调用来获取K线数据。这.*", caplog)
    with pytest.raises(OperationalException, match=r"此策略需要 2500.*"):
        ex.validate_required_startup_candles(2500, "5m")

    # 确保在初始化时也会发生同样的情况
    default_conf["startup_candle_count"] = 6000
    with pytest.raises(OperationalException, match=r"此策略需要 6000.*"):
        Exchange(default_conf)

    # 模拟kraken模式
    ex._ft_has["ohlcv_has_history"] = False
    with pytest.raises(
        OperationalException,
        match=r"此策略需要 2500.*，这超过了可用数量.*",
    ):
        ex.validate_required_startup_candles(2500, "5m")


def test_exchange_has(default_conf, mocker):
    exchange = get_patched_exchange(mocker, default_conf)
    assert not exchange.exchange_has("ASDFASDF")
    api_mock = MagicMock()

    type(api_mock).has = PropertyMock(return_value={"deadbeef": True})
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    assert exchange.exchange_has("deadbeef")

    type(api_mock).has = PropertyMock(return_value={"deadbeef": False})
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    assert not exchange.exchange_has("deadbeef")

    exchange._ft_has["exchange_has_overrides"] = {"deadbeef": True}
    assert exchange.exchange_has("deadbeef")


@pytest.mark.parametrize(
    "side,leverage",
    [
        ("buy", 1),
        ("buy", 5),
        ("sell", 1.0),
        ("sell", 5.0),
    ],
)
@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_create_dry_run_order(default_conf, mocker, side, exchange_name, leverage):
    default_conf["dry_run"] = True
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)

    order = exchange.create_dry_run_order(
        pair="ETH/BTC", ordertype="limit", side=side, amount=1, rate=200, leverage=leverage
    )
    assert "id" in order
    assert f"dry_run_{side}_" in order["id"]
    assert order["side"] == side
    assert order["type"] == "limit"
    assert order["symbol"] == "ETH/BTC"
    assert order["amount"] == 1
    assert order["cost"] == 1 * 200


@pytest.mark.parametrize(
    "side,is_short,order_reason",
    [
        ("buy", False, "entry"),
        ("sell", False, "exit"),
        ("buy", True, "exit"),
        ("sell", True, "entry"),
    ],
)
@pytest.mark.parametrize(
    "order_type,price_side,fee",
    [
        ("limit", "same", 1.0),
        ("limit", "other", 2.0),
        ("market", "same", 2.0),
        ("market", "other", 2.0),
    ],
)
def test_create_dry_run_order_fees(
    default_conf,
    mocker,
    side,
    order_type,
    is_short,
    order_reason,
    price_side,
    fee,
):
    exchange = get_patched_exchange(mocker, default_conf)
    mocker.patch(
        f"{EXMS}.get_fee",
        side_effect=lambda symbol, taker_or_maker: 2.0 if taker_or_maker == "taker" else 1.0,
    )
    mocker.patch(f"{EXMS}._dry_is_price_crossed", return_value=price_side == "other")

    order = exchange.create_dry_run_order(
        pair="LTC/USDT", ordertype=order_type, side=side, amount=10, rate=2.0, leverage=1.0
    )
    if price_side == "other" or order_type == "market":
        assert order["fee"]["rate"] == fee
        return
    else:
        assert order["fee"] is None

    mocker.patch(f"{EXMS}._dry_is_price_crossed", return_value=price_side != "other")

    order1 = exchange.fetch_dry_run_order(order["id"])
    assert order1["fee"]["rate"] == fee


@pytest.mark.parametrize(
    "side,price,filled,converted",
    [
        # 订单簿l2_usd点差:
        # 最佳卖价: 25.566
        # 最佳买价: 25.563
        ("buy", 25.563, False, False),
        ("buy", 25.566, True, False),
        ("sell", 25.566, False, False),
        ("sell", 25.563, True, False),
        ("buy", 29.563, True, True),
        ("sell", 21.563, True, True),
    ],
)
@pytest.mark.parametrize("leverage", [1, 2, 5])
@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_create_dry_run_order_limit_fill(
    default_conf,
    mocker,
    side,
    price,
    filled,
    caplog,
    exchange_name,
    order_book_l2_usd,
    converted,
    leverage,
):
    default_conf["dry_run"] = True
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    mocker.patch.multiple(
        EXMS,
        exchange_has=MagicMock(return_value=True),
        fetch_l2_order_book=order_book_l2_usd,
    )

    order = exchange.create_order(
        pair="LTC/USDT",
        ordertype="limit",
        side=side,
        amount=1,
        rate=price,
        leverage=leverage,
    )
    assert order_book_l2_usd.call_count == 1
    assert "id" in order
    assert f"dry_run_{side}_" in order["id"]
    assert order["side"] == side
    if not converted:
        assert order["average"] == price
        assert order["type"] == "limit"
    else:
        # 转换为市价单
        assert order["type"] == "market"
        assert 25.5 < order["average"] < 25.6
        assert log_has_re(r"已将 .* 转换为市价单.*", caplog)

    assert order["symbol"] == "LTC/USDT"
    assert order["status"] == "open" if not filled else "closed"
    order_book_l2_usd.reset_mock()

    # 再次获取订单...
    order_closed = exchange.fetch_dry_run_order(order["id"])
    assert order_book_l2_usd.call_count == (1 if not filled else 0)
    assert order_closed["status"] == ("open" if not filled else "closed")
    assert order_closed["filled"] == (0 if not filled else 1)
    assert order_closed["cost"] == 1 * order_closed["average"]

    order_book_l2_usd.reset_mock()

    # 空订单簿测试
    mocker.patch(f"{EXMS}.fetch_l2_order_book", return_value={"asks": [], "bids": []})
    exchange._dry_run_open_orders[order["id"]]["status"] = "open"
    order_closed = exchange.fetch_dry_run_order(order["id"])


@pytest.mark.parametrize(
    "side,rate,amount,endprice",
    [
        # 点差为 25.263-25.266
        ("buy", 25.564, 1, 25.566),
        ("buy", 25.564, 100, 25.5672),  # 需要插值
        ("buy", 25.590, 100, 25.5672),  # 价格高于点差...平均值较低
        ("buy", 25.564, 1000, 25.575),  # 超过订单簿返回量
        ("buy", 24.000, 100000, 25.200),  # 触及5%的最大滑点
        ("sell", 25.564, 1, 25.563),
        ("sell", 25.564, 100, 25.5625),  # 需要插值
        ("sell", 25.510, 100, 25.5625),  # 价格低于点差 - 平均值较高
        ("sell", 25.564, 1000, 25.5555),  # 超过订单簿返回量
        ("sell", 27, 10000, 25.65),  # 最大滑点5%
    ],
)
@pytest.mark.parametrize("leverage", [1, 2, 5])
@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_create_dry_run_order_market_fill(
    default_conf, mocker, side, rate, amount, endprice, exchange_name, order_book_l2_usd, leverage
):
    default_conf["dry_run"] = True
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    mocker.patch.multiple(
        EXMS,
        exchange_has=MagicMock(return_value=True),
        fetch_l2_order_book=order_book_l2_usd,
    )

    order = exchange.create_order(
        pair="LTC/USDT",
        ordertype="market",
        side=side,
        amount=amount,
        rate=rate,
        leverage=leverage,
    )
    assert "id" in order
    assert f"dry_run_{side}_" in order["id"]
    assert order["side"] == side
    assert order["type"] == "market"
    assert order["symbol"] == "LTC/USDT"
    assert order["status"] == "closed"
    assert order["filled"] == amount
    assert order["amount"] == amount
    assert pytest.approx(order["cost"]) == amount * order["average"]
    assert round(order["average"], 4) == round(endprice, 4)


@pytest.mark.parametrize("side", ["buy", "sell"])
@pytest.mark.parametrize(
    "ordertype,rate,marketprice",
    [
        ("market", None, None),
        ("market", 200, True),
        ("limit", 200, None),
        ("stop_loss_limit", 200, None),
    ],
)
@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_create_order(default_conf, mocker, side, ordertype, rate, marketprice, exchange_name):
    api_mock = MagicMock()
    order_id = f"test_prod_{side}_{randint(0, 10**6)}"
    api_mock.options = {} if not marketprice else {"createMarketBuyOrderRequiresPrice": True}
    api_mock.create_order = MagicMock(
        return_value={"id": order_id, "info": {"foo": "bar"}, "symbol": "XLTCUSDT", "amount": 1}
    )
    default_conf["dry_run"] = False
    default_conf["margin_mode"] = "isolated"
    mocker.patch(f"{EXMS}.amount_to_precision", lambda s, x, y: y)
    mocker.patch(f"{EXMS}.price_to_precision", lambda s, x, y: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    exchange._set_leverage = MagicMock()
    exchange.set_margin_mode = MagicMock()

    # 仅适用于gate
    price_req = exchange._ft_has.get("marketOrderRequiresPrice", False)

    order = exchange.create_order(
        pair="XLTCUSDT", ordertype=ordertype, side=side, amount=1, rate=rate, leverage=1.0
    )

    assert "id" in order
    assert "info" in order
    assert order["id"] == order_id
    assert order["amount"] == 1
    assert api_mock.create_order.call_args[0][0] == "XLTCUSDT"
    assert api_mock.create_order.call_args[0][1] == ordertype
    assert api_mock.create_order.call_args[0][2] == side
    assert api_mock.create_order.call_args[0][3] == 1
    assert api_mock.create_order.call_args[0][4] == (
        rate if price_req or not (bool(marketprice) and side == "sell") else None
    )
    assert exchange._set_leverage.call_count == 0
    assert exchange.set_margin_mode.call_count == 0

    api_mock.create_order = MagicMock(
        return_value={
            "id": order_id,
            "info": {"foo": "bar"},
            "symbol": "ADA/USDT:USDT",
            "amount": 1,
        }
    )
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    exchange.trading_mode = TradingMode.FUTURES
    exchange._set_leverage = MagicMock()
    exchange.set_margin_mode = MagicMock()
    order = exchange.create_order(
        pair="ADA/USDT:USDT", ordertype=ordertype, side=side, amount=1, rate=200, leverage=3.0
    )

    if exchange_name != "okx":
        assert exchange._set_leverage.call_count == 1
        assert exchange.set_margin_mode.call_count == 1
    else:
        assert api_mock.set_leverage.call_count == 1
    assert order["amount"] == 0.01


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_buy_dry_run(default_conf, mocker, exchange_name):
    default_conf["dry_run"] = True
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)

    order = exchange.create_order(
        pair="ETH/BTC",
        ordertype="limit",
        side="buy",
        amount=1,
        rate=200,
        leverage=1.0,
        time_in_force="gtc",
    )
    assert "id" in order
    assert "dry_run_buy_" in order["id"]


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_buy_prod(default_conf, mocker, exchange_name):
    api_mock = MagicMock()
    order_id = f"test_prod_buy_{randint(0, 10**6)}"
    order_type = "market"
    time_in_force = "gtc"
    api_mock.options = {}
    api_mock.create_order = MagicMock(
        return_value={"id": order_id, "symbol": "ETH/BTC", "info": {"foo": "bar"}}
    )
    default_conf["dry_run"] = False
    mocker.patch(f"{EXMS}.amount_to_precision", lambda s, x, y: y)
    mocker.patch(f"{EXMS}.price_to_precision", lambda s, x, y: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)

    order = exchange.create_order(
        pair="ETH/BTC",
        ordertype=order_type,
        side="buy",
        amount=1,
        rate=200,
        leverage=1.0,
        time_in_force=time_in_force,
    )

    assert "id" in order
    assert "info" in order
    assert order["id"] == order_id
    assert api_mock.create_order.call_args[0][0] == "ETH/BTC"
    assert api_mock.create_order.call_args[0][1] == order_type
    assert api_mock.create_order.call_args[0][2] == "buy"
    assert api_mock.create_order.call_args[0][3] == 1
    if exchange._order_needs_price("buy", order_type):
        assert api_mock.create_order.call_args[0][4] == 200
    else:
        assert api_mock.create_order.call_args[0][4] is None

    api_mock.create_order.reset_mock()
    order_type = "limit"
    order = exchange.create_order(
        pair="ETH/BTC",
        ordertype=order_type,
        side="buy",
        amount=1,
        rate=200,
        leverage=1.0,
        time_in_force=time_in_force,
    )
    assert api_mock.create_order.call_args[0][0] == "ETH/BTC"
    assert api_mock.create_order.call_args[0][1] == order_type
    assert api_mock.create_order.call_args[0][2] == "buy"
    assert api_mock.create_order.call_args[0][3] == 1
    assert api_mock.create_order.call_args[0][4] == 200

    # 测试异常处理
    with pytest.raises(DependencyException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InsufficientFunds("资金不足"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.create_order(
            pair="ETH/BTC",
            ordertype=order_type,
            side="buy",
            amount=1,
            rate=200,
            leverage=1.0,
            time_in_force=time_in_force,
        )

    with pytest.raises(DependencyException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InvalidOrder("订单未找到"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.create_order(
            pair="ETH/BTC",
            ordertype="limit",
            side="buy",
            amount=1,
            rate=200,
            leverage=1.0,
            time_in_force=time_in_force,
        )

    with pytest.raises(DependencyException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InvalidOrder("订单未找到"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.create_order(
            pair="ETH/BTC",
            ordertype="market",
            side="buy",
            amount=1,
            rate=200,
            leverage=1.0,
            time_in_force=time_in_force,
        )

    with pytest.raises(TemporaryError):
        api_mock.create_order = MagicMock(side_effect=ccxt.NetworkError("网络断开连接"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.create_order(
            pair="ETH/BTC",
            ordertype=order_type,
            side="buy",
            amount=1,
            rate=200,
            leverage=1.0,
            time_in_force=time_in_force,
        )

    with pytest.raises(OperationalException):
        api_mock.create_order = MagicMock(side_effect=ccxt.BaseError("未知错误"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.create_order(
            pair="ETH/BTC",
            ordertype=order_type,
            side="buy",
            amount=1,
            rate=200,
            leverage=1.0,
            time_in_force=time_in_force,
        )


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_buy_considers_time_in_force(default_conf, mocker, exchange_name):
    api_mock = MagicMock()
    order_id = f"test_prod_buy_{randint(0, 10**6)}"
    api_mock.options = {}
    api_mock.create_order = MagicMock(
        return_value={"id": order_id, "symbol": "ETH/BTC", "info": {"foo": "bar"}}
    )
    default_conf["dry_run"] = False
    mocker.patch(f"{EXMS}.amount_to_precision", lambda s, x, y: y)
    mocker.patch(f"{EXMS}.price_to_precision", lambda s, x, y: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)

    order_type = "limit"
    time_in_force = "ioc"

    order = exchange.create_order(
        pair="ETH/BTC",
        ordertype=order_type,
        side="buy",
        amount=1,
        rate=200,
        leverage=1.0,
        time_in_force=time_in_force,
    )

    assert "id" in order
    assert "info" in order
    assert order["status"] == "open"
    assert order["id"] == order_id
    assert api_mock.create_order.call_args[0][0] == "ETH/BTC"
    assert api_mock.create_order.call_args[0][1] == order_type
    assert api_mock.create_order.call_args[0][2] == "buy"
    assert api_mock.create_order.call_args[0][3] == 1
    assert api_mock.create_order.call_args[0][4] == 200
    assert "timeInForce" in api_mock.create_order.call_args[0][5]
    assert api_mock.create_order.call_args[0][5]["timeInForce"] == time_in_force.upper()

    order_type = "market"
    time_in_force = "ioc"

    order = exchange.create_order(
        pair="ETH/BTC",
        ordertype=order_type,
        side="buy",
        amount=1,
        rate=200,
        leverage=1.0,
        time_in_force=time_in_force,
    )

    assert "id" in order
    assert "info" in order
    assert order["id"] == order_id
    assert api_mock.create_order.call_args[0][0] == "ETH/BTC"
    assert api_mock.create_order.call_args[0][1] == order_type
    assert api_mock.create_order.call_args[0][2] == "buy"
    assert api_mock.create_order.call_args[0][3] == 1
    if exchange._order_needs_price("buy", order_type):
        assert api_mock.create_order.call_args[0][4] == 200
    else:
        assert api_mock.create_order.call_args[0][4] is None
    # 市价单不应发送timeInForce!!
    assert "timeInForce" not in api_mock.create_order.call_args[0][5]


def test_sell_dry_run(default_conf, mocker):
    default_conf["dry_run"] = True
    exchange = get_patched_exchange(mocker, default_conf)

    order = exchange.create_order(
        pair="ETH/BTC", ordertype="limit", side="sell", amount=1, rate=200, leverage=1.0
    )
    assert "id" in order
    assert "dry_run_sell_" in order["id"]


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_sell_prod(default_conf, mocker, exchange_name):
    api_mock = MagicMock()
    order_id = f"test_prod_sell_{randint(0, 10**6)}"
    order_type = "market"
    api_mock.options = {}
    api_mock.create_order = MagicMock(
        return_value={"id": order_id, "symbol": "ETH/BTC", "info": {"foo": "bar"}}
    )
    default_conf["dry_run"] = False

    mocker.patch(f"{EXMS}.amount_to_precision", lambda s, x, y: y)
    mocker.patch(f"{EXMS}.price_to_precision", lambda s, x, y: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)

    order = exchange.create_order(
        pair="ETH/BTC", ordertype=order_type, side="sell", amount=1, rate=200, leverage=1.0
    )

    assert "id" in order
    assert "info" in order
    assert order["id"] == order_id
    assert api_mock.create_order.call_args[0][0] == "ETH/BTC"
    assert api_mock.create_order.call_args[0][1] == order_type
    assert api_mock.create_order.call_args[0][2] == "sell"
    assert api_mock.create_order.call_args[0][3] == 1
    if exchange._order_needs_price("sell", order_type):
        assert api_mock.create_order.call_args[0][4] == 200
    else:
        assert api_mock.create_order.call_args[0][4] is None

    api_mock.create_order.reset_mock()
    order_type = "limit"
    order = exchange.create_order(
        pair="ETH/BTC", ordertype=order_type, side="sell", amount=1, rate=200, leverage=1.0
    )
    assert api_mock.create_order.call_args[0][0] == "ETH/BTC"
    assert api_mock.create_order.call_args[0][1] == order_type
    assert api_mock.create_order.call_args[0][2] == "sell"
    assert api_mock.create_order.call_args[0][3] == 1
    assert api_mock.create_order.call_args[0][4] == 200

    # 测试异常处理
    with pytest.raises(InsufficientFundsError):
        api_mock.create_order = MagicMock(side_effect=ccxt.InsufficientFunds("余额为0"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.create_order(
            pair="ETH/BTC", ordertype=order_type, side="sell", amount=1, rate=200, leverage=1.0
        )

    with pytest.raises(InvalidOrderException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InvalidOrder("订单未找到"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.create_order(
            pair="ETH/BTC", ordertype="limit", side="sell", amount=1, rate=200, leverage=1.0
        )

    # 市价单不需要价格，因此行为略有不同
    with pytest.raises(DependencyException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InvalidOrder("订单未找到"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.create_order(
            pair="ETH/BTC", ordertype="market", side="sell", amount=1, rate=200, leverage=1.0
        )

    with pytest.raises(TemporaryError):
        api_mock.create_order = MagicMock(side_effect=ccxt.NetworkError("无连接"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.create_order(
            pair="ETH/BTC", ordertype=order_type, side="sell", amount=1, rate=200, leverage=1.0
        )

    with pytest.raises(OperationalException):
        api_mock.create_order = MagicMock(side_effect=ccxt.BaseError("DeadBeef"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.create_order(
            pair="ETH/BTC", ordertype=order_type, side="sell", amount=1, rate=200, leverage=1.0
        )


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_sell_considers_time_in_force(default_conf, mocker, exchange_name):
    api_mock = MagicMock()
    order_id = f"test_prod_sell_{randint(0, 10**6)}"
    api_mock.create_order = MagicMock(
        return_value={"id": order_id, "symbol": "ETH/BTC", "info": {"foo": "bar"}}
    )
    api_mock.options = {}
    default_conf["dry_run"] = False
    mocker.patch(f"{EXMS}.amount_to_precision", lambda s, x, y: y)
    mocker.patch(f"{EXMS}.price_to_precision", lambda s, x, y: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)

    order_type = "limit"
    time_in_force = "ioc"

    order = exchange.create_order(
        pair="ETH/BTC",
        ordertype=order_type,
        side="sell",
        amount=1,
        rate=200,
        leverage=1.0,
        time_in_force=time_in_force,
    )

    assert "id" in order
    assert "info" in order
    assert order["id"] == order_id
    assert api_mock.create_order.call_args[0][0] == "ETH/BTC"
    assert api_mock.create_order.call_args[0][1] == order_type
    assert api_mock.create_order.call_args[0][2] == "sell"
    assert api_mock.create_order.call_args[0][3] == 1
    assert api_mock.create_order.call_args[0][4] == 200
    assert "timeInForce" in api_mock.create_order.call_args[0][5]
    assert api_mock.create_order.call_args[0][5]["timeInForce"] == time_in_force.upper()

    order_type = "market"
    time_in_force = "IOC"
    order = exchange.create_order(
        pair="ETH/BTC",
        ordertype=order_type,
        side="sell",
        amount=1,
        rate=200,
        leverage=1.0,
        time_in_force=time_in_force,
    )

    assert "id" in order
    assert "info" in order
    assert order["id"] == order_id
    assert api_mock.create_order.call_args[0][0] == "ETH/BTC"
    assert api_mock.create_order.call_args[0][1] == order_type
    assert api_mock.create_order.call_args[0][2] == "sell"
    assert api_mock.create_order.call_args[0][3] == 1
    if exchange._order_needs_price("sell", order_type):
        assert api_mock.create_order.call_args[0][4] == 200
    else:
        assert api_mock.create_order.call_args[0][4] is None
    # 市价单不应发送timeInForce!!
    assert "timeInForce" not in api_mock.create_order.call_args[0][5]


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_get_balances_prod(default_conf, mocker, exchange_name):
    balance_item = {"free": 10.0, "total": 10.0, "used": 0.0}

    api_mock = MagicMock()
    api_mock.fetch_balance = MagicMock(
        return_value={"1ST": balance_item, "2ND": balance_item, "3RD": balance_item}
    )
    api_mock.commonCurrencies = {}
    default_conf["dry_run"] = False
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    assert len(exchange.get_balances()) == 3
    assert exchange.get_balances()["1ST"]["free"] == 10.0
    assert exchange.get_balances()["1ST"]["total"] == 10.0
    assert exchange.get_balances()["1ST"]["used"] == 0.0

    ccxt_exceptionhandlers(
        mocker, default_conf, api_mock, exchange_name, "get_balances", "fetch_balance"
    )


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_fetch_positions(default_conf, mocker, exchange_name):
    mocker.patch(f"{EXMS}.validate_trading_mode_and_margin_mode")
    api_mock = MagicMock()
    api_mock.fetch_positions = MagicMock(
        return_value=[
            {"symbol": "ETH/USDT:USDT", "leverage": 5},
            {"symbol": "XRP/USDT:USDT", "leverage": 5},
        ]
    )
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    assert exchange.fetch_positions() == []
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = "futures"

    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    res = exchange.fetch_positions()
    assert len(res) == 2

    ccxt_exceptionhandlers(
        mocker, default_conf, api_mock, exchange_name, "fetch_positions", "fetch_positions"
    )


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_fetch_orders(default_conf, mocker, exchange_name, limit_order):
    api_mock = MagicMock()
    call_count = 1

    def return_value(*args, **kwargs):
        nonlocal call_count
        call_count += 2
        return [
            {**limit_order["buy"], "id": call_count},
            {**limit_order["sell"], "id": call_count + 1},
        ]

    api_mock.fetch_orders = MagicMock(side_effect=return_value)
    api_mock.fetch_open_orders = MagicMock(return_value=[limit_order["buy"]])
    api_mock.fetch_closed_orders = MagicMock(return_value=[limit_order["buy"]])

    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    start_time = datetime.now(timezone.utc) - timedelta(days=20)
    expected = 1
    if exchange_name == "bybit":
        expected = 3

    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    # 模拟环境中不可用
    assert exchange.fetch_orders("mocked", start_time) == []
    assert api_mock.fetch_orders.call_count == 0
    default_conf["dry_run"] = False

    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    res = exchange.fetch_orders("mocked", start_time)
    assert api_mock.fetch_orders.call_count == expected
    assert api_mock.fetch_open_orders.call_count == 0
    assert api_mock.fetch_closed_orders.call_count == 0
    assert len(res) == 2 * expected

    res = exchange.fetch_orders("mocked", start_time)

    api_mock.fetch_orders.reset_mock()

    def has_resp(_, endpoint):
        if endpoint == "fetchOrders":
            return False
        if endpoint == "fetchClosedOrders":
            return True
        if endpoint == "fetchOpenOrders":
            return True

    if exchange_name == "okx":
        # OKX特殊情况单独测试
        return

    mocker.patch(f"{EXMS}.exchange_has", has_resp)

    # 没有fetchOrders的正常路径
    exchange.fetch_orders("mocked", start_time)
    assert api_mock.fetch_orders.call_count == 0
    assert api_mock.fetch_open_orders.call_count == expected
    assert api_mock.fetch_closed_orders.call_count == expected

    mocker.patch(f"{EXMS}.exchange_has", return_value=True)

    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        api_mock,
        exchange_name,
        "fetch_orders",
        "fetch_orders",
        retries=1,
        pair="mocked",
        since=start_time,
    )

    # 异常路径 - 首次fetch-orders调用失败。
    api_mock.fetch_orders = MagicMock(side_effect=ccxt.NotSupported())
    api_mock.fetch_open_orders.reset_mock()
    api_mock.fetch_closed_orders.reset_mock()

    exchange.fetch_orders("mocked", start_time)

    assert api_mock.fetch_orders.call_count == expected
    assert api_mock.fetch_open_orders.call_count == expected
    assert api_mock.fetch_closed_orders.call_count == expected


def test_fetch_trading_fees(default_conf, mocker):
    api_mock = MagicMock()
    tick = {
        "1INCH/USDT:USDT": {
            "info": {
                "user_id": "",
                "taker_fee": "0.0018",
                "maker_fee": "0.0018",
                "gt_discount": False,
                "gt_taker_fee": "0",
                "gt_maker_fee": "0",
                "loan_fee": "0.18",
                "point_type": "1",
                "futures_taker_fee": "0.0005",
                "futures_maker_fee": "0",
            },
            "symbol": "1INCH/USDT:USDT",
            "maker": 0.0,
            "taker": 0.0005,
        },
        "ETH/USDT:USDT": {
            "info": {
                "user_id": "",
                "taker_fee": "0.0018",
                "maker_fee": "0.0018",
                "gt_discount": False,
                "gt_taker_fee": "0",
                "gt_maker_fee": "0",
                "loan_fee": "0.18",
                "point_type": "1",
                "futures_taker_fee": "0.0005",
                "futures_maker_fee": "0",
            },
            "symbol": "ETH/USDT:USDT",
            "maker": 0.0,
            "taker": 0.0005,
        },
    }
    exchange_name = "gate"
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    api_mock.fetch_trading_fees = MagicMock(return_value=tick)
    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)

    assert "1INCH/USDT:USDT" in exchange._trading_fees
    assert "ETH/USDT:USDT" in exchange._trading_fees
    assert api_mock.fetch_trading_fees.call_count == 1

    api_mock.fetch_trading_fees.reset_mock()
    # Reload-markets也会调用fetch_trading_fees，因此下面异常测试中的显式调用会被调用两次。
    mocker.patch(f"{EXMS}.reload_markets")
    ccxt_exceptionhandlers(
        mocker, default_conf, api_mock, exchange_name, "fetch_trading_fees", "fetch_trading_fees"
    )

    api_mock.fetch_trading_fees = MagicMock(return_value={})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    exchange.fetch_trading_fees()
    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    assert exchange.fetch_trading_fees() == {}


def test_fetch_bids_asks(default_conf, mocker):
    api_mock = MagicMock()
    tick = {
        "ETH/BTC": {
            "symbol": "ETH/BTC",
            "bid": 0.5,
            "ask": 1,
            "last": 42,
        },
        "BCH/BTC": {
            "symbol": "BCH/BTC",
            "bid": 0.6,
            "ask": 0.5,
            "last": 41,
        },
    }
    exchange_name = "binance"
    api_mock.fetch_bids_asks = MagicMock(return_value=tick)
    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    # 获取原始报价
    bidsasks = exchange.fetch_bids_asks()

    assert "ETH/BTC" in bidsasks
    assert "BCH/BTC" in bidsasks
    assert bidsasks["ETH/BTC"]["bid"] == 0.5
    assert bidsasks["ETH/BTC"]["ask"] == 1
    assert bidsasks["BCH/BTC"]["bid"] == 0.6
    assert bidsasks["BCH/BTC"]["ask"] == 0.5
    assert api_mock.fetch_bids_asks.call_count == 1

    api_mock.fetch_bids_asks.reset_mock()

    # 缓存的报价不应再次调用API
    tickers2 = exchange.fetch_bids_asks(cached=True)
    assert tickers2 == bidsasks
    assert api_mock.fetch_bids_asks.call_count == 0
    tickers2 = exchange.fetch_bids_asks(cached=False)
    assert api_mock.fetch_bids_asks.call_count == 1

    ccxt_exceptionhandlers(
        mocker, default_conf, api_mock, exchange_name, "fetch_bids_asks", "fetch_bids_asks"
    )

    with pytest.raises(OperationalException):
        api_mock.fetch_bids_asks = MagicMock(side_effect=ccxt.NotSupported("DeadBeef"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.fetch_bids_asks()

    api_mock.fetch_bids_asks = MagicMock(return_value={})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    exchange.fetch_bids_asks()
    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    assert exchange.fetch_bids_asks() == {}


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_get_tickers(default_conf, mocker, exchange_name, caplog):
    api_mock = MagicMock()
    tick = {
        "ETH/BTC": {
            "symbol": "ETH/BTC",
            "bid": 0.5,
            "ask": 1,
            "last": 42,
        },
        "BCH/BTC": {
            "symbol": "BCH/BTC",
            "bid": 0.6,
            "ask": 0.5,
            "last": 41,
        },
    }
    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    api_mock.fetch_tickers = MagicMock(return_value=tick)
    api_mock.fetch_bids_asks = MagicMock(return_value={})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    # 获取原始报价
    tickers = exchange.get_tickers()

    assert "ETH/BTC" in tickers
    assert "BCH/BTC" in tickers
    assert tickers["ETH/BTC"]["bid"] == 0.5
    assert tickers["ETH/BTC"]["ask"] == 1
    assert tickers["BCH/BTC"]["bid"] == 0.6
    assert tickers["BCH/BTC"]["ask"] == 0.5
    assert api_mock.fetch_tickers.call_count == 1
    assert api_mock.fetch_bids_asks.call_count == 0

    api_mock.fetch_tickers.reset_mock()

    # 缓存的报价不应再次调用API
    tickers2 = exchange.get_tickers(cached=True)
    assert tickers2 == tickers
    assert api_mock.fetch_tickers.call_count == 0
    assert api_mock.fetch_bids_asks.call_count == 0
    tickers2 = exchange.get_tickers(cached=False)
    assert api_mock.fetch_tickers.call_count == 1
    assert api_mock.fetch_bids_asks.call_count == 0

    ccxt_exceptionhandlers(
        mocker, default_conf, api_mock, exchange_name, "get_tickers", "fetch_tickers"
    )

    with pytest.raises(OperationalException):
        api_mock.fetch_tickers = MagicMock(side_effect=ccxt.NotSupported("DeadBeef"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.get_tickers()

    caplog.clear()
    api_mock.fetch_tickers = MagicMock(side_effect=[ccxt.BadSymbol("SomeSymbol"), []])
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    x = exchange.get_tickers()
    assert x == []
    assert log_has_re(r"由于BadSymbol无法加载报价..*SomeSymbol", caplog)
    caplog.clear()

    api_mock.fetch_tickers = MagicMock(return_value={})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    exchange.get_tickers()

    api_mock.fetch_tickers.reset_mock()
    api_mock.fetch_bids_asks.reset_mock()
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)

    exchange.get_tickers()
    assert api_mock.fetch_tickers.call_count == 1
    assert api_mock.fetch_bids_asks.call_count == (1 if exchange_name == "binance" else 0)

    api_mock.fetch_tickers.reset_mock()
    api_mock.fetch_bids_asks.reset_mock()
    mocker.patch(f"{EXMS}.exchange_has", return_value=False)
    assert exchange.get_tickers() == {}
@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_get_conversion_rate(default_conf_usdt, mocker, exchange_name):
    api_mock = MagicMock()
    行情数据 = {
        "ETH/USDT": {
            "last": 42,
        },
        "BCH/USDT": {
            "last": 41,
        },
        "ETH/BTC": {
            "last": 250,
        },
    }
    行情数据2 = {
        "ADA/USDT:USDT": {
            "last": 2.5,
        }
    }
    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    api_mock.fetch_tickers = MagicMock(side_effect=[行情数据, 行情数据2])
    api_mock.fetch_bids_asks = MagicMock(return_value={})
    default_conf_usdt["trading_mode"] = "futures"

    exchange = get_patched_exchange(mocker, default_conf_usdt, api_mock, exchange=exchange_name)
    # 获取原始价格数据
    assert exchange.get_conversion_rate("USDT", "USDT") == 1
    assert api_mock.fetch_tickers.call_count == 0
    assert exchange.get_conversion_rate("ETH", "USDT") == 42
    assert exchange.get_conversion_rate("ETH", "USDC") is None
    assert exchange.get_conversion_rate("ETH", "BTC") == 250
    assert exchange.get_conversion_rate("BTC", "ETH") == 0.004

    assert api_mock.fetch_tickers.call_count == 1
    api_mock.fetch_tickers.reset_mock()

    assert exchange.get_conversion_rate("ADA", "USDT") == 2.5
    # 只调用了一次"其他"市场
    assert api_mock.fetch_tickers.call_count == 1

    if exchange_name == "binance":
        # 币安特殊情况：BNFCR与USDT匹配
        assert exchange.get_conversion_rate("BNFCR", "USDT") is None
        assert exchange.get_conversion_rate("BNFCR", "USDC") == 1
        assert exchange.get_conversion_rate("USDT", "BNFCR") is None
        assert exchange.get_conversion_rate("USDC", "BNFCR") == 1


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_fetch_ticker(default_conf, mocker, exchange_name):
    api_mock = MagicMock()
    价格数据 = {
        "symbol": "ETH/BTC",
        "bid": 0.00001098,
        "ask": 0.00001099,
        "last": 0.0001,
    }
    api_mock.fetch_ticker = MagicMock(return_value=价格数据)
    api_mock.markets = {"ETH/BTC": {"active": True}}
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    # 获取原始价格数据
    ticker = exchange.fetch_ticker(pair="ETH/BTC")

    assert ticker["bid"] == 0.00001098
    assert ticker["ask"] == 0.00001099

    # 更改价格数据
    价格数据 = {
        "symbol": "ETH/BTC",
        "bid": 0.5,
        "ask": 1,
        "last": 42,
    }
    api_mock.fetch_ticker = MagicMock(return_value=价格数据)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)

    # 如果不缓存结果，我们应该得到相同的价格数据
    # 如果不获取新结果，我们应该得到缓存的价格数据
    ticker = exchange.fetch_ticker(pair="ETH/BTC")

    assert api_mock.fetch_ticker.call_count == 1
    assert ticker["bid"] == 0.5
    assert ticker["ask"] == 1

    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        api_mock,
        exchange_name,
        "fetch_ticker",
        "fetch_ticker",
        pair="ETH/BTC",
    )

    api_mock.fetch_ticker = MagicMock(return_value={})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    exchange.fetch_ticker(pair="ETH/BTC")

    with pytest.raises(DependencyException, match=r"交易对 XRP/ETH 不可用"):
        exchange.fetch_ticker(pair="XRP/ETH")


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test___now_is_time_to_refresh(default_conf, mocker, exchange_name, time_machine):
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    交易对 = "BTC/USDT"
    蜡烛类型 = CandleType.SPOT
    开始时间 = datetime(2023, 12, 1, 0, 10, 0, tzinfo=timezone.utc)
    time_machine.move_to(开始时间, tick=False)
    assert (交易对, "5m", 蜡烛类型) not in exchange._pairs_last_refresh_time

    # 尚未刷新
    assert exchange._now_is_time_to_refresh(交易对, "5m", 蜡烛类型) is True

    最后关闭的蜡烛 = dt_ts(开始时间 - timedelta(minutes=5))
    exchange._pairs_last_refresh_time[(交易对, "5m", 蜡烛类型)] = 最后关闭的蜡烛

    # 下一根蜡烛尚未关闭
    time_machine.move_to(开始时间 + timedelta(minutes=4, seconds=59), tick=False)
    assert exchange._now_is_time_to_refresh(交易对, "5m", 蜡烛类型) is False

    # 下一根蜡烛已关闭
    time_machine.move_to(开始时间 + timedelta(minutes=5, seconds=0), tick=False)
    assert exchange._now_is_time_to_refresh(交易对, "5m", 蜡烛类型) is True

    # 1秒后（last_refresh_time未改变）
    time_machine.move_to(开始时间 + timedelta(minutes=5, seconds=1), tick=False)
    assert exchange._now_is_time_to_refresh(交易对, "5m", 蜡烛类型) is True

    # 测试1天数据
    开始日时间 = datetime(2023, 12, 1, 0, 0, 0, tzinfo=timezone.utc)
    最后关闭的蜡烛_1d = dt_ts(开始日时间 - timedelta(days=1))
    exchange._pairs_last_refresh_time[(交易对, "1d", 蜡烛类型)] = 最后关闭的蜡烛_1d

    time_machine.move_to(开始日时间 - timedelta(seconds=5), tick=False)
    assert exchange._now_is_time_to_refresh(交易对, "1d", 蜡烛类型) is False

    time_machine.move_to(开始日时间 + timedelta(hours=20, seconds=5), tick=False)
    assert exchange._now_is_time_to_refresh(交易对, "1d", 蜡烛类型) is False

    # 下一根蜡烛已关闭 - 现在我们刷新
    time_machine.move_to(开始日时间 + timedelta(days=1, seconds=0), tick=False)
    assert exchange._now_is_time_to_refresh(交易对, "1d", 蜡烛类型) is True


@pytest.mark.parametrize("candle_type", ["mark", ""])
@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_get_historic_ohlcv(default_conf, mocker, caplog, exchange_name, candle_type):
    caplog.set_level(logging.DEBUG)
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    交易对 = "ETH/BTC"
    调用次数 = 0
    现在 = dt_now()

    async def mock_candle_hist(交易对, 时间周期, 蜡烛类型, 起始时间戳):
        nonlocal 调用次数
        调用次数 += 1
        ohlcv = [
            [
                dt_ts(现在 + timedelta(minutes=5 * (调用次数 + i))),  #  Unix时间戳（毫秒）
                1,  # 开盘价
                2,  # 最高价
                3,  # 最低价
                4,  # 收盘价
                5,  # 成交量（以计价货币计）
            ]
            for i in range(2)
        ]
        return (交易对, 时间周期, 蜡烛类型, ohlcv, True)

    exchange._async_get_candle_history = Mock(wraps=mock_candle_hist)
    # 一次调用计算 * 1.8 应该进行2次调用

    时长 = 5 * 60 * exchange.ohlcv_candle_limit("5m", candle_type) * 1.8
    结果 = exchange.get_historic_ohlcv(
        交易对, "5m", dt_ts(dt_now() - timedelta(seconds=时长)), candle_type=candle_type
    )

    assert exchange._async_get_candle_history.call_count == 2
    # 截断未完成的蜡烛后，返回两倍的上述OHLCV数据
    assert len(结果) == 2
    assert log_has_re(r"从ccxt下载了.*的数据，长度为.*\.", caplog)

    caplog.clear()

    exchange._async_get_candle_history = get_mock_coro(side_effect=TimeoutError())
    with pytest.raises(TimeoutError):
        exchange.get_historic_ohlcv(
            交易对, "5m", dt_ts(dt_now() - timedelta(seconds=时长)), candle_type=candle_type
        )
    assert log_has_re(r"异步代码抛出异常: .*", caplog)


@pytest.mark.parametrize("exchange_name", EXCHANGES)
@pytest.mark.parametrize("candle_type", [CandleType.MARK, CandleType.SPOT])
async def test__async_get_historic_ohlcv(default_conf, mocker, caplog, exchange_name, candle_type):
    ohlcv = [
        [
            int((datetime.now(timezone.utc).timestamp() - 1000) * 1000),
            1,  # 开盘价
            2,  # 最高价
            3,  # 最低价
            4,  # 收盘价
            5,  # 成交量（以计价货币计）
        ]
    ]
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    # 猴子补丁异步函数
    exchange._api_async.fetch_ohlcv = get_mock_coro(ohlcv)

    交易对 = "ETH/USDT"
    响应交易对, 响应时间周期, _, 响应结果, _ = await exchange._async_get_historic_ohlcv(
        交易对, "5m", 1500000000000, candle_type=candle_type
    )
    assert 响应交易对 == 交易对
    assert 响应时间周期 == "5m"
    # 使用非常旧的时间戳调用 - 导致大量请求
    assert exchange._api_async.fetch_ohlcv.call_count > 200
    assert 响应结果[0] == ohlcv[0]

    exchange._api_async.fetch_ohlcv.reset_mock()
    结束时间戳 = 1_500_500_000_000
    开始时间戳 = 1_500_000_000_000
    响应交易对, 响应时间周期, _, 响应结果, _ = await exchange._async_get_historic_ohlcv(
        交易对, "5m", since_ms=开始时间戳, candle_type=candle_type, until_ms=结束时间戳
    )
    # 所需蜡烛数
    蜡烛数 = (结束时间戳 - 开始时间戳) / 300_000
    预期调用次数 = 蜡烛数 // exchange.ohlcv_candle_limit("5m", candle_type, 开始时间戳) + 1

    # 根据交易所不同，这应该被调用1到6次之间
    assert exchange._api_async.fetch_ohlcv.call_count == 预期调用次数


@pytest.mark.parametrize("candle_type", [CandleType.FUTURES, CandleType.MARK, CandleType.SPOT])
def test_refresh_latest_ohlcv(mocker, default_conf, caplog, candle_type) -> None:
    ohlcv = [
        [
            dt_ts(dt_now() - timedelta(minutes=5)),  # Unix时间戳（毫秒）
            1,  # 开盘价
            2,  # 最高价
            3,  # 最低价
            4,  # 收盘价
            5,  # 成交量（以计价货币计）
        ],
        [
            dt_ts(),  # Unix时间戳（毫秒）
            3,  # 开盘价
            1,  # 最高价
            4,  # 最低价
            6,  # 收盘价
            5,  # 成交量（以计价货币计）
        ],
    ]

    caplog.set_level(logging.DEBUG)
    exchange = get_patched_exchange(mocker, default_conf)
    exchange._api_async.fetch_ohlcv = get_mock_coro(ohlcv)

    交易对列表 = [("IOTA/ETH", "5m", candle_type), ("XRP/ETH", "5m", candle_type)]
    # 空字典
    assert not exchange._klines
    结果 = exchange.refresh_latest_ohlcv(交易对列表, cache=False)
    # 不缓存
    assert not exchange._klines

    assert len(结果) == len(交易对列表)
    assert exchange._api_async.fetch_ohlcv.call_count == 2
    exchange._api_async.fetch_ohlcv.reset_mock()

    exchange.required_candle_call_count = 2
    结果 = exchange.refresh_latest_ohlcv(交易对列表)
    assert len(结果) == len(交易对列表)

    assert log_has(f"刷新 {len(交易对列表)} 个交易对的蜡烛图（OHLCV）数据", caplog)
    assert exchange._klines
    assert exchange._api_async.fetch_ohlcv.call_count == 4
    exchange._api_async.fetch_ohlcv.reset_mock()
    for 交易对 in 交易对列表:
        assert isinstance(exchange.klines(交易对), DataFrame)
        assert len(exchange.klines(交易对)) > 0

        # 如果copy为"True"，klines函数每次调用应返回不同的对象
        assert exchange.klines(交易对) is not exchange.klines(交易对)
        assert exchange.klines(交易对) is not exchange.klines(交易对, copy=True)
        assert exchange.klines(交易对, copy=True) is not exchange.klines(交易对, copy=True)
        assert exchange.klines(交易对, copy=False) is exchange.klines(交易对, copy=False)

    # 测试缓存
    结果 = exchange.refresh_latest_ohlcv(
        [("IOTA/ETH", "5m", candle_type), ("XRP/ETH", "5m", candle_type)]
    )
    assert len(结果) == len(交易对列表)

    assert exchange._api_async.fetch_ohlcv.call_count == 0
    assert log_has(
        f"使用缓存的蜡烛图（OHLCV）数据：{交易对列表[0][0]}, {交易对列表[0][1]}, {candle_type} ...",
        caplog,
    )
    caplog.clear()
    # 重置刷新时间 - 由于缓存过期，每个交易对应进行2次调用
    exchange._pairs_last_refresh_time = {}
    结果 = exchange.refresh_latest_ohlcv(
        [("IOTA/ETH", "5m", candle_type), ("XRP/ETH", "5m", candle_type)]
    )
    assert len(结果) == len(交易对列表)

    assert exchange._api_async.fetch_ohlcv.call_count == 4

    # 缓存 - 但禁用缓存
    exchange._api_async.fetch_ohlcv.reset_mock()
    exchange.required_candle_call_count = 1

    交易对列表 = [
        ("IOTA/ETH", "5m", candle_type),
        ("XRP/ETH", "5m", candle_type),
        ("XRP/ETH", "1d", candle_type),
    ]
    结果 = exchange.refresh_latest_ohlcv(交易对列表, cache=False)
    assert len(结果) == 3
    assert exchange._api_async.fetch_ohlcv.call_count == 3

    # 再次测试，不应从缓存返回！
    exchange._api_async.fetch_ohlcv.reset_mock()
    结果 = exchange.refresh_latest_ohlcv(交易对列表, cache=False)
    assert len(结果) == 3
    assert exchange._api_async.fetch_ohlcv.call_count == 3
    exchange._api_async.fetch_ohlcv.reset_mock()
    caplog.clear()

    # 使用无效时间周期调用
    结果 = exchange.refresh_latest_ohlcv([("IOTA/ETH", "3m", candle_type)], cache=False)
    if candle_type != CandleType.MARK:
        assert not 结果
        assert len(结果) == 0
        assert log_has_re(r"无法下载 \(IOTA\/ETH, 3m\).*", caplog)
    else:
        assert len(结果) == 1


@pytest.mark.parametrize("candle_type", [CandleType.FUTURES, CandleType.SPOT])
def test_refresh_latest_trades(
    mocker, default_conf, caplog, candle_type, tmp_path, time_machine
) -> None:
    time_machine.move_to(dt_now(), tick=False)
    交易记录 = [
        {
            # Unix时间戳（毫秒）
            "timestamp": dt_ts(dt_now() - timedelta(minutes=5)),
            "amount": 16.512,
            "cost": 10134.07488,
            "fee": None,
            "fees": [],
            "id": "354669639",
            "order": None,
            "price": 613.74,
            "side": "sell",
            "takerOrMaker": None,
            "type": None,
        },
        {
            "timestamp": dt_ts(),  # Unix时间戳（毫秒）
            "amount": 12.512,
            "cost": 1000,
            "fee": None,
            "fees": [],
            "id": "354669640",
            "order": None,
            "price": 613.84,
            "side": "buy",
            "takerOrMaker": None,
            "type": None,
        },
    ]

    caplog.set_level(logging.DEBUG)
    使用交易配置 = default_conf
    使用交易配置["exchange"]["use_public_trades"] = True
    使用交易配置["exchange"]["only_from_ccxt"] = True

    使用交易配置["datadir"] = tmp_path
    使用交易配置["orderflow"] = {"max_candles": 1500}
    exchange = get_patched_exchange(mocker, 使用交易配置)
    exchange._api_async.fetch_trades = get_mock_coro(交易记录)
    exchange._ft_has["exchange_has_overrides"]["fetchTrades"] = True

    交易对列表 = [("IOTA/USDT:USDT", "5m", candle_type), ("XRP/USDT:USDT", "5m", candle_type)]
    # 空字典
    assert not exchange._trades
    结果 = exchange.refresh_latest_trades(交易对列表, cache=False)
    # 不缓存
    assert not exchange._trades

    assert len(结果) == len(交易对列表)
    assert exchange._api_async.fetch_trades.call_count == 4
    exchange._api_async.fetch_trades.reset_mock()

    exchange.required_candle_call_count = 2
    结果 = exchange.refresh_latest_trades(交易对列表)
    assert len(结果) == len(交易对列表)

    assert log_has(f"刷新 {len(交易对列表)} 个交易对的交易数据", caplog)
    assert exchange._trades
    assert exchange._api_async.fetch_trades.call_count == 4
    exchange._api_async.fetch_trades.reset_mock()
    for 交易对 in 交易对列表:
        assert isinstance(exchange.trades(交易对), DataFrame)
        assert len(exchange.trades(交易对)) > 0

        # 如果copy为"True"，trades函数每次调用应返回不同的对象
        assert exchange.trades(交易对) is not exchange.trades(交易对)
        assert exchange.trades(交易对) is not exchange.trades(交易对, copy=True)
        assert exchange.trades(交易对, copy=True) is not exchange.trades(交易对, copy=True)
        assert exchange.trades(交易对, copy=False) is exchange.trades(交易对, copy=False)

        # 测试缓存
        ohlcv = [
            [
                dt_ts(dt_now() - timedelta(minutes=5)),  # Unix时间戳（毫秒）
                1,  # 开盘价
                2,  # 最高价
                3,  # 最低价
                4,  # 收盘价
                5,  # 成交量（以计价货币计）
            ],
            [
                dt_ts(),  # Unix时间戳（毫秒）
                3,  # 开盘价
                1,  # 最高价
                4,  # 最低价
                6,  # 收盘价
                5,  # 成交量（以计价货币计）
            ],
        ]
        列名 = DEFAULT_DATAFRAME_COLUMNS
        交易数据框 = DataFrame(ohlcv, columns=列名)

        交易数据框["date"] = to_datetime(交易数据框["date"], unit="ms", utc=True)
        交易数据框["date"] = 交易数据框["date"].apply(lambda date: timeframe_to_prev_date("5m", date))
        exchange._klines[交易对] = 交易数据框
    结果 = exchange.refresh_latest_trades(
        [("IOTA/USDT:USDT", "5m", candle_type), ("XRP/USDT:USDT", "5m", candle_type)]
    )
    assert len(结果) == 0
    assert exchange._api_async.fetch_trades.call_count == 0
    caplog.clear()

    # 重置刷新时间
    for 交易对 in 交易对列表:
        # 测试带有"过期"蜡烛的缓存
        交易记录 = [
            {
                # Unix时间戳（毫秒）
                "timestamp": dt_ts(exchange._klines[交易对].iloc[-1].date - timedelta(minutes=5)),
                "amount": 16.512,
                "cost": 10134.07488,
                "fee": None,
                "fees": [],
                "id": "354669639",
                "order": None,
                "price": 613.74,
                "side": "sell",
                "takerOrMaker": None,
                "type": None,
            }
        ]
        交易数据框 = DataFrame(交易记录)
        交易数据框["date"] = to_datetime(交易数据框["timestamp"], unit="ms", utc=True)
        exchange._trades[交易对] = 交易数据框
    结果 = exchange.refresh_latest_trades(
        [("IOTA/USDT:USDT", "5m", candle_type), ("XRP/USDT:USDT", "5m", candle_type)]
    )
    assert len(结果) == len(交易对列表)

    assert exchange._api_async.fetch_trades.call_count == 4

    # 缓存 - 但禁用缓存
    exchange._api_async.fetch_trades.reset_mock()
    exchange.required_candle_call_count = 1

    交易对列表 = [
        ("IOTA/ETH", "5m", candle_type),
        ("XRP/ETH", "5m", candle_type),
        ("XRP/ETH", "1d", candle_type),
    ]
    结果 = exchange.refresh_latest_trades(交易对列表, cache=False)
    assert len(结果) == 3
    assert exchange._api_async.fetch_trades.call_count == 6

    # 再次测试，不应从缓存返回！
    exchange._api_async.fetch_trades.reset_mock()
    结果 = exchange.refresh_latest_trades(交易对列表, cache=False)
    assert len(结果) == 3
    assert exchange._api_async.fetch_trades.call_count == 6
    exchange._api_async.fetch_trades.reset_mock()
    caplog.clear()


@pytest.mark.parametrize("candle_type", [CandleType.FUTURES, CandleType.MARK, CandleType.SPOT])
def test_refresh_latest_ohlcv_cache(mocker, default_conf, candle_type, time_machine) -> None:
    开始时间 = datetime(2021, 8, 1, 0, 0, 0, 0, tzinfo=timezone.utc)
    ohlcv = generate_test_data_raw("1h", 100, 开始时间.strftime("%Y-%m-%d"))
    time_machine.move_to(开始时间 + timedelta(hours=99, minutes=30))

    exchange = get_patched_exchange(mocker, default_conf)
    mocker.patch(f"{EXMS}.ohlcv_candle_limit", return_value=100)
    assert exchange._startup_candle_count == 0

    exchange._api_async.fetch_ohlcv = get_mock_coro(ohlcv)
    交易对1 = ("IOTA/ETH", "1h", candle_type)
    交易对2 = ("XRP/ETH", "1h", candle_type)
    交易对列表 = [交易对1, 交易对2]

    # 不缓存
    assert not exchange._klines
    结果 = exchange.refresh_latest_ohlcv(交易对列表, cache=False)
    assert exchange._api_async.fetch_ohlcv.call_count == 2
    assert len(结果) == 2
    assert len(结果[交易对1]) == 99
    assert len(结果[交易对2]) == 99
    assert not exchange._klines
    exchange._api_async.fetch_ohlcv.reset_mock()

    # 带缓存
    结果 = exchange.refresh_latest_ohlcv(交易对列表)
    assert exchange._api_async.fetch_ohlcv.call_count == 2
    assert len(结果) == 2
    assert len(结果[交易对1]) == 99
    assert len(结果[交易对2]) == 99
    assert exchange._klines
    assert exchange._pairs_last_refresh_time[交易对1] == ohlcv[-2][0]
    exchange._api_async.fetch_ohlcv.reset_mock()

    # 从缓存返回
    结果 = exchange.refresh_latest_ohlcv(交易对列表)
    assert exchange._api_async.fetch_ohlcv.call_count == 0
    assert len(结果) == 2
    assert len(结果[交易对1]) == 99
    assert len(结果[交易对2]) == 99
    assert exchange._pairs_last_refresh_time[交易对1] == ohlcv[-2][0]

    # 时间移动到下一根蜡烛，但结果尚未改变
    time_machine.move_to(开始时间 + timedelta(hours=101))
    结果 = exchange.refresh_latest_ohlcv(交易对列表)
    assert exchange._api_async.fetch_ohlcv.call_count == 2
    assert len(结果) == 2
    assert len(结果[交易对1]) == 99
    assert len(结果[交易对2]) == 99
    assert 结果[交易对2].at[0, "open"]
    assert exchange._pairs_last_refresh_time[交易对1] == ohlcv[-2][0]
    之前的刷新时间 = exchange._pairs_last_refresh_time[交易对1]

    # 交易所出现新蜡烛 - 返回100根蜡烛 - 但跳过一根蜡烛，所以我们实际上一次获得2根蜡烛
    新开始日期 = (开始时间 + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M")
    # mocker.patch(f"{EXMS}.ohlcv_candle_limit", return_value=100)
    ohlcv = generate_test_data_raw("1h", 100, 新开始日期)
    exchange._api_async.fetch_ohlcv = get_mock_coro(ohlcv)
    结果 = exchange.refresh_latest_ohlcv(交易对列表)
    assert exchange._api_async.fetch_ohlcv.call_count == 2
    assert len(结果) == 2
    assert len(结果[交易对1]) == 100
    assert len(结果[交易对2]) == 100
    # 验证索引从0开始
    assert 结果[交易对2].at[0, "open"]
    assert 之前的刷新时间 != exchange._pairs_last_refresh_time[交易对1]

    assert exchange._pairs_last_refresh_time[交易对1] == ohlcv[-2][0]
    assert exchange._pairs_last_refresh_time[交易对2] == ohlcv[-2][0]
    exchange._api_async.fetch_ohlcv.reset_mock()

    # 重试相同调用 - 从缓存
    结果 = exchange.refresh_latest_ohlcv(交易对列表)
    assert exchange._api_async.fetch_ohlcv.call_count == 0
    assert len(结果) == 2
    assert len(结果[交易对1]) == 100
    assert len(结果[交易对2]) == 100
    assert 结果[交易对2].at[0, "open"]

    # 移动到遥远的未来（因此一次调用会导致数据中的漏洞）
    time_machine.move_to(开始时间 + timedelta(hours=2000))
    ohlcv = generate_test_data_raw("1h", 100, 开始时间 + timedelta(hours=1900))
    exchange._api_async.fetch_ohlcv = get_mock_coro(ohlcv)
    结果 = exchange.refresh_latest_ohlcv(交易对列表)

    assert exchange._api_async.fetch_ohlcv.call_count == 2
    assert len(结果) == 2
    # 缓存清除 - 新数据
    assert len(结果[交易对1]) == 99
    assert len(结果[交易对2]) == 99
    assert 结果[交易对2].at[0, "open"]


def test_refresh_ohlcv_with_cache(mocker, default_conf, time_machine) -> None:
    开始时间 = datetime(2021, 8, 1, 0, 0, 0, 0, tzinfo=timezone.utc)
    ohlcv = generate_test_data_raw("1h", 100, 开始时间.strftime("%Y-%m-%d"))
    time_machine.move_to(开始时间, tick=False)
    交易对列表 = [
        ("ETH/BTC", "1d", CandleType.SPOT),
        ("TKN/BTC", "1d", CandleType.SPOT),
        ("LTC/BTC", "1d", CandleType.SPOT),
        ("LTC/BTC", "5m", CandleType.SPOT),
        ("LTC/BTC", "1h", CandleType.SPOT),
    ]

    ohlcv_data = {p: ohlcv for p in 交易对列表}
    ohlcv_mock = mocker.patch(f"{EXMS}.refresh_latest_ohlcv", return_value=ohlcv_data)
    mocker.patch(f"{EXMS}.ohlcv_candle_limit", return_value=100)
    exchange = get_patched_exchange(mocker, default_conf)

    assert len(exchange._expiring_candle_cache) == 0

    结果 = exchange.refresh_ohlcv_with_cache(交易对列表, 开始时间.timestamp())
    assert ohlcv_mock.call_count == 1
    assert ohlcv_mock.call_args_list[0][0][0] == 交易对列表
    assert len(ohlcv_mock.call_args_list[0][0][0]) == 5

    assert len(结果) == 5
    # 长度为3 - 因为我们有3种不同的时间周期
    assert len(exchange._expiring_candle_cache) == 3

    ohlcv_mock.reset_mock()
    结果 = exchange.refresh_ohlcv_with_cache(交易对列表, 开始时间.timestamp())
    assert ohlcv_mock.call_count == 0

    # 使5m缓存过期
    time_machine.move_to(开始时间 + timedelta(minutes=6), tick=False)

    ohlcv_mock.reset_mock()
    结果 = exchange.refresh_ohlcv_with_cache(交易对列表, 开始时间.timestamp())
    assert ohlcv_mock.call_count == 1
    assert len(ohlcv_mock.call_args_list[0][0][0]) == 1

    # 使5m和1h缓存过期
    time_machine.move_to(开始时间 + timedelta(hours=2), tick=False)

    ohlcv_mock.reset_mock()
    结果 = exchange.refresh_ohlcv_with_cache(交易对列表, 开始时间.timestamp())
    assert ohlcv_mock.call_count == 1
    assert len(ohlcv_mock.call_args_list[0][0][0]) == 2

    # 使所有缓存过期
    time_machine.move_to(开始时间 + timedelta(days=1, hours=2), tick=False)

    ohlcv_mock.reset_mock()
    结果 = exchange.refresh_ohlcv_with_cache(交易对列表, 开始时间.timestamp())
    assert ohlcv_mock.call_count == 1
    assert len(ohlcv_mock.call_args_list[0][0][0]) == 5
    assert ohlcv_mock.call_args_list[0][0][0] == 交易对列表


@pytest.mark.parametrize("exchange_name", EXCHANGES)
async def test__async_get_candle_history(default_conf, mocker, caplog, exchange_name):
    ohlcv = [
        [
            dt_ts(),  # Unix时间戳（毫秒）
            1,  # 开盘价
            2,  # 最高价
            3,  # 最低价
            4,  # 收盘价
            5,  # 成交量（以计价货币计）
        ]
    ]

    caplog.set_level(logging.DEBUG)
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    # 猴子补丁异步函数
    exchange._api_async.fetch_ohlcv = get_mock_coro(ohlcv)

    交易对 = "ETH/BTC"
    结果 = await exchange._async_get_candle_history(交易对, "5m", CandleType.SPOT)
    assert type(结果) is tuple
    assert len(结果) == 5
    assert 结果[0] == 交易对
    assert 结果[1] == "5m"
    assert 结果[2] == CandleType.SPOT
    assert 结果[3] == ohlcv
    assert exchange._api_async.fetch_ohlcv.call_count == 1
    assert not log_has(f"使用缓存的蜡烛图（OHLCV）数据：{交易对} ...", caplog)
    exchange.close()
    # exchange = Exchange(default_conf)
    await async_ccxt_exception(
        mocker,
        default_conf,
        MagicMock(),
        "_async_get_candle_history",
        "fetch_ohlcv",
        pair="ABCD/BTC",
        timeframe=default_conf["timeframe"],
        candle_type=CandleType.SPOT,
    )

    api_mock = MagicMock()
    with pytest.raises(
        OperationalException, match=r"无法获取历史蜡烛图（OHLCV）数据.*"
    ):
        api_mock.fetch_ohlcv = MagicMock(side_effect=ccxt.BaseError("未知错误"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        await exchange._async_get_candle_history(
            交易对, "5m", CandleType.SPOT, dt_ts(dt_now() - timedelta(seconds=2000))
        )

    exchange.close()

    with pytest.raises(
        OperationalException,
        match=r"交易所.*不支持获取历史蜡烛图（OHLCV）数据\..*",
    ):
        api_mock.fetch_ohlcv = MagicMock(side_effect=ccxt.NotSupported("不支持"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        await exchange._async_get_candle_history(
            交易对, "5m", CandleType.SPOT, dt_ts(dt_now() - timedelta(seconds=2000))
        )
    exchange.close()


async def test__async_kucoin_get_candle_history(default_conf, mocker, caplog):
    from freqtrade.exchange.common import _reset_logging_mixin

    _reset_logging_mixin()
    caplog.set_level(logging.INFO)
    api_mock = MagicMock()
    api_mock.fetch_ohlcv = MagicMock(
        side_effect=ccxt.DDoSProtection(
            "kucoin GET https://openapi-v2.kucoin.com/api/v1/market/candles?"
            "symbol=ETH-BTC&type=5min&startAt=1640268735&endAt=1640418735"
            "429 太多请求"
            '{"code":"429000","msg":"太多请求"}'
        )
    )
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="kucoin")
    mocker.patch(f"{EXMS}.name", PropertyMock(return_value="KuCoin"))

    消息 = "Kucoin 429错误，避免触发DDosProtection退避延迟"
    assert not num_log_has_re(消息, caplog)

    for _ in range(3):
        with pytest.raises(DDosProtection, match=r"429 太多请求"):
            await exchange._async_get_candle_history(
                "ETH/BTC",
                "5m",
                CandleType.SPOT,
                since_ms=dt_ts(dt_now() - timedelta(seconds=2000)),
                count=3,
            )
    assert num_log_has_re(消息, caplog) == 3

    caplog.clear()
    # 测试常规非kucoin消息
    api_mock.fetch_ohlcv = MagicMock(
        side_effect=ccxt.DDoSProtection(
            "kucoin GET https://openapi-v2.kucoin.com/api/v1/market/candles?"
            "symbol=ETH-BTC&type=5min&startAt=1640268735&endAt=1640418735"
            "429 太多请求"
            '{"code":"2222222","msg":"太多请求"}'
        )
    )

    消息 = r"_async_get_candle_history\(\) 返回异常: .*"
    消息2 = r"应用DDosProtection退避延迟: .*"
    with patch("freqtrade.exchange.common.asyncio.sleep", get_mock_coro(None)):
        for _ in range(3):
            with pytest.raises(DDosProtection, match=r"429 太多请求"):
                await exchange._async_get_candle_history(
                    "ETH/BTC",
                    "5m",
                    CandleType.SPOT,
                    dt_ts(dt_now() - timedelta(seconds=2000)),
                    count=3,
                )
        # 期望"返回异常"消息12次（4次重试 * 3次循环）
        assert num_log_has_re(消息, caplog) == 12
        assert num_log_has_re(消息2, caplog) == 9
    exchange.close()


async def test__async_get_candle_history_empty(default_conf, mocker, caplog):
    """测试空的交易所结果"""
    ohlcv = []

    caplog.set_level(logging.DEBUG)
    exchange = get_patched_exchange(mocker, default_conf)
    # 猴子补丁异步函数
    exchange._api_async.fetch_ohlcv = get_mock_coro([])

    exchange = Exchange(default_conf)
    交易对 = "ETH/BTC"
    结果 = await exchange._async_get_candle_history(交易对, "5m", CandleType.SPOT)
    assert type(结果) is tuple
    assert len(结果) == 5
    assert 结果[0] == 交易对
    assert 结果[1] == "5m"
    assert 结果[2] == CandleType.SPOT
    assert 结果[3] == ohlcv
    assert exchange._api_async.fetch_ohlcv.call_count == 1
    exchange.close()


def test_refresh_latest_ohlcv_inv_result(default_conf, mocker, caplog):
    async def mock_get_candle_hist(交易对, *args, **kwargs):
        if 交易对 == "ETH/BTC":
            return [[]]
        else:
            raise TypeError()

    exchange = get_patched_exchange(mocker, default_conf)

    # 猴子补丁异步函数，返回空结果
    exchange._api_async.fetch_ohlcv = MagicMock(side_effect=mock_get_candle_hist)

    交易对列表 = [("ETH/BTC", "5m", ""), ("XRP/BTC", "5m", "")]
    结果 = exchange.refresh_latest_ohlcv(交易对列表)
    assert exchange._klines
    assert exchange._api_async.fetch_ohlcv.call_count == 2

    assert isinstance(结果, dict)
    assert len(结果) == 1
    # 测试每个至少在列表中出现一次，因为顺序不保证
    assert log_has("加载ETH/BTC错误。结果为[[]]。", caplog)
    assert log_has("异步代码抛出异常: TypeError()", caplog)


def test_get_next_limit_in_list():
    限制范围 = [5, 10, 20, 50, 100, 500, 1000]
    assert Exchange.get_next_limit_in_list(1, 限制范围) == 5
    assert Exchange.get_next_limit_in_list(5, 限制范围) == 5
    assert Exchange.get_next_limit_in_list(6, 限制范围) == 10
    assert Exchange.get_next_limit_in_list(9, 限制范围) == 10
    assert Exchange.get_next_limit_in_list(10, 限制范围) == 10
    assert Exchange.get_next_limit_in_list(11, 限制范围) == 20
    assert Exchange.get_next_limit_in_list(19, 限制范围) == 20
    assert Exchange.get_next_limit_in_list(21, 限制范围) == 50
    assert Exchange.get_next_limit_in_list(51, 限制范围) == 100
    assert Exchange.get_next_limit_in_list(1000, 限制范围) == 1000
    # 超过限制...
    assert Exchange.get_next_limit_in_list(1001, 限制范围) == 1000
    assert Exchange.get_next_limit_in_list(2000, 限制范围) == 1000
    # 没有所需范围
    assert Exchange.get_next_limit_in_list(2000, 限制范围, False) is None
    assert Exchange.get_next_limit_in_list(15, 限制范围, False) == 20

    assert Exchange.get_next_limit_in_list(21, None) == 21
    assert Exchange.get_next_limit_in_list(100, None) == 100
    assert Exchange.get_next_limit_in_list(1000, None) == 1000
    # 有上限
    assert Exchange.get_next_limit_in_list(1000, None, upper_limit=None) == 1000
    assert Exchange.get_next_limit_in_list(1000, None, upper_limit=500) == 500
    # 有上限和范围，限制范围优先
    assert Exchange.get_next_limit_in_list(1000, 限制范围, upper_limit=500) == 1000


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_fetch_l2_order_book(default_conf, mocker, order_book_l2, exchange_name):
    default_conf["exchange"]["name"] = exchange_name
    api_mock = MagicMock()

    api_mock.fetch_l2_order_book = order_book_l2
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    订单簿 = exchange.fetch_l2_order_book(pair="ETH/BTC", limit=10)
    assert "bids" in 订单簿
    assert "asks" in 订单簿
    assert len(订单簿["bids"]) == 10
    assert len(订单簿["asks"]) == 10
    assert api_mock.fetch_l2_order_book.call_args_list[0][0][0] == "ETH/BTC"

    for val in [1, 5, 10, 12, 20, 50, 100]:
        api_mock.fetch_l2_order_book.reset_mock()

        订单簿 = exchange.fetch_l2_order_book(pair="ETH/BTC", limit=val)
        assert api_mock.fetch_l2_order_book.call_args_list[0][0][0] == "ETH/BTC"
        # 并非所有交易所都支持订单簿的所有限制
        if not exchange.get_option("l2_limit_range") or val in exchange.get_option(
            "l2_limit_range"
        ):
            assert api_mock.fetch_l2_order_book.call_args_list[0][0][1] == val
        else:
            下一个限制 = exchange.get_next_limit_in_list(val, exchange.get_option("l2_limit_range"))
            assert api_mock.fetch_l2_order_book.call_args_list[0][0][1] == 下一个限制


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_fetch_l2_order_book_exception(default_conf, mocker, exchange_name):
    api_mock = MagicMock()
    with pytest.raises(OperationalException):
        api_mock.fetch_l2_order_book = MagicMock(side_effect=ccxt.NotSupported("不支持"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.fetch_l2_order_book(pair="ETH/BTC", limit=50)
    with pytest.raises(TemporaryError):
        api_mock.fetch_l2_order_book = MagicMock(side_effect=ccxt.NetworkError("DeadBeef"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.fetch_l2_order_book(pair="ETH/BTC", limit=50)
    with pytest.raises(OperationalException):
        api_mock.fetch_l2_order_book = MagicMock(side_effect=ccxt.BaseError("DeadBeef"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.fetch_l2_order_book(pair="ETH/BTC", limit=50)


@pytest.mark.parametrize("side,ask,bid,last,last_ab,expected", get_entry_rate_data)
def test_get_entry_rate(
    mocker, default_conf, caplog, side, ask, bid, last, last_ab, expected, time_machine
) -> None:
    caplog.set_level(logging.DEBUG)
    开始时间 = datetime(2023, 12, 1, 0, 10, 0, tzinfo=timezone.utc)
    time_machine.move_to(开始时间, tick=False)
    if last_ab is None:
        del default_conf["entry_pricing"]["price_last_balance"]
    else:
        default_conf["entry_pricing"]["price_last_balance"] = last_ab
    default_conf["entry_pricing"]["price_side"] = side
    exchange = get_patched_exchange(mocker, default_conf)
    mocker.patch(f"{EXMS}.fetch_ticker", return_value={"ask": ask, "last": last, "bid": bid})
    日志消息 = "使用缓存的ETH/BTC入场费率。"

    assert exchange.get_rate("ETH/BTC", side="entry", is_short=False, refresh=True) == expected
    assert not log_has(日志消息, caplog)

    time_machine.move_to(开始时间 + timedelta(minutes=4), tick=False)
    # 第二次运行，不刷新！
    caplog.clear()
    assert exchange.get_rate("ETH/BTC", side="entry", is_short=False, refresh=False) == expected
    assert log_has(日志消息, caplog)

    time_machine.move_to(开始时间 + timedelta(minutes=6), tick=False)
    # 第二次运行 - 由于ttl超时，强制刷新
    caplog.clear()
    assert exchange.get_rate("ETH/BTC", side="entry", is_short=False, refresh=False) == expected
    assert not log_has(日志消息, caplog)

    # 第二次运行，带刷新！
    caplog.clear()
    assert exchange.get_rate("ETH/BTC", side="entry", is_short=False, refresh=True) == expected
    assert not log_has(日志消息, caplog)


@pytest.mark.parametrize("side,ask,bid,last,last_ab,expected", get_exit_rate_data)
def test_get_exit_rate(
    default_conf, mocker, caplog, side, bid, ask, last, last_ab, expected, time_machine
) -> None:
    caplog.set_level(logging.DEBUG)
    开始时间 = datetime(2023, 12, 1, 0, 10, 0, tzinfo=timezone.utc)
    time_machine.move_to(开始时间, tick=False)

    default_conf["exit_pricing"]["price_side"] = side
    if last_ab is not None:
        default_conf["exit_pricing"]["price_last_balance"] = last_ab
    mocker.patch(f"{EXMS}.fetch_ticker", return_value={"ask": ask, "bid": bid, "last": last})
    交易对 = "ETH/BTC"
    日志消息 = "使用缓存的ETH/BTC出场费率。"

    # 测试常规模式
    exchange = get_patched_exchange(mocker, default_conf)
    费率 = exchange.get_rate(交易对, side="exit", is_short=False, refresh=True)
    assert not log_has(日志消息, caplog)
    assert isinstance(费率, float)
    assert 费率 == expected
    # 使用缓存
    caplog.clear()
    assert exchange.get_rate(交易对, side="exit", is_short=False, refresh=False) == expected
    assert log_has(日志消息, caplog)

    time_machine.move_to(开始时间 + timedelta(minutes=4), tick=False)
    # 缓存仍有效 - TTL未过期
    caplog.clear()
    assert exchange.get_rate(交易对, side="exit", is_short=False, refresh=False) == expected
    assert log_has(日志消息, caplog)

    time_machine.move_to(开始时间 + timedelta(minutes=6), tick=False)
    # 缓存过期 - 强制刷新
    caplog.clear()
    assert exchange.get_rate(交易对, side="exit", is_short=False, refresh=False) == expected
    assert not log_has(日志消息, caplog)


@pytest.mark.parametrize(
    "entry,is_short,side,ask,bid,last,last_ab,expected",
    [
        ("entry", False, "ask", None, 4, 4, 0, 4),  # 卖一不可用
        ("entry", False, "ask", None, None, 4, 0, 4),  # 卖一不可用
        ("entry", False, "bid", 6, None, 4, 0, 5),  # 买一不可用
        ("entry", False, "bid", None, None, 4, 0, 5),  # 无可用费率
        ("exit", False, "ask", None, 4, 4, 0, 4),  # 卖一不可用
        ("exit", False, "ask", None, None, 4, 0, 4),  # 卖一不可用
        ("exit", False, "bid", 6, None, 4, 0, 5),  # 买一不可用
        ("exit", False, "bid", None, None, 4, 0, 5),  # 买一不可用
    ],
)
def test_get_ticker_rate_error(
    mocker, entry, default_conf, caplog, side, is_short, ask, bid, last, last_ab, expected
) -> None:
    caplog.set_level(logging.DEBUG)
    default_conf["entry_pricing"]["price_last_balance"] = last_ab
    default_conf["entry_pricing"]["price_side"] = side
    default_conf["exit_pricing"]["price_side"] = side
    default_conf["exit_pricing"]["price_last_balance"] = last_ab
    exchange = get_patched_exchange(mocker, default_conf)
    mocker.patch(f"{EXMS}.fetch_ticker", return_value={"ask": ask, "last": last, "bid": bid})


    
    with pytest.raises(PricingError):
        exchange.get_rate("ETH/BTC", refresh=True, side=entry, is_short=is_short)


@pytest.mark.parametrize(
    "is_short,side,expected",
    [
        (False, "bid", 0.043936),  # 来自order_book_l2测试数据 - 买单侧
        (False, "ask", 0.043949),  # 来自order_book_l2测试数据 - 卖单侧
        (False, "other", 0.043936),  # 来自order_book_l2测试数据 - 买单侧
        (False, "same", 0.043949),  # 来自order_book_l2测试数据 - 卖单侧
        (True, "bid", 0.043936),  # 来自order_book_l2测试数据 - 买单侧
        (True, "ask", 0.043949),  # 来自order_book_l2测试数据 - 卖单侧
        (True, "other", 0.043949),  # 来自order_book_l2测试数据 - 卖单侧
        (True, "same", 0.043936),  # 来自order_book_l2测试数据 - 买单侧
    ],
)
def test_get_exit_rate_orderbook(
    default_conf, mocker, caplog, is_short, side, expected, order_book_l2
):
    caplog.set_level(logging.DEBUG)
    # 测试订单簿模式
    default_conf["exit_pricing"]["price_side"] = side
    default_conf["exit_pricing"]["use_order_book"] = True
    default_conf["exit_pricing"]["order_book_top"] = 1
    交易对 = "ETH/BTC"
    mocker.patch(f"{EXMS}.fetch_l2_order_book", order_book_l2)
    exchange = get_patched_exchange(mocker, default_conf)
    汇率 = exchange.get_rate(交易对, refresh=True, side="exit", is_short=is_short)
    assert not log_has("使用ETH/BTC的缓存退出汇率。", caplog)
    assert isinstance(汇率, float)
    assert 汇率 == expected
    汇率 = exchange.get_rate(交易对, refresh=False, side="exit", is_short=is_short)
    assert 汇率 == expected
    assert log_has("使用ETH/BTC的缓存退出汇率。", caplog)


def test_get_exit_rate_orderbook_exception(default_conf, mocker, caplog):
    # 测试订单簿模式
    default_conf["exit_pricing"]["price_side"] = "ask"
    default_conf["exit_pricing"]["use_order_book"] = True
    default_conf["exit_pricing"]["order_book_top"] = 1
    交易对 = "ETH/BTC"
    # 测试当交易所返回空订单簿时的情况
    mocker.patch(f"{EXMS}.fetch_l2_order_book", return_value={"bids": [[]], "asks": [[]]})
    exchange = get_patched_exchange(mocker, default_conf)
    with pytest.raises(PricingError):
        exchange.get_rate(交易对, refresh=True, side="exit", is_short=False)
    assert log_has_re(
        rf"{交易对} - 无法确定订单簿位置1的退出价格\..*",
        caplog,
    )


@pytest.mark.parametrize("is_short", [True, False])
def test_get_exit_rate_exception(default_conf, mocker, is_short):
    # 在某些情况下，一侧的报价可能为空
    default_conf["exit_pricing"]["price_side"] = "ask"
    交易对 = "ETH/BTC"
    mocker.patch(f"{EXMS}.fetch_ticker", return_value={"ask": None, "bid": 0.12, "last": None})
    exchange = get_patched_exchange(mocker, default_conf)
    with pytest.raises(PricingError, match=r"ETH/BTC的退出汇率为空。"):
        exchange.get_rate(交易对, refresh=True, side="exit", is_short=is_short)

    exchange._config["exit_pricing"]["price_side"] = "bid"
    assert exchange.get_rate(交易对, refresh=True, side="exit", is_short=is_short) == 0.12
    # 反转两侧
    mocker.patch(f"{EXMS}.fetch_ticker", return_value={"ask": 0.13, "bid": None, "last": None})
    with pytest.raises(PricingError, match=r"ETH/BTC的退出汇率为空。"):
        exchange.get_rate(交易对, refresh=True, side="exit", is_short=is_short)

    exchange._config["exit_pricing"]["price_side"] = "ask"
    assert exchange.get_rate(交易对, refresh=True, side="exit", is_short=is_short) == 0.13


@pytest.mark.parametrize("side,ask,bid,last,last_ab,expected", get_entry_rate_data)
@pytest.mark.parametrize("side2", ["bid", "ask"])
@pytest.mark.parametrize("use_order_book", [True, False])
def test_get_rates_testing_entry(
    mocker,
    default_conf,
    caplog,
    side,
    ask,
    bid,
    last,
    last_ab,
    expected,
    side2,
    use_order_book,
    order_book_l2,
) -> None:
    caplog.set_level(logging.DEBUG)
    if last_ab is None:
        del default_conf["entry_pricing"]["price_last_balance"]
    else:
        default_conf["entry_pricing"]["price_last_balance"] = last_ab
    default_conf["entry_pricing"]["price_side"] = side
    default_conf["exit_pricing"]["price_side"] = side2
    default_conf["exit_pricing"]["use_order_book"] = use_order_book
    api_mock = MagicMock()
    api_mock.fetch_l2_order_book = order_book_l2
    api_mock.fetch_ticker = MagicMock(return_value={"ask": ask, "last": last, "bid": bid})
    exchange = get_patched_exchange(mocker, default_conf, api_mock)

    assert exchange.get_rates("ETH/BTC", refresh=True, is_short=False)[0] == expected
    assert not log_has("使用ETH/BTC的缓存买入汇率。", caplog)

    api_mock.fetch_l2_order_book.reset_mock()
    api_mock.fetch_ticker.reset_mock()
    assert exchange.get_rates("ETH/BTC", refresh=False, is_short=False)[0] == expected
    assert log_has("使用ETH/BTC的缓存买入汇率。", caplog)
    assert api_mock.fetch_l2_order_book.call_count == 0
    assert api_mock.fetch_ticker.call_count == 0
    # 第二次运行，开启刷新！
    caplog.clear()

    assert exchange.get_rates("ETH/BTC", refresh=True, is_short=False)[0] == expected
    assert not log_has("使用ETH/BTC的缓存买入汇率。", caplog)

    assert api_mock.fetch_l2_order_book.call_count == int(use_order_book)
    assert api_mock.fetch_ticker.call_count == 1


@pytest.mark.parametrize("side,ask,bid,last,last_ab,expected", get_exit_rate_data)
@pytest.mark.parametrize("side2", ["bid", "ask"])
@pytest.mark.parametrize("use_order_book", [True, False])
def test_get_rates_testing_exit(
    default_conf,
    mocker,
    caplog,
    side,
    bid,
    ask,
    last,
    last_ab,
    expected,
    side2,
    use_order_book,
    order_book_l2,
) -> None:
    caplog.set_level(logging.DEBUG)

    default_conf["exit_pricing"]["price_side"] = side
    if last_ab is not None:
        default_conf["exit_pricing"]["price_last_balance"] = last_ab

    default_conf["entry_pricing"]["price_side"] = side2
    default_conf["entry_pricing"]["use_order_book"] = use_order_book
    api_mock = MagicMock()
    api_mock.fetch_l2_order_book = order_book_l2
    api_mock.fetch_ticker = MagicMock(return_value={"ask": ask, "last": last, "bid": bid})
    exchange = get_patched_exchange(mocker, default_conf, api_mock)

    交易对 = "ETH/BTC"

    # 测试常规模式
    汇率 = exchange.get_rates(交易对, refresh=True, is_short=False)[1]
    assert not log_has("使用ETH/BTC的缓存卖出汇率。", caplog)
    assert isinstance(汇率, float)
    assert 汇率 == expected
    # 使用缓存
    api_mock.fetch_l2_order_book.reset_mock()
    api_mock.fetch_ticker.reset_mock()

    汇率 = exchange.get_rates(交易对, refresh=False, is_short=False)[1]
    assert 汇率 == expected
    assert log_has("使用ETH/BTC的缓存卖出汇率。", caplog)

    assert api_mock.fetch_l2_order_book.call_count == 0
    assert api_mock.fetch_ticker.call_count == 0


@pytest.mark.parametrize("exchange_name", EXCHANGES)
async def test___async_get_candle_history_sort(default_conf, mocker, exchange_name):
    def sort_data(data, key):
        return sorted(data, key=key)

    # GDAX用例（来自GDAX的真实数据）
    # 此OHLCV数据按降序排列（最新的在前，最旧的在后）
    ohlcv = [
        [1527833100000, 0.07666, 0.07671, 0.07666, 0.07668, 16.65244264],
        [1527832800000, 0.07662, 0.07666, 0.07662, 0.07666, 1.30051526],
        [1527832500000, 0.07656, 0.07661, 0.07656, 0.07661, 12.034778840000001],
        [1527832200000, 0.07658, 0.07658, 0.07655, 0.07656, 0.59780186],
        [1527831900000, 0.07658, 0.07658, 0.07658, 0.07658, 1.76278136],
        [1527831600000, 0.07658, 0.07658, 0.07658, 0.07658, 2.22646521],
        [1527831300000, 0.07655, 0.07657, 0.07655, 0.07657, 1.1753],
        [1527831000000, 0.07654, 0.07654, 0.07651, 0.07651, 0.8073060299999999],
        [1527830700000, 0.07652, 0.07652, 0.07651, 0.07652, 10.04822687],
        [1527830400000, 0.07649, 0.07651, 0.07649, 0.07651, 2.5734867],
    ]
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    exchange._api_async.fetch_ohlcv = get_mock_coro(ohlcv)
    sort_mock = mocker.patch("freqtrade.exchange.exchange.sorted", MagicMock(side_effect=sort_data))
    # 测试OHLCV数据排序
    res = await exchange._async_get_candle_history(
        "ETH/BTC", default_conf["timeframe"], CandleType.SPOT
    )
    assert res[0] == "ETH/BTC"
    res_ohlcv = res[3]

    assert sort_mock.call_count == 1
    assert res_ohlcv[0][0] == 1527830400000
    assert res_ohlcv[0][1] == 0.07649
    assert res_ohlcv[0][2] == 0.07651
    assert res_ohlcv[0][3] == 0.07649
    assert res_ohlcv[0][4] == 0.07651
    assert res_ohlcv[0][5] == 2.5734867

    assert res_ohlcv[9][0] == 1527833100000
    assert res_ohlcv[9][1] == 0.07666
    assert res_ohlcv[9][2] == 0.07671
    assert res_ohlcv[9][3] == 0.07666
    assert res_ohlcv[9][4] == 0.07668
    assert res_ohlcv[9][5] == 16.65244264

    # 此OHLCV数据按升序排列（最旧的在前，最新的在后）
    ohlcv = [
        [1527827700000, 0.07659999, 0.0766, 0.07627, 0.07657998, 1.85216924],
        [1527828000000, 0.07657995, 0.07657995, 0.0763, 0.0763, 26.04051037],
        [1527828300000, 0.0763, 0.07659998, 0.0763, 0.0764, 10.36434124],
        [1527828600000, 0.0764, 0.0766, 0.0764, 0.0766, 5.71044773],
        [1527828900000, 0.0764, 0.07666998, 0.0764, 0.07666998, 47.48888565],
        [1527829200000, 0.0765, 0.07672999, 0.0765, 0.07672999, 3.37640326],
        [1527829500000, 0.0766, 0.07675, 0.0765, 0.07675, 8.36203831],
        [1527829800000, 0.07675, 0.07677999, 0.07620002, 0.076695, 119.22963884],
        [1527830100000, 0.076695, 0.07671, 0.07624171, 0.07671, 1.80689244],
        [1527830400000, 0.07671, 0.07674399, 0.07629216, 0.07655213, 2.31452783],
    ]
    exchange._api_async.fetch_ohlcv = get_mock_coro(ohlcv)
    # 重置排序模拟
    sort_mock = mocker.patch("freqtrade.exchange.sorted", MagicMock(side_effect=sort_data))
    # 测试OHLCV数据排序
    res = await exchange._async_get_candle_history(
        "ETH/BTC", default_conf["timeframe"], CandleType.SPOT
    )
    assert res[0] == "ETH/BTC"
    assert res[1] == default_conf["timeframe"]
    res_ohlcv = res[3]
    # 排序未再次调用 - 数据已按顺序排列
    assert sort_mock.call_count == 0
    assert res_ohlcv[0][0] == 1527827700000
    assert res_ohlcv[0][1] == 0.07659999
    assert res_ohlcv[0][2] == 0.0766
    assert res_ohlcv[0][3] == 0.07627
    assert res_ohlcv[0][4] == 0.07657998
    assert res_ohlcv[0][5] == 1.85216924

    assert res_ohlcv[9][0] == 1527830400000
    assert res_ohlcv[9][1] == 0.07671
    assert res_ohlcv[9][2] == 0.07674399
    assert res_ohlcv[9][3] == 0.07629216
    assert res_ohlcv[9][4] == 0.07655213
    assert res_ohlcv[9][5] == 2.31452783


@pytest.mark.parametrize("exchange_name", EXCHANGES)
async def test__async_fetch_trades(
    default_conf, mocker, caplog, exchange_name, fetch_trades_result
):
    caplog.set_level(logging.DEBUG)
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    # 猴子补丁异步函数
    exchange._api_async.fetch_trades = get_mock_coro(fetch_trades_result)

    交易对 = "ETH/BTC"
    res, pagid = await exchange._async_fetch_trades(交易对, since=None, params=None)
    assert isinstance(res, list)
    assert isinstance(res[0], list)
    assert isinstance(res[1], list)
    if exchange._trades_pagination == "id":
        if exchange_name == "kraken":
            assert pagid == 1565798399872512133
        else:
            assert pagid == "126181333"
    else:
        assert pagid == 1565798399872

    assert exchange._api_async.fetch_trades.call_count == 1
    assert exchange._api_async.fetch_trades.call_args[0][0] == 交易对
    assert exchange._api_async.fetch_trades.call_args[1]["limit"] == 1000

    assert log_has_re(f"获取交易对{交易对}的交易，since .*", caplog)
    caplog.clear()
    exchange._api_async.fetch_trades.reset_mock()
    res, pagid = await exchange._async_fetch_trades(交易对, since=None, params={"from": "123"})
    assert exchange._api_async.fetch_trades.call_count == 1
    assert exchange._api_async.fetch_trades.call_args[0][0] == 交易对
    assert exchange._api_async.fetch_trades.call_args[1]["limit"] == 1000
    assert exchange._api_async.fetch_trades.call_args[1]["params"] == {"from": "123"}

    if exchange._trades_pagination == "id":
        if exchange_name == "kraken":
            assert pagid == 1565798399872512133
        else:
            assert pagid == "126181333"
    else:
        assert pagid == 1565798399872

    assert log_has_re(f"获取交易对{交易对}的交易，参数: .*", caplog)
    exchange.close()

    await async_ccxt_exception(
        mocker,
        default_conf,
        MagicMock(),
        "_async_fetch_trades",
        "fetch_trades",
        pair="ABCD/BTC",
        since=None,
    )

    api_mock = MagicMock()
    with pytest.raises(OperationalException, match=r"无法获取交易数据*"):
        api_mock.fetch_trades = MagicMock(side_effect=ccxt.BaseError("未知错误"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        await exchange._async_fetch_trades(交易对, since=dt_ts(dt_now() - timedelta(seconds=2000)))
    exchange.close()

    with pytest.raises(
        OperationalException,
        match=r"交易所.*不支持获取历史交易数据\..*",
    ):
        api_mock.fetch_trades = MagicMock(side_effect=ccxt.NotSupported("不支持"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        await exchange._async_fetch_trades(交易对, since=dt_ts(dt_now() - timedelta(seconds=2000)))
    exchange.close()


@pytest.mark.parametrize("exchange_name", EXCHANGES)
async def test__async_fetch_trades_contract_size(
    default_conf, mocker, caplog, exchange_name, fetch_trades_result
):
    caplog.set_level(logging.DEBUG)
    default_conf["margin_mode"] = "isolated"
    default_conf["trading_mode"] = "futures"
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    # 猴子补丁异步函数
    exchange._api_async.fetch_trades = get_mock_coro(
        [
            {
                "info": {
                    "a": 126181333,
                    "p": "0.01952600",
                    "q": "0.01200000",
                    "f": 138604158,
                    "l": 138604158,
                    "T": 1565798399872,
                    "m": True,
                    "M": True,
                },
                "timestamp": 1565798399872,
                "datetime": "2019-08-14T15:59:59.872Z",
                "symbol": "ETH/USDT:USDT",
                "id": "126181383",
                "order": None,
                "type": None,
                "takerOrMaker": None,
                "side": "sell",
                "price": 2.0,
                "amount": 30.0,
                "cost": 60.0,
                "fee": None,
            }
        ]
    )

    交易对 = "ETH/USDT:USDT"
    res, pagid = await exchange._async_fetch_trades(交易对, since=None, params=None)
    assert res[0][5] == 300
    assert pagid is not None
    exchange.close()


@pytest.mark.parametrize("exchange_name", EXCHANGES)
async def test__async_get_trade_history_id(
    default_conf, mocker, exchange_name, fetch_trades_result
):
    default_conf["exchange"]["only_from_ccxt"] = True
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    if exchange._trades_pagination != "id":
        exchange.close()
        pytest.skip("交易所不支持按交易ID分页")
    pagination_arg = exchange._trades_pagination_arg

    async def mock_get_trade_hist(交易对, *args, **kwargs):
        if "since" in kwargs:
            # 返回前3条
            return fetch_trades_result[:-2]
        elif kwargs.get("params", {}).get(pagination_arg) in (
            fetch_trades_result[-3]["id"],
            1565798399752,
        ):
            # 返回2条
            return fetch_trades_result[-3:-1]
        else:
            # 返回最后2条
            return fetch_trades_result[-2:]

    # 猴子补丁异步函数
    exchange._api_async.fetch_trades = MagicMock(side_effect=mock_get_trade_hist)

    交易对 = "ETH/BTC"
    ret = await exchange._async_get_trade_history_id(
        交易对,
        since=fetch_trades_result[0]["timestamp"],
        until=fetch_trades_result[-1]["timestamp"] - 1,
    )
    assert isinstance(ret, tuple)
    assert ret[0] == 交易对
    assert isinstance(ret[1], list)
    if exchange_name != "kraken":
        assert len(ret[1]) == len(fetch_trades_result)
    assert exchange._api_async.fetch_trades.call_count == 3
    fetch_trades_cal = exchange._api_async.fetch_trades.call_args_list
    # 第一次调用（使用since，不使用fromId）
    assert fetch_trades_cal[0][0][0] == 交易对
    assert fetch_trades_cal[0][1]["since"] == fetch_trades_result[0]["timestamp"]

    # 第二次调用
    assert fetch_trades_cal[1][0][0] == 交易对
    assert "params" in fetch_trades_cal[1][1]
    assert exchange._ft_has["trades_pagination_arg"] in fetch_trades_cal[1][1]["params"]


@pytest.mark.parametrize(
    "trade_id, expected",
    [
        ("1234", True),
        ("170544369512007228", True),
        ("1705443695120072285", True),
        ("170544369512007228555", True),
    ],
)
@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test__valid_trade_pagination_id(mocker, default_conf_usdt, exchange_name, trade_id, expected):
    if exchange_name == "kraken":
        pytest.skip("Kraken有不同的分页ID格式，并有专门的测试")
    exchange = get_patched_exchange(mocker, default_conf_usdt, exchange=exchange_name)

    assert exchange._valid_trade_pagination_id("XRP/USDT", trade_id) == expected


@pytest.mark.parametrize("exchange_name", EXCHANGES)
async def test__async_get_trade_history_time(
    default_conf, mocker, caplog, exchange_name, fetch_trades_result
):
    caplog.set_level(logging.DEBUG)

    async def mock_get_trade_hist(交易对, *args, **kwargs):
        if kwargs["since"] == fetch_trades_result[0]["timestamp"]:
            return fetch_trades_result[:-1]
        else:
            return fetch_trades_result[-1:]

    caplog.set_level(logging.DEBUG)
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    if exchange._trades_pagination != "time":
        exchange.close()
        pytest.skip("交易所不支持按时间戳分页")
    # 猴子补丁异步函数
    exchange._api_async.fetch_trades = MagicMock(side_effect=mock_get_trade_hist)
    交易对 = "ETH/BTC"
    ret = await exchange._async_get_trade_history_time(
        交易对,
        since=fetch_trades_result[0]["timestamp"],
        until=fetch_trades_result[-1]["timestamp"] - 1,
    )
    assert isinstance(ret, tuple)
    assert ret[0] == 交易对
    assert isinstance(ret[1], list)
    assert len(ret[1]) == len(fetch_trades_result)
    assert exchange._api_async.fetch_trades.call_count == 2
    fetch_trades_cal = exchange._api_async.fetch_trades.call_args_list
    # 第一次调用（使用since，不使用fromId）
    assert fetch_trades_cal[0][0][0] == 交易对
    assert fetch_trades_cal[0][1]["since"] == fetch_trades_result[0]["timestamp"]

    # 第二次调用
    assert fetch_trades_cal[1][0][0] == 交易对
    assert fetch_trades_cal[1][1]["since"] == fetch_trades_result[-2]["timestamp"]
    assert log_has_re(r"因达到until条件而停止.*", caplog)


@pytest.mark.parametrize("exchange_name", EXCHANGES)
async def test__async_get_trade_history_time_empty(
    default_conf, mocker, caplog, exchange_name, trades_history
):
    caplog.set_level(logging.DEBUG)

    async def mock_get_trade_hist(交易对, *args, **kwargs):
        if kwargs["since"] == trades_history[0][0]:
            return trades_history[:-1], trades_history[:-1][-1][0]
        else:
            return [], None

    caplog.set_level(logging.DEBUG)
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    # 猴子补丁异步函数
    exchange._async_fetch_trades = MagicMock(side_effect=mock_get_trade_hist)
    交易对 = "ETH/BTC"
    ret = await exchange._async_get_trade_history_time(
        交易对, since=trades_history[0][0], until=trades_history[-1][0] - 1
    )
    assert isinstance(ret, tuple)
    assert ret[0] == 交易对
    assert isinstance(ret[1], list)
    assert len(ret[1]) == len(trades_history) - 1
    assert exchange._async_fetch_trades.call_count == 2
    fetch_trades_cal = exchange._async_fetch_trades.call_args_list
    # 第一次调用（使用since，不使用fromId）
    assert fetch_trades_cal[0][0][0] == 交易对
    assert fetch_trades_cal[0][1]["since"] == trades_history[0][0]


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_get_historic_trades(default_conf, mocker, caplog, exchange_name, trades_history):
    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)

    交易对 = "ETH/BTC"

    exchange._async_get_trade_history_id = get_mock_coro((交易对, trades_history))
    exchange._async_get_trade_history_time = get_mock_coro((交易对, trades_history))
    ret = exchange.get_historic_trades(
        交易对, since=trades_history[0][0], until=trades_history[-1][0]
    )

    # 根据交易所不同，应该调用其中一个方法
    assert (
        sum(
            [
                exchange._async_get_trade_history_id.call_count,
                exchange._async_get_trade_history_time.call_count,
            ]
        )
        == 1
    )

    assert len(ret) == 2
    assert ret[0] == 交易对
    assert len(ret[1]) == len(trades_history)


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_get_historic_trades_notsupported(
    default_conf, mocker, caplog, exchange_name, trades_history
):
    mocker.patch(f"{EXMS}.exchange_has", return_value=False)
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)

    交易对 = "ETH/BTC"

    with pytest.raises(
        OperationalException, match="此交易所不支持下载交易记录。"
    ):
        exchange.get_historic_trades(交易对, since=trades_history[0][0], until=trades_history[-1][0])


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_cancel_order_dry_run(default_conf, mocker, exchange_name):
    default_conf["dry_run"] = True
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    mocker.patch(f"{EXMS}._dry_is_price_crossed", return_value=True)
    assert exchange.cancel_order(order_id="123", pair="TKN/BTC") == {}
    assert exchange.cancel_stoploss_order(order_id="123", pair="TKN/BTC") == {}

    order = exchange.create_order(
        pair="ETH/BTC",
        ordertype="limit",
        side="buy",
        amount=5,
        rate=0.55,
        time_in_force="gtc",
        leverage=1.0,
    )

    cancel_order = exchange.cancel_order(order_id=order["id"], pair="ETH/BTC")
    assert order["id"] == cancel_order["id"]
    assert order["amount"] == cancel_order["amount"]
    assert order["symbol"] == cancel_order["symbol"]
    assert cancel_order["status"] == "canceled"


@pytest.mark.parametrize("exchange_name", EXCHANGES)
@pytest.mark.parametrize(
    "order,result",
    [
        ({"status": "closed", "filled": 10}, False),
        ({"status": "closed", "filled": 0.0}, True),
        ({"status": "canceled", "filled": 0.0}, True),
        ({"status": "canceled", "filled": 10.0}, False),
        ({"status": "unknown", "filled": 10.0}, False),
        ({"result": "testest123"}, False),
    ],
)
def test_check_order_canceled_empty(mocker, default_conf, exchange_name, order, result):
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    assert exchange.check_order_canceled_empty(order) == result


@pytest.mark.parametrize("exchange_name", EXCHANGES)
@pytest.mark.parametrize(
    "order,result",
    [
        ({"status": "closed", "amount": 10, "fee": {}}, True),
        ({"status": "closed", "amount": 0.0, "fee": {}}, True),
        ({"status": "canceled", "amount": 0.0, "fee": {}}, True),
        ({"status": "canceled", "amount": 10.0}, False),
        ({"amount": 10.0, "fee": {}}, False),
        ({"result": "testest123"}, False),
        ("hello_world", False),
        ({"status": "canceled", "amount": None, "fee": None}, False),
        ({"status": "canceled", "filled": None, "amount": None, "fee": None}, False),
    ],
)
def test_is_cancel_order_result_suitable(mocker, default_conf, exchange_name, order, result):
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    assert exchange.is_cancel_order_result_suitable(order) == result


@pytest.mark.parametrize("exchange_name", EXCHANGES)
@pytest.mark.parametrize(
    "corder,call_corder,call_forder",
    [
        ({"status": "closed", "amount": 10, "fee": {}}, 1, 0),
        ({"amount": 10, "fee": {}}, 1, 1),
    ],
)
def test_cancel_order_with_result(
    default_conf, mocker, exchange_name, corder, call_corder, call_forder
):
    default_conf["dry_run"] = False
    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    api_mock = MagicMock()
    api_mock.cancel_order = MagicMock(return_value=corder)
    api_mock.fetch_order = MagicMock(return_value={"id": "1234"})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    res = exchange.cancel_order_with_result("1234", "ETH/BTC", 1234)
    assert isinstance(res, dict)
    assert api_mock.cancel_order.call_count == call_corder
    assert api_mock.fetch_order.call_count == call_forder


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_cancel_order_with_result_error(default_conf, mocker, exchange_name, caplog):
    default_conf["dry_run"] = False
    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    api_mock = MagicMock()
    api_mock.cancel_order = MagicMock(side_effect=ccxt.InvalidOrder("未找到订单"))
    api_mock.fetch_order = MagicMock(side_effect=ccxt.InvalidOrder("未找到订单"))
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)

    res = exchange.cancel_order_with_result("1234", "ETH/BTC", 1541)
    assert isinstance(res, dict)
    assert log_has("无法取消ETH/BTC的订单1234。", caplog)
    assert log_has("无法获取已取消的订单1234。", caplog)
    assert res["amount"] == 1541


# 确保在非模拟交易模式下，我们应该调用API
@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_cancel_order(default_conf, mocker, exchange_name):
    default_conf["dry_run"] = False
    api_mock = MagicMock()
    api_mock.cancel_order = MagicMock(return_value={"id": "123"})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    assert exchange.cancel_order(order_id="_", pair="TKN/BTC") == {"id": "123"}

    with pytest.raises(InvalidOrderException):
        api_mock.cancel_order = MagicMock(side_effect=ccxt.InvalidOrder("未找到订单"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.cancel_order(order_id="_", pair="TKN/BTC")
    assert api_mock.cancel_order.call_count == 1

    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        api_mock,
        exchange_name,
        "cancel_order",
        "cancel_order",
        order_id="_",
        pair="TKN/BTC",
    )


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_cancel_stoploss_order(default_conf, mocker, exchange_name):
    default_conf["dry_run"] = False
    api_mock = MagicMock()
    api_mock.cancel_order = MagicMock(return_value={"id": "123"})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    assert exchange.cancel_stoploss_order(order_id="_", pair="TKN/BTC") == {"id": "123"}

    with pytest.raises(InvalidOrderException):
        api_mock.cancel_order = MagicMock(side_effect=ccxt.InvalidOrder("未找到订单"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.cancel_stoploss_order(order_id="_", pair="TKN/BTC")
    assert api_mock.cancel_order.call_count == 1

    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        api_mock,
        exchange_name,
        "cancel_stoploss_order",
        "cancel_order",
        order_id="_",
        pair="TKN/BTC",
    )


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_cancel_stoploss_order_with_result(default_conf, mocker, exchange_name):
    default_conf["dry_run"] = False
    mock_prefix = "freqtrade.exchange.gate.Gate"
    if exchange_name == "okx":
        mock_prefix = "freqtrade.exchange.okx.Okx"
    mocker.patch(f"{EXMS}.fetch_stoploss_order", return_value={"for": 123})
    mocker.patch(f"{mock_prefix}.fetch_stoploss_order", return_value={"for": 123})
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)

    res = {"fee": {}, "status": "canceled", "amount": 1234}
    mocker.patch(f"{EXMS}.cancel_stoploss_order", return_value=res)
    mocker.patch(f"{mock_prefix}.cancel_stoploss_order", return_value=res)
    co = exchange.cancel_stoploss_order_with_result(order_id="_", pair="TKN/BTC", amount=555)
    assert co == res

    mocker.patch(f"{EXMS}.cancel_stoploss_order", return_value="canceled")
    mocker.patch(f"{mock_prefix}.cancel_stoploss_order", return_value="canceled")
    # 回退到fetch_stoploss_order
    co = exchange.cancel_stoploss_order_with_result(order_id="_", pair="TKN/BTC", amount=555)
    assert co == {"for": 123}

    exc = InvalidOrderException("")
    mocker.patch(f"{EXMS}.fetch_stoploss_order", side_effect=exc)
    mocker.patch(f"{mock_prefix}.fetch_stoploss_order", side_effect=exc)
    co = exchange.cancel_stoploss_order_with_result(order_id="_", pair="TKN/BTC", amount=555)
    assert co["amount"] == 555
    assert co == {"id": "_", "fee": {}, "status": "canceled", "amount": 555, "info": {}}

    with pytest.raises(InvalidOrderException):
        exc = InvalidOrderException("未找到订单")
        mocker.patch(f"{EXMS}.cancel_stoploss_order", side_effect=exc)
        mocker.patch(f"{mock_prefix}.cancel_stoploss_order", side_effect=exc)
        exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
        exchange.cancel_stoploss_order_with_result(order_id="_", pair="TKN/BTC", amount=123)


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_fetch_order(default_conf, mocker, exchange_name, caplog):
    default_conf["dry_run"] = True
    default_conf["exchange"]["log_responses"] = True
    order = MagicMock()
    order.myid = 123
    order.symbol = "TKN/BTC"

    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    exchange._dry_run_open_orders["X"] = order
    assert exchange.fetch_order("X", "TKN/BTC").myid == 123

    with pytest.raises(InvalidOrderException, match=r"尝试获取无效的模拟交易订单.*"):
        exchange.fetch_order("Y", "TKN/BTC")

    default_conf["dry_run"] = False
    api_mock = MagicMock()
    api_mock.fetch_order = MagicMock(return_value={"id": "123", "amount": 2, "symbol": "TKN/BTC"})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    assert exchange.fetch_order("X", "TKN/BTC") == {"id": "123", "amount": 2, "symbol": "TKN/BTC"}
    assert log_has(("API fetch_order: {'id': '123', 'amount': 2, 'symbol': 'TKN/BTC'}"), caplog)

    with pytest.raises(InvalidOrderException):
        api_mock.fetch_order = MagicMock(side_effect=ccxt.InvalidOrder("订单未找到"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.fetch_order(order_id="_", pair="TKN/BTC")
    assert api_mock.fetch_order.call_count == 1

    api_mock.fetch_order = MagicMock(side_effect=ccxt.OrderNotFound("订单未找到"))
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    with patch("freqtrade.exchange.common.time.sleep") as tm:
        with pytest.raises(InvalidOrderException):
            exchange.fetch_order(order_id="_", pair="TKN/BTC")
        # 确保重试机制被调用
        assert tm.call_args_list[0][0][0] == 1
        assert tm.call_args_list[1][0][0] == 2
        if API_FETCH_ORDER_RETRY_COUNT > 2:
            assert tm.call_args_list[2][0][0] == 5
        if API_FETCH_ORDER_RETRY_COUNT > 3:
            assert tm.call_args_list[3][0][0] == 10
    assert api_mock.fetch_order.call_count == API_FETCH_ORDER_RETRY_COUNT + 1

    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        api_mock,
        exchange_name,
        "fetch_order",
        "fetch_order",
        retries=API_FETCH_ORDER_RETRY_COUNT + 1,
        order_id="_",
        pair="TKN/BTC",
    )


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_fetch_order_emulated(default_conf, mocker, exchange_name, caplog):
    default_conf["dry_run"] = True
    default_conf["exchange"]["log_responses"] = True
    order = MagicMock()
    order.myid = 123
    order.symbol = "TKN/BTC"

    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    mocker.patch(f"{EXMS}.exchange_has", return_value=False)
    exchange._dry_run_open_orders["X"] = order
    # 模拟交易 - 常规fetch_order行为
    assert exchange.fetch_order("X", "TKN/BTC").myid == 123

    with pytest.raises(InvalidOrderException, match=r"尝试获取无效的模拟交易订单.*"):
        exchange.fetch_order("Y", "TKN/BTC")

    default_conf["dry_run"] = False
    mocker.patch(f"{EXMS}.exchange_has", return_value=False)
    api_mock = MagicMock()
    api_mock.fetch_open_order = MagicMock(
        return_value={"id": "123", "amount": 2, "symbol": "TKN/BTC"}
    )
    api_mock.fetch_closed_order = MagicMock(
        return_value={"id": "123", "amount": 2, "symbol": "TKN/BTC"}
    )
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    assert exchange.fetch_order("X", "TKN/BTC") == {"id": "123", "amount": 2, "symbol": "TKN/BTC"}
    assert log_has(
        ("API fetch_open_order: {'id': '123', 'amount': 2, 'symbol': 'TKN/BTC'}"), caplog
    )
    assert api_mock.fetch_open_order.call_count == 1
    assert api_mock.fetch_closed_order.call_count == 0
    caplog.clear()

    # open_order未找到订单
    api_mock.fetch_open_order = MagicMock(side_effect=ccxt.OrderNotFound("订单未找到"))
    api_mock.fetch_closed_order = MagicMock(
        return_value={"id": "123", "amount": 2, "symbol": "TKN/BTC"}
    )
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    assert exchange.fetch_order("X", "TKN/BTC") == {"id": "123", "amount": 2, "symbol": "TKN/BTC"}
    assert log_has(
        ("API fetch_closed_order: {'id': '123', 'amount': 2, 'symbol': 'TKN/BTC'}"), caplog
    )
    assert api_mock.fetch_open_order.call_count == 1
    assert api_mock.fetch_closed_order.call_count == 1
    caplog.clear()

    with pytest.raises(InvalidOrderException):
        api_mock.fetch_open_order = MagicMock(side_effect=ccxt.InvalidOrder("订单未找到"))
        api_mock.fetch_closed_order = MagicMock(side_effect=ccxt.InvalidOrder("订单未找到"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.fetch_order(order_id="_", pair="TKN/BTC")
    assert api_mock.fetch_open_order.call_count == 1

    api_mock.fetch_open_order = MagicMock(side_effect=ccxt.OrderNotFound("订单未找到"))
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)

    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        api_mock,
        exchange_name,
        "fetch_order_emulated",
        "fetch_open_order",
        retries=1,
        order_id="_",
        pair="TKN/BTC",
        params={},
    )


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_fetch_stoploss_order(default_conf, mocker, exchange_name):
    default_conf["dry_run"] = True
    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    order = MagicMock()
    order.myid = 123
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    exchange._dry_run_open_orders["X"] = order
    assert exchange.fetch_stoploss_order("X", "TKN/BTC").myid == 123

    with pytest.raises(InvalidOrderException, match=r"尝试获取无效的模拟交易订单.*"):
        exchange.fetch_stoploss_order("Y", "TKN/BTC")

    default_conf["dry_run"] = False
    api_mock = MagicMock()
    api_mock.fetch_order = MagicMock(return_value={"id": "123", "symbol": "TKN/BTC"})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    res = {"id": "123", "symbol": "TKN/BTC"}
    if exchange_name == "okx":
        res = {"id": "123", "symbol": "TKN/BTC", "type": "stoploss"}
    assert exchange.fetch_stoploss_order("X", "TKN/BTC") == res

    if exchange_name == "okx":
        # 单独测试
        return
    with pytest.raises(InvalidOrderException):
        api_mock.fetch_order = MagicMock(side_effect=ccxt.InvalidOrder("订单未找到"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        exchange.fetch_stoploss_order(order_id="_", pair="TKN/BTC")
    assert api_mock.fetch_order.call_count == 1

    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        api_mock,
        exchange_name,
        "fetch_stoploss_order",
        "fetch_order",
        retries=API_FETCH_ORDER_RETRY_COUNT + 1,
        order_id="_",
        pair="TKN/BTC",
    )


def test_fetch_order_or_stoploss_order(default_conf, mocker):
    exchange = get_patched_exchange(mocker, default_conf, exchange="binance")
    fetch_order_mock = MagicMock()
    fetch_stoploss_order_mock = MagicMock()
    mocker.patch.multiple(
        EXMS,
        fetch_order=fetch_order_mock,
        fetch_stoploss_order=fetch_stoploss_order_mock,
    )

    exchange.fetch_order_or_stoploss_order("1234", "ETH/BTC", False)
    assert fetch_order_mock.call_count == 1
    assert fetch_order_mock.call_args_list[0][0][0] == "1234"
    assert fetch_order_mock.call_args_list[0][0][1] == "ETH/BTC"
    assert fetch_stoploss_order_mock.call_count == 0

    fetch_order_mock.reset_mock()
    fetch_stoploss_order_mock.reset_mock()

    exchange.fetch_order_or_stoploss_order("1234", "ETH/BTC", True)
    assert fetch_order_mock.call_count == 0
    assert fetch_stoploss_order_mock.call_count == 1
    assert fetch_stoploss_order_mock.call_args_list[0][0][0] == "1234"
    assert fetch_stoploss_order_mock.call_args_list[0][0][1] == "ETH/BTC"


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_name(default_conf, mocker, exchange_name):
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)

    assert exchange.name == exchange_name.title()
    assert exchange.id == exchange_name


@pytest.mark.parametrize(
    "trading_mode,amount",
    [
        ("spot", 0.2340606),
        ("futures", 2.340606),
    ],
)
@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_get_trades_for_order(default_conf, mocker, exchange_name, trading_mode, amount):
    order_id = "ABCD-ABCD"
    since = datetime(2018, 5, 5, 0, 0, 0)
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = trading_mode
    default_conf["margin_mode"] = "isolated"
    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    api_mock = MagicMock()

    api_mock.fetch_my_trades = MagicMock(
        return_value=[
            {
                "id": "TTR67E-3PFBD-76IISV",
                "order": "ABCD-ABCD",
                "info": {
                    "pair": "XLTCZBTC",
                    "time": 1519860024.4388,
                    "type": "buy",
                    "ordertype": "limit",
                    "price": "20.00000",
                    "cost": "38.62000",
                    "fee": "0.06179",
                    "vol": "5",
                    "id": "ABCD-ABCD",
                },
                "timestamp": 1519860024438,
                "datetime": "2018-02-28T23:20:24.438Z",
                "symbol": "ETH/USDT:USDT",
                "type": "limit",
                "side": "buy",
                "price": 165.0,
                "amount": 0.2340606,
                "fee": {"cost": 0.06179, "currency": "BTC"},
            }
        ]
    )

    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)

    orders = exchange.get_trades_for_order(order_id, "ETH/USDT:USDT", since)
    assert len(orders) == 1
    assert orders[0]["price"] == 165
    assert pytest.approx(orders[0]["amount"]) == amount
    assert api_mock.fetch_my_trades.call_count == 1
    # since参数应该是
    assert isinstance(api_mock.fetch_my_trades.call_args[0][1], int)
    assert api_mock.fetch_my_trades.call_args[0][0] == "ETH/USDT:USDT"
    # 两次相同测试，硬编码数字和执行相同计算
    assert api_mock.fetch_my_trades.call_args[0][1] == 1525478395000
    assert (
        api_mock.fetch_my_trades.call_args[0][1]
        == int(since.replace(tzinfo=timezone.utc).timestamp() - 5) * 1000
    )

    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        api_mock,
        exchange_name,
        "get_trades_for_order",
        "fetch_my_trades",
        order_id=order_id,
        pair="ETH/USDT:USDT",
        since=since,
    )

    mocker.patch(f"{EXMS}.exchange_has", MagicMock(return_value=False))
    assert exchange.get_trades_for_order(order_id, "ETH/USDT:USDT", since) == []


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_get_fee(default_conf, mocker, exchange_name):
    api_mock = MagicMock()
    api_mock.calculate_fee = MagicMock(
        return_value={"type": "taker", "currency": "BTC", "rate": 0.025, "cost": 0.05}
    )
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    exchange._config.pop("fee", None)

    assert exchange.get_fee("ETH/BTC") == 0.025
    assert api_mock.calculate_fee.call_count == 1

    ccxt_exceptionhandlers(
        mocker, default_conf, api_mock, exchange_name, "get_fee", "calculate_fee", symbol="ETH/BTC"
    )

    api_mock.calculate_fee.reset_mock()
    exchange._config["fee"] = 0.001

    assert exchange.get_fee("ETH/BTC") == 0.001
    assert api_mock.calculate_fee.call_count == 0


def test_stoploss_order_unsupported_exchange(default_conf, mocker):
    exchange = get_patched_exchange(mocker, default_conf, exchange="bitpanda")
    with pytest.raises(OperationalException, match=r"止损未在.*实现"):
        exchange.create_stoploss(
            pair="ETH/BTC", amount=1, stop_price=220, order_types={}, side="sell", leverage=1.0
        )

    with pytest.raises(OperationalException, match=r"止损未在.*实现"):
        exchange.stoploss_adjust(1, {}, side="sell")


@pytest.mark.parametrize(
    "side,ratio,expected",
    [
        ("sell", 0.99, 99.0),  # 默认
        ("sell", 0.999, 99.9),
        ("sell", 1, 100),
        ("sell", 1.1, InvalidOrderException),
        ("buy", 0.99, 101.0),  # 默认
        ("buy", 0.999, 100.1),
        ("buy", 1, 100),
        ("buy", 1.1, InvalidOrderException),
    ],
)
def test__get_stop_limit_rate(default_conf_usdt, mocker, side, ratio, expected):
    exchange = get_patched_exchange(mocker, default_conf_usdt, exchange="binance")

    order_types = {"stoploss_on_exchange_limit_ratio": ratio}
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            exchange._get_stop_limit_rate(100, order_types, side)
    else:
        assert exchange._get_stop_limit_rate(100, order_types, side) == expected


def test_merge_ft_has_dict(default_conf, mocker):
    mocker.patch.multiple(
        EXMS,
        _init_ccxt=MagicMock(return_value=MagicMock()),
        _load_async_markets=MagicMock(),
        validate_timeframes=MagicMock(),
        validate_stakecurrency=MagicMock(),
        validate_pricing=MagicMock(),
    )
    ex = Exchange(default_conf)
    assert ex._ft_has == Exchange._ft_has_default

    ex = Kraken(default_conf)
    assert ex._ft_has != Exchange._ft_has_default
    assert ex.get_option("trades_pagination") == "id"
    assert ex.get_option("trades_pagination_arg") == "since"

    # Binance定义了不同的值
    ex = Binance(default_conf)
    assert ex._ft_has != Exchange._ft_has_default
    assert ex.get_option("stoploss_on_exchange")
    assert ex.get_option("order_time_in_force") == ["GTC", "FOK", "IOC", "PO"]
    assert ex.get_option("trades_pagination") == "id"
    assert ex.get_option("trades_pagination_arg") == "fromId"

    conf = copy.deepcopy(default_conf)
    conf["exchange"]["_ft_has_params"] = {"DeadBeef": 20, "stoploss_on_exchange": False}
    # 使用配置中的设置（覆盖stoploss_on_exchange）
    ex = Binance(conf)
    assert ex._ft_has != Exchange._ft_has_default
    assert not ex._ft_has["stoploss_on_exchange"]
    assert ex._ft_has["DeadBeef"] == 20


def test_get_valid_pair_combination(default_conf, mocker, markets):
    mocker.patch.multiple(
        EXMS,
        _init_ccxt=MagicMock(return_value=MagicMock()),
        _load_async_markets=MagicMock(),
        validate_timeframes=MagicMock(),
        validate_pricing=MagicMock(),
        markets=PropertyMock(return_value=markets),
    )
    ex = Exchange(default_conf)

    assert next(ex.get_valid_pair_combination("ETH", "BTC")) == "ETH/BTC"
    assert next(ex.get_valid_pair_combination("BTC", "ETH")) == "ETH/BTC"
    multicombs = list(ex.get_valid_pair_combination("ETH", "USDT"))
    assert len(multicombs) == 2
    assert "ETH/USDT" in multicombs
    assert "ETH/USDT:USDT" in multicombs

    with pytest.raises(ValueError, match=r"无法组合.*以获得有效交易对。"):
        for x in ex.get_valid_pair_combination("NOPAIR", "ETH"):
            pass


@pytest.mark.parametrize(
    "base_currencies,quote_currencies,tradable_only,active_only,spot_only,"
    "futures_only,expected_keys,test_comment",
    [
        # 测试市场（在conftest.py中）：
        # 'BLK/BTC':  'active': True
        # 'BTT/BTC':  'active': True
        # 'ETH/BTC':  'active': True
        # 'ETH/USDT': 'active': True
        # 'LTC/BTC':  'active': False
        # 'LTC/ETH':  'active': True
        # 'LTC/USD':  'active': True
        # 'LTC/USDT': 'active': True
        # 'NEO/BTC':  'active': False
        # 'TKN/BTC':  'active' 未设置
        # 'XLTCUSDT': 'active': True, 不是交易对
        # 'XRP/BTC':  'active': False
        (
            [],
            [],
            False,
            False,
            False,
            False,
            [
                "BLK/BTC",
                "BTT/BTC",
                "ETH/BTC",
                "ETH/USDT",
                "LTC/BTC",
                "LTC/ETH",
                "LTC/USD",
                "LTC/USDT",
                "NEO/BTC",
                "TKN/BTC",
                "XLTCUSDT",
                "XRP/BTC",
                "ADA/USDT:USDT",
                "ETH/USDT:USDT",
            ],
            "所有市场",
        ),
        (
            [],
            [],
            False,
            False,
            True,
            False,
            [
                "BLK/BTC",
                "BTT/BTC",
                "ETH/BTC",
                "ETH/USDT",
                "LTC/BTC",
                "LTC/ETH",
                "LTC/USD",
                "LTC/USDT",
                "NEO/BTC",
                "TKN/BTC",
                "XRP/BTC",
            ],
            "所有市场，仅现货交易对",
        ),
        (
            [],
            [],
            False,
            True,
            False,
            False,
            [
                "BLK/BTC",
                "ETH/BTC",
                "ETH/USDT",
                "LTC/BTC",
                "LTC/ETH",
                "LTC/USD",
                "NEO/BTC",
                "TKN/BTC",
                "XLTCUSDT",
                "XRP/BTC",
                "ADA/USDT:USDT",
                "ETH/USDT:USDT",
            ],
            "活跃市场",
        ),
        (
            [],
            [],
            True,
            False,
            False,
            False,
            [
                "BLK/BTC",
                "BTT/BTC",
                "ETH/BTC",
                "ETH/USDT",
                "LTC/BTC",
                "LTC/ETH",
                "LTC/USD",
                "LTC/USDT",
                "NEO/BTC",
                "TKN/BTC",
                "XRP/BTC",
            ],
            "所有交易对",
        ),
        (
            [],
            [],
            True,
            True,
            False,
            False,
            [
                "BLK/BTC",
                "ETH/BTC",
                "ETH/USDT",
                "LTC/BTC",
                "LTC/ETH",
                "LTC/USD",
                "NEO/BTC",
                "TKN/BTC",
                "XRP/BTC",
            ],
            "活跃交易对",
        ),
        (
            ["ETH", "LTC"],
            [],
            False,
            False,
            False,
            False,
            [
                "ETH/BTC",
                "ETH/USDT",
                "LTC/BTC",
                "LTC/ETH",
                "LTC/USD",
                "LTC/USDT",
                "XLTCUSDT",
                "ETH/USDT:USDT",
            ],
            "所有市场，基础货币=ETH, LTC",
        ),
        (
            ["LTC"],
            [],
            False,
            False,
            False,
            False,
            ["LTC/BTC", "LTC/ETH", "LTC/USD", "LTC/USDT", "XLTCUSDT"],
            "所有市场，基础货币=LTC",
        ),
        (
            ["LTC"],
            [],
            False,
            False,
            True,
            False,
            ["LTC/BTC", "LTC/ETH", "LTC/USD", "LTC/USDT"],
            "现货市场，基础货币=LTC",
        ),
        (
            [],
            ["USDT"],
            False,
            False,
            False,
            False,
            ["ETH/USDT", "LTC/USDT", "XLTCUSDT", "ADA/USDT:USDT", "ETH/USDT:USDT"],
            "所有市场，计价货币=USDT",
        ),
        (
            [],
            ["USDT"],
            False,
            False,
            False,
            True,
            ["ADA/USDT:USDT", "ETH/USDT:USDT"],
            "期货市场，计价货币=USDT",
        ),
        (
            [],
            ["USDT", "USD"],
            False,
            False,
            False,
            False,
            ["ETH/USDT", "LTC/USD", "LTC/USDT", "XLTCUSDT", "ADA/USDT:USDT", "ETH/USDT:USDT"],
            "所有市场，计价货币=USDT, USD",
        ),
        (
            [],
            ["USDT", "USD"],
            False,
            False,
            True,
            False,
            ["ETH/USDT", "LTC/USD", "LTC/USDT"],
            "现货市场，计价货币=USDT, USD",
        ),
        (
            ["LTC"],
            ["USDT"],
            False,
            False,
            False,
            False,
            ["LTC/USDT", "XLTCUSDT"],
            "所有市场，基础货币=LTC, 计价货币=USDT",
        ),
        (
            ["LTC"],
            ["USDT"],
            True,
            False,
            False,
            False,
            ["LTC/USDT"],
            "所有交易对，基础货币=LTC, 计价货币=USDT",
        ),
        (
            ["LTC"],
            ["USDT", "NONEXISTENT"],
            False,
            False,
            False,
            False,
            ["LTC/USDT", "XLTCUSDT"],
            "所有市场，基础货币=LTC, 计价货币=USDT, NONEXISTENT",
        ),
        (
            ["LTC"],
            ["NONEXISTENT"],
            False,
            False,
            False,
            False,
            [],
            "所有市场，基础货币=LTC, 计价货币=NONEXISTENT",
        ),
    ],
)
def test_get_markets(
    default_conf,
    mocker,
    markets_static,
    base_currencies,
    quote_currencies,
    tradable_only,
    active_only,
    spot_only,
    futures_only,
    expected_keys,
    test_comment,  # 用于调试目的（不在方法内使用）
):
    mocker.patch.multiple(
        EXMS,
        _init_ccxt=MagicMock(return_value=MagicMock()),
        _load_async_markets=MagicMock(),
        validate_timeframes=MagicMock(),
        validate_pricing=MagicMock(),
        markets=PropertyMock(return_value=markets_static),
    )
    ex = Exchange(default_conf)
    pairs = ex.get_markets(
        base_currencies,
        quote_currencies,
        tradable_only=tradable_only,
        spot_only=spot_only,
        futures_only=futures_only,
        active_only=active_only,
    )
    assert sorted(pairs.keys()) == sorted(expected_keys)


def test_get_markets_error(default_conf, mocker):
    ex = get_patched_exchange(mocker, default_conf)
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=None))
    with pytest.raises(OperationalException, match="市场未加载。"):
        ex.get_markets("LTC", "USDT", True, False)


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_ohlcv_candle_limit(default_conf, mocker, exchange_name):
    if exchange_name == "okx":
        pytest.skip("okx单独测试")
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    timeframes = ("1m", "5m", "1h")
    expected = exchange._ft_has.get("ohlcv_candle_limit", 500)
    for timeframe in timeframes:
        # if 'ohlcv_candle_limit_per_timeframe' in exchange._ft_has:
        # expected = exchange._ft_has['ohlcv_candle_limit_per_timeframe'][timeframe]
        # 这应该只对bittrex运行
        # assert exchange_name == 'bittrex'
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.SPOT) == expected


@pytest.mark.parametrize(
    "market_symbol,base,quote,exchange,spot,margin,futures,trademode,add_dict,expected_result",
    [
        ("BTC/USDT", "BTC", "USDT", "binance", True, False, False, "spot", {}, True),
        ("USDT/BTC", "USDT", "BTC", "binance", True, False, False, "spot", {}, True),
        # 没有分隔符/
        ("BTCUSDT", "BTC", "USDT", "binance", True, False, False, "spot", {}, True),
        ("BTCUSDT", None, "USDT", "binance", True, False, False, "spot", {}, False),
        ("USDT/BTC", "BTC", None, "binance", True, False, False, "spot", {}, False),
        ("BTCUSDT", "BTC", None, "binance", True, False, False, "spot", {}, False),
        ("BTC/USDT", "BTC", "USDT", "binance", True, False, False, "spot", {}, True),
        # 期货模式，现货交易对
        ("BTC/USDT", "BTC", "USDT", "binance", True, False, False, "futures", {}, False),
        ("BTC/USDT", "BTC", "USDT", "binance", True, False, False, "margin", {}, False),
        ("BTC/USDT", "BTC", "USDT", "binance", True, True, True, "margin", {}, True),
        ("BTC/USDT", "BTC", "USDT", "binance", False, True, False, "margin", {}, True),
        # 期货模式，期货交易对
        ("BTC/USDT", "BTC", "USDT", "binance", False, False, True, "futures", {}, True),
        # 期货市场
        ("BTC/UNK", "BTC", "UNK", "binance", False, False, True, "spot", {}, False),
        ("BTC/EUR", "BTC", "EUR", "kraken", True, False, False, "spot", {"darkpool": False}, True),
        ("EUR/BTC", "EUR", "BTC", "kraken", True, False, False, "spot", {"darkpool": False}, True),
        # 没有暗池
        ("BTC/EUR", "BTC", "EUR", "kraken", True, False, False, "spot", {"darkpool": True}, False),
        # 没有暗池
        (
            "BTC/EUR.d",
            "BTC",
            "EUR",
            "kraken",
            True,
            False,
            False,
            "spot",
            {"darkpool": True},
            False,
        ),
        ("BTC/USDT:USDT", "BTC", "USD", "okx", False, False, True, "spot", {}, False),
        ("BTC/USDT:USDT", "BTC", "USD", "okx", False, False, True, "margin", {}, False),
        ("BTC/USDT:USDT", "BTC", "USD", "okx", False, False, True, "futures", {}, True),
    ],
)
def test_market_is_tradable(
    mocker,
    default_conf,
    market_symbol,
    base,
    quote,
    spot,
    margin,
    futures,
    trademode,
    add_dict,
    exchange,
    expected_result,
) -> None:
    default_conf["trading_mode"] = trademode
    mocker.patch(f"{EXMS}.validate_trading_mode_and_margin_mode")
    ex = get_patched_exchange(mocker, default_conf, exchange=exchange)
    market = {
        "symbol": market_symbol,
        "type": "swap",
        "base": base,
        "quote": quote,
        "spot": spot,
        "future": futures,
        "swap": futures,
        "margin": margin,
        "linear": True,
        **(add_dict),
    }
    assert ex.market_is_tradable(market) == expected_result


@pytest.mark.parametrize(
    "market,expected_result",
    [
        ({"symbol": "ETH/BTC", "active": True}, True),
        ({"symbol": "ETH/BTC", "active": False}, False),
        (
            {
                "symbol": "ETH/BTC",
            },
            True,
        ),
    ],
)
def test_market_is_active(market, expected_result) -> None:
    assert market_is_active(market) == expected_result


@pytest.mark.parametrize(
    "order,expected",
    [
        ([{"fee"}], False),
        ({"fee": None}, False),
        ({"fee": {"currency": "ETH/BTC"}}, False),
        ({"fee": {"currency": "ETH/BTC", "cost": None}}, False),
        ({"fee": {"currency": "ETH/BTC", "cost": 0.01}}, True),
    ],
)
def test_order_has_fee(order, expected) -> None:
    assert Exchange.order_has_fee(order) == expected


@pytest.mark.parametrize(
    "order,expected",
    [
        ({"symbol": "ETH/BTC", "fee": {"currency": "ETH", "cost": 0.43}}, (0.43, "ETH", 0.01)),
        ({"symbol": "ETH/USDT", "fee": {"currency": "USDT", "cost": 0.01}}, (0.01, "USDT", 0.01)),
        (
            {"symbol": "BTC/USDT", "fee": {"currency": "USDT", "cost": 0.34, "rate": 0.01}},
            (0.34, "USDT", 0.01),
        ),
    ],
)
def test_extract_cost_curr_rate(mocker, default_conf, order, expected) -> None:
    mocker.patch(f"{EXMS}.calculate_fee_rate", MagicMock(return_value=0.01))
    ex = get_patched_exchange(mocker, default_conf)
    assert ex.extract_cost_curr_rate(order["fee"], order["symbol"], cost=20, amount=1) == expected


@pytest.mark.parametrize(
    "order,unknown_fee_rate,expected",
    [
        # 使用基础货币
        (
            {
                "symbol": "ETH/BTC",
                "amount": 0.04,
                "cost": 0.05,
                "fee": {"currency": "ETH", "cost": 0.004, "rate": None},
            },
            None,
            0.1,
        ),
        (
            {
                "symbol": "ETH/BTC",
                "amount": 0.05,
                "cost": 0.05,
                "fee": {"currency": "ETH", "cost": 0.004, "rate": None},
            },
            None,
            0.08,
        ),
        # 使用计价货币
        (
            {
                "symbol": "ETH/BTC",
                "amount": 0.04,
                "cost": 0.05,
                "fee": {"currency": "BTC", "cost": 0.005},
            },
            None,
            0.1,
        ),
        (
            {
                "symbol": "ETH/BTC",
                "amount": 0.04,
                "cost": 0.05,
                "fee": {"currency": "BTC", "cost": 0.002, "rate": None},
            },
            None,
            0.04,
        ),
        # 使用外币
        (
            {
                "symbol": "ETH/BTC",
                "amount": 0.04,
                "cost": 0.05,
                "fee": {"currency": "NEO", "cost": 0.0012},
            },
            None,
            0.001944,
        ),
        (
            {
                "symbol": "ETH/BTC",
                "amount": 2.21,
                "cost": 0.02992561,
                "fee": {"currency": "NEO", "cost": 0.00027452},
            },
            None,
            0.00074305,
        ),
        # 费率包含在返回中 - 按原样返回
        (
            {
                "symbol": "ETH/BTC",
                "amount": 0.04,
                "cost": 0.05,
                "fee": {"currency": "USDT", "cost": 0.34, "rate": 0.01},
            },
            None,
            0.01,
        ),
        (
            {
                "symbol": "ETH/BTC",
                "amount": 0.04,
                "cost": 0.05,
                "fee": {"currency": "USDT", "cost": 0.34, "rate": 0.005},
            },
            None,
            0.005,
        ),
        # 0.1% 成交 - 无成本 (kraken - #3431)
        (
            {
                "symbol": "ETH/BTC",
                "amount": 0.04,
                "cost": 0.0,
                "fee": {"currency": "BTC", "cost": 0.0, "rate": None},
            },
            None,
            None,
        ),
        (
            {
                "symbol": "ETH/BTC",
                "amount": 0.04,
                "cost": 0.0,
                "fee": {"currency": "ETH", "cost": 0.0, "rate": None},
            },
            None,
            0.0,
        ),
        (
            {
                "symbol": "ETH/BTC",
                "amount": 0.04,
                "cost": 0.0,
                "fee": {"currency": "NEO", "cost": 0.0, "rate": None},
            },
            None,
            None,
        ),
        # 无效交易对组合 - POINT/BTC不是交易对
        (
            {
                "symbol": "POINT/BTC",
                "amount": 0.04,
                "cost": 0.5,
                "fee": {"currency": "POINT", "cost": 2.0, "rate": None},
            },
            None,
            None,
        ),
        (
            {
                "symbol": "POINT/BTC",
                "amount": 0.04,
                "cost": 0.5,
                "fee": {"currency": "POINT", "cost": 2.0, "rate": None},
            },
            1,
            4.0,
        ),
        (
            {
                "symbol": "POINT/BTC",
                "amount": 0.04,
                "cost": 0.5,
                "fee": {"currency": "POINT", "cost": 2.0, "rate": None},
            },
            2,
            8.0,
        ),
        # 缺少货币
        (
            {
                "symbol": "ETH/BTC",
                "amount": 0.04,
                "cost": 0.05,
                "fee": {"currency": None, "cost": 0.005},
            },
            None,
            None,
        ),
    ],
)
def test_calculate_fee_rate(mocker, default_conf, order, expected, unknown_fee_rate) -> None:
    mocker.patch(f"{EXMS}.get_tickers", return_value={"NEO/BTC": {"last": 0.081}})
    if unknown_fee_rate:
        default_conf["exchange"]["unknown_fee_rate"] = unknown_fee_rate

    ex = get_patched_exchange(mocker, default_conf)

    assert (
        ex.calculate_fee_rate(
            order["fee"], order["symbol"], cost=order["cost"], amount=order["amount"]
        )
        == expected
    )


@pytest.mark.parametrize(
    "retrycount,max_retries,expected",
    [
        (0, 3, 10),
        (1, 3, 5),
        (2, 3, 2),
        (3, 3, 1),
        (0, 1, 2),
        (1, 1, 1),
        (0, 4, 17),
        (1, 4, 10),
        (2, 4, 5),
        (3, 4, 2),
        (4, 4, 1),
        (0, 5, 26),
        (1, 5, 17),
        (2, 5, 10),
        (3, 5, 5),
        (4, 5, 2),
        (5, 5, 1),
    ],
)
def test_calculate_backoff(retrycount, max_retries, expected):
    assert calculate_backoff(retrycount, max_retries) == expected


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_get_funding_fees(default_conf_usdt, mocker, exchange_name, caplog):
    now = datetime.now(timezone.utc)
    default_conf_usdt["trading_mode"] = "futures"
    default_conf_usdt["margin_mode"] = "isolated"
    exchange = get_patched_exchange(mocker, default_conf_usdt, exchange=exchange_name)
    exchange._fetch_and_calculate_funding_fees = MagicMock(side_effect=ExchangeError)
    assert exchange.get_funding_fees("BTC/USDT:USDT", 1, False, now) == 0.0
    assert exchange._fetch_and_calculate_funding_fees.call_count == 1
    assert log_has("无法更新BTC/USDT:USDT的资金费用。", caplog)


@pytest.mark.parametrize("exchange_name", ["binance"])
def test__get_funding_fees_from_exchange(default_conf, mocker, exchange_name):
    api_mock = MagicMock()
    api_mock.fetch_funding_history = MagicMock(
        return_value=[
            {
                "amount": 0.14542,
                "code": "USDT",
                "datetime": "2021-09-01T08:00:01.000Z",
                "id": "485478",
                "info": {
                    "asset": "USDT",
                    "income": "0.14542",
                    "incomeType": "FUNDING_FEE",
                    "info": "FUNDING_FEE",
                    "symbol": "XRPUSDT",
                    "time": "1630382001000",
                    "tradeId": "",
                    "tranId": "993203",
                },
                "symbol": "XRP/USDT",
                "timestamp": 1630382001000,
            },
            {
                "amount": -0.14642,
                "code": "USDT",
                "datetime": "2021-09-01T16:00:01.000Z",
                "id": "485479",
                "info": {
                    "asset": "USDT",
                    "income": "-0.14642",
                    "incomeType": "FUNDING_FEE",
                    "info": "FUNDING_FEE",
                    "symbol": "XRPUSDT",
                    "time": "1630314001000",
                    "tradeId": "",
                    "tranId": "993204",
                },
                "symbol": "XRP/USDT",
                "timestamp": 1630314001000,
            },
        ]
    )
    type(api_mock).has = PropertyMock(return_value={"fetchFundingHistory": True})

    # mocker.patch(f'{EXMS}.get_funding_fees', lambda pair, since: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    date_time = datetime.strptime("2021-09-01T00:00:01.000Z", "%Y-%m-%dT%H:%M:%S.%fZ")
    unix_time = int(date_time.timestamp())
    expected_fees = -0.001  # 0.14542341 + -0.14642341
    fees_from_datetime = exchange._get_funding_fees_from_exchange(pair="XRP/USDT", since=date_time)
    fees_from_unix_time = exchange._get_funding_fees_from_exchange(pair="XRP/USDT", since=unix_time)

    assert pytest.approx(expected_fees) == fees_from_datetime
    assert pytest.approx(expected_fees) == fees_from_unix_time

    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        api_mock,
        exchange_name,
        "_get_funding_fees_from_exchange",
        "fetch_funding_history",
        pair="XRP/USDT",
        since=unix_time,
    )


@pytest.mark.parametrize("exchange", ["binance", "kraken"])
@pytest.mark.parametrize(
    "stake_amount,leverage,min_stake_with_lev",
    [(9.0, 3.0, 3.0), (20.0, 5.0, 4.0), (100.0, 100.0, 1.0)],
)
def test_get_stake_amount_considering_leverage(
    exchange, stake_amount, leverage, min_stake_with_lev, mocker, default_conf
):
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange)
    assert (
        exchange._get_stake_amount_considering_leverage(stake_amount, leverage)
        == min_stake_with_lev
    )


@pytest.mark.parametrize("margin_mode", [(MarginMode.CROSS), (MarginMode.ISOLATED)])
def test_set_margin_mode(mocker, default_conf, margin_mode):
    api_mock = MagicMock()
    api_mock.set_margin_mode = MagicMock()
    type(api_mock).has = PropertyMock(return_value={"setMarginMode": True})
    default_conf["dry_run"] = False

    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        api_mock,
        "binance",
        "set_margin_mode",
        "set_margin_mode",
        pair="XRP/USDT",
        margin_mode=margin_mode,
    )


@pytest.mark.parametrize(
    "exchange_name, trading_mode, margin_mode, exception_thrown",
    [
        ("binance", TradingMode.SPOT, None, False),
        ("binance", TradingMode.MARGIN, MarginMode.ISOLATED, True),
        ("kraken", TradingMode.SPOT, None, False),
        ("kraken", TradingMode.MARGIN, MarginMode.ISOLATED, True),
        ("kraken", TradingMode.FUTURES, MarginMode.ISOLATED, True),
        ("bitmart", TradingMode.SPOT, None, False),
        ("bitmart", TradingMode.MARGIN, MarginMode.CROSS, True),
        ("bitmart", TradingMode.MARGIN, MarginMode.ISOLATED, True),
        ("bitmart", TradingMode.FUTURES, MarginMode.CROSS, True),
        ("bitmart", TradingMode.FUTURES, MarginMode.ISOLATED, True),
        ("gate", TradingMode.MARGIN, MarginMode.ISOLATED, True),
        ("okx", TradingMode.SPOT, None, False),
        ("okx", TradingMode.MARGIN, MarginMode.CROSS, True),
        ("okx", TradingMode.MARGIN, MarginMode.ISOLATED, True),
        ("okx", TradingMode.FUTURES, MarginMode.CROSS, True),
        ("binance", TradingMode.FUTURES, MarginMode.ISOLATED, False),
        ("gate", TradingMode.FUTURES, MarginMode.ISOLATED, False),
        ("okx", TradingMode.FUTURES, MarginMode.ISOLATED, False),
        # * 实现后移除
        ("binance", TradingMode.MARGIN, MarginMode.CROSS, True),
        ("binance", TradingMode.FUTURES, MarginMode.CROSS, False),
        ("kraken", TradingMode.MARGIN, MarginMode.CROSS, True),
        ("kraken", TradingMode.FUTURES, MarginMode.CROSS, True),
        ("gate", TradingMode.MARGIN, MarginMode.CROSS, True),
        ("gate", TradingMode.FUTURES, MarginMode.CROSS, True),
        # * 实现后取消注释
        # ("binance", TradingMode.MARGIN, MarginMode.CROSS, False),
        # ("binance", TradingMode.FUTURES, MarginMode.CROSS, False),
        # ("kraken", TradingMode.MARGIN, MarginMode.CROSS, False),
        # ("kraken", TradingMode.FUTURES, MarginMode.CROSS, False),
        # ("gate", TradingMode.MARGIN, MarginMode.CROSS, False),
        # ("gate", TradingMode.FUTURES, MarginMode.CROSS, False),
    ],
)
def test_validate_trading_mode_and_margin_mode(
    default_conf, mocker, exchange_name, trading_mode, margin_mode, exception_thrown
):
    exchange = get_patched_exchange(
        mocker, default_conf, exchange=exchange_name, mock_supported_modes=False
    )
    if exception_thrown:
        with pytest.raises(OperationalException):
            exchange.validate_trading_mode_and_margin_mode(trading_mode, margin_mode)
    else:
        exchange.validate_trading_mode_and_margin_mode(trading_mode, margin_mode)


@pytest.mark.parametrize(
    "exchange_name,trading_mode,ccxt_config",
    [
        ("binance", "spot", {}),
        ("binance", "margin", {"options": {"defaultType": "margin"}}),
        ("binance", "futures", {"options": {"defaultType": "swap"}}),
        ("bybit", "spot", {"options": {"defaultType": "spot"}}),
        ("bybit", "futures", {"options": {"defaultType": "swap"}}),
        ("gate", "futures", {"options": {"defaultType": "swap"}}),
        ("hitbtc", "futures", {"options": {"defaultType": "swap"}}),
        ("kraken", "futures", {"options": {"defaultType": "swap"}}),
        ("kucoin", "futures", {"options": {"defaultType": "swap"}}),
        ("okx", "futures", {"options": {"defaultType": "swap"}}),
    ],
)
def test__ccxt_config(default_conf, mocker, exchange_name, trading_mode, ccxt_config):
    default_conf["trading_mode"] = trading_mode
    default_conf["margin_mode"] = "isolated"
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    assert exchange._ccxt_config == ccxt_config


@pytest.mark.parametrize(
    "pair,nominal_value,max_lev",
    [
        ("ETH/BTC", 0.0, 2.0),
        ("TKN/BTC", 100.0, 5.0),
        ("BLK/BTC", 173.31, 3.0),
        ("LTC/BTC", 0.0, 1.0),
        ("TKN/USDT", 210.30, 1.0),
    ],
)
def test_get_max_leverage_from_margin(default_conf, mocker, pair, nominal_value, max_lev):
    default_conf["trading_mode"] = "margin"
    default_conf["margin_mode"] = "isolated"
    api_mock = MagicMock()
    type(api_mock).has = PropertyMock(return_value={"fetchLeverageTiers": False})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="gate")
    assert exchange.get_max_leverage(pair, nominal_value) == max_lev


@pytest.mark.parametrize(
    "size,funding_rate,mark_price,time_in_ratio,funding_fee,kraken_fee",
    [
        (10, 0.0001, 2.0, 1.0, 0.002, 0.002),
        (10, 0.0002, 2.0, 0.01, 0.004, 0.00004),
        (10, 0.0002, 2.5, None, 0.005, None),
        (10, 0.0002, nan, None, 0.0, None),
    ],
)
  
def test_calculate_funding_fees(
    default_conf, mocker, size, funding_rate, mark_price, funding_fee, kraken_fee, time_in_ratio
):
    exchange = get_patched_exchange(mocker, default_conf)
    kraken = get_patched_exchange(mocker, default_conf, exchange="kraken")
    prior_date = timeframe_to_prev_date("1h", datetime.now(timezone.utc) - timedelta(hours=1))
    trade_date = timeframe_to_prev_date("1h", datetime.now(timezone.utc))
    funding_rates = DataFrame(
        [
            {"date": prior_date, "open": funding_rate},  # 未使用的行
            {"date": trade_date, "open": funding_rate},
        ]
    )
    mark_rates = DataFrame(
        [
            {"date": prior_date, "open": mark_price},
            {"date": trade_date, "open": mark_price},
        ]
    )
    df = exchange.combine_funding_and_mark(funding_rates, mark_rates)

    assert (
        exchange.calculate_funding_fees(
            df,
            amount=size,  # 数量
            is_short=True,  # 是否做空
            open_date=trade_date,  # 开仓日期
            close_date=trade_date,  # 平仓日期
            time_in_ratio=time_in_ratio,  # 时间占比
        )
        == funding_fee
    )

    if kraken_fee is None:
        with pytest.raises(OperationalException):  # 操作异常
            kraken.calculate_funding_fees(
                df,
                amount=size,
                is_short=True,
                open_date=trade_date,
                close_date=trade_date,
                time_in_ratio=time_in_ratio,
            )

    else:
        assert (
            kraken.calculate_funding_fees(
                df,
                amount=size,
                is_short=True,
                open_date=trade_date,
                close_date=trade_date,
                time_in_ratio=time_in_ratio,
            )
            == kraken_fee
        )


@pytest.mark.parametrize(
    "mark_price,funding_rate,futures_funding_rate",
    [
        (1000, 0.001, None),
        (1000, 0.001, 0.01),
        (1000, 0.001, 0.0),
        (1000, 0.001, -0.01),
    ],
)
def test_combine_funding_and_mark(
    default_conf,
    mocker,
    funding_rate,
    mark_price,
    futures_funding_rate,
):
    exchange = get_patched_exchange(mocker, default_conf)
    prior2_date = timeframe_to_prev_date("1h", datetime.now(timezone.utc) - timedelta(hours=2))
    prior_date = timeframe_to_prev_date("1h", datetime.now(timezone.utc) - timedelta(hours=1))
    trade_date = timeframe_to_prev_date("1h", datetime.now(timezone.utc))
    funding_rates = DataFrame(
        [
            {"date": prior2_date, "open": funding_rate},
            {"date": prior_date, "open": funding_rate},
            {"date": trade_date, "open": funding_rate},
        ]
    )
    mark_rates = DataFrame(
        [
            {"date": prior2_date, "open": mark_price},
            {"date": prior_date, "open": mark_price},
            {"date": trade_date, "open": mark_price},
        ]
    )

    df = exchange.combine_funding_and_mark(funding_rates, mark_rates, futures_funding_rate)
    assert "open_mark" in df.columns  # 标记价格开盘价
    assert "open_fund" in df.columns  # 资金费率开盘价
    assert len(df) == 3

    funding_rates = DataFrame(
        [
            {"date": trade_date, "open": funding_rate},
        ]
    )
    mark_rates = DataFrame(
        [
            {"date": prior2_date, "open": mark_price},
            {"date": prior_date, "open": mark_price},
            {"date": trade_date, "open": mark_price},
        ]
    )
    df = exchange.combine_funding_and_mark(funding_rates, mark_rates, futures_funding_rate)

    if futures_funding_rate is not None:
        assert len(df) == 3
        assert df.iloc[0]["open_fund"] == futures_funding_rate
        assert df.iloc[1]["open_fund"] == futures_funding_rate
        assert df.iloc[2]["open_fund"] == funding_rate
    else:
        assert len(df) == 1

    # 空的资金费率数据
    funding_rates2 = DataFrame([], columns=["date", "open"])
    df = exchange.combine_funding_and_mark(funding_rates2, mark_rates, futures_funding_rate)
    if futures_funding_rate is not None:
        assert len(df) == 3
        assert df.iloc[0]["open_fund"] == futures_funding_rate
        assert df.iloc[1]["open_fund"] == futures_funding_rate
        assert df.iloc[2]["open_fund"] == futures_funding_rate
    else:
        assert len(df) == 0

    # 空的标记K线
    mark_candles = DataFrame([], columns=["date", "open"])
    df = exchange.combine_funding_and_mark(funding_rates, mark_candles, futures_funding_rate)

    assert len(df) == 0


@pytest.mark.parametrize(
    "exchange,rate_start,rate_end,d1,d2,amount,expected_fees",
    [
        ("binance", 0, 2, "2021-09-01 01:00:00", "2021-09-01 04:00:00", 30.0, 0.0),
        ("binance", 0, 2, "2021-09-01 00:00:00", "2021-09-01 08:00:00", 30.0, -0.00091409999),
        ("binance", 0, 2, "2021-09-01 00:00:15", "2021-09-01 08:00:00", 30.0, -0.0002493),
        ("binance", 1, 2, "2021-09-01 01:00:14", "2021-09-01 08:00:00", 30.0, -0.0002493),
        ("binance", 1, 2, "2021-09-01 00:00:16", "2021-09-01 08:00:00", 30.0, -0.0002493),
        ("binance", 0, 1, "2021-09-01 00:00:00", "2021-09-01 07:59:59", 30.0, -0.00066479999),
        ("binance", 0, 2, "2021-09-01 00:00:00", "2021-09-01 12:00:00", 30.0, -0.00091409999),
        # :01必须向下取整
        ("binance", 0, 2, "2021-09-01 00:00:01", "2021-09-01 08:00:00", 30.0, -0.00091409999),
        ("binance", 0, 2, "2021-08-31 23:58:00", "2021-09-01 08:00:00", 30.0, -0.00091409999),
        ("binance", 0, 2, "2021-09-01 00:10:01", "2021-09-01 08:00:00", 30.0, -0.0002493),
        # TODO: 一旦_calculate_funding_fees可以将time_in_ratio传递给交易所，就取消注释
        # ('kraken', "2021-09-01 00:00:00", "2021-09-01 08:00:00",  30.0, -0.0014937),
        # ('kraken', "2021-09-01 00:00:15", "2021-09-01 08:00:00",  30.0, -0.0008289),
        # ('kraken', "2021-09-01 01:00:14", "2021-09-01 08:00:00",  30.0, -0.0008289),
        # ('kraken', "2021-09-01 00:00:00", "2021-09-01 07:59:59",  30.0, -0.0012443999999999999),
        # ('kraken', "2021-09-01 00:00:00", "2021-09-01 12:00:00", 30.0,  0.0045759),
        # ('kraken', "2021-09-01 00:00:01", "2021-09-01 08:00:00",  30.0, -0.0008289),
        ("gate", 0, 2, "2021-09-01 00:10:00", "2021-09-01 04:00:00", 30.0, 0.0),
        ("gate", 0, 2, "2021-09-01 00:00:00", "2021-09-01 08:00:00", 30.0, -0.0009140999),
        ("gate", 0, 2, "2021-09-01 00:00:00", "2021-09-01 12:00:00", 30.0, -0.0009140999),
        ("gate", 1, 2, "2021-09-01 00:00:01", "2021-09-01 08:00:00", 30.0, -0.0002493),
        ("binance", 0, 2, "2021-09-01 00:00:00", "2021-09-01 08:00:00", 50.0, -0.0015235),
        # TODO: 一旦_calculate_funding_fees可以将time_in_ratio传递给交易所，就取消注释
        # ('kraken', "2021-09-01 00:00:00", "2021-09-01 08:00:00",  50.0, -0.0024895),
    ],
)
def test__fetch_and_calculate_funding_fees(
    mocker,
    default_conf,
    funding_rate_history_hourly,
    funding_rate_history_octohourly,
    rate_start,
    rate_end,
    mark_ohlcv,
    exchange,
    d1,
    d2,
    amount,
    expected_fees,
):
    """
    名义价值 = 标记价格 * 合约数量
    资金费用 = 名义价值 * 资金费率
    size: 30
        time: 0, mark: 2.77, nominal_value: 83.1, fundRate: -0.000008, fundFee: -0.0006648
        time: 1, mark: 2.73, nominal_value: 81.9, fundRate: -0.000004, fundFee: -0.0003276
        time: 2, mark: 2.74, nominal_value: 82.2, fundRate: 0.000012, fundFee: 0.0009864
        time: 3, mark: 2.76, nominal_value: 82.8, fundRate: -0.000003, fundFee: -0.0002484
        time: 4, mark: 2.76, nominal_value: 82.8, fundRate: -0.000007, fundFee: -0.0005796
        time: 5, mark: 2.77, nominal_value: 83.1, fundRate: 0.000003, fundFee: 0.0002493
        time: 6, mark: 2.78, nominal_value: 83.4, fundRate: 0.000019, fundFee: 0.0015846
        time: 7, mark: 2.78, nominal_value: 83.4, fundRate: 0.000003, fundFee: 0.0002502
        time: 8, mark: 2.77, nominal_value: 83.1, fundRate: -0.000003, fundFee: -0.0002493
        time: 9, mark: 2.77, nominal_value: 83.1, fundRate: 0, fundFee: 0.0
        time: 10, mark: 2.84, nominal_value: 85.2, fundRate: 0.000013, fundFee: 0.0011076
        time: 11, mark: 2.81, nominal_value: 84.3, fundRate: 0.000077, fundFee: 0.0064911
        time: 12, mark: 2.81, nominal_value: 84.3, fundRate: 0.000072, fundFee: 0.0060696
        time: 13, mark: 2.82, nominal_value: 84.6, fundRate: 0.000097, fundFee: 0.0082062

    size: 50
        time: 0, mark: 2.77, nominal_value: 138.5, fundRate: -0.000008, fundFee: -0.001108
        time: 1, mark: 2.73, nominal_value: 136.5, fundRate: -0.000004, fundFee: -0.000546
        time: 2, mark: 2.74, nominal_value: 137.0, fundRate: 0.000012, fundFee: 0.001644
        time: 3, mark: 2.76, nominal_value: 138.0, fundRate: -0.000003, fundFee: -0.000414
        time: 4, mark: 2.76, nominal_value: 138.0, fundRate: -0.000007, fundFee: -0.000966
        time: 5, mark: 2.77, nominal_value: 138.5, fundRate: 0.000003, fundFee: 0.0004155
        time: 6, mark: 2.78, nominal_value: 139.0, fundRate: 0.000019, fundFee: 0.002641
        time: 7, mark: 2.78, nominal_value: 139.0, fundRate: 0.000003, fundFee: 0.000417
        time: 8, mark: 2.77, nominal_value: 138.5, fundRate: -0.000003, fundFee: -0.0004155
        time: 9, mark: 2.77, nominal_value: 138.5, fundRate: 0, fundFee: 0.0
        time: 10, mark: 2.84, nominal_value: 142.0, fundRate: 0.000013, fundFee: 0.001846
        time: 11, mark: 2.81, nominal_value: 140.5, fundRate: 0.000077, fundFee: 0.0108185
        time: 12, mark: 2.81, nominal_value: 140.5, fundRate: 0.000072, fundFee: 0.010116
        time: 13, mark: 2.82, nominal_value: 141.0, fundRate: 0.000097, fundFee: 0.013677
    """
    d1 = datetime.strptime(f"{d1} +0000", "%Y-%m-%d %H:%M:%S %z")
    d2 = datetime.strptime(f"{d2} +0000", "%Y-%m-%d %H:%M:%S %z")
    funding_rate_history = {
        "binance": funding_rate_history_octohourly,
        "gate": funding_rate_history_octohourly,
    }[exchange][rate_start:rate_end]
    api_mock = MagicMock()
    api_mock.fetch_funding_rate_history = get_mock_coro(return_value=funding_rate_history)
    api_mock.fetch_ohlcv = get_mock_coro(return_value=mark_ohlcv)
    type(api_mock).has = PropertyMock(return_value={"fetchOHLCV": True})
    type(api_mock).has = PropertyMock(return_value={"fetchFundingRateHistory": True})

    ex = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange)
    mocker.patch(f"{EXMS}.timeframes", PropertyMock(return_value=["1h", "4h", "8h"]))
    funding_fees = ex._fetch_and_calculate_funding_fees(
        pair="ADA/USDT:USDT", amount=amount, is_short=True, open_date=d1, close_date=d2
    )
    assert pytest.approx(funding_fees) == expected_fees
    # 多单费用与空单相反
    funding_fees = ex._fetch_and_calculate_funding_fees(
        pair="ADA/USDT:USDT", amount=amount, is_short=False, open_date=d1, close_date=d2
    )
    assert pytest.approx(funding_fees) == -expected_fees

    # 返回空的"refresh_latest"
    mocker.patch(f"{EXMS}.refresh_latest_ohlcv", return_value={})
    ex = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange)
    with pytest.raises(ExchangeError, match="无法找到资金费率。"):
        ex._fetch_and_calculate_funding_fees(
            pair="ADA/USDT:USDT", amount=amount, is_short=False, open_date=d1, close_date=d2
        )


@pytest.mark.parametrize(
    "exchange,expected_fees",
    [
        ("binance", -0.0009140999999999999),
        ("gate", -0.0009140999999999999),
    ],
)
def test__fetch_and_calculate_funding_fees_datetime_called(
    mocker,
    default_conf,
    funding_rate_history_octohourly,
    mark_ohlcv,
    exchange,
    time_machine,
    expected_fees,
):
    api_mock = MagicMock()
    api_mock.fetch_ohlcv = get_mock_coro(return_value=mark_ohlcv)
    api_mock.fetch_funding_rate_history = get_mock_coro(
        return_value=funding_rate_history_octohourly
    )
    type(api_mock).has = PropertyMock(return_value={"fetchOHLCV": True})
    type(api_mock).has = PropertyMock(return_value={"fetchFundingRateHistory": True})
    mocker.patch(f"{EXMS}.timeframes", PropertyMock(return_value=["4h", "8h"]))
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange)
    d1 = datetime.strptime("2021-08-31 23:00:01 +0000", "%Y-%m-%d %H:%M:%S %z")

    time_machine.move_to("2021-09-01 08:00:00 +00:00")
    funding_fees = exchange._fetch_and_calculate_funding_fees("ADA/USDT", 30.0, True, d1)
    assert funding_fees == expected_fees
    funding_fees = exchange._fetch_and_calculate_funding_fees("ADA/USDT", 30.0, False, d1)
    assert funding_fees == 0 - expected_fees


@pytest.mark.parametrize(
    "pair,expected_size,trading_mode",
    [
        ("XLTCUSDT", 1, "spot"),
        ("LTC/USD", 1, "futures"),
        ("XLTCUSDT", 0.01, "futures"),
        ("ETH/USDT:USDT", 10, "futures"),
        ("TORN/USDT:USDT", None, "futures"),  # 对不可用的交易对不报错
    ],
)
def test__get_contract_size(mocker, default_conf, pair, expected_size, trading_mode):
    api_mock = MagicMock()
    default_conf["trading_mode"] = trading_mode
    default_conf["margin_mode"] = "isolated"
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    mocker.patch(
        f"{EXMS}.markets",
        {
            "LTC/USD": {
                "symbol": "LTC/USD",
                "contractSize": None,
            },
            "XLTCUSDT": {
                "symbol": "XLTCUSDT",
                "contractSize": "0.01",
            },
            "ETH/USDT:USDT": {
                "symbol": "ETH/USDT:USDT",
                "contractSize": "10",
            },
        },
    )
    size = exchange.get_contract_size(pair)
    assert expected_size == size


@pytest.mark.parametrize(
    "pair,contract_size,trading_mode",
    [
        ("XLTCUSDT", 1, "spot"),
        ("LTC/USD", 1, "futures"),
        ("ADA/USDT:USDT", 0.01, "futures"),
        ("LTC/ETH", 1, "futures"),
        ("ETH/USDT:USDT", 10, "futures"),
    ],
)
def test__order_contracts_to_amount(
    mocker,
    default_conf,
    markets,
    pair,
    contract_size,
    trading_mode,
):
    api_mock = MagicMock()
    default_conf["trading_mode"] = trading_mode
    default_conf["margin_mode"] = "isolated"
    mocker.patch(f"{EXMS}.markets", markets)
    exchange = get_patched_exchange(mocker, default_conf, api_mock)

    orders = [
        {
            "id": "123456320",
            "clientOrderId": "12345632018",
            "timestamp": 1640124992000,
            "datetime": "Tue 21 Dec 2021 22:16:32 UTC",
            "lastTradeTimestamp": 1640124911000,
            "status": "active",
            "symbol": pair,
            "type": "limit",
            "timeInForce": "gtc",
            "postOnly": None,
            "side": "buy",
            "price": 2.0,
            "stopPrice": None,
            "average": None,
            "amount": 30.0,
            "cost": 60.0,
            "filled": None,
            "remaining": 30.0,
            "fee": {
                "currency": "USDT",
                "cost": 0.06,
            },
            "fees": [
                {
                    "currency": "USDT",
                    "cost": 0.06,
                }
            ],
            "trades": None,
            "info": {},
        },
        {
            "id": "123456380",
            "clientOrderId": "12345638203",
            "timestamp": 1640124992000,
            "datetime": "Tue 21 Dec 2021 22:16:32 UTC",
            "lastTradeTimestamp": 1640124911000,
            "status": "active",
            "symbol": pair,
            "type": "limit",
            "timeInForce": "gtc",
            "postOnly": None,
            "side": "sell",
            "price": 2.2,
            "stopPrice": None,
            "average": None,
            "amount": 40.0,
            "cost": 80.0,
            "filled": None,
            "remaining": 40.0,
            "fee": {
                "currency": "USDT",
                "cost": 0.08,
            },
            "fees": [
                {
                    "currency": "USDT",
                    "cost": 0.08,
                }
            ],
            "trades": None,
            "info": {},
        },
        {
            # Gate交易所的实际止损订单
            "id": "123456380",
            "clientOrderId": "12345638203",
            "timestamp": None,
            "datetime": None,
            "lastTradeTimestamp": None,
            "status": None,
            "symbol": None,
            "type": None,
            "timeInForce": None,
            "postOnly": None,
            "side": None,
            "price": None,
            "stopPrice": None,
            "average": None,
            "amount": None,
            "cost": None,
            "filled": None,
            "remaining": None,
            "fee": None,
            "fees": [],
            "trades": None,
            "info": {},
        },
    ]
    order1_bef = orders[0]
    order2_bef = orders[1]
    order1 = exchange._order_contracts_to_amount(deepcopy(order1_bef))
    order2 = exchange._order_contracts_to_amount(deepcopy(order2_bef))
    assert order1["amount"] == order1_bef["amount"] * contract_size
    assert order1["cost"] == order1_bef["cost"] * contract_size

    assert order2["amount"] == order2_bef["amount"] * contract_size
    assert order2["cost"] == order2_bef["cost"] * contract_size

    # 不报错
    exchange._order_contracts_to_amount(orders[2])


@pytest.mark.parametrize(
    "pair,contract_size,trading_mode",
    [
        ("XLTCUSDT", 1, "spot"),
        ("LTC/USD", 1, "futures"),
        ("ADA/USDT:USDT", 0.01, "futures"),
        ("LTC/ETH", 1, "futures"),
        ("ETH/USDT:USDT", 10, "futures"),
    ],
)
def test__trades_contracts_to_amount(
    mocker,
    default_conf,
    markets,
    pair,
    contract_size,
    trading_mode,
):
    api_mock = MagicMock()
    default_conf["trading_mode"] = trading_mode
    default_conf["margin_mode"] = "isolated"
    mocker.patch(f"{EXMS}.markets", markets)
    exchange = get_patched_exchange(mocker, default_conf, api_mock)

    trades = [
        {
            "symbol": pair,
            "amount": 30.0,
        },
        {
            "symbol": pair,
            "amount": 40.0,
        },
    ]

    new_amount_trades = exchange._trades_contracts_to_amount(trades)
    assert new_amount_trades[0]["amount"] == 30.0 * contract_size
    assert new_amount_trades[1]["amount"] == 40.0 * contract_size


@pytest.mark.parametrize(
    "pair,param_amount,param_size",
    [
        ("ADA/USDT:USDT", 40, 4000),
        ("LTC/ETH", 30, 30),
        ("LTC/USD", 30, 30),
        ("ETH/USDT:USDT", 10, 1),
    ],
)
def test__amount_to_contracts(mocker, default_conf, pair, param_amount, param_size):
    api_mock = MagicMock()
    default_conf["trading_mode"] = "spot"
    default_conf["margin_mode"] = "isolated"
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    mocker.patch(
        f"{EXMS}.markets",
        {
            "LTC/USD": {
                "symbol": "LTC/USD",
                "contractSize": None,
            },
            "XLTCUSDT": {
                "symbol": "XLTCUSDT",
                "contractSize": "0.01",
            },
            "LTC/ETH": {
                "symbol": "LTC/ETH",
            },
            "ETH/USDT:USDT": {
                "symbol": "ETH/USDT:USDT",
                "contractSize": "10",
            },
        },
    )
    result_size = exchange._amount_to_contracts(pair, param_amount)
    assert result_size == param_amount
    result_amount = exchange._contracts_to_amount(pair, param_size)
    assert result_amount == param_size

    default_conf["trading_mode"] = "futures"
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    result_size = exchange._amount_to_contracts(pair, param_amount)
    assert result_size == param_size
    result_amount = exchange._contracts_to_amount(pair, param_size)
    assert result_amount == param_amount


@pytest.mark.parametrize(
    "pair,amount,expected_spot,expected_fut",
    [
        # 合约大小为0.01
        ("ADA/USDT:USDT", 40, 40, 40),
        ("ADA/USDT:USDT", 10.4445555, 10.4, 10.444),
        ("LTC/ETH", 30, 30, 30),
        ("LTC/USD", 30, 30, 30),
        ("ADA/USDT:USDT", 1.17, 1.1, 1.17),
        # 合约大小为10
        ("ETH/USDT:USDT", 10.111, 10.1, 10),
        ("ETH/USDT:USDT", 10.188, 10.1, 10),
        ("ETH/USDT:USDT", 10.988, 10.9, 10),
    ],
)
def test_amount_to_contract_precision(
    mocker,
    default_conf,
    pair,
    amount,
    expected_spot,
    expected_fut,
):
    api_mock = MagicMock()
    default_conf["trading_mode"] = "spot"
    default_conf["margin_mode"] = "isolated"
    exchange = get_patched_exchange(mocker, default_conf, api_mock)

    result_size = exchange.amount_to_contract_precision(pair, amount)
    assert result_size == expected_spot

    default_conf["trading_mode"] = "futures"
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    result_size = exchange.amount_to_contract_precision(pair, amount)
    assert result_size == expected_fut


@pytest.mark.parametrize(
    "exchange_name,open_rate,is_short,trading_mode,margin_mode",
    [
        # Bybit交易所
        ("bybit", 2.0, False, "spot", None),
        ("bybit", 2.0, False, "spot", "cross"),
        ("bybit", 2.0, True, "spot", "isolated"),
        # Binance交易所
        ("binance", 2.0, False, "spot", None),
        ("binance", 2.0, False, "spot", "cross"),
        ("binance", 2.0, True, "spot", "isolated"),
    ],
)
def test_liquidation_price_is_none(
    mocker, default_conf, exchange_name, open_rate, is_short, trading_mode, margin_mode
):
    default_conf["trading_mode"] = trading_mode
    default_conf["margin_mode"] = margin_mode
    exchange = get_patched_exchange(mocker, default_conf, exchange=exchange_name)
    assert (
        exchange.get_liquidation_price(
            pair="DOGE/USDT",
            open_rate=open_rate,  # 开仓价格
            is_short=is_short,  # 是否做空
            amount=71200.81144,  # 数量
            stake_amount=open_rate * 71200.81144,  # 保证金金额
            leverage=5,  # 杠杆倍数
            wallet_balance=-56354.57,  # 钱包余额
        )
        is None
    )


def test_get_max_pair_stake_amount(
    mocker,
    default_conf,
    leverage_tiers,  # 杠杆层级
):
    api_mock = MagicMock()
    default_conf["margin_mode"] = "isolated"  # 逐仓模式
    default_conf["trading_mode"] = "futures"  # 期货模式
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    exchange._leverage_tiers = leverage_tiers
    markets = {
        "XRP/USDT:USDT": {
            "limits": {  # 限制
                "amount": {"min": 0.001, "max": 10000},
                "cost": {"min": 5, "max": None},
            },
            "contractSize": None,  # 合约大小
            "spot": False,  # 是否现货
        },
        "LTC/USDT:USDT": {
            "limits": {
                "amount": {"min": 0.001, "max": None},
                "cost": {"min": 5, "max": None},
            },
            "contractSize": 0.01,
            "spot": False,
        },
        "ETH/USDT:USDT": {
            "limits": {
                "amount": {"min": 0.001, "max": 10000},
                "cost": {
                    "min": 5,
                    "max": 30000,
                },
            },
            "contractSize": 0.01,
            "spot": False,
        },
        "BTC/USDT": {
            "limits": {
                "amount": {"min": 0.001, "max": 10000},
                "cost": {"min": 5, "max": None},
            },
            "contractSize": 0.01,
            "spot": True,
        },
        "ADA/USDT": {
            "limits": {
                "amount": {"min": 0.001, "max": 10000},
                "cost": {
                    "min": 5,
                    "max": 500,
                },
            },
            "contractSize": 0.01,
            "spot": True,
        },
        "DOGE/USDT:USDT": {
            "limits": {
                "amount": {"min": 0.001, "max": 10000},
                "cost": {"min": 5, "max": 500},
            },
            "contractSize": None,
            "spot": False,
        },
        "LUNA/USDT:USDT": {
            "limits": {
                "amount": {"min": 0.001, "max": 10000},
                "cost": {"min": 5, "max": 500},
            },
            "contractSize": 0.01,
            "spot": False,
        },
        "ZEC/USDT:USDT": {
            "limits": {
                "amount": {"min": 0.001, "max": None},
                "cost": {"min": 5, "max": None},
            },
            "contractSize": 1,
            "spot": False,
        },
    }

    mocker.patch(f"{EXMS}.markets", markets)
    assert exchange.get_max_pair_stake_amount("XRP/USDT:USDT", 2.0) == 20000
    assert exchange.get_max_pair_stake_amount("XRP/USDT:USDT", 2.0, 5) == 4000
    # 限制杠杆层级
    assert exchange.get_max_pair_stake_amount("ZEC/USDT:USDT", 2.0, 5) == 100_000
    assert exchange.get_max_pair_stake_amount("ZEC/USDT:USDT", 2.0, 50) == 1000

    assert exchange.get_max_pair_stake_amount("LTC/USDT:USDT", 2.0) == float("inf")
    assert exchange.get_max_pair_stake_amount("ETH/USDT:USDT", 2.0) == 200
    assert exchange.get_max_pair_stake_amount("DOGE/USDT:USDT", 2.0) == 500
    assert exchange.get_max_pair_stake_amount("LUNA/USDT:USDT", 2.0) == 5.0

    default_conf["trading_mode"] = "spot"  # 现货模式
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    mocker.patch(f"{EXMS}.markets", markets)
    assert exchange.get_max_pair_stake_amount("BTC/USDT", 2.0) == 20000
    assert exchange.get_max_pair_stake_amount("ADA/USDT", 2.0) == 500


@pytest.mark.parametrize("exchange_name", EXCHANGES)
def test_load_leverage_tiers(mocker, default_conf, exchange_name):
    if exchange_name == "bybit":
        # TODO: 一旦get_leverage_tiers的解决方案被移除就取消注释
        pytest.skip("当前跳过")
    api_mock = MagicMock()
    api_mock.fetch_leverage_tiers = MagicMock()
    type(api_mock).has = PropertyMock(return_value={"fetchLeverageTiers": True})
    default_conf["dry_run"] = False  # 非模拟运行
    mocker.patch(f"{EXMS}.validate_trading_mode_and_margin_mode")

    api_mock.fetch_leverage_tiers = MagicMock(
        return_value={
            "ADA/USDT:USDT": [
                {
                    "tier": 1,  # 层级
                    "minNotional": 0,  # 最小名义价值
                    "maxNotional": 500,  # 最大名义价值
                    "maintenanceMarginRate": 0.02,  # 维持保证金率
                    "maxLeverage": 75,  # 最大杠杆
                    "info": {
                        "baseMaxLoan": "",
                        "imr": "0.013",
                        "instId": "",
                        "maxLever": "75",
                        "maxSz": "500",
                        "minSz": "0",
                        "mmr": "0.01",
                        "optMgnFactor": "0",
                        "quoteMaxLoan": "",
                        "tier": "1",
                        "uly": "ADA-USDT",
                    },
                },
            ]
        }
    )

    # 现货模式
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    assert exchange.load_leverage_tiers() == {}

    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"

    if exchange_name != "binance":
        # 期货模式 has.fetchLeverageTiers == False
        type(api_mock).has = PropertyMock(return_value={"fetchLeverageTiers": False})
        exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
        assert exchange.load_leverage_tiers() == {}

    # 常规期货模式
    type(api_mock).has = PropertyMock(return_value={"fetchLeverageTiers": True})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    assert exchange.load_leverage_tiers() == {
        "ADA/USDT:USDT": [
            {
                "tier": 1,
                "minNotional": 0,
                "maxNotional": 500,
                "maintenanceMarginRate": 0.02,
                "maxLeverage": 75,
                "info": {
                    "baseMaxLoan": "",
                    "imr": "0.013",
                    "instId": "",
                    "maxLever": "75",
                    "maxSz": "500",
                    "minSz": "0",
                    "mmr": "0.01",
                    "optMgnFactor": "0",
                    "quoteMaxLoan": "",
                    "tier": "1",
                    "uly": "ADA-USDT",
                },
            },
        ]
    }

    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        api_mock,
        exchange_name,
        "load_leverage_tiers",
        "fetch_leverage_tiers",
    )


@pytest.mark.parametrize("exchange_name", EXCHANGES)
async def test_get_market_leverage_tiers(mocker, default_conf, exchange_name):
    default_conf["exchange"]["name"] = exchange_name
    await async_ccxt_exception(
        mocker,
        default_conf,
        MagicMock(),
        "get_market_leverage_tiers",
        "fetch_market_leverage_tiers",
        symbol="BTC/USDT:USDT",
    )


def test_parse_leverage_tier(mocker, default_conf):
    exchange = get_patched_exchange(mocker, default_conf)

    tier = {
        "tier": 1,
        "minNotional": 0,
        "maxNotional": 100000,
        "maintenanceMarginRate": 0.025,
        "maxLeverage": 20,
        "info": {
            "bracket": "1",
            "initialLeverage": "20",
            "maxNotional": "100000",
            "minNotional": "0",
            "maintMarginRatio": "0.025",
            "cum": "0.0",
        },
    }

    assert exchange.parse_leverage_tier(tier) == {
        "minNotional": 0,
        "maxNotional": 100000,
        "maintenanceMarginRate": 0.025,
        "maxLeverage": 20,
        "maintAmt": 0.0,  # 维持保证金金额
    }

    tier2 = {
        "tier": 1,
        "minNotional": 0,
        "maxNotional": 2000,
        "maintenanceMarginRate": 0.01,
        "maxLeverage": 75,
        "info": {
            "baseMaxLoan": "",
            "imr": "0.013",
            "instId": "",
            "maxLever": "75",
            "maxSz": "2000",
            "minSz": "0",
            "mmr": "0.01",
            "optMgnFactor": "0",
            "quoteMaxLoan": "",
            "tier": "1",
            "uly": "SHIB-USDT",
        },
    }

    assert exchange.parse_leverage_tier(tier2) == {
        "minNotional": 0,
        "maxNotional": 2000,
        "maintenanceMarginRate": 0.01,
        "maxLeverage": 75,
        "maintAmt": None,
    }


def test_get_maintenance_ratio_and_amt_exceptions(mocker, default_conf, leverage_tiers):
    api_mock = MagicMock()
    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    exchange = get_patched_exchange(mocker, default_conf, api_mock)

    exchange._leverage_tiers = leverage_tiers
    with pytest.raises(
        DependencyException,
        match="名义价值不能小于0",
    ):
        exchange.get_maintenance_ratio_and_amt("1000SHIB/USDT:USDT", -1)

    exchange._leverage_tiers = {}

    with pytest.raises(
        InvalidOrderException,
        match="1000SHIB/USDT:USDT的维持保证金率不可用",
    ):
        exchange.get_maintenance_ratio_and_amt("1000SHIB/USDT:USDT", 10000)


@pytest.mark.parametrize(
    "pair,value,mmr,maintAmt",
    [
        ("ADA/USDT:USDT", 500, 0.025, 0.0),
        ("ADA/USDT:USDT", 20000000, 0.5, 1527500.0),
        ("ZEC/USDT:USDT", 500, 0.01, 0.0),
        ("ZEC/USDT:USDT", 20000000, 0.5, 654500.0),
    ],
)
def test_get_maintenance_ratio_and_amt(
    mocker, default_conf, leverage_tiers, pair, value, mmr, maintAmt
):
    api_mock = MagicMock()
    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    exchange._leverage_tiers = leverage_tiers
    assert exchange.get_maintenance_ratio_and_amt(pair, value) == (mmr, maintAmt)


def test_get_max_leverage_futures(default_conf, mocker, leverage_tiers):
    # 测试现货
    exchange = get_patched_exchange(mocker, default_conf, exchange="binance")
    assert exchange.get_max_leverage("BNB/USDT", 100.0) == 1.0

    # 测试期货
    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    exchange = get_patched_exchange(mocker, default_conf, exchange="binance")

    exchange._leverage_tiers = leverage_tiers

    assert exchange.get_max_leverage("XRP/USDT:USDT", 1.0) == 20.0
    assert exchange.get_max_leverage("BNB/USDT:USDT", 100.0) == 75.0
    assert exchange.get_max_leverage("BTC/USDT:USDT", 170.30) == 125.0
    assert pytest.approx(exchange.get_max_leverage("XRP/USDT:USDT", 99999.9)) == 5
    assert pytest.approx(exchange.get_max_leverage("BNB/USDT:USDT", 1500)) == 25
    assert exchange.get_max_leverage("BTC/USDT:USDT", 300000000) == 2.0
    assert exchange.get_max_leverage("BTC/USDT:USDT", 600000000) == 1.0  # 最后一层级

    assert exchange.get_max_leverage("SPONGE/USDT:USDT", 200) == 1.0  # 交易对不在杠杆层级中
    assert exchange.get_max_leverage("BTC/USDT:USDT", 0.0) == 125.0  # 没有保证金金额
    with pytest.raises(
        InvalidOrderException, match=r"金额1000000000.01对于BTC/USDT:USDT过高"
    ):
        exchange.get_max_leverage("BTC/USDT:USDT", 1000000000.01)


@pytest.mark.parametrize("exchange_name", ["binance", "kraken", "gate", "okx", "bybit"])
def test__get_params(mocker, default_conf, exchange_name):
    api_mock = MagicMock()
    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    exchange._params = {"test": True}

    params1 = {"test": True}
    params2 = {
        "test": True,
        "timeInForce": "IOC",
        "reduceOnly": True,
    }

    if exchange_name == "kraken":
        params2["leverage"] = 3.0

    if exchange_name == "okx":
        params2["tdMode"] = "isolated"
        params2["posSide"] = "net"

    if exchange_name == "bybit":
        params2["position_idx"] = 0

    assert (
        exchange._get_params(
            side="buy",
            ordertype="market",
            reduceOnly=False,
            time_in_force="GTC",
            leverage=1.0,
        )
        == params1
    )

    assert (
        exchange._get_params(
            side="buy",
            ordertype="market",
            reduceOnly=False,
            time_in_force="IOC",
            leverage=1.0,
        )
        == params1
    )

    assert (
        exchange._get_params(
            side="buy",
            ordertype="limit",
            reduceOnly=False,
            time_in_force="GTC",
            leverage=1.0,
        )
        == params1
    )

    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange=exchange_name)
    exchange._params = {"test": True}
    assert (
            exchange._get_params(
                side="buy",
                ordertype="limit",
                reduceOnly=True,
                time_in_force="IOC",
                leverage=3.0,
            )
            == params2
        )


def test_get_liquidation_price1(mocker, default_conf):
    api_mock = MagicMock()
    leverage = 9.97
    positions = [
        {
            "info": {},
            "symbol": "NEAR/USDT:USDT",
            "timestamp": 1642164737148,
            "datetime": "2022-01-14T12:52:17.148Z",
            "initialMargin": 1.51072,
            "initialMarginPercentage": 0.1,
            "maintenanceMargin": 0.38916147,
            "maintenanceMarginPercentage": 0.025,
            "entryPrice": 18.884,
            "notional": 15.1072,
            "leverage": leverage,
            "unrealizedPnl": 0.0048,
            "contracts": 8,
            "contractSize": 0.1,
            "marginRatio": None,
            "liquidationPrice": 17.47,
            "markPrice": 18.89,
            "margin_mode": 1.52549075,
            "marginType": "isolated",
            "side": "buy",
            "percentage": 0.003177292946409658,
        }
    ]
    api_mock.fetch_positions = MagicMock(return_value=positions)
    mocker.patch.multiple(
        EXMS,
        exchange_has=MagicMock(return_value=True),
    )
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    default_conf["liquidation_buffer"] = 0.0

    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    清算价格 = exchange.get_liquidation_price(
        pair="NEAR/USDT:USDT",
        open_rate=18.884,
        is_short=False,
        amount=0.8,
        stake_amount=18.884 * 0.8,
        leverage=leverage,
        wallet_balance=18.884 * 0.8,
    )
    assert 清算价格 == 17.47

    default_conf["liquidation_buffer"] = 0.05
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    清算价格 = exchange.get_liquidation_price(
        pair="NEAR/USDT:USDT",
        open_rate=18.884,
        is_short=False,
        amount=0.8,
        stake_amount=18.884 * 0.8,
        leverage=leverage,
        wallet_balance=18.884 * 0.8,
    )
    assert 清算价格 == 17.540699999999998

    api_mock.fetch_positions = MagicMock(return_value=[])
    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    清算价格 = exchange.get_liquidation_price(
        pair="NEAR/USDT:USDT",
        open_rate=18.884,
        is_short=False,
        amount=0.8,
        stake_amount=18.884 * 0.8,
        leverage=leverage,
        wallet_balance=18.884 * 0.8,
    )
    assert 清算价格 is None
    default_conf["trading_mode"] = "margin"

    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    with pytest.raises(OperationalException, match=r".*不支持.*保证金"):
        exchange.get_liquidation_price(
            pair="NEAR/USDT:USDT",
            open_rate=18.884,
            is_short=False,
            amount=0.8,
            stake_amount=18.884 * 0.8,
            leverage=leverage,
            wallet_balance=18.884 * 0.8,
            open_trades=[],
        )


@pytest.mark.parametrize("liquidation_buffer", [0.0])
@pytest.mark.parametrize(
    "is_short,trading_mode,exchange_name,margin_mode,leverage,open_rate,amount,mramt,expected_liq",
    [
        (False, "spot", "binance", "", 5.0, 10.0, 1.0, (0.01, 0.01), None),
        (True, "spot", "binance", "", 5.0, 10.0, 1.0, (0.01, 0.01), None),
        (False, "spot", "gate", "", 5.0, 10.0, 1.0, (0.01, 0.01), None),
        (True, "spot", "gate", "", 5.0, 10.0, 1.0, (0.01, 0.01), None),
        (False, "spot", "okx", "", 5.0, 10.0, 1.0, (0.01, 0.01), None),
        (True, "spot", "okx", "", 5.0, 10.0, 1.0, (0.01, 0.01), None),
        # 币安，做空
        (True, "futures", "binance", "isolated", 5.0, 10.0, 1.0, (0.01, 0.01), 11.89108910891089),
        (True, "futures", "binance", "isolated", 3.0, 10.0, 1.0, (0.01, 0.01), 13.211221122079207),
        (True, "futures", "binance", "isolated", 5.0, 8.0, 1.0, (0.01, 0.01), 9.514851485148514),
        (True, "futures", "binance", "isolated", 5.0, 10.0, 0.6, (0.01, 0.01), 11.897689768976898),
        # 币安，做多
        (False, "futures", "binance", "isolated", 5, 10, 1.0, (0.01, 0.01), 8.070707070707071),
        (False, "futures", "binance", "isolated", 5, 8, 1.0, (0.01, 0.01), 6.454545454545454),
        (False, "futures", "binance", "isolated", 3, 10, 1.0, (0.01, 0.01), 6.723905723905723),
        (False, "futures", "binance", "isolated", 5, 10, 0.6, (0.01, 0.01), 8.063973063973064),
        # Gate/okx，做空
        (True, "futures", "gate", "isolated", 5, 10, 1.0, (0.01, 0.01), 11.87413417771621),
        (True, "futures", "gate", "isolated", 5, 10, 2.0, (0.01, 0.01), 11.87413417771621),
        (True, "futures", "gate", "isolated", 3, 10, 1.0, (0.01, 0.01), 13.193482419684678),
        (True, "futures", "gate", "isolated", 5, 8, 1.0, (0.01, 0.01), 9.499307342172967),
        (True, "futures", "okx", "isolated", 3, 10, 1.0, (0.01, 0.01), 13.193482419684678),
        # Gate/okx，做多
        (False, "futures", "gate", "isolated", 5.0, 10.0, 1.0, (0.01, 0.01), 8.085708510208207),
        (False, "futures", "gate", "isolated", 3.0, 10.0, 1.0, (0.01, 0.01), 6.738090425173506),
        (False, "futures", "okx", "isolated", 3.0, 10.0, 1.0, (0.01, 0.01), 6.738090425173506),
        # bybit，做多
        (False, "futures", "bybit", "isolated", 1.0, 10.0, 1.0, (0.01, 0.01), 0.1),
        (False, "futures", "bybit", "isolated", 3.0, 10.0, 1.0, (0.01, 0.01), 6.7666666),
        (False, "futures", "bybit", "isolated", 5.0, 10.0, 1.0, (0.01, 0.01), 8.1),
        (False, "futures", "bybit", "isolated", 10.0, 10.0, 1.0, (0.01, 0.01), 9.1),
        # 来自bybit示例 - 无额外保证金
        (False, "futures", "bybit", "isolated", 50.0, 40000.0, 1.0, (0.005, None), 39400),
        (False, "futures", "bybit", "isolated", 50.0, 20000.0, 1.0, (0.005, None), 19700),
        # bybit，做空
        (True, "futures", "bybit", "isolated", 1.0, 10.0, 1.0, (0.01, 0.01), 19.9),
        (True, "futures", "bybit", "isolated", 3.0, 10.0, 1.0, (0.01, 0.01), 13.233333),
        (True, "futures", "bybit", "isolated", 5.0, 10.0, 1.0, (0.01, 0.01), 11.9),
        (True, "futures", "bybit", "isolated", 10.0, 10.0, 1.0, (0.01, 0.01), 10.9),
    ],
)
def test_get_liquidation_price(
    mocker,
    default_conf_usdt,
    is_short,
    trading_mode,
    exchange_name,
    margin_mode,
    leverage,
    open_rate,
    amount,
    mramt,
    expected_liq,
    liquidation_buffer,
):
    """
    position = 0.2 * 5
    wb: 钱包余额（如果是逐仓则为保证金金额）
    cum_b: 维持保证金金额
    side_1: 做空为-1，做多为1
    ep1: 入场价格
    mmr_b: 维持保证金比率

    币安，做空
    杠杆 = 5，入场价 = 10，数量 = 1.0
        ((wb + cum_b) - (side_1 * position * ep1)) / ((position * mmr_b) - (side_1 * position))
        ((2 + 0.01) - ((-1) * 1 * 10)) / ((1 * 0.01) - ((-1) * 1)) = 11.89108910891089
    杠杆 = 3，入场价 = 10，数量 = 1.0
        ((3.3333333333 + 0.01) - ((-1) * 1.0 * 10)) / ((1.0 * 0.01) - ((-1) * 1.0)) = 13.2112211220
    杠杆 = 5，入场价 = 8，数量 = 1.0
        ((1.6 + 0.01) - ((-1) * 1 * 8)) / ((1 * 0.01) - ((-1) * 1)) = 9.514851485148514
    杠杆 = 5，入场价 = 10，数量 = 0.6
        ((1.6 + 0.01) - ((-1) * 0.6 * 10)) / ((0.6 * 0.01) - ((-1) * 0.6)) = 12.557755775577558

    币安，做多
    杠杆 = 5，入场价 = 10，数量 = 1.0
        ((wb + cum_b) - (side_1 * position * ep1)) / ((position * mmr_b) - (side_1 * position))
        ((2 + 0.01) - (1 * 1 * 10)) / ((1 * 0.01) - (1 * 1)) = 8.070707070707071
    杠杆 = 5，入场价 = 8，数量 = 1.0
        ((1.6 + 0.01) - (1 * 1 * 8)) / ((1 * 0.01) - (1 * 1)) = 6.454545454545454
    杠杆 = 3，入场价 = 10，数量 = 1.0
        ((2 + 0.01) - (1 * 0.6 * 10)) / ((0.6 * 0.01) - (1 * 0.6)) = 6.717171717171718
    杠杆 = 5，入场价 = 10，数量 = 0.6
        ((1.6 + 0.01) - (1 * 0.6 * 10)) / ((0.6 * 0.01) - (1 * 0.6)) = 7.39057239057239

    Gate/Okx，做空
    杠杆 = 5，入场价 = 10，数量 = 1.0
        (入场价 + (钱包余额 / 持仓量)) / (1 + (维持保证金比率 + 吃单费率))
        (10 + (2 / 1.0)) / (1 + (0.01 + 0.0006)) = 11.87413417771621
    杠杆 = 5，入场价 = 10，数量 = 2.0
        (10 + (4 / 2.0)) / (1 + (0.01 + 0.0006)) = 11.87413417771621
    杠杆 = 3，入场价 = 10，数量 = 1.0
        (10 + (3.3333333333333 / 1.0)) / (1 - (0.01 + 0.0006)) = 13.476180850346978
    杠杆 = 5，入场价 = 8，数量 = 1.0
        (8 + (1.6 / 1.0)) / (1 + (0.01 + 0.0006)) = 9.499307342172967

    Gate/Okx，做多
    杠杆 = 5，入场价 = 10，数量 = 1.0
        (入场价 - (钱包余额 / 持仓量)) / (1 - (维持保证金比率 + 吃单费率))
        (10 - (2 / 1)) / (1 - (0.01 + 0.0006)) = 8.085708510208207
    杠杆 = 5，入场价 = 10，数量 = 2.0
        (10 - (4 / 2.0)) / (1 + (0.01 + 0.0006)) = 7.916089451810806
    杠杆 = 3，入场价 = 10，数量 = 1.0
        (10 - (3.333333333333333333 / 1.0)) / (1 - (0.01 + 0.0006)) = 6.738090425173506
    杠杆 = 5，入场价 = 8，数量 = 1.0
        (8 - (1.6 / 1.0)) / (1 + (0.01 + 0.0006)) = 6.332871561448645
    """
    default_conf_usdt["liquidation_buffer"] = liquidation_buffer
    default_conf_usdt["trading_mode"] = trading_mode
    default_conf_usdt["exchange"]["name"] = exchange_name
    default_conf_usdt["margin_mode"] = margin_mode
    mocker.patch("freqtrade.exchange.gate.Gate.validate_ordertypes")
    mocker.patch(f"{EXMS}.price_to_precision", lambda s, x, y, **kwargs: y)
    exchange = get_patched_exchange(mocker, default_conf_usdt, exchange=exchange_name)

    exchange.get_maintenance_ratio_and_amt = MagicMock(return_value=mramt)
    exchange.name = exchange_name
    # default_conf_usdt.update({
    #     "dry_run": False,
    # })
    清算价格 = exchange.get_liquidation_price(
        pair="ETH/USDT:USDT",
        open_rate=open_rate,
        amount=amount,
        stake_amount=amount * open_rate / leverage,
        wallet_balance=amount * open_rate / leverage,
        leverage=leverage,
        is_short=is_short,
        open_trades=[],
    )
    if expected_liq is None:
        assert 清算价格 is None
    else:
        缓冲金额 = liquidation_buffer * abs(open_rate - expected_liq)
        expected_liq = expected_liq - 缓冲金额 if is_short else expected_liq + 缓冲金额
        assert pytest.approx(expected_liq) == 清算价格


@pytest.mark.parametrize(
    "contract_size,order_amount",
    [
        (10, 10),
        (0.01, 10000),
    ],
)
def test_stoploss_contract_size(mocker, default_conf, contract_size, order_amount):
    api_mock = MagicMock()
    order_id = f"test_prod_buy_{randint(0, 10**6)}"

    api_mock.create_order = MagicMock(
        return_value={
            "id": order_id,
            "info": {"foo": "bar"},
            "amount": order_amount,
            "cost": order_amount,
            "filled": order_amount,
            "remaining": order_amount,
            "symbol": "ETH/BTC",
        }
    )
    default_conf["dry_run"] = False
    mocker.patch(f"{EXMS}.amount_to_precision", lambda s, x, y: y)
    mocker.patch(f"{EXMS}.price_to_precision", lambda s, x, y, **kwargs: y)

    exchange = get_patched_exchange(mocker, default_conf, api_mock)
    exchange.get_contract_size = MagicMock(return_value=contract_size)

    api_mock.create_order.reset_mock()
    order = exchange.create_stoploss(
        pair="ETH/BTC", amount=100, stop_price=220, order_types={}, side="buy", leverage=1.0
    )

    assert api_mock.create_order.call_args_list[0][1]["amount"] == order_amount
    assert order["amount"] == 100
    assert order["cost"] == order_amount
    assert order["filled"] == 100
    assert order["remaining"] == 100


def test_price_to_precision_with_default_conf(default_conf, mocker):
    conf = copy.deepcopy(default_conf)
    patched_ex = get_patched_exchange(mocker, conf)
    精确价格 = patched_ex.price_to_precision("XRP/USDT", 1.0000000101)
    assert 精确价格 == 1.00000001
    assert 精确价格 == 1.00000001


def test_exchange_features(default_conf, mocker):
    conf = copy.deepcopy(default_conf)
    exchange = get_patched_exchange(mocker, conf)
    exchange._api_async.features = {
        "spot": {
            "fetchOHLCV": {
                "limit": 995,
            }
        },
        "swap": {
            "linear": {
                "fetchOHLCV": {
                    "limit": 997,
                }
            }
        },
    }
    assert exchange.features("spot", "fetchOHLCV", "limit", 500) == 995
    assert exchange.features("futures", "fetchOHLCV", "limit", 500) == 997
    # 回退到默认值
    assert exchange.features("futures", "fetchOHLCV_else", "limit", 601) == 601
