from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import ccxt
import pytest

from freqtrade.enums import CandleType, MarginMode, TradingMode
from freqtrade.exceptions import RetryableOrderError, TemporaryError
from freqtrade.exchange.common import API_RETRY_COUNT
from freqtrade.exchange.exchange import timeframe_to_minutes
from tests.conftest import EXMS, get_patched_exchange, log_has
from tests.exchange.test_exchange import ccxt_exceptionhandlers


def test_okx_ohlcv_candle_limit(default_conf, mocker):
    """测试OKX交易所的OHLCV蜡烛图数据限制"""
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")
    timeframes = ("1m", "5m", "1h")
    start_time = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

    for timeframe in timeframes:
        # 验证不同蜡烛图类型的默认限制
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.SPOT) == 300
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUTURES) == 300
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.MARK) == 100
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUNDING_RATE) == 100

        # 验证指定起始时间时的限制
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.SPOT, start_time) == 100
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUTURES, start_time) == 100
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.MARK, start_time) == 100
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUNDING_RATE, start_time) == 100
        
        # 计算单批次调用的时间点（290个时间单位前）
        one_call = int(
            (
                datetime.now(timezone.utc)
                - timedelta(minutes=290 * timeframe_to_minutes(timeframe))
            ).timestamp()
            * 1000
        )

        # 验证该时间点的限制
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.SPOT, one_call) == 300
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUTURES, one_call) == 300

        # 计算需要分页的时间点（320个时间单位前）
        one_call = int(
            (
                datetime.now(timezone.utc)
                - timedelta(minutes=320 * timeframe_to_minutes(timeframe))
            ).timestamp()
            * 1000
        )
        # 验证该时间点的限制
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.SPOT, one_call) == 100
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUTURES, one_call) == 100


def test_get_maintenance_ratio_and_amt_okx(
    default_conf,
    mocker,
):
    """测试OKX交易所获取维持保证金比率和金额"""
    api_mock = MagicMock()
    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    default_conf["dry_run"] = False
    mocker.patch.multiple(
        "freqtrade.exchange.okx.Okx",
        exchange_has=MagicMock(return_value=True),
        load_leverage_tiers=MagicMock(
            return_value={
                "ETH/USDT:USDT": [
                    {
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
                            "uly": "ETH-USDT",
                        },
                    },
                    {
                        "tier": 2,
                        "minNotional": 2001,
                        "maxNotional": 4000,
                        "maintenanceMarginRate": 0.015,
                        "maxLeverage": 50,
                        "info": {
                            "baseMaxLoan": "",
                            "imr": "0.02",
                            "instId": "",
                            "maxLever": "50",
                            "maxSz": "4000",
                            "minSz": "2001",
                            "mmr": "0.015",
                            "optMgnFactor": "0",
                            "quoteMaxLoan": "",
                            "tier": "2",
                            "uly": "ETH-USDT",
                        },
                    },
                    {
                        "tier": 3,
                        "minNotional": 4001,
                        "maxNotional": 8000,
                        "maintenanceMarginRate": 0.02,
                        "maxLeverage": 20,
                        "info": {
                            "baseMaxLoan": "",
                            "imr": "0.05",
                            "instId": "",
                            "maxLever": "20",
                            "maxSz": "8000",
                            "minSz": "4001",
                            "mmr": "0.02",
                            "optMgnFactor": "0",
                            "quoteMaxLoan": "",
                            "tier": "3",
                            "uly": "ETH-USDT",
                        },
                    },
                ],
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
                    {
                        "tier": 2,
                        "minNotional": 501,
                        "maxNotional": 1000,
                        "maintenanceMarginRate": 0.025,
                        "maxLeverage": 50,
                        "info": {
                            "baseMaxLoan": "",
                            "imr": "0.02",
                            "instId": "",
                            "maxLever": "50",
                            "maxSz": "1000",
                            "minSz": "501",
                            "mmr": "0.015",
                            "optMgnFactor": "0",
                            "quoteMaxLoan": "",
                            "tier": "2",
                            "uly": "ADA-USDT",
                        },
                    },
                    {
                        "tier": 3,
                        "minNotional": 1001,
                        "maxNotional": 2000,
                        "maintenanceMarginRate": 0.03,
                        "maxLeverage": 20,
                        "info": {
                            "baseMaxLoan": "",
                            "imr": "0.05",
                            "instId": "",
                            "maxLever": "20",
                            "maxSz": "2000",
                            "minSz": "1001",
                            "mmr": "0.02",
                            "optMgnFactor": "0",
                            "quoteMaxLoan": "",
                            "tier": "3",
                            "uly": "ADA-USDT",
                        },
                    },
                ],
            }
        ),
    )
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="okx")
    
    # 验证ETH/USDT不同金额对应的维持保证金比率
    assert exchange.get_maintenance_ratio_and_amt("ETH/USDT:USDT", 2000) == (0.01, None)
    assert exchange.get_maintenance_ratio_and_amt("ETH/USDT:USDT", 2001) == (0.015, None)
    assert exchange.get_maintenance_ratio_and_amt("ETH/USDT:USDT", 4001) == (0.02, None)
    assert exchange.get_maintenance_ratio_and_amt("ETH/USDT:USDT", 8000) == (0.02, None)

    # 验证ADA/USDT不同金额对应的维持保证金比率
    assert exchange.get_maintenance_ratio_and_amt("ADA/USDT:USDT", 1) == (0.02, None)
    assert exchange.get_maintenance_ratio_and_amt("ADA/USDT:USDT", 2000) == (0.03, None)


def test_get_max_pair_stake_amount_okx(default_conf, mocker, leverage_tiers):
    """测试OKX交易所获取最大交易对投注金额"""
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")
    # 现货模式下无限制
    assert exchange.get_max_pair_stake_amount("BNB/BUSD", 1.0) == float("inf")

    # 期货模式下的限制
    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")
    exchange._leverage_tiers = leverage_tiers

    # 验证不同交易对的最大金额限制
    assert exchange.get_max_pair_stake_amount("XRP/USDT:USDT", 1.0) == 30000000
    assert exchange.get_max_pair_stake_amount("BNB/USDT:USDT", 1.0) == 50000000
    assert exchange.get_max_pair_stake_amount("BTC/USDT:USDT", 1.0) == 1000000000
    # 考虑杠杆后的最大金额（1000000000 / 10 = 100000000）
    assert exchange.get_max_pair_stake_amount("BTC/USDT:USDT", 1.0, 10.0) == 100000000

    # 不在杠杆层级中的交易对无限制
    assert exchange.get_max_pair_stake_amount("TTT/USDT:USDT", 1.0) == float("inf")


@pytest.mark.parametrize(
    "mode,side,reduceonly,result",
    [
        ("net", "buy", False, "net"),
        ("net", "sell", True, "net"),
        ("net", "sell", False, "net"),
        ("net", "buy", True, "net"),
        ("longshort", "buy", False, "long"),
        ("longshort", "sell", True, "long"),
        ("longshort", "sell", False, "short"),
        ("longshort", "buy", True, "short"),
    ],
)
def test__get_posSide(default_conf, mocker, mode, side, reduceonly, result):
    """测试OKX交易所获取仓位方向"""
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")
    exchange.net_only = mode == "net"
    assert exchange._get_posSide(side, reduceonly) == result


def test_additional_exchange_init_okx(default_conf, mocker):
    """测试OKX交易所的额外初始化逻辑"""
    api_mock = MagicMock()
    api_mock.fetch_accounts = MagicMock(
        return_value=[
            {
                "id": "2555",
                "type": "2",
                "currency": None,
                "info": {
                    "acctLv": "2",
                    "autoLoan": False,
                    "ctIsoMode": "automatic",
                    "greeksType": "PA",
                    "level": "Lv1",
                    "levelTmp": "",
                    "mgnIsoMode": "automatic",
                    "posMode": "long_short_mode",
                    "uid": "2555",
                },
            }
        ]
    )
    default_conf["dry_run"] = False
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx", api_mock=api_mock)
    # 初始化时未调用fetch_accounts
    assert api_mock.fetch_accounts.call_count == 0
    exchange.trading_mode = TradingMode.FUTURES
    # 默认启用netOnly
    assert exchange.net_only
    # 执行额外初始化
    exchange.additional_exchange_init()
    # 调用了fetch_accounts
    assert api_mock.fetch_accounts.call_count == 1
    # 长/短模式下net_only应为False
    assert not exchange.net_only

    # 测试净持仓模式
    api_mock.fetch_accounts = MagicMock(
        return_value=[
            {
                "id": "2555",
                "type": "2",
                "currency": None,
                "info": {
                    "acctLv": "2",
                    "autoLoan": False,
                    "ctIsoMode": "automatic",
                    "greeksType": "PA",
                    "level": "Lv1",
                    "levelTmp": "",
                    "mgnIsoMode": "automatic",
                    "posMode": "net_mode",
                    "uid": "2555",
                },
            }
        ]
    )
    exchange.additional_exchange_init()
    # 净持仓模式下net_only应为True
    assert exchange.net_only

    # 测试异常处理
    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    ccxt_exceptionhandlers(
        mocker, default_conf, api_mock, "okx", "additional_exchange_init", "fetch_accounts"
    )


def test_load_leverage_tiers_okx(default_conf, mocker, markets, tmp_path, caplog, time_machine):
    """测试OKX交易所加载杠杆层级"""
    default_conf["datadir"] = tmp_path
    api_mock = MagicMock()
    type(api_mock).has = PropertyMock(
        return_value={
            "fetchLeverageTiers": False,
            "fetchMarketLeverageTiers": True,
        }
    )
    api_mock.fetch_market_leverage_tiers = AsyncMock(
        side_effect=[
            # 第一次调用返回ADA的杠杆层级
            [
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
                {
                    "tier": 2,
                    "minNotional": 501,
                    "maxNotional": 1000,
                    "maintenanceMarginRate": 0.025,
                    "maxLeverage": 50,
                    "info": {
                        "baseMaxLoan": "",
                        "imr": "0.02",
                        "instId": "",
                        "maxLever": "50",
                        "maxSz": "1000",
                        "minSz": "501",
                        "mmr": "0.015",
                        "optMgnFactor": "0",
                        "quoteMaxLoan": "",
                        "tier": "2",
                        "uly": "ADA-USDT",
                    },
                },
                {
                    "tier": 3,
                    "minNotional": 1001,
                    "maxNotional": 2000,
                    "maintenanceMarginRate": 0.03,
                    "maxLeverage": 20,
                    "info": {
                        "baseMaxLoan": "",
                        "imr": "0.05",
                        "instId": "",
                        "maxLever": "20",
                        "maxSz": "2000",
                        "minSz": "1001",
                        "mmr": "0.02",
                        "optMgnFactor": "0",
                        "quoteMaxLoan": "",
                        "tier": "3",
                        "uly": "ADA-USDT",
                    },
                },
            ],
            # 第二次调用模拟失败
            TemporaryError("获取失败"),
            # 第三次调用返回ETH的杠杆层级
            [
                {
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
                        "uly": "ETH-USDT",
                    },
                },
                {
                    "tier": 2,
                    "minNotional": 2001,
                    "maxNotional": 4000,
                    "maintenanceMarginRate": 0.015,
                    "maxLeverage": 50,
                    "info": {
                        "baseMaxLoan": "",
                        "imr": "0.02",
                        "instId": "",
                        "maxLever": "50",
                        "maxSz": "4000",
                        "minSz": "2001",
                        "mmr": "0.015",
                        "optMgnFactor": "0",
                        "quoteMaxLoan": "",
                        "tier": "2",
                        "uly": "ETH-USDT",
                    },
                },
                {
                    "tier": 3,
                    "minNotional": 4001,
                    "maxNotional": 8000,
                    "maintenanceMarginRate": 0.02,
                    "maxLeverage": 20,
                    "info": {
                        "baseMaxLoan": "",
                        "imr": "0.05",
                        "instId": "",
                        "maxLever": "20",
                        "maxSz": "8000",
                        "minSz": "4001",
                        "mmr": "0.02",
                        "optMgnFactor": "0",
                        "quoteMaxLoan": "",
                        "tier": "3",
                        "uly": "ETH-USDT",
                    },
                },
            ],
        ]
    )
    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    default_conf["stake_currency"] = "USDT"
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="okx")
    exchange.trading_mode = TradingMode.FUTURES
    exchange.margin_mode = MarginMode.ISOLATED
    exchange.markets = markets
    
    # 验证加载的杠杆层级是否正确
    assert exchange._leverage_tiers == {
        "ADA/USDT:USDT": [
            {
                "minNotional": 0,
                "maxNotional": 500,
                "maintenanceMarginRate": 0.02,
                "maxLeverage": 75,
                "maintAmt": None,
            },
            {
                "minNotional": 501,
                "maxNotional": 1000,
                "maintenanceMarginRate": 0.025,
                "maxLeverage": 50,
                "maintAmt": None,
            },
            {
                "minNotional": 1001,
                "maxNotional": 2000,
                "maintenanceMarginRate": 0.03,
                "maxLeverage": 20,
                "maintAmt": None,
            },
        ],
        "ETH/USDT:USDT": [
            {
                "minNotional": 0,
                "maxNotional": 2000,
                "maintenanceMarginRate": 0.01,
                "maxLeverage": 75,
                "maintAmt": None,
            },
            {
                "minNotional": 2001,
                "maxNotional": 4000,
                "maintenanceMarginRate": 0.015,
                "maxLeverage": 50,
                "maintAmt": None,
            },
            {
                "minNotional": 4001,
                "maxNotional": 8000,
                "maintenanceMarginRate": 0.02,
                "maxLeverage": 20,
                "maintAmt": None,
            },
        ],
    }
    
    # 验证缓存文件是否创建
    filename = (
        default_conf["datadir"] / f"futures/leverage_tiers_{default_conf['stake_currency']}.json"
    )
    assert filename.is_file()

    # 验证日志中没有更新缓存的消息
    logmsg = "Cached leverage tiers are outdated. Will update."
    assert not log_has(logmsg, caplog)

    # 重置模拟并再次加载杠杆层级
    api_mock.fetch_market_leverage_tiers.reset_mock()
    exchange.load_leverage_tiers()
    # 应该使用缓存，不会调用API
    assert not log_has(logmsg, caplog)
    assert api_mock.fetch_market_leverage_tiers.call_count == 0

    # 模拟时间过了5周，缓存过期
    time_machine.move_to(datetime.now() + timedelta(weeks=5))
    exchange.load_leverage_tiers()
    # 应该有更新缓存的日志
    assert log_has(logmsg, caplog)


def test__set_leverage_okx(mocker, default_conf):
    """测试OKX交易所设置杠杆"""
    api_mock = MagicMock()
    api_mock.set_leverage = MagicMock()
    type(api_mock).has = PropertyMock(return_value={"setLeverage": True})
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED

    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="okx")
    exchange._lev_prep("BTC/USDT:USDT", 3.2, "buy")
    # 验证调用了set_leverage
    assert api_mock.set_leverage.call_count == 1
    # 验证杠杆值是否正确传递（保留小数）
    assert api_mock.set_leverage.call_args_list[0][1]["leverage"] == 3.2
    assert api_mock.set_leverage.call_args_list[0][1]["symbol"] == "BTC/USDT:USDT"
    assert api_mock.set_leverage.call_args_list[0][1]["params"] == {
        "mgnMode": "isolated",
        "posSide": "net",
    }

    # 测试网络错误时的处理
    api_mock.set_leverage = MagicMock(side_effect=ccxt.NetworkError())
    exchange._lev_prep("BTC/USDT:USDT", 3.2, "buy")
    # 应该尝试获取当前杠杆
    assert api_mock.fetch_leverage.call_count == 1

    # 测试获取杠杆也失败的情况
    api_mock.fetch_leverage = MagicMock(side_effect=ccxt.NetworkError())
    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        api_mock,
        "okx",
        "_lev_prep",
        "set_leverage",
        pair="XRP/USDT:USDT",
        leverage=5.0,
        side="buy",
    )


@pytest.mark.usefixtures("init_persistence")
def test_fetch_stoploss_order_okx(default_conf, mocker):
    """测试OKX交易所获取止损订单"""
    default_conf["dry_run"] = False
    mocker.patch("freqtrade.exchange.common.time.sleep")
    api_mock = MagicMock()
    api_mock.fetch_order = MagicMock()

    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="okx")

    # 测试正常获取止损订单
    exchange.fetch_stoploss_order("1234", "ETH/BTC")
    assert api_mock.fetch_order.call_count == 1
    assert api_mock.fetch_order.call_args_list[0][0][0] == "1234"
    assert api_mock.fetch_order.call_args_list[0][0][1] == "ETH/BTC"
    assert api_mock.fetch_order.call_args_list[0][1]["params"] == {"stop": True}

    # 测试订单未找到的情况
    api_mock.fetch_order = MagicMock(side_effect=ccxt.OrderNotFound)
    api_mock.fetch_open_orders = MagicMock(return_value=[])
    api_mock.fetch_closed_orders = MagicMock(return_value=[])
    api_mock.fetch_canceled_orders = MagicMock(return_value=[])

    with pytest.raises(RetryableOrderError):
        exchange.fetch_stoploss_order("1234", "ETH/BTC")
    # 验证重试次数
    assert api_mock.fetch_order.call_count == API_RETRY_COUNT + 1
    assert api_mock.fetch_open_orders.call_count == API_RETRY_COUNT + 1
    assert api_mock.fetch_closed_orders.call_count == API_RETRY_COUNT + 1
    assert api_mock.fetch_canceled_orders.call_count == API_RETRY_COUNT + 1

    # 重置模拟
    api_mock.fetch_order.reset_mock()
    api_mock.fetch_open_orders.reset_mock()
    api_mock.fetch_closed_orders.reset_mock()
    api_mock.fetch_canceled_orders.reset_mock()

    # 测试在已关闭订单中找到的情况
    api_mock.fetch_closed_orders = MagicMock(
        return_value=[{"id": "1234", "status": "closed", "info": {"ordId": "123455"}}]
    )
    mocker.patch(f"{EXMS}.fetch_order", MagicMock(return_value={"id": "123455"}))
    resp = exchange.fetch_stoploss_order("1234", "ETH/BTC")
    assert api_mock.fetch_order.call_count == 1
    assert api_mock.fetch_open_orders.call_count == 1
    assert api_mock.fetch_closed_orders.call_count == 1
    assert api_mock.fetch_canceled_orders.call_count == 0

    # 验证返回结果
    assert resp["id"] == "1234"
    assert resp["id_stop"] == "123455"
    assert resp["type"] == "stoploss"

    # 测试模拟交易模式
    default_conf["dry_run"] = True
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="okx")
    dro_mock = mocker.patch(f"{EXMS}.fetch_dry_run_order", MagicMock(return_value={"id": "123455"}))

    # 重置模拟
    api_mock.fetch_order.reset_mock()
    api_mock.fetch_open_orders.reset_mock()
    api_mock.fetch_closed_orders.reset_mock()
    api_mock.fetch_canceled_orders.reset_mock()
    resp = exchange.fetch_stoploss_order("1234", "ETH/BTC")

    # 模拟模式下不应调用真实API
    assert api_mock.fetch_order.call_count == 0
    assert api_mock.fetch_open_orders.call_count == 0
    assert api_mock.fetch_closed_orders.call_count == 0
    assert api_mock.fetch_canceled_orders.call_count == 0
    # 应该调用模拟订单获取方法
    assert dro_mock.call_count == 1


def test_fetch_stoploss_order_okx_exceptions(default_conf_usdt, mocker):
    """测试OKX交易所获取止损订单的异常处理"""
    default_conf_usdt["dry_run"] = False
    api_mock = MagicMock()
    ccxt_exceptionhandlers(
        mocker,
        default_conf_usdt,
        api_mock,
        "okx",
        "fetch_stoploss_order",
        "fetch_order",
        retries=API_RETRY_COUNT + 1,
        order_id="12345",
        pair="ETH/USDT",
    )

    # 测试函数的第二部分异常处理
    api_mock.fetch_order = MagicMock(side_effect=ccxt.OrderNotFound())
    api_mock.fetch_closed_orders = MagicMock(return_value=[])
    api_mock.fetch_canceled_orders = MagicMock(return_value=[])

    ccxt_exceptionhandlers(
        mocker,
        default_conf_usdt,
        api_mock,
        "okx",
        "fetch_stoploss_order",
        "fetch_open_orders",
        retries=API_RETRY_COUNT + 1,
        order_id="12345",
        pair="ETH/USDT",
    )


@pytest.mark.parametrize(
    "sl1,sl2,sl3,side", [(1501, 1499, 1501, "sell"), (1499, 1501, 1499, "buy")]
)
def test_stoploss_adjust_okx(mocker, default_conf, sl1, sl2, sl3, side):
    """测试OKX交易所的止损调整功能"""
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")
    order = {
        "type": "stoploss",
        "price": 1500,
        "stopLossPrice": 1500,
    }
    # 验证调整逻辑
    assert exchange.stoploss_adjust(sl1, order, side=side)
    assert not exchange.stoploss_adjust(sl2, order, side=side)


def test_stoploss_cancel_okx(mocker, default_conf):
    """测试OKX交易所取消止损订单"""
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")

    exchange.cancel_order = MagicMock()

    exchange.cancel_stoploss_order("1234", "ETH/USDT")
    # 验证取消订单的调用参数
    assert exchange.cancel_order.call_count == 1
    assert exchange.cancel_order.call_args_list[0][1]["order_id"] == "1234"
    assert exchange.cancel_order.call_args_list[0][1]["pair"] == "ETH/USDT"
    assert exchange.cancel_order.call_args_list[0][1]["params"] == {"stop": True}


def test__get_stop_params_okx(mocker, default_conf):
    """测试OKX交易所获取止损订单参数"""
    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")
    params = exchange._get_stop_params("sell", "market", 1500)

    # 验证参数是否正确
    assert params["tdMode"] == "isolated"
    assert params["posSide"] == "net"


def test_fetch_orders_okx(default_conf, mocker, limit_order):
    """测试OKX交易所获取订单历史"""
    api_mock = MagicMock()
    api_mock.fetch_orders = MagicMock(
        return_value=[
            limit_order["buy"],
            limit_order["sell"],
        ]
    )
    api_mock.fetch_open_orders = MagicMock(return_value=[limit_order["buy"]])
    api_mock.fetch_closed_orders = MagicMock(return_value=[limit_order["buy"]])

    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    start_time = datetime.now(timezone.utc) - timedelta(days=20)

    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="okx")
    # 模拟模式下不返回订单
    assert exchange.fetch_orders("mocked", start_time) == []
    assert api_mock.fetch_orders.call_count == 0
    # 切换到实盘模式
    default_conf["dry_run"] = False

    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="okx")

    # 模拟交易所支持情况
    def has_resp(_, endpoint):
        if endpoint == "fetchOrders":
            return False
        if endpoint == "fetchClosedOrders":
            return True
        if endpoint == "fetchOpenOrders":
            return True

    mocker.patch(f"{EXMS}.exchange_has", has_resp)

    history_params = {"method": "privateGetTradeOrdersHistoryArchive"}

    # 测试不支持fetchOrders的情况
    exchange.fetch_orders("mocked", start_time)
    assert api_mock.fetch_orders.call_count == 0
    assert api_mock.fetch_open_orders.call_count == 1
    assert api_mock.fetch_closed_orders.call_count == 2
    assert "params" not in api_mock.fetch_closed_orders.call_args_list[0][1]
    assert api_mock.fetch_closed_orders.call_args_list[1][1]["params"] == history_params

    # 重置模拟
    api_mock.fetch_open_orders.reset_mock()
    api_mock.fetch_closed_orders.reset_mock()

    # 测试7天内的订单查询
    exchange.fetch_orders("mocked", datetime.now(timezone.utc) - timedelta(days=6))
    assert api_mock.fetch_orders.call_count == 0
    assert api_mock.fetch_open_orders.call_count == 1
    assert api_mock.fetch_closed_orders.call_count == 1
    assert "params" not in api_mock.fetch_closed_orders.call_args_list[0][1]

    # 测试fetchOrders不支持的情况
    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    api_mock.fetch_orders = MagicMock(side_effect=ccxt.NotSupported())
    api_mock.fetch_open_orders.reset_mock()
    api_mock.fetch_closed_orders.reset_mock()

    exchange.fetch_orders("mocked", start_time)

    assert api_mock.fetch_orders.call_count == 1
    assert api_mock.fetch_open_orders.call_count == 1
    assert api_mock.fetch_closed_orders.call_count == 2
    assert "params" not in api_mock.fetch_closed_orders.call_args_list[0][1]
    assert api_mock.fetch_closed_orders.call_args_list[1][1]["params"] == history_params