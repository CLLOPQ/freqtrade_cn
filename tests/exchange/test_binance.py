from datetime import datetime, timedelta
from random import randint
from unittest.mock import MagicMock, PropertyMock

import ccxt
import pandas as pd
import pytest

from freqtrade.data.converter.trade_converter import trades_dict_to_list
from freqtrade.enums import CandleType, MarginMode, TradingMode
from freqtrade.exceptions import DependencyException, InvalidOrderException, OperationalException
from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_seconds
from freqtrade.persistence import Trade
from freqtrade.util.datetime_helpers import dt_from_ts, dt_ts, dt_utc
from tests.conftest import EXMS, get_patched_exchange
from tests.exchange.test_exchange import ccxt_exceptionhandlers


@pytest.mark.parametrize(
    "方向,订单类型,有效时间,预期结果",
    [
        ("buy", "limit", "gtc", {"timeInForce": "GTC"}),
        ("buy", "limit", "IOC", {"timeInForce": "IOC"}),
        ("buy", "market", "IOC", {}),
        ("buy", "limit", "PO", {"timeInForce": "PO"}),
        ("sell", "limit", "PO", {"timeInForce": "PO"}),
        ("sell", "market", "PO", {}),
    ],
)
def test__get_params_binance(默认配置, mocker, 方向, 订单类型, 有效时间, 预期结果):
    交易所 = get_patched_exchange(mocker, 默认配置, exchange="binance")
    assert 交易所._get_params(方向, 订单类型, 1, False, 有效时间) == 预期结果


@pytest.mark.parametrize("交易模式", [TradingMode.FUTURES, TradingMode.SPOT])
@pytest.mark.parametrize(
    "限价比例,预期价格,方向",
    [
        (None, 220 * 0.99, "sell"),
        (0.99, 220 * 0.99, "sell"),
        (0.98, 220 * 0.98, "sell"),
        (None, 220 * 1.01, "buy"),
        (0.99, 220 * 1.01, "buy"),
        (0.98, 220 * 1.02, "buy"),
    ],
)
def test_create_stoploss_order_binance(默认配置, mocker, 限价比例, 预期价格, 方向, 交易模式):
    api_mock = MagicMock()
    订单ID = f"test_prod_buy_{randint(0, 10**6)}"
    订单类型 = "stop_loss_limit" if 交易模式 == TradingMode.SPOT else "stop"

    api_mock.create_order = MagicMock(return_value={"id": 订单ID, "info": {"foo": "bar"}})
    默认配置["dry_run"] = False
    默认配置["margin_mode"] = MarginMode.ISOLATED
    默认配置["trading_mode"] = 交易模式
    mocker.patch(f"{EXMS}.amount_to_precision", lambda s, x, y: y)
    mocker.patch(f"{EXMS}.price_to_precision", lambda s, x, y, **kwargs: y)

    交易所 = get_patched_exchange(mocker, 默认配置, api_mock, "binance")

    with pytest.raises(InvalidOrderException):
        订单 = 交易所.create_stoploss(
            pair="ETH/BTC",
            amount=1,
            stop_price=190,
            side=方向,
            order_types={"stoploss": "limit", "stoploss_on_exchange_limit_ratio": 1.05},
            leverage=1.0,
        )

    api_mock.create_order.reset_mock()
    订单类型字典 = {"stoploss": "limit", "stoploss_price_type": "mark"}
    if 限价比例 is not None:
        订单类型字典.update({"stoploss_on_exchange_limit_ratio": 限价比例})

    订单 = 交易所.create_stoploss(
        pair="ETH/BTC", amount=1, stop_price=220, order_types=订单类型字典, side=方向, leverage=1.0
    )

    assert "id" in 订单
    assert "info" in 订单
    assert 订单["id"] == 订单ID
    assert api_mock.create_order.call_args_list[0][1]["symbol"] == "ETH/BTC"
    assert api_mock.create_order.call_args_list[0][1]["type"] == 订单类型
    assert api_mock.create_order.call_args_list[0][1]["side"] == 方向
    assert api_mock.create_order.call_args_list[0][1]["amount"] == 1
    # 价格应为止损价的1%以下
    assert api_mock.create_order.call_args_list[0][1]["price"] == 预期价格
    if 交易模式 == TradingMode.SPOT:
        参数字典 = {"stopPrice": 220}
    else:
        参数字典 = {"stopPrice": 220, "reduceOnly": True, "workingType": "MARK_PRICE"}
    assert api_mock.create_order.call_args_list[0][1]["params"] == 参数字典

    # 测试异常处理
    with pytest.raises(DependencyException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InsufficientFunds("0 balance"))
        交易所 = get_patched_exchange(mocker, 默认配置, api_mock, "binance")
        交易所.create_stoploss(
            pair="ETH/BTC", amount=1, stop_price=220, order_types={}, side=方向, leverage=1.0
        )

    with pytest.raises(InvalidOrderException):
        api_mock.create_order = MagicMock(
            side_effect=ccxt.InvalidOrder("binance Order would trigger immediately.")
        )
        交易所 = get_patched_exchange(mocker, 默认配置, api_mock, "binance")
        交易所.create_stoploss(
            pair="ETH/BTC", amount=1, stop_price=220, order_types={}, side=方向, leverage=1.0
        )

    ccxt_exceptionhandlers(
        mocker,
        默认配置,
        api_mock,
        "binance",
        "create_stoploss",
        "create_order",
        retries=1,
        pair="ETH/BTC",
        amount=1,
        stop_price=220,
        order_types={},
        side=方向,
        leverage=1.0,
    )


def test_create_stoploss_order_dry_run_binance(默认配置, mocker):
    api_mock = MagicMock()
    订单类型 = "stop_loss_limit"
    默认配置["dry_run"] = True
    mocker.patch(f"{EXMS}.amount_to_precision", lambda s, x, y: y)
    mocker.patch(f"{EXMS}.price_to_precision", lambda s, x, y, **kwargs: y)

    交易所 = get_patched_exchange(mocker, 默认配置, api_mock, "binance")

    with pytest.raises(InvalidOrderException):
        订单 = 交易所.create_stoploss(
            pair="ETH/BTC",
            amount=1,
            stop_price=190,
            side="sell",
            order_types={"stoploss_on_exchange_limit_ratio": 1.05},
            leverage=1.0,
        )

    api_mock.create_order.reset_mock()

    订单 = 交易所.create_stoploss(
        pair="ETH/BTC", amount=1, stop_price=220, order_types={}, side="sell", leverage=1.0
    )

    assert "id" in 订单
    assert "info" in 订单
    assert "type" in 订单

    assert 订单["type"] == 订单类型
    assert 订单["price"] == 220
    assert 订单["amount"] == 1


@pytest.mark.parametrize(
    "止损1,止损2,止损3,方向", [(1501, 1499, 1501, "sell"), (1499, 1501, 1499, "buy")]
)
def test_stoploss_adjust_binance(mocker, 默认配置, 止损1, 止损2, 止损3, 方向):
    交易所 = get_patched_exchange(mocker, 默认配置, exchange="binance")
    订单 = {
        "type": "stop_loss_limit",
        "price": 1500,
        "stopPrice": 1500,
        "info": {"stopPrice": 1500},
    }
    assert 交易所.stoploss_adjust(止损1, 订单, side=方向)
    assert not 交易所.stoploss_adjust(止损2, 订单, side=方向)


@pytest.mark.parametrize(
    "交易对, 是否做空, 交易模式, 保证金模式, 钱包余额, "
    "维持保证金金额, 数量, 开仓价格, 未平仓交易,"
    "维持保证金比例, 预期结果",
    [
        (
            "ETH/USDT:USDT",
            False,
            "futures",
            "isolated",
            1535443.01,
            135365.00,
            3683.979,
            1456.84,
            [],
            0.10,
            1114.78,
        ),
        (
            "ETH/USDT:USDT",
            False,
            "futures",
            "isolated",
            1535443.01,
            16300.000,
            109.488,
            32481.980,
            [],
            0.025,
            18778.73,
        ),
        (
            "ETH/USDT:USDT",
            False,
            "futures",
            "cross",
            1535443.01,
            135365.00,
            3683.979,  # 数量
            1456.84,  # 开仓价格
            [
                {
                    # 计算示例
                    "pair": "BTC/USDT:USDT",
                    "open_rate": 32481.98,
                    "amount": 109.488,
                    "stake_amount": 3556387.02624,  # 开仓价格 * 数量
                    "mark_price": 31967.27,
                    "mm_ratio": 0.025,
                    "maintenance_amt": 16300.0,
                },
                {
                    # 计算示例
                    "pair": "ETH/USDT:USDT",
                    "open_rate": 1456.84,
                    "amount": 3683.979,
                    "stake_amount": 5366967.96,
                    "mark_price": 1335.18,
                    "mm_ratio": 0.10,
                    "maintenance_amt": 135365.00,
                },
            ],
            0.10,
            1153.26,
        ),
        (
            "BTC/USDT:USDT",
            False,
            "futures",
            "cross",
            1535443.01,
            16300.0,
            109.488,  # 数量
            32481.980,  # 开仓价格
            [
                {
                    # 计算示例
                    "pair": "BTC/USDT:USDT",
                    "open_rate": 32481.98,
                    "amount": 109.488,
                    "stake_amount": 3556387.02624,  # 开仓价格 * 数量
                    "mark_price": 31967.27,
                    "mm_ratio": 0.025,
                    "maintenance_amt": 16300.0,
                },
                {
                    # 计算示例
                    "pair": "ETH/USDT:USDT",
                    "open_rate": 1456.84,
                    "amount": 3683.979,
                    "stake_amount": 5366967.96,
                    "mark_price": 1335.18,
                    "mm_ratio": 0.10,
                    "maintenance_amt": 135365.00,
                },
            ],
            0.025,
            26316.89,
        ),
    ],
)
def test_liquidation_price_binance(
    mocker,
    默认配置,
    交易对,
    是否做空,
    交易模式,
    保证金模式,
    钱包余额,
    维持保证金金额,
    数量,
    开仓价格,
    未平仓交易,
    维持保证金比例,
    预期结果,
):
    默认配置["trading_mode"] = 交易模式
    默认配置["margin_mode"] = 保证金模式
    默认配置["liquidation_buffer"] = 0.0
    mocker.patch(f"{EXMS}.price_to_precision", lambda s, x, y, **kwargs: y)
    交易所 = get_patched_exchange(mocker, 默认配置, exchange="binance")

    def get_maint_ratio(交易对_, 保证金金额):
        if 交易对_ != 交易对:
            oc = next(c for c in 未平仓交易 if c["pair"] == 交易对_)
            return oc["mm_ratio"], oc["maintenance_amt"]
        return 维持保证金比例, 维持保证金金额

    def fetch_funding_rates(*args, **kwargs):
        return {
            t["pair"]: {
                "symbol": t["pair"],
                "markPrice": t["mark_price"],
            }
            for t in 未平仓交易
        }

    交易所.get_maintenance_ratio_and_amt = get_maint_ratio
    交易所.fetch_funding_rates = fetch_funding_rates

    未平仓交易对象 = [
        Trade(
            pair=t["pair"],
            open_rate=t["open_rate"],
            amount=t["amount"],
            stake_amount=t["stake_amount"],
            fee_open=0,
        )
        for t in 未平仓交易
    ]

    assert (
        pytest.approx(
            round(
                交易所.get_liquidation_price(
                    pair=交易对,
                    open_rate=开仓价格,
                    is_short=是否做空,
                    wallet_balance=钱包余额,
                    amount=数量,
                    stake_amount=开仓价格 * 数量,
                    leverage=5,
                    open_trades=未平仓交易对象,
                ),
                2,
            )
        )
        == 预期结果
    )


def test_fill_leverage_tiers_binance(默认配置, mocker):
    api_mock = MagicMock()
    api_mock.fetch_leverage_tiers = MagicMock(
        return_value={
            "ADA/BUSD": [
                {
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
                },
                {
                    "tier": 2,
                    "minNotional": 100000,
                    "maxNotional": 500000,
                    "maintenanceMarginRate": 0.05,
                    "maxLeverage": 10,
                    "info": {
                        "bracket": "2",
                        "initialLeverage": "10",
                        "maxNotional": "500000",
                        "minNotional": "100000",
                        "maintMarginRatio": "0.05",
                        "cum": "2500.0",
                    },
                },
                {
                    "tier": 3,
                    "minNotional": 500000,
                    "maxNotional": 1000000,
                    "maintenanceMarginRate": 0.1,
                    "maxLeverage": 5,
                    "info": {
                        "bracket": "3",
                        "initialLeverage": "5",
                        "maxNotional": "1000000",
                        "minNotional": "500000",
                        "maintMarginRatio": "0.1",
                        "cum": "27500.0",
                    },
                },
                {
                    "tier": 4,
                    "minNotional": 1000000,
                    "maxNotional": 2000000,
                    "maintenanceMarginRate": 0.15,
                    "maxLeverage": 3,
                    "info": {
                        "bracket": "4",
                        "initialLeverage": "3",
                        "maxNotional": "2000000",
                        "minNotional": "1000000",
                        "maintMarginRatio": "0.15",
                        "cum": "77500.0",
                    },
                },
                {
                    "tier": 5,
                    "minNotional": 2000000,
                    "maxNotional": 5000000,
                    "maintenanceMarginRate": 0.25,
                    "maxLeverage": 2,
                    "info": {
                        "bracket": "5",
                        "initialLeverage": "2",
                        "maxNotional": "5000000",
                        "minNotional": "2000000",
                        "maintMarginRatio": "0.25",
                        "cum": "277500.0",
                    },
                },
                {
                    "tier": 6,
                    "minNotional": 5000000,
                    "maxNotional": 30000000,
                    "maintenanceMarginRate": 0.5,
                    "maxLeverage": 1,
                    "info": {
                        "bracket": "6",
                        "initialLeverage": "1",
                        "maxNotional": "30000000",
                        "minNotional": "5000000",
                        "maintMarginRatio": "0.5",
                        "cum": "1527500.0",
                    },
                },
            ],
            "ZEC/USDT": [
                {
                    "tier": 1,
                    "minNotional": 0,
                    "maxNotional": 50000,
                    "maintenanceMarginRate": 0.01,
                    "maxLeverage": 50,
                    "info": {
                        "bracket": "1",
                        "initialLeverage": "50",
                        "maxNotional": "50000",
                        "minNotional": "0",
                        "maintMarginRatio": "0.01",
                        "cum": "0.0",
                    },
                },
                {
                    "tier": 2,
                    "minNotional": 50000,
                    "maxNotional": 150000,
                    "maintenanceMarginRate": 0.025,
                    "maxLeverage": 20,
                    "info": {
                        "bracket": "2",
                        "initialLeverage": "20",
                        "maxNotional": "150000",
                        "minNotional": "50000",
                        "maintMarginRatio": "0.025",
                        "cum": "750.0",
                    },
                },
                {
                    "tier": 3,
                    "minNotional": 150000,
                    "maxNotional": 250000,
                    "maintenanceMarginRate": 0.05,
                    "maxLeverage": 10,
                    "info": {
                        "bracket": "3",
                        "initialLeverage": "10",
                        "maxNotional": "250000",
                        "minNotional": "150000",
                        "maintMarginRatio": "0.05",
                        "cum": "4500.0",
                    },
                },
                {
                    "tier": 4,
                    "minNotional": 250000,
                    "maxNotional": 500000,
                    "maintenanceMarginRate": 0.1,
                    "maxLeverage": 5,
                    "info": {
                        "bracket": "4",
                        "initialLeverage": "5",
                        "maxNotional": "500000",
                        "minNotional": "250000",
                        "maintMarginRatio": "0.1",
                        "cum": "17000.0",
                    },
                },
                {
                    "tier": 5,
                    "minNotional": 500000,
                    "maxNotional": 1000000,
                    "maintenanceMarginRate": 0.125,
                    "maxLeverage": 4,
                    "info": {
                        "bracket": "5",
                        "initialLeverage": "4",
                        "maxNotional": "1000000",
                        "minNotional": "500000",
                        "maintMarginRatio": "0.125",
                        "cum": "29500.0",
                    },
                },
                {
                    "tier": 6,
                    "minNotional": 1000000,
                    "maxNotional": 2000000,
                    "maintenanceMarginRate": 0.25,
                    "maxLeverage": 2,
                    "info": {
                        "bracket": "6",
                        "initialLeverage": "2",
                        "maxNotional": "2000000",
                        "minNotional": "1000000",
                        "maintMarginRatio": "0.25",
                        "cum": "154500.0",
                    },
                },
                {
                    "tier": 7,
                    "minNotional": 2000000,
                    "maxNotional": 30000000,
                    "maintenanceMarginRate": 0.5,
                    "maxLeverage": 1,
                    "info": {
                        "bracket": "7",
                        "initialLeverage": "1",
                        "maxNotional": "30000000",
                        "minNotional": "2000000",
                        "maintMarginRatio": "0.5",
                        "cum": "654500.0",
                    },
                },
            ],
        }
    )
    默认配置["dry_run"] = False
    默认配置["trading_mode"] = TradingMode.FUTURES
    默认配置["margin_mode"] = MarginMode.ISOLATED
    交易所 = get_patched_exchange(mocker, 默认配置, api_mock, exchange="binance")
    交易所.fill_leverage_tiers()

    assert 交易所._leverage_tiers == {
        "ADA/BUSD": [
            {
                "minNotional": 0,
                "maxNotional": 100000,
                "maintenanceMarginRate": 0.025,
                "maxLeverage": 20,
                "maintAmt": 0.0,
            },
            {
                "minNotional": 100000,
                "maxNotional": 500000,
                "maintenanceMarginRate": 0.05,
                "maxLeverage": 10,
                "maintAmt": 2500.0,
            },
            {
                "minNotional": 500000,
                "maxNotional": 1000000,
                "maintenanceMarginRate": 0.1,
                "maxLeverage": 5,
                "maintAmt": 27500.0,
            },
            {
                "minNotional": 1000000,
                "maxNotional": 2000000,
                "maintenanceMarginRate": 0.15,
                "maxLeverage": 3,
                "maintAmt": 77500.0,
            },
            {
                "minNotional": 2000000,
                "maxNotional": 5000000,
                "maintenanceMarginRate": 0.25,
                "maxLeverage": 2,
                "maintAmt": 277500.0,
            },
            {
                "minNotional": 5000000,
                "maxNotional": 30000000,
                "maintenanceMarginRate": 0.5,
                "maxLeverage": 1,
                "maintAmt": 1527500.0,
            },
        ],
        "ZEC/USDT": [
            {
                "minNotional": 0,
                "maxNotional": 50000,
                "maintenanceMarginRate": 0.01,
                "maxLeverage": 50,
                "maintAmt": 0.0,
            },
            {
                "minNotional": 50000,
                "maxNotional": 150000,
                "maintenanceMarginRate": 0.025,
                "maxLeverage": 20,
                "maintAmt": 750.0,
            },
            {
                "minNotional": 150000,
                "maxNotional": 250000,
                "maintenanceMarginRate": 0.05,
                "maxLeverage": 10,
                "maintAmt": 4500.0,
            },
            {
                "minNotional": 250000,
                "maxNotional": 500000,
                "maintenanceMarginRate": 0.1,
                "maxLeverage": 5,
                "maintAmt": 17000.0,
            },
            {
                "minNotional": 500000,
                "maxNotional": 1000000,
                "maintenanceMarginRate": 0.125,
                "maxLeverage": 4,
                "maintAmt": 29500.0,
            },
            {
                "minNotional": 1000000,
                "maxNotional": 2000000,
                "maintenanceMarginRate": 0.25,
                "maxLeverage": 2,
                "maintAmt": 154500.0,
            },
            {
                "minNotional": 2000000,
                "maxNotional": 30000000,
                "maintenanceMarginRate": 0.5,
                "maxLeverage": 1,
                "maintAmt": 654500.0,
            },
        ],
    }

    api_mock = MagicMock()
    api_mock.load_leverage_tiers = MagicMock()
    type(api_mock).has = PropertyMock(return_value={"fetchLeverageTiers": True})

    ccxt_exceptionhandlers(
        mocker,
        默认配置,
        api_mock,
        "binance",
        "fill_leverage_tiers",
        "fetch_leverage_tiers",
    )


def test_fill_leverage_tiers_binance_dryrun(默认配置, mocker, leverage_tiers):
    api_mock = MagicMock()
    默认配置["trading_mode"] = TradingMode.FUTURES
    默认配置["margin_mode"] = MarginMode.ISOLATED
    交易所 = get_patched_exchange(mocker, 默认配置, api_mock, exchange="binance")
    交易所.fill_leverage_tiers()
    assert len(交易所._leverage_tiers.keys()) > 100
    for key, value in leverage_tiers.items():
        v = 交易所._leverage_tiers[key]
        assert isinstance(v, list)
        # 确保conftest杠杆层级不超过交易所的层级
        assert len(v) >= len(value)


def test_additional_exchange_init_binance(默认配置, mocker):
    api_mock = MagicMock()
    api_mock.fapiPrivateGetPositionSideDual = MagicMock(return_value={"dualSidePosition": True})
    api_mock.fapiPrivateGetMultiAssetsMargin = MagicMock(return_value={"multiAssetsMargin": True})
    默认配置["dry_run"] = False
    默认配置["trading_mode"] = TradingMode.FUTURES
    默认配置["margin_mode"] = MarginMode.ISOLATED
    with pytest.raises(
        OperationalException,
        match=r"Hedge Mode is not supported.*\nMulti-Asset Mode is not supported.*",
    ):
        get_patched_exchange(mocker, 默认配置, exchange="binance", api_mock=api_mock)
    api_mock.fapiPrivateGetPositionSideDual = MagicMock(return_value={"dualSidePosition": False})
    api_mock.fapiPrivateGetMultiAssetsMargin = MagicMock(return_value={"multiAssetsMargin": False})
    交易所 = get_patched_exchange(mocker, 默认配置, exchange="binance", api_mock=api_mock)
    assert 交易所
    ccxt_exceptionhandlers(
        mocker,
        默认配置,
        api_mock,
        "binance",
        "additional_exchange_init",
        "fapiPrivateGetPositionSideDual",
    )


def test__set_leverage_binance(mocker, 默认配置):
    api_mock = MagicMock()
    api_mock.set_leverage = MagicMock()
    type(api_mock).has = PropertyMock(return_value={"setLeverage": True})
    默认配置["dry_run"] = False
    默认配置["trading_mode"] = TradingMode.FUTURES
    默认配置["margin_mode"] = MarginMode.ISOLATED

    交易所 = get_patched_exchange(mocker, 默认配置, api_mock, exchange="binance")
    交易所._set_leverage(3.2, "BTC/USDT:USDT")
    assert api_mock.set_leverage.call_count == 1
    # 杠杆会被四舍五入为3
    assert api_mock.set_leverage.call_args_list[0][1]["leverage"] == 3
    assert api_mock.set_leverage.call_args_list[0][1]["symbol"] == "BTC/USDT:USDT"

    ccxt_exceptionhandlers(
        mocker,
        默认配置,
        api_mock,
        "binance",
        "_set_leverage",
        "set_leverage",
        pair="XRP/USDT",
        leverage=5.0,
    )


def patch_binance_vision_ohlcv(mocker, start, archive_end, api_end, timeframe):
    def make_storage(start: datetime, end: datetime, timeframe: str):
        date = pd.date_range(start, end, freq=timeframe.replace("m", "min"))
        df = pd.DataFrame(
            data=dict(date=date, open=1.0, high=1.0, low=1.0, close=1.0),
        )
        return df

    archive_storage = make_storage(start, archive_end, timeframe)
    api_storage = make_storage(start, api_end, timeframe)

    ohlcv = [[dt_ts(start), 1, 1, 1, 1]]
    # (交易对, 时间框架, K线类型, ohlcv, True)
    candle_history = [None, None, None, ohlcv, None]

    def get_historic_ohlcv(
        # self,
        pair: str,
        timeframe: str,
        since_ms: int,
        candle_type: CandleType,
        is_new_pair: bool = False,
        until_ms: int | None = None,
    ):
        since = dt_from_ts(since_ms)
        until = dt_from_ts(until_ms) if until_ms else api_end + timedelta(seconds=1)
        return api_storage.loc[(api_storage["date"] >= since) & (api_storage["date"] < until)]

    async def download_archive_ohlcv(
        candle_type,
        pair,
        timeframe,
        since_ms,
        until_ms,
        markets=None,
        stop_on_404=False,
    ):
        since = dt_from_ts(since_ms)
        until = dt_from_ts(until_ms) if until_ms else archive_end + timedelta(seconds=1)
        if since < start:
            pass
        return archive_storage.loc[
            (archive_storage["date"] >= since) & (archive_storage["date"] < until)
        ]

    candle_mock = mocker.patch(f"{EXMS}._async_get_candle_history", return_value=candle_history)
    api_mock = mocker.patch(f"{EXMS}.get_historic_ohlcv", side_effect=get_historic_ohlcv)
    archive_mock = mocker.patch(
        "freqtrade.exchange.binance.download_archive_ohlcv", side_effect=download_archive_ohlcv
    )
    return candle_mock, api_mock, archive_mock


@pytest.mark.parametrize(
    "时间框架,是否新交易对,开始时间,结束时间,首个日期,最后日期,调用K线,调用存档,"
    "调用API",
    [
        (
            "1m",
            True,
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 2),
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 1, 23, 59),
            True,
            True,
            False,
        ),
        (
            "1m",
            True,
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 3),
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 2, 23, 59),
            True,
            True,
            True,
        ),
        (
            "1m",
            True,
            dt_utc(2020, 1, 2),
            dt_utc(2020, 1, 2, 1),
            dt_utc(2020, 1, 2),
            dt_utc(2020, 1, 2, 0, 59),
            True,
            False,
            True,
        ),
        (
            "1m",
            False,
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 2),
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 1, 23, 59),
            False,
            True,
            False,
        ),
        (
            "1m",
            True,
            dt_utc(2019, 1, 1),
            dt_utc(2020, 1, 2),
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 1, 23, 59),
            True,
            True,
            False,
        ),
        (
            "1m",
            False,
            dt_utc(2019, 1, 1),
            dt_utc(2020, 1, 2),
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 1, 23, 59),
            False,
            True,
            False,
        ),
        (
            "1m",
            False,
            dt_utc(2019, 1, 1),
            dt_utc(2019, 1, 2),
            None,
            None,
            False,
            True,
            True,
        ),
        (
            "1m",
            True,
            dt_utc(2019, 1, 1),
            dt_utc(2019, 1, 2),
            None,
            None,
            True,
            False,
            False,
        ),
        (
            "1m",
            False,
            dt_utc(2021, 1, 1),
            dt_utc(2021, 1, 2),
            None,
            None,
            False,
            False,
            False,
        ),
        (
            "1m",
            True,
            dt_utc(2021, 1, 1),
            dt_utc(2021, 1, 2),
            None,
            None,
            True,
            False,
            False,
        ),
        (
            "1h",
            False,
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 2),
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 1, 23),
            False,
            False,
            True,
        ),
        (
            "1m",
            False,
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 1, 3, 50, 30),
            dt_utc(2020, 1, 1),
            dt_utc(2020, 1, 1, 3, 50),
            False,
            True,
            False,
        ),
    ],
)
def test_get_historic_ohlcv_binance(
    mocker,
    默认配置,
    时间框架,
    是否新交易对,
    开始时间,
    结束时间,
    首个日期,
    最后日期,
    调用K线,
    调用存档,
    调用API,
):
    交易所 = get_patched_exchange(mocker, 默认配置, exchange="binance")

    start = dt_utc(2020, 1, 1)
    archive_end = dt_utc(2020, 1, 2)
    api_end = dt_utc(2020, 1, 3)
    candle_mock, api_mock, archive_mock = patch_binance_vision_ohlcv(
        mocker, start=start, archive_end=archive_end, api_end=api_end, timeframe=时间框架
    )

    candle_type = CandleType.SPOT
    交易对 = "BTC/USDT"

    since_ms = dt_ts(开始时间)
    until_ms = dt_ts(结束时间)

    df = 交易所.get_historic_ohlcv(交易对, 时间框架, since_ms, candle_type, 是否新交易对, until_ms)

    if df.empty:
        assert 首个日期 is None
        assert 最后日期 is None
    else:
        assert df["date"].iloc[0] == 首个日期
        assert df["date"].iloc[-1] == 最后日期
        assert (
            df["date"].diff().iloc[1:] == timedelta(seconds=timeframe_to_seconds(时间框架))
        ).all()

    if 调用K线:
        candle_mock.assert_called_once()
    if 调用存档:
        archive_mock.assert_called_once()
    if 调用API:
        api_mock.assert_called_once()


@pytest.mark.parametrize(
    "交易对,名义价值,维持保证金比例,金额",
    [
        ("XRP/USDT:USDT", 0.0, 0.025, 0),
        ("BNB/USDT:USDT", 100.0, 0.0065, 0),
        ("BTC/USDT:USDT", 170.30, 0.004, 0),
        ("XRP/USDT:USDT", 999999.9, 0.1, 27500.0),
        ("BNB/USDT:USDT", 5000000.0, 0.15, 233035.0),
        ("BTC/USDT:USDT", 600000000, 0.5, 1.997038e8),
    ],
)
def test_get_maintenance_ratio_and_amt_binance(
    默认配置,
    mocker,
    leverage_tiers,
    交易对,
    名义价值,
    维持保证金比例,
    金额,
):
    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    交易所 = get_patched_exchange(mocker, 默认配置, exchange="binance")
    交易所._leverage_tiers = leverage_tiers
    (result_ratio, result_amt) = 交易所.get_maintenance_ratio_and_amt(交易对, 名义价值)
    assert (round(result_ratio, 8), round(result_amt, 8)) == (维持保证金比例, 金额)


async def test__async_get_trade_history_id_binance(默认配置_usdt, mocker, fetch_trades_result):
    默认配置_usdt["exchange"]["only_from_ccxt"] = True
    交易所 = get_patched_exchange(mocker, 默认配置_usdt, exchange="binance")

    async def mock_get_trade_hist(交易对, *args, **kwargs):
        if "since" in kwargs:
            # 早于初始调用
            if kwargs["since"] < 1565798399752:
                return []
            else:
                # 不期望到达这里
                raise ValueError("Unexpected call")
                # return fetch_trades_result[:-2]
        elif kwargs.get("params", {}).get(交易所._trades_pagination_arg) == "0":
            # 返回前3个
            return fetch_trades_result[:-2]
        elif kwargs.get("params", {}).get(交易所._trades_pagination_arg) in (
            fetch_trades_result[-3]["id"],
            1565798399752,
        ):
            # 返回2个
            return fetch_trades_result[-3:-1]
        else:
            # 返回最后2个
            return fetch_trades_result[-2:]

    交易所._api_async.fetch_trades = MagicMock(side_effect=mock_get_trade_hist)

    交易对 = "ETH/BTC"
    ret = await 交易所._async_get_trade_history_id(
        交易对,
        since=fetch_trades_result[0]["timestamp"],
        until=fetch_trades_result[-1]["timestamp"] - 1,
    )
    assert ret[0] == 交易对
    assert isinstance(ret[1], list)
    assert 交易所._api_async.fetch_trades.call_count == 4

    fetch_trades_cal = 交易所._api_async.fetch_trades.call_args_list
    # 第一次调用（使用since，不使用fromId）
    assert fetch_trades_cal[0][0][0] == 交易对
    assert fetch_trades_cal[0][1]["since"] == fetch_trades_result[0]["timestamp"]

    # 第二次调用
    assert fetch_trades_cal[1][0][0] == 交易对
    assert "params" in fetch_trades_cal[1][1]
    pagination_arg = 交易所._ft_has["trades_pagination_arg"]
    assert pagination_arg in fetch_trades_cal[1][1]["params"]
    # 初始调用使用from_id = "0"
    assert fetch_trades_cal[1][1]["params"][pagination_arg] == "0"

    assert fetch_trades_cal[2][1]["params"][pagination_arg] != "0"
    assert fetch_trades_cal[3][1]["params"][pagination_arg] != "0"

    # 清理事件循环以避免警告
    交易所.close()


async def test__async_get_trade_history_id_binance_fast(
    默认配置_usdt, mocker, fetch_trades_result
):
    默认配置_usdt["exchange"]["only_from_ccxt"] = False
    交易所 = get_patched_exchange(mocker, 默认配置_usdt, exchange="binance")

    async def mock_get_trade_hist(交易对, *args, **kwargs):
        if "since" in kwargs:
            pass
            # 早于初始调用
            # if kwargs["since"] < 1565798399752:
            #     return []
            # else:
            #     # 不期望到达这里
            #     raise ValueError("Unexpected call")
            #     # return fetch_trades_result[:-2]
        elif kwargs.get("params", {}).get(交易所._trades_pagination_arg) == "0":
            # 返回前3个
            return fetch_trades_result[:-2]
        # elif kwargs.get("params", {}).get(交易所._trades_pagination_arg) in (
        #     fetch_trades_result[-3]["id"],
        #     1565798399752,
        # ):
        #     # 返回2个
        #     return fetch_trades_result[-3:-1]
        # else:
        #     # 返回最后2个
        #     return fetch_trades_result[-2:]

    交易对 = "ETH/BTC"
    mocker.patch(
        "freqtrade.exchange.binance.download_archive_trades",
        return_value=(交易对, trades_dict_to_list(fetch_trades_result[-2:])),
    )

    交易所._api_async.fetch_trades = MagicMock(side_effect=mock_get_trade_hist)

    ret = await 交易所._async_get_trade_history(
        交易对,
        since=fetch_trades_result[0]["timestamp"],
        until=fetch_trades_result[-1]["timestamp"] - 1,
    )

    assert ret[0] == 交易对
    assert isinstance(ret[1], list)

    # 清理事件循环以避免警告
    交易所.close()
