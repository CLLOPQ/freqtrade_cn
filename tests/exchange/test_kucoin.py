from random import randint
from unittest.mock import MagicMock

import ccxt
import pytest

from freqtrade.exceptions import DependencyException, InvalidOrderException
from tests.conftest import EXMS, get_patched_exchange
from tests.exchange.test_exchange import ccxt_exceptionhandlers


@pytest.mark.parametrize("order_type", ["market", "limit"])
@pytest.mark.parametrize(
    "limitratio,expected,side",
    [
        (None, 220 * 0.99, "sell"),
        (0.99, 220 * 0.99, "sell"),
        (0.98, 220 * 0.98, "sell"),
    ],
)
def test_create_stoploss_order_kucoin(default_conf, mocker, limitratio, expected, side, order_type):
    """测试Kucoin交易所创建止损订单"""
    api_mock = MagicMock()
    order_id = f"test_prod_buy_{randint(0, 10**6)}"

    api_mock.create_order = MagicMock(return_value={"id": order_id, "info": {"foo": "bar"}})
    default_conf["dry_run"] = False
    mocker.patch(f"{EXMS}.amount_to_precision", lambda s, x, y: y)
    mocker.patch(f"{EXMS}.price_to_precision", lambda s, x, y, **kwargs: y)

    exchange = get_patched_exchange(mocker, default_conf, api_mock, "kucoin")
    if order_type == "limit":
        # 测试无效的订单比例会抛出异常
        with pytest.raises(InvalidOrderException):
            order = exchange.create_stoploss(
                pair="ETH/BTC",
                amount=1,
                stop_price=190,
                order_types={"stoploss": order_type, "stoploss_on_exchange_limit_ratio": 1.05},
                side=side,
                leverage=1.0,
            )

    api_mock.create_order.reset_mock()
    order_types = {"stoploss": order_type}
    if limitratio is not None:
        order_types.update({"stoploss_on_exchange_limit_ratio": limitratio})
    order = exchange.create_stoploss(
        pair="ETH/BTC", amount=1, stop_price=220, order_types=order_types, side=side, leverage=1.0
    )

    assert "id" in order
    assert "info" in order
    assert order["id"] == order_id
    assert api_mock.create_order.call_args_list[0][1]["symbol"] == "ETH/BTC"
    assert api_mock.create_order.call_args_list[0][1]["type"] == order_type
    assert api_mock.create_order.call_args_list[0][1]["side"] == "sell"
    assert api_mock.create_order.call_args_list[0][1]["amount"] == 1
    # 价格应该比止损价低1%
    if order_type == "limit":
        assert api_mock.create_order.call_args_list[0][1]["price"] == expected
    else:
        assert api_mock.create_order.call_args_list[0][1]["price"] is None

    assert api_mock.create_order.call_args_list[0][1]["params"] == {
        "stopPrice": 220,
        "stop": "loss",
    }

    # 测试异常处理
    with pytest.raises(DependencyException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InsufficientFunds("余额为0"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, "kucoin")
        exchange.create_stoploss(
            pair="ETH/BTC", amount=1, stop_price=220, order_types={}, side=side, leverage=1.0
        )

    with pytest.raises(InvalidOrderException):
        api_mock.create_order = MagicMock(
            side_effect=ccxt.InvalidOrder("kucoin订单将立即触发。")
        )
        exchange = get_patched_exchange(mocker, default_conf, api_mock, "kucoin")
        exchange.create_stoploss(
            pair="ETH/BTC", amount=1, stop_price=220, order_types={}, side=side, leverage=1.0
        )

    # 测试CCXT异常处理函数
    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        api_mock,
        "kucoin",
        "create_stoploss",
        "create_order",
        retries=1,
        pair="ETH/BTC",
        amount=1,
        stop_price=220,
        order_types={},
        side=side,
        leverage=1.0,
    )


def test_stoploss_order_dry_run_kucoin(default_conf, mocker):
    """测试Kucoin交易所模拟交易模式下的止损订单"""
    api_mock = MagicMock()
    order_type = "market"
    default_conf["dry_run"] = True
    mocker.patch(f"{EXMS}.amount_to_precision", lambda s, x, y: y)
    mocker.patch(f"{EXMS}.price_to_precision", lambda s, x, y, **kwargs: y)

    exchange = get_patched_exchange(mocker, default_conf, api_mock, "kucoin")

    # 测试无效的订单比例会抛出异常
    with pytest.raises(InvalidOrderException):
        order = exchange.create_stoploss(
            pair="ETH/BTC",
            amount=1,
            stop_price=190,
            order_types={"stoploss": "limit", "stoploss_on_exchange_limit_ratio": 1.05},
            side="sell",
            leverage=1.0,
        )

    api_mock.create_order.reset_mock()

    order = exchange.create_stoploss(
        pair="ETH/BTC", amount=1, stop_price=220, order_types={}, side="sell", leverage=1.0
    )

    assert "id" in order
    assert "info" in order
    assert "type" in order

    assert order["type"] == order_type
    assert order["price"] == 220
    assert order["amount"] == 1


def test_stoploss_adjust_kucoin(mocker, default_conf):
    """测试Kucoin交易所的止损调整功能"""
    exchange = get_patched_exchange(mocker, default_conf, exchange="kucoin")
    order = {
        "type": "limit",
        "price": 1500,
        "stopPrice": 1500,
        "info": {"stopPrice": 1500, "stop": "limit"},
    }
    # 验证调整逻辑
    assert exchange.stoploss_adjust(1501, order, "sell")
    assert not exchange.stoploss_adjust(1499, order, "sell")
    # 测试无效订单情况下的调整逻辑
    order["stopPrice"] = None
    assert exchange.stoploss_adjust(1501, order, "sell")


@pytest.mark.parametrize("side", ["buy", "sell"])
@pytest.mark.parametrize(
    "ordertype,rate", [("market", None), ("market", 200), ("limit", 200), ("stop_loss_limit", 200)]
)
def test_kucoin_create_order(default_conf, mocker, side, ordertype, rate):
    """测试Kucoin交易所创建订单功能"""
    api_mock = MagicMock()
    order_id = f"test_prod_{side}_{randint(0, 10**6)}"
    api_mock.create_order = MagicMock(
        return_value={"id": order_id, "info": {"foo": "bar"}, "symbol": "XRP/USDT", "amount": 1}
    )
    default_conf["dry_run"] = False
    mocker.patch(f"{EXMS}.amount_to_precision", lambda s, x, y: y)
    mocker.patch(f"{EXMS}.price_to_precision", lambda s, x, y: y)
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="kucoin")
    exchange._set_leverage = MagicMock()
    exchange.set_margin_mode = MagicMock()

    order = exchange.create_order(
        pair="XRP/USDT", ordertype=ordertype, side=side, amount=1, rate=rate, leverage=1.0
    )

    assert "id" in order
    assert "info" in order
    assert order["id"] == order_id
    assert order["amount"] == 1
    # Kucoin的订单状态必须模拟为"open"
    assert order["status"] == "open"