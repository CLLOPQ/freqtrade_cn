from random import randint
from unittest.mock import MagicMock

import ccxt
import pytest

from freqtrade.exceptions import DependencyException, InvalidOrderException
from tests.conftest import EXMS, get_patched_exchange
from tests.exchange.test_exchange import ccxt_exceptionhandlers


@pytest.mark.parametrize(
    "limitratio,expected,side",
    [
        (None, 220 * 0.99, "sell"),
        (0.99, 220 * 0.99, "sell"),
        (0.98, 220 * 0.98, "sell"),
    ],
)
def test_create_stoploss_order_htx(default_conf, mocker, limitratio, expected, side):
    """测试HTX交易所创建止损订单功能"""
    api_mock = MagicMock()
    order_id = f"test_prod_buy_{randint(0, 10**6)}"
    order_type = "stop-limit"

    api_mock.create_order = MagicMock(return_value={"id": order_id, "info": {"foo": "bar"}})
    default_conf["dry_run"] = False
    mocker.patch(f"{EXMS}.amount_to_precision", lambda s, x, y: y)
    mocker.patch(f"{EXMS}.price_to_precision", lambda s, x, y, **kwargs: y)

    exchange = get_patched_exchange(mocker, default_conf, api_mock, "htx")

    # 测试无效的订单比例会抛出异常
    with pytest.raises(InvalidOrderException):
        order = exchange.create_stoploss(
            pair="ETH/BTC",
            amount=1,
            stop_price=190,
            order_types={"stoploss_on_exchange_limit_ratio": 1.05},
            side=side,
            leverage=1.0,
        )

    api_mock.create_order.reset_mock()
    order_types = {} if limitratio is None else {"stoploss_on_exchange_limit_ratio": limitratio}
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
    assert api_mock.create_order.call_args_list[0][1]["price"] == expected
    assert api_mock.create_order.call_args_list[0][1]["params"] == {
        "stopPrice": 220,
        "operator": "lte",
    }

    # 测试异常处理
    with pytest.raises(DependencyException):
        api_mock.create_order = MagicMock(side_effect=ccxt.InsufficientFunds("余额为0"))
        exchange = get_patched_exchange(mocker, default_conf, api_mock, "htx")
        exchange.create_stoploss(
            pair="ETH/BTC", amount=1, stop_price=220, order_types={}, side=side, leverage=1.0
        )

    with pytest.raises(InvalidOrderException):
        api_mock.create_order = MagicMock(
            side_effect=ccxt.InvalidOrder("币安订单将立即触发。")
        )
        exchange = get_patched_exchange(mocker, default_conf, api_mock, "binance")
        exchange.create_stoploss(
            pair="ETH/BTC", amount=1, stop_price=220, order_types={}, side=side, leverage=1.0
        )

    # 测试CCXT异常处理函数
    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        api_mock,
        "htx",
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


def test_create_stoploss_order_dry_run_htx(default_conf, mocker):
    """测试HTX交易所模拟交易模式下创建止损订单"""
    api_mock = MagicMock()
    order_type = "stop-limit"
    default_conf["dry_run"] = True
    mocker.patch(f"{EXMS}.amount_to_precision", lambda s, x, y: y)
    mocker.patch(f"{EXMS}.price_to_precision", lambda s, x, y, **kwargs: y)

    exchange = get_patched_exchange(mocker, default_conf, api_mock, "htx")

    # 测试无效的订单比例会抛出异常
    with pytest.raises(InvalidOrderException):
        order = exchange.create_stoploss(
            pair="ETH/BTC",
            amount=1,
            stop_price=190,
            order_types={"stoploss_on_exchange_limit_ratio": 1.05},
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


def test_stoploss_adjust_htx(mocker, default_conf):
    """测试HTX交易所的止损调整功能"""
    exchange = get_patched_exchange(mocker, default_conf, exchange="htx")
    order = {
        "type": "stop",
        "price": 1500,
        "stopPrice": "1500",
    }
    # 测试调整逻辑
    assert exchange.stoploss_adjust(1501, order, "sell")
    assert not exchange.stoploss_adjust(1499, order, "sell")
    # 测试无效订单情况下的调整逻辑
    assert exchange.stoploss_adjust(1501, order, "sell")