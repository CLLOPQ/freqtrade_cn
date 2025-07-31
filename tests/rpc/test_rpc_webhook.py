# pragma pylint: disable=missing-docstring, C0103, protected-access

import logging
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from requests import RequestException

from freqtrade.enums import ExitType, RPCMessageType
from freqtrade.rpc import RPC
from freqtrade.rpc.discord import Discord
from freqtrade.rpc.webhook import Webhook
from tests.conftest import get_patched_freqtradebot, log_has


def get_webhook_dict() -> dict:
    return {
        "enabled": True,
        "url": "https://maker.ifttt.com/trigger/freqtrade_test/with/key/c764udvJ5jfSlswVRukZZ2/",
        "webhookentry": {
            # 故意设置错误，因为"entry"应该有优先级
            "value1": "购买 {pair55555}",
        },
        "entry": {
            "value1": "购买 {pair}",
            "value2": "限价 {limit:8f}",
            "value3": "{stake_amount:8f} {stake_currency}",
            "value4": "杠杆 {leverage:.1f}",
            "value5": "方向 {direction}",
        },
        "webhookentrycancel": {
            "value1": "取消 {pair} 的未成交买入订单",
            "value2": "限价 {limit:8f}",
            "value3": "{stake_amount:8f} {stake_currency}",
            "value4": "杠杆 {leverage:.1f}",
            "value5": "方向 {direction}",
        },
        "webhookentryfill": {
            "value1": "{pair} 的买入订单已成交",
            "value2": "成交价格 {open_rate:8f}",
            "value3": "{stake_amount:8f} {stake_currency}",
            "value4": "杠杆 {leverage:.1f}",
            "value5": "方向 {direction}",
        },
        "webhookexit": {
            "value1": "卖出 {pair}",
            "value2": "限价 {limit:8f}",
            "value3": "利润: {profit_amount:8f} {stake_currency} ({profit_ratio})",
        },
        "webhookexitcancel": {
            "value1": "取消 {pair} 的未成交卖出订单",
            "value2": "限价 {limit:8f}",
            "value3": "利润: {profit_amount:8f} {stake_currency} ({profit_ratio})",
        },
        "webhookexitfill": {
            "value1": "{pair} 的卖出订单已成交",
            "value2": "成交价格 {close_rate:8f}",
            "value3": "",
        },
        "webhookstatus": {"value1": "状态: {status}", "value2": "", "value3": ""},
    }


def test__init__(mocker, default_conf):
    default_conf["webhook"] = {"enabled": True, "url": "https://DEADBEEF.com"}
    webhook = Webhook(RPC(get_patched_freqtradebot(mocker, default_conf)), default_conf)
    assert webhook._config == default_conf


def test_send_msg_webhook(default_conf, mocker):
    default_conf["webhook"] = get_webhook_dict()
    msg_mock = MagicMock()
    mocker.patch("freqtrade.rpc.webhook.Webhook._send_msg", msg_mock)
    webhook = Webhook(RPC(get_patched_freqtradebot(mocker, default_conf)), default_conf)
    # 测试买入
    msg_mock = MagicMock()
    mocker.patch("freqtrade.rpc.webhook.Webhook._send_msg", msg_mock)
    msg = {
        "type": RPCMessageType.ENTRY,
        "exchange": "Binance",
        "pair": "ETH/BTC",
        "leverage": 1.0,
        "direction": "Long",
        "limit": 0.005,
        "stake_amount": 0.8,
        "stake_amount_fiat": 500,
        "stake_currency": "BTC",
        "fiat_currency": "EUR",
    }
    webhook.send_msg(msg=msg)
    assert msg_mock.call_count == 1
    assert msg_mock.call_args[0][0]["value1"] == default_conf["webhook"]["entry"]["value1"].format(
        **msg
    )
    assert msg_mock.call_args[0][0]["value2"] == default_conf["webhook"]["entry"]["value2"].format(
        **msg
    )
    assert msg_mock.call_args[0][0]["value3"] == default_conf["webhook"]["entry"]["value3"].format(
        **msg
    )
    assert msg_mock.call_args[0][0]["value4"] == default_conf["webhook"]["entry"]["value4"].format(
        **msg
    )
    assert msg_mock.call_args[0][0]["value5"] == default_conf["webhook"]["entry"]["value5"].format(
        **msg
    )
    # 测试做空
    msg_mock.reset_mock()

    msg = {
        "type": RPCMessageType.ENTRY,
        "exchange": "Binance",
        "pair": "ETH/BTC",
        "leverage": 2.0,
        "direction": "Short",
        "limit": 0.005,
        "stake_amount": 0.8,
        "stake_amount_fiat": 500,
        "stake_currency": "BTC",
        "fiat_currency": "EUR",
    }
    webhook.send_msg(msg=msg)
    assert msg_mock.call_count == 1
    assert msg_mock.call_args[0][0]["value1"] == default_conf["webhook"]["entry"]["value1"].format(
        **msg
    )
    assert msg_mock.call_args[0][0]["value2"] == default_conf["webhook"]["entry"]["value2"].format(
        **msg
    )
    assert msg_mock.call_args[0][0]["value3"] == default_conf["webhook"]["entry"]["value3"].format(
        **msg
    )
    assert msg_mock.call_args[0][0]["value4"] == default_conf["webhook"]["entry"]["value4"].format(
        **msg
    )
    assert msg_mock.call_args[0][0]["value5"] == default_conf["webhook"]["entry"]["value5"].format(
        **msg
    )
    # 测试取消买入
    msg_mock.reset_mock()

    msg = {
        "type": RPCMessageType.ENTRY_CANCEL,
        "exchange": "Binance",
        "pair": "ETH/BTC",
        "leverage": 1.0,
        "direction": "Long",
        "limit": 0.005,
        "stake_amount": 0.8,
        "stake_amount_fiat": 500,
        "stake_currency": "BTC",
        "fiat_currency": "EUR",
    }
    webhook.send_msg(msg=msg)
    assert msg_mock.call_count == 1
    assert msg_mock.call_args[0][0]["value1"] == default_conf["webhook"]["webhookentrycancel"][
        "value1"
    ].format(**msg)
    assert msg_mock.call_args[0][0]["value2"] == default_conf["webhook"]["webhookentrycancel"][
        "value2"
    ].format(**msg)
    assert msg_mock.call_args[0][0]["value3"] == default_conf["webhook"]["webhookentrycancel"][
        "value3"
    ].format(**msg)
    # 测试取消做空
    msg_mock.reset_mock()

    msg = {
        "type": RPCMessageType.ENTRY_CANCEL,
        "exchange": "Binance",
        "pair": "ETH/BTC",
        "leverage": 2.0,
        "direction": "Short",
        "limit": 0.005,
        "stake_amount": 0.8,
        "stake_amount_fiat": 500,
        "stake_currency": "BTC",
        "fiat_currency": "EUR",
    }
    webhook.send_msg(msg=msg)
    assert msg_mock.call_count == 1
    assert msg_mock.call_args[0][0]["value1"] == default_conf["webhook"]["webhookentrycancel"][
        "value1"
    ].format(**msg)
    assert msg_mock.call_args[0][0]["value2"] == default_conf["webhook"]["webhookentrycancel"][
        "value2"
    ].format(**msg)
    assert msg_mock.call_args[0][0]["value3"] == default_conf["webhook"]["webhookentrycancel"][
        "value3"
    ].format(**msg)
    assert msg_mock.call_args[0][0]["value4"] == default_conf["webhook"]["webhookentrycancel"][
        "value4"
    ].format(**msg)
    assert msg_mock.call_args[0][0]["value5"] == default_conf["webhook"]["webhookentrycancel"][
        "value5"
    ].format(**msg)
    # 测试买入成交
    msg_mock.reset_mock()

    msg = {
        "type": RPCMessageType.ENTRY_FILL,
        "exchange": "Binance",
        "pair": "ETH/BTC",
        "leverage": 1.0,
        "direction": "Long",
        "open_rate": 0.005,
        "stake_amount": 0.8,
        "stake_amount_fiat": 500,
        "stake_currency": "BTC",
        "fiat_currency": "EUR",
    }
    webhook.send_msg(msg=msg)
    assert msg_mock.call_count == 1
    assert msg_mock.call_args[0][0]["value1"] == default_conf["webhook"]["webhookentryfill"][
        "value1"
    ].format(**msg)
    assert msg_mock.call_args[0][0]["value2"] == default_conf["webhook"]["webhookentryfill"][
        "value2"
    ].format(**msg)
    assert msg_mock.call_args[0][0]["value3"] == default_conf["webhook"]["webhookentryfill"][
        "value3"
    ].format(**msg)
    assert msg_mock.call_args[0][0]["value4"] == default_conf["webhook"]["webhookentrycancel"][
        "value4"
    ].format(**msg)
    assert msg_mock.call_args[0][0]["value5"] == default_conf["webhook"]["webhookentrycancel"][
        "value5"
    ].format(**msg)
    # 测试做空成交
    msg_mock.reset_mock()

    msg = {
        "type": RPCMessageType.ENTRY_FILL,
        "exchange": "Binance",
        "pair": "ETH/BTC",
        "leverage": 2.0,
        "direction": "Short",
        "open_rate": 0.005,
        "stake_amount": 0.8,
        "stake_amount_fiat": 500,
        "stake_currency": "BTC",
        "fiat_currency": "EUR",
    }
    webhook.send_msg(msg=msg)
    assert msg_mock.call_count == 1
    assert msg_mock.call_args[0][0]["value1"] == default_conf["webhook"]["webhookentryfill"][
        "value1"
    ].format(**msg)
    assert msg_mock.call_args[0][0]["value2"] == default_conf["webhook"]["webhookentryfill"][
        "value2"
    ].format(**msg)
    assert msg_mock.call_args[0][0]["value3"] == default_conf["webhook"]["webhookentryfill"][
        "value3"
    ].format(**msg)
    assert msg_mock.call_args[0][0]["value4"] == default_conf["webhook"]["webhookentrycancel"][
        "value4"
    ].format(**msg)
    assert msg_mock.call_args[0][0]["value5"] == default_conf["webhook"]["webhookentrycancel"][
        "value5"
    ].format(**msg)
    # 测试卖出
    msg_mock.reset_mock()

    msg = {
        "type": RPCMessageType.EXIT,
        "exchange": "Binance",
        "pair": "ETH/BTC",
        "gain": "profit",
        "limit": 0.005,
        "amount": 0.8,
        "order_type": "limit",
        "open_rate": 0.004,
        "current_rate": 0.005,
        "profit_amount": 0.001,
        "profit_ratio": 0.20,
        "stake_currency": "BTC",
        "sell_reason": ExitType.STOP_LOSS.value,
    }
    webhook.send_msg(msg=msg)
    assert msg_mock.call_count == 1
    assert msg_mock.call_args[0][0]["value1"] == default_conf["webhook"]["webhookexit"][
        "value1"
    ].format(**msg)
    assert msg_mock.call_args[0][0]["value2"] == default_conf["webhook"]["webhookexit"][
        "value2"
    ].format(**msg)
    assert msg_mock.call_args[0][0]["value3"] == default_conf["webhook"]["webhookexit"][
        "value3"
    ].format(**msg)
    # 测试取消卖出
    msg_mock.reset_mock()
    msg = {
        "type": RPCMessageType.EXIT_CANCEL,
        "exchange": "Binance",
        "pair": "ETH/BTC",
        "gain": "profit",
        "limit": 0.005,
        "amount": 0.8,
        "order_type": "limit",
        "open_rate": 0.004,
        "current_rate": 0.005,
        "profit_amount": 0.001,
        "profit_ratio": 0.20,
        "stake_currency": "BTC",
        "sell_reason": ExitType.STOP_LOSS.value,
    }
    webhook.send_msg(msg=msg)
    assert msg_mock.call_count == 1
    assert msg_mock.call_args[0][0]["value1"] == default_conf["webhook"]["webhookexitcancel"][
        "value1"
    ].format(**msg)
    assert msg_mock.call_args[0][0]["value2"] == default_conf["webhook"]["webhookexitcancel"][
        "value2"
    ].format(**msg)
    assert msg_mock.call_args[0][0]["value3"] == default_conf["webhook"]["webhookexitcancel"][
        "value3"
    ].format(**msg)
    # 测试卖出成交
    msg_mock.reset_mock()
    msg = {
        "type": RPCMessageType.EXIT_FILL,
        "exchange": "Binance",
        "pair": "ETH/BTC",
        "gain": "profit",
        "close_rate": 0.005,
        "amount": 0.8,
        "order_type": "limit",
        "open_rate": 0.004,
        "current_rate": 0.005,
        "profit_amount": 0.001,
        "profit_ratio": 0.20,
        "stake_currency": "BTC",
        "sell_reason": ExitType.STOP_LOSS.value,
    }
    webhook.send_msg(msg=msg)
    assert msg_mock.call_count == 1
    assert msg_mock.call_args[0][0]["value1"] == default_conf["webhook"]["webhookexitfill"][
        "value1"
    ].format(**msg)
    assert msg_mock.call_args[0][0]["value2"] == default_conf["webhook"]["webhookexitfill"][
        "value2"
    ].format(**msg)
    assert msg_mock.call_args[0][0]["value3"] == default_conf["webhook"]["webhookexitfill"][
        "value3"
    ].format(**msg)

    for msgtype in [RPCMessageType.STATUS, RPCMessageType.WARNING, RPCMessageType.STARTUP]:
        # 测试通知
        msg = {"type": msgtype, "status": "由于超时，BTC的未成交卖出订单已取消"}
        msg_mock = MagicMock()
        mocker.patch("freqtrade.rpc.webhook.Webhook._send_msg", msg_mock)
        webhook.send_msg(msg)
        assert msg_mock.call_count == 1
        assert msg_mock.call_args[0][0]["value1"] == default_conf["webhook"]["webhookstatus"][
            "value1"
        ].format(**msg)
        assert msg_mock.call_args[0][0]["value2"] == default_conf["webhook"]["webhookstatus"][
            "value2"
        ].format(**msg)
        assert msg_mock.call_args[0][0]["value3"] == default_conf["webhook"]["webhookstatus"][
            "value3"
        ].format(**msg)


def test_exception_send_msg(default_conf, mocker, caplog):
    caplog.set_level(logging.DEBUG)
    default_conf["webhook"] = get_webhook_dict()
    del default_conf["webhook"]["entry"]
    del default_conf["webhook"]["webhookentry"]

    webhook = Webhook(RPC(get_patched_freqtradebot(mocker, default_conf)), default_conf)
    webhook.send_msg({"type": RPCMessageType.ENTRY})
    assert log_has(f"Webhook未配置消息类型 '{RPCMessageType.ENTRY}'", caplog)

    default_conf["webhook"] = get_webhook_dict()
    default_conf["webhook"]["strategy_msg"] = {"value1": "{DEADBEEF:8f}"}
    msg_mock = MagicMock()
    mocker.patch("freqtrade.rpc.webhook.Webhook._send_msg", msg_mock)
    webhook = Webhook(RPC(get_patched_freqtradebot(mocker, default_conf)), default_conf)
    msg = {
        "type": RPCMessageType.STRATEGY_MSG,
        "msg": "hello world",
    }
    webhook.send_msg(msg)
    assert log_has(
        "调用Webhook时出现问题。请检查您的webhook配置。异常: 'DEADBEEF'",
        caplog,
    )

    # 测试已知但未实现的消息类型不会失败
    for e in RPCMessageType:
        msg = {"type": e, "status": "随便什么内容"}
        webhook.send_msg(msg)

    # 再次测试已知但未实现的消息类型不会失败
    for e in RPCMessageType:
        msg = {"type": e, "status": "随便什么内容"}
        webhook.send_msg(msg)


def test__send_msg(default_conf, mocker, caplog):
    default_conf["webhook"] = get_webhook_dict()
    webhook = Webhook(RPC(get_patched_freqtradebot(mocker, default_conf)), default_conf)
    msg = {"value1": "DEADBEEF", "value2": "ALIVEBEEF", "value3": "FREQTRADE"}
    post = MagicMock()
    mocker.patch("freqtrade.rpc.webhook.post", post)
    webhook._send_msg(msg)

    assert post.call_count == 1
    assert post.call_args[1] == {"data": msg, "timeout": 10}
    assert post.call_args[0] == (default_conf["webhook"]["url"],)

    post = MagicMock(side_effect=RequestException)
    mocker.patch("freqtrade.rpc.webhook.post", post)
    webhook._send_msg(msg)
    assert log_has("无法调用webhook URL。异常: ", caplog)


def test__send_msg_with_json_format(default_conf, mocker, caplog):
    default_conf["webhook"] = get_webhook_dict()
    default_conf["webhook"]["format"] = "json"
    webhook = Webhook(RPC(get_patched_freqtradebot(mocker, default_conf)), default_conf)
    msg = {"text": "你好"}
    post = MagicMock()
    mocker.patch("freqtrade.rpc.webhook.post", post)
    webhook._send_msg(msg)

    assert post.call_args[1] == {"json": msg, "timeout": 10}


def test__send_msg_with_raw_format(default_conf, mocker, caplog):
    default_conf["webhook"] = get_webhook_dict()
    default_conf["webhook"]["format"] = "raw"
    webhook = Webhook(RPC(get_patched_freqtradebot(mocker, default_conf)), default_conf)
    msg = {"data": "你好"}
    post = MagicMock()
    mocker.patch("freqtrade.rpc.webhook.post", post)
    webhook._send_msg(msg)

    assert post.call_args[1] == {
        "data": msg["data"],
        "headers": {"Content-Type": "text/plain"},
        "timeout": 10,
    }


def test_send_msg_discord(default_conf, mocker):
    default_conf["discord"] = {"enabled": True, "webhook_url": "https://webhookurl..."}
    msg_mock = MagicMock()
    mocker.patch("freqtrade.rpc.webhook.Webhook._send_msg", msg_mock)
    discord = Discord(RPC(get_patched_freqtradebot(mocker, default_conf)), default_conf)

    msg = {
        "type": RPCMessageType.EXIT_FILL,
        "trade_id": 1,
        "exchange": "Binance",
        "pair": "ETH/BTC",
        "direction": "Long",
        "gain": "profit",
        "close_rate": 0.005,
        "amount": 0.8,
        "order_type": "limit",
        "open_date": datetime.now() - timedelta(days=1),
        "close_date": datetime.now(),
        "open_rate": 0.004,
        "current_rate": 0.005,
        "profit_amount": 0.001,
        "profit_ratio": 0.20,
        "stake_currency": "BTC",
        "enter_tag": "enter_tagggg",
        "exit_reason": ExitType.STOP_LOSS.value,
    }
    discord.send_msg(msg=msg)

    assert msg_mock.call_count == 1
    assert "embeds" in msg_mock.call_args_list[0][0][0]
    assert "title" in msg_mock.call_args_list[0][0][0]["embeds"][0]
    assert "color" in msg_mock.call_args_list[0][0][0]["embeds"][0]
    assert "fields" in msg_mock.call_args_list[0][0][0]["embeds"][0]