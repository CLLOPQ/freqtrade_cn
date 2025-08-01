# pragma pylint: disable=missing-docstring, C0103
import logging
import time
from collections import deque
from unittest.mock import MagicMock

from freqtrade.enums import RPCMessageType
from freqtrade.rpc import RPCManager
from freqtrade.rpc.api_server.webserver import ApiServer
from tests.conftest import get_patched_freqtradebot, log_has


def test__init__(mocker, default_conf) -> None:
    default_conf["telegram"]["enabled"] = False

    rpc_manager = RPCManager(get_patched_freqtradebot(mocker, default_conf))
    assert rpc_manager.registered_modules == []


def test_init_telegram_disabled(mocker, default_conf, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    default_conf["telegram"]["enabled"] = False
    rpc_manager = RPCManager(get_patched_freqtradebot(mocker, default_conf))

    assert not log_has("启用 rpc.telegram ...", caplog)
    assert rpc_manager.registered_modules == []


def test_init_telegram_enabled(mocker, default_conf, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    default_conf["telegram"]["enabled"] = True
    mocker.patch("freqtrade.rpc.telegram.Telegram._init", MagicMock())
    rpc_manager = RPCManager(get_patched_freqtradebot(mocker, default_conf))

    assert log_has("启用 rpc.telegram ...", caplog)
    len_modules = len(rpc_manager.registered_modules)
    assert len_modules == 1
    assert "telegram" in [mod.name for mod in rpc_manager.registered_modules]


def test_cleanup_telegram_disabled(mocker, default_conf, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    telegram_mock = mocker.patch("freqtrade.rpc.telegram.Telegram.cleanup", MagicMock())
    default_conf["telegram"]["enabled"] = False

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc_manager = RPCManager(freqtradebot)
    rpc_manager.cleanup()

    assert not log_has("清理 rpc.telegram ...", caplog)
    assert telegram_mock.call_count == 0


def test_cleanup_telegram_enabled(mocker, default_conf, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    default_conf["telegram"]["enabled"] = True
    mocker.patch("freqtrade.rpc.telegram.Telegram._init", MagicMock())
    telegram_mock = mocker.patch("freqtrade.rpc.telegram.Telegram.cleanup", MagicMock())

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc_manager = RPCManager(freqtradebot)

    # 检查是否已注册Telegram模块
    assert "telegram" in [mod.name for mod in rpc_manager.registered_modules]

    rpc_manager.cleanup()
    assert log_has("清理 rpc.telegram ...", caplog)
    assert "telegram" not in [mod.name for mod in rpc_manager.registered_modules]
    assert telegram_mock.call_count == 1


def test_send_msg_telegram_disabled(mocker, default_conf, caplog) -> None:
    telegram_mock = mocker.patch("freqtrade.rpc.telegram.Telegram.send_msg", MagicMock())
    default_conf["telegram"]["enabled"] = False

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc_manager = RPCManager(freqtradebot)
    rpc_manager.send_msg({"type": RPCMessageType.STATUS, "status": "test"})

    assert log_has("发送 rpc 消息: {'type': status, 'status': 'test'}", caplog)
    assert telegram_mock.call_count == 0


def test_send_msg_telegram_error(mocker, default_conf, caplog) -> None:
    mocker.patch("freqtrade.rpc.telegram.Telegram._init", MagicMock())
    mocker.patch("freqtrade.rpc.telegram.Telegram.send_msg", side_effect=ValueError())
    default_conf["telegram"]["enabled"] = True
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc_manager = RPCManager(freqtradebot)
    rpc_manager.send_msg({"type": RPCMessageType.STATUS, "status": "test"})

    assert log_has("发送 rpc 消息: {'type': status, 'status': 'test'}", caplog)
    assert log_has("RPC模块telegram内发生异常", caplog)


def test_process_msg_queue(mocker, default_conf, caplog) -> None:
    telegram_mock = mocker.patch("freqtrade.rpc.telegram.Telegram.send_msg")
    default_conf["telegram"]["enabled"] = True
    default_conf["telegram"]["allow_custom_messages"] = True
    mocker.patch("freqtrade.rpc.telegram.Telegram._init")

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc_manager = RPCManager(freqtradebot)
    queue = deque()
    queue.append("测试消息")
    queue.append("测试消息 2")
    rpc_manager.process_msg_queue(queue)

    assert log_has("发送 rpc strategy_msg: 测试消息", caplog)
    assert log_has("发送 rpc strategy_msg: 测试消息 2", caplog)
    assert telegram_mock.call_count == 2


def test_send_msg_telegram_enabled(mocker, default_conf, caplog) -> None:
    default_conf["telegram"]["enabled"] = True
    telegram_mock = mocker.patch("freqtrade.rpc.telegram.Telegram.send_msg")
    mocker.patch("freqtrade.rpc.telegram.Telegram._init")
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc_manager = RPCManager(freqtradebot)
    rpc_manager.send_msg({"type": RPCMessageType.STATUS, "status": "test"})

    assert log_has("发送 rpc 消息: {'type': status, 'status': 'test'}", caplog)
    assert telegram_mock.call_count == 1


def test_init_webhook_disabled(mocker, default_conf, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    default_conf["telegram"]["enabled"] = False
    default_conf["webhook"] = {"enabled": False}
    rpc_manager = RPCManager(get_patched_freqtradebot(mocker, default_conf))

    assert not log_has("启用 rpc.webhook ...", caplog)
    assert rpc_manager.registered_modules == []


def test_init_webhook_enabled(mocker, default_conf, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    default_conf["telegram"]["enabled"] = False
    default_conf["webhook"] = {"enabled": True, "url": "https://DEADBEEF.com"}
    rpc_manager = RPCManager(get_patched_freqtradebot(mocker, default_conf))

    assert log_has("启用 rpc.webhook ...", caplog)
    assert len(rpc_manager.registered_modules) == 1
    assert "webhook" in [mod.name for mod in rpc_manager.registered_modules]


def test_send_msg_webhook_CustomMessagetype(mocker, default_conf, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    default_conf["telegram"]["enabled"] = False
    default_conf["webhook"] = {"enabled": True, "url": "https://DEADBEEF.com"}
    mocker.patch(
        "freqtrade.rpc.webhook.Webhook.send_msg", MagicMock(side_effect=NotImplementedError)
    )
    rpc_manager = RPCManager(get_patched_freqtradebot(mocker, default_conf))

    assert "webhook" in [mod.name for mod in rpc_manager.registered_modules]
    rpc_manager.send_msg({"type": RPCMessageType.STARTUP, "status": "TestMessage"})
    assert log_has("处理程序webhook未实现消息类型'startup'。", caplog)


def test_startupmessages_telegram_enabled(mocker, default_conf) -> None:
    default_conf["telegram"]["enabled"] = True
    telegram_mock = mocker.patch("freqtrade.rpc.telegram.Telegram.send_msg", MagicMock())
    mocker.patch("freqtrade.rpc.telegram.Telegram._init", MagicMock())

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc_manager = RPCManager(freqtradebot)
    rpc_manager.startup_messages(default_conf, freqtradebot.pairlists, freqtradebot.protections)

    assert telegram_mock.call_count == 3
    assert "*交易所:* `binance`" in telegram_mock.call_args_list[1][0][0]["status"]

    telegram_mock.reset_mock()
    default_conf["dry_run"] = True
    default_conf["whitelist"] = {"method": "VolumePairList", "config": {"number_assets": 20}}
    default_conf["_strategy_protections"] = [
        {"method": "StoplossGuard", "lookback_period": 60, "trade_limit": 2, "stop_duration": 60}
    ]
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)

    rpc_manager.startup_messages(default_conf, freqtradebot.pairlists, freqtradebot.protections)
    assert telegram_mock.call_count == 4
    assert "已启用模拟交易。" in telegram_mock.call_args_list[0][0][0]["status"]
    assert "StoplossGuard" in telegram_mock.call_args_list[-1][0][0]["status"]


def test_init_apiserver_disabled(mocker, default_conf, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    run_mock = MagicMock()
    mocker.patch("freqtrade.rpc.api_server.ApiServer.start_api", run_mock)
    default_conf["telegram"]["enabled"] = False
    rpc_manager = RPCManager(get_patched_freqtradebot(mocker, default_conf))

    assert not log_has("启用 rpc.api_server", caplog)
    assert rpc_manager.registered_modules == []
    assert run_mock.call_count == 0


def test_init_apiserver_enabled(mocker, default_conf, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    run_mock = MagicMock()
    mocker.patch("freqtrade.rpc.api_server.ApiServer.start_api", run_mock)

    default_conf["telegram"]["enabled"] = False
    default_conf["api_server"] = {
        "enabled": True,
        "listen_ip_address": "127.0.0.1",
        "listen_port": 8080,
        "username": "TestUser",
        "password": "TestPass",
    }
    rpc_manager = RPCManager(get_patched_freqtradebot(mocker, default_conf))

    # 等待线程启动
    time.sleep(0.5)
    assert log_has("启用 rpc.api_server", caplog)
    assert len(rpc_manager.registered_modules) == 1
    assert "apiserver" in [mod.name for mod in rpc_manager.registered_modules]
    assert run_mock.call_count == 1
    ApiServer.shutdown()