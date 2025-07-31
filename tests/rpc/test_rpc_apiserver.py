"""
rpc/api_server.py 的单元测试文件
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import ANY, MagicMock, PropertyMock

import pandas as pd
import pytest
import rapidjson
import uvicorn
from fastapi import FastAPI, WebSocketDisconnect
from fastapi.exceptions import HTTPException
from fastapi.testclient import TestClient
from requests.auth import _basic_auth_str
from sqlalchemy import select

from freqtrade.__init__ import __version__
from freqtrade.enums import CandleType, RunMode, State, TradingMode
from freqtrade.exceptions import DependencyException, ExchangeError, OperationalException
from freqtrade.loggers import setup_logging, setup_logging_pre
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.persistence import CustomDataWrapper, Trade
from freqtrade.rpc import RPC
from freqtrade.rpc.api_server import ApiServer
from freqtrade.rpc.api_server.api_auth import create_token, get_user_from_token
from freqtrade.rpc.api_server.uvicorn_threaded import UvicornServer
from freqtrade.rpc.api_server.webserver_bgwork import ApiBG
from freqtrade.util.datetime_helpers import format_date
from tests.conftest import (
    CURRENT_TEST_STRATEGY,
    EXMS,
    create_mock_trades,
    create_mock_trades_usdt,
    generate_test_data,
    get_mock_coro,
    get_patched_freqtradebot,
    log_has,
    log_has_re,
    patch_get_signal,
)


BASE_URI = "/api/v1"
_TEST_USER = "FreqTrader"  # 测试用户名
_TEST_PASS = "SuperSecurePassword1!"  # 测试密码
_TEST_WS_TOKEN = "secret_Ws_t0ken"  # 测试WebSocket令牌


@pytest.fixture
def botclient(default_conf, mocker):
    """机器人客户端测试固件"""
    setup_logging_pre()
    setup_logging(default_conf)
    default_conf["runmode"] = RunMode.DRY_RUN
    default_conf.update(
        {
            "api_server": {
                "enabled": True,
                "listen_ip_address": "127.0.0.1",
                "listen_port": 8080,
                "CORS_origins": ["http://example.com"],
                "username": _TEST_USER,
                "password": _TEST_PASS,
                "ws_token": _TEST_WS_TOKEN,
            }
        }
    )

    ftbot = get_patched_freqtradebot(mocker, default_conf)
    rpc = RPC(ftbot)
    mocker.patch("freqtrade.rpc.api_server.ApiServer.start_api", MagicMock())
    apiserver = None
    try:
        apiserver = ApiServer(default_conf)
        apiserver.add_rpc_handler(rpc)
        # 我们需要使用TestClient作为上下文管理器来
        # 正确处理生命周期事件
        with TestClient(apiserver.app) as client:
            yield ftbot, client
        # 清理...？
    finally:
        if apiserver:
            apiserver.cleanup()
        ApiServer.shutdown()


def client_post(client: TestClient, url, data=None):
    """客户端POST请求辅助函数"""
    if data is None:
        data = {}
    return client.post(
        url,
        json=data,
        headers={
            "Authorization": _basic_auth_str(_TEST_USER, _TEST_PASS),
            "Origin": "http://example.com",
            "content-type": "application/json",
        },
    )


def client_patch(client: TestClient, url, data=None):
    """客户端PATCH请求辅助函数"""
    if data is None:
        data = {}
    return client.patch(
        url,
        json=data,
        headers={
            "Authorization": _basic_auth_str(_TEST_USER, _TEST_PASS),
            "Origin": "http://example.com",
            "content-type": "application/json",
        },
    )


def client_get(client: TestClient, url):
    """客户端GET请求辅助函数"""
    # 添加虚假的Origin以确保CORS生效
    return client.get(
        url,
        headers={
            "Authorization": _basic_auth_str(_TEST_USER, _TEST_PASS),
            "Origin": "http://example.com",
        },
    )


def client_delete(client: TestClient, url):
    """客户端DELETE请求辅助函数"""
    # 添加虚假的Origin以确保CORS生效
    return client.delete(
        url,
        headers={
            "Authorization": _basic_auth_str(_TEST_USER, _TEST_PASS),
            "Origin": "http://example.com",
        },
    )


def assert_response(response, expected_code=200, needs_cors=True):
    """断言响应状态码和头部信息"""
    assert response.status_code == expected_code
    assert response.headers.get("content-type") == "application/json"
    if needs_cors:
        assert ("access-control-allow-credentials", "true") in response.headers.items()
        assert ("access-control-allow-origin", "http://example.com") in response.headers.items()


def test_api_not_found(botclient):
    """测试API未找到路径"""
    _ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/invalid_url")
    assert_response(rc, 404)
    assert rc.json() == {"detail": "未找到"}


def test_api_ui_fallback(botclient, mocker):
    """测试API UI后备页面"""
    _ftbot, client = botclient

    rc = client_get(client, "/favicon.ico")
    assert rc.status_code == 200

    rc = client_get(client, "/fallback_file.html")
    assert rc.status_code == 200
    assert "`freqtrade install-ui`" in rc.text

    # 转发到fallback_html或index.html（取决于是否已安装）
    rc = client_get(client, "/something")
    assert rc.status_code == 200

    rc = client_get(client, "/something.js")
    assert rc.status_code == 200

    # 测试目录遍历而不模拟
    rc = client_get(client, "%2F%2F%2Fetc/passwd")
    assert rc.status_code == 200
    # 允许后备或真实UI
    assert "`freqtrade install-ui`" in rc.text or "<!DOCTYPE html>" in rc.text

    mocker.patch.object(Path, "is_file", MagicMock(side_effect=[True, False]))
    rc = client_get(client, "%2F%2F%2Fetc/passwd")
    assert rc.status_code == 200

    assert "`freqtrade install-ui`" in rc.text


def test_api_ui_version(botclient, mocker):
    """测试API UI版本"""
    _ftbot, client = botclient

    mocker.patch("freqtrade.commands.deploy_ui.read_ui_version", return_value="0.1.2")
    rc = client_get(client, "/ui_version")
    assert rc.status_code == 200
    assert rc.json()["version"] == "0.1.2"


def test_api_auth():
    """测试API认证"""
    with pytest.raises(ValueError):
        create_token({"identity": {"u": "Freqtrade"}}, "secret1234", token_type="NotATokenType")

    token = create_token({"identity": {"u": "Freqtrade"}}, "secret1234")
    assert isinstance(token, str)

    u = get_user_from_token(token, "secret1234")
    assert u == "Freqtrade"
    with pytest.raises(HTTPException):
        get_user_from_token(token, "secret1234", token_type="refresh")
    # 创建无效令牌
    token = create_token({"identity": {"u1": "Freqrade"}}, "secret1234")
    with pytest.raises(HTTPException):
        get_user_from_token(token, "secret1234")

    with pytest.raises(HTTPException):
        get_user_from_token(b"not_a_token", "secret1234")


def test_api_ws_auth(botclient):
    """测试API WebSocket认证"""
    ftbot, client = botclient

    def url(token):
        return f"/api/v1/message/ws?token={token}"

    bad_token = "bad-ws_token"
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(url(bad_token)):
            pass

    good_token = _TEST_WS_TOKEN
    with client.websocket_connect(url(good_token)):
        pass

    jwt_secret = ftbot.config["api_server"].get("jwt_secret_key", "super-secret")
    jwt_token = create_token({"identity": {"u": "Freqtrade"}}, jwt_secret)
    with client.websocket_connect(url(jwt_token)):
        pass


def test_api_unauthorized(botclient):
    """测试API未授权访问"""
    ftbot, client = botclient
    rc = client.get(f"{BASE_URI}/ping")
    assert_response(rc, needs_cors=False)
    assert rc.json() == {"status": "pong"}

    # 不发送用户名/密码信息
    rc = client.get(f"{BASE_URI}/version")
    assert_response(rc, 401, needs_cors=False)
    assert rc.json() == {"detail": "未授权"}

    # 只更改用户名
    ftbot.config["api_server"]["username"] = "Ftrader"
    rc = client_get(client, f"{BASE_URI}/version")
    assert_response(rc, 401)
    assert rc.json() == {"detail": "未授权"}

    # 只更改密码
    ftbot.config["api_server"]["username"] = _TEST_USER
    ftbot.config["api_server"]["password"] = "WrongPassword"
    rc = client_get(client, f"{BASE_URI}/version")
    assert_response(rc, 401)
    assert rc.json() == {"detail": "未授权"}

    ftbot.config["api_server"]["username"] = "Ftrader"
    ftbot.config["api_server"]["password"] = "WrongPassword"

    rc = client_get(client, f"{BASE_URI}/version")
    assert_response(rc, 401)
    assert rc.json() == {"detail": "未授权"}


def test_api_token_login(botclient):
    """测试API令牌登录"""
    _ftbot, client = botclient
    rc = client.post(
        f"{BASE_URI}/token/login",
        data=None,
        headers={
            "Authorization": _basic_auth_str("WRONG_USER", "WRONG_PASS"),
            "Origin": "http://example.com",
        },
    )
    assert_response(rc, 401)
    rc = client_post(client, f"{BASE_URI}/token/login")
    assert_response(rc)
    assert "access_token" in rc.json()
    assert "refresh_token" in rc.json()

    # 测试JWT令牌也能正常工作的认证
    rc = client.get(
        f"{BASE_URI}/count",
        headers={
            "Authorization": f"Bearer {rc.json()['access_token']}",
            "Origin": "http://example.com",
        },
    )
    assert_response(rc)


def test_api_token_refresh(botclient):
    """测试API令牌刷新"""
    _ftbot, client = botclient
    rc = client_post(client, f"{BASE_URI}/token/login")
    assert_response(rc)
    rc = client.post(
        f"{BASE_URI}/token/refresh",
        data=None,
        headers={
            "Authorization": f"Bearer {rc.json()['refresh_token']}",
            "Origin": "http://example.com",
        },
    )
    assert_response(rc)
    assert "access_token" in rc.json()
    assert "refresh_token" not in rc.json()


def test_api_stop_workflow(botclient):
    """测试API停止工作流程"""
    ftbot, client = botclient
    assert ftbot.state == State.RUNNING
    rc = client_post(client, f"{BASE_URI}/stop")
    assert_response(rc)
    assert rc.json() == {"status": "正在停止交易机器人..."}
    assert ftbot.state == State.STOPPED

    # 再次停止机器人
    rc = client_post(client, f"{BASE_URI}/stop")
    assert_response(rc)
    assert rc.json() == {"status": "已经停止"}

    # 启动机器人
    rc = client_post(client, f"{BASE_URI}/start")
    assert_response(rc)
    assert rc.json() == {"status": "正在启动交易机器人..."}
    assert ftbot.state == State.RUNNING

    # 再次调用启动
    rc = client_post(client, f"{BASE_URI}/start")
    assert_response(rc)
    assert rc.json() == {"status": "已经在运行"}


def test_api__init__(default_conf, mocker):
    """
    测试__init__()方法
    """
    default_conf.update(
        {
            "api_server": {
                "enabled": True,
                "listen_ip_address": "127.0.0.1",
                "listen_port": 8080,
                "username": "TestUser",
                "password": "testPass",
            }
        }
    )
    mocker.patch("freqtrade.rpc.telegram.Telegram._init")
    mocker.patch("freqtrade.rpc.api_server.webserver.ApiServer.start_api", MagicMock())
    apiserver = ApiServer(default_conf)
    apiserver.add_rpc_handler(RPC(get_patched_freqtradebot(mocker, default_conf)))
    assert apiserver._config == default_conf
    with pytest.raises(OperationalException, match="RPC处理器已经附加。"):
        apiserver.add_rpc_handler(RPC(get_patched_freqtradebot(mocker, default_conf)))

    apiserver.cleanup()
    ApiServer.shutdown()


def test_api_UvicornServer(mocker):
    """测试UvicornServer"""
    thread_mock = mocker.patch("freqtrade.rpc.api_server.uvicorn_threaded.threading.Thread")
    s = UvicornServer(uvicorn.Config(MagicMock(), port=8080, host="127.0.0.1"))
    assert thread_mock.call_count == 0

    # 伪造启动以避免永久睡眠
    s.started = True
    s.run_in_thread()
    assert thread_mock.call_count == 1

    s.cleanup()
    assert s.should_exit is True


def test_api_UvicornServer_run(mocker):
    """测试UvicornServer运行"""
    serve_mock = mocker.patch(
        "freqtrade.rpc.api_server.uvicorn_threaded.UvicornServer.serve", get_mock_coro(None)
    )
    s = UvicornServer(uvicorn.Config(MagicMock(), port=8080, host="127.0.0.1"))
    assert serve_mock.call_count == 0

    # 伪造启动以避免永久睡眠
    s.started = True
    s.run()
    assert serve_mock.call_count == 1


def test_api_UvicornServer_run_no_uvloop(mocker, import_fails):
    """测试UvicornServer运行无uvloop"""
    serve_mock = mocker.patch(
        "freqtrade.rpc.api_server.uvicorn_threaded.UvicornServer.serve", get_mock_coro(None)
    )
    asyncio.set_event_loop(asyncio.new_event_loop())
    s = UvicornServer(uvicorn.Config(MagicMock(), port=8080, host="127.0.0.1"))
    assert serve_mock.call_count == 0

    # 伪造启动以避免永久睡眠
    s.started = True
    s.run()
    assert serve_mock.call_count == 1


def test_api_run(default_conf, mocker, caplog):
    """测试API运行"""
    default_conf.update(
        {
            "api_server": {
                "enabled": True,
                "listen_ip_address": "127.0.0.1",
                "listen_port": 8080,
                "username": "TestUser",
                "password": "testPass",
            }
        }
    )
    mocker.patch("freqtrade.rpc.telegram.Telegram._init")

    server_inst_mock = MagicMock()
    server_inst_mock.run_in_thread = MagicMock()
    server_inst_mock.run = MagicMock()
    server_mock = MagicMock(return_value=server_inst_mock)
    mocker.patch("freqtrade.rpc.api_server.webserver.UvicornServer", server_mock)

    apiserver = ApiServer(default_conf)
    apiserver.add_rpc_handler(RPC(get_patched_freqtradebot(mocker, default_conf)))

    assert server_mock.call_count == 1
    assert apiserver._config == default_conf
    apiserver.start_api()
    assert server_mock.call_count == 2
    assert server_inst_mock.run_in_thread.call_count == 2
    assert server_inst_mock.run.call_count == 0
    assert server_mock.call_args_list[0][0][0].host == "127.0.0.1"
    assert server_mock.call_args_list[0][0][0].port == 8080
    assert isinstance(server_mock.call_args_list[0][0][0].app, FastAPI)

    assert log_has("在 127.0.0.1:8080 启动HTTP服务器", caplog)
    assert log_has("启动本地Rest服务器。", caplog)

    # 测试绑定到公共地址
    caplog.clear()
    server_mock.reset_mock()
    apiserver._config.update(
        {
            "api_server": {
                "enabled": True,
                "listen_ip_address": "0.0.0.0",
                "listen_port": 8089,
                "password": "",
            }
        }
    )
    apiserver.start_api()

    assert server_mock.call_count == 1
    assert server_inst_mock.run_in_thread.call_count == 1
    assert server_inst_mock.run.call_count == 0
    assert server_mock.call_args_list[0][0][0].host == "0.0.0.0"
    assert server_mock.call_args_list[0][0][0].port == 8089
    assert isinstance(server_mock.call_args_list[0][0][0].app, FastAPI)
    assert log_has("在 0.0.0.0:8089 启动HTTP服务器", caplog)
    assert log_has("启动本地Rest服务器。", caplog)
    assert log_has("安全警告 - 本地Rest服务器正在监听外部连接", caplog)
    assert log_has(
        "安全警告 - 这是不安全的，请在config.json中设置为您的回环地址，"
        "例如 127.0.0.1",
        caplog,
    )
    assert log_has(
        "安全警告 - 本地REST服务器未定义密码。"
        "请确保这是有意的！",
        caplog,
    )
    assert log_has_re("安全警告 - `jwt_secret_key`似乎是默认值.*", caplog)

    server_mock.reset_mock()
    apiserver._standalone = True
    apiserver.start_api()
    assert server_inst_mock.run_in_thread.call_count == 0
    assert server_inst_mock.run.call_count == 1

    apiserver1 = ApiServer(default_conf)
    assert id(apiserver1) == id(apiserver)

    apiserver._standalone = False

    # 测试崩溃的API服务器
    caplog.clear()
    mocker.patch(
        "freqtrade.rpc.api_server.webserver.UvicornServer", MagicMock(side_effect=Exception)
    )
    apiserver.start_api()
    assert log_has("API服务器启动失败。", caplog)
    apiserver.cleanup()
    ApiServer.shutdown()


def test_api_cleanup(default_conf, mocker, caplog):
    """测试API清理"""
    default_conf.update(
        {
            "api_server": {
                "enabled": True,
                "listen_ip_address": "127.0.0.1",
                "listen_port": 8080,
                "username": "TestUser",
                "password": "testPass",
            }
        }
    )
    mocker.patch("freqtrade.rpc.telegram.Telegram._init")

    server_mock = MagicMock()
    server_mock.cleanup = MagicMock()
    mocker.patch("freqtrade.rpc.api_server.webserver.UvicornServer", server_mock)

    apiserver = ApiServer(default_conf)
    apiserver.add_rpc_handler(RPC(get_patched_freqtradebot(mocker, default_conf)))

    apiserver.cleanup()
    assert apiserver._server.cleanup.call_count == 1
    assert log_has("停止API服务器", caplog)
    ApiServer.shutdown()


def test_api_reloadconf(botclient):
    """测试API重新加载配置"""
    ftbot, client = botclient

    rc = client_post(client, f"{BASE_URI}/reload_config")
    assert_response(rc)
    assert rc.json() == {"status": "正在重新加载配置..."}
    assert ftbot.state == State.RELOAD_CONFIG


def test_api_pause(botclient):
    """测试API暂停"""
    ftbot, client = botclient

    rc = client_post(client, f"{BASE_URI}/pause")
    assert_response(rc)
    assert rc.json() == {
        "status": "已暂停，从现在开始不会有新的入场。运行 /start 来启用入场。"
    }

    rc = client_post(client, f"{BASE_URI}/pause")
    assert_response(rc)
    assert rc.json() == {
        "status": "已暂停，从现在开始不会有新的入场。运行 /start 来启用入场。"
    }

    rc = client_post(client, f"{BASE_URI}/stopentry")
    assert_response(rc)
    assert rc.json() == {
        "status": "已暂停，从现在开始不会有新的入场。运行 /start 来启用入场。"
    }


def test_api_balance(botclient, mocker, rpc_balance, tickers):
    """测试API余额"""
    ftbot, client = botclient

    ftbot.config["dry_run"] = False
    mocker.patch(f"{EXMS}.get_balances", return_value=rpc_balance)
    mocker.patch(f"{EXMS}.get_tickers", tickers)
    mocker.patch(f"{EXMS}.get_valid_pair_combination", side_effect=lambda a, b: [f"{a}/{b}"])
    ftbot.wallets.update()

    rc = client_get(client, f"{BASE_URI}/balance")
    assert_response(rc)
    response = rc.json()
    assert "currencies" in response
    assert len(response["currencies"]) == 5
    assert response["currencies"][0] == {
        "currency": "BTC",
        "free": 12.0,
        "balance": 12.0,
        "used": 0.0,
        "bot_owned": pytest.approx(11.879999),
        "est_stake": 12.0,
        "est_stake_bot": pytest.approx(11.879999),
        "stake": "BTC",
        "is_position": False,
        "position": 0.0,
        "side": "long",
        "is_bot_managed": True,
    }
    assert response["total"] == 12.159513094
    assert response["total_bot"] == pytest.approx(11.879999)
    assert "starting_capital" in response
    assert "starting_capital_fiat" in response
    assert "starting_capital_pct" in response
    assert "starting_capital_ratio" in response


@pytest.mark.parametrize("is_short", [True, False])
def test_api_count(botclient, mocker, ticker, fee, markets, is_short):
    """测试API计数"""
    ftbot, client = botclient
    patch_get_signal(ftbot)
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets),
    )
    rc = client_get(client, f"{BASE_URI}/count")
    assert_response(rc)

    assert rc.json()["current"] == 0
    assert rc.json()["max"] == 1

    # 创建一些测试数据
    create_mock_trades(fee, is_short=is_short)
    rc = client_get(client, f"{BASE_URI}/count")
    assert_response(rc)
    assert rc.json()["current"] == 4
    assert rc.json()["max"] == 1

    ftbot.config["max_open_trades"] = float("inf")
    rc = client_get(client, f"{BASE_URI}/count")
    assert rc.json()["max"] == -1


def test_api_locks(botclient):
    """测试API锁定"""
    _ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/locks")
    assert_response(rc)

    assert "locks" in rc.json()

    assert rc.json()["lock_count"] == 0
    assert rc.json()["lock_count"] == len(rc.json()["locks"])

    rc = client_post(
        client,
        f"{BASE_URI}/locks",
        [
            {
                "pair": "ETH/BTC",
                "until": f"{format_date(datetime.now(timezone.utc) + timedelta(minutes=4))}Z",
                "reason": "随机原因",
            },
            {
                "pair": "XRP/BTC",
                "until": f"{format_date(datetime.now(timezone.utc) + timedelta(minutes=20))}Z",
                "reason": "deadbeef",
            },
        ],
    )
    assert_response(rc)
    assert rc.json()["lock_count"] == 2

    rc = client_get(client, f"{BASE_URI}/locks")
    assert_response(rc)

    assert rc.json()["lock_count"] == 2
    assert rc.json()["lock_count"] == len(rc.json()["locks"])
    assert "ETH/BTC" in (rc.json()["locks"][0]["pair"], rc.json()["locks"][1]["pair"])
    assert "随机原因" in (rc.json()["locks"][0]["reason"], rc.json()["locks"][1]["reason"])
    assert "deadbeef" in (rc.json()["locks"][0]["reason"], rc.json()["locks"][1]["reason"])

    # 测试删除
    rc = client_delete(client, f"{BASE_URI}/locks/1")
    assert_response(rc)
    assert rc.json()["lock_count"] == 1

    rc = client_post(client, f"{BASE_URI}/locks/delete", data={"pair": "XRP/BTC"})
    assert_response(rc)
    assert rc.json()["lock_count"] == 0


def test_api_show_config(botclient):
    """测试API显示配置"""
    ftbot, client = botclient
    patch_get_signal(ftbot)

    rc = client_get(client, f"{BASE_URI}/show_config")
    assert_response(rc)
    response = rc.json()
    assert "dry_run" in response
    assert response["exchange"] == "binance"
    assert response["timeframe"] == "5m"
    assert response["timeframe_ms"] == 300000
    assert response["timeframe_min"] == 5
    assert response["state"] == "running"
    assert response["bot_name"] == "freqtrade"
    assert response["trading_mode"] == "spot"
    assert response["strategy_version"] is None
    assert not response["trailing_stop"]
    assert "entry_pricing" in response
    assert "exit_pricing" in response
    assert "unfilledtimeout" in response
    assert "version" in response
    assert "api_version" in response
    assert 2.1 <= response["api_version"] < 3.0


def test_api_daily(botclient, mocker, ticker, fee, markets):
    """测试API每日统计"""
    ftbot, client = botclient

    ftbot.config["dry_run"] = False
    mocker.patch(f"{EXMS}.get_balances", return_value=ticker)
    mocker.patch(f"{EXMS}.get_tickers", ticker)
    mocker.patch(f"{EXMS}.get_fee", fee)
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets))
    ftbot.wallets.update()

    rc = client_get(client, f"{BASE_URI}/daily")
    assert_response(rc)
    response = rc.json()
    assert "data" in response
    assert len(response["data"]) == 7
    assert response["stake_currency"] == "BTC"
    assert response["fiat_display_currency"] == "USD"
    assert response["data"][0]["date"] == str(datetime.now(timezone.utc).date())


def test_api_weekly(botclient, mocker, ticker, fee, markets, time_machine):
    """测试API每周统计"""
    ftbot, client = botclient
    patch_get_signal(ftbot)
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets),
    )
    time_machine.move_to("2023-03-31 21:45:05 +00:00")
    rc = client_get(client, f"{BASE_URI}/weekly")
    assert_response(rc)
    assert len(rc.json()["data"]) == 4
    assert rc.json()["stake_currency"] == "BTC"
    assert rc.json()["fiat_display_currency"] == "USD"
    # 移动到周一
    assert rc.json()["data"][0]["date"] == "2023-03-27"
    assert rc.json()["data"][1]["date"] == "2023-03-20"


def test_api_monthly(botclient, mocker, ticker, fee, markets, time_machine):
    """测试API每月统计"""
    ftbot, client = botclient
    patch_get_signal(ftbot)
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets),
    )
    time_machine.move_to("2023-03-31 21:45:05 +00:00")
    rc = client_get(client, f"{BASE_URI}/monthly")
    assert_response(rc)
    assert len(rc.json()["data"]) == 3
    assert rc.json()["stake_currency"] == "BTC"
    assert rc.json()["fiat_display_currency"] == "USD"
    assert rc.json()["data"][0]["date"] == "2023-03-01"
    assert rc.json()["data"][1]["date"] == "2023-02-01"


@pytest.mark.parametrize("is_short", [True, False])
def test_api_trades(botclient, mocker, fee, markets, is_short):
    """测试API交易"""
    ftbot, client = botclient
    patch_get_signal(ftbot)
    mocker.patch.multiple(EXMS, markets=PropertyMock(return_value=markets))
    rc = client_get(client, f"{BASE_URI}/trades")
    assert_response(rc)
    assert len(rc.json()) == 4
    assert rc.json()["trades_count"] == 0
    assert rc.json()["total_trades"] == 0
    assert rc.json()["offset"] == 0

    create_mock_trades(fee, is_short=is_short)
    Trade.session.flush()

    rc = client_get(client, f"{BASE_URI}/trades")
    assert_response(rc)
    assert len(rc.json()["trades"]) == 2
    assert rc.json()["trades_count"] == 2
    assert rc.json()["total_trades"] == 2
    assert rc.json()["trades"][0]["is_short"] == is_short
    # 确保交易按trade_id排序（默认，见下文）
    assert rc.json()["trades"][0]["trade_id"] == 2
    assert rc.json()["trades"][1]["trade_id"] == 3

    rc = client_get(client, f"{BASE_URI}/trades?limit=1")
    assert_response(rc)
    assert len(rc.json()["trades"]) == 1
    assert rc.json()["trades_count"] == 1
    assert rc.json()["total_trades"] == 2

    # 测试升序排列（默认）
    rc = client_get(client, f"{BASE_URI}/trades?order_by_id=true")
    assert_response(rc)
    assert rc.json()["trades"][0]["trade_id"] == 2
    assert rc.json()["trades"][1]["trade_id"] == 3

    # 测试降序排列
    rc = client_get(client, f"{BASE_URI}/trades?order_by_id=false")
    assert_response(rc)
    assert rc.json()["trades"][0]["trade_id"] == 3
    assert rc.json()["trades"][1]["trade_id"] == 2


@pytest.mark.parametrize("is_short", [True, False])
def test_api_trade_single(botclient, mocker, fee, ticker, markets, is_short):
    """测试API单个交易"""
    ftbot, client = botclient
    patch_get_signal(ftbot, enter_long=not is_short, enter_short=is_short)
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        fetch_ticker=ticker,
    )
    rc = client_get(client, f"{BASE_URI}/trade/3")
    assert_response(rc, 404)
    assert rc.json()["detail"] == "未找到交易。"

    Trade.rollback()
    create_mock_trades(fee, is_short=is_short)

    rc = client_get(client, f"{BASE_URI}/trade/3")
    assert_response(rc)
    assert rc.json()["trade_id"] == 3
    assert rc.json()["is_short"] == is_short


@pytest.mark.usefixtures("init_persistence")
def test_api_custom_data_single_trade(botclient, fee):
    """测试API单个交易自定义数据"""
    Trade.reset_trades()
    CustomDataWrapper.reset_custom_data()

    create_mock_trades_usdt(fee, use_db=True)

    trade1 = Trade.get_trades_proxy()[0]

    assert trade1.get_all_custom_data() == []

    trade1.set_custom_data("test_str", "test_value")
    trade1.set_custom_data("test_int", 0)
    trade1.set_custom_data("test_float", 1.54)
    trade1.set_custom_data("test_bool", True)
    trade1.set_custom_data("test_dict", {"test": "vl"})

    trade1.set_custom_data("test_int", 1)

    _, client = botclient

    # 情况1：检查交易1的所有自定义数据
    rc = client_get(client, f"{BASE_URI}/trades/1/custom-data")
    assert_response(rc)

    # 验证响应JSON结构
    response_json = rc.json()

    assert len(response_json) == 1

    res_cust_data = response_json[0]["custom_data"]
    expected_data_td_1 = [
        {"key": "test_str", "type": "str", "value": "test_value"},
        {"key": "test_int", "type": "int", "value": 1},
        {"key": "test_float", "type": "float", "value": 1.54},
        {"key": "test_bool", "type": "bool", "value": True},
        {"key": "test_dict", "type": "dict", "value": {"test": "vl"}},
    ]

    # 确保响应包含确切预期的条目数
    assert len(res_cust_data) == len(expected_data_td_1), (
        f"期望 {len(expected_data_td_1)} 个条目，但得到 {len(res_cust_data)} 个。\n"
    )

    # 验证每个预期条目
    for expected in expected_data_td_1:
        matched_item = None
        for item in res_cust_data:
            if item["key"] == expected["key"]:
                matched_item = item
                break

        assert matched_item is not None, (
            f"缺少键 '{expected['key']}' 的预期条目\n期望：{expected}\n"
        )

        # 验证各个字段并仅打印不正确的值
        mismatches = []
        for field in ["key", "type", "value"]:
            if matched_item[field] != expected[field]:
                mismatches.append(f"{field}: 期望 {expected[field]}, 得到 {matched_item[field]}")

        assert not mismatches, f"条目 '{expected['key']}' 中的错误：\n" + "\n".join(mismatches)

    # 情况2：检查交易1的特定存在键自定义数据
    rc = client_get(client, f"{BASE_URI}/trades/1/custom-data?key=test_dict")
    assert_response(rc, 200)

    # 情况3：检查交易1的特定不存在键自定义数据
    rc = client_get(client, f"{BASE_URI}/trades/1/custom-data&key=test")
    assert_response(rc, 404)

    # 情况4：尝试从不存在的交易获取自定义数据
    rc = client_get(client, f"{BASE_URI}/trades/13/custom-data")
    assert_response(rc, 404)
    assert rc.json()["detail"] == "未找到交易ID为 13 的交易"


@pytest.mark.usefixtures("init_persistence")
def test_api_custom_data_multiple_open_trades(botclient, fee):
    """测试API多个开放交易自定义数据"""
    use_db = True
    Trade.use_db = use_db
    Trade.reset_trades()
    CustomDataWrapper.reset_custom_data()
    create_mock_trades(fee, False, use_db)
    trades = Trade.get_trades_proxy()
    assert len(trades) == 6

    assert isinstance(trades[0], Trade)

    trades = Trade.get_trades_proxy(is_open=True)
    assert len(trades) == 4

    create_mock_trades_usdt(fee, use_db=True)

    trade1 = Trade.get_trades_proxy(is_open=True)[0]
    trade2 = Trade.get_trades_proxy(is_open=True)[1]

    # 最初，不应该有自定义数据。
    assert trade1.get_all_custom_data() == []
    assert trade2.get_all_custom_data() == []

    # 为两个开放交易设置自定义数据。
    trade1.set_custom_data("test_str", "test_value_t1")
    trade1.set_custom_data("test_float", 1.54)
    trade1.set_custom_data("test_dict", {"test_t1": "vl_t1"})

    trade2.set_custom_data("test_str", "test_value_t2")
    trade2.set_custom_data("test_float", 1.55)
    trade2.set_custom_data("test_dict", {"test_t2": "vl_t2"})

    _, client = botclient

    # 情况1：检查两个交易的所有自定义数据。
    rc = client_get(client, f"{BASE_URI}/trades/open/custom-data")
    assert_response(rc)

    response_json = rc.json()

    # 期望响应中有两个交易条目
    assert len(response_json) == 2, f"期望2个交易条目，但得到 {len(response_json)} 个。\n"

    # 为每个交易定义预期的自定义数据。
    # 键现在使用来自自定义数据的实际trade_ids。
    expected_custom_data = {
        1: [
            {
                "key": "test_str",
                "type": "str",
                "value": "test_value_t1",
            },
            {
                "key": "test_float",
                "type": "float",
                "value": 1.54,
            },
            {
                "key": "test_dict",
                "type": "dict",
                "value": {"test_t1": "vl_t1"},
            },
        ],
        4: [
            {
                "key": "test_str",
                "type": "str",
                "value": "test_value_t2",
            },
            {
                "key": "test_float",
                "type": "float",
                "value": 1.55,
            },
            {
                "key": "test_dict",
                "type": "dict",
                "value": {"test_t2": "vl_t2"},
            },
        ],
    }

    # 遍历响应中每个交易的数据并验证条目。
    for trade_entry in response_json:
        trade_id = trade_entry.get("trade_id")
        assert trade_id in expected_custom_data, f"\n意外的trade_id: {trade_id}"

        custom_data_list = trade_entry.get("custom_data")
        expected_data = expected_custom_data[trade_id]
        assert len(custom_data_list) == len(expected_data), (
            f"trade_id {trade_id} 错误："
            f"期望 {len(expected_data)} 个条目，但得到 {len(custom_data_list)} 个。\n"
        )

        # 对于每个预期条目，检查响应是否包含正确的条目。
        for expected in expected_data:
            matched_item = None
            for item in custom_data_list:
                if item["key"] == expected["key"]:
                    matched_item = item
                    break

            assert matched_item is not None, (
                f"对于trade_id {trade_id}，"
                f"缺少键 '{expected['key']}' 的预期条目\n"
                f"期望：{expected}\n"
            )

            # 验证关键字段。
            mismatches = []
            for field in ["key", "type", "value"]:
                if matched_item[field] != expected[field]:
                    mismatches.append(
                        f"{field}: 期望 {expected[field]}, 得到 {matched_item[field]}"
                    )
            # 检查created_at和updated_at字段的存在而不比较值。
            for field in ["created_at", "updated_at"]:
                if field not in matched_item:
                    mismatches.append(f"缺少字段: {field}")

            assert not mismatches, (
                f"trade_id {trade_id} 的条目 '{expected['key']}' 中的错误：\n"
                + "\n".join(mismatches)
            )


@pytest.mark.parametrize("is_short", [True, False])
def test_api_delete_trade(botclient, mocker, fee, markets, is_short):
    """测试API删除交易"""
    ftbot, client = botclient
    patch_get_signal(ftbot, enter_long=not is_short, enter_short=is_short)
    stoploss_mock = MagicMock()
    cancel_mock = MagicMock()
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        cancel_order=cancel_mock,
        cancel_stoploss_order=stoploss_mock,
    )

    create_mock_trades(fee, is_short=is_short)

    ftbot.strategy.order_types["stoploss_on_exchange"] = True
    trades = Trade.session.scalars(select(Trade)).all()
    Trade.commit()
    assert len(trades) > 2

    rc = client_delete(client, f"{BASE_URI}/trades/1")
    assert_response(rc)
    assert rc.json()["result_msg"] == "已删除交易1。关闭了1个开放订单。"
    assert len(trades) - 1 == len(Trade.session.scalars(select(Trade)).all())
    assert cancel_mock.call_count == 1

    cancel_mock.reset_mock()
    rc = client_delete(client, f"{BASE_URI}/trades/1")
    # 交易现在已经消失了。
    assert_response(rc, 502)
    assert cancel_mock.call_count == 0

    assert len(trades) - 1 == len(Trade.session.scalars(select(Trade)).all())
    rc = client_delete(client, f"{BASE_URI}/trades/5")
    assert_response(rc)
    assert rc.json()["result_msg"] == "已删除交易5。关闭了1个开放订单。"
    assert len(trades) - 2 == len(Trade.session.scalars(select(Trade)).all())
    assert stoploss_mock.call_count == 1

    rc = client_delete(client, f"{BASE_URI}/trades/502")
    # 错误 - 交易不存在。
    assert_response(rc, 502)


@pytest.mark.parametrize("is_short", [True, False])
def test_api_delete_open_order(botclient, mocker, fee, markets, ticker, is_short):
    """测试API删除开放订单"""
    ftbot, client = botclient
    patch_get_signal(ftbot, enter_long=not is_short, enter_short=is_short)
    stoploss_mock = MagicMock()
    cancel_mock = MagicMock()
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        fetch_ticker=ticker,
        cancel_order=cancel_mock,
        cancel_stoploss_order=stoploss_mock,
    )

    rc = client_delete(client, f"{BASE_URI}/trades/10/open-order")
    assert_response(rc, 502)
    assert "无效的trade_id。" in rc.json()["error"]

    create_mock_trades(fee, is_short=is_short)
    Trade.commit()

    rc = client_delete(client, f"{BASE_URI}/trades/5/open-order")
    assert_response(rc, 502)
    assert "trade_id没有开放订单" in rc.json()["error"]
    trade = Trade.get_trades([Trade.id == 6]).first()
    mocker.patch(f"{EXMS}.fetch_order", side_effect=ExchangeError)
    rc = client_delete(client, f"{BASE_URI}/trades/6/open-order")
    assert_response(rc, 502)
    assert "未找到订单。" in rc.json()["error"]

    trade = Trade.get_trades([Trade.id == 6]).first()
    mocker.patch(f"{EXMS}.fetch_order", return_value=trade.orders[-1].to_ccxt_object())

    rc = client_delete(client, f"{BASE_URI}/trades/6/open-order")
    assert_response(rc)
    assert cancel_mock.call_count == 1


@pytest.mark.parametrize("is_short", [True, False])
def test_api_trade_reload_trade(botclient, mocker, fee, markets, ticker, is_short):
    """测试API交易重载交易"""
    ftbot, client = botclient
    patch_get_signal(ftbot, enter_long=not is_short, enter_short=is_short)
    stoploss_mock = MagicMock()
    cancel_mock = MagicMock()
    ftbot.handle_onexchange_order = MagicMock()
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        fetch_ticker=ticker,
        cancel_order=cancel_mock,
        cancel_stoploss_order=stoploss_mock,
    )

    rc = client_post(client, f"{BASE_URI}/trades/10/reload")
    assert_response(rc, 502)
    assert "找不到ID为10的交易。" in rc.json()["error"]
    assert ftbot.handle_onexchange_order.call_count == 0

    create_mock_trades(fee, is_short=is_short)
    Trade.commit()

    rc = client_post(client, f"{BASE_URI}/trades/5/reload")
    assert ftbot.handle_onexchange_order.call_count == 1


def test_api_logs(botclient):
    """测试API日志"""
    _ftbot, client = botclient
    rc = client_get(client, f"{BASE_URI}/logs")
    assert_response(rc)
    assert len(rc.json()) == 2
    assert "logs" in rc.json()
    # 在这里使用固定比较会使此测试失败！
    assert rc.json()["log_count"] > 1
    assert len(rc.json()["logs"]) == rc.json()["log_count"]

    assert isinstance(rc.json()["logs"][0], list)
    # 日期
    assert isinstance(rc.json()["logs"][0][0], str)
    # created_timestamp
    assert isinstance(rc.json()["logs"][0][1], float)
    assert isinstance(rc.json()["logs"][0][2], str)
    assert isinstance(rc.json()["logs"][0][3], str)
    assert isinstance(rc.json()["logs"][0][4], str)

    rc1 = client_get(client, f"{BASE_URI}/logs?limit=5")
    assert_response(rc1)
    assert len(rc1.json()) == 2
    assert "logs" in rc1.json()
    # 在这里使用固定比较会使此测试失败！
    if rc1.json()["log_count"] < 5:
        # 帮助调试随机测试失败
        print(f"rc={rc.json()}")
        print(f"rc1={rc1.json()}")
    assert rc1.json()["log_count"] > 2
    assert len(rc1.json()["logs"]) == rc1.json()["log_count"]


@pytest.mark.parametrize(
    "is_short,expected",
    [
        (
            True,
            {
                "best_pair": "XRP/BTC",
                "best_rate": -0.02,
                "best_pair_profit_ratio": -0.00018780487,
                "best_pair_profit_abs": -0.001155,
                "profit_all_coin": 15.382312,
                "profit_all_fiat": 189894.6470718,
                "profit_all_percent_mean": 49.62,
                "profit_all_ratio_mean": 0.49620917,
                "profit_all_percent_sum": 198.48,
                "profit_all_ratio_sum": 1.98483671,
                "profit_all_percent": 1.54,
                "profit_all_ratio": 0.01538214,
                "profit_closed_coin": -0.00673913,
                "profit_closed_fiat": -83.19455985,
                "profit_closed_ratio_mean": -0.0075,
                "profit_closed_percent_mean": -0.75,
                "profit_closed_ratio_sum": -0.015,
                "profit_closed_percent_sum": -1.5,
                "profit_closed_ratio": -6.739057628404269e-06,
                "profit_closed_percent": -0.0,
                "winning_trades": 0,
                "losing_trades": 2,
                "profit_factor": 0.0,
                "winrate": 0.0,
                "expectancy": -0.0033695635,
                "expectancy_ratio": -1.0,
                "trading_volume": 75.945,
            },
        ),
        (
            False,
            {
                "best_pair": "ETC/BTC",
                "best_rate": 0.0,
                "best_pair_profit_ratio": 0.00003860975,
                "best_pair_profit_abs": 0.000584127,
                "profit_all_coin": -15.46546305,
                "profit_all_fiat": -190921.14135225,
                "profit_all_percent_mean": -49.62,
                "profit_all_ratio_mean": -0.49620955,
                "profit_all_percent_sum": -198.48,
                "profit_all_ratio_sum": -1.9848382,
                "profit_all_percent": -1.55,
                "profit_all_ratio": -0.0154654126,
                "profit_closed_coin": 0.00073913,
                "profit_closed_fiat": 9.124559849999999,
                "profit_closed_ratio_mean": 0.0075,
                "profit_closed_percent_mean": 0.75,
                "profit_closed_ratio_sum": 0.015,
                "profit_closed_percent_sum": 1.5,
                "profit_closed_ratio": 7.391275897987988e-07,
                "profit_closed_percent": 0.0,
                "winning_trades": 2,
                "losing_trades": 0,
                "profit_factor": None,
                "winrate": 1.0,
                "expectancy": 0.0003695635,
                "expectancy_ratio": 100,
                "trading_volume": 75.945,
            },
        ),
        (
            None,
            {
                "best_pair": "XRP/BTC",
                "best_rate": 0.0,
                "best_pair_profit_ratio": 0.000025203252,
                "best_pair_profit_abs": 0.000155,
                "profit_all_coin": -14.87167525,
                "profit_all_fiat": -183590.83096125,
                "profit_all_percent_mean": 0.13,
                "profit_all_ratio_mean": 0.0012538324,
                "profit_all_percent_sum": 0.5,
                "profit_all_ratio_sum": 0.005015329,
                "profit_all_percent": -1.49,
                "profit_all_ratio": -0.0148715350,
                "profit_closed_coin": -0.00542913,
                "profit_closed_fiat": -67.02260985,
                "profit_closed_ratio_mean": 0.0025,
                "profit_closed_percent_mean": 0.25,
                "profit_closed_ratio_sum": 0.005,
                "profit_closed_percent_sum": 0.5,
                "profit_closed_ratio": -5.429078808526421e-06,
                "profit_closed_percent": -0.0,
                "winning_trades": 1,
                "losing_trades": 1,
                "profit_factor": 0.02775724835771106,
                "winrate": 0.5,
                "expectancy": -0.0027145635000000003,
                "expectancy_ratio": -0.48612137582114445,
                "trading_volume": 75.945,
            },
        ),
    ],
)
def test_api_profit(botclient, mocker, ticker, fee, markets, is_short, expected):
    """测试API利润"""
    ftbot, client = botclient
    ftbot.config["tradable_balance_ratio"] = 1
    patch_get_signal(ftbot)
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets),
    )

    rc = client_get(client, f"{BASE_URI}/profit")
    assert_response(rc, 200)
    assert rc.json()["trade_count"] == 0

    create_mock_trades(fee, is_short=is_short)
    # 模拟已完成的LIMIT_BUY订单进行交易

    rc = client_get(client, f"{BASE_URI}/profit")
    assert_response(rc)
    # raise ValueError(rc.json())
    assert rc.json() == {
        "avg_duration": ANY,
        "best_pair": expected["best_pair"],
        "best_pair_profit_ratio": pytest.approx(expected["best_pair_profit_ratio"]),
        "best_pair_profit_abs": expected["best_pair_profit_abs"],
        "best_rate": expected["best_rate"],
        "first_trade_date": ANY,
        "first_trade_humanized": ANY,
        "first_trade_timestamp": ANY,
        "latest_trade_date": ANY,
        "latest_trade_humanized": "5分钟前",
        "latest_trade_timestamp": ANY,
        "profit_all_coin": pytest.approx(expected["profit_all_coin"]),
        "profit_all_fiat": pytest.approx(expected["profit_all_fiat"]),
        "profit_all_percent_mean": pytest.approx(expected["profit_all_percent_mean"]),
        "profit_all_ratio_mean": pytest.approx(expected["profit_all_ratio_mean"]),
        "profit_all_percent_sum": pytest.approx(expected["profit_all_percent_sum"]),
        "profit_all_ratio_sum": pytest.approx(expected["profit_all_ratio_sum"]),
        "profit_all_percent": pytest.approx(expected["profit_all_percent"]),
        "profit_all_ratio": pytest.approx(expected["profit_all_ratio"]),
        "profit_closed_coin": pytest.approx(expected["profit_closed_coin"]),
        "profit_closed_fiat": pytest.approx(expected["profit_closed_fiat"]),
        "profit_closed_ratio_mean": pytest.approx(expected["profit_closed_ratio_mean"]),
        "profit_closed_percent_mean": pytest.approx(expected["profit_closed_percent_mean"]),
        "profit_closed_ratio_sum": pytest.approx(expected["profit_closed_ratio_sum"]),
        "profit_closed_percent_sum": pytest.approx(expected["profit_closed_percent_sum"]),
        "profit_closed_ratio": pytest.approx(expected["profit_closed_ratio"]),
        "profit_closed_percent": pytest.approx(expected["profit_closed_percent"]),
        "trade_count": 6,
        "closed_trade_count": 2,
        "winning_trades": expected["winning_trades"],
        "losing_trades": expected["losing_trades"],
        "profit_factor": expected["profit_factor"],
        "winrate": expected["winrate"],
        "expectancy": expected["expectancy"],
        "expectancy_ratio": expected["expectancy_ratio"],
        "max_drawdown": ANY,
        "max_drawdown_abs": ANY,
        "max_drawdown_start": ANY,
        "max_drawdown_start_timestamp": ANY,
        "max_drawdown_end": ANY,
        "max_drawdown_end_timestamp": ANY,
        "trading_volume": expected["trading_volume"],
        "bot_start_timestamp": 0,
        "bot_start_date": "",
    }


@pytest.mark.parametrize("is_short", [True, False])
def test_api_stats(botclient, mocker, ticker, fee, markets, is_short):
    """测试API统计"""
    ftbot, client = botclient
    patch_get_signal(ftbot, enter_long=not is_short, enter_short=is_short)
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets),
    )

    rc = client_get(client, f"{BASE_URI}/stats")
    assert_response(rc, 200)
    assert "durations" in rc.json()
    assert "exit_reasons" in rc.json()

    create_mock_trades(fee, is_short=is_short)

    rc = client_get(client, f"{BASE_URI}/stats")
    assert_response(rc, 200)
    assert "durations" in rc.json()
    assert "exit_reasons" in rc.json()

    assert "wins" in rc.json()["durations"]
    assert "losses" in rc.json()["durations"]
    assert "draws" in rc.json()["durations"]


def test_api_performance(botclient, fee):
    """测试API性能"""
    ftbot, client = botclient
    patch_get_signal(ftbot)

    create_mock_trades_usdt(fee)

    rc = client_get(client, f"{BASE_URI}/performance")
    assert_response(rc)
    assert len(rc.json()) == 3
    assert rc.json() == [
        {
            "count": 1,
            "pair": "NEO/USDT",
            "profit": 1.99,
            "profit_pct": 1.99,
            "profit_ratio": 0.0199375,
            "profit_abs": 3.9875,
        },
        {
            "count": 1,
            "pair": "XRP/USDT",
            "profit": 9.47,
            "profit_abs": 2.8425,
            "profit_pct": 9.47,
            "profit_ratio": pytest.approx(0.094749999),
        },
        {
            "count": 1,
            "pair": "LTC/USDT",
            "profit": -20.45,
            "profit_abs": -4.09,
            "profit_pct": -20.45,
            "profit_ratio": -0.2045,
        },
    ]


def test_api_entries(botclient, fee):
    """测试API入场"""
    ftbot, client = botclient
    patch_get_signal(ftbot)
    # 空的
    rc = client_get(client, f"{BASE_URI}/entries")
    assert_response(rc)
    assert len(rc.json()) == 0

    create_mock_trades(fee)
    rc = client_get(client, f"{BASE_URI}/entries")
    assert_response(rc)
    response = rc.json()
    assert len(response) == 2
    resp = response[0]
    assert resp["enter_tag"] == "TEST1"
    assert resp["count"] == 1
    assert resp["profit_pct"] == 0.0
    assert pytest.approx(resp["profit_ratio"]) == 0.000038609756


def test_api_exits(botclient, fee):
    """测试API出场"""
    ftbot, client = botclient
    patch_get_signal(ftbot)
    # 空的
    rc = client_get(client, f"{BASE_URI}/exits")
    assert_response(rc)
    assert len(rc.json()) == 0

    create_mock_trades(fee)
    rc = client_get(client, f"{BASE_URI}/exits")
    assert_response(rc)
    response = rc.json()
    assert len(response) == 2
    resp = response[0]
    assert resp["exit_reason"] == "sell_signal"
    assert resp["count"] == 1
    assert resp["profit_pct"] == 0.0
    assert pytest.approx(resp["profit_ratio"]) == 0.000038609756


def test_api_mix_tag(botclient, fee):
    """测试API混合标签"""
    ftbot, client = botclient
    patch_get_signal(ftbot)
    # 空的
    rc = client_get(client, f"{BASE_URI}/mix_tags")
    assert_response(rc)
    assert len(rc.json()) == 0

    create_mock_trades(fee)
    rc = client_get(client, f"{BASE_URI}/mix_tags")
    assert_response(rc)
    response = rc.json()
    assert len(response) == 2
    resp = response[0]
    assert resp["mix_tag"] == "TEST1 sell_signal"
    assert resp["count"] == 1
    assert resp["profit_pct"] == 0.5


@pytest.mark.parametrize(
    "is_short,current_rate,open_trade_value",
    [(True, 1.098e-05, 6.134625), (False, 1.099e-05, 6.165375)],
)
def test_api_status(
    botclient, mocker, ticker, fee, markets, is_short, current_rate, open_trade_value
):
    """测试API状态"""
    ftbot, client = botclient
    patch_get_signal(ftbot)
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets),
        fetch_order=MagicMock(return_value={}),
    )

    rc = client_get(client, f"{BASE_URI}/status")
    assert_response(rc, 200)
    assert rc.json() == []
    create_mock_trades(fee, is_short=is_short)

    rc = client_get(client, f"{BASE_URI}/status")
    assert_response(rc)
    assert len(rc.json()) == 4
    assert rc.json()[0] == {
        "amount": 50.0,
        "amount_requested": 123.0,
        "close_date": None,
        "close_timestamp": None,
        "close_profit": None,
        "close_profit_pct": None,
        "close_profit_abs": None,
        "close_rate": None,
        "profit_ratio": ANY,
        "profit_pct": ANY,
        "profit_abs": ANY,
        "profit_fiat": ANY,
        "total_profit_abs": ANY,
        "total_profit_fiat": ANY,
        "total_profit_ratio": ANY,
        "realized_profit": 0.0,
        "realized_profit_ratio": None,
        "current_rate": current_rate,
        "open_date": ANY,
        "open_timestamp": ANY,
        "open_fill_date": ANY,
        "open_fill_timestamp": ANY,
        "open_rate": 0.123,
        "pair": "ETH/BTC",
        "base_currency": "ETH",
        "quote_currency": "BTC",
        "stake_amount": 0.001,
        "max_stake_amount": ANY,
        "stop_loss_abs": ANY,
        "stop_loss_pct": ANY,
        "stop_loss_ratio": ANY,
        "stoploss_last_update": ANY,
        "stoploss_last_update_timestamp": ANY,
        "initial_stop_loss_abs": 0.0,
        "initial_stop_loss_pct": ANY,
        "initial_stop_loss_ratio": ANY,
        "stoploss_current_dist": ANY,
        "stoploss_current_dist_ratio": ANY,
        "stoploss_current_dist_pct": ANY,
        "stoploss_entry_dist": ANY,
        "stoploss_entry_dist_ratio": ANY,
        "trade_id": 1,
        "close_rate_requested": ANY,
        "fee_close": 0.0025,
        "fee_close_cost": None,
        "fee_close_currency": None,
        "fee_open": 0.0025,
        "fee_open_cost": None,
        "fee_open_currency": None,
        "is_open": True,
        "is_short": is_short,
        "max_rate": ANY,
        "min_rate": ANY,
        "open_rate_requested": ANY,
        "open_trade_value": open_trade_value,
        "exit_reason": None,
        "exit_order_status": None,
        "strategy": CURRENT_TEST_STRATEGY,
        "enter_tag": None,
        "timeframe": 5,
        "exchange": "binance",
        "leverage": 1.0,
        "interest_rate": 0.0,
        "liquidation_price": None,
        "funding_fees": None,
        "trading_mode": ANY,
        "amount_precision": None,
        "price_precision": None,
        "precision_mode": None,
        "orders": [ANY],
        "has_open_orders": True,
    }

    mocker.patch(
        f"{EXMS}.get_rate", MagicMock(side_effect=ExchangeError("交易对 'ETH/BTC' 不可用"))
    )

    rc = client_get(client, f"{BASE_URI}/status")
    assert_response(rc)
    resp_values = rc.json()
    assert len(resp_values) == 4
    assert resp_values[0]["profit_abs"] == 0.0


def test_api_version(botclient):
    """测试API版本"""
    _ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/version")
    assert_response(rc)
    assert rc.json() == {"version": __version__}


def test_api_blacklist(botclient, mocker):
    """测试API黑名单"""
    _ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/blacklist")
    assert_response(rc)
    # DOGE和HOT不在markets模拟中！
    assert rc.json() == {
        "blacklist": ["DOGE/BTC", "HOT/BTC"],
        "blacklist_expanded": [],
        "length": 2,
        "method": ["StaticPairList"],
        "errors": {},
    }

    # 将ETH/BTC添加到黑名单
    rc = client_post(client, f"{BASE_URI}/blacklist", data={"blacklist": ["ETH/BTC"]})
    assert_response(rc)
    assert rc.json() == {
        "blacklist": ["DOGE/BTC", "HOT/BTC", "ETH/BTC"],
        "blacklist_expanded": ["ETH/BTC"],
        "length": 3,
        "method": ["StaticPairList"],
        "errors": {},
    }

    rc = client_post(client, f"{BASE_URI}/blacklist", data={"blacklist": ["XRP/.*"]})
    assert_response(rc)
    assert rc.json() == {
        "blacklist": ["DOGE/BTC", "HOT/BTC", "ETH/BTC", "XRP/.*"],
        "blacklist_expanded": ["ETH/BTC", "XRP/BTC", "XRP/USDT"],
        "length": 4,
        "method": ["StaticPairList"],
        "errors": {},
    }

    rc = client_delete(client, f"{BASE_URI}/blacklist?pairs_to_delete=DOGE/BTC")
    assert_response(rc)
    assert rc.json() == {
        "blacklist": ["HOT/BTC", "ETH/BTC", "XRP/.*"],
        "blacklist_expanded": ["ETH/BTC", "XRP/BTC", "XRP/USDT"],
        "length": 3,
        "method": ["StaticPairList"],
        "errors": {},
    }

    rc = client_delete(client, f"{BASE_URI}/blacklist?pairs_to_delete=NOTHING/BTC")
    assert_response(rc)
    assert rc.json() == {
        "blacklist": ["HOT/BTC", "ETH/BTC", "XRP/.*"],
        "blacklist_expanded": ["ETH/BTC", "XRP/BTC", "XRP/USDT"],
        "length": 3,
        "method": ["StaticPairList"],
        "errors": {
            "NOTHING/BTC": {"error_msg": "交易对NOTHING/BTC不在当前黑名单中。"}
        },
    }
    rc = client_delete(
        client, f"{BASE_URI}/blacklist?pairs_to_delete=HOT/BTC&pairs_to_delete=ETH/BTC"
    )
    assert_response(rc)
    assert rc.json() == {
        "blacklist": ["XRP/.*"],
        "blacklist_expanded": ["XRP/BTC", "XRP/USDT"],
        "length": 1,
        "method": ["StaticPairList"],
        "errors": {},
    }


def test_api_whitelist(botclient):
    """测试API白名单"""
    _ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/whitelist")
    assert_response(rc)
    assert rc.json() == {
        "whitelist": ["ETH/BTC", "LTC/BTC", "XRP/BTC", "NEO/BTC"],
        "length": 4,
        "method": ["StaticPairList"],
    }


@pytest.mark.parametrize(
    "endpoint",
    [
        "forcebuy",
        "forceenter",
    ],
)
def test_api_force_entry(botclient, mocker, fee, endpoint):
    """测试API强制入场"""
    ftbot, client = botclient

    rc = client_post(client, f"{BASE_URI}/{endpoint}", data={"pair": "ETH/BTC"})
    assert_response(rc, 502)
    assert rc.json() == {"error": f"查询 /api/v1/{endpoint} 时出错：未启用强制入场。"}

    # 启用强制买入
    ftbot.config["force_entry_enable"] = True

    fbuy_mock = MagicMock(return_value=None)
    mocker.patch("freqtrade.rpc.rpc.RPC._rpc_force_entry", fbuy_mock)
    rc = client_post(client, f"{BASE_URI}/{endpoint}", data={"pair": "ETH/BTC"})
    assert_response(rc)
    assert rc.json() == {"status": "为交易对ETH/BTC进入多头交易出错。"}

    # 测试创建交易
    fbuy_mock = MagicMock(
        return_value=Trade(
            pair="ETH/BTC",
            amount=1,
            amount_requested=1,
            exchange="binance",
            stake_amount=1,
            open_rate=0.245441,
            open_date=datetime.now(timezone.utc),
            is_open=False,
            is_short=False,
            fee_close=fee.return_value,
            fee_open=fee.return_value,
            close_rate=0.265441,
            id=22,
            timeframe=5,
            strategy=CURRENT_TEST_STRATEGY,
            trading_mode=TradingMode.SPOT,
        )
    )
    mocker.patch("freqtrade.rpc.rpc.RPC._rpc_force_entry", fbuy_mock)

    rc = client_post(client, f"{BASE_URI}/{endpoint}", data={"pair": "ETH/BTC"})
    assert_response(rc)
    assert rc.json() == {
        "amount": 1.0,
        "amount_requested": 1.0,
        "trade_id": 22,
        "close_date": None,
        "close_timestamp": None,
        "close_rate": 0.265441,
        "open_date": ANY,
        "open_timestamp": ANY,
        "open_fill_date": ANY,
        "open_fill_timestamp": ANY,
        "open_rate": 0.245441,
        "pair": "ETH/BTC",
        "base_currency": "ETH",
        "quote_currency": "BTC",
        "stake_amount": 1,
        "max_stake_amount": ANY,
        "stop_loss_abs": None,
        "stop_loss_pct": None,
        "stop_loss_ratio": None,
        "stoploss_last_update": None,
        "stoploss_last_update_timestamp": None,
        "initial_stop_loss_abs": None,
        "initial_stop_loss_pct": None,
        "initial_stop_loss_ratio": None,
        "close_profit": None,
        "close_profit_pct": None,
        "close_profit_abs": None,
        "close_rate_requested": None,
        "profit_ratio": None,
        "profit_pct": None,
        "profit_abs": None,
        "profit_fiat": None,
        "realized_profit": 0.0,
        "realized_profit_ratio": None,
        "fee_close": 0.0025,
        "fee_close_cost": None,
        "fee_close_currency": None,
        "fee_open": 0.0025,
        "fee_open_cost": None,
        "fee_open_currency": None,
        "is_open": False,
        "is_short": False,
        "max_rate": None,
        "min_rate": None,
        "open_rate_requested": None,
        "open_trade_value": 0.24605460,
        "exit_reason": None,
        "exit_order_status": None,
        "strategy": CURRENT_TEST_STRATEGY,
        "enter_tag": None,
        "timeframe": 5,
        "exchange": "binance",
        "leverage": None,
        "interest_rate": None,
        "liquidation_price": None,
        "funding_fees": None,
        "trading_mode": "spot",
        "amount_precision": None,
        "price_precision": None,
        "precision_mode": None,
        "has_open_orders": False,
        "orders": [],
    }


def test_api_forceexit(botclient, mocker, ticker, fee, markets):
    """测试API强制出场"""
    ftbot, client = botclient
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets),
        _dry_is_price_crossed=MagicMock(return_value=True),
    )
    patch_get_signal(ftbot)

    rc = client_post(client, f"{BASE_URI}/forceexit", data={"tradeid": "1"})
    assert_response(rc, 502)
    assert rc.json() == {"error": "查询 /api/v1/forceexit 时出错：无效参数"}
    Trade.rollback()

    create_mock_trades(fee)
    trade = Trade.get_trades([Trade.id == 5]).first()
    assert pytest.approx(trade.amount) == 123
    rc = client_post(
        client, f"{BASE_URI}/forceexit", data={"tradeid": "5", "ordertype": "market", "amount": 23}
    )
    assert_response(rc)
    assert rc.json() == {"result": "为交易5创建了出场订单。"}
    Trade.rollback()

    trade = Trade.get_trades([Trade.id == 5]).first()
    assert pytest.approx(trade.amount) == 100
    assert trade.is_open is True

    rc = client_post(client, f"{BASE_URI}/forceexit", data={"tradeid": "5"})
    assert_response(rc)
    assert rc.json() == {"result": "为交易5创建了出场订单。"}
    Trade.rollback()

    trade = Trade.get_trades([Trade.id == 5]).first()
    assert trade.is_open is False


def test_api_pair_candles(botclient, ohlcv_history):
    """测试API交易对K线"""
    ftbot, client = botclient
    timeframe = "5m"
    amount = 3

    # 没有交易对
    rc = client_get(client, f"{BASE_URI}/pair_candles?limit={amount}&timeframe={timeframe}")
    assert_response(rc, 422)

    # 没有时间框架
    rc = client_get(client, f"{BASE_URI}/pair_candles?pair=XRP%2FBTC")
    assert_response(rc, 422)

    rc = client_get(
        client, f"{BASE_URI}/pair_candles?limit={amount}&pair=XRP%2FBTC&timeframe={timeframe}"
    )
    assert_response(rc)
    assert "columns" in rc.json()
    assert "data_start_ts" in rc.json()
    assert "data_start" in rc.json()
    assert "data_stop" in rc.json()
    assert "data_stop_ts" in rc.json()
    assert len(rc.json()["data"]) == 0
    ohlcv_history["sma"] = ohlcv_history["close"].rolling(2).mean()
    ohlcv_history["sma2"] = ohlcv_history["close"].rolling(2).mean()
    ohlcv_history["enter_long"] = 0
    ohlcv_history.loc[1, "enter_long"] = 1
    ohlcv_history["exit_long"] = 0
    ohlcv_history["enter_short"] = 0
    ohlcv_history["exit_short"] = 0

    ftbot.dataprovider._set_cached_df("XRP/BTC", timeframe, ohlcv_history, CandleType.SPOT)
    fake_plot_annotations = [
        {
            "type": "area",
            "start": "2024-01-01 15:00:00",
            "end": "2024-01-01 16:00:00",
            "y_start": 94000.2,
            "y_end": 98000,
            "color": "",
            "label": "某个标签",
        }
    ]
    plot_annotations_mock = MagicMock(return_value=fake_plot_annotations)
    ftbot.strategy.plot_annotations = plot_annotations_mock
    for call in ("get", "post"):
        plot_annotations_mock.reset_mock()
        if call == "get":
            rc = client_get(
                client,
                f"{BASE_URI}/pair_candles?limit={amount}&pair=XRP%2FBTC&timeframe={timeframe}",
            )
        else:
            rc = client_post(
                client,
                f"{BASE_URI}/pair_candles",
                data={
                    "pair": "XRP/BTC",
                    "timeframe": timeframe,
                    "limit": amount,
                    "columns": ["sma"],
                },
            )
        assert_response(rc)
        resp = rc.json()
        assert "strategy" in resp
        assert resp["strategy"] == CURRENT_TEST_STRATEGY
        assert "columns" in resp
        assert "data_start_ts" in resp
        assert "data_start" in resp
        assert "data_stop" in resp
        assert "data_stop_ts" in resp
        assert resp["data_start"] == "2017-11-26 08:50:00+00:00"
        assert resp["data_start_ts"] == 1511686200000
        assert resp["data_stop"] == "2017-11-26 09:00:00+00:00"
        assert resp["data_stop_ts"] == 1511686800000
        assert resp["annotations"] == fake_plot_annotations
        assert plot_annotations_mock.call_count == 1
        assert isinstance(resp["columns"], list)
        base_cols = {
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "sma",
            "enter_long",
            "exit_long",
            "enter_short",
            "exit_short",
            "__date_ts",
            "_enter_long_signal_close",
            "_exit_long_signal_close",
            "_enter_short_signal_close",
            "_exit_short_signal_close",
        }
        if call == "get":
            assert set(resp["columns"]) == base_cols.union({"sma2"})
        else:
            assert set(resp["columns"]) == base_cols

        # 所有列不包括内部列
        assert set(resp["all_columns"]) == {
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "sma",
            "sma2",
            "enter_long",
            "exit_long",
            "enter_short",
            "exit_short",
        }
        assert "pair" in resp
        assert resp["pair"] == "XRP/BTC"

        assert "data" in resp
        assert len(resp["data"]) == amount
        if call == "get":
            assert len(resp["data"][0]) == 17
            assert resp["data"] == [
                [
                    "2017-11-26T08:50:00Z",
                    8.794e-05,
                    8.948e-05,
                    8.794e-05,
                    8.88e-05,
                    0.0877869,
                    None,
                    None,
                    0,
                    0,
                    0,
                    0,
                    1511686200000,
                    None,
                    None,
                    None,
                    None,
                ],
                [
                    "2017-11-26T08:55:00Z",
                    8.88e-05,
                    8.942e-05,
                    8.88e-05,
                    8.893e-05,
                    0.05874751,
                    8.886500000000001e-05,
                    8.886500000000001e-05,
                    1,
                    0,
                    0,
                    0,
                    1511686500000,
                    8.893e-05,
                    None,
                    None,
                    None,
                ],
                [
                    "2017-11-26T09:00:00Z",
                    8.891e-05,
                    8.893e-05,
                    8.875e-05,
                    8.877e-05,
                    0.7039405,
                    8.885e-05,
                    8.885e-05,
                    0,
                    0,
                    0,
                    0,
                    1511686800000,
                    None,
                    None,
                    None,
                    None,
                ],
            ]
        else:
            assert len(resp["data"][0]) == 16
            assert resp["data"] == [
                [
                    "2017-11-26T08:50:00Z",
                    8.794e-05,
                    8.948e-05,
                    8.794e-05,
                    8.88e-05,
                    0.0877869,
                    None,
                    0,
                    0,
                    0,
                    0,
                    1511686200000,
                    None,
                    None,
                    None,
                    None,
                ],
                [
                    "2017-11-26T08:55:00Z",
                    8.88e-05,
                    8.942e-05,
                    8.88e-05,
                    8.893e-05,
                    0.05874751,
                    8.886500000000001e-05,
                    1,
                    0,
                    0,
                    0,
                    1511686500000,
                    8.893e-05,
                    None,
                    None,
                    None,
                ],
                [
                    "2017-11-26T09:00:00Z",
                    8.891e-05,
                    8.893e-05,
                    8.875e-05,
                    8.877e-05,
                    0.7039405,
                    8.885e-05,
                    0,
                    0,
                    0,
                    0,
                    1511686800000,
                    None,
                    None,
                    None,
                    None,
                ],
            ]

    # 为下一个测试做准备
    ohlcv_history["exit_long"] = ohlcv_history["exit_long"].astype("float64")
    ohlcv_history.at[0, "exit_long"] = float("inf")
    ohlcv_history["date1"] = ohlcv_history["date"]
    ohlcv_history.at[0, "date1"] = pd.NaT

    ftbot.dataprovider._set_cached_df("XRP/BTC", timeframe, ohlcv_history, CandleType.SPOT)
    rc = client_get(
        client, f"{BASE_URI}/pair_candles?limit={amount}&pair=XRP%2FBTC&timeframe={timeframe}"
    )
    assert_response(rc)
    assert rc.json()["data"] == [
        [
            "2017-11-26T08:50:00Z",
            8.794e-05,
            8.948e-05,
            8.794e-05,
            8.88e-05,
            0.0877869,
            None,
            None,
            0,
            None,
            0,
            0,
            None,
            1511686200000,
            None,
            None,
            None,
            None,
        ],
        [
            "2017-11-26T08:55:00Z",
            8.88e-05,
            8.942e-05,
            8.88e-05,
            8.893e-05,
            0.05874751,
            8.886500000000001e-05,
            8.886500000000001e-05,
            1,
            0.0,
            0,
            0,
            "2017-11-26T08:55:00Z",
            1511686500000,
            8.893e-05,
            None,
            None,
            None,
        ],
        [
            "2017-11-26T09:00:00Z",
            8.891e-05,
            8.893e-05,
            8.875e-05,
            8.877e-05,
            0.7039405,
            8.885e-05,
            8.885e-05,
            0,
            0.0,
            0,
            0,
            "2017-11-26T09:00:00Z",
            1511686800000,
            None,
            None,
            None,
            None,
        ],
    ]


def test_api_pair_history(botclient, tmp_path, mocker):
    """测试API交易对历史"""
    _ftbot, client = botclient
    _ftbot.config["user_data_dir"] = tmp_path

    timeframe = "5m"
    lfm = mocker.patch("freqtrade.strategy.interface.IStrategy.load_freqAI_model")
    # 错误模式
    rc = client_get(
        client,
        f"{BASE_URI}/pair_history?timeframe={timeframe}"
        f"&timerange=20180111-20180112&strategy={CURRENT_TEST_STRATEGY}",
    )
    assert_response(rc, 503)
    _ftbot.config["runmode"] = RunMode.WEBSERVER

    # 没有交易对
    rc = client_get(
        client,
        f"{BASE_URI}/pair_history?timeframe={timeframe}"
        f"&timerange=20180111-20180112&strategy={CURRENT_TEST_STRATEGY}",
    )
    assert_response(rc, 422)

    # 没有时间框架
    rc = client_get(
        client,
        f"{BASE_URI}/pair_history?pair=UNITTEST%2FBTC"
        f"&timerange=20180111-20180112&strategy={CURRENT_TEST_STRATEGY}",
    )
    assert_response(rc, 422)

    # 没有时间范围
    rc = client_get(
        client,
        f"{BASE_URI}/pair_history?pair=UNITTEST%2FBTC&timeframe={timeframe}"
        f"&strategy={CURRENT_TEST_STRATEGY}",
    )
    assert_response(rc, 422)

    # 没有策略
    rc = client_get(
        client,
        f"{BASE_URI}/pair_history?pair=UNITTEST%2FBTC&timeframe={timeframe}"
        "&timerange=20180111-20180112",
    )
    assert_response(rc, 422)

    # 无效策略
    rc = client_get(
        client,
        f"{BASE_URI}/pair_history?pair=UNITTEST%2FBTC&timeframe={timeframe}"
        "&timerange=20180111-20180112&strategy={CURRENT_TEST_STRATEGY}11",
    )
    assert_response(rc, 502)

    # 正常工作
    for call in ("get", "post"):
        if call == "get":
            rc = client_get(
                client,
                f"{BASE_URI}/pair_history?pair=UNITTEST%2FBTC&timeframe={timeframe}"
                f"&timerange=20180111-20180112&strategy={CURRENT_TEST_STRATEGY}",
            )
        else:
            rc = client_post(
                client,
                f"{BASE_URI}/pair_history",
                data={
                    "pair": "UNITTEST/BTC",
                    "timeframe": timeframe,
                    "timerange": "20180111-20180112",
                    "strategy": CURRENT_TEST_STRATEGY,
                    "columns": ["rsi", "fastd", "fastk"],
                },
            )

        assert_response(rc, 200)
        result = rc.json()
        assert result["length"] == 289
        assert len(result["data"]) == result["length"]
        assert "columns" in result
        assert "data" in result
        data = result["data"]
        assert len(data) == 289
        col_count = 30 if call == "get" else 18
        # 分析的DF有30列
        assert len(result["columns"]) == col_count
        assert len(result["all_columns"]) == 25
        assert len(data[0]) == col_count
        date_col_idx = next(idx for idx, c in enumerate(result["columns"]) if c == "date")
        rsi_col_idx = next(idx for idx, c in enumerate(result["columns"]) if c == "rsi")

        assert data[0][date_col_idx] == "2018-01-11T00:00:00Z"
        assert data[0][rsi_col_idx] is not None
        assert data[0][rsi_col_idx] > 0
        assert lfm.call_count == 1
        assert result["pair"] == "UNITTEST/BTC"
        assert result["strategy"] == CURRENT_TEST_STRATEGY
        assert result["data_start"] == "2018-01-11 00:00:00+00:00"
        assert result["data_start_ts"] == 1515628800000
        assert result["data_stop"] == "2018-01-12 00:00:00+00:00"
        assert result["data_stop_ts"] == 1515715200000
        assert result["annotations"] == []
        lfm.reset_mock()

        # 未找到数据
        if call == "get":
            rc = client_get(
                client,
                f"{BASE_URI}/pair_history?pair=UNITTEST%2FBTC&timeframe={timeframe}"
                f"&timerange=20200111-20200112&strategy={CURRENT_TEST_STRATEGY}",
            )
        else:
            rc = client_post(
                client,
                f"{BASE_URI}/pair_history",
                data={
                    "pair": "UNITTEST/BTC",
                    "timeframe": timeframe,
                    "timerange": "20200111-20200112",
                    "strategy": CURRENT_TEST_STRATEGY,
                    "columns": ["rsi", "fastd", "fastk"],
                },
            )
        assert_response(rc, 502)
        assert rc.json()["detail"] == ("未找到UNITTEST/BTC, 5m在20200111-20200112中的数据。")

    # 没有策略
    rc = client_post(
        client,
        f"{BASE_URI}/pair_history",
        data={
            "pair": "UNITTEST/BTC",
            "timeframe": timeframe,
            "timerange": "20180111-20180112",
            # "strategy": CURRENT_TEST_STRATEGY,
            "columns": ["rsi", "fastd", "fastk"],
        },
    )
    assert_response(rc, 200)
    result = rc.json()
    assert result["length"] == 289
    assert len(result["data"]) == result["length"]
    assert "columns" in result
    assert "data" in result
    # 没有策略的结果不会分配enter_long。
    assert "enter_long" not in result["columns"]
    assert result["columns"] == ["date", "open", "high", "low", "close", "volume", "__date_ts"]


def test_api_pair_history_live_mode(botclient, tmp_path, mocker):
    """测试API交易对历史实时模式"""
    _ftbot, client = botclient
    _ftbot.config["user_data_dir"] = tmp_path
    _ftbot.config["runmode"] = RunMode.WEBSERVER

    mocker.patch("freqtrade.strategy.interface.IStrategy.load_freqAI_model")
    # 没有策略，实时数据
    gho = mocker.patch(
        "freqtrade.exchange.binance.Binance.get_historic_ohlcv",
        return_value=generate_test_data("1h", 100),
    )
    rc = client_post(
        client,
        f"{BASE_URI}/pair_history",
        data={
            "pair": "UNITTEST/BTC",
            "timeframe": "1h",
            "timerange": "20240101-",
            # "strategy": CURRENT_TEST_STRATEGY,
            "columns": ["rsi", "fastd", "fastk"],
            "live_mode": True,
        },
    )

    assert_response(rc, 200)
    result = rc.json()
    # 100根K线 - 如上面generate_test_data调用中所示
    assert result["length"] == 100
    assert len(result["data"]) == result["length"]
    assert result["columns"] == ["date", "open", "high", "low", "close", "volume", "__date_ts"]
    assert gho.call_count == 1

    gho.reset_mock()
    rc = client_post(
        client,
        f"{BASE_URI}/pair_history",
        data={
            "pair": "UNITTEST/BTC",
            "timeframe": "1h",
            "timerange": "20240101-",
            "strategy": CURRENT_TEST_STRATEGY,
            "columns": ["rsi", "fastd", "fastk"],
            "live_mode": True,
        },
    )

    assert_response(rc, 200)
    result = rc.json()
    # 80根K线 - 如上面generate_test_data调用中所示 - 20根启动K线
    assert result["length"] == 100 - 20
    assert len(result["data"]) == result["length"]

    assert "rsi" in result["columns"]
    assert "enter_long" in result["columns"]
    assert "fastd" in result["columns"]
    assert "date" in result["columns"]
    assert gho.call_count == 1


def test_api_plot_config(botclient, mocker, tmp_path):
    """测试API绘图配置"""
    ftbot, client = botclient
    ftbot.config["user_data_dir"] = tmp_path

    rc = client_get(client, f"{BASE_URI}/plot_config")
    assert_response(rc)
    assert rc.json() == {}

    ftbot.strategy.plot_config = {
        "main_plot": {"sma": {}},
        "subplots": {"RSI": {"rsi": {"color": "red"}}},
    }
    rc = client_get(client, f"{BASE_URI}/plot_config")
    assert_response(rc)
    assert rc.json() == ftbot.strategy.plot_config
    assert isinstance(rc.json()["main_plot"], dict)
    assert isinstance(rc.json()["subplots"], dict)

    ftbot.strategy.plot_config = {"main_plot": {"sma": {}}}
    rc = client_get(client, f"{BASE_URI}/plot_config")
    assert_response(rc)

    assert isinstance(rc.json()["main_plot"], dict)
    assert isinstance(rc.json()["subplots"], dict)

    rc = client_get(client, f"{BASE_URI}/plot_config?strategy=freqai_test_classifier")
    assert_response(rc)
    res = rc.json()
    assert "target_roi" in res["subplots"]
    assert "do_predict" in res["subplots"]

    rc = client_get(client, f"{BASE_URI}/plot_config?strategy=HyperoptableStrategy")
    assert_response(rc)
    assert rc.json()["subplots"] == {}

    rc = client_get(client, f"{BASE_URI}/plot_config?strategy=NotAStrategy")
    assert_response(rc, 502)
    assert rc.json()["detail"] is not None

    mocker.patch("freqtrade.rpc.api_server.api_v1.get_rpc_optional", return_value=None)

    rc = client_get(client, f"{BASE_URI}/plot_config")
    assert_response(rc)


def test_api_strategies(botclient, tmp_path):
    """测试API策略"""
    ftbot, client = botclient
    ftbot.config["user_data_dir"] = tmp_path

    rc = client_get(client, f"{BASE_URI}/strategies")

    assert_response(rc)

    assert rc.json() == {
        "strategies": [
            "HyperoptableStrategy",
            "HyperoptableStrategyV2",
            "InformativeDecoratorTest",
            "StrategyTestV2",
            "StrategyTestV3",
            "StrategyTestV3CustomEntryPrice",
            "StrategyTestV3Futures",
            "freqai_rl_test_strat",
            "freqai_test_classifier",
            "freqai_test_multimodel_classifier_strat",
            "freqai_test_multimodel_strat",
            "freqai_test_strat",
            "strategy_test_v3_recursive_issue",
        ]
    }


def test_api_strategy(botclient, tmp_path, mocker):
    """测试API策略"""
    _ftbot, client = botclient
    _ftbot.config["user_data_dir"] = tmp_path

    rc = client_get(client, f"{BASE_URI}/strategy/{CURRENT_TEST_STRATEGY}")

    assert_response(rc)
    assert rc.json()["strategy"] == CURRENT_TEST_STRATEGY

    data = (Path(__file__).parents[1] / "strategy/strats/strategy_test_v3.py").read_text()
    assert rc.json()["code"] == data

    rc = client_get(client, f"{BASE_URI}/strategy/NoStrat")
    assert_response(rc, 404)

    # 不允许base64策略
    rc = client_get(client, f"{BASE_URI}/strategy/xx:cHJpbnQoImhlbGxvIHdvcmxkIik=")
    assert_response(rc, 500)
    mocker.patch(
        "freqtrade.resolvers.strategy_resolver.StrategyResolver._load_strategy",
        side_effect=Exception("测试"),
    )

    rc = client_get(client, f"{BASE_URI}/strategy/NoStrat")
    assert_response(rc, 502)


def test_api_exchanges(botclient):
    """测试API交易所"""
    _ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/exchanges")
    assert_response(rc)
    response = rc.json()
    assert isinstance(response["exchanges"], list)
    assert len(response["exchanges"]) > 20
    okx = next(x for x in response["exchanges"] if x["classname"] == "okx")
    assert okx == {
        "classname": "okx",
        "name": "OKX",
        "valid": True,
        "supported": True,
        "comment": "",
        "dex": False,
        "is_alias": False,
        "alias_for": None,
        "trade_modes": [
            {"trading_mode": "spot", "margin_mode": ""},
            {"trading_mode": "futures", "margin_mode": "isolated"},
        ],
    }

    mexc = next(x for x in response["exchanges"] if x["classname"] == "mexc")
    assert mexc == {
        "classname": "mexc",
        "name": "MEXC Global",
        "valid": True,
        "supported": False,
        "dex": False,
        "comment": "",
        "is_alias": False,
        "alias_for": None,
        "trade_modes": [{"trading_mode": "spot", "margin_mode": ""}],
    }
    waves = next(x for x in response["exchanges"] if x["classname"] == "wavesexchange")
    assert waves == {
        "classname": "wavesexchange",
        "name": "Waves.Exchange",
        "valid": True,
        "supported": False,
        "dex": True,
        "comment": ANY,
        "is_alias": False,
        "alias_for": None,
        "trade_modes": [{"trading_mode": "spot", "margin_mode": ""}],
    }


def test_list_hyperoptloss(botclient, tmp_path):
    """测试列出超参数优化损失函数"""
    ftbot, client = botclient
    ftbot.config["user_data_dir"] = tmp_path

    rc = client_get(client, f"{BASE_URI}/hyperoptloss")
    assert_response(rc)
    response = rc.json()
    assert isinstance(response["loss_functions"], list)
    assert len(response["loss_functions"]) > 0

    sharpeloss = [r for r in response["loss_functions"] if r["name"] == "SharpeHyperOptLoss"]
    assert len(sharpeloss) == 1
    assert "夏普比率计算" in sharpeloss[0]["description"]
    assert len([r for r in response["loss_functions"] if r["name"] == "SortinoHyperOptLoss"]) == 1


def test_api_freqaimodels(botclient, tmp_path, mocker):
    """测试API FreqAI模型"""
    ftbot, client = botclient
    ftbot.config["user_data_dir"] = tmp_path
    mocker.patch(
        "freqtrade.resolvers.freqaimodel_resolver.FreqaiModelResolver.search_all_objects",
        return_value=[
            {"name": "LightGBMClassifier"},
            {"name": "LightGBMClassifierMultiTarget"},
            {"name": "LightGBMRegressor"},
            {"name": "LightGBMRegressorMultiTarget"},
            {"name": "ReinforcementLearner"},
            {"name": "ReinforcementLearner_multiproc"},
            {"name": "SKlearnRandomForestClassifier"},
            {"name": "XGBoostClassifier"},
            {"name": "XGBoostRFClassifier"},
            {"name": "XGBoostRFRegressor"},
            {"name": "XGBoostRegressor"},
            {"name": "XGBoostRegressorMultiTarget"},
        ],
    )

    rc = client_get(client, f"{BASE_URI}/freqaimodels")

    assert_response(rc)

    assert rc.json() == {
        "freqaimodels": [
            "LightGBMClassifier",
            "LightGBMClassifierMultiTarget",
            "LightGBMRegressor",
            "LightGBMRegressorMultiTarget",
            "ReinforcementLearner",
            "ReinforcementLearner_multiproc",
            "SKlearnRandomForestClassifier",
            "XGBoostClassifier",
            "XGBoostRFClassifier",
            "XGBoostRFRegressor",
            "XGBoostRegressor",
            "XGBoostRegressorMultiTarget",
        ]
    }


def test_api_pairlists_available(botclient, tmp_path):
    """测试API可用交易对列表"""
    ftbot, client = botclient
    ftbot.config["user_data_dir"] = tmp_path

    rc = client_get(client, f"{BASE_URI}/pairlists/available")

    assert_response(rc, 503)
    assert rc.json()["detail"] == "机器人不在正确的状态。"

    ftbot.config["runmode"] = RunMode.WEBSERVER

    rc = client_get(client, f"{BASE_URI}/pairlists/available")
    assert_response(rc)
    response = rc.json()
    assert isinstance(response["pairlists"], list)
    assert len(response["pairlists"]) > 0

    assert len([r for r in response["pairlists"] if r["name"] == "AgeFilter"]) == 1
    assert len([r for r in response["pairlists"] if r["name"] == "VolumePairList"]) == 1
    assert len([r for r in response["pairlists"] if r["name"] == "StaticPairList"]) == 1

    volumepl = next(r for r in response["pairlists"] if r["name"] == "VolumePairList")
    assert volumepl["is_pairlist_generator"] is True
    assert len(volumepl["params"]) > 1
    age_pl = next(r for r in response["pairlists"] if r["name"] == "AgeFilter")
    assert age_pl["is_pairlist_generator"] is False
    assert len(volumepl["params"]) > 2


def test_api_pairlists_evaluate(botclient, tmp_path, mocker):
    """测试API交易对列表评估"""
    ftbot, client = botclient
    ftbot.config["user_data_dir"] = tmp_path

    rc = client_get(client, f"{BASE_URI}/pairlists/evaluate/randomJob")

    assert_response(rc, 503)
    assert rc.json()["detail"] == "机器人不在正确的状态。"

    ftbot.config["runmode"] = RunMode.WEBSERVER

    rc = client_get(client, f"{BASE_URI}/pairlists/evaluate/randomJob")
    assert_response(rc, 404)
    assert rc.json()["detail"] == "未找到任务。"

    body = {
        "pairlists": [
            {
                "method": "StaticPairList",
            },
        ],
        "blacklist": [],
        "stake_currency": "BTC",
    }
    # 失败，已经在运行
    ApiBG.pairlist_running = True
    rc = client_post(client, f"{BASE_URI}/pairlists/evaluate", body)
    assert_response(rc, 400)
    assert rc.json()["detail"] == "交易对列表评估已在运行。"

    # 应该开始运行
    ApiBG.pairlist_running = False
    rc = client_post(client, f"{BASE_URI}/pairlists/evaluate", body)
    assert_response(rc)
    assert rc.json()["status"] == "交易对列表评估已在后台启动。"
    job_id = rc.json()["job_id"]

    rc = client_get(client, f"{BASE_URI}/background/RandomJob")
    assert_response(rc, 404)
    assert rc.json()["detail"] == "未找到任务。"

    # 后台列表
    rc = client_get(client, f"{BASE_URI}/background")
    assert_response(rc)
    response = rc.json()
    assert isinstance(response, list)
    assert len(response) == 1
    assert response[0]["job_id"] == job_id

    # 获取单个任务
    rc = client_get(client, f"{BASE_URI}/background/{job_id}")
    assert_response(rc)
    response = rc.json()
    assert response["job_id"] == job_id
    assert response["job_category"] == "pairlist"

    rc = client_get(client, f"{BASE_URI}/pairlists/evaluate/{job_id}")
    assert_response(rc)
    response = rc.json()
    assert response["result"]["whitelist"] == ["ETH/BTC", "LTC/BTC", "XRP/BTC", "NEO/BTC"]
    assert response["result"]["length"] == 4

    # 使用额外过滤器重新启动，将列表减少到2个
    body["pairlists"].append({"method": "OffsetFilter", "number_assets": 2})
    rc = client_post(client, f"{BASE_URI}/pairlists/evaluate", body)
    assert_response(rc)
    assert rc.json()["status"] == "交易对列表评估已在后台启动。"
    job_id = rc.json()["job_id"]

    rc = client_get(client, f"{BASE_URI}/pairlists/evaluate/{job_id}")
    assert_response(rc)
    response = rc.json()
    assert response["result"]["whitelist"] == [
        "ETH/BTC",
        "LTC/BTC",
    ]
    assert response["result"]["length"] == 2
    # 修补__run_pairlists
    plm = mocker.patch("freqtrade.rpc.api_server.api_pairlists.__run_pairlist", return_value=None)
    body = {
        "pairlists": [
            {
                "method": "StaticPairList",
            },
        ],
        "blacklist": [],
        "stake_currency": "BTC",
        "exchange": "randomExchange",
        "trading_mode": "futures",
        "margin_mode": "isolated",
    }
    rc = client_post(client, f"{BASE_URI}/pairlists/evaluate", body)
    assert_response(rc)
    assert plm.call_count == 1
    call_config = plm.call_args_list[0][0][1]
    assert call_config["exchange"]["name"] == "randomExchange"
    assert call_config["trading_mode"] == "futures"
    assert call_config["margin_mode"] == "isolated"


def test_list_available_pairs(botclient):
    """测试列出可用交易对"""
    ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/available_pairs")

    assert_response(rc)
    assert rc.json()["length"] == 12
    assert isinstance(rc.json()["pairs"], list)

    rc = client_get(client, f"{BASE_URI}/available_pairs?timeframe=5m")
    assert_response(rc)
    assert rc.json()["length"] == 12

    rc = client_get(client, f"{BASE_URI}/available_pairs?stake_currency=ETH")
    assert_response(rc)
    assert rc.json()["length"] == 1
    assert rc.json()["pairs"] == ["XRP/ETH"]
    assert len(rc.json()["pair_interval"]) == 2

    rc = client_get(client, f"{BASE_URI}/available_pairs?stake_currency=ETH&timeframe=5m")
    assert_response(rc)
    assert rc.json()["length"] == 1
    assert rc.json()["pairs"] == ["XRP/ETH"]
    assert len(rc.json()["pair_interval"]) == 1

    ftbot.config["trading_mode"] = "futures"
    rc = client_get(client, f"{BASE_URI}/available_pairs?timeframe=1h")
    assert_response(rc)
    assert rc.json()["length"] == 1
    assert rc.json()["pairs"] == ["XRP/USDT:USDT"]

    rc = client_get(client, f"{BASE_URI}/available_pairs?timeframe=1h&candletype=mark")
    assert_response(rc)
    assert rc.json()["length"] == 2
    assert rc.json()["pairs"] == ["UNITTEST/USDT:USDT", "XRP/USDT:USDT"]
    assert len(rc.json()["pair_interval"]) == 2


def test_sysinfo(botclient):
    """测试系统信息"""
    _ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/sysinfo")
    assert_response(rc)
    result = rc.json()
    assert "cpu_pct" in result
    assert "ram_pct" in result


def test_api_backtesting(botclient, mocker, fee, caplog, tmp_path):
    """测试API回测"""
    try:
        ftbot, client = botclient
        mocker.patch(f"{EXMS}.get_fee", fee)

        rc = client_get(client, f"{BASE_URI}/backtest")
        # 在默认模式下阻止回测
        assert_response(rc, 503)
        assert rc.json()["detail"] == "机器人不在正确的状态。"

        ftbot.config["runmode"] = RunMode.WEBSERVER
        # 回测尚未开始
        rc = client_get(client, f"{BASE_URI}/backtest")
        assert_response(rc)

        result = rc.json()
        assert result["status"] == "not_started"
        assert not result["running"]
        assert result["status_msg"] == "回测尚未执行"
        assert result["progress"] == 0

        # 重置回测
        rc = client_delete(client, f"{BASE_URI}/backtest")
        assert_response(rc)
        result = rc.json()
        assert result["status"] == "reset"
        assert not result["running"]
        assert result["status_msg"] == "回测重置"
        ftbot.config["export"] = "trades"
        ftbot.config["backtest_cache"] = "day"
        ftbot.config["user_data_dir"] = tmp_path
        ftbot.config["exportfilename"] = tmp_path / "backtest_results"
        ftbot.config["exportfilename"].mkdir()

        # 开始回测
        data = {
            "strategy": CURRENT_TEST_STRATEGY,
            "timeframe": "5m",
            "timerange": "20180110-20180111",
            "max_open_trades": 3,
            "stake_amount": 100,
            "dry_run_wallet": 1000,
            "enable_protections": False,
        }
        rc = client_post(client, f"{BASE_URI}/backtest", data=data)
        assert_response(rc)
        result = rc.json()

        assert result["status"] == "running"
        assert result["progress"] == 0
        assert result["running"]
        assert result["status_msg"] == "回测已启动"

        rc = client_get(client, f"{BASE_URI}/backtest")
        assert_response(rc)

        result = rc.json()
        assert result["status"] == "ended"
        assert not result["running"]
        assert result["status_msg"] == "回测已结束"
        assert result["progress"] == 1
        assert result["backtest_result"]

        rc = client_get(client, f"{BASE_URI}/backtest/abort")
        assert_response(rc)
        result = rc.json()
        assert result["status"] == "not_running"
        assert not result["running"]
        assert result["status_msg"] == "回测已结束"

        # 模拟运行中的回测
        ApiBG.bgtask_running = True
        rc = client_get(client, f"{BASE_URI}/backtest/abort")
        assert_response(rc)
        result = rc.json()
        assert result["status"] == "stopping"
        assert not result["running"]
        assert result["status_msg"] == "回测已结束"

        # 获取运行中的回测...
        rc = client_get(client, f"{BASE_URI}/backtest")
        assert_response(rc)
        result = rc.json()
        assert result["status"] == "running"
        assert result["running"]
        assert result["step"] == "backtest"
        assert result["status_msg"] == "回测运行中"

        # 尝试在任务仍在运行时删除
        rc = client_delete(client, f"{BASE_URI}/backtest")
        assert_response(rc)
        result = rc.json()
        assert result["status"] == "running"

        # 在仍在运行时发布到回测
        rc = client_post(client, f"{BASE_URI}/backtest", data=data)
        assert_response(rc, 502)
        result = rc.json()
        assert "机器人后台任务已在运行" in result["error"]

        ApiBG.bgtask_running = False

        # 重新运行回测（应该获得之前的结果）
        rc = client_post(client, f"{BASE_URI}/backtest", data=data)
        assert_response(rc)
        result = rc.json()
        assert log_has_re("重用之前回测的结果.*", caplog)

        data["stake_amount"] = 101

        mocker.patch(
            "freqtrade.optimize.backtesting.Backtesting.backtest_one_strategy",
            side_effect=DependencyException("DeadBeef"),
        )
        rc = client_post(client, f"{BASE_URI}/backtest", data=data)
        assert log_has("回测导致错误: DeadBeef", caplog)

        rc = client_get(client, f"{BASE_URI}/backtest")
        assert_response(rc)
        result = rc.json()
        assert result["status"] == "error"
        assert "回测失败" in result["status_msg"]

        # 删除回测以避免泄漏，因为回测对象可能会保留。
        rc = client_delete(client, f"{BASE_URI}/backtest")
        assert_response(rc)

        result = rc.json()
        assert result["status"] == "reset"
        assert not result["running"]
        assert result["status_msg"] == "回测重置"

        # 不允许base64策略
        data["strategy"] = "xx:cHJpbnQoImhlbGxvIHdvcmxkIik="
        rc = client_post(client, f"{BASE_URI}/backtest", data=data)
        assert_response(rc, 500)
    finally:
        Backtesting.cleanup()


def test_api_backtest_history(botclient, mocker, testdatadir):
    """测试API回测历史"""
    ftbot, client = botclient
    mocker.patch(
        "freqtrade.data.btanalysis.bt_fileutils._get_backtest_files",
        return_value=[
            testdatadir / "backtest_results/backtest-result_multistrat.json",
            testdatadir / "backtest_results/backtest-result.json",
        ],
    )

    rc = client_get(client, f"{BASE_URI}/backtest/history")
    assert_response(rc, 503)
    assert rc.json()["detail"] == "机器人不在正确的状态。"

    ftbot.config["user_data_dir"] = testdatadir
    ftbot.config["runmode"] = RunMode.WEBSERVER

    rc = client_get(client, f"{BASE_URI}/backtest/history")
    assert_response(rc)
    result = rc.json()
    assert len(result) == 3
    fn = result[0]["filename"]
    assert fn == "backtest-result_multistrat"
    assert result[0]["notes"] == ""
    strategy = result[0]["strategy"]
    rc = client_get(client, f"{BASE_URI}/backtest/history/result?filename={fn}&strategy={strategy}")
    assert_response(rc)
    result2 = rc.json()
    assert result2
    assert result2["status"] == "ended"
    assert not result2["running"]
    assert result2["progress"] == 1
    # 只加载了一个策略 - 即使我们使用多结果
    assert len(result2["backtest_result"]["strategy"]) == 1
    assert result2["backtest_result"]["strategy"][strategy]


def test_api_delete_backtest_history_entry(botclient, tmp_path: Path):
    """测试API删除回测历史条目"""
    ftbot, client = botclient

    # 创建临时目录和文件
    bt_results_base = tmp_path / "backtest_results"
    bt_results_base.mkdir()
    file_path = bt_results_base / "test.json"
    file_path.touch()
    meta_path = file_path.with_suffix(".meta.json")
    meta_path.touch()
    market_change_path = file_path.with_name(file_path.stem + "_market_change.feather")
    market_change_path.touch()

    rc = client_delete(client, f"{BASE_URI}/backtest/history/randomFile.json")
    assert_response(rc, 503)
    assert rc.json()["detail"] == "机器人不在正确的状态。"

    ftbot.config["user_data_dir"] = tmp_path
    ftbot.config["runmode"] = RunMode.WEBSERVER
    rc = client_delete(client, f"{BASE_URI}/backtest/history/randomFile.json")
    assert rc.status_code == 404
    assert rc.json()["detail"] == "未找到文件。"

    rc = client_delete(client, f"{BASE_URI}/backtest/history/{file_path.name}")
    assert rc.status_code == 200

    assert not file_path.exists()
    assert not meta_path.exists()
    assert not market_change_path.exists()


def test_api_patch_backtest_history_entry(botclient, tmp_path: Path):
    """测试API修补回测历史条目"""
    ftbot, client = botclient

    # 创建临时目录和文件
    bt_results_base = tmp_path / "backtest_results"
    bt_results_base.mkdir()
    file_path = bt_results_base / "test.json"
    file_path.touch()
    meta_path = file_path.with_suffix(".meta.json")
    with meta_path.open("w") as metafile:
        rapidjson.dump(
            {
                CURRENT_TEST_STRATEGY: {
                    "run_id": "6e542efc8d5e62cef6e5be0ffbc29be81a6e751d",
                    "backtest_start_time": 1690176003,
                }
            },
            metafile,
        )

    def read_metadata():
        with meta_path.open("r") as metafile:
            return rapidjson.load(metafile)

    rc = client_patch(client, f"{BASE_URI}/backtest/history/randomFile.json")
    assert_response(rc, 503)

    ftbot.config["user_data_dir"] = tmp_path
    ftbot.config["runmode"] = RunMode.WEBSERVER

    rc = client_patch(
        client,
        f"{BASE_URI}/backtest/history/randomFile.json",
        {
            "strategy": CURRENT_TEST_STRATEGY,
        },
    )
    assert rc.status_code == 404

    # 不存在的策略
    rc = client_patch(
        client,
        f"{BASE_URI}/backtest/history/{file_path.name}",
        {
            "strategy": f"{CURRENT_TEST_STRATEGY}xxx",
        },
    )
    assert rc.status_code == 400
    assert rc.json()["detail"] == "策略不在元数据中。"

    # 没有备注
    rc = client_patch(
        client,
        f"{BASE_URI}/backtest/history/{file_path.name}",
        {
            "strategy": CURRENT_TEST_STRATEGY,
        },
    )
    assert rc.status_code == 200
    res = rc.json()
    assert isinstance(res, list)
    assert len(res) == 1
    assert res[0]["strategy"] == CURRENT_TEST_STRATEGY
    assert res[0]["notes"] == ""

    fileres = read_metadata()
    assert fileres[CURRENT_TEST_STRATEGY]["run_id"] == res[0]["run_id"]
    assert fileres[CURRENT_TEST_STRATEGY]["notes"] == ""

    rc = client_patch(
        client,
        f"{BASE_URI}/backtest/history/{file_path.name}",
        {
            "strategy": CURRENT_TEST_STRATEGY,
            "notes": "FooBar",
        },
    )
    assert rc.status_code == 200
    res = rc.json()
    assert isinstance(res, list)
    assert len(res) == 1
    assert res[0]["strategy"] == CURRENT_TEST_STRATEGY
    assert res[0]["notes"] == "FooBar"

    fileres = read_metadata()
    assert fileres[CURRENT_TEST_STRATEGY]["run_id"] == res[0]["run_id"]
    assert fileres[CURRENT_TEST_STRATEGY]["notes"] == "FooBar"


def test_api_patch_backtest_market_change(botclient, tmp_path: Path):
    """测试API修补回测市场变化"""
    ftbot, client = botclient

    # 创建临时目录和文件
    bt_results_base = tmp_path / "backtest_results"
    bt_results_base.mkdir()
    file_path = bt_results_base / "test_22_market_change.feather"
    df = pd.DataFrame(
        {
            "date": ["2018-01-01T00:00:00Z", "2018-01-01T00:05:00Z"],
            "count": [2, 4],
            "mean": [2555, 2556],
            "rel_mean": [0, 0.022],
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df.to_feather(file_path, compression_level=9, compression="lz4")
    # 不存在的文件
    rc = client_get(client, f"{BASE_URI}/backtest/history/randomFile.json/market_change")
    assert_response(rc, 503)

    ftbot.config["user_data_dir"] = tmp_path
    ftbot.config["runmode"] = RunMode.WEBSERVER

    rc = client_get(client, f"{BASE_URI}/backtest/history/randomFile.json/market_change")
    assert_response(rc, 404)

    rc = client_get(client, f"{BASE_URI}/backtest/history/test_22/market_change")
    assert_response(rc, 200)
    result = rc.json()
    assert result["length"] == 2
    assert result["columns"] == ["date", "count", "mean", "rel_mean", "__date_ts"]
    assert result["data"] == [
        ["2018-01-01T00:00:00Z", 2, 2555, 0.0, 1514764800000],
        ["2018-01-01T00:05:00Z", 4, 2556, 0.022, 1514765100000],
    ]


def test_health(botclient):
    """测试健康检查"""
    _ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/health")

    assert_response(rc)
    ret = rc.json()
    assert ret["last_process_ts"] is None
    assert ret["last_process"] is None


def test_api_ws_subscribe(botclient, mocker):
    """测试API WebSocket订阅"""
    _ftbot, client = botclient
    ws_url = f"/api/v1/message/ws?token={_TEST_WS_TOKEN}"

    sub_mock = mocker.patch("freqtrade.rpc.api_server.ws.WebSocketChannel.set_subscriptions")

    with client.websocket_connect(ws_url) as ws:
        ws.send_json({"type": "subscribe", "data": ["whitelist"]})
        time.sleep(0.2)

    # 检查调用计数现在是1，因为我们发送了有效的订阅请求
    assert sub_mock.call_count == 1

    with client.websocket_connect(ws_url) as ws:
        ws.send_json({"type": "subscribe", "data": "whitelist"})
        time.sleep(0.2)

    # 调用计数没有改变，因为订阅请求无效
    assert sub_mock.call_count == 1


def test_api_ws_requests(botclient, caplog):
    """测试API WebSocket请求"""
    caplog.set_level(logging.DEBUG)

    _ftbot, client = botclient
    ws_url = f"/api/v1/message/ws?token={_TEST_WS_TOKEN}"

    # 测试白名单请求
    with client.websocket_connect(ws_url) as ws:
        ws.send_json({"type": "whitelist", "data": None})
        response = ws.receive_json()

    assert log_has_re(r"来自.+的白名单类型请求", caplog)
    assert response["type"] == "whitelist"

    # 测试analyzed_df请求
    with client.websocket_connect(ws_url) as ws:
        ws.send_json({"type": "analyzed_df", "data": {}})
        response = ws.receive_json()

    assert log_has_re(r"来自.+的analyzed_df类型请求", caplog)
    assert response["type"] == "analyzed_df"

    caplog.clear()
    # 测试带数据的analyzed_df请求
    with client.websocket_connect(ws_url) as ws:
        ws.send_json({"type": "analyzed_df", "data": {"limit": 100}})
        response = ws.receive_json()

    assert log_has_re(r"来自.+的analyzed_df类型请求", caplog)
    assert response["type"] == "analyzed_df"


def test_api_ws_send_msg(default_conf, mocker, caplog):
    """测试API WebSocket发送消息"""
    try:
        caplog.set_level(logging.DEBUG)

        default_conf.update(
            {
                "api_server": {
                    "enabled": True,
                    "listen_ip_address": "127.0.0.1",
                    "listen_port": 8080,
                    "CORS_origins": ["http://example.com"],
                    "username": _TEST_USER,
                    "password": _TEST_PASS,
                    "ws_token": _TEST_WS_TOKEN,
                }
            }
        )
        mocker.patch("freqtrade.rpc.telegram.Telegram._init")
        mocker.patch("freqtrade.rpc.api_server.ApiServer.start_api")
        apiserver = ApiServer(default_conf)
        apiserver.add_rpc_handler(RPC(get_patched_freqtradebot(mocker, default_conf)))

        # 启动测试客户端上下文管理器以运行生命周期事件
        with TestClient(apiserver.app):
            # 测试消息在消息流上发布
            test_message = {"type": "status", "data": "test"}
            first_waiter = apiserver._message_stream._waiter
            apiserver.send_msg(test_message)
            assert first_waiter.result()[0] == test_message

            second_waiter = apiserver._message_stream._waiter
            apiserver.send_msg(test_message)
            assert first_waiter != second_waiter

    finally:
        ApiServer.shutdown()
        ApiServer.shutdown()


def test_api_download_data(botclient, mocker, tmp_path):
    """测试API下载数据"""
    ftbot, client = botclient

    rc = client_post(client, f"{BASE_URI}/download_data", data={})
    assert_response(rc, 503)
    assert rc.json()["detail"] == "机器人不在正确的状态。"

    ftbot.config["runmode"] = RunMode.WEBSERVER
    ftbot.config["user_data_dir"] = tmp_path

    body = {
        "pairs": ["ETH/BTC", "XRP/BTC"],
        "timeframes": ["5m"],
    }

    # 失败，已经在运行
    ApiBG.download_data_running = True
    rc = client_post(client, f"{BASE_URI}/download_data", body)
    assert_response(rc, 400)
    assert rc.json()["detail"] == "数据下载已在运行。"

    # 重置运行状态
    ApiBG.download_data_running = False

    # 测试成功下载
    mocker.patch(
        "freqtrade.data.history.history_utils.download_data",
        return_value=None,
    )

    rc = client_post(client, f"{BASE_URI}/download_data", body)
    assert_response(rc)
    assert rc.json()["status"] == "数据下载已在后台启动。"
    job_id = rc.json()["job_id"]

    rc = client_get(client, f"{BASE_URI}/background/{job_id}")
    assert_response(rc)
    response = rc.json()
    assert response["job_id"] == job_id
    assert response["job_category"] == "download_data"
    # 由于模拟，任务立即完成。
    assert response["status"] == "success"

    # 后台列表包含该任务
    rc = client_get(client, f"{BASE_URI}/background")
    assert_response(rc)
    response = rc.json()
    assert isinstance(response, list)
    assert len(response) == 1
    assert response[0]["job_id"] == job_id

    # 测试错误情况
    ApiBG.download_data_running = False
    mocker.patch(
        "freqtrade.data.history.history_utils.download_data",
        side_effect=OperationalException("下载错误"),
    )
    rc = client_post(client, f"{BASE_URI}/download_data", body)
    assert_response(rc)
    assert rc.json()["status"] == "数据下载已在后台启动。"
    job_id = rc.json()["job_id"]

    rc = client_get(client, f"{BASE_URI}/background/{job_id}")
    assert_response(rc)
    response = rc.json()
    assert response["job_id"] == job_id
    assert response["job_category"] == "download_data"
    assert response["status"] == "failed"
    assert response["error"] == "下载错误"


def test_api_markets_live(botclient):
    """测试API市场实时"""
    ftbot, client = botclient

    rc = client_get(client, f"{BASE_URI}/markets")
    assert_response(rc, 200)
    response = rc.json()
    assert "markets" in response
    assert len(response["markets"]) >= 0
    assert response["markets"]["XRP/USDT"] == {
        "base": "XRP",
        "quote": "USDT",
        "symbol": "XRP/USDT",
        "spot": True,
        "swap": False,
    }

    assert "BTC/USDT" in response["markets"]
    assert "XRP/BTC" in response["markets"]

    rc = client_get(
        client,
        f"{BASE_URI}/markets?base=XRP",
    )
    assert_response(rc, 200)
    response = rc.json()
    assert "XRP/USDT" in response["markets"]
    assert "XRP/BTC" in response["markets"]

    assert "BTC/USDT" not in response["markets"]


def test_api_markets_webserver(botclient):
    """测试API市场网络服务器"""
    # 确保网络服务器交易所被重置
    ApiBG.exchanges = {}
    ftbot, client = botclient
    # 在网络服务器模式下测试
    ftbot.config["runmode"] = RunMode.WEBSERVER

    rc = client_get(client, f"{BASE_URI}/markets?exchange=binance")
    assert_response(rc, 200)
    response = rc.json()
    assert "markets" in response
    assert len(response["markets"]) >= 0
    assert response["exchange_id"] == "binance"

    rc = client_get(client, f"{BASE_URI}/markets?exchange=hyperliquid")
    assert_response(rc, 200)

    assert "hyperliquid_spot" in ApiBG.exchanges
    assert "binance_spot" in ApiBG.exchanges