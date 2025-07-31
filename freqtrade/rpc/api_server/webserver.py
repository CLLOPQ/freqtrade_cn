import logging
from ipaddress import ip_address
from typing import Any

import orjson
import uvicorn
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from freqtrade.configuration import running_in_docker
from freqtrade.constants import Config
from freqtrade.exceptions import OperationalException
from freqtrade.rpc.api_server.uvicorn_threaded import UvicornServer
from freqtrade.rpc.api_server.webserver_bgwork import ApiBG
from freqtrade.rpc.api_server.ws.message_stream import MessageStream
from freqtrade.rpc.rpc import RPC, RPCException, RPCHandler
from freqtrade.rpc.rpc_types import RPCSendMsg


logger = logging.getLogger(__name__)


class FTJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        """
        使用快速JSON进行响应。默认以JavaScript方式处理NaN和Inf/-Inf。
        """
        return orjson.dumps(content, option=orjson.OPT_SERIALIZE_NUMPY)


class ApiServer(RPCHandler):
    __instance = None
    __initialized = False

    _rpc: RPC
    _has_rpc: bool = False
    _config: Config = {}
    # websocket消息相关
    _message_stream: MessageStream | None = None

    def __new__(cls, *args, **kwargs):
        """
        此类是单例模式。我们将只存在一个实例。
        """
        if ApiServer.__instance is None:
            ApiServer.__instance = object.__new__(cls)
            ApiServer.__initialized = False
        return ApiServer.__instance

    def __init__(self, config: Config, standalone: bool = False) -> None:
        ApiServer._config = config
        if self.__initialized and (standalone or self._standalone):
            return
        self._standalone: bool = standalone
        self._server = None

        ApiServer.__initialized = True

        api_config = self._config["api_server"]

        self.app = FastAPI(
            title="Freqtrade API",
            docs_url="/docs" if api_config.get("enable_openapi", False) else None,
            redoc_url=None,
            default_response_class=FTJSONResponse,
        )
        self.configure_app(self.app, self._config)
        self.start_api()

    def add_rpc_handler(self, rpc: RPC):
        """
        附加RPC处理器
        """
        if not ApiServer._has_rpc:
            ApiServer._rpc = rpc
            ApiServer._has_rpc = True
        else:
            # 假设我们没有出错，这应该不会发生。
            raise OperationalException("RPC处理器已附加。")

    def cleanup(self) -> None:
        """清理待处理的模块资源"""
        ApiServer._has_rpc = False
        del ApiServer._rpc
        ApiBG.exchanges = {}
        ApiBG.jobs = {}
        if self._server and not self._standalone:
            logger.info("正在停止API服务器")
            # self._server.force_exit, self._server.should_exit = True, True
            self._server.cleanup()

    @classmethod
    def shutdown(cls):
        cls.__initialized = False
        del cls.__instance
        cls.__instance = None
        cls._has_rpc = False
        cls._rpc = None

    def send_msg(self, msg: RPCSendMsg) -> None:
        """
        将消息发布到消息流
        """
        if ApiServer._message_stream:
            ApiServer._message_stream.publish(msg)

    def handle_rpc_exception(self, request, exc):
        logger.error(f"API调用错误: {exc}")
        return JSONResponse(
            status_code=502, content={"error": f"查询{request.url.path}时出错：{exc.message}"}
        )

    def configure_app(self, app: FastAPI, config):
        from freqtrade.rpc.api_server.api_auth import http_basic_or_jwt_token, router_login
        from freqtrade.rpc.api_server.api_background_tasks import router as api_bg_tasks
        from freqtrade.rpc.api_server.api_backtest import router as api_backtest
        from freqtrade.rpc.api_server.api_download_data import router as api_download_data
        from freqtrade.rpc.api_server.api_pair_history import router as api_pair_history
        from freqtrade.rpc.api_server.api_pairlists import router as api_pairlists
        from freqtrade.rpc.api_server.api_v1 import router as api_v1
        from freqtrade.rpc.api_server.api_v1 import router_public as api_v1_public
        from freqtrade.rpc.api_server.api_ws import router as ws_router
        from freqtrade.rpc.api_server.deps import is_webserver_mode
        from freqtrade.rpc.api_server.web_ui import router_ui

        app.include_router(api_v1_public, prefix="/api/v1")

        app.include_router(router_login, prefix="/api/v1", tags=["auth"])
        app.include_router(
            api_v1,
            prefix="/api/v1",
            dependencies=[Depends(http_basic_or_jwt_token)],
        )
        app.include_router(
            api_backtest,
            prefix="/api/v1",
            dependencies=[Depends(http_basic_or_jwt_token), Depends(is_webserver_mode)],
        )
        app.include_router(
            api_bg_tasks,
            prefix="/api/v1",
            dependencies=[Depends(http_basic_or_jwt_token), Depends(is_webserver_mode)],
        )
        app.include_router(
            api_pair_history,
            prefix="/api/v1",
            dependencies=[Depends(http_basic_or_jwt_token), Depends(is_webserver_mode)],
        )
        app.include_router(
            api_pairlists,
            prefix="/api/v1",
            dependencies=[Depends(http_basic_or_jwt_token), Depends(is_webserver_mode)],
        )
        app.include_router(
            api_download_data,
            prefix="/api/v1",
            dependencies=[Depends(http_basic_or_jwt_token), Depends(is_webserver_mode)],
        )
        app.include_router(ws_router, prefix="/api/v1")
        # UI路由必须放在最后！
        app.include_router(router_ui, prefix="")

        app.add_middleware(
            CORSMiddleware,
            allow_origins=config["api_server"].get("CORS_origins", []),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        app.add_exception_handler(RPCException, self.handle_rpc_exception)
        app.add_event_handler(event_type="startup", func=self._api_startup_event)
        app.add_event_handler(event_type="shutdown", func=self._api_shutdown_event)

    async def _api_startup_event(self):
        """
        在启动时创建MessageStream类，以便它可以访问与uvicorn相同的事件循环
        """
        if not ApiServer._message_stream:
            ApiServer._message_stream = MessageStream()

    async def _api_shutdown_event(self):
        """
        在关闭时移除MessageStream类
        """
        if ApiServer._message_stream:
            ApiServer._message_stream = None

    def start_api(self):
        """
        启动API...应在线程中运行。
        """
        rest_ip = self._config["api_server"]["listen_ip_address"]
        rest_port = self._config["api_server"]["listen_port"]

        logger.info(f"在{rest_ip}:{rest_port}启动HTTP服务器")
        if not ip_address(rest_ip).is_loopback and not running_in_docker():
            logger.warning("安全警告 - 本地REST服务器正在监听外部连接")
            logger.warning(
                "安全警告 - 这很不安全，请设置为回环地址，例如在config.json中设置为127.0.0.1"
            )

        if not self._config["api_server"].get("password"):
            logger.warning(
                "安全警告 - 未定义本地REST服务器的密码。请确保这是有意为之！"
            )

        if self._config["api_server"].get("jwt_secret_key", "super-secret") in (
            "super-secret, somethingrandom"
        ):
            logger.warning(
                "安全警告 - `jwt_secret_key`似乎是默认值。其他人可能能够登录您的机器人。"
            )

        logger.info("正在启动本地REST服务器。")
        verbosity = self._config["api_server"].get("verbosity", "error")

        uvconfig = uvicorn.Config(
            self.app,
            port=rest_port,
            host=rest_ip,
            use_colors=False,
            log_config=None,
            access_log=True if verbosity != "error" else False,
            ws_ping_interval=None,  # 我们自己显式设置
        )
        try:
            self._server = UvicornServer(uvconfig)
            if self._standalone:
                self._server.run()
            else:
                self._server.run_in_thread()
        except Exception:
            logger.exception("API服务器启动失败。")