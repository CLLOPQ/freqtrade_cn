"""
此模块管理Webhook通信
"""

import logging
import time
from typing import Any

from requests import RequestException, post

from freqtrade.constants import Config
from freqtrade.enums import RPCMessageType
from freqtrade.rpc import RPC, RPCHandler
from freqtrade.rpc.rpc_types import RPCSendMsg


logger = logging.getLogger(__name__)

logger.debug("已包含模块 rpc.webhook ...")


class Webhook(RPCHandler):
    """此类处理所有Webhook通信"""

    def __init__(self, rpc: RPC, config: Config) -> None:
        """
        初始化Webhook类，并初始化父类RPCHandler
        :param rpc: RPC辅助类实例
        :param config: 配置对象
        :return: None
        """
        super().__init__(rpc, config)

        self._url = self._config["webhook"]["url"]
        self._format = self._config["webhook"].get("format", "form")
        self._retries = self._config["webhook"].get("retries", 0)
        self._retry_delay = self._config["webhook"].get("retry_delay", 0.1)
        self._timeout = self._config["webhook"].get("timeout", 10)

    def cleanup(self) -> None:
        """
        清理待处理的模块资源。
        对于Webhook，此方法不执行任何操作，它们将不再被调用
        """
        pass

    def _get_value_dict(self, msg: RPCSendMsg) -> dict[str, Any] | None:
        """
        获取给定消息类型的值字典
        """
        whconfig = self._config["webhook"]
        if msg["type"].value in whconfig:
            # 显式类型应具有优先级
            valuedict = whconfig.get(msg["type"].value)
        # 已弃用 2022.10 - 仅保留通用方法。
        elif msg["type"] in [RPCMessageType.ENTRY]:
            valuedict = whconfig.get("webhookentry")
        elif msg["type"] in [RPCMessageType.ENTRY_CANCEL]:
            valuedict = whconfig.get("webhookentrycancel")
        elif msg["type"] in [RPCMessageType.ENTRY_FILL]:
            valuedict = whconfig.get("webhookentryfill")
        elif msg["type"] == RPCMessageType.EXIT:
            valuedict = whconfig.get("webhookexit")
        elif msg["type"] == RPCMessageType.EXIT_FILL:
            valuedict = whconfig.get("webhookexitfill")
        elif msg["type"] == RPCMessageType.EXIT_CANCEL:
            valuedict = whconfig.get("webhookexitcancel")
        elif msg["type"] in (
            RPCMessageType.STATUS,
            RPCMessageType.STARTUP,
            RPCMessageType.EXCEPTION,
            RPCMessageType.WARNING,
        ):
            valuedict = whconfig.get("webhookstatus")
        elif msg["type"] in (
            RPCMessageType.PROTECTION_TRIGGER,
            RPCMessageType.PROTECTION_TRIGGER_GLOBAL,
            RPCMessageType.WHITELIST,
            RPCMessageType.ANALYZED_DF,
            RPCMessageType.NEW_CANDLE,
            RPCMessageType.STRATEGY_MSG,
        ):
            # 对于未实现的类型不失败
            return None
        return valuedict

    def send_msg(self, msg: RPCSendMsg) -> None:
        """向Webhook URL发送消息"""
        try:
            valuedict = self._get_value_dict(msg)

            if not valuedict:
                logger.debug("消息类型 '%s' 未为Webhook配置", msg["type"])
                return

            payload = {key: value.format(**msg) for (key, value) in valuedict.items()}
            self._send_msg(payload)
        except KeyError as exc:
            logger.exception(
                "调用Webhook时出错。请检查您的Webhook配置。异常: %s",
                exc,
            )

    def _send_msg(self, payload: dict) -> None:
        """执行实际的Webhook调用"""

        success = False
        attempts = 0
        while not success and attempts <= self._retries:
            if attempts:
                if self._retry_delay:
                    time.sleep(self._retry_delay)
                logger.info("正在重试Webhook...")

            attempts += 1

            try:
                if self._format == "form":
                    response = post(self._url, data=payload, timeout=self._timeout)
                elif self._format == "json":
                    response = post(self._url, json=payload, timeout=self._timeout)
                elif self._format == "raw":
                    response = post(
                        self._url,
                        data=payload["data"],
                        headers={"Content-Type": "text/plain"},
                        timeout=self._timeout,
                    )
                else:
                    raise NotImplementedError(f"未知格式: {self._format}")

                # 如果请求未成功，抛出RequestException
                response.raise_for_status()
                success = True

            except RequestException as exc:
                logger.warning("无法调用Webhook URL。异常: %s", exc)