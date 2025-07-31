import logging

from freqtrade.constants import Config
from freqtrade.enums import RPCMessageType
from freqtrade.rpc import RPC
from freqtrade.rpc.webhook import Webhook


logger = logging.getLogger(__name__)


class Discord(Webhook):
    def __init__(self, rpc: "RPC", config: Config):
        self._config = config
        self.rpc = rpc
        self.strategy = config.get("strategy", "")
        self.timeframe = config.get("timeframe", "")
        self.bot_name = config.get("bot_name", "")

        self._url = config["discord"]["Webhook URL"]
        self._format = "json"
        self._retries = 1
        self._retry_delay = 0.1
        self._timeout = self._config["discord"].get("超时时间", 10)

    def cleanup(self) -> None:
        """
        清理待处理的模块资源。
        对于Webhook来说，这不会执行任何操作，它们将简单地不再被调用。
        """
        pass

    def send_msg(self, msg) -> None:
        if fields := self._config["discord"].get(msg["type"].value):
            logger.info(f"发送Discord消息：{msg}")

            msg["strategy"] = self.strategy
            msg["timeframe"] = self.timeframe
            msg["bot_name"] = self.bot_name
            color = 0x0000FF
            if msg["type"] in (RPCMessageType.EXIT, RPCMessageType.EXIT_FILL):
                profit_ratio = msg.get("profit_ratio")
                color = 0x00FF00 if profit_ratio > 0 else 0xFF0000
            title = msg["type"].value
            if "pair" in msg:
                title = f"交易: {msg['pair']} {msg['type'].value}"
            embeds = [
                {
                    "标题": title,
                    "颜色": color,
                    "字段": [],
                }
            ]
            for f in fields:
                for k, v in f.items():
                    v = v.format(**msg)
                    embeds[0]["字段"].append({"名称": k, "值": v, "内联": True})

            # 发送消息到Discord频道
            payload = {"字段": embeds}
            self._send_msg(payload)