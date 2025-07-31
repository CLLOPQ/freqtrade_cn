"""
此模块包含用于管理RPC通信的类（Telegram、API等）
"""

import logging
from collections import deque

from freqtrade.constants import Config
from freqtrade.enums import NO_ECHO_MESSAGES, RPCMessageType
from freqtrade.rpc import RPC, RPCHandler
from freqtrade.rpc.rpc_types import RPCSendMsg


logger = logging.getLogger(__name__)


class RPCManager:
    """
    用于管理RPC对象的类（Telegram、API等）
    """

    def __init__(self, freqtrade) -> None:
        """初始化所有已启用的RPC模块"""
        self.registered_modules: list[RPCHandler] = []
        self._rpc = RPC(freqtrade)
        config = freqtrade.config
        # 启用Telegram
        if config.get("telegram", {}).get("enabled", False):
            logger.info("启用 rpc.telegram ...")
            from freqtrade.rpc.telegram import Telegram

            self.registered_modules.append(Telegram(self._rpc, config))

        # 启用Discord
        if config.get("discord", {}).get("enabled", False):
            logger.info("启用 rpc.discord ...")
            from freqtrade.rpc.discord import Discord

            self.registered_modules.append(Discord(self._rpc, config))

        # 启用Webhook
        if config.get("webhook", {}).get("enabled", False):
            logger.info("启用 rpc.webhook ...")
            from freqtrade.rpc.webhook import Webhook

            self.registered_modules.append(Webhook(self._rpc, config))

        # 启用本地REST API服务器以进行命令行控制
        if config.get("api_server", {}).get("enabled", False):
            logger.info("启用 rpc.api_server")
            from freqtrade.rpc.api_server import ApiServer

            apiserver = ApiServer(config)
            apiserver.add_rpc_handler(self._rpc)
            self.registered_modules.append(apiserver)

    def cleanup(self) -> None:
        """停止所有已启用的RPC模块"""
        logger.info("清理RPC模块 ...")
        while self.registered_modules:
            mod = self.registered_modules.pop()
            logger.info("清理 rpc.%s ...", mod.name)
            mod.cleanup()
            del mod

    def send_msg(self, msg: RPCSendMsg) -> None:
        """
        向所有已注册的RPC模块发送给定消息。
        消息由一个或多个字符串键值对组成。
        例如：
        {
            'status': '停止机器人'
        }
        """
        if msg.get("type") not in NO_ECHO_MESSAGES:
            logger.info("发送RPC消息: %s", msg)
        for mod in self.registered_modules:
            logger.debug("将消息转发给 rpc.%s", mod.name)
            try:
                mod.send_msg(msg)
            except NotImplementedError:
                logger.error(f"消息类型 '{msg['type']}' 未被处理器 {mod.name} 实现。")
            except Exception:
                logger.exception("RPC模块 %s 内发生异常", mod.name)

    def process_msg_queue(self, queue: deque) -> None:
        """
        处理队列中的所有消息。
        """
        while queue:
            msg = queue.popleft()
            logger.info("发送RPC策略消息: %s", msg)
            for mod in self.registered_modules:
                if mod._config.get(mod.name, {}).get("allow_custom_messages", False):
                    mod.send_msg(
                        {
                            "type": RPCMessageType.STRATEGY_MSG,
                            "msg": msg,
                        }
                    )

    def startup_messages(self, config: Config, pairlist, protections) -> None:
        if config["dry_run"]:
            self.send_msg(
                {
                    "type": RPCMessageType.WARNING,
                    "status": "干运行已启用。所有交易均为模拟。",
                }
            )
        stake_currency = config["stake_currency"]
        stake_amount = config["stake_amount"]
        minimal_roi = config["minimal_roi"]
        stoploss = config["stoploss"]
        trailing_stop = config["trailing_stop"]
        timeframe = config["timeframe"]
        exchange_name = config["exchange"]["name"]
        strategy_name = config.get("strategy", "")
        pos_adjust_enabled = "开启" if config["position_adjustment_enable"] else "关闭"
        self.send_msg(
            {
                "type": RPCMessageType.STARTUP,
                "status": f"*交易所:* `{exchange_name}`\n"
                f"*每笔交易 stake:* `{stake_amount} {stake_currency}`\n"
                f"*最小ROI:* `{minimal_roi}`\n"
                f"*{'跟踪 ' if trailing_stop else ''}止损:* `{stoploss}`\n"
                f"*仓位调整:* `{pos_adjust_enabled}`\n"
                f"*时间周期:* `{timeframe}`\n"
                f"*策略:* `{strategy_name}`",
            }
        )
        self.send_msg(
            {
                "type": RPCMessageType.STARTUP,
                "status": f"正在根据 {pairlist.short_desc()} 搜索 {stake_currency} 交易对进行买卖",
            }
        )
        if len(protections.name_list) > 0:
            prots = "\n".join([p for prot in protections.short_desc() for k, p in prot.items()])
            self.send_msg(
                {"type": RPCMessageType.STARTUP, "status": f"正在使用防护措施: \n{prots}"}
            )