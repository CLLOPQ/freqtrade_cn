# pragma pylint: disable=unused-argument, unused-variable, protected-access, invalid-name

"""
该模块管理 Telegram 通信
"""

import asyncio
import json
import logging
import re
from collections.abc import Callable, Coroutine
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import partial, wraps
from html import escape
from itertools import chain
from math import isnan
from threading import Thread
from typing import Any, Literal

from tabulate import tabulate
from telegram import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.constants import MessageLimit, ParseMode
from telegram.error import BadRequest, NetworkError, TelegramError
from telegram.ext import Application, CallbackContext, CallbackQueryHandler, CommandHandler
from telegram.helpers import escape_markdown

from freqtrade.__init__ import __version__
from freqtrade.constants import DUST_PER_COIN, Config
from freqtrade.enums import MarketDirection, RPCMessageType, SignalDirection, TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.misc import chunks, plural
from freqtrade.persistence import Trade
from freqtrade.rpc import RPC, RPCException, RPCHandler
from freqtrade.rpc.rpc_types import RPCEntryMsg, RPCExitMsg, RPCOrderMsg, RPCSendMsg
from freqtrade.util import (
    dt_from_ts,
    dt_humanize_delta,
    fmt_coin,
    fmt_coin2,
    format_date,
    round_value,
)


MAX_MESSAGE_LENGTH = MessageLimit.MAX_TEXT_LENGTH


logger = logging.getLogger(__name__)

logger.debug("已包含模块 rpc.telegram ...")


def safe_async_db(func: Callable[..., Any]):
    """
    用于在切换异步上下文时安全处理会话的装饰器
    :param func: 要装饰的函数
    :return: 装饰后的函数
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """装饰器逻辑"""
        try:
            return func(*args, **kwargs)
        finally:
            Trade.session.remove()

    return wrapper


@dataclass
class TimeunitMappings:
    header: str
    message: str
    message2: str
    callback: str
    default: int
    dateformat: str


def authorized_only(command_handler: Callable[..., Coroutine[Any, Any, None]]):
    """
    检查消息是否来自正确 chat_id 的装饰器
    只能与 Telegram 类一起用于装饰实例方法。
    :param command_handler: Telegram CommandHandler
    :return: 装饰后的函数
    """

    @wraps(command_handler)
    async def wrapper(self, *args, **kwargs) -> None:
        """装饰器逻辑"""
        update = kwargs.get("update") or args[0]

        # 拒绝未经授权的消息
        message: Message = (
            update.message if update.callback_query is None else update.callback_query.message
        )
        cchat_id: int = int(message.chat_id)
        ctopic_id: int | None = message.message_thread_id
        from_user_id: str = str(update.effective_user.id if update.effective_user else "")

        chat_id = int(self._config["telegram"]["chat_id"])
        if cchat_id != chat_id:
            logger.info(f"拒绝了来自: {cchat_id} 的未经授权的消息")
            return None
        if (topic_id := self._config["telegram"].get("topic_id")) is not None:
            if str(ctopic_id) != topic_id:
                # 在多话题环境中这很常见。
                logger.debug(f"拒绝了来自错误频道的短信: {cchat_id}, {ctopic_id}")
                return None

        authorized = self._config["telegram"].get("authorized_users", None)
        if authorized is not None and from_user_id not in authorized:
            logger.info(f"未经授权的用户试图控制机器人: {from_user_id}")
            return None
        # 回滚会话以避免获取事务中存储的数据。
        Trade.rollback()
        logger.debug("正在执行处理程序: %s，针对 chat_id: %s", command_handler.__name__, chat_id)
        try:
            return await command_handler(self, *args, **kwargs)
        except RPCException as e:
            await self._send_msg(str(e))
        except BaseException:
            logger.exception("Telegram 模块中发生异常")
        finally:
            Trade.session.remove()

    return wrapper


class Telegram(RPCHandler):
    """该类处理所有 Telegram 通信"""

    def __init__(self, rpc: RPC, config: Config) -> None:
        """
        初始化 Telegram 调用，并初始化父类 RPCHandler
        :param rpc: RPC 辅助类实例
        :param config: 配置对象
        :return: 无
        """
        super().__init__(rpc, config)

        self._app: Application
        self._loop: asyncio.AbstractEventLoop
        self._init_keyboard()
        self._start_thread()

    def _start_thread(self):
        """
        创建并启动轮询线程
        """
        self._thread = Thread(target=self._init, name="FTTelegram")
        self._thread.start()

    def _init_keyboard(self) -> None:
        """
        验证 Telegram 配置部分中的键盘配置。
        """
        self._keyboard: list[list[str | KeyboardButton]] = [
            ["/daily", "/profit", "/balance"],
            ["/status", "/status table", "/performance"],
            ["/count", "/start", "/stop", "/help"],
        ]
        # 不允许带有强制参数和关键命令的命令
        # TODO: DRY! - 在这里列出所有有效命令不好。否则
        #       这需要重构整个 Telegram 模块（与 _help() 中存在相同的问题）。
        valid_keys: list[str] = [
            r"/start$",
            r"/pause$",
            r"/stop$",
            r"/status$",
            r"/status table$",
            r"/trades$",
            r"/performance$",
            r"/buys",
            r"/entries",
            r"/sells",
            r"/exits",
            r"/mix_tags",
            r"/daily$",
            r"/daily \d+$",
            r"/profit$",
            r"/profit \d+",
            r"/stats$",
            r"/count$",
            r"/locks$",
            r"/balance$",
            r"/stopbuy$",
            r"/stopentry$",
            r"/reload_config$",
            r"/show_config$",
            r"/logs$",
            r"/whitelist$",
            r"/whitelist(\ssorted|\sbaseonly)+$",
            r"/blacklist$",
            r"/bl_delete$",
            r"/weekly$",
            r"/weekly \d+$",
            r"/monthly$",
            r"/monthly \d+$",
            r"/forcebuy$",
            r"/forcelong$",
            r"/forceshort$",
            r"/forcesell$",
            r"/forceexit$",
            r"/health$",
            r"/help$",
            r"/version$",
            r"/marketdir (long|short|even|none)$",
            r"/marketdir$",
        ]
        # 创建用于生成键
        valid_keys_print = [k.replace("$", "") for k in valid_keys]

        # config.json 中指定的自定义键盘
        cust_keyboard = self._config["telegram"].get("keyboard", [])
        if cust_keyboard:
            combined = "(" + ")|(".join(valid_keys) + ")"
            # 检查有效快捷方式
            invalid_keys = [
                b for b in chain.from_iterable(cust_keyboard) if not re.match(combined, b)
            ]
            if len(invalid_keys):
                err_msg = (
                    "config.telegram.keyboard: 自定义 Telegram 键盘的命令无效: "
                    f"{invalid_keys}"
                    f"\n有效命令为: {valid_keys_print}"
                )
                raise OperationalException(err_msg)
            else:
                self._keyboard = cust_keyboard
                logger.info(f"正在使用 config.json 中的自定义键盘: {self._keyboard}")

    def _init_telegram_app(self):
        return Application.builder().token(self._config["telegram"]["token"]).build()

    def _init(self) -> None:
        """
        用给定配置初始化此模块，
        注册所有已知命令处理程序
        并开始轮询消息更新
        在单独的线程中运行。
        """
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        self._app = self._init_telegram_app()

        # 注册命令处理程序并开始 Telegram 消息轮询
        handles = [
            CommandHandler("status", self._status),
            CommandHandler("profit", self._profit),
            CommandHandler("balance", self._balance),
            CommandHandler("start", self._start),
            CommandHandler("stop", self._stop),
            CommandHandler(["forcesell", "forceexit", "fx"], self._force_exit),
            CommandHandler(
                ["forcebuy", "forcelong"],
                partial(self._force_enter, order_side=SignalDirection.LONG),
            ),
            CommandHandler(
                "forceshort", partial(self._force_enter, order_side=SignalDirection.SHORT)
            ),
            CommandHandler("reload_trade", self._reload_trade_from_exchange),
            CommandHandler("trades", self._trades),
            CommandHandler("delete", self._delete_trade),
            CommandHandler(["coo", "cancel_open_order"], self._cancel_open_order),
            CommandHandler("performance", self._performance),
            CommandHandler(["buys", "entries"], self._enter_tag_performance),
            CommandHandler(["sells", "exits"], self._exit_reason_performance),
            CommandHandler("mix_tags", self._mix_tag_performance),
            CommandHandler("stats", self._stats),
            CommandHandler("daily", self._daily),
            CommandHandler("weekly", self._weekly),
            CommandHandler("monthly", self._monthly),
            CommandHandler("count", self._count),
            CommandHandler("locks", self._locks),
            CommandHandler(["unlock", "delete_locks"], self._delete_locks),
            CommandHandler(["reload_config", "reload_conf"], self._reload_config),
            CommandHandler(["show_config", "show_conf"], self._show_config),
            CommandHandler(["stopbuy", "stopentry", "pause"], self._pause),
            CommandHandler("whitelist", self._whitelist),
            CommandHandler("blacklist", self._blacklist),
            CommandHandler(["blacklist_delete", "bl_delete"], self._blacklist_delete),
            CommandHandler("logs", self._logs),
            CommandHandler("health", self._health),
            CommandHandler("help", self._help),
            CommandHandler("version", self._version),
            CommandHandler("marketdir", self._changemarketdir),
            CommandHandler("order", self._order),
            CommandHandler("list_custom_data", self._list_custom_data),
            CommandHandler("tg_info", self._tg_info),
        ]
        callbacks = [
            CallbackQueryHandler(self._status_table, pattern="update_status_table"),
            CallbackQueryHandler(self._daily, pattern="update_daily"),
            CallbackQueryHandler(self._weekly, pattern="update_weekly"),
            CallbackQueryHandler(self._monthly, pattern="update_monthly"),
            CallbackQueryHandler(self._profit, pattern="update_profit"),
            CallbackQueryHandler(self._balance, pattern="update_balance"),
            CallbackQueryHandler(self._performance, pattern="update_performance"),
            CallbackQueryHandler(
                self._enter_tag_performance, pattern="update_enter_tag_performance"
            ),
            CallbackQueryHandler(
                self._exit_reason_performance, pattern="update_exit_reason_performance"
            ),
            CallbackQueryHandler(self._mix_tag_performance, pattern="update_mix_tag_performance"),
            CallbackQueryHandler(self._count, pattern="update_count"),
            CallbackQueryHandler(self._force_exit_inline, pattern=r"force_exit__\S+"),
            CallbackQueryHandler(self._force_enter_inline, pattern=r"force_enter__\S+"),
        ]
        for handle in handles:
            self._app.add_handler(handle)

        for callback in callbacks:
            self._app.add_handler(callback)

        logger.info(
            "rpc.telegram 正在监听以下命令: %s",
            [[x for x in sorted(h.commands)] for h in handles],
        )
        self._loop.run_until_complete(self._startup_telegram())

    async def _startup_telegram(self) -> None:
        await self._app.initialize()
        await self._app.start()
        if self._app.updater:
            await self._app.updater.start_polling(
                bootstrap_retries=-1,
                timeout=20,
                # read_latency=60,  # 假定的传输延迟
                drop_pending_updates=True,
                # stop_signals=[],  # 我们不在主线程运行，所以这是必要的
            )
            while True:
                await asyncio.sleep(10)
                if not self._app.updater.running:
                    break

    async def _cleanup_telegram(self) -> None:
        if self._app.updater:
            await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()

    def cleanup(self) -> None:
        """
        停止所有正在运行的 Telegram 线程。
        :return: 无
        """
        # 这可能需要从调用 `start_polling` 开始到 `timeout`。
        asyncio.run_coroutine_threadsafe(self._cleanup_telegram(), self._loop)
        self._thread.join()

    def _exchange_from_msg(self, msg: RPCOrderMsg) -> str:
        """
        从给定消息中提取交易所名称。
        :param msg: 要从中提取交易所名称的消息。
        :return: 交易所名称。
        """
        return f"{msg['exchange']}{' (模拟)' if self._config['dry_run'] else ''}"

    def _add_analyzed_candle(self, pair: str) -> str:
        candle_val = (
            self._config["telegram"].get("notification_settings", {}).get("show_candle", "off")
        )
        if candle_val != "off":
            if candle_val == "ohlc":
                analyzed_df, _ = self._rpc._freqtrade.dataprovider.get_analyzed_dataframe(
                    pair, self._config["timeframe"]
                )
                candle = analyzed_df.iloc[-1].squeeze() if len(analyzed_df) > 0 else None
                if candle is not None:
                    return (
                        f"*蜡烛图 OHLC*: `{candle['open']}, {candle['high']}, "
                        f"{candle['low']}, {candle['close']}`\n"
                    )

        return ""

    def _format_entry_msg(self, msg: RPCEntryMsg) -> str:
        is_fill = msg["type"] in [RPCMessageType.ENTRY_FILL]
        emoji = "\N{CHECK MARK}" if is_fill else "\N{LARGE BLUE CIRCLE}"

        terminology = {
            "1_enter": "新交易",
            "1_entered": "新交易已成交",
            "x_enter": "增加头寸",
            "x_entered": "头寸增加已成交",
        }

        key = f"{'x' if msg['sub_trade'] else '1'}_{'entered' if is_fill else 'enter'}"
        wording = terminology[key]

        message = (
            f"{emoji} *{self._exchange_from_msg(msg)}:*"
            f" {wording} (#{msg['trade_id']})\n"
            f"*交易对:* `{msg['pair']}`\n"
        )
        message += self._add_analyzed_candle(msg["pair"])
        message += f"*入场标签:* `{msg['enter_tag']}`\n" if msg.get("enter_tag") else ""
        message += f"*数量:* `{round_value(msg['amount'], 8)}`\n"
        message += f"*方向:* `{msg['direction']}"
        if msg.get("leverage") and msg.get("leverage", 1.0) != 1.0:
            message += f" ({msg['leverage']:.3g}x)"
        message += "`\n"
        message += f"*开仓价:* `{fmt_coin2(msg['open_rate'], msg['quote_currency'])}`\n"
        if msg["type"] == RPCMessageType.ENTRY and msg["current_rate"]:
            message += (
                f"*当前价格:* `{fmt_coin2(msg['current_rate'], msg['quote_currency'])}`\n"
            )

        profit_fiat_extra = self.__format_profit_fiat(msg, "stake_amount")  # type: ignore
        total = fmt_coin(msg["stake_amount"], msg["quote_currency"])

        message += f"*{'新 ' if msg['sub_trade'] else ''}总额:* `{total}{profit_fiat_extra}`"

        return message

    def _format_exit_msg(self, msg: RPCExitMsg) -> str:
        duration = msg["close_date"].replace(microsecond=0) - msg["open_date"].replace(
            microsecond=0
        )
        duration_min = duration.total_seconds() / 60

        leverage_text = (
            f" ({msg['leverage']:.3g}x)"
            if msg.get("leverage") and msg.get("leverage", 1.0) != 1.0
            else ""
        )

        profit_fiat_extra = self.__format_profit_fiat(msg, "profit_amount")

        profit_extra = (
            f" ({msg['gain']}: {fmt_coin(msg['profit_amount'], msg['quote_currency'])}"
            f"{profit_fiat_extra})"
        )

        is_fill = msg["type"] == RPCMessageType.EXIT_FILL
        is_sub_trade = msg.get("sub_trade")
        is_sub_profit = msg["profit_amount"] != msg.get("cumulative_profit")
        is_final_exit = msg.get("is_final_exit", False) and is_sub_profit
        profit_prefix = "子 " if is_sub_trade else ""
        cp_extra = ""
        exit_wording = "已退出" if is_fill else "正在退出"
        if is_sub_trade or is_final_exit:
            cp_fiat = self.__format_profit_fiat(msg, "cumulative_profit")

            if is_final_exit:
                profit_prefix = "子 "
                cp_extra = (
                    f"*最终利润:* `{msg['final_profit_ratio']:.2%} "
                    f"({msg['cumulative_profit']:.8f} {msg['quote_currency']}{cp_fiat})`\n"
                )
            else:
                exit_wording = f"部分{exit_wording.lower()}"
                if msg["cumulative_profit"]:
                    cp_extra = (
                        f"*累计利润:* `"
                        f"{fmt_coin(msg['cumulative_profit'], msg['stake_currency'])}{cp_fiat}`\n"
                    )
        enter_tag = f"*入场标签:* `{msg['enter_tag']}`\n" if msg.get("enter_tag") else ""
        message = (
            f"{self._get_exit_emoji(msg)} *{self._exchange_from_msg(msg)}:* "
            f"{exit_wording} {msg['pair']} (#{msg['trade_id']})\n"
            f"{self._add_analyzed_candle(msg['pair'])}"
            f"*{f'{profit_prefix}利润' if is_fill else f'未实现{profit_prefix}利润'}:* "
            f"`{msg['profit_ratio']:.2%}{profit_extra}`\n"
            f"{cp_extra}"
            f"{enter_tag}"
            f"*退出原因:* `{msg['exit_reason']}`\n"
            f"*方向:* `{msg['direction']}"
            f"{leverage_text}`\n"
            f"*数量:* `{round_value(msg['amount'], 8)}`\n"
            f"*开仓价:* `{fmt_coin2(msg['open_rate'], msg['quote_currency'])}`\n"
        )
        if msg["type"] == RPCMessageType.EXIT and msg["current_rate"]:
            message += (
                f"*当前价格:* `{fmt_coin2(msg['current_rate'], msg['quote_currency'])}`\n"
            )
            if msg["order_rate"]:
                message += f"*退出价格:* `{fmt_coin2(msg['order_rate'], msg['quote_currency'])}`"
        elif msg["type"] == RPCMessageType.EXIT_FILL:
            message += f"*退出价格:* `{fmt_coin2(msg['close_rate'], msg['quote_currency'])}`"

        if is_sub_trade:
            stake_amount_fiat = self.__format_profit_fiat(msg, "stake_amount")

            rem = fmt_coin(msg["stake_amount"], msg["quote_currency"])
            message += f"\n*剩余:* `{rem}{stake_amount_fiat}`"
        else:
            message += f"\n*持续时间:* `{duration} ({duration_min:.1f} 分钟)`"
        return message

    def __format_profit_fiat(
        self, msg: RPCExitMsg, key: Literal["stake_amount", "profit_amount", "cumulative_profit"]
    ) -> str:
        """
        格式化法币，附加到常规利润输出
        """
        profit_fiat_extra = ""
        if self._rpc._fiat_converter and (fiat_currency := msg.get("fiat_currency")):
            profit_fiat = self._rpc._fiat_converter.convert_amount(
                msg[key], msg["stake_currency"], fiat_currency
            )
            profit_fiat_extra = f" / {profit_fiat:.3f} {fiat_currency}"
        return profit_fiat_extra

    def compose_message(self, msg: RPCSendMsg) -> str | None:
        if msg["type"] == RPCMessageType.ENTRY or msg["type"] == RPCMessageType.ENTRY_FILL:
            message = self._format_entry_msg(msg)

        elif msg["type"] == RPCMessageType.EXIT or msg["type"] == RPCMessageType.EXIT_FILL:
            message = self._format_exit_msg(msg)

        elif (
            msg["type"] == RPCMessageType.ENTRY_CANCEL or msg["type"] == RPCMessageType.EXIT_CANCEL
        ):
            message_side = "入场" if msg["type"] == RPCMessageType.ENTRY_CANCEL else "退出"
            message = (
                f"\N{WARNING SIGN} *{self._exchange_from_msg(msg)}:* "
                f"正在取消 {'部分' if msg.get('sub_trade') else ''}"
                f" {message_side} 订单 {msg['pair']} "
                f"(#{msg['trade_id']})。原因: {msg['reason']}。"
            )

        elif msg["type"] == RPCMessageType.PROTECTION_TRIGGER:
            message = (
                f"*保护* 已触发，原因: {msg['reason']}。 "
                f"`{msg['pair']}` 将被锁定直到 `{msg['lock_end_time']}`。"
            )

        elif msg["type"] == RPCMessageType.PROTECTION_TRIGGER_GLOBAL:
            message = (
                f"*保护* 已触发，原因: {msg['reason']}。 "
                f"*所有交易对* 将被锁定直到 `{msg['lock_end_time']}`。"
            )

        elif msg["type"] == RPCMessageType.STATUS:
            message = f"*状态:* `{msg['status']}`"

        elif msg["type"] == RPCMessageType.WARNING:
            message = f"\N{WARNING SIGN} *警告:* `{msg['status']}`"
        elif msg["type"] == RPCMessageType.EXCEPTION:
            # 错误将包含异常，它们用三重反引号括起来。
            message = f"\N{WARNING SIGN} *错误:* \n {msg['status']}"

        elif msg["type"] == RPCMessageType.STARTUP:
            message = f"{msg['status']}"
        elif msg["type"] == RPCMessageType.STRATEGY_MSG:
            message = f"{msg['msg']}"
        else:
            logger.debug("未知消息类型: %s", msg["type"])
            return None
        return message

    def _message_loudness(self, msg: RPCSendMsg) -> str:
        """确定消息的响度 - on（开）、off（关）或 silent（静音）"""
        default_noti = "on"

        msg_type = msg["type"]
        noti = ""
        if msg["type"] == RPCMessageType.EXIT or msg["type"] == RPCMessageType.EXIT_FILL:
            sell_noti = (
                self._config["telegram"].get("notification_settings", {}).get(str(msg_type), {})
            )

            # 为了向后兼容，sell 仍然可以是字符串
            if isinstance(sell_noti, str):
                noti = sell_noti
            else:
                default_noti = sell_noti.get("*", default_noti)
                noti = sell_noti.get(str(msg["exit_reason"]), default_noti)
        else:
            noti = (
                self._config["telegram"]
                .get("notification_settings", {})
                .get(str(msg_type), default_noti)
            )

        return noti

    def send_msg(self, msg: RPCSendMsg) -> None:
        """向 Telegram 频道发送消息"""
        noti = self._message_loudness(msg)

        if noti == "off":
            logger.info(f"通知 '{msg['type']}' 未发送。")
            # 通知已禁用
            return

        message = self.compose_message(deepcopy(msg))
        if message:
            asyncio.run_coroutine_threadsafe(
                self._send_msg(message, disable_notification=(noti == "silent")), self._loop
            )

    def _get_exit_emoji(self, msg):
        """
        获取退出消息的表情符号
        """

        if float(msg["profit_ratio"]) >= 0.05:
            return "\N{ROCKET}"
        elif float(msg["profit_ratio"]) >= 0.0:
            return "\N{EIGHT SPOKED ASTERISK}"
        elif msg["exit_reason"] == "stop_loss":
            return "\N{WARNING SIGN}"
        else:
            return "\N{CROSS MARK}"

    def _prepare_order_details(self, filled_orders: list, quote_currency: str, is_open: bool):
        """
        准备启用入场调整的交易详情
        """
        lines_detail: list[str] = []
        if len(filled_orders) > 0:
            first_avg = filled_orders[0]["safe_price"]
        order_nr = 0
        for order in filled_orders:
            lines: list[str] = []
            if order["is_open"] is True:
                continue
            order_nr += 1
            wording = "入场" if order["ft_is_entry"] else "出场"

            cur_entry_amount = order["filled"] or order["amount"]
            cur_entry_average = order["safe_price"]
            lines.append("  ")
            lines.append(f"*{wording} #{order_nr}:*")
            if order_nr == 1:
                lines.append(
                    f"*数量:* {round_value(cur_entry_amount, 8)} "
                    f"({fmt_coin(order['cost'], quote_currency)})"
                )
                lines.append(f"*平均价格:* {round_value(cur_entry_average, 8)}")
            else:
                # TODO: 此计算忽略手续费。
                price_to_1st_entry = (cur_entry_average - first_avg) / first_avg
                if is_open:
                    lines.append("({})".format(dt_humanize_delta(order["order_filled_date"])))
                lines.append(
                    f"*数量:* {round_value(cur_entry_amount, 8)} "
                    f"({fmt_coin(order['cost'], quote_currency)})"
                )
                lines.append(
                    f"*平均 {wording} 价格:* {round_value(cur_entry_average, 8)} "
                    f"（距首次入场价 {price_to_1st_entry:.2%}）"
                )
                lines.append(f"*订单成交:* {order['order_filled_date']}")

            lines_detail.append("\n".join(lines))

        return lines_detail

    async def __send_order_msg(self, lines: list[str], r: dict[str, Any]) -> None:
        """
        发送状态消息。
        """
        msg = ""

        for line in lines:
            if line:
                if (len(msg) + len(line) + 1) < MAX_MESSAGE_LENGTH:
                    msg += line + "\n"
                else:
                    await self._send_msg(msg.format(**r))
                    msg = "*交易订单列表 #*`{trade_id}` - 续\n" + line + "\n"

        await self._send_msg(msg.format(**r))

    @authorized_only
    async def _order(self, update: Update, context: CallbackContext) -> None:
        """
        /order 的处理程序。
        返回交易的订单
        :param bot: telegram 机器人
        :param update: 消息更新
        :return: 无
        """

        trade_ids = []
        if context.args and len(context.args) > 0:
            trade_ids = [int(i) for i in context.args if i.isnumeric()]

        results = self._rpc._rpc_trade_status(trade_ids=trade_ids)
        for r in results:
            lines = ["*交易订单列表 #*`{trade_id}`"]

            lines_detail = self._prepare_order_details(
                r["orders"], r["quote_currency"], r["is_open"]
            )
            lines.extend(lines_detail if lines_detail else "")
            await self.__send_order_msg(lines, r)

    @authorized_only
    async def _status(self, update: Update, context: CallbackContext) -> None:
        """
        /status 的处理程序。
        返回当前 TradeThread 状态
        :param bot: telegram 机器人
        :param update: 消息更新
        :return: 无
        """

        if context.args and "table" in context.args:
            await self._status_table(update, context)
            return
        else:
            await self._status_msg(update, context)

    async def _status_msg(self, update: Update, context: CallbackContext) -> None:
        """
        /status 和 /status <id> 的处理程序。

        """
        # 检查是否提供了至少一个数字 ID。
        # 如果是，则尝试仅获取这些交易。
        trade_ids = []
        if context.args and len(context.args) > 0:
            trade_ids = [int(i) for i in context.args if i.isnumeric()]

        results = self._rpc._rpc_trade_status(trade_ids=trade_ids)
        position_adjust = self._config.get("position_adjustment_enable", False)
        max_entries = self._config.get("max_entry_position_adjustment", -1)
        for r in results:
            r["open_date_hum"] = dt_humanize_delta(r["open_date"])
            r["num_entries"] = len([o for o in r["orders"] if o["ft_is_entry"]])
            r["num_exits"] = len(
                [
                    o
                    for o in r["orders"]
                    if not o["ft_is_entry"] and not o["ft_order_side"] == "stoploss"
                ]
            )
            r["exit_reason"] = r.get("exit_reason", "")
            r["stake_amount_r"] = fmt_coin(r["stake_amount"], r["quote_currency"])
            r["max_stake_amount_r"] = fmt_coin(
                r["max_stake_amount"] or r["stake_amount"], r["quote_currency"]
            )
            r["profit_abs_r"] = fmt_coin(r["profit_abs"], r["quote_currency"])
            r["realized_profit_r"] = fmt_coin(r["realized_profit"], r["quote_currency"])
            r["total_profit_abs_r"] = fmt_coin(r["total_profit_abs"], r["quote_currency"])
            lines = [
                "*交易 ID:* `{trade_id}`" + (" `（自 {open_date_hum} 起）`" if r["is_open"] else ""),
                "*当前交易对:* {pair}",
                (
                    f"*方向:* {'`做空`' if r.get('is_short') else '`做多`'}"
                    + " ` ({leverage}x)`"
                    if r.get("leverage")
                    else ""
                ),
                "*数量:* `{amount} ({stake_amount_r})`",
                "*总投资:* `{max_stake_amount_r}`" if position_adjust else "",
                "*入场标签:* `{enter_tag}`" if r["enter_tag"] else "",
                "*退出原因:* `{exit_reason}`" if r["exit_reason"] else "",
            ]

            if position_adjust:
                max_buy_str = f"/{max_entries + 1}" if (max_entries > 0) else ""
                lines.extend(
                    [
                        "*入场次数:* `{num_entries}" + max_buy_str + "`",
                        "*出场次数:* `{num_exits}`",
                    ]
                )

            lines.extend(
                [
                    f"*开仓价:* `{round_value(r['open_rate'], 8)}`",
                    f"*平仓价:* `{round_value(r['close_rate'], 8)}`" if r["close_rate"] else "",
                    "*开仓日期:* `{open_date}`",
                    "*平仓日期:* `{close_date}`" if r["close_date"] else "",
                    (
                        f" \n*当前价格:* `{round_value(r['current_rate'], 8)}`"
                        if r["is_open"]
                        else ""
                    ),
                    ("*未实现利润:* " if r["is_open"] else "*平仓利润: *")
                    + "`{profit_ratio:.2%}` `({profit_abs_r})`",
                ]
            )

            if r["is_open"]:
                if r.get("realized_profit"):
                    lines.extend(
                        [
                            "*已实现利润:* `{realized_profit_ratio:.2%} "
                            "({realized_profit_r})`",
                            "*总利润:* `{total_profit_ratio:.2%} ({total_profit_abs_r})`",
                        ]
                    )

                # 添加空行以提高可读性
                lines.append(" ")
                if (
                    r["stop_loss_abs"] != r["initial_stop_loss_abs"]
                    and r["initial_stop_loss_ratio"] is not None
                ):
                    # 仅在止损与初始止损不同时添加初始止损
                    lines.append(
                        "*初始止损:* `{initial_stop_loss_abs:.8f}` "
                        "`({initial_stop_loss_ratio:.2%})`"
                    )

                # 仅当止损和止损百分比不为 None 时添加
                lines.append(
                    f"*止损:* `{round_value(r['stop_loss_abs'], 8)}` "
                    + ("`({stop_loss_ratio:.2%})`" if r["stop_loss_ratio"] else "")
                )
                lines.append(
                    f"*止损距离:* `{round_value(r['stoploss_current_dist'], 8)}` "
                    "`({stoploss_current_dist_ratio:.2%})`"
                )
                if r.get("open_orders"):
                    lines.append(
                        "*开放订单:* `{open_orders}`"
                        + ("- `{exit_order_status}`" if r["exit_order_status"] else "")
                    )

            await self.__send_status_msg(lines, r)

    async def __send_status_msg(self, lines: list[str], r: dict[str, Any]) -> None:
        """
        发送状态消息。
        """
        msg = ""

        for line in lines:
            if line:
                if (len(msg) + len(line) + 1) < MAX_MESSAGE_LENGTH:
                    msg += line + "\n"
                else:
                    await self._send_msg(msg.format(**r))
                    msg = "*交易 ID:* `{trade_id}` - 续\n" + line + "\n"

        await self._send_msg(msg.format(**r))

    @authorized_only
    async def _status_table(self, update: Update, context: CallbackContext) -> None:
        """
        /status table 的处理程序。
        以表格格式返回当前 TradeThread 状态
        :param bot: telegram 机器人
        :param update: 消息更新
        :return: 无
        """
        fiat_currency = self._config.get("fiat_display_currency", "")
        statlist, head, fiat_profit_sum, fiat_total_profit_sum = self._rpc._rpc_status_table(
            self._config["stake_currency"], fiat_currency
        )

        show_total = not isnan(fiat_profit_sum) and len(statlist) > 1
        show_total_realized = (
            not isnan(fiat_total_profit_sum) and len(statlist) > 1 and fiat_profit_sum
        ) != fiat_total_profit_sum
        max_trades_per_msg = 50
        """
        计算每条消息 50 笔交易的消息数量
        使用 0.99 确保没有额外的（空）消息
        例如，50 笔交易，消息数为 int(50/50 + 0.99) = 1 条
        """
        messages_count = max(int(len(statlist) / max_trades_per_msg + 0.99), 1)
        for i in range(0, messages_count):
            trades = statlist[i * max_trades_per_msg : (i + 1) * max_trades_per_msg]
            if show_total and i == messages_count - 1:
                # 添加总计行
                trades.append(["总计", "", "", f"{fiat_profit_sum:.2f} {fiat_currency}"])
                if show_total_realized:
                    trades.append(
                        [
                            "总计",
                            "（含已实现利润）",
                            "",
                            f"{fiat_total_profit_sum:.2f} {fiat_currency}",
                        ]
                    )

            message = tabulate(trades, headers=head, tablefmt="simple")
            if show_total and i == messages_count - 1:
                # 在总计行之间插入分隔符
                lines = message.split("\n")
                offset = 2 if show_total_realized else 1
                message = "\n".join(lines[:-offset] + [lines[1]] + lines[-offset:])
            await self._send_msg(
                f"<pre>{message}</pre>",
                parse_mode=ParseMode.HTML,
                reload_able=True,
                callback_path="update_status_table",
                query=update.callback_query,
            )

    async def _timeunit_stats(self, update: Update, context: CallbackContext, unit: str) -> None:
        """
        /daily <n> 的处理程序
        返回过去 n 天的每日利润（以 BTC 计）。
        :param bot: telegram 机器人
        :param update: 消息更新
        :return: 无
        """

        vals = {
            "days": TimeunitMappings("日", "每日", "天", "update_daily", 7, "%Y-%m-%d"),
            "weeks": TimeunitMappings(
                "周一", "每周", "周（从周一开始）", "update_weekly", 8, "%Y-%m-%d"
            ),
            "months": TimeunitMappings("月", "每月", "月", "update_monthly", 6, "%Y-%m"),
        }
        val = vals[unit]

        stake_cur = self._config["stake_currency"]
        fiat_disp_cur = self._config.get("fiat_display_currency", "")
        try:
            timescale = int(context.args[0]) if context.args else val.default
        except (TypeError, ValueError, IndexError):
            timescale = val.default
        stats = self._rpc._rpc_timeunit_profit(timescale, stake_cur, fiat_disp_cur, unit)
        stats_tab = tabulate(
            [
                [
                    f"{period['date']:{val.dateformat}} ({period['trade_count']})",
                    f"{fmt_coin(period['abs_profit'], stats['stake_currency'])}",
                    f"{period['fiat_value']:.2f} {stats['fiat_display_currency']}",
                    f"{period['rel_profit']:.2%}",
                ]
                for period in stats["data"]
            ],
            headers=[
                f"{val.header} (计数)",
                f"{stake_cur}",
                f"{fiat_disp_cur}",
                "利润 %",
                "交易数",
            ],
            tablefmt="simple",
        )
        message = (
            f"<b>过去 {timescale} {val.message2} 的{val.message}利润</b>:\n"
            f"<pre>{stats_tab}</pre>"
        )
        await self._send_msg(
            message,
            parse_mode=ParseMode.HTML,
            reload_able=True,
            callback_path=val.callback,
            query=update.callback_query,
        )

    @authorized_only
    async def _daily(self, update: Update, context: CallbackContext) -> None:
        """
        /daily <n> 的处理程序
        返回过去 n 天的每日利润（以 BTC 计）。
        :param bot: telegram 机器人
        :param update: 消息更新
        :return: 无
        """
        await self._timeunit_stats(update, context, "days")

    @authorized_only
    async def _weekly(self, update: Update, context: CallbackContext) -> None:
        """
        /weekly <n> 的处理程序
        返回过去 n 周的每周利润（以 BTC 计）。
        :param bot: telegram 机器人
        :param update: 消息更新
        :return: 无
        """
        await self._timeunit_stats(update, context, "weeks")

    @authorized_only
    async def _monthly(self, update: Update, context: CallbackContext) -> None:
        """
        /monthly <n> 的处理程序
        返回过去 n 个月的每月利润（以 BTC 计）。
        :param bot: telegram 机器人
        :param update: 消息更新
        :return: 无
        """
        await self._timeunit_stats(update, context, "months")

    @authorized_only
    async def _profit(self, update: Update, context: CallbackContext) -> None:
        """
        /profit 的处理程序。
        返回累计利润统计。
        :param bot: telegram 机器人
        :param update: 消息更新
        :return: 无
        """
        stake_cur = self._config["stake_currency"]
        fiat_disp_cur = self._config.get("fiat_display_currency", "")

        start_date = datetime.fromtimestamp(0)
        timescale = None
        try:
            if context.args:
                timescale = int(context.args[0]) - 1
                today_start = datetime.combine(date.today(), datetime.min.time())
                start_date = today_start - timedelta(days=timescale)
        except (TypeError, ValueError, IndexError):
            pass

        stats = self._rpc._rpc_trade_statistics(stake_cur, fiat_disp_cur, start_date)
        profit_closed_coin = stats["profit_closed_coin"]
        profit_closed_ratio_mean = stats["profit_closed_ratio_mean"]
        profit_closed_percent = stats["profit_closed_percent"]
        profit_closed_fiat = stats["profit_closed_fiat"]
        profit_all_coin = stats["profit_all_coin"]
        profit_all_ratio_mean = stats["profit_all_ratio_mean"]
        profit_all_percent = stats["profit_all_percent"]
        profit_all_fiat = stats["profit_all_fiat"]
        trade_count = stats["trade_count"]
        first_trade_date = f"{stats['first_trade_humanized']} ({stats['first_trade_date']})"
        latest_trade_date = f"{stats['latest_trade_humanized']} ({stats['latest_trade_date']})"
        avg_duration = stats["avg_duration"]
        best_pair = stats["best_pair"]
        best_pair_profit_ratio = stats["best_pair_profit_ratio"]
        best_pair_profit_abs = fmt_coin(stats["best_pair_profit_abs"], stake_cur)
        winrate = stats["winrate"]
        expectancy = stats["expectancy"]
        expectancy_ratio = stats["expectancy_ratio"]

        if stats["trade_count"] == 0:
            markdown_msg = f"尚未有交易。\n*机器人启动:* `{stats['bot_start_date']}`"
        else:
            # 要显示的消息
            if stats["closed_trade_count"] > 0:
                fiat_closed_trades = (
                    f"∙ `{fmt_coin(profit_closed_fiat, fiat_disp_cur)}`\n" if fiat_disp_cur else ""
                )
                markdown_msg = (
                    "*投资回报率 (ROI):* 已平仓交易\n"
                    f"∙ `{fmt_coin(profit_closed_coin, stake_cur)} "
                    f"({profit_closed_ratio_mean:.2%}) "
                    f"({profit_closed_percent} \N{GREEK CAPITAL LETTER SIGMA}%)`\n"
                    f"{fiat_closed_trades}"
                )
            else:
                markdown_msg = "`无已平仓交易` \n"
            fiat_all_trades = (
                f"∙ `{fmt_coin(profit_all_fiat, fiat_disp_cur)}`\n" if fiat_disp_cur else ""
            )
            markdown_msg += (
                f"*投资回报率 (ROI):* 所有交易\n"
                f"∙ `{fmt_coin(profit_all_coin, stake_cur)} "
                f"({profit_all_ratio_mean:.2%}) "
                f"({profit_all_percent} \N{GREEK CAPITAL LETTER SIGMA}%)`\n"
                f"{fiat_all_trades}"
                f"*总交易数:* `{trade_count}`\n"
                f"*机器人启动:* `{stats['bot_start_date']}`\n"
                f"*{'首次交易开仓' if not timescale else '显示利润自'}:* "
                f"`{first_trade_date}`\n"
                f"*最新交易开仓:* `{latest_trade_date}`\n"
                f"*胜 / 负:* `{stats['winning_trades']} / {stats['losing_trades']}`\n"
                f"*胜率:* `{winrate:.2%}`\n"
                f"*预期值 (比率):* `{expectancy:.2f} ({expectancy_ratio:.2f})`"
            )
            if stats["closed_trade_count"] > 0:
                markdown_msg += (
                    f"\n*平均持有时间:* `{avg_duration}`\n"
                    f"*表现最佳:* `{best_pair}: {best_pair_profit_abs} "
                    f"({best_pair_profit_ratio:.2%})`\n"
                    f"*交易量:* `{fmt_coin(stats['trading_volume'], stake_cur)}`\n"
                    f"*利润因子:* `{stats['profit_factor']:.2f}`\n"
                    f"*最大回撤:* `{stats['max_drawdown']:.2%} "
                    f"({fmt_coin(stats['max_drawdown_abs'], stake_cur)})`\n"
                    f"    从 `{stats['max_drawdown_start']} "
                    f"({fmt_coin(stats['drawdown_high'], stake_cur)})`\n"
                    f"    到 `{stats['max_drawdown_end']} "
                    f"({fmt_coin(stats['drawdown_low'], stake_cur)})`\n"
                )
        await self._send_msg(
            markdown_msg,
            reload_able=True,
            callback_path="update_profit",
            query=update.callback_query,
        )

    @authorized_only
    async def _stats(self, update: Update, context: CallbackContext) -> None:
        """
        /stats 的处理程序
        显示近期交易的统计数据
        """
        stats = self._rpc._rpc_stats()

        reason_map = {
            "roi": "ROI",
            "stop_loss": "止损",
            "trailing_stop_loss": "追踪止损",
            "stoploss_on_exchange": "交易所止损",
            "exit_signal": "退出信号",
            "force_exit": "强制退出",
            "emergency_exit": "紧急退出",
        }
        exit_reasons_tabulate = [
            [reason_map.get(reason, reason), sum(count.values()), count["wins"], count["losses"]]
            for reason, count in stats["exit_reasons"].items()
        ]
        exit_reasons_msg = "尚未有交易。"
        for reason in chunks(exit_reasons_tabulate, 25):
            exit_reasons_msg = tabulate(reason, headers=["退出原因", "退出数", "胜利数", "失败数"])
            if len(exit_reasons_tabulate) > 25:
                await self._send_msg(f"```\n{exit_reasons_msg}```", ParseMode.MARKDOWN)
                exit_reasons_msg = ""

        durations = stats["durations"]
        duration_msg = tabulate(
            [
                [
                    "胜利",
                    (
                        str(timedelta(seconds=durations["wins"]))
                        if durations["wins"] is not None
                        else "不适用"
                    ),
                ],
                [
                    "失败",
                    (
                        str(timedelta(seconds=durations["losses"]))
                        if durations["losses"] is not None
                        else "不适用"
                    ),
                ],
            ],
            headers=["", "平均持续时间"],
        )
        msg = f"""```\n{exit_reasons_msg}```\n```\n{duration_msg}```"""

        await self._send_msg(msg, ParseMode.MARKDOWN)

    @authorized_only
    async def _balance(self, update: Update, context: CallbackContext) -> None:
        """/balance 的处理程序"""
        full_result = context.args and "full" in context.args
        result = self._rpc._rpc_balance(
            self._config["stake_currency"], self._config.get("fiat_display_currency", "")
        )

        balance_dust_level = self._config["telegram"].get("balance_dust_level", 0.0)
        if not balance_dust_level:
            balance_dust_level = DUST_PER_COIN.get(self._config["stake_currency"], 1.0)

        output = ""
        if self._config["dry_run"]:
            output += "*警告:* 模拟模式下的余额。\n"
        starting_cap = fmt_coin(result["starting_capital"], self._config["stake_currency"])
        output += f"起始资金: `{starting_cap}`"
        starting_cap_fiat = (
            fmt_coin(result["starting_capital_fiat"], self._config["fiat_display_currency"])
            if result["starting_capital_fiat"] > 0
            else ""
        )
        output += (f" `, {starting_cap_fiat}`。\n") if result["starting_capital_fiat"] > 0 else "。\n"

        total_dust_balance = 0
        total_dust_currencies = 0
        for curr in result["currencies"]:
            curr_output = ""
            if (curr["is_position"] or curr["est_stake"] > balance_dust_level) and (
                full_result or curr["is_bot_managed"]
            ):
                if curr["is_position"]:
                    curr_output = (
                        f"*{curr['currency']}:*\n"
                        f"\t`方向: {curr['position']:.8f}`\n"
                        f"\t`估值 {curr['stake']}: "
                        f"{fmt_coin(curr['est_stake'], curr['stake'], False)}`\n"
                    )
                else:
                    est_stake = fmt_coin(
                        curr["est_stake" if full_result else "est_stake_bot"], curr["stake"], False
                    )

                    curr_output = (
                        f"*{curr['currency']}:*\n"
                        f"\t`可用: {curr['free']:.8f}`\n"
                        f"\t`余额: {curr['balance']:.8f}`\n"
                        f"\t`挂单: {curr['used']:.8f}`\n"
                        f"\t`机器人持有: {curr['bot_owned']:.8f}`\n"
                        f"\t`估值 {curr['stake']}: {est_stake}`\n"
                    )

            elif curr["est_stake"] <= balance_dust_level:
                total_dust_balance += curr["est_stake"]
                total_dust_currencies += 1

            # 处理溢出的消息长度
            if len(output + curr_output) >= MAX_MESSAGE_LENGTH:
                await self._send_msg(output)
                output = curr_output
            else:
                output += curr_output

        if total_dust_balance > 0:
            output += (
                f"*{total_dust_currencies} 其他 "
                f"{plural(total_dust_currencies, '货币', '货币')} "
                f"（< {balance_dust_level} {result['stake']}）:*\n"
                f"\t`估值 {result['stake']}: "
                f"{fmt_coin(total_dust_balance, result['stake'], False)}`\n"
            )
        tc = result["trade_count"] > 0
        stake_improve = f" `({result['starting_capital_ratio']:.2%})`" if tc else ""
        fiat_val = f" `({result['starting_capital_fiat_ratio']:.2%})`" if tc else ""
        value = fmt_coin(result["value" if full_result else "value_bot"], result["symbol"], False)
        total_stake = fmt_coin(
            result["total" if full_result else "total_bot"], result["stake"], False
        )
        fiat_estimated_value = (
            f"\t`{result['symbol']}: {value}`{fiat_val}\n" if result["symbol"] else ""
        )
        output += (
            f"\n*估值{' (仅机器人管理资产)' if not full_result else ''}*:\n"
            f"\t`{result['stake']}: {total_stake}`{stake_improve}\n"
            f"{fiat_estimated_value}"
        )
        await self._send_msg(
            output, reload_able=True, callback_path="update_balance", query=update.callback_query
        )

    @authorized_only
    async def _start(self, update: Update, context: CallbackContext) -> None:
        """
        /start 的处理程序。
        启动 TradeThread
        :param bot: telegram 机器人
        :param update: 消息更新
        :return: 无
        """
        msg = self._rpc._rpc_start()
        await self._send_msg(f"状态: `{msg['status']}`")

    @authorized_only
    async def _stop(self, update: Update, context: CallbackContext) -> None:
        """
        /stop 的处理程序。
        停止 TradeThread
        :param bot: telegram 机器人
        :param update: 消息更新
        :return: 无
        """
        msg = self._rpc._rpc_stop()
        await self._send_msg(f"状态: `{msg['status']}`")

    @authorized_only
    async def _reload_config(self, update: Update, context: CallbackContext) -> None:
        """
        /reload_config 的处理程序。
        触发配置文件重新加载
        :param bot: telegram 机器人
        :param update: 消息更新
        :return: 无
        """
        msg = self._rpc._rpc_reload_config()
        await self._send_msg(f"状态: `{msg['status']}`")

    @authorized_only
    async def _pause(self, update: Update, context: CallbackContext) -> None:
        """
        /stop_buy /stop_entry 和 /pause 的处理程序。
        将机器人状态设置为暂停
        :param bot: telegram 机器人
        :param update: 消息更新
        :return: 无
        """
        msg = self._rpc._rpc_pause()
        await self._send_msg(f"状态: `{msg['status']}`")

    @authorized_only
    async def _reload_trade_from_exchange(self, update: Update, context: CallbackContext) -> None:
        """
        /reload_trade <tradeid> 的处理程序。
        """
        if not context.args or len(context.args) == 0:
            raise RPCException("未设置交易 ID。")
        trade_id = int(context.args[0])
        msg = self._rpc._rpc_reload_trade_from_exchange(trade_id)
        await self._send_msg(f"状态: `{msg['status']}`")

    @authorized_only
    async def _force_exit(self, update: Update, context: CallbackContext) -> None:
        """
        /forceexit <id> 的处理程序。
        以当前价格出售给定交易
        :param bot: telegram 机器人
        :param update: 消息更新
        :return: 无
        """

        if context.args:
            trade_id = context.args[0]
            await self._force_exit_action(trade_id)
        else:
            fiat_currency = self._config.get("fiat_display_currency", "")
            try:
                statlist, _, _, _ = self._rpc._rpc_status_table(
                    self._config["stake_currency"], fiat_currency
                )
            except RPCException:
                await self._send_msg(msg="未找到开放交易。")
                return
            trades = []
            for trade in statlist:
                trades.append((trade[0], f"{trade[0]} {trade[1]} {trade[2]} {trade[3]}"))

            trade_buttons = [
                InlineKeyboardButton(text=trade[1], callback_data=f"force_exit__{trade[0]}")
                for trade in trades
            ]
            buttons_aligned = self._layout_inline_keyboard(trade_buttons, cols=1)

            buttons_aligned.append(
                [InlineKeyboardButton(text="取消", callback_data="force_exit__cancel")]
            )
            await self._send_msg(msg="哪个交易？", keyboard=buttons_aligned)

    async def _force_exit_action(self, trade_id: str):
        if trade_id != "cancel":
            try:
                loop = asyncio.get_running_loop()
                # 避免嵌套循环的变通方法
                await loop.run_in_executor(None, safe_async_db(self._rpc._rpc_force_exit), trade_id)
            except RPCException as e:
                await self._send_msg(str(e))

    async def _force_exit_inline(self, update: Update, _: CallbackContext) -> None:
        if update.callback_query:
            query = update.callback_query
            if query.data and "__" in query.data:
                # 输入数据为 "force_exit__<tradid|cancel>"
                trade_id = query.data.split("__")[1].split(" ")[0]
                if trade_id == "cancel":
                    await query.answer()
                    await query.edit_message_text(text="强制退出已取消。")
                    return
                trade: Trade | None = Trade.get_trades(trade_filter=Trade.id == trade_id).first()
                await query.answer()
                if trade:
                    await query.edit_message_text(
                        text=f"正在手动退出交易 #{trade_id}, {trade.pair}"
                    )
                    await self._force_exit_action(trade_id)
                else:
                    await query.edit_message_text(text=f"未找到交易 {trade_id}。")

    async def _force_enter_action(self, pair, price: float | None, order_side: SignalDirection):
        if pair != "cancel":
            try:

                @safe_async_db
                def _force_enter():
                    self._rpc._rpc_force_entry(pair, price, order_side=order_side)

                loop = asyncio.get_running_loop()
                # 避免嵌套循环的变通方法
                await loop.run_in_executor(None, _force_enter)
            except RPCException as e:
                logger.exception("强制买入错误！")
                await self._send_msg(str(e), ParseMode.HTML)

    async def _force_enter_inline(self, update: Update, _: CallbackContext) -> None:
        if update.callback_query:
            query = update.callback_query
            if query.data and "__" in query.data:
                # 输入数据为 "force_enter__<pair|cancel>_<side>"
                payload = query.data.split("__")[1]
                if payload == "cancel":
                    await query.answer()
                    await query.edit_message_text(text="强制入场已取消。")
                    return
                if payload and "_||_" in payload:
                    pair, side = payload.split("_||_")
                    order_side = SignalDirection(side)
                    await query.answer()
                    await query.edit_message_text(text=f"正在手动 {order_side} {pair}")
                    await self._force_enter_action(pair, None, order_side)

    @staticmethod
    def _layout_inline_keyboard(
        buttons: list[InlineKeyboardButton], cols=3
    ) -> list[list[InlineKeyboardButton]]:
        return [buttons[i : i + cols] for i in range(0, len(buttons), cols)]

    @authorized_only
    async def _force_enter(
        self, update: Update, context: CallbackContext, order_side: SignalDirection
    ) -> None:
        """
        /forcelong <asset> <price> 和 /forceshort <asset> <price> 的处理程序
        以给定或当前价格买入交易对
        :param bot: telegram 机器人
        :param update: 消息更新
        :return: 无
        """
        if context.args:
            pair = context.args[0]
            price = float(context.args[1]) if len(context.args) > 1 else None
            await self._force_enter_action(pair, price, order_side)
        else:
            whitelist = self._rpc._rpc_whitelist()["whitelist"]
            pair_buttons = [
                InlineKeyboardButton(
                    text=pair, callback_data=f"force_enter__{pair}_||_{order_side}"
                )
                for pair in sorted(whitelist)
            ]
            buttons_aligned = self._layout_inline_keyboard(pair_buttons)

            buttons_aligned.append(
                [InlineKeyboardButton(text="取消", callback_data="force_enter__cancel")]
            )
            await self._send_msg(
                msg="哪个交易对？", keyboard=buttons_aligned, query=update.callback_query
            )

    @authorized_only
    async def _trades(self, update: Update, context: CallbackContext) -> None:
        """
        /trades <n> 的处理程序
        返回最近 n 笔交易。
        :param bot: telegram 机器人
        :param update: 消息更新
        :return: 无
        """
        stake_cur = self._config["stake_currency"]
        try:
            nrecent = int(context.args[0]) if context.args else 10
        except (TypeError, ValueError, IndexError):
            nrecent = 10
        nonspot = self._config.get("trading_mode", TradingMode.SPOT) != TradingMode.SPOT
        trades = self._rpc._rpc_trade_history(nrecent)
        trades_tab = tabulate(
            [
                [
                    dt_humanize_delta(dt_from_ts(trade["close_timestamp"])),
                    f"{trade['pair']} (#{trade['trade_id']}"
                    f"{(' ' + ('S' if trade['is_short'] else 'L')) if nonspot else ''})",
                    f"{(trade['close_profit']):.2%} ({trade['close_profit_abs']})",
                ]
                for trade in trades["trades"]
            ],
            headers=[
                "平仓日期",
                "交易对 (ID L/S)" if nonspot else "交易对 (ID)",
                f"利润 ({stake_cur})",
            ],
            tablefmt="simple",
        )
        message = f"<b>{min(trades['trades_count'], nrecent)} 笔近期交易</b>:\n" + (
            f"<pre>{trades_tab}</pre>" if trades["trades_count"] > 0 else ""
        )
        await self._send_msg(message, parse_mode=ParseMode.HTML)

    @authorized_only
    async def _delete_trade(self, update: Update, context: CallbackContext) -> None:
        """
        /delete <id> 的处理程序。
        删除给定交易
        :param bot: telegram 机器人
        :param update: 消息更新
        :return: 无
        """
        if not context.args or len(context.args) == 0:
            raise RPCException("未设置交易 ID。")
        trade_id = int(context.args[0])
        msg = self._rpc._rpc_delete(trade_id)
        await self._send_msg(
            f"`{msg['result_msg']}`\n"
            "请务必在交易所手动处理此资产。"
        )

    @authorized_only
    async def _cancel_open_order(self, update: Update, context: CallbackContext) -> None:
        """
        /cancel_open_order <id> 的处理程序。
        取消交易 ID 的开放订单
        :param bot: telegram 机器人
        :param update: 消息更新
        :return: 无
        """
        if not context.args or len(context.args) == 0:
            raise RPCException("未设置交易 ID。")
        trade_id = int(context.args[0])
        self._rpc._rpc_cancel_open_order(trade_id)
        await self._send_msg("开放订单已取消。")

    @authorized_only
    async def _performance(self, update: Update, context: CallbackContext) -> None:
        """
        /performance 的处理程序。
        显示已完成交易的性能统计数据
        :param bot: telegram 机器人
        :param update: 消息更新
        :return: 无
        """
        trades = self._rpc._rpc_performance()
        output = "<b>性能:</b>\n"
        for i, trade in enumerate(trades):
            stat_line = (
                f"{i + 1}.\t <code>{trade['pair']}\t"
                f"{fmt_coin(trade['profit_abs'], self._config['stake_currency'])} "
                f"({trade['profit_ratio']:.2%}) "
                f"({trade['count']})</code>\n"
            )

            if len(output + stat_line) >= MAX_MESSAGE_LENGTH:
                await self._send_msg(output, parse_mode=ParseMode.HTML)
                output = stat_line
            else:
                output += stat_line

        await self._send_msg(
            output,
            parse_mode=ParseMode.HTML,
            reload_able=True,
            callback_path="update_performance",
            query=update.callback_query,
        )

    @authorized_only
    async def _enter_tag_performance(self, update: Update, context: CallbackContext) -> None:
        """
        /entries PAIR 的处理程序。
        显示已完成交易的性能统计数据
        :param bot: telegram 机器人
        :param update: 消息更新
        :return: 无
        """
        pair = None
        if context.args and isinstance(context.args[0], str):
            pair = context.args[0]

        trades = self._rpc._rpc_enter_tag_performance(pair)
        output = "*入场标签表现:*\n"
        for i, trade in enumerate(trades):
            stat_line = (
                f"{i + 1}.\t `{trade['enter_tag']}\t"
                f"{fmt_coin(trade['profit_abs'], self._config['stake_currency'])} "
                f"({trade['profit_ratio']:.2%}) "
                f"({trade['count']})`\n"
            )

            if len(output + stat_line) >= MAX_MESSAGE_LENGTH:
                await self._send_msg(output, parse_mode=ParseMode.MARKDOWN)
                output = stat_line
            else:
                output += stat_line

        await self._send_msg(
            output,
            parse_mode=ParseMode.MARKDOWN,
            reload_able=True,
            callback_path="update_enter_tag_performance",
            query=update.callback_query,
        )

    @authorized_only
    async def _exit_reason_performance(self, update: Update, context: CallbackContext) -> None:
        """
        /exits 的处理程序。
        显示已完成交易的性能统计数据
        :param bot: telegram 机器人
        :param update: 消息更新
        :return: 无
        """
        pair = None
        if context.args and isinstance(context.args[0], str):
            pair = context.args[0]

        trades = self._rpc._rpc_exit_reason_performance(pair)
        output = "*退出原因表现:*\n"
        for i, trade in enumerate(trades):
            stat_line = (
                f"{i + 1}.\t `{trade['exit_reason']}\t"
                f"{fmt_coin(trade['profit_abs'], self._config['stake_currency'])} "
                f"({trade['profit_ratio']:.2%}) "
                f"({trade['count']})`\n"
            )

            if len(output + stat_line) >= MAX_MESSAGE_LENGTH:
                await self._send_msg(output, parse_mode=ParseMode.MARKDOWN)
                output = stat_line
            else:
                output += stat_line

        await self._send_msg(
            output,
            parse_mode=ParseMode.MARKDOWN,
            reload_able=True,
            callback_path="update_exit_reason_performance",
            query=update.callback_query,
        )

    @authorized_only
    async def _mix_tag_performance(self, update: Update, context: CallbackContext) -> None:
        """
        /mix_tags 的处理程序。
        显示已完成交易的性能统计数据
        :param bot: telegram 机器人
        :param update: 消息更新
        :return: 无
        """
        pair = None
        if context.args and isinstance(context.args[0], str):
            pair = context.args[0]

        trades = self._rpc._rpc_mix_tag_performance(pair)
        output = "*混合标签表现:*\n"
        for i, trade in enumerate(trades):
            stat_line = (
                f"{i + 1}.\t `{trade['mix_tag']}\t"
                f"{fmt_coin(trade['profit_abs'], self._config['stake_currency'])} "
                f"({trade['profit_ratio']:.2%}) "
                f"({trade['count']})`\n"
            )

            if len(output + stat_line) >= MAX_MESSAGE_LENGTH:
                await self._send_msg(output, parse_mode=ParseMode.MARKDOWN)
                output = stat_line
            else:
                output += stat_line

        await self._send_msg(
            output,
            parse_mode=ParseMode.MARKDOWN,
            reload_able=True,
            callback_path="update_mix_tag_performance",
            query=update.callback_query,
        )

    @authorized_only
    async def _count(self, update: Update, context: CallbackContext) -> None:
        """
        /count 的处理程序。
        返回正在运行的交易数量
        :param bot: telegram 机器人
        :param update: 消息更新
        :return: 无
        """
        counts = self._rpc._rpc_count()
        message = tabulate(
            {k: [v] for k, v in counts.items()},
            headers=["当前", "最大", "总抵押"],
            tablefmt="simple",
        )
        message = f"<pre>{message}</pre>"
        logger.debug(message)
        await self._send_msg(
            message,
            parse_mode=ParseMode.HTML,
            reload_able=True,
            callback_path="update_count",
            query=update.callback_query,
        )

    @authorized_only
    async def _locks(self, update: Update, context: CallbackContext) -> None:
        """
        /locks 的处理程序。
        返回当前活动的锁定
        """
        rpc_locks = self._rpc._rpc_locks()
        if not rpc_locks["locks"]:
            await self._send_msg("没有活动的锁定。", parse_mode=ParseMode.HTML)

        for locks in chunks(rpc_locks["locks"], 25):
            message = tabulate(
                [
                    [lock["id"], lock["pair"], lock["lock_end_time"], lock["reason"]]
                    for lock in locks
                ],
                headers=["ID", "交易对", "直到", "原因"],
                tablefmt="simple",
            )
            message = f"<pre>{escape(message)}</pre>"
            logger.debug(message)
            await self._send_msg(message, parse_mode=ParseMode.HTML)

    @authorized_only
    async def _delete_locks(self, update: Update, context: CallbackContext) -> None:
        """
        /delete_locks 的处理程序。
        返回当前活动的锁定
        """
        arg = context.args[0] if context.args and len(context.args) > 0 else None
        lockid = None
        pair = None
        if arg:
            try:
                lockid = int(arg)
            except ValueError:
                pair = arg

        self._rpc._rpc_delete_lock(lockid=lockid, pair=pair)
        await self._locks(update, context)

    @authorized_only
    async def _whitelist(self, update: Update, context: CallbackContext) -> None:
        """
        /whitelist 的处理程序
        显示当前活跃的白名单
        """
        whitelist = self._rpc._rpc_whitelist()

        if context.args:
            if "sorted" in context.args:
                whitelist["whitelist"] = sorted(whitelist["whitelist"])
            if "baseonly" in context.args:
                whitelist["whitelist"] = [pair.split("/")[0] for pair in whitelist["whitelist"]]

        message = f"正在使用白名单 `{whitelist['method']}`，包含 {whitelist['length']} 个交易对\n"
        message += f"`{', '.join(whitelist['whitelist'])}`"

        logger.debug(message)
        await self._send_msg(message)

    @authorized_only
    async def _blacklist(self, update: Update, context: CallbackContext) -> None:
        """
        /blacklist 的处理程序
        显示当前活跃的黑名单
        """
        await self.send_blacklist_msg(self._rpc._rpc_blacklist(context.args))

    async def send_blacklist_msg(self, blacklist: dict):
        errmsgs = []
        for _, error in blacklist["errors"].items():
            errmsgs.append(f"错误: {error['error_msg']}")
        if errmsgs:
            await self._send_msg("\n".join(errmsgs))

        message = f"黑名单包含 {blacklist['length']} 个交易对\n"
        message += f"`{', '.join(blacklist['blacklist'])}`"

        logger.debug(message)
        await self._send_msg(message)

    @authorized_only
    async def _blacklist_delete(self, update: Update, context: CallbackContext) -> None:
        """
        /bl_delete 的处理程序
        从当前黑名单中删除交易对
        """
        await self.send_blacklist_msg(self._rpc._rpc_blacklist_delete(context.args or []))

    @authorized_only
    async def _logs(self, update: Update, context: CallbackContext) -> None:
        """
        /logs 的处理程序
        显示最新日志
        """
        try:
            limit = int(context.args[0]) if context.args else 10
        except (TypeError, ValueError, IndexError):
            limit = 10
        logs = RPC._rpc_get_logs(limit)["logs"]
        msgs = ""
        msg_template = "*{}* {}: {} \\- `{}`"
        for logrec in logs:
            msg = msg_template.format(
                escape_markdown(logrec[0], version=2),
                escape_markdown(logrec[2], version=2),
                escape_markdown(logrec[3], version=2),
                escape_markdown(logrec[4], version=2),
            )
            if len(msgs + msg) + 10 >= MAX_MESSAGE_LENGTH:
                # 如果消息变得太长，则立即发送消息
                await self._send_msg(msgs, parse_mode=ParseMode.MARKDOWN_V2)
                msgs = msg + "\n"
            else:
                # 将消息添加到要发送的消息中
                msgs += msg + "\n"

        if msgs:
            await self._send_msg(msgs, parse_mode=ParseMode.MARKDOWN_V2)

    @authorized_only
    async def _help(self, update: Update, context: CallbackContext) -> None:
        """
        /help 的处理程序。
        显示机器人的命令
        :param bot: telegram 机器人
        :param update: 消息更新
        :return: 无
        """
        force_enter_text = (
            "*/forcelong <pair> [<rate>]:* `立即买入给定交易对。 "
            "可选地指定买入价格（仅适用于限价订单）。` \n"
        )
        if self._rpc._freqtrade.trading_mode != TradingMode.SPOT:
            force_enter_text += (
                "*/forceshort <pair> [<rate>]:* `立即做空给定交易对。 "
                "可选地指定卖出价格（仅适用于限价订单）。` \n"
            )
        message = (
            "_机器人控制_\n"
            "------------\n"
            "*/start:* `启动交易机器人`\n"
            "*/pause:* `暂停交易机器人新入场，但会优雅处理开放交易`\n"
            "*/stop:* `停止交易机器人`\n"
            "*/stopentry:* `停止入场，但会优雅处理开放交易` \n"
            "*/forceexit <trade_id>|all:* `立即退出给定交易或所有交易，无论盈亏`\n"
            "*/fx <trade_id>|all:* `/forceexit 的别名`\n"
            f"{force_enter_text if self._config.get('force_entry_enable', False) else ''}"
            "*/delete <trade_id>:* `立即从数据库中删除给定交易`\n"
            "*/reload_trade <trade_id>:* `从交易所订单重新加载交易`\n"
            "*/cancel_open_order <trade_id>:* `取消交易的开放订单。 "
            "仅在交易有开放订单时有效。`\n"
            "*/coo <trade_id>|all:* `/cancel_open_order 的别名`\n"
            "*/whitelist [sorted] [baseonly]:* `显示当前白名单。可选地按顺序显示和/或仅显示每个交易对的基础货币。`\n"
            "*/blacklist [pair]:* `显示当前黑名单，或将一个或多个交易对添加到黑名单。` \n"
            "*/blacklist_delete [pairs]| /bl_delete [pairs]:* "
            "`从黑名单中删除交易对/模式。重新加载配置时会重置。` \n"
            "*/reload_config:* `重新加载配置文件` \n"
            "*/unlock <pair|id>:* `解锁此交易对（如果为数字，则为锁定ID）`\n"
            "_当前状态_\n"
            "------------\n"
            "*/show_config:* `显示运行配置` \n"
            "*/locks:* `显示当前锁定的交易对`\n"
            "*/balance:* `显示机器人管理的每种货币余额`\n"
            "*/balance total:* `显示每种货币的账户余额`\n"
            "*/logs [limit]:* `显示最新日志 - 默认为10条` \n"
            "*/count:* `显示活跃交易数量与允许交易数量的对比`\n"
            "*/health* `显示最新进程时间戳 - 默认为1970-01-01 00:00:00` \n"
            "*/marketdir [long | short | even | none]:* `更新表示当前市场方向的用户管理变量。如果未提供方向，将输出当前设置的市场方向。` \n"
            "*/list_custom_data <trade_id> <key>:* `列出交易ID和Key组合的自定义数据。`\n"
            "`如果未提供Key，它将列出该交易ID找到的所有键值对。`\n"
            "_统计数据_\n"
            "------------\n"
            "*/status <trade_id>|[table]:* `列出所有开放交易`\n"
            "         *<trade_id> :* `列出一个或多个特定交易。`\n"
            "                        `多个<trade_id>之间用空格分隔。`\n"
            "         *table :* `将以表格形式显示交易`\n"
            "                `待处理买入订单标有星号 (*)`\n"
            "                `待处理卖出订单标有双星号 (**)`\n"
            "*/entries <pair|none>:* `显示入场标签表现`\n"
            "*/exits <pair|none>:* `显示退出原因表现`\n"
            "*/mix_tags <pair|none>:* `显示组合入场标签+退出原因表现`\n"
            "*/trades [limit]:* `列出最近平仓交易（默认限制10笔）`\n"
            "*/profit [<n>]:* `列出所有已完成交易的累计利润，按过去n天计算`\n"
            "*/performance:* `显示每个已完成交易的性能统计（按交易对分组）`\n"
            "*/daily <n>:* `显示过去n天的每日盈亏`\n"
            "*/weekly <n>:* `显示过去n周的每周统计`\n"
            "*/monthly <n>:* `显示过去n月的每月统计`\n"
            "*/stats:* `显示按卖出原因划分的盈亏，以及买入和卖出的平均持有持续时间。`\n"
            "*/help:* `此帮助消息`\n"
            "*/version:* `显示版本`\n"
        )

        await self._send_msg(message, parse_mode=ParseMode.MARKDOWN)

    @authorized_only
    async def _health(self, update: Update, context: CallbackContext) -> None:
        """
        /health 的处理程序
        显示上次进程时间戳
        """
        health = self._rpc.health()
        message = f"最后进程: `{health['last_process_loc']}`\n"
        message += f"初始机器人启动: `{health['bot_start_loc']}`\n"
        message += f"最后机器人重启: `{health['bot_startup_loc']}`"
        await self._send_msg(message)

    @authorized_only
    async def _version(self, update: Update, context: CallbackContext) -> None:
        """
        /version 的处理程序。
        显示版本信息
        :param bot: telegram 机器人
        :param update: 消息更新
        :return: 无
        """
        strategy_version = self._rpc._freqtrade.strategy.version()
        version_string = f"*版本:* `{__version__}`"
        if strategy_version is not None:
            version_string += f"\n*策略版本: * `{strategy_version}`"

        await self._send_msg(version_string)

    @authorized_only
    async def _show_config(self, update: Update, context: CallbackContext) -> None:
        """
        /show_config 的处理程序。
        显示配置信息
        :param bot: telegram 机器人
        :param update: 消息更新
        :return: 无
        """
        val = RPC._rpc_show_config(self._config, self._rpc._freqtrade.state)

        if val["trailing_stop"]:
            sl_info = (
                f"*初始止损:* `{val['stoploss']}`\n"
                f"*追踪止损正向值:* `{val['trailing_stop_positive']}`\n"
                f"*追踪止损偏移:* `{val['trailing_stop_positive_offset']}`\n"
                f"*仅在达到偏移后追踪:* `{val['trailing_only_offset_is_reached']}`\n"
            )

        else:
            sl_info = f"*止损:* `{val['stoploss']}`\n"

        if val["position_adjustment_enable"]:
            pa_info = (
                f"*头寸调整:* 开启\n"
                f"*最大入场头寸调整:* `{val['max_entry_position_adjustment']}`\n"
            )
        else:
            pa_info = "*头寸调整:* 关闭\n"

        await self._send_msg(
            f"*模式:* `{'模拟运行' if val['dry_run'] else '实盘'}`\n"
            f"*交易所:* `{val['exchange']}`\n"
            f"*市场: * `{val['trading_mode']}`\n"
            f"*每笔交易抵押:* `{val['stake_amount']} {val['stake_currency']}`\n"
            f"*最大开放交易数:* `{val['max_open_trades']}`\n"
            f"*最小投资回报率:* `{val['minimal_roi']}`\n"
            f"*入场策略:* ```\n{json.dumps(val['entry_pricing'])}```\n"
            f"*出场策略:* ```\n{json.dumps(val['exit_pricing'])}```\n"
            f"{sl_info}"
            f"{pa_info}"
            f"*时间框架:* `{val['timeframe']}`\n"
            f"*策略:* `{val['strategy']}`\n"
            f"*当前状态:* `{val['state']}`"
        )

    @authorized_only
    async def _list_custom_data(self, update: Update, context: CallbackContext) -> None:
        """
        /list_custom_data <id> <key> 的处理程序。
        列出指定交易（如果提供则包括键）的自定义数据。
        :param bot: telegram 机器人
        :param update: 消息更新
        :return: 无
        """
        try:
            if not context.args or len(context.args) == 0:
                raise RPCException("未设置交易 ID。")
            trade_id = int(context.args[0])
            key = None if len(context.args) < 2 else str(context.args[1])

            results = self._rpc._rpc_list_custom_data(trade_id, key)
            messages = []
            if len(results) > 0:
                trade_custom_data = results[0]["custom_data"]
                messages.append(
                    "找到自定义数据条目" + ("： " if len(trade_custom_data) > 1 else "： ")
                )
                for custom_data in trade_custom_data:
                    lines = [
                        f"*键:* `{custom_data['key']}`",
                        f"*类型:* `{custom_data['type']}`",
                        f"*值:* `{custom_data['value']}`",
                        f"*创建日期:* `{format_date(custom_data['created_at'])}`",
                        f"*更新日期:* `{format_date(custom_data['updated_at'])}`",
                    ]
                    # 使用列表推导式过滤空行
                    messages.append("\n".join([line for line in lines if line]))
                for msg in messages:
                    if len(msg) > MAX_MESSAGE_LENGTH:
                        msg = "消息因长度超过允许的最大字符数而丢弃: "
                        msg += f"{MAX_MESSAGE_LENGTH}"
                        logger.warning(msg)
                    await self._send_msg(msg)
            else:
                message = f"未找到交易 ID 为 `{trade_id}` 的自定义数据条目"
                message += f"，且键为: `{key}`。" if key is not None else ""
                await self._send_msg(message)

        except RPCException as e:
            await self._send_msg(str(e))

    async def _update_msg(
        self,
        query: CallbackQuery,
        msg: str,
        callback_path: str = "",
        reload_able: bool = False,
        parse_mode: str = ParseMode.MARKDOWN,
    ) -> None:
        if reload_able:
            reply_markup = InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton("刷新", callback_data=callback_path)],
                ]
            )
        else:
            reply_markup = InlineKeyboardMarkup([[]])
        msg += f"\n更新于: {datetime.now().ctime()}"
        if not query.message:
            return

        try:
            await query.edit_message_text(
                text=msg, parse_mode=parse_mode, reply_markup=reply_markup
            )
        except BadRequest as e:
            if "not modified" in e.message.lower():
                pass
            else:
                logger.warning("TelegramError: %s", e.message)
        except TelegramError as telegram_err:
            logger.warning("TelegramError: %s! 放弃该消息。", telegram_err.message)

    async def _send_msg(
        self,
        msg: str,
        parse_mode: str = ParseMode.MARKDOWN,
        disable_notification: bool = False,
        keyboard: list[list[InlineKeyboardButton]] | None = None,
        callback_path: str = "",
        reload_able: bool = False,
        query: CallbackQuery | None = None,
    ) -> None:
        """
        发送给定 markdown 消息
        :param msg: 消息
        :param bot: 备用机器人
        :param parse_mode: telegram 解析模式
        :return: 无
        """
        reply_markup: InlineKeyboardMarkup | ReplyKeyboardMarkup
        if query:
            await self._update_msg(
                query=query,
                msg=msg,
                parse_mode=parse_mode,
                callback_path=callback_path,
                reload_able=reload_able,
            )
            return
        if reload_able and self._config["telegram"].get("reload", True):
            reply_markup = InlineKeyboardMarkup(
                [[InlineKeyboardButton("刷新", callback_data=callback_path)]]
            )
        else:
            if keyboard is not None:
                reply_markup = InlineKeyboardMarkup(keyboard)
            else:
                reply_markup = ReplyKeyboardMarkup(self._keyboard, resize_keyboard=True)
        try:
            try:
                await self._app.bot.send_message(
                    self._config["telegram"]["chat_id"],
                    text=msg,
                    parse_mode=parse_mode,
                    reply_markup=reply_markup,
                    disable_notification=disable_notification,
                    message_thread_id=self._config["telegram"].get("topic_id"),
                )
            except NetworkError as network_err:
                # 有时 Telegram 服务器会重置当前连接，
                # 如果是这种情况，我们再次发送消息。
                logger.warning(
                    "Telegram NetworkError: %s! 再次尝试。", network_err.message
                )
                await self._app.bot.send_message(
                    self._config["telegram"]["chat_id"],
                    text=msg,
                    parse_mode=parse_mode,
                    reply_markup=reply_markup,
                    disable_notification=disable_notification,
                    message_thread_id=self._config["telegram"].get("topic_id"),
                )
        except TelegramError as telegram_err:
            logger.warning("TelegramError: %s! 放弃该消息。", telegram_err.message)

    @authorized_only
    async def _changemarketdir(self, update: Update, context: CallbackContext) -> None:
        """
        /marketdir 的处理程序。
        更新机器人的 market_direction
        :param bot: telegram 机器人
        :param update: 消息更新
        :return: 无
        """
        if context.args and len(context.args) == 1:
            new_market_dir_arg = context.args[0]
            old_market_dir = self._rpc._get_market_direction()
            new_market_dir = None
            if new_market_dir_arg == "long":
                new_market_dir = MarketDirection.LONG
            elif new_market_dir_arg == "short":
                new_market_dir = MarketDirection.SHORT
            elif new_market_dir_arg == "even":
                new_market_dir = MarketDirection.EVEN
            elif new_market_dir_arg == "none":
                new_market_dir = MarketDirection.NONE

            if new_market_dir is not None:
                self._rpc._update_market_direction(new_market_dir)
                await self._send_msg(
                    "成功更新市场方向"
                    f" 从 *{old_market_dir}* 到 *{new_market_dir}*。"
                )
            else:
                raise RPCException(
                    "提供了无效的市场方向。\n"
                    "有效的市场方向: *long, short, even, none*"
                )
        elif context.args is not None and len(context.args) == 0:
            old_market_dir = self._rpc._get_market_direction()
            await self._send_msg(f"当前设置的市场方向: *{old_market_dir}*")
        else:
            raise RPCException(
                "命令 /marketdir 使用无效。\n"
                "用法: */marketdir [short | long | even | none]*"
            )

    async def _tg_info(self, update: Update, context: CallbackContext) -> None:
        """
        有意地未认证的 /tg_info 处理程序。
        返回有关当前 Telegram 聊天的信息 - 即使 chat_id 与此聊天不对应。

        :param update: 消息更新
        :return: 无
        """
        if not update.message:
            return
        chat_id = update.message.chat_id
        topic_id = update.message.message_thread_id
        user_id = (
            update.effective_user.id if topic_id is not None and update.effective_user else None
        )

        msg = f"""Freqtrade 机器人信息:
        ```json
            {{
                "enabled": true,
                "token": "********",
                "chat_id": "{chat_id}",
                {f'"topic_id": "{topic_id}",' if topic_id else ""}
                {f'//"authorized_users": ["{user_id}"]' if topic_id and user_id else ""}
            }}
        ```
        """
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=msg,
                parse_mode=ParseMode.MARKDOWN_V2,
                message_thread_id=topic_id,
            )
        except TelegramError as telegram_err:
            logger.warning("TelegramError: %s! 放弃该消息。", telegram_err.message)
