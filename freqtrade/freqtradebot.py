"""
Freqtrade 是此机器人的主要模块。它包含 Freqtrade() 类。
"""

import logging
import traceback
from copy import deepcopy
from datetime import datetime, time, timedelta, timezone
from math import isclose
from threading import Lock
from time import sleep
from typing import Any

from schedule import Scheduler

from freqtrade import constants
from freqtrade.configuration import remove_exchange_credentials, validate_config_consistency
from freqtrade.constants import BuySell, Config, EntryExecuteMode, ExchangeConfig, LongShort
from freqtrade.data.converter import order_book_to_dataframe
from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import (
    ExitCheckTuple,
    ExitType,
    MarginMode,
    RPCMessageType,
    SignalDirection,
    State,
    TradingMode,
)
from freqtrade.exceptions import (
    DependencyException,
    ExchangeError,
    InsufficientFundsError,
    InvalidOrderException,
    PricingError,
)
from freqtrade.exchange import (
    ROUND_DOWN,
    ROUND_UP,
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_seconds,
)
from freqtrade.exchange.exchange_types import CcxtOrder
from freqtrade.leverage.liquidation_price import update_liquidation_prices
from freqtrade.misc import safe_value_fallback, safe_value_fallback2
from freqtrade.mixins import LoggingMixin
from freqtrade.persistence import Order, PairLocks, Trade, init_db
from freqtrade.persistence.key_value_store import set_startup_time
from freqtrade.plugins.pairlistmanager import PairListManager
from freqtrade.plugins.protectionmanager import ProtectionManager
from freqtrade.resolvers import ExchangeResolver, StrategyResolver
from freqtrade.rpc import RPCManager
from freqtrade.rpc.external_message_consumer import ExternalMessageConsumer
from freqtrade.rpc.rpc_types import (
    ProfitLossStr,
    RPCCancelMsg,
    RPCEntryMsg,
    RPCExitCancelMsg,
    RPCExitMsg,
    RPCProtectionMsg,
)
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy.strategy_wrapper import strategy_safe_wrapper
from freqtrade.util import FtPrecise, MeasureTime, PeriodicCache, dt_from_ts, dt_now
from freqtrade.util.migrations.binance_mig import migrate_binance_futures_names
from freqtrade.wallets import Wallets


logger = logging.getLogger(__name__)


class FreqtradeBot(LoggingMixin):
    """
    Freqtrade 是机器人的主类。
    机器人的逻辑从这里开始。
    """

    def __init__(self, config: Config) -> None:
        """
        初始化机器人所需的所有变量和对象
        :param config: 配置字典，您可以使用 Configuration.get_config()
        来获取配置字典。
        """
        self.active_pair_whitelist: list[str] = []

        # 初始化机器人状态
        self.state = State.STOPPED

        # 初始化对象
        self.config = config
        exchange_config: ExchangeConfig = deepcopy(config["exchange"])
        # 从原始交易所配置中移除凭据，以避免意外的凭据暴露
        remove_exchange_credentials(config["exchange"], True)

        self.strategy: IStrategy = StrategyResolver.load_strategy(self.config)

        # 在此处检查配置一致性，因为策略可以设置某些选项
        validate_config_consistency(config)

        self.exchange = ExchangeResolver.load_exchange(
            self.config, exchange_config=exchange_config, load_leverage_tiers=True
        )

        init_db(self.config["db_url"])

        self.wallets = Wallets(self.config, self.exchange)

        PairLocks.timeframe = self.config["timeframe"]

        self.trading_mode: TradingMode = self.config.get("trading_mode", TradingMode.SPOT)
        self.margin_mode: MarginMode = self.config.get("margin_mode", MarginMode.NONE)
        self.last_process: datetime | None = None

        # RPC 在单独的线程中运行，可以在初始化后立即开始处理外部命令，
        # 甚至在 FreqtradeBot 有机会启动其节流之前，
        # 因此 FreqtradeBot 实例中的任何内容都应该准备好（已初始化），包括
        # 机器人的初始状态。
        # 将此放在初始化方法的末尾。
        self.rpc: RPCManager = RPCManager(self)

        self.dataprovider = DataProvider(self.config, self.exchange, rpc=self.rpc)
        self.pairlists = PairListManager(self.exchange, self.config, self.dataprovider)

        self.dataprovider.add_pairlisthandler(self.pairlists)

        # 将 Dataprovider 附加到策略实例
        self.strategy.dp = self.dataprovider
        # 将 Wallets 附加到策略实例
        self.strategy.wallets = self.wallets

        # 如果启用，初始化 ExternalMessageConsumer
        self.emc = (
            ExternalMessageConsumer(self.config, self.dataprovider)
            if self.config.get("external_message_consumer", {}).get("enabled", False)
            else None
        )

        logger.info("开始初始交易对列表刷新")
        with MeasureTime(
            lambda duration, _: logger.info(f"初始交易对列表刷新耗时 {duration:.2f}秒"), 0
        ):
            self.active_pair_whitelist = self._refresh_active_whitelist()

        # 从配置中设置初始机器人状态
        initial_state = self.config.get("initial_state")
        self.state = State[initial_state.upper()] if initial_state else State.STOPPED

        # 保护退出逻辑免受强制平仓的影响，反之亦然
        self._exit_lock = Lock()
        timeframe_secs = timeframe_to_seconds(self.strategy.timeframe)
        self._exit_reason_cache = PeriodicCache(100, ttl=timeframe_secs)
        LoggingMixin.__init__(self, logger, timeframe_secs)

        self._schedule = Scheduler()

        if self.trading_mode == TradingMode.FUTURES:

            def update():
                self.update_funding_fees()
                self.update_all_liquidation_prices()
                self.wallets.update()

            # 如果按 UTC 时间调度，并在每个
            # 资金间隔（由交易所类的 funding_fee_times 指定）执行，
            # 这将更有效率。
            # 但是，这会降低精度 - 因此可能导致问题。
            for time_slot in range(0, 24):
                for minutes in [1, 31]:
                    t = str(time(time_slot, minutes, 2))
                    self._schedule.every().day.at(t).do(update)

        self._schedule.every().day.at("00:02").do(self.exchange.ws_connection_reset)

        self.strategy.ft_bot_start()
        # 在机器人启动后初始化保护机制 - 否则参数不会被加载。
        self.protections = ProtectionManager(self.config, self.strategy.protections)

        def log_took_too_long(duration: float, time_limit: float):
            logger.warning(
                f"策略分析耗时 {duration:.2f}秒，超过了时间周期 "
                f"({time_limit:.2f}秒)的25%。这可能导致订单延迟和信号错过。"
                "请考虑减少策略执行的工作量或减少交易对列表中的交易对数量。"
            )

        self._measure_execution = MeasureTime(log_took_too_long, timeframe_secs * 0.25)

    def notify_status(self, msg: str, msg_type=RPCMessageType.STATUS) -> None:
        """
        此类的用户（worker 等）用于通过 RPC 发送有关机器人状态变化的通知的公共方法。
        """
        self.rpc.send_msg({"type": msg_type, "status": msg})

    def cleanup(self) -> None:
        """
        清理已停止机器人上的待处理资源
        :return: 无
        """
        logger.info("正在清理模块...")
        try:
            # 将数据库活动封装在关机中，以避免数据库消失时出现问题，
            # 并引发进一步的异常。
            if self.config["cancel_open_orders_on_exit"]:
                self.cancel_all_open_orders()

            self.check_for_open_trades()
        except Exception as e:
            logger.warning(f"清理期间发生异常: {e.__class__.__name__} {e}")

        finally:
            self.strategy.ft_bot_cleanup()

        self.rpc.cleanup()
        if self.emc:
            self.emc.shutdown()
        self.exchange.close()
        try:
            Trade.commit()
        except Exception:
            # 如果数据库消失，则会发生异常。
            # 在这种情况下，我们无法提交。
            logger.exception("清理期间发生错误")

    def startup(self) -> None:
        """
        在启动和重新加载机器人时调用 - 触发通知并执行启动任务
        """
        migrate_binance_futures_names(self.config)
        set_startup_time()

        self.rpc.startup_messages(self.config, self.pairlists, self.protections)
        # 使用精度和精度模式更新旧交易
        self.startup_backpopulate_precision()
        # 如果止损被更改，则调整止损
        Trade.stoploss_reinitialization(self.strategy.stoploss)

        # 只在启动时更新未平仓订单
        # 这将在初始迁移后更新数据库
        self.startup_update_open_orders()
        self.update_all_liquidation_prices()
        self.update_funding_fees()

    def process(self) -> None:
        """
        查询持久层以获取未平仓交易并处理它们，
        否则创建新交易。
        :return: 如果一个或多个交易已创建或平仓，则为 True，否则为 False
        """

        # 检查市场是否需要重新加载，并在需要时重新加载
        self.exchange.reload_markets()

        self.update_trades_without_assigned_fees()

        # 从持久层查询交易
        trades: list[Trade] = Trade.get_open_trades()

        self.active_pair_whitelist = self._refresh_active_whitelist(trades)

        # 刷新蜡烛图
        self.dataprovider.refresh(
            self.pairlists.create_pair_list(self.active_pair_whitelist),
            self.strategy.gather_informative_pairs(),
        )

        strategy_safe_wrapper(self.strategy.bot_loop_start, supress_error=True)(
            current_time=datetime.now(timezone.utc)
        )

        with self._measure_execution:
            self.strategy.analyze(self.active_pair_whitelist)

        with self._exit_lock:
            # 检查交易所取消、超时和用户请求的替换
            self.manage_open_orders()

        # 防止与 force_exit 冲突。
        # 否则，freqtrade 可能会在退出过程中尝试重新创建 stoploss_on_exchange 订单，
        # 因为 Telegram 消息在不同的线程中到达。
        with self._exit_lock:
            trades = Trade.get_open_trades()
            # 首先处理当前未平仓交易（头寸）
            self.exit_positions(trades)
            Trade.commit()

        # 在尝试进入新交易之前，检查是否需要调整当前头寸。
        if self.strategy.position_adjustment_enable:
            with self._exit_lock:
                self.process_open_trade_positions()

        # 然后寻找入场机会
        if self.state == State.RUNNING and self.get_free_open_trades():
            self.enter_positions()
        self._schedule.run_pending()
        Trade.commit()
        self.rpc.process_msg_queue(self.dataprovider._msg_queue)
        self.last_process = datetime.now(timezone.utc)

    def process_stopped(self) -> None:
        """
        关闭所有未平仓订单
        """
        if self.config["cancel_open_orders_on_exit"]:
            self.cancel_all_open_orders()

    def check_for_open_trades(self):
        """
        当机器人停止（未重新加载）且仍有未平仓交易活跃时通知用户。
        """
        open_trades = Trade.get_open_trades()

        if len(open_trades) != 0 and self.state != State.RELOAD_CONFIG:
            msg = {
                "type": RPCMessageType.WARNING,
                "status": f"{len(open_trades)} 笔未平仓交易活跃。\n\n"
                f"请在 {self.exchange.name} 上手动处理这些交易，"
                f"或者再次 '/start' 机器人并使用 '/stopentry' "
                f"以优雅地处理未平仓交易。\n"
                f"{'注意：交易是模拟的（干运行）。' if self.config['dry_run'] else ''}",
            }
            self.rpc.send_msg(msg)

    def _refresh_active_whitelist(self, trades: list[Trade] | None = None) -> list[str]:
        """
        从交易对列表刷新活跃白名单，并使用具有未平仓交易的交易对进行扩展。
        """
        # 刷新白名单
        _prev_whitelist = self.pairlists.whitelist
        self.pairlists.refresh_pairlist()
        _whitelist = self.pairlists.whitelist

        if trades:
            # 使用未平仓交易的交易对扩展活跃交易对白名单
            # 它确保也为未平仓交易下载蜡烛图（OHLCV）数据
            _whitelist.extend([trade.pair for trade in trades if trade.pair not in _whitelist])

        # 最后调用以包含已包含的交易对
        if _prev_whitelist != _whitelist:
            self.rpc.send_msg({"type": RPCMessageType.WHITELIST, "data": _whitelist})

        return _whitelist

    def get_free_open_trades(self) -> int:
        """
        返回未平仓交易槽的数量，如果达到最大未平仓交易数，则返回 0
        """
        open_trades = Trade.get_open_trade_count()
        return max(0, self.config["max_open_trades"] - open_trades)

    def update_all_liquidation_prices(self) -> None:
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode == MarginMode.CROSS:
            # 在全仓模式下更新所有交易的强平价格
            update_liquidation_prices(
                exchange=self.exchange,
                wallets=self.wallets,
                stake_currency=self.config["stake_currency"],
                dry_run=self.config["dry_run"],
            )

    def update_funding_fees(self) -> None:
        if self.trading_mode == TradingMode.FUTURES:
            trades: list[Trade] = Trade.get_open_trades()
            for trade in trades:
                trade.set_funding_fees(
                    self.exchange.get_funding_fees(
                        pair=trade.pair,
                        amount=trade.amount,
                        is_short=trade.is_short,
                        open_date=trade.date_last_filled_utc,
                    )
                )

    def startup_backpopulate_precision(self) -> None:
        trades = Trade.get_trades([Trade.contract_size.is_(None)])
        for trade in trades:
            if trade.exchange != self.exchange.id:
                continue
            trade.precision_mode = self.exchange.precisionMode
            trade.precision_mode_price = self.exchange.precision_mode_price
            trade.amount_precision = self.exchange.get_precision_amount(trade.pair)
            trade.price_precision = self.exchange.get_precision_price(trade.pair)
            trade.contract_size = self.exchange.get_contract_size(trade.pair)
        Trade.commit()

    def startup_update_open_orders(self):
        """
        根据数据库中保留的订单列表更新未平仓订单。
        主要更新订单的状态 - 但也可能平仓交易
        """
        if self.config["dry_run"] or self.config["exchange"].get("skip_open_order_update", False):
            # 在干运行中更新未平仓订单没有意义，并且会失败。
            return

        orders = Order.get_open_orders()
        logger.info(f"正在更新 {len(orders)} 个未平仓订单。")
        for order in orders:
            try:
                fo = self.exchange.fetch_order_or_stoploss_order(
                    order.order_id, order.ft_pair, order.ft_order_side == "stoploss"
                )
                if not order.trade:
                    # 这不应该发生，但如果交易被手动删除，则会发生。
                    # 这只能在 sqlite 上发生，它不强制执行外键约束。
                    logger.warning(
                        f"订单 {order.order_id} 没有关联交易。"
                        "这可能表明数据库已损坏。"
                        f"预期的交易 ID 是 {order.ft_trade_id}。忽略此订单。"
                    )
                    continue
                self.update_trade_state(
                    order.trade,
                    order.order_id,
                    fo,
                    stoploss_order=(order.ft_order_side == "stoploss"),
                )

            except InvalidOrderException as e:
                logger.warning(f"更新订单 {order.order_id} 失败，原因：{e}。")
                if order.order_date_utc - timedelta(days=5) < datetime.now(timezone.utc):
                    logger.warning(
                        "订单已超过 5 天。假设订单已完全取消。"
                    )
                    fo = order.to_ccxt_object()
                    fo["status"] = "canceled"
                    self.handle_cancel_order(
                        fo, order, order.trade, constants.CANCEL_REASON["TIMEOUT"]
                    )

            except ExchangeError as e:
                logger.warning(f"更新订单 {order.order_id} 失败，原因：{e}")

    def update_trades_without_assigned_fees(self) -> None:
        """
        更新未分配平仓费用的已平仓交易。
        仅当数据库中有订单时才执行操作，否则上次订单 ID 未知。
        """
        if self.config["dry_run"]:
            # 在干运行中更新未平仓订单没有意义，并且会失败。
            return

        trades: list[Trade] = Trade.get_closed_trades_without_assigned_fees()
        for trade in trades:
            if not trade.is_open and not trade.fee_updated(trade.exit_side):
                # 获取卖出费用
                order = trade.select_order(trade.exit_side, False, only_filled=True)
                if not order:
                    order = trade.select_order("stoploss", False)
                if order:
                    logger.info(
                        f"正在更新交易 {trade} 的 {trade.exit_side} 费用，"
                        f"订单号为 {order.order_id}。"
                    )
                    self.update_trade_state(
                        trade,
                        order.order_id,
                        stoploss_order=order.ft_order_side == "stoploss",
                        send_msg=False,
                    )

        trades = Trade.get_open_trades_without_assigned_fees()
        for trade in trades:
            with self._exit_lock:
                if trade.is_open and not trade.fee_updated(trade.entry_side):
                    order = trade.select_order(trade.entry_side, False, only_filled=True)
                    open_order = trade.select_order(trade.entry_side, True)
                    if order and open_order is None:
                        logger.info(
                            f"正在更新交易 {trade} 的 {trade.entry_side} 费用，"
                            f"订单号为 {order.order_id}。"
                        )
                        self.update_trade_state(trade, order.order_id, send_msg=False)

    def handle_insufficient_funds(self, trade: Trade):
        """
        尝试找回丢失的交易。
        仅当止损订单（止损或多头卖出/空头买入）出现资金不足时使用。
        尝试遍历存储的订单并根据需要更新交易状态。
        """
        logger.info(f"正在尝试找回 {trade} 的丢失订单")
        for order in trade.orders:
            logger.info(f"正在尝试找回 {order}")
            fo = None
            if not order.ft_is_open:
                logger.debug(f"订单 {order} 已不再是未平仓状态。")
                continue
            try:
                fo = self.exchange.fetch_order_or_stoploss_order(
                    order.order_id, order.ft_pair, order.ft_order_side == "stoploss"
                )
                if fo:
                    logger.info(f"找到了交易 {trade} 的 {order}。")
                    self.update_trade_state(
                        trade, order.order_id, fo, stoploss_order=order.ft_order_side == "stoploss"
                    )

            except ExchangeError:
                logger.warning(f"更新 {order.order_id} 失败。")

    def handle_onexchange_order(self, trade: Trade) -> bool:
        """
        尝试找回数据库中没有的订单。
        仅当余额消失时使用，这将导致无法退出。
        :return: 如果交易被删除，则为 True，否则为 False
        """
        try:
            orders = self.exchange.fetch_orders(
                trade.pair, trade.open_date_utc - timedelta(seconds=10)
            )
            prev_exit_reason = trade.exit_reason
            prev_trade_state = trade.is_open
            prev_trade_amount = trade.amount
            for order in orders:
                trade_order = [o for o in trade.orders if o.order_id == order["id"]]

                if trade_order:
                    # 我们知道这个订单，但没有正确更新它
                    order_obj = trade_order[0]
                else:
                    logger.info(f"找到了 {trade.pair} 的先前未知订单 {order['id']}。")

                    order_obj = Order.parse_from_ccxt_object(order, trade.pair, order["side"])
                    order_obj.order_filled_date = dt_from_ts(
                        safe_value_fallback(order, "lastTradeTimestamp", "timestamp")
                    )
                    trade.orders.append(order_obj)
                    Trade.commit()
                    trade.exit_reason = ExitType.SOLD_ON_EXCHANGE.value

                self.update_trade_state(trade, order["id"], order, send_msg=False)

                logger.info(f"已处理订单 {order['id']}")

            # 从数据库刷新交易
            Trade.session.refresh(trade)
            if not trade.is_open:
                # 交易刚刚平仓
                trade.close_date = trade.date_last_filled_utc
                self.order_close_notify(
                    trade,
                    order_obj,
                    order_obj.ft_order_side == "stoploss",
                    send_msg=prev_trade_state != trade.is_open,
                )
            else:
                trade.exit_reason = prev_exit_reason
                total = (
                    self.wallets.get_owned(trade.pair, trade.base_currency)
                    if trade.base_currency
                    else 0
                )
                if total < trade.amount:
                    if trade.fully_canceled_entry_order_count == len(trade.orders):
                        logger.warning(
                            f"交易只有完全取消的入场订单。"
                            f"正在从数据库中移除 {trade}。"
                        )

                        self._notify_enter_cancel(
                            trade,
                            order_type=self.strategy.order_types["entry"],
                            reason=constants.CANCEL_REASON["FULLY_CANCELLED"],
                        )
                        trade.delete()
                        return True
                    if total > trade.amount * 0.98:
                        logger.warning(
                            f"{trade} 的总 {trade.base_currency} 为 {trade.amount}，"
                            f"但钱包显示总计 {trade.base_currency} 为 {total}。"
                            f"正在将交易金额调整为 {total}。"
                            "但这可能会导致进一步的问题。"
                        )
                        trade.amount = total
                    else:
                        logger.warning(
                            f"{trade} 的总 {trade.base_currency} 为 {trade.amount}，"
                            f"但钱包显示总计 {trade.base_currency} 为 {total}。"
                            "拒绝调整，因为差异过大。"
                            "但这可能会导致进一步的问题。"
                        )
                if prev_trade_amount != trade.amount:
                    # 如果金额发生变化，则取消交易所上的止损
                    trade = self.cancel_stoploss_on_exchange(trade)
            Trade.commit()

        except ExchangeError:
            logger.warning("查找交易所订单失败。")
        except Exception:
            # 捕获 https://github.com/freqtrade/freqtrade/issues/9025
            logger.warning("查找交易所订单失败", exc_info=True)
        return False

    #
    # 进入头寸 / 开仓交易逻辑和方法
    #

    def enter_positions(self) -> int:
        """
        尝试为新交易（头寸）执行入场订单
        """
        trades_created = 0

        whitelist = deepcopy(self.active_pair_whitelist)
        if not whitelist:
            self.log_once("活跃交易对白名单为空。", logger.info)
            return trades_created
        # 从白名单中移除当前已开仓交易的交易对
        for trade in Trade.get_open_trades():
            if trade.pair in whitelist:
                whitelist.remove(trade.pair)
                logger.debug("忽略交易对白名单中的 %s", trade.pair)

        if not whitelist:
            self.log_once(
                "活跃交易对白名单中没有货币对，但正在检查是否退出未平仓交易。",
                logger.info,
            )
            return trades_created
        if PairLocks.is_global_lock(side="*"):
            # 这只检查总锁定（双向）。
            # 每侧锁定将在 create_trade 中由 `is_pair_locked` 进行评估，
            # 一旦交易方向明确。
            lock = PairLocks.get_pair_longest_lock("*")
            if lock:
                self.log_once(
                    f"全局交易对锁定活跃，直到 "
                    f"{lock.lock_end_time.strftime(constants.DATETIME_PRINT_FORMAT)}。 "
                    f"不创建新交易，原因：{lock.reason}。",
                    logger.info,
                )
            else:
                self.log_once("全局交易对锁定活跃。不创建新交易。", logger.info)
            return trades_created
        # 为白名单中的每个交易对创建实体并执行交易
        for pair in whitelist:
            try:
                with self._exit_lock:
                    trades_created += self.create_trade(pair)
            except DependencyException as exception:
                logger.warning("无法为 %s 创建交易: %s", pair, exception)

        if not trades_created:
            logger.debug("未找到白名单货币的入场信号。正在重试...")

        return trades_created

    def create_trade(self, pair: str) -> bool:
        """
        检查已实施的交易策略以获取入场信号。

        如果交易对触发入场信号，则创建新的交易记录，
        并向交易所发出开仓订单以开仓交易。

        :return: 如果已创建交易，则为 True。
        """
        logger.debug(f"为交易对 {pair} 创建交易")

        analyzed_df, _ = self.dataprovider.get_analyzed_dataframe(pair, self.strategy.timeframe)
        nowtime = analyzed_df.iloc[-1]["date"] if len(analyzed_df) > 0 else None

        # create_trade 被调用前检查 get_free_open_trades
        # 但它仍在此处用于防止在一次迭代中开立过多交易
        if not self.get_free_open_trades():
            logger.debug(f"无法为 {pair} 开新交易：已达到最大交易数。")
            return False

        # 在获取的历史数据上运行 get_signal
        (signal, enter_tag) = self.strategy.get_entry_signal(
            pair, self.strategy.timeframe, analyzed_df
        )

        if signal:
            if self.strategy.is_pair_locked(pair, candle_date=nowtime, side=signal):
                lock = PairLocks.get_pair_longest_lock(pair, nowtime, signal)
                if lock:
                    self.log_once(
                        f"交易对 {pair} {lock.side} 被锁定直到 "
                        f"{lock.lock_end_time.strftime(constants.DATETIME_PRINT_FORMAT)} "
                        f"原因：{lock.reason}。",
                        logger.info,
                    )
                else:
                    self.log_once(f"交易对 {pair} 当前被锁定。", logger.info)
                return False
            stake_amount = self.wallets.get_trade_stake_amount(pair, self.config["max_open_trades"])

            bid_check_dom = self.config.get("entry_pricing", {}).get("check_depth_of_market", {})
            if (bid_check_dom.get("enabled", False)) and (
                bid_check_dom.get("bids_to_ask_delta", 0) > 0
            ):
                if self._check_depth_of_market(pair, bid_check_dom, side=signal):
                    return self.execute_entry(
                        pair,
                        stake_amount,
                        enter_tag=enter_tag,
                        is_short=(signal == SignalDirection.SHORT),
                    )
                else:
                    return False

            return self.execute_entry(
                pair, stake_amount, enter_tag=enter_tag, is_short=(signal == SignalDirection.SHORT)
            )
        else:
            return False

    #
    # 修改头寸 / DCA 逻辑和方法
    #
    def process_open_trade_positions(self):
        """
        尝试为未平仓交易（头寸）执行额外的买入或卖出订单
        """
        # 遍历每个交易对并检查是否需要更改
        for trade in Trade.get_open_trades():
            # 如果有任何未平仓订单，等待它们完成。
            # TODO：移除以允许多个未平仓订单
            if trade.has_open_position or trade.has_open_orders:
                # 更新钱包（将限制为每小时一次）
                self.wallets.update(False)
                try:
                    self.check_and_call_adjust_trade_position(trade)
                except DependencyException as exception:
                    logger.warning(
                        f"无法调整交易 {trade.pair} 的头寸: {exception}"
                    )

    def check_and_call_adjust_trade_position(self, trade: Trade):
        """
        检查已实施的交易策略以获取调整命令。
        如果策略触发调整，则发出新订单。
        一旦完成，现有交易将被修改以匹配新数据。
        """
        current_entry_rate, current_exit_rate = self.exchange.get_rates(
            trade.pair, True, trade.is_short
        )

        current_entry_profit = trade.calc_profit_ratio(current_entry_rate)
        current_exit_profit = trade.calc_profit_ratio(current_exit_rate)

        min_entry_stake = self.exchange.get_min_pair_stake_amount(
            trade.pair, current_entry_rate, 0.0, trade.leverage
        )
        min_exit_stake = self.exchange.get_min_pair_stake_amount(
            trade.pair, current_exit_rate, self.strategy.stoploss, trade.leverage
        )
        max_entry_stake = self.exchange.get_max_pair_stake_amount(
            trade.pair, current_entry_rate, trade.leverage
        )
        stake_available = self.wallets.get_available_stake_amount()
        logger.debug(f"正在调用 adjust_trade_position，交易对为 {trade.pair}")
        stake_amount, order_tag = self.strategy._adjust_trade_position_internal(
            trade=trade,
            current_time=datetime.now(timezone.utc),
            current_rate=current_entry_rate,
            current_profit=current_entry_profit,
            min_stake=min_entry_stake,
            max_stake=min(max_entry_stake, stake_available),
            current_entry_rate=current_entry_rate,
            current_exit_rate=current_exit_rate,
            current_entry_profit=current_entry_profit,
            current_exit_profit=current_exit_profit,
        )

        if stake_amount is not None and stake_amount > 0.0:
            if self.state == State.PAUSED:
                logger.debug("由于机器人处于 PAUSED 状态，头寸调整已中止")
                return

            # 我们应该增加头寸
            if self.strategy.max_entry_position_adjustment > -1:
                count_of_entries = trade.nr_of_successful_entries
                if count_of_entries > self.strategy.max_entry_position_adjustment:
                    logger.debug(f"已达到 {trade.pair} 的最大调整入场次数。")
                    return
                else:
                    logger.debug("最大调整入场次数设置为无限制。")

            self.execute_entry(
                trade.pair,
                stake_amount,
                price=current_entry_rate,
                trade=trade,
                is_short=trade.is_short,
                mode="pos_adjust",
                enter_tag=order_tag,
            )

        if stake_amount is not None and stake_amount < 0.0:
            # 我们应该减少头寸
            amount = self.exchange.amount_to_contract_precision(
                trade.pair,
                abs(
                    float(
                        FtPrecise(stake_amount)
                        * FtPrecise(trade.amount)
                        / FtPrecise(trade.stake_amount)
                    )
                ),
            )

            if amount == 0.0:
                logger.info(
                    f"想要退出 {stake_amount} 的金额，"
                    "但由于交易所限制，退出金额现在为 0.0 - 不退出。"
                )
                return

            remaining = (trade.amount - amount) * current_exit_rate
            if min_exit_stake and remaining != 0 and remaining < min_exit_stake:
                logger.info(
                    f"剩余金额 {remaining} 将小于 "
                    f"最小金额 {min_exit_stake}。"
                )
                return

            self.execute_trade_exit(
                trade,
                current_exit_rate,
                exit_check=ExitCheckTuple(exit_type=ExitType.PARTIAL_EXIT),
                sub_trade_amt=amount,
                exit_tag=order_tag,
            )

    def _check_depth_of_market(self, pair: str, conf: dict, side: SignalDirection) -> bool:
        """
        在执行入场前检查市场深度
        """
        conf_bids_to_ask_delta = conf.get("bids_to_ask_delta", 0)
        logger.info(f"正在检查 {pair} 的市场深度...")
        order_book = self.exchange.fetch_l2_order_book(pair, 1000)
        order_book_data_frame = order_book_to_dataframe(order_book["bids"], order_book["asks"])
        order_book_bids = order_book_data_frame["b_size"].sum()
        order_book_asks = order_book_data_frame["a_size"].sum()

        entry_side = order_book_bids if side == SignalDirection.LONG else order_book_asks
        exit_side = order_book_asks if side == SignalDirection.LONG else order_book_bids
        bids_ask_delta = entry_side / exit_side

        bids = f"买盘: {order_book_bids}"
        asks = f"卖盘: {order_book_asks}"
        delta = f"Delta: {bids_ask_delta}"

        logger.info(
            f"{bids}, {asks}, {delta}, 方向: {side.value} "
            f"买入价: {order_book['bids'][0][0]}, 卖出价: {order_book['asks'][0][0]}, "
            f"即时买入量: {order_book['bids'][0][1]}, "
            f"即时卖出量: {order_book['asks'][0][1]}。"
        )
        if bids_ask_delta >= conf_bids_to_ask_delta:
            logger.info(f"{pair} 的买卖盘差 DOES 满足条件。")
            return True
        else:
            logger.info(f"{pair} 的买卖盘差不满足条件。")
            return False

    def execute_entry(
        self,
        pair: str,
        stake_amount: float,
        price: float | None = None,
        *,
        is_short: bool = False,
        ordertype: str | None = None,
        enter_tag: str | None = None,
        trade: Trade | None = None,
        mode: EntryExecuteMode = "initial",
        leverage_: float | None = None,
    ) -> bool:
        """
        为给定交易对执行入场操作
        :param pair: 我们要创建 LIMIT 订单的交易对
        :param stake_amount: 交易对的质押金额
        :return: 如果已创建入场订单，则为 True，如果失败，则为 False。
        :raise: DependencyException 或其子类，如 ExchangeError。
        """
        time_in_force = self.strategy.order_time_in_force["entry"]

        side: BuySell = "sell" if is_short else "buy"
        name = "空头" if is_short else "多头"
        trade_side: LongShort = "short" if is_short else "long"
        pos_adjust = trade is not None

        enter_limit_requested, stake_amount, leverage = self.get_valid_enter_price_and_stake(
            pair, price, stake_amount, trade_side, enter_tag, trade, mode, leverage_
        )

        if not stake_amount:
            return False

        msg = (
            f"头寸调整：正在为 {pair} 创建新订单，质押金额： "
            f"{stake_amount}，价格：{enter_limit_requested}，针对交易：{trade}"
            if mode == "pos_adjust"
            else (
                f"正在替换 {side} 订单：正在为 {pair} 创建新订单，质押金额： "
                f"{stake_amount}，价格：{enter_limit_requested} ..."
                if mode == "replace"
                else f"{name} 信号已找到：正在为 {pair} 创建新交易，质押金额： "
                f"{stake_amount}，价格：{enter_limit_requested} ..."
            )
        )
        logger.info(msg)
        amount = (stake_amount / enter_limit_requested) * leverage
        order_type = ordertype or self.strategy.order_types["entry"]

        if mode == "initial" and not strategy_safe_wrapper(
            self.strategy.confirm_trade_entry, default_retval=True
        )(
            pair=pair,
            order_type=order_type,
            amount=amount,
            rate=enter_limit_requested,
            time_in_force=time_in_force,
            current_time=datetime.now(timezone.utc),
            entry_tag=enter_tag,
            side=trade_side,
        ):
            logger.info(f"用户拒绝 {pair} 的入场。")
            return False

        if trade and self.handle_similar_open_order(trade, enter_limit_requested, amount, side):
            return False

        order = self.exchange.create_order(
            pair=pair,
            ordertype=order_type,
            side=side,
            amount=amount,
            rate=enter_limit_requested,
            reduceOnly=False,
            time_in_force=time_in_force,
            leverage=leverage,
        )
        order_obj = Order.parse_from_ccxt_object(order, pair, side, amount, enter_limit_requested)
        order_obj.ft_order_tag = enter_tag
        order_id = order["id"]
        order_status = order.get("status")
        logger.info(f"订单 {order_id} 已为 {pair} 创建，状态为 {order_status}。")

        # 我们假设订单以请求的价格执行
        enter_limit_filled_price = enter_limit_requested
        amount_requested = amount

        if order_status == "expired" or order_status == "rejected":
            # 如果订单未成交，则返回 false
            if float(order["filled"]) == 0:
                logger.warning(
                    f"{name} {time_in_force} 订单，强制时间 {order_type} "
                    f"针对 {pair}，被 {self.exchange.name} {order_status}。"
                    "成交金额为零。"
                )
                return False
            else:
                # 订单部分成交
                # 对于 IOC 订单，我们可以立即检查
                # 订单是否完全或部分成交
                logger.warning(
                    "%s %s 订单，强制时间 %s 针对 %s，被 %s %s。"
                    "%s 成交金额占 %s (剩余 %s 已取消)。",
                    name,
                    time_in_force,
                    order_type,
                    pair,
                    order_status,
                    self.exchange.name,
                    order["filled"],
                    order["amount"],
                    order["remaining"],
                )
                amount = safe_value_fallback(order, "filled", "amount", amount)
                enter_limit_filled_price = safe_value_fallback(
                    order, "average", "price", enter_limit_filled_price
                )

        # 在 FOK 的情况下，订单可能会立即完全成交
        elif order_status == "closed":
            amount = safe_value_fallback(order, "filled", "amount", amount)
            enter_limit_filled_price = safe_value_fallback(
                order, "average", "price", enter_limit_requested
            )

        # 费用被应用两次，因为我们执行了 LIMIT_BUY 和 LIMIT_SELL
        fee = self.exchange.get_fee(symbol=pair, taker_or_maker="maker")
        base_currency = self.exchange.get_pair_base_currency(pair)
        open_date = datetime.now(timezone.utc)

        funding_fees = self.exchange.get_funding_fees(
            pair=pair,
            amount=amount + trade.amount if trade else amount,
            is_short=is_short,
            open_date=trade.date_last_filled_utc if trade else open_date,
        )

        # 这是一个新交易
        if trade is None:
            trade = Trade(
                pair=pair,
                base_currency=base_currency,
                stake_currency=self.config["stake_currency"],
                stake_amount=stake_amount,
                amount=0,
                is_open=True,
                amount_requested=amount_requested,
                fee_open=fee,
                fee_close=fee,
                open_rate=enter_limit_filled_price,
                open_rate_requested=enter_limit_requested,
                open_date=open_date,
                exchange=self.exchange.id,
                strategy=self.strategy.get_strategy_name(),
                enter_tag=enter_tag,
                timeframe=timeframe_to_minutes(self.config["timeframe"]),
                leverage=leverage,
                is_short=is_short,
                trading_mode=self.trading_mode,
                funding_fees=funding_fees,
                amount_precision=self.exchange.get_precision_amount(pair),
                price_precision=self.exchange.get_precision_price(pair),
                precision_mode=self.exchange.precisionMode,
                precision_mode_price=self.exchange.precision_mode_price,
                contract_size=self.exchange.get_contract_size(pair),
            )
            stoploss = self.strategy.stoploss
            trade.adjust_stop_loss(trade.open_rate, stoploss, initial=True)

        else:
            trade.is_open = True
            trade.set_funding_fees(funding_fees)

        trade.orders.append(order_obj)
        trade.recalc_trade_from_orders()
        Trade.session.add(trade)
        Trade.commit()

        # 更新钱包
        self.wallets.update()

        self._notify_enter(trade, order_obj, order_type, sub_trade=pos_adjust)

        if pos_adjust:
            if order_status == "closed":
                logger.info(f"DCA 订单已平仓，交易应已更新：{trade}")
                trade = self.cancel_stoploss_on_exchange(trade)
            else:
                logger.info(f"DCA 订单 {order_status}，将等待解决：{trade}")

        # 如果订单未开仓，则更新费用
        if order_status in constants.NON_OPEN_EXCHANGE_STATES:
            fully_canceled = self.update_trade_state(trade, order_id, order)
            if fully_canceled and mode != "replace":
                # 完全取消的订单，可能发生在某些强制时间设置（IOC）中。
                # 应立即处理。
                self.handle_cancel_enter(
                    trade, order, order_obj, constants.CANCEL_REASON["TIMEOUT"]
                )

        return True

    def cancel_stoploss_on_exchange(self, trade: Trade) -> Trade:
        # 首先取消交易所上的止损...
        for oslo in trade.open_sl_orders:
            try:
                logger.info(f"正在取消交易 {trade} 的交易所止损订单：{oslo.order_id}")
                co = self.exchange.cancel_stoploss_order_with_result(
                    oslo.order_id, trade.pair, trade.amount
                )
                self.update_trade_state(trade, oslo.order_id, co, stoploss_order=True)
            except InvalidOrderException:
                logger.exception(
                    f"无法取消交易对 {trade.pair} 的止损订单 {oslo.order_id}"
                )
        return trade

    def get_valid_enter_price_and_stake(
        self,
        pair: str,
        price: float | None,
        stake_amount: float,
        trade_side: LongShort,
        entry_tag: str | None,
        trade: Trade | None,
        mode: EntryExecuteMode,
        leverage_: float | None,
    ) -> tuple[float, float, float]:
        """
        验证并最终调整（在限制范围内）限价、金额和杠杆
        :return: 包含 (价格, 金额, 杠杆) 的元组
        """

        if price:
            enter_limit_requested = price
        else:
            # 计算价格
            enter_limit_requested = self.exchange.get_rate(
                pair, side="entry", is_short=(trade_side == "short"), refresh=True
            )
        if mode != "replace":
            # 在订单调整场景中不调用 custom_entry_price
            custom_entry_price = strategy_safe_wrapper(
                self.strategy.custom_entry_price, default_retval=enter_limit_requested
            )(
                pair=pair,
                trade=trade,
                current_time=datetime.now(timezone.utc),
                proposed_rate=enter_limit_requested,
                entry_tag=entry_tag,
                side=trade_side,
            )

            enter_limit_requested = self.get_valid_price(custom_entry_price, enter_limit_requested)

        if not enter_limit_requested:
            raise PricingError("无法确定入场价格。")

        if self.trading_mode != TradingMode.SPOT and trade is None:
            max_leverage = self.exchange.get_max_leverage(pair, stake_amount)
            if leverage_:
                leverage = leverage_
            else:
                leverage = strategy_safe_wrapper(self.strategy.leverage, default_retval=1.0)(
                    pair=pair,
                    current_time=datetime.now(timezone.utc),
                    current_rate=enter_limit_requested,
                    proposed_leverage=1.0,
                    max_leverage=max_leverage,
                    side=trade_side,
                    entry_tag=entry_tag,
                )
            # 将杠杆限制在 1.0 和 max_leverage 之间。
            leverage = min(max(leverage, 1.0), max_leverage)
        else:
            # 当前无法更改杠杆
            leverage = trade.leverage if trade else 1.0

        # 最小质押金额实际上应该包括杠杆 - 这样我们的“最小”
        # 质押金额可能高于实际需要。
        # 但是，我们还需要最小质押金额来确定杠杆，因此目前将其视为
        # 边缘情况而忽略。
        min_stake_amount = self.exchange.get_min_pair_stake_amount(
            pair,
            enter_limit_requested,
            self.strategy.stoploss if not mode == "pos_adjust" else 0.0,
            leverage,
        )
        max_stake_amount = self.exchange.get_max_pair_stake_amount(
            pair, enter_limit_requested, leverage
        )

        if trade is None:
            stake_available = self.wallets.get_available_stake_amount()
            stake_amount = strategy_safe_wrapper(
                self.strategy.custom_stake_amount, default_retval=stake_amount
            )(
                pair=pair,
                current_time=datetime.now(timezone.utc),
                current_rate=enter_limit_requested,
                proposed_stake=stake_amount,
                min_stake=min_stake_amount,
                max_stake=min(max_stake_amount, stake_available),
                leverage=leverage,
                entry_tag=entry_tag,
                side=trade_side,
            )

        stake_amount = self.wallets.validate_stake_amount(
            pair=pair,
            stake_amount=stake_amount,
            min_stake_amount=min_stake_amount,
            max_stake_amount=max_stake_amount,
            trade_amount=trade.stake_amount if trade else None,
        )

        return enter_limit_requested, stake_amount, leverage

    def _notify_enter(
        self,
        trade: Trade,
        order: Order,
        order_type: str | None,
        fill: bool = False,
        sub_trade: bool = False,
    ) -> None:
        """
        当入场订单发生时发送 RPC 通知。
        """
        open_rate = order.safe_price

        if open_rate is None:
            open_rate = trade.open_rate

        current_rate = self.exchange.get_rate(
            trade.pair, side="entry", is_short=trade.is_short, refresh=False
        )
        stake_amount = trade.stake_amount
        if not fill and trade.nr_of_successful_entries > 0:
            # 如果我们有未平仓订单，我们需要添加未平仓订单的质押金额
            # 因为它尚未包含在 trade.stake_amount 中
            stake_amount += sum(
                o.stake_amount for o in trade.open_orders if o.ft_order_side == trade.entry_side
            )

        msg: RPCEntryMsg = {
            "trade_id": trade.id,
            "type": RPCMessageType.ENTRY_FILL if fill else RPCMessageType.ENTRY,
            "buy_tag": trade.enter_tag,
            "enter_tag": trade.enter_tag,
            "exchange": trade.exchange.capitalize(),
            "pair": trade.pair,
            "leverage": trade.leverage if trade.leverage else None,
            "direction": "Short" if trade.is_short else "Long",
            "limit": open_rate,  # 已弃用 (?)
            "open_rate": open_rate,
            "order_type": order_type or "unknown",
            "stake_amount": stake_amount,
            "stake_currency": self.config["stake_currency"],
            "base_currency": self.exchange.get_pair_base_currency(trade.pair),
            "quote_currency": self.exchange.get_pair_quote_currency(trade.pair),
            "fiat_currency": self.config.get("fiat_display_currency", None),
            "amount": order.safe_amount_after_fee if fill else (order.safe_amount or trade.amount),
            "open_date": trade.open_date_utc or datetime.now(timezone.utc),
            "current_rate": current_rate,
            "sub_trade": sub_trade,
        }

        # 发送消息
        self.rpc.send_msg(msg)

    def _notify_enter_cancel(
        self, trade: Trade, order_type: str, reason: str, sub_trade: bool = False
    ) -> None:
        """
        当入场订单取消时发送 RPC 通知。
        """
        current_rate = self.exchange.get_rate(
            trade.pair, side="entry", is_short=trade.is_short, refresh=False
        )

        msg: RPCCancelMsg = {
            "trade_id": trade.id,
            "type": RPCMessageType.ENTRY_CANCEL,
            "buy_tag": trade.enter_tag,
            "enter_tag": trade.enter_tag,
            "exchange": trade.exchange.capitalize(),
            "pair": trade.pair,
            "leverage": trade.leverage,
            "direction": "Short" if trade.is_short else "Long",
            "limit": trade.open_rate,
            "order_type": order_type,
            "stake_amount": trade.stake_amount,
            "open_rate": trade.open_rate,
            "stake_currency": self.config["stake_currency"],
            "base_currency": self.exchange.get_pair_base_currency(trade.pair),
            "quote_currency": self.exchange.get_pair_quote_currency(trade.pair),
            "fiat_currency": self.config.get("fiat_display_currency", None),
            "amount": trade.amount,
            "open_date": trade.open_date,
            "current_rate": current_rate,
            "reason": reason,
            "sub_trade": sub_trade,
        }

        # 发送消息
        self.rpc.send_msg(msg)

    #
    # 卖出 / 退出头寸 / 平仓交易逻辑和方法
    #

    def exit_positions(self, trades: list[Trade]) -> int:
        """
        尝试为未平仓交易（头寸）执行退出订单
        """
        trades_closed = 0
        for trade in trades:
            if (
                not trade.has_open_orders
                and not trade.has_open_sl_orders
                and trade.fee_open_currency is not None
                and not self.wallets.check_exit_amount(trade)
            ):
                logger.warning(
                    f"钱包中没有足够的 {trade.safe_base_currency} 来退出 {trade}。 "
                    "正在尝试恢复。"
                )
                if self.handle_onexchange_order(trade):
                    # 交易被删除。不要继续。
                    continue

            try:
                try:
                    if self.strategy.order_types.get(
                        "stoploss_on_exchange"
                    ) and self.handle_stoploss_on_exchange(trade):
                        trades_closed += 1
                        Trade.commit()
                        continue

                except InvalidOrderException as exception:
                    logger.warning(
                        f"无法处理 {trade.pair} 的交易所止损：{exception}"
                    )
                # 检查我们是否可以退出此交易的当前头寸
                if trade.has_open_position and trade.is_open and self.handle_trade(trade):
                    trades_closed += 1

            except DependencyException as exception:
                logger.warning(f"无法退出交易 {trade.pair}: {exception}")

        # 如果发生任何交易，则更新钱包
        if trades_closed:
            self.wallets.update()

        return trades_closed

    def handle_trade(self, trade: Trade) -> bool:
        """
        如果达到阈值，则退出当前交易对并更新交易记录。
        :return: 如果交易已卖出/空头退出，则为 True，否则为 False
        """
        if not trade.is_open:
            raise DependencyException(f"尝试处理已平仓交易：{trade}")

        logger.debug("正在处理 %s ...", trade)

        (enter, exit_) = (False, False)
        exit_tag = None
        exit_signal_type = "空头退出" if trade.is_short else "多头退出"

        if self.config.get("use_exit_signal", True) or self.config.get(
            "ignore_roi_if_entry_signal", False
        ):
            analyzed_df, _ = self.dataprovider.get_analyzed_dataframe(
                trade.pair, self.strategy.timeframe
            )

            (enter, exit_, exit_tag) = self.strategy.get_exit_signal(
                trade.pair, self.strategy.timeframe, analyzed_df, is_short=trade.is_short
            )

        logger.debug("正在检查退出")
        exit_rate = self.exchange.get_rate(
            trade.pair, side="exit", is_short=trade.is_short, refresh=True
        )
        if self._check_and_execute_exit(trade, exit_rate, enter, exit_, exit_tag):
            return True

        logger.debug(f"未找到 {exit_signal_type} 信号，交易对为 %s。", trade)
        return False

    def _check_and_execute_exit(
        self, trade: Trade, exit_rate: float, enter: bool, exit_: bool, exit_tag: str | None
    ) -> bool:
        """
        检查并执行交易退出
        """
        exits: list[ExitCheckTuple] = self.strategy.should_exit(
            trade,
            exit_rate,
            datetime.now(timezone.utc),
            enter=enter,
            exit_=exit_,
            force_stoploss=0,
        )
        for should_exit in exits:
            if should_exit.exit_flag:
                exit_tag1 = exit_tag if should_exit.exit_type == ExitType.EXIT_SIGNAL else None
                if trade.has_open_orders:
                    if prev_eval := self._exit_reason_cache.get(
                        f"{trade.pair}_{trade.id}_{exit_tag1 or should_exit.exit_reason}", None
                    ):
                        logger.debug(
                            f"此蜡烛图已见到退出原因，首次见于 {prev_eval}"
                        )
                        continue

                logger.info(
                    f"检测到 {trade.pair} 的退出。原因：{should_exit.exit_type}"
                    f"{f' 标签: {exit_tag1}' if exit_tag1 is not None else ''}"
                )
                exited = self.execute_trade_exit(trade, exit_rate, should_exit, exit_tag=exit_tag1)
                if exited:
                    return True
        return False

    def create_stoploss_order(self, trade: Trade, stop_price: float) -> bool:
        """
        从逻辑中抽象创建止损订单。
        处理错误并更新交易数据库对象。
        如果创建订单出现问题，则强制卖出交易对（使用 EmergencySell 原因）。
        :return: 如果订单成功，则为 True，如果出现问题，则为 False。
        """
        try:
            stoploss_order = self.exchange.create_stoploss(
                pair=trade.pair,
                amount=trade.amount,
                stop_price=stop_price,
                order_types=self.strategy.order_types,
                side=trade.exit_side,
                leverage=trade.leverage,
            )

            order_obj = Order.parse_from_ccxt_object(
                stoploss_order, trade.pair, "stoploss", trade.amount, stop_price
            )
            trade.orders.append(order_obj)
            return True
        except InsufficientFundsError as e:
            logger.warning(f"无法下止损订单 {e}。")
            # 尝试找出问题所在
            self.handle_insufficient_funds(trade)

        except InvalidOrderException as e:
            logger.error(f"无法在交易所下止损订单。{e}")
            logger.warning("正在强制平仓")
            self.emergency_exit(trade, stop_price)

        except ExchangeError:
            logger.exception("无法在交易所下止损订单。")
        return False

    def handle_stoploss_on_exchange(self, trade: Trade) -> bool:
        """
        检查交易是否已完成，如果是，则应立即添加交易所止损
        如果交易所止损已启用。
        # TODO: 强平价格始终在交易所，即使没有止损
        # 因此获取未平仓交易对的账户强平信息可能是有意义的。
        """

        logger.debug("正在处理交易所止损 %s ...", trade)

        stoploss_orders = []
        for slo in trade.open_sl_orders:
            stoploss_order = None
            try:
                # 首先我们检查是否已存在交易所止损
                stoploss_order = (
                    self.exchange.fetch_stoploss_order(slo.order_id, trade.pair)
                    if slo.order_id
                    else None
                )
            except InvalidOrderException as exception:
                logger.warning("无法获取止损订单：%s", exception)

            if stoploss_order:
                stoploss_orders.append(stoploss_order)
                self.update_trade_state(trade, slo.order_id, stoploss_order, stoploss_order=True)

            # 我们检查止损订单是否已完成
            if stoploss_order and stoploss_order["status"] in ("closed", "triggered"):
                trade.exit_reason = ExitType.STOPLOSS_ON_EXCHANGE.value
                self._notify_exit(trade, "stoploss", True)
                self.handle_protections(trade.pair, trade.trade_direction)
                return True

        if (
            not trade.has_open_position
            or not trade.is_open
            or (trade.has_open_orders and self.exchange.get_option("stoploss_blocks_assets", True))
        ):
            # 交易可能已经平仓（本次迭代中收到了卖出订单成交确认）
            return False

        # 如果入场订单已完成但没有止损，我们添加交易所止损
        if len(stoploss_orders) == 0:
            stop_price = trade.stoploss_or_liquidation

            if self.create_stoploss_order(trade=trade, stop_price=stop_price):
                # 如果放置失败并且交易被强制卖出，则上述将返回 False。
                # 在这种情况下，交易将被平仓 - 我们必须在下面进行检查。
                return False

        self.manage_trade_stoploss_orders(trade, stoploss_orders)

        return False

    def handle_trailing_stoploss_on_exchange(self, trade: Trade, order: CcxtOrder) -> None:
        """
        检查是否应更新交易所上的追踪止损
        在追踪止损的情况下
        :param trade: 相应的交易
        :param order: 当前交易所上的止损订单
        :return: 无
        """
        stoploss_norm = self.exchange.price_to_precision(
            trade.pair,
            trade.stoploss_or_liquidation,
            rounding_mode=ROUND_DOWN if trade.is_short else ROUND_UP,
        )

        if self.exchange.stoploss_adjust(stoploss_norm, order, side=trade.exit_side):
            # 我们检查是否需要更新
            update_beat = self.strategy.order_types.get("stoploss_on_exchange_interval", 60)
            upd_req = datetime.now(timezone.utc) - timedelta(seconds=update_beat)
            if trade.stoploss_last_update_utc and upd_req >= trade.stoploss_last_update_utc:
                # 首先取消当前交易所上的止损...
                logger.info(
                    f"正在取消交易对 {trade.pair} 的当前交易所止损 "
                    f"(订单号:{order['id']}) 以便添加另一个..."
                )

                self.cancel_stoploss_on_exchange(trade)
                if not trade.is_open:
                    logger.warning(
                        f"交易 {trade} 已平仓，不创建追踪止损订单。"
                    )
                    return

                # 创建新的止损订单
                if not self.create_stoploss_order(trade=trade, stop_price=stoploss_norm):
                    logger.warning(
                        f"无法为交易对 {trade.pair} 创建追踪止损订单。"
                    )

    def manage_trade_stoploss_orders(self, trade: Trade, stoploss_orders: list[CcxtOrder]):
        """
        根据交易的现有止损订单执行所需操作
        :param trade: 相应的交易
        :param stoploss_orders: 当前交易所上的止损订单
        :return: 无
        """
        # 如果所有止损订单因某种原因被取消，我们再次添加它
        canceled_sl_orders = [
            o for o in stoploss_orders if o["status"] in ("canceled", "cancelled")
        ]
        if (
            trade.is_open
            and len(stoploss_orders) > 0
            and len(stoploss_orders) == len(canceled_sl_orders)
        ):
            if self.create_stoploss_order(trade=trade, stop_price=trade.stoploss_or_liquidation):
                return False
            else:
                logger.warning("所有止损订单均已取消，但无法重新创建。")

        active_sl_orders = [o for o in stoploss_orders if o not in canceled_sl_orders]
        if len(active_sl_orders) > 0:
            last_active_sl_order = active_sl_orders[-1]
            # 最后我们检查追踪止损是否应上移。
            # 触发的订单现在是真实订单 - 所以不再替换止损
            if (
                trade.is_open
                and last_active_sl_order.get("status_stop") != "triggered"
                and (
                    self.config.get("trailing_stop", False)
                    or self.config.get("use_custom_stoploss", False)
                )
            ):
                # 如果启用了追踪止损，我们检查止损值是否已更改
                # 如果是，则取消止损订单并立即放置一个新值
                self.handle_trailing_stoploss_on_exchange(trade, last_active_sl_order)

        return

    def manage_open_orders(self) -> None:
        """
        管理交易所的未平仓订单。未成交订单可能在达到超时后被取消，
        或者在新蜡烛出现且用户请求时被替换。
        超时设置优先于限价订单调整请求。
        :return: 无
        """
        for trade in Trade.get_open_trades():
            open_order: Order
            for open_order in trade.open_orders:
                try:
                    order = self.exchange.fetch_order(open_order.order_id, trade.pair)

                except ExchangeError:
                    logger.info(
                        "由于 %s，无法查询订单 %s", trade, traceback.format_exc()
                    )
                    continue

                fully_cancelled = self.update_trade_state(trade, open_order.order_id, order)
                not_closed = order["status"] == "open" or fully_cancelled

                if not_closed:
                    if fully_cancelled or (
                        open_order
                        and self.strategy.ft_check_timed_out(
                            trade, open_order, datetime.now(timezone.utc)
                        )
                    ):
                        self.handle_cancel_order(
                            order, open_order, trade, constants.CANCEL_REASON["TIMEOUT"]
                        )
                    else:
                        self.replace_order(order, open_order, trade)

    def handle_cancel_order(
        self, order: CcxtOrder, order_obj: Order, trade: Trade, reason: str, replacing: bool = False
    ) -> bool:
        """
        检查当前分析的订单是否超时并根据需要取消。
        :param order: 通过 exchange.fetch_order() 获取的订单字典
        :param order_obj: 数据库中的订单对象。
        :param trade: 交易对象。
        :return: 如果订单被取消，则为 True，否则为 False。
        """
        if order["side"] == trade.entry_side:
            return self.handle_cancel_enter(trade, order, order_obj, reason, replacing)
        else:
            canceled = self.handle_cancel_exit(trade, order, order_obj, reason)
            if not replacing:
                canceled_count = trade.get_canceled_exit_order_count()
                max_timeouts = self.config.get("unfilledtimeout", {}).get("exit_timeout_count", 0)
                if canceled and max_timeouts > 0 and canceled_count >= max_timeouts:
                    logger.warning(
                        f"紧急退出交易 {trade}，因为退出订单 "
                        f"超时 {max_timeouts} 次。强制卖出 {order['amount']}。"
                    )
                    self.emergency_exit(trade, order["price"], order["amount"])
            return canceled

    def emergency_exit(
        self, trade: Trade, price: float, sub_trade_amt: float | None = None
    ) -> None:
        try:
            self.execute_trade_exit(
                trade,
                price,
                exit_check=ExitCheckTuple(exit_type=ExitType.EMERGENCY_EXIT),
                sub_trade_amt=sub_trade_amt,
            )
        except DependencyException as exception:
            logger.warning(f"无法紧急退出交易 {trade.pair}: {exception}")

    def replace_order_failed(self, trade: Trade, msg: str) -> None:
        """
        订单替换失败处理。
        必要时删除交易。
        :param trade: 交易对象。
        :param msg: 错误消息。
        """
        logger.warning(msg)
        if trade.nr_of_successful_entries == 0:
            # 这是第一次入场，我们还没有成交，删除交易
            logger.warning(f"正在从数据库中移除 {trade}。")
            self._notify_enter_cancel(
                trade,
                order_type=self.strategy.order_types["entry"],
                reason=constants.CANCEL_REASON["REPLACE_FAILED"],
            )
            trade.delete()

    def replace_order(self, order: CcxtOrder, order_obj: Order | None, trade: Trade) -> None:
        """
        检查当前分析的入场订单是否应该被替换或简单取消。
        要简单取消现有订单（不替换），adjust_order_price() 应返回 None
        要保留现有订单，adjust_order_price() 应返回 order_obj.price
        要替换现有订单，adjust_order_price() 应返回限价订单所需的价格
        :param order: 通过 exchange.fetch_order() 获取的订单字典
        :param order_obj: 订单对象。
        :param trade: 交易对象。
        :return: 无
        """
        analyzed_df, _ = self.dataprovider.get_analyzed_dataframe(
            trade.pair, self.strategy.timeframe
        )
        latest_candle_open_date = analyzed_df.iloc[-1]["date"] if len(analyzed_df) > 0 else None
        latest_candle_close_date = timeframe_to_next_date(
            self.strategy.timeframe, latest_candle_open_date
        )
        # 检查是否是新蜡烛
        if order_obj and latest_candle_close_date > order_obj.order_date_utc:
            is_entry = order_obj.side == trade.entry_side
            # 新蜡烛
            proposed_rate = self.exchange.get_rate(
                trade.pair,
                side="entry" if is_entry else "exit",
                is_short=trade.is_short,
                refresh=True,
            )
            adjusted_price = strategy_safe_wrapper(
                self.strategy.adjust_order_price, default_retval=order_obj.safe_placement_price
            )(
                trade=trade,
                order=order_obj,
                pair=trade.pair,
                current_time=datetime.now(timezone.utc),
                proposed_rate=proposed_rate,
                current_order_rate=order_obj.safe_placement_price,
                entry_tag=trade.enter_tag,
                side=trade.trade_direction,
                is_entry=is_entry,
            )

            replacing = True
            cancel_reason = constants.CANCEL_REASON["REPLACE"]
            if not adjusted_price:
                replacing = False
                cancel_reason = constants.CANCEL_REASON["USER_CANCEL"]

            if order_obj.safe_placement_price != adjusted_price:
                self.handle_replace_order(
                    order,
                    order_obj,
                    trade,
                    adjusted_price,
                    is_entry,
                    cancel_reason,
                    replacing=replacing,
                )

    def handle_replace_order(
        self,
        order: CcxtOrder | None,
        order_obj: Order,
        trade: Trade,
        new_order_price: float | None,
        is_entry: bool,
        cancel_reason: str,
        replacing: bool = False,
    ) -> None:
        """
        如果提供了新价格，则取消现有订单，如果取消成功，
        则使用剩余资金下新订单。
        """
        if not order:
            order = self.exchange.fetch_order(order_obj.order_id, trade.pair)
        res = self.handle_cancel_order(order, order_obj, trade, cancel_reason, replacing=replacing)
        if not res:
            self.replace_order_failed(
                trade, f"无法完全取消交易 {trade} 的订单，因此不替换。"
            )
            return
        if new_order_price:
            # 仅在提供新价格时下新订单
            try:
                if is_entry:
                    succeeded = self.execute_entry(
                        pair=trade.pair,
                        stake_amount=(
                            order_obj.safe_remaining * order_obj.safe_price / trade.leverage
                        ),
                        price=new_order_price,
                        trade=trade,
                        is_short=trade.is_short,
                        mode="replace",
                    )
                else:
                    succeeded = self.execute_trade_exit(
                        trade,
                        new_order_price,
                        exit_check=ExitCheckTuple(
                            exit_type=ExitType.CUSTOM_EXIT,
                            exit_reason=order_obj.ft_order_tag or "订单已替换",
                        ),
                        ordertype="limit",
                        sub_trade_amt=order_obj.safe_remaining,
                    )
                if not succeeded:
                    self.replace_order_failed(trade, f"无法替换交易 {trade} 的订单。")
            except DependencyException as exception:
                logger.warning(f"无法替换交易对 {trade.pair} 的订单: {exception}")
                self.replace_order_failed(trade, f"无法替换交易 {trade} 的订单。")

    def cancel_open_orders_of_trade(
        self, trade: Trade, sides: list[str], reason: str, replacing: bool = False
    ) -> None:
        """
        取消当前未平仓的指定方向的交易订单
        :param trade: 我们正在分析的交易对象
        :param reason: 取消的原因
        :param sides: 应取消的方向
        :return: 无
        """

        for open_order in trade.open_orders:
            try:
                order = self.exchange.fetch_order(open_order.order_id, trade.pair)
            except ExchangeError:
                logger.info("由于 %s，无法查询订单 %s", trade, traceback.format_exc())
                continue

            if order["side"] in sides:
                if order["side"] == trade.entry_side:
                    self.handle_cancel_enter(trade, order, open_order, reason, replacing)

                elif order["side"] == trade.exit_side:
                    self.handle_cancel_exit(trade, order, open_order, reason)

    def cancel_all_open_orders(self) -> None:
        """
        取消当前所有未平仓订单
        :return: 无
        """

        for trade in Trade.get_open_trades():
            self.cancel_open_orders_of_trade(
                trade, [trade.entry_side, trade.exit_side], constants.CANCEL_REASON["ALL_CANCELLED"]
            )

        Trade.commit()

    def handle_similar_open_order(
        self, trade: Trade, price: float, amount: float, side: str
    ) -> bool:
        """
        如果金额和方向相同，则保留现有未平仓订单，否则取消
        :param trade: 我们正在分析的交易对象
        :param price: 潜在新订单的限价
        :param amount: 潜在新订单的资产数量
        :param side: 潜在新订单的方向
        :return: 如果找到现有类似订单，则为 True
        """
        if trade.has_open_orders:
            oo = trade.select_order(side, True)
            if oo is not None:
                if price == oo.price and side == oo.side and amount == oo.amount:
                    logger.info(
                        f"已为 {trade.pair} 找到类似的未平仓订单。 "
                        f"保留现有 {trade.exit_side} 订单。 价格={price}，金额={amount}"
                    )
                    return True
            # 如果订单不同，则取消此交易的未平仓订单
            self.cancel_open_orders_of_trade(
                trade,
                [trade.entry_side, trade.exit_side],
                constants.CANCEL_REASON["REPLACE"],
                True,
            )
            Trade.commit()
            return False

        return False

    def handle_cancel_enter(
        self,
        trade: Trade,
        order: CcxtOrder,
        order_obj: Order,
        reason: str,
        replacing: bool | None = False,
    ) -> bool:
        """
        入场取消 - 取消订单
        :param order_obj: 数据库中的订单对象。
        :param replacing: 替换订单 - 防止交易删除。
        :return: 如果交易完全取消，则为 True
        """
        was_trade_fully_canceled = False
        order_id = order_obj.order_id
        side = trade.entry_side.capitalize()

        if order["status"] not in constants.NON_OPEN_EXCHANGE_STATES:
            filled_val: float = order.get("filled", 0.0) or 0.0
            filled_stake = filled_val * trade.open_rate
            minstake = self.exchange.get_min_pair_stake_amount(
                trade.pair, trade.open_rate, self.strategy.stoploss
            )

            if filled_val > 0 and minstake and filled_stake < minstake:
                logger.warning(
                    f"订单 {order_id} 针对 {trade.pair} 未取消，"
                    f"因为成交金额 {filled_val} 将导致无法退出的交易。"
                )
                return False
            corder = self.exchange.cancel_order_with_result(order_id, trade.pair, trade.amount)
            order_obj.ft_cancel_reason = reason
            # 如果是替换，如果状态不是我们需要的，则重试获取订单 3 次
            if replacing:
                retry_count = 0
                while (
                    corder.get("status") not in constants.NON_OPEN_EXCHANGE_STATES
                    and retry_count < 3
                ):
                    sleep(0.5)
                    corder = self.exchange.fetch_order(order_id, trade.pair)
                    retry_count += 1

            # 避免竞态条件，即订单可能因为已经成交而无法取消。
            # 在这里简单地放弃是唯一安全的方法 - 因为此订单将在
            # 下一次迭代中处理。
            if corder.get("status") not in constants.NON_OPEN_EXCHANGE_STATES:
                logger.warning(f"订单 {order_id} 针对 {trade.pair} 未取消。")
                return False
        else:
            # 订单已被取消，因此我们可以重用现有字典
            corder = order
            if order_obj.ft_cancel_reason is None:
                order_obj.ft_cancel_reason = constants.CANCEL_REASON["CANCELLED_ON_EXCHANGE"]

        logger.info(f"交易 {trade} 的 {side} 订单 {order_obj.ft_cancel_reason}。")

        # 使用 filled 来确定成交金额
        filled_amount = safe_value_fallback2(corder, order, "filled", "filled")
        if isclose(filled_amount, 0.0, abs_tol=constants.MATH_CLOSE_PREC):
            was_trade_fully_canceled = True
            # 如果交易未部分完成且它是唯一的订单，则删除交易
            open_order_count = len(
                [order for order in trade.orders if order.ft_is_open and order.order_id != order_id]
            )
            if open_order_count < 1 and trade.nr_of_successful_entries == 0 and not replacing:
                logger.info(f"{side} 订单已完全取消。正在从数据库中移除 {trade}。")
                trade.delete()
                order_obj.ft_cancel_reason += f", {constants.CANCEL_REASON['FULLY_CANCELLED']}"
            else:
                self.update_trade_state(trade, order_id, corder)
                logger.info(f"交易 {trade} 的 {side} 订单超时。")
        else:
            # update_trade_state（以及随后的 recalc_trade_from_orders）将处理对
            # 交易对象的更新
            self.update_trade_state(trade, order_id, corder)

            logger.info(
                f"交易 {trade} 的 {trade.entry_side} 订单部分超时。已成交：{filled_amount}，"
                f"总计：{order_obj.ft_amount}"
            )
            order_obj.ft_cancel_reason += f", {constants.CANCEL_REASON['PARTIALLY_FILLED']}"

        self.wallets.update()
        self._notify_enter_cancel(
            trade, order_type=self.strategy.order_types["entry"], reason=order_obj.ft_cancel_reason
        )
        return was_trade_fully_canceled

    def handle_cancel_exit(
        self, trade: Trade, order: CcxtOrder, order_obj: Order, reason: str
    ) -> bool:
        """
        退出订单取消 - 取消订单并更新交易
        :return: 如果退出订单被取消，则为 True，否则为 False
        """
        order_id = order_obj.order_id
        cancelled = False
        # 已取消订单的状态可能为 'canceled' 或 'closed'
        if order["status"] not in constants.NON_OPEN_EXCHANGE_STATES:
            filled_amt: float = order.get("filled", 0.0) or 0.0
            # Filled val is in quote currency (after leverage)
            filled_rem_stake = trade.stake_amount - (filled_amt * trade.open_rate / trade.leverage)
            minstake = self.exchange.get_min_pair_stake_amount(
                trade.pair, trade.open_rate, self.strategy.stoploss
            )
            # 再次检查剩余金额
            if filled_amt > 0:
                reason = constants.CANCEL_REASON["PARTIALLY_FILLED"]
                if minstake and filled_rem_stake < minstake:
                    logger.warning(
                        f"订单 {order_id} 针对 {trade.pair} 未取消，因为 "
                        f"成交金额 {filled_amt} 将导致无法退出的交易。"
                    )
                    reason = constants.CANCEL_REASON["PARTIALLY_FILLED_KEEP_OPEN"]

                    self._notify_exit_cancel(
                        trade,
                        order_type=self.strategy.order_types["exit"],
                        reason=reason,
                        order_id=order["id"],
                        sub_trade=trade.amount != order["amount"],
                    )
                    return False
            order_obj.ft_cancel_reason = reason
            try:
                order = self.exchange.cancel_order_with_result(
                    order["id"], trade.pair, trade.amount
                )
            except InvalidOrderException:
                logger.exception(f"无法取消 {trade.exit_side} 订单 {order_id}")
                return False

            # 设置退出原因以供填充消息
            exit_reason_prev = trade.exit_reason
            trade.exit_reason = trade.exit_reason + f", {reason}" if trade.exit_reason else reason
            # 订单可能在奇怪的时间问题中被填充。
            if order.get("status") in ("canceled", "cancelled"):
                trade.exit_reason = None
            else:
                trade.exit_reason = exit_reason_prev
            cancelled = True
        else:
            if order_obj.ft_cancel_reason is None:
                order_obj.ft_cancel_reason = constants.CANCEL_REASON["CANCELLED_ON_EXCHANGE"]
            trade.exit_reason = None

        self.update_trade_state(trade, order["id"], order)

        logger.info(
            f"交易 {trade} 的 {trade.exit_side.capitalize()} 订单 {order_obj.ft_cancel_reason}。"
        )
        trade.close_rate = None
        trade.close_rate_requested = None

        self._notify_exit_cancel(
            trade,
            order_type=self.strategy.order_types["exit"],
            reason=order_obj.ft_cancel_reason,
            order_id=order["id"],
            sub_trade=trade.amount != order["amount"],
        )
        return cancelled

    def _safe_exit_amount(self, trade: Trade, pair: str, amount: float) -> float:
        """
        获取可卖出金额。
        应该是 trade.amount - 但在必要时会回退到可用金额。
        这应该涵盖 get_real_amount() 因某种原因无法更新金额的情况。
        :param trade: 我们正在处理的交易
        :param pair: 我们正在尝试卖出的交易对
        :param amount: 我们预期可用的金额
        :return: 卖出金额
        :raise: DependencyException: 如果可用余额与所需金额的差值超过 2%。
        """
        # 更新钱包以确保止损中锁定的金额现在是自由的！
        self.wallets.update()
        if self.trading_mode == TradingMode.FUTURES:
            # 期货不需要安全退出金额，您可以直接退出/平仓
            return amount

        trade_base_currency = self.exchange.get_pair_base_currency(pair)
        # 可用 + 已用 - 未平仓订单最终仍将被取消。
        wallet_amount = self.wallets.get_free(trade_base_currency) + self.wallets.get_used(
            trade_base_currency
        )

        logger.debug(f"{pair} - 钱包：{wallet_amount} - 交易金额：{amount}")
        if wallet_amount >= amount:
            return amount
        elif wallet_amount > amount * 0.98:
            logger.info(f"{pair} - 回退到钱包金额 {wallet_amount} -> {amount}。")
            trade.amount = wallet_amount
            return wallet_amount
        else:
            raise DependencyException(
                f"金额不足以退出交易。交易金额：{amount}，钱包：{wallet_amount}"
            )

    def execute_trade_exit(
        self,
        trade: Trade,
        limit: float,
        exit_check: ExitCheckTuple,
        *,
        exit_tag: str | None = None,
        ordertype: str | None = None,
        sub_trade_amt: float | None = None,
    ) -> bool:
        """
        为给定交易和限价执行交易退出
        :param trade: 交易实例
        :param limit: 卖出订单的限价
        :param exit_check: 包含信号和原因的 CheckTuple
        :return: 如果成功则为 True，否则为 False
        """
        trade.set_funding_fees(
            self.exchange.get_funding_fees(
                pair=trade.pair,
                amount=trade.amount,
                is_short=trade.is_short,
                open_date=trade.date_last_filled_utc,
            )
        )

        exit_type = "退出"
        exit_reason = exit_tag or exit_check.exit_reason
        if exit_check.exit_type in (
            ExitType.STOP_LOSS,
            ExitType.TRAILING_STOP_LOSS,
            ExitType.LIQUIDATION,
        ):
            exit_type = "止损"

        # 如果可用，设置 custom_exit_price
        proposed_limit_rate = limit
        current_profit = trade.calc_profit_ratio(limit)
        custom_exit_price = strategy_safe_wrapper(
            self.strategy.custom_exit_price, default_retval=proposed_limit_rate
        )(
            pair=trade.pair,
            trade=trade,
            current_time=datetime.now(timezone.utc),
            proposed_rate=proposed_limit_rate,
            current_profit=current_profit,
            exit_tag=exit_reason,
        )

        limit = self.get_valid_price(custom_exit_price, proposed_limit_rate)

        # 首先取消交易所上的止损...
        trade = self.cancel_stoploss_on_exchange(trade)

        order_type = ordertype or self.strategy.order_types[exit_type]
        if exit_check.exit_type == ExitType.EMERGENCY_EXIT:
            # 紧急卖出（默认为市价！）
            order_type = self.strategy.order_types.get("emergency_exit", "market")

        amount = self._safe_exit_amount(trade, trade.pair, sub_trade_amt or trade.amount)
        time_in_force = self.strategy.order_time_in_force["exit"]

        if (
            exit_check.exit_type != ExitType.LIQUIDATION
            and not sub_trade_amt
            and not strategy_safe_wrapper(self.strategy.confirm_trade_exit, default_retval=True)(
                pair=trade.pair,
                trade=trade,
                order_type=order_type,
                amount=amount,
                rate=limit,
                time_in_force=time_in_force,
                exit_reason=exit_reason,
                sell_reason=exit_reason,  # 卖出原因 -> 兼容性
                current_time=datetime.now(timezone.utc),
            )
        ):
            logger.info(f"用户拒绝 {trade.pair} 的退出。")
            return False

        if trade.has_open_orders:
            if self.handle_similar_open_order(trade, limit, amount, trade.exit_side):
                return False

        try:
            # 执行卖出并更新交易记录
            order = self.exchange.create_order(
                pair=trade.pair,
                ordertype=order_type,
                side=trade.exit_side,
                amount=amount,
                rate=limit,
                leverage=trade.leverage,
                reduceOnly=self.trading_mode == TradingMode.FUTURES,
                time_in_force=time_in_force,
            )
        except InsufficientFundsError as e:
            logger.warning(f"无法下订单 {e}。")
            # 尝试找出问题所在
            self.handle_insufficient_funds(trade)
            return False

        self._exit_reason_cache[f"{trade.pair}_{trade.id}_{exit_reason}"] = dt_now()
        order_obj = Order.parse_from_ccxt_object(order, trade.pair, trade.exit_side, amount, limit)
        order_obj.ft_order_tag = exit_reason
        trade.orders.append(order_obj)

        trade.exit_order_status = ""
        trade.close_rate_requested = limit
        trade.exit_reason = exit_reason

        self._notify_exit(trade, order_type, sub_trade=bool(sub_trade_amt), order=order_obj)
        # 在市价卖出订单的情况下，订单可以立即平仓
        if order.get("status", "unknown") in ("closed", "expired"):
            self.update_trade_state(trade, order_obj.order_id, order)
        Trade.commit()

        return True

    def _notify_exit(
        self,
        trade: Trade,
        order_type: str | None,
        fill: bool = False,
        sub_trade: bool = False,
        order: Order | None = None,
    ) -> None:
        """
        当发生卖出时发送 RPC 通知。
        """
        # 在这里使用缓存费率 - 它在几秒钟前已更新。
        current_rate = (
            self.exchange.get_rate(trade.pair, side="exit", is_short=trade.is_short, refresh=False)
            if not fill
            else None
        )

        # 第二个条件仅用于 mypy；在子交易期间，order 总是会被传递
        if sub_trade and order is not None:
            amount = order.safe_filled if fill else order.safe_amount
            order_rate: float = order.safe_price

            profit = trade.calculate_profit(order_rate, amount, trade.open_rate)
        else:
            order_rate = trade.safe_close_rate
            profit = trade.calculate_profit(rate=order_rate)
            amount = trade.amount
        gain: ProfitLossStr = "盈利" if profit.profit_ratio > 0 else "亏损"

        msg: RPCExitMsg = {
            "type": (RPCMessageType.EXIT_FILL if fill else RPCMessageType.EXIT),
            "trade_id": trade.id,
            "exchange": trade.exchange.capitalize(),
            "pair": trade.pair,
            "leverage": trade.leverage,
            "direction": "空头" if trade.is_short else "多头",
            "gain": gain,
            "limit": order_rate,  # 已弃用
            "order_rate": order_rate,
            "order_type": order_type or "unknown",
            "amount": amount,
            "open_rate": trade.open_rate,
            "close_rate": order_rate,
            "current_rate": current_rate,
            "profit_amount": profit.profit_abs,
            "profit_ratio": profit.profit_ratio,
            "buy_tag": trade.enter_tag,
            "enter_tag": trade.enter_tag,
            "exit_reason": trade.exit_reason,
            "open_date": trade.open_date_utc,
            "close_date": trade.close_date_utc or datetime.now(timezone.utc),
            "stake_amount": trade.stake_amount,
            "stake_currency": self.config["stake_currency"],
            "base_currency": self.exchange.get_pair_base_currency(trade.pair),
            "quote_currency": self.exchange.get_pair_quote_currency(trade.pair),
            "fiat_currency": self.config.get("fiat_display_currency"),
            "sub_trade": sub_trade,
            "cumulative_profit": trade.realized_profit,
            "final_profit_ratio": trade.close_profit if not trade.is_open else None,
            "is_final_exit": trade.is_open is False,
        }

        # 发送消息
        self.rpc.send_msg(msg)

    def _notify_exit_cancel(
        self, trade: Trade, order_type: str, reason: str, order_id: str, sub_trade: bool = False
    ) -> None:
        """
        当卖出取消时发送 RPC 通知。
        """
        if trade.exit_order_status == reason:
            return
        else:
            trade.exit_order_status = reason

        order_or_none = trade.select_order_by_order_id(order_id)
        order = self.order_obj_or_raise(order_id, order_or_none)

        profit_rate: float = trade.safe_close_rate
        profit = trade.calculate_profit(rate=profit_rate)
        current_rate = self.exchange.get_rate(
            trade.pair, side="exit", is_short=trade.is_short, refresh=False
        )
        gain: ProfitLossStr = "盈利" if profit.profit_ratio > 0 else "亏损"

        msg: RPCExitCancelMsg = {
            "type": RPCMessageType.EXIT_CANCEL,
            "trade_id": trade.id,
            "exchange": trade.exchange.capitalize(),
            "pair": trade.pair,
            "leverage": trade.leverage,
            "direction": "空头" if trade.is_short else "多头",
            "gain": gain,
            "limit": profit_rate or 0,
            "order_type": order_type,
            "amount": order.safe_amount_after_fee,
            "open_rate": trade.open_rate,
            "current_rate": current_rate,
            "profit_amount": profit.profit_abs,
            "profit_ratio": profit.profit_ratio,
            "buy_tag": trade.enter_tag,
            "enter_tag": trade.enter_tag,
            "exit_reason": trade.exit_reason,
            "open_date": trade.open_date,
            "close_date": trade.close_date or datetime.now(timezone.utc),
            "stake_currency": self.config["stake_currency"],
            "base_currency": self.exchange.get_pair_base_currency(trade.pair),
            "quote_currency": self.exchange.get_pair_quote_currency(trade.pair),
            "fiat_currency": self.config.get("fiat_display_currency", None),
            "reason": reason,
            "sub_trade": sub_trade,
            "stake_amount": trade.stake_amount,
        }

        # 发送消息
        self.rpc.send_msg(msg)

    def order_obj_or_raise(self, order_id: str, order_obj: Order | None) -> Order:
        if not order_obj:
            raise DependencyException(
                f"未找到订单 {order_id} 的 order_obj。这不应该发生。"
            )
        return order_obj

    #
    # 常用更新交易状态方法
    #

    def update_trade_state(
        self,
        trade: Trade,
        order_id: str | None,
        action_order: CcxtOrder | None = None,
        *,
        stoploss_order: bool = False,
        send_msg: bool = True,
    ) -> bool:
        """
        检查具有未平仓订单的交易并根据需要更新金额
        处理同时平仓买入和卖出订单。
        :param trade: 我们正在分析的交易对象
        :param order_id: 我们正在分析的订单 ID
        :param action_order: 已经获取的订单对象
        :param send_msg: 发送通知 - 除了在“恢复”方法中，应始终为 True
        :return: 如果订单已取消且未部分成交，则为 True，否则为 False
        """
        if not order_id:
            logger.warning(f"交易 {trade} 的订单 ID 为空。")
            return False

        # 使用订单值更新交易
        if not stoploss_order:
            logger.info(f"找到交易 {trade} 的未平仓订单")
        try:
            order = action_order or self.exchange.fetch_order_or_stoploss_order(
                order_id, trade.pair, stoploss_order
            )
        except InvalidOrderException as exception:
            logger.warning("无法获取订单 %s：%s", order_id, exception)
            return False

        trade.update_order(order)

        if self.exchange.check_order_canceled_empty(order):
            # 交易已在交易所取消
            # 这将在 handle_cancel_order 中处理。
            return True

        order_obj_or_none = trade.select_order_by_order_id(order_id)
        order_obj = self.order_obj_or_raise(order_id, order_obj_or_none)

        self.handle_order_fee(trade, order_obj, order)

        trade.update_trade(order_obj, not send_msg)

        trade = self._update_trade_after_fill(trade, order_obj, send_msg)
        Trade.commit()

        self.order_close_notify(trade, order_obj, stoploss_order, send_msg)

        return False

    def _update_trade_after_fill(self, trade: Trade, order: Order, send_msg: bool) -> Trade:
        if order.status in constants.NON_OPEN_EXCHANGE_STATES:
            strategy_safe_wrapper(self.strategy.order_filled, default_retval=None)(
                pair=trade.pair, trade=trade, order=order, current_time=datetime.now(timezone.utc)
            )
            # 如果入场订单已平仓，则强制更新交易所上的止损
            if order.ft_order_side == trade.entry_side:
                if send_msg:
                    if trade.nr_of_successful_entries > 1:
                        # 重置 fee_open_currency 以使费用检查生效
                        # 仅对额外的入场生效
                        trade.fee_open_currency = None
                    # 在恢复模式下不立即取消止损
                    trade = self.cancel_stoploss_on_exchange(trade)
                trade.adjust_stop_loss(trade.open_rate, self.strategy.stoploss, initial=True)
            if (
                order.ft_order_side == trade.entry_side
                or (trade.amount > 0 and trade.is_open)
                or self.margin_mode == MarginMode.CROSS
            ):
                # 必须也适用于部分退出
                # TODO: 保证金也需要使用 interest_rate。
                # interest_rate = self.exchange.get_interest_rate()
                update_liquidation_prices(
                    trade,
                    exchange=self.exchange,
                    wallets=self.wallets,
                    stake_currency=self.config["stake_currency"],
                    dry_run=self.config["dry_run"],
                )
                if self.strategy.use_custom_stoploss:
                    current_rate = self.exchange.get_rate(
                        trade.pair, side="exit", is_short=trade.is_short, refresh=True
                    )
                    profit = trade.calc_profit_ratio(current_rate)
                    self.strategy.ft_stoploss_adjust(
                        current_rate, trade, datetime.now(timezone.utc), profit, 0, after_fill=True
                    )
            # 订单平仓时更新钱包
            self.wallets.update()
        return trade

    def order_close_notify(self, trade: Trade, order: Order, stoploss_order: bool, send_msg: bool):
        """发送“填充”通知"""

        if order.ft_order_side == trade.exit_side:
            # 退出通知
            if send_msg and not stoploss_order and order.order_id not in trade.open_orders_ids:
                self._notify_exit(
                    trade, order.order_type, fill=True, sub_trade=trade.is_open, order=order
                )
            if not trade.is_open:
                self.handle_protections(trade.pair, trade.trade_direction)
        elif send_msg and order.order_id not in trade.open_orders_ids and not stoploss_order:
            sub_trade = not isclose(
                order.safe_amount_after_fee, trade.amount, abs_tol=constants.MATH_CLOSE_PREC
            )
            # 入场填充
            self._notify_enter(trade, order, order.order_type, fill=True, sub_trade=sub_trade)

    def handle_protections(self, pair: str, side: LongShort) -> None:
        # 锁定交易对一个蜡烛周期，以防止立即重新入场
        self.strategy.lock_pair(pair, datetime.now(timezone.utc), reason="自动锁定", side=side)
        prot_trig = self.protections.stop_per_pair(pair, side=side)
        if prot_trig:
            msg: RPCProtectionMsg = {
                "type": RPCMessageType.PROTECTION_TRIGGER,
                "base_currency": self.exchange.get_pair_base_currency(prot_trig.pair),
                **prot_trig.to_json(),  # type: ignore
            }
            self.rpc.send_msg(msg)

        prot_trig_glb = self.protections.global_stop(side=side)
        if prot_trig_glb:
            msg = {
                "type": RPCMessageType.PROTECTION_TRIGGER_GLOBAL,
                "base_currency": self.exchange.get_pair_base_currency(prot_trig_glb.pair),
                **prot_trig_glb.to_json(),  # type: ignore
            }
            self.rpc.send_msg(msg)

    def apply_fee_conditional(
        self,
        trade: Trade,
        trade_base_currency: str,
        amount: float,
        fee_abs: float,
        order_obj: Order,
    ) -> float | None:
        """
        将费用应用于金额（来自订单或交易）。
        如果可用资产多于所需资产，可能会造成粉尘。
        在交易调整订单的情况下，trade.amount 可能尚未调整。
        在期货模式下不会发生 - 费用总是以结算货币收取，
        从不以基础货币收取。
        """
        self.wallets.update()
        amount_ = trade.amount
        if order_obj.ft_order_side == trade.exit_side or order_obj.ft_order_side == "stoploss":
            # 检查剩余金额！
            amount_ = trade.amount - amount

        if trade.nr_of_successful_entries >= 1 and order_obj.ft_order_side == trade.entry_side:
            # 在重新入场的情况下，trade.amount 不包含上次入场的金额。
            amount_ = trade.amount + amount

        if fee_abs != 0 and self.wallets.get_free(trade_base_currency) >= amount_:
            # 如果我们拥有超过基础货币的资产，则消耗粉尘
            logger.info(
                f"交易 {trade} 的费用以基础货币计价 - 正在消耗费用 {fee_abs}。"
            )
        elif fee_abs != 0:
            logger.info(f"正在将费用应用于交易 {trade} 的金额，费用={fee_abs}。")
            return fee_abs
        return None

    def handle_order_fee(self, trade: Trade, order_obj: Order, order: CcxtOrder) -> None:
        # 尝试更新金额（币安修复 - 但也适用于不同交易所）
        try:
            if (fee_abs := self.get_real_amount(trade, order, order_obj)) is not None:
                order_obj.ft_fee_base = fee_abs
        except DependencyException as exception:
            logger.warning("无法更新交易金额：%s", exception)

    def get_real_amount(self, trade: Trade, order: CcxtOrder, order_obj: Order) -> float | None:
        """
        检测并更新交易费用。
        在正确检测后调用 trade.update_fee()。
        如果费用是从目标货币中扣除的，则返回修改后的金额。
        对于以基础货币收取费用的交易所（例如币安）是必要的。
        :return: 此订单要应用的绝对费用或 None
        """
        # 初始化变量
        order_amount = safe_value_fallback(order, "filled", "amount")
        # 仅对已平仓订单运行
        if (
            trade.fee_updated(order.get("side", "")) or order["status"] == "open"
            # 或 order_obj.ft_fee_base
        ):
            return None

        trade_base_currency = self.exchange.get_pair_base_currency(trade.pair)
        # 如果可能，使用订单字典中的费用
        if self.exchange.order_has_fee(order):
            fee_cost, fee_currency, fee_rate = self.exchange.extract_cost_curr_rate(
                order["fee"], order["symbol"], order["cost"], order_obj.safe_filled
            )
            logger.info(
                f"交易 {trade} [{order_obj.ft_order_side}] 的费用: "
                f"{fee_cost:.8g} {fee_currency} - 费率: {fee_rate}"
            )
            if fee_rate is None or fee_rate < 0.02:
                # 拒绝所有报告大于 2% 的费用。
                # 这些很可能是由 ccxt 中的解析错误引起的
                # 由于多个交易（https://github.com/ccxt/ccxt/issues/8025）
                trade.update_fee(fee_cost, fee_currency, fee_rate, order.get("side", ""))
                if trade_base_currency == fee_currency:
                    # 将费用应用于金额
                    return self.apply_fee_conditional(
                        trade,
                        trade_base_currency,
                        amount=order_amount,
                        fee_abs=fee_cost,
                        order_obj=order_obj,
                    )
                return None
        return self.fee_detection_from_trades(
            trade, order, order_obj, order_amount, order.get("trades", [])
        )

    def _trades_valid_for_fee(self, trades: list[dict[str, Any]]) -> bool:
        """
        检查交易是否有效以便于费用检测。
        :return: 如果交易有效以便于费用检测，则为 True，否则为 False
        """
        if not trades:
            return False
        # 我们期望所有交易对象中都存在 amount 和 cost。
        if any(trade.get("amount") is None or trade.get("cost") is None for trade in trades):
            return False
        return True

    def fee_detection_from_trades(
        self, trade: Trade, order: CcxtOrder, order_obj: Order, order_amount: float, trades: list
    ) -> float | None:
        """
        费用检测回退到交易。
        使用提供的交易列表或 fetch_my_trades 的结果来获取正确的费用。
        """
        if not self._trades_valid_for_fee(trades):
            trades = self.exchange.get_trades_for_order(
                self.exchange.get_order_id_conditional(order), trade.pair, order_obj.order_date
            )

        if len(trades) == 0:
            logger.info("将费用应用于 %s 的金额失败：未找到我的交易字典", trade)
            return None
        fee_currency = None
        amount = 0
        fee_abs = 0.0
        fee_cost = 0.0
        trade_base_currency = self.exchange.get_pair_base_currency(trade.pair)
        fee_rate_array: list[float] = []
        for exectrade in trades:
            amount += exectrade["amount"]
            if self.exchange.order_has_fee(exectrade):
                # 优先使用单一费用
                fees = [exectrade["fee"]]
            else:
                fees = exectrade.get("fees", [])
            for fee in fees:
                fee_cost_, fee_currency, fee_rate_ = self.exchange.extract_cost_curr_rate(
                    fee, exectrade["symbol"], exectrade["cost"], exectrade["amount"]
                )
                fee_cost += fee_cost_
                if fee_rate_ is not None:
                    fee_rate_array.append(fee_rate_)
                # 仅当费用以报价货币计算时才适用！
                if trade_base_currency == fee_currency:
                    fee_abs += fee_cost_
        # 确保至少找到一个交易：
        if fee_currency:
            # fee_rate 应该使用平均值
            fee_rate = sum(fee_rate_array) / float(len(fee_rate_array)) if fee_rate_array else None
            if fee_rate is not None and fee_rate < 0.02:
                # 仅在费率 < 2% 时更新
                trade.update_fee(fee_cost, fee_currency, fee_rate, order.get("side", ""))
            else:
                logger.warning(
                    f"未更新 {order.get('side', '')} 费用 - 费率: {fee_rate}, {fee_currency}。"
                )

        if not isclose(amount, order_amount, abs_tol=constants.MATH_CLOSE_PREC):
            # * 杠杆可能是导致此警告的原因
            logger.warning(f"金额 {amount} 与金额 {trade.amount} 不匹配")
            raise DependencyException("半买？金额不匹配")

        if fee_abs != 0:
            return self.apply_fee_conditional(
                trade, trade_base_currency, amount=amount, fee_abs=fee_abs, order_obj=order_obj
            )
        return None

    def get_valid_price(self, custom_price: float, proposed_price: float) -> float:
        """
        返回有效价格。
        检查自定义价格类型是否正确，如果不是则返回 proposed_price
        :return: 订单的有效价格
        """
        if custom_price:
            try:
                valid_custom_price = float(custom_price)
            except ValueError:
                valid_custom_price = proposed_price
        else:
            valid_custom_price = proposed_price

        cust_p_max_dist_r = self.config.get("custom_price_max_distance_ratio", 0.02)
        min_custom_price_allowed = proposed_price - (proposed_price * cust_p_max_dist_r)
        max_custom_price_allowed = proposed_price + (proposed_price * cust_p_max_dist_r)

        # 限制在 min_custom_price_allowed 和 max_custom_price_allowed 之间
        final_price = max(
            min(valid_custom_price, max_custom_price_allowed), min_custom_price_allowed
        )

        # 如果自定义价格通过限制进行了调整，则记录警告。
        if final_price != valid_custom_price:
            logger.info(
                f"自定义价格从 {valid_custom_price} 调整到 {final_price}，基于 "
                f"custom_price_max_distance_ratio 为 {cust_p_max_dist_r}。"
            )

        return final_price
