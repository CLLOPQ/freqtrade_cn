"""
这个模块包含用于定义RPC通信的类
"""

import logging
from abc import abstractmethod
from collections.abc import Generator, Sequence
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import psutil
from dateutil.relativedelta import relativedelta
from dateutil.tz import tzlocal
from numpy import inf, int64, isnan, mean, nan
from pandas import DataFrame, NaT
from sqlalchemy import func, select

from freqtrade import __version__
from freqtrade.configuration.timerange import TimeRange
from freqtrade.constants import CANCEL_REASON, DEFAULT_DATAFRAME_COLUMNS, Config
from freqtrade.data.history import load_data
from freqtrade.data.metrics import DrawDownResult, calculate_expectancy, calculate_max_drawdown
from freqtrade.enums import (
    CandleType,
    ExitCheckTuple,
    ExitType,
    MarketDirection,
    SignalDirection,
    State,
    TradingMode,
)
from freqtrade.exceptions import ExchangeError, PricingError
from freqtrade.exchange import Exchange, timeframe_to_minutes, timeframe_to_msecs
from freqtrade.exchange.exchange_utils import price_to_precision
from freqtrade.ft_types import AnnotationType
from freqtrade.loggers import bufferHandler
from freqtrade.persistence import CustomDataWrapper, KeyValueStore, PairLocks, Trade
from freqtrade.persistence.models import PairLock, custom_data_rpc_wrapper
from freqtrade.plugins.pairlist.pairlist_helpers import expand_pairlist
from freqtrade.rpc.fiat_convert import CryptoToFiatConverter
from freqtrade.rpc.rpc_types import RPCSendMsg
from freqtrade.util import (
    decimals_per_coin,
    dt_from_ts,
    dt_humanize_delta,
    dt_now,
    dt_ts,
    dt_ts_def,
    format_date,
    shorten_date,
)
from freqtrade.wallets import PositionWallet, Wallet


logger = logging.getLogger(__name__)


class RPCException(Exception):
    """
    如果所需状态错误，应该在 _rpc_* 方法中抛出带有rpc格式消息的此异常，
    例如：

    raise RPCException('*状态:* `没有活跃交易`')
    """

    def __init__(self, message: str) -> None:
        super().__init__(self)
        self.message = message

    def __str__(self):
        return self.message

    def __json__(self):
        return {"msg": self.message}


class RPCHandler:
    def __init__(self, rpc: "RPC", config: Config) -> None:
        """
        初始化RPCHandler
        :param rpc: RPC辅助类实例
        :param config: 配置对象
        :return: 无
        """
        self._rpc = rpc
        self._config: Config = config

    @property
    def name(self) -> str:
        """返回实现的小写名称"""
        return self.__class__.__name__.lower()

    @abstractmethod
    def cleanup(self) -> None:
        """清理待处理的模块资源"""

    @abstractmethod
    def send_msg(self, msg: RPCSendMsg) -> None:
        """向所有注册的rpc模块发送消息"""


class RPC:
    """
    RPC类可用于提供额外功能，如机器人数据和访问数据库数据
    """

    # 如果需要，绑定 _fiat_converter
    _fiat_converter: CryptoToFiatConverter | None = None
    if TYPE_CHECKING:
        from freqtrade.freqtradebot import FreqtradeBot

        _freqtrade: FreqtradeBot

    def __init__(self, freqtrade) -> None:
        """
        初始化所有已启用的rpc模块
        :param freqtrade: freqtrade机器人实例
        :return: 无
        """
        self._freqtrade = freqtrade
        self._config: Config = freqtrade.config
        if self._config.get("fiat_display_currency"):
            self._fiat_converter = CryptoToFiatConverter(self._config)

    @staticmethod
    def _rpc_show_config(
        config, botstate: State | str, strategy_version: str | None = None
    ) -> dict[str, Any]:
        """
        返回一个包含配置选项的字典。
        明确地不返回完整配置，以避免通过rpc泄露敏感信息。
        """
        val = {
            "version": __version__,
            "strategy_version": strategy_version,
            "dry_run": config["dry_run"],
            "trading_mode": config.get("trading_mode", "spot"),
            "short_allowed": config.get("trading_mode", "spot") != "spot",
            "stake_currency": config["stake_currency"],
            "stake_currency_decimals": decimals_per_coin(config["stake_currency"]),
            "stake_amount": str(config["stake_amount"]),
            "available_capital": config.get("available_capital"),
            "max_open_trades": (
                config.get("max_open_trades", 0)
                if config.get("max_open_trades", 0) != float("inf")
                else -1
            ),
            "minimal_roi": config["minimal_roi"].copy() if "minimal_roi" in config else {},
            "stoploss": config.get("stoploss"),
            "stoploss_on_exchange": config.get("order_types", {}).get(
                "stoploss_on_exchange", False
            ),
            "trailing_stop": config.get("trailing_stop"),
            "trailing_stop_positive": config.get("trailing_stop_positive"),
            "trailing_stop_positive_offset": config.get("trailing_stop_positive_offset"),
            "trailing_only_offset_is_reached": config.get("trailing_only_offset_is_reached"),
            "unfilledtimeout": config.get("unfilledtimeout"),
            "use_custom_stoploss": config.get("use_custom_stoploss"),
            "order_types": config.get("order_types"),
            "bot_name": config.get("bot_name", "freqtrade"),
            "timeframe": config.get("timeframe"),
            "timeframe_ms": timeframe_to_msecs(config["timeframe"]) if "timeframe" in config else 0,
            "timeframe_min": (
                timeframe_to_minutes(config["timeframe"]) if "timeframe" in config else 0
            ),
            "exchange": config["exchange"]["name"],
            "strategy": config["strategy"],
            "force_entry_enable": config.get("force_entry_enable", False),
            "exit_pricing": config.get("exit_pricing", {}),
            "entry_pricing": config.get("entry_pricing", {}),
            "state": str(botstate),
            "runmode": config["runmode"].value,
            "position_adjustment_enable": config.get("position_adjustment_enable", False),
            "max_entry_position_adjustment": (
                config.get("max_entry_position_adjustment", -1)
                if config.get("max_entry_position_adjustment") != float("inf")
                else -1
            ),
        }
        return val

    def _rpc_trade_status(self, trade_ids: list[int] | None = None) -> list[dict[str, Any]]:
        """
        以下是RPC后端，其前缀为rpc_以提高对它是远程暴露函数的认识。
        """
        # 获取开放交易
        if trade_ids:
            trades: Sequence[Trade] = Trade.get_trades(trade_filter=Trade.id.in_(trade_ids)).all()
        else:
            trades = Trade.get_open_trades()

        if not trades:
            raise RPCException("没有活跃交易")
        else:
            results = []
            for trade in trades:
                current_profit_fiat: float | None = None
                total_profit_fiat: float | None = None

                # 准备开放订单详情
                oo_details: str | None = ""
                oo_details_lst = [
                    f"({oo.order_type} {oo.side} rem={oo.safe_remaining:.8f})"
                    for oo in trade.open_orders
                    if oo.ft_order_side not in ["stoploss"]
                ]
                oo_details = ", ".join(oo_details_lst)

                total_profit_abs = 0.0
                total_profit_ratio: float | None = None
                # 计算利润并向用户发送消息
                if trade.is_open:
                    try:
                        current_rate: float = self._freqtrade.exchange.get_rate(
                            trade.pair, side="exit", is_short=trade.is_short, refresh=False
                        )
                    except (ExchangeError, PricingError):
                        current_rate = nan
                    if len(trade.select_filled_orders(trade.entry_side)) > 0:
                        current_profit = current_profit_abs = current_profit_fiat = nan
                        if not isnan(current_rate):
                            prof = trade.calculate_profit(current_rate)
                            current_profit = prof.profit_ratio
                            current_profit_abs = prof.profit_abs
                            total_profit_abs = prof.total_profit
                            total_profit_ratio = prof.total_profit_ratio
                    else:
                        current_profit = current_profit_abs = current_profit_fiat = 0.0

                else:
                    # 已关闭交易...
                    current_rate = trade.close_rate or 0.0
                    current_profit = trade.close_profit or 0.0
                    current_profit_abs = trade.close_profit_abs or 0.0

                # 计算法币利润
                if not isnan(current_profit_abs) and self._fiat_converter:
                    current_profit_fiat = self._fiat_converter.convert_amount(
                        current_profit_abs,
                        self._freqtrade.config["stake_currency"],
                        self._freqtrade.config["fiat_display_currency"],
                    )
                    total_profit_fiat = self._fiat_converter.convert_amount(
                        total_profit_abs,
                        self._freqtrade.config["stake_currency"],
                        self._freqtrade.config["fiat_display_currency"],
                    )

                # 计算保证利润（在追踪止损的情况下）
                stop_entry = trade.calculate_profit(trade.stop_loss)

                stoploss_entry_dist = stop_entry.profit_abs
                stoploss_entry_dist_ratio = stop_entry.profit_ratio

                # 计算到止损的距离
                stoploss_current_dist = price_to_precision(
                    trade.stop_loss - current_rate,
                    trade.price_precision,
                    trade.precision_mode_price,
                )
                stoploss_current_dist_ratio = stoploss_current_dist / current_rate

                trade_dict = trade.to_json()
                trade_dict.update(
                    dict(
                        close_profit=trade.close_profit if not trade.is_open else None,
                        current_rate=current_rate,
                        profit_ratio=current_profit,
                        profit_pct=round(current_profit * 100, 2),
                        profit_abs=current_profit_abs,
                        profit_fiat=current_profit_fiat,
                        total_profit_abs=total_profit_abs,
                        total_profit_fiat=total_profit_fiat,
                        total_profit_ratio=total_profit_ratio,
                        stoploss_current_dist=stoploss_current_dist,
                        stoploss_current_dist_ratio=round(stoploss_current_dist_ratio, 8),
                        stoploss_current_dist_pct=round(stoploss_current_dist_ratio * 100, 2),
                        stoploss_entry_dist=stoploss_entry_dist,
                        stoploss_entry_dist_ratio=round(stoploss_entry_dist_ratio, 8),
                        open_orders=oo_details,
                        nr_of_successful_entries=trade.nr_of_successful_entries,
                    )
                )
                results.append(trade_dict)
            return results

    def _rpc_status_table(
        self, stake_currency: str, fiat_display_currency: str
    ) -> tuple[list, list, float, float]:
        """
        :return: 交易列表、列列表、法币利润总和
        """
        nonspot = self._config.get("trading_mode", TradingMode.SPOT) != TradingMode.SPOT
        if not Trade.get_open_trades():
            raise RPCException("没有活跃交易")

        trades_list = []
        fiat_profit_sum = nan
        fiat_total_profit_sum = nan
        for trade in self._rpc_trade_status():
            # 将利润格式化为带正确符号的字符串
            profit = f"{trade['profit_ratio']:.2%}"
            fiat_profit = trade.get("profit_fiat", None)
            if fiat_profit is None or isnan(fiat_profit):
                fiat_profit = trade.get("profit_abs", 0.0)
            if not isnan(fiat_profit):
                profit += f" ({fiat_profit:.2f})"
                fiat_profit_sum = (
                    fiat_profit if isnan(fiat_profit_sum) else fiat_profit_sum + fiat_profit
                )
            total_profit = trade.get("total_profit_fiat", None)
            if total_profit is None or isnan(total_profit):
                total_profit = trade.get("total_profit_abs", 0.0)
            if not isnan(total_profit):
                fiat_total_profit_sum = (
                    total_profit
                    if isnan(fiat_total_profit_sum)
                    else fiat_total_profit_sum + total_profit
                )

            # 格式化活跃订单侧符号
            active_order_side = ""
            orders = trade.get("orders", [])
            if orders:
                active_order_side = ".".join(
                    "*" if (o.get("is_open") and o.get("ft_is_entry")) else "**"
                    for o in orders
                    if o.get("is_open") and o.get("ft_order_side") != "stoploss"
                )

            # 非现货交易方向字符串
            direction_str = ""
            if nonspot:
                leverage = trade.get("leverage", 1.0)
                direction_str = f"{'S' if trade.get('is_short') else 'L'} {leverage:.3g}x"

            detail_trade = [
                f"{trade['trade_id']} {direction_str}",
                f"{trade['pair']}{active_order_side}",
                shorten_date(dt_humanize_delta(dt_from_ts(trade["open_timestamp"]))),
                profit,
            ]

            # 如果启用了头寸调整，添加入场次数
            if self._config.get("position_adjustment_enable", False):
                max_entry_str = ""
                if self._config.get("max_entry_position_adjustment", -1) > 0:
                    max_entry_str = f"/{self._config['max_entry_position_adjustment'] + 1}"
                filled_entries = trade.get("nr_of_successful_entries", 0)
                detail_trade.append(f"{filled_entries}{max_entry_str}")

            trades_list.append(detail_trade)

        columns = [
            "ID L/S" if nonspot else "ID",
            "交易对",
            "起始",
            f"利润 ({fiat_display_currency if self._fiat_converter else stake_currency})",
        ]

        if self._config.get("position_adjustment_enable", False):
            columns.append("入场次数")

        return trades_list, columns, fiat_profit_sum, fiat_total_profit_sum

    def _rpc_timeunit_profit(
        self,
        timescale: int,
        stake_currency: str,
        fiat_display_currency: str,
        timeunit: str = "days",
    ) -> dict[str, Any]:
        """
        :param timeunit: 有效条目为 'days' (天), 'weeks' (周), 'months' (月)
        """
        start_date = datetime.now(timezone.utc).date()
        if timeunit == "weeks":
            # 周
            start_date = start_date - timedelta(days=start_date.weekday())  # 星期一
        if timeunit == "months":
            start_date = start_date.replace(day=1)

        def time_offset(step: int):
            if timeunit == "months":
                return relativedelta(months=step)
            return timedelta(**{timeunit: step})

        if not (isinstance(timescale, int) and timescale > 0):
            raise RPCException("timescale 必须是大于0的整数")

        profit_units: dict[date, dict] = {}
        daily_stake = self._freqtrade.wallets.get_total_stake_amount()

        for day in range(0, timescale):
            profitday = start_date - time_offset(day)
            # 仅查询必要列以提高性能。
            trades = Trade.session.execute(
                select(Trade.close_profit_abs)
                .filter(
                    Trade.is_open.is_(False),
                    Trade.close_date >= profitday,
                    Trade.close_date < (profitday + time_offset(1)),
                )
                .order_by(Trade.close_date)
            ).all()

            curdayprofit = sum(
                trade.close_profit_abs for trade in trades if trade.close_profit_abs is not None
            )
            # 计算此期间的起始余额
            daily_stake = daily_stake - curdayprofit
            profit_units[profitday] = {
                "amount": curdayprofit,
                "daily_stake": daily_stake,
                "rel_profit": round(curdayprofit / daily_stake, 8) if daily_stake > 0 else 0,
                "trades": len(trades),
            }

        data = [
            {
                "date": key,
                "abs_profit": value["amount"],
                "starting_balance": value["daily_stake"],
                "rel_profit": value["rel_profit"],
                "fiat_value": (
                    self._fiat_converter.convert_amount(
                        value["amount"], stake_currency, fiat_display_currency
                    )
                    if self._fiat_converter
                    else 0
                ),
                "trade_count": value["trades"],
            }
            for key, value in profit_units.items()
        ]
        return {
            "stake_currency": stake_currency,
            "fiat_display_currency": fiat_display_currency,
            "data": data,
        }

    def _rpc_trade_history(self, limit: int, offset: int = 0, order_by_id: bool = False) -> dict:
        """返回最近X笔交易"""
        order_by: Any = Trade.id if order_by_id else Trade.close_date.desc()
        if limit:
            trades = Trade.session.scalars(
                Trade.get_trades_query([Trade.is_open.is_(False)])
                .order_by(order_by)
                .limit(limit)
                .offset(offset)
            )
        else:
            trades = Trade.session.scalars(
                Trade.get_trades_query([Trade.is_open.is_(False)]).order_by(Trade.close_date.desc())
            )

        output = [trade.to_json() for trade in trades]
        total_trades = Trade.session.scalar(
            select(func.count(Trade.id)).filter(Trade.is_open.is_(False))
        )

        return {
            "trades": output,
            "trades_count": len(output),
            "offset": offset,
            "total_trades": total_trades,
        }

    def _rpc_stats(self) -> dict[str, Any]:
        """
        为数据库中的交易生成通用统计数据
        """

        def trade_win_loss(trade):
            if trade.close_profit > 0:
                return "wins"
            elif trade.close_profit < 0:
                return "losses"
            else:
                return "draws"

        trades = Trade.get_trades([Trade.is_open.is_(False)], include_orders=False)
        # 持续时间
        dur: dict[str, list[float]] = {"wins": [], "draws": [], "losses": []}
        # 退出原因
        exit_reasons = {}
        for trade in trades:
            if trade.exit_reason not in exit_reasons:
                exit_reasons[trade.exit_reason] = {"wins": 0, "losses": 0, "draws": 0}
            exit_reasons[trade.exit_reason][trade_win_loss(trade)] += 1

            if trade.close_date is not None and trade.open_date is not None:
                trade_dur = (trade.close_date - trade.open_date).total_seconds()
                dur[trade_win_loss(trade)].append(trade_dur)

        wins_dur = sum(dur["wins"]) / len(dur["wins"]) if len(dur["wins"]) > 0 else None
        draws_dur = sum(dur["draws"]) / len(dur["draws"]) if len(dur["draws"]) > 0 else None
        losses_dur = sum(dur["losses"]) / len(dur["losses"]) if len(dur["losses"]) > 0 else None

        durations = {"wins": wins_dur, "draws": draws_dur, "losses": losses_dur}
        return {"exit_reasons": exit_reasons, "durations": durations}

    def _rpc_trade_statistics(
        self, stake_currency: str, fiat_display_currency: str, start_date: datetime | None = None
    ) -> dict[str, Any]:
        """返回累计利润统计数据"""

        start_date = datetime.fromtimestamp(0) if start_date is None else start_date

        trade_filter = (
            Trade.is_open.is_(False) & (Trade.close_date >= start_date)
        ) | Trade.is_open.is_(True)
        trades: Sequence[Trade] = Trade.session.scalars(
            Trade.get_trades_query(trade_filter, include_orders=False).order_by(Trade.id)
        ).all()

        profit_all_coin = []
        profit_all_ratio = []
        profit_closed_coin = []
        profit_closed_ratio = []
        durations = []
        winning_trades = 0
        losing_trades = 0
        winning_profit = 0.0
        losing_profit = 0.0

        for trade in trades:
            current_rate: float = 0.0

            if trade.close_date:
                durations.append((trade.close_date - trade.open_date).total_seconds())

            if not trade.is_open:
                profit_ratio = trade.close_profit or 0.0
                profit_abs = trade.close_profit_abs or 0.0
                profit_closed_coin.append(profit_abs)
                profit_closed_ratio.append(profit_ratio)
                if profit_ratio >= 0:
                    winning_trades += 1
                    winning_profit += profit_abs
                else:
                    losing_trades += 1
                    losing_profit += profit_abs
            else:
                # 获取当前汇率
                if len(trade.select_filled_orders(trade.entry_side)) == 0:
                    # 跳过没有填充订单的交易
                    continue
                try:
                    current_rate = self._freqtrade.exchange.get_rate(
                        trade.pair, side="exit", is_short=trade.is_short, refresh=False
                    )
                except (PricingError, ExchangeError):
                    current_rate = nan
                    profit_ratio = nan
                    profit_abs = nan
                else:
                    _profit = trade.calculate_profit(trade.close_rate or current_rate)

                    profit_ratio = _profit.profit_ratio
                    profit_abs = _profit.total_profit

            profit_all_coin.append(profit_abs)
            profit_all_ratio.append(profit_ratio)

        closed_trade_count = len([t for t in trades if not t.is_open])

        best_pair = Trade.get_best_pair(start_date)
        trading_volume = Trade.get_trading_volume(start_date)

        # 准备显示数据
        profit_closed_coin_sum = round(sum(profit_closed_coin), 8)
        profit_closed_ratio_mean = float(mean(profit_closed_ratio) if profit_closed_ratio else 0.0)
        profit_closed_ratio_sum = sum(profit_closed_ratio) if profit_closed_ratio else 0.0

        profit_closed_fiat = (
            self._fiat_converter.convert_amount(
                profit_closed_coin_sum, stake_currency, fiat_display_currency
            )
            if self._fiat_converter
            else 0
        )

        profit_all_coin_sum = round(sum(profit_all_coin), 8)
        profit_all_ratio_mean = float(mean(profit_all_ratio) if profit_all_ratio else 0.0)
        # 求和不正确 - 整体利润需要基于初始资本
        profit_all_ratio_sum = sum(profit_all_ratio) if profit_all_ratio else 0.0
        starting_balance = self._freqtrade.wallets.get_starting_balance()
        profit_closed_ratio_fromstart = 0.0
        profit_all_ratio_fromstart = 0.0
        if starting_balance:
            profit_closed_ratio_fromstart = profit_closed_coin_sum / starting_balance
            profit_all_ratio_fromstart = profit_all_coin_sum / starting_balance

        profit_factor = winning_profit / abs(losing_profit) if losing_profit else float("inf")

        winrate = (winning_trades / closed_trade_count) if closed_trade_count > 0 else 0

        trades_df = DataFrame(
            [
                {
                    "close_date": format_date(trade.close_date),
                    "close_date_dt": trade.close_date,
                    "profit_abs": trade.close_profit_abs,
                }
                for trade in trades
                if not trade.is_open and trade.close_date
            ]
        )

        expectancy, expectancy_ratio = calculate_expectancy(trades_df)

        drawdown = DrawDownResult()
        if len(trades_df) > 0:
            try:
                drawdown = calculate_max_drawdown(
                    trades_df,
                    value_col="profit_abs",
                    date_col="close_date_dt",
                    starting_balance=starting_balance,
                )
            except ValueError:
                # 如果没有亏损交易，则出现 ValueError。
                pass

        profit_all_fiat = (
            self._fiat_converter.convert_amount(
                profit_all_coin_sum, stake_currency, fiat_display_currency
            )
            if self._fiat_converter
            else 0
        )

        first_date = trades[0].open_date_utc if trades else None
        last_date = trades[-1].open_date_utc if trades else None
        num = float(len(durations) or 1)
        bot_start = KeyValueStore.get_datetime_value("bot_start_time")
        return {
            "profit_closed_coin": profit_closed_coin_sum,
            "profit_closed_percent_mean": round(profit_closed_ratio_mean * 100, 2),
            "profit_closed_ratio_mean": profit_closed_ratio_mean,
            "profit_closed_percent_sum": round(profit_closed_ratio_sum * 100, 2),
            "profit_closed_ratio_sum": profit_closed_ratio_sum,
            "profit_closed_ratio": profit_closed_ratio_fromstart,
            "profit_closed_percent": round(profit_closed_ratio_fromstart * 100, 2),
            "profit_closed_fiat": profit_closed_fiat,
            "profit_all_coin": profit_all_coin_sum,
            "profit_all_percent_mean": round(profit_all_ratio_mean * 100, 2),
            "profit_all_ratio_mean": profit_all_ratio_mean,
            "profit_all_percent_sum": round(profit_all_ratio_sum * 100, 2),
            "profit_all_ratio_sum": profit_all_ratio_sum,
            "profit_all_ratio": profit_all_ratio_fromstart,
            "profit_all_percent": round(profit_all_ratio_fromstart * 100, 2),
            "profit_all_fiat": profit_all_fiat,
            "trade_count": len(trades),
            "closed_trade_count": closed_trade_count,
            "first_trade_date": format_date(first_date),
            "first_trade_humanized": dt_humanize_delta(first_date) if first_date else "",
            "first_trade_timestamp": dt_ts_def(first_date, 0),
            "latest_trade_date": format_date(last_date),
            "latest_trade_humanized": dt_humanize_delta(last_date) if last_date else "",
            "latest_trade_timestamp": dt_ts_def(last_date, 0),
            "avg_duration": str(timedelta(seconds=sum(durations) / num)).split(".")[0],
            "best_pair": best_pair[0] if best_pair else "",
            "best_rate": round(best_pair[1] * 100, 2) if best_pair else 0,  # 已弃用
            "best_pair_profit_ratio": best_pair[1] if best_pair else 0,
            "best_pair_profit_abs": best_pair[2] if best_pair else 0,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "profit_factor": profit_factor,
            "winrate": winrate,
            "expectancy": expectancy,
            "expectancy_ratio": expectancy_ratio,
            "max_drawdown": drawdown.relative_account_drawdown,
            "max_drawdown_abs": drawdown.drawdown_abs,
            "max_drawdown_start": format_date(drawdown.high_date),
            "max_drawdown_start_timestamp": dt_ts_def(drawdown.high_date),
            "max_drawdown_end": format_date(drawdown.low_date),
            "max_drawdown_end_timestamp": dt_ts_def(drawdown.low_date),
            "drawdown_high": drawdown.high_value,
            "drawdown_low": drawdown.low_value,
            "trading_volume": trading_volume,
            "bot_start_timestamp": dt_ts_def(bot_start, 0),
            "bot_start_date": format_date(bot_start),
        }

    def __balance_get_est_stake(
        self, coin: str, stake_currency: str, amount: float, balance: Wallet
    ) -> tuple[float, float]:
        est_stake = 0.0
        est_bot_stake = 0.0
        is_futures = self._config.get("trading_mode", TradingMode.SPOT) == TradingMode.FUTURES
        if coin == self._freqtrade.exchange.get_proxy_coin():
            est_stake = balance.total
            if is_futures:
                # 在期货中，"total" 包含锁定的资金，因此包括所有头寸
                est_stake = balance.free
            est_bot_stake = amount
        else:
            try:
                rate: float | None = self._freqtrade.exchange.get_conversion_rate(
                    coin, stake_currency
                )
                if rate:
                    est_stake = rate * (balance.free if is_futures else balance.total)
                    est_bot_stake = rate * amount

                return est_stake, est_bot_stake
            except (ExchangeError, PricingError) as e:
                logger.warning(f"获取 {coin} 的汇率时出错: {e}")
                pass
        return est_stake, est_bot_stake

    def _rpc_balance(self, stake_currency: str, fiat_display_currency: str) -> dict:
        """返回每个加密货币的当前账户余额"""
        currencies: list[dict] = []
        total = 0.0
        total_bot = 0.0

        open_trades: list[Trade] = Trade.get_open_trades()
        open_assets: dict[str, Trade] = {t.safe_base_currency: t for t in open_trades}
        self._freqtrade.wallets.update(require_update=False)
        starting_capital = self._freqtrade.wallets.get_starting_balance()
        starting_cap_fiat = (
            self._fiat_converter.convert_amount(
                starting_capital, stake_currency, fiat_display_currency
            )
            if self._fiat_converter
            else 0
        )
        coin: str
        balance: Wallet
        for coin, balance in self._freqtrade.wallets.get_all_balances().items():
            if not balance.total and not balance.free:
                continue

            trade = (
                open_assets.get(coin, None)
                if self._freqtrade.trading_mode != TradingMode.FUTURES
                else None
            )
            is_stake_currency = coin == self._freqtrade.exchange.get_proxy_coin()
            is_bot_managed = is_stake_currency or trade is not None
            trade_amount = trade.amount if trade else 0
            if is_stake_currency:
                trade_amount = self._freqtrade.wallets.get_available_stake_amount()

            try:
                est_stake, est_stake_bot = self.__balance_get_est_stake(
                    coin, stake_currency, trade_amount, balance
                )
            except ValueError:
                continue

            total += est_stake

            if is_bot_managed:
                total_bot += est_stake_bot
            currencies.append(
                {
                    "currency": coin,
                    "free": balance.free,
                    "balance": balance.total,
                    "used": balance.used,
                    "bot_owned": trade_amount,
                    "est_stake": est_stake or 0,
                    "est_stake_bot": est_stake_bot if is_bot_managed else 0,
                    "stake": stake_currency,
                    "side": "long",
                    "position": 0,
                    "is_bot_managed": is_bot_managed,
                    "is_position": False,
                }
            )
        symbol: str
        position: PositionWallet
        for symbol, position in self._freqtrade.wallets.get_all_positions().items():
            total += position.collateral
            total_bot += position.collateral

            currencies.append(
                {
                    "currency": symbol,
                    "free": 0,
                    "balance": 0,
                    "used": 0,
                    "position": position.position,
                    "est_stake": position.collateral,
                    "est_stake_bot": position.collateral,
                    "stake": stake_currency,
                    "side": position.side,
                    "is_bot_managed": True,
                    "is_position": True,
                }
            )

        value = (
            self._fiat_converter.convert_amount(total, stake_currency, fiat_display_currency)
            if self._fiat_converter
            else 0
        )
        value_bot = (
            self._fiat_converter.convert_amount(total_bot, stake_currency, fiat_display_currency)
            if self._fiat_converter
            else 0
        )

        trade_count = len(Trade.get_trades_proxy())
        starting_capital_ratio = (total_bot / starting_capital) - 1 if starting_capital else 0.0
        starting_cap_fiat_ratio = (value_bot / starting_cap_fiat) - 1 if starting_cap_fiat else 0.0

        return {
            "currencies": currencies,
            "total": total,
            "total_bot": total_bot,
            "symbol": fiat_display_currency,
            "value": value,
            "value_bot": value_bot,
            "stake": stake_currency,
            "starting_capital": starting_capital,
            "starting_capital_ratio": starting_capital_ratio,
            "starting_capital_pct": round(starting_capital_ratio * 100, 2),
            "starting_capital_fiat": starting_cap_fiat,
            "starting_capital_fiat_ratio": starting_cap_fiat_ratio,
            "starting_capital_fiat_pct": round(starting_cap_fiat_ratio * 100, 2),
            "trade_count": trade_count,
            "note": "模拟余额" if self._freqtrade.config["dry_run"] else "",
        }

    def _rpc_start(self) -> dict[str, str]:
        """启动处理程序"""
        if self._freqtrade.state == State.RUNNING:
            return {"status": "已经在运行"}

        self._freqtrade.state = State.RUNNING
        return {"status": "正在启动交易员..."}

    def _rpc_stop(self) -> dict[str, str]:
        """停止处理程序"""
        if self._freqtrade.state != State.STOPPED:
            self._freqtrade.state = State.STOPPED
            return {"status": "正在停止交易员..."}

        return {"status": "已经停止"}

    def _rpc_reload_config(self) -> dict[str, str]:
        """reload_config 处理程序。"""
        self._freqtrade.state = State.RELOAD_CONFIG
        return {"status": "正在重新加载配置..."}

    def _rpc_pause(self) -> dict[str, str]:
        """
        暂停交易（停止进入新交易），但优雅地处理开放交易的处理器。
        """
        if self._freqtrade.state == State.RUNNING:
            self._freqtrade.state = State.PAUSED

        if self._freqtrade.state == State.STOPPED:
            self._freqtrade.state = State.PAUSED
            return {
                "status": (
                    "正在以暂停状态启动机器人，不会再有入场。 "
                    "运行 /start 以启用入场。"
                )
            }

        return {
            "status": "已暂停，从现在开始将不再有入场。运行 /start 以启用入场。"
        }

    def _rpc_reload_trade_from_exchange(self, trade_id: int) -> dict[str, str]:
        """
        reload_trade_from_exchange 处理程序。
        如果发生手动交互，则从其订单重新加载交易。
        """
        trade = Trade.get_trades(trade_filter=[Trade.id == trade_id]).first()
        if not trade:
            raise RPCException(f"找不到ID为 {trade_id} 的交易。")

        self._freqtrade.handle_onexchange_order(trade)
        return {"status": "从交易所订单重新加载"}

    def __exec_force_exit(
        self, trade: Trade, ordertype: str | None, amount: float | None = None
    ) -> bool:
        # 检查是否有开放订单
        trade_entry_cancelation_registry = []
        for oo in trade.open_orders:
            trade_entry_cancelation_res = {"order_id": oo.order_id, "cancel_state": False}
            order = self._freqtrade.exchange.fetch_order(oo.order_id, trade.pair)

            if order["side"] == trade.entry_side:
                fully_canceled = self._freqtrade.handle_cancel_enter(
                    trade, order, oo, CANCEL_REASON["FORCE_EXIT"]
                )
                trade_entry_cancelation_res["cancel_state"] = fully_canceled
                trade_entry_cancelation_registry.append(trade_entry_cancelation_res)

            if order["side"] == trade.exit_side:
                # 取消订单 - 以便以新价格重新下达。
                self._freqtrade.handle_cancel_exit(trade, order, oo, CANCEL_REASON["FORCE_EXIT"])

        if all(tocr["cancel_state"] is False for tocr in trade_entry_cancelation_registry):
            if trade.has_open_orders:
                # 订单取消失败，所以无法退出。
                return False
            # 获取当前汇率并执行卖出
            current_rate = self._freqtrade.exchange.get_rate(
                trade.pair, side="exit", is_short=trade.is_short, refresh=True
            )
            exit_check = ExitCheckTuple(exit_type=ExitType.FORCE_EXIT)
            order_type = ordertype or self._freqtrade.strategy.order_types.get(
                "force_exit", self._freqtrade.strategy.order_types["exit"]
            )
            sub_amount: float | None = None
            if amount and amount < trade.amount:
                # 部分退出...
                min_exit_stake = self._freqtrade.exchange.get_min_pair_stake_amount(
                    trade.pair, current_rate, trade.stop_loss_pct or 0.0
                )
                remaining = (trade.amount - amount) * current_rate
                if min_exit_stake and remaining < min_exit_stake:
                    raise RPCException(f"剩余金额 {remaining} 会太小。")
                sub_amount = amount

            self._freqtrade.execute_trade_exit(
                trade, current_rate, exit_check, ordertype=order_type, sub_trade_amt=sub_amount
            )

            return True
        return False

    def _rpc_force_exit(
        self, trade_id: str, ordertype: str | None = None, *, amount: float | None = None
    ) -> dict[str, str]:
        """
        forceexit <id> 的处理程序。
        以当前价格出售给定的交易
        """

        if self._freqtrade.state == State.STOPPED:
            raise RPCException("交易员未运行")

        with self._freqtrade._exit_lock:
            if trade_id == "all":
                # 对所有开放订单执行退出
                for trade in Trade.get_open_trades():
                    self.__exec_force_exit(trade, ordertype)
                Trade.commit()
                self._freqtrade.wallets.update()
                return {"result": "已为所有开放交易创建退出订单。"}

            # 查询交易
            trade = Trade.get_trades(
                trade_filter=[
                    Trade.id == trade_id,
                    Trade.is_open.is_(True),
                ]
            ).first()
            if not trade:
                logger.warning("force_exit: 收到无效参数")
                raise RPCException("无效参数")

            result = self.__exec_force_exit(trade, ordertype, amount)
            Trade.commit()
            self._freqtrade.wallets.update()
            if not result:
                raise RPCException("退出交易失败。")
            return {"result": f"已为交易 {trade_id} 创建退出订单。"}

    def _force_entry_validations(self, pair: str, order_side: SignalDirection):
        if not self._freqtrade.config.get("force_entry_enable", False):
            raise RPCException("force_entry 未启用。")

        if self._freqtrade.state != State.RUNNING:
            raise RPCException("交易员未运行")

        if order_side == SignalDirection.SHORT and self._freqtrade.trading_mode == TradingMode.SPOT:
            raise RPCException("不能在现货市场做空。")

        if pair not in self._freqtrade.exchange.get_markets(tradable_only=True):
            raise RPCException("符号不存在或市场不活跃。")
        # 检查交易对报价货币是否等于抵押货币。
        stake_currency = self._freqtrade.config.get("stake_currency")
        if not self._freqtrade.exchange.get_pair_quote_currency(pair) == stake_currency:
            raise RPCException(
                f"选择了错误的交易对。只允许使用抵押货币 {stake_currency} 的交易对。"
            )

    def _rpc_force_entry(
        self,
        pair: str,
        price: float | None,
        *,
        order_type: str | None = None,
        order_side: SignalDirection = SignalDirection.LONG,
        stake_amount: float | None = None,
        enter_tag: str | None = "force_entry",
        leverage: float | None = None,
    ) -> Trade | None:
        """
        forcebuy <asset> <price> 的处理程序
        以给定价格或当前价格买入交易对
        """
        self._force_entry_validations(pair, order_side)

        # 检查是否是有效交易对

        # 检查交易对是否已有开放交易
        trade: Trade | None = Trade.get_trades(
            [Trade.is_open.is_(True), Trade.pair == pair]
        ).first()
        is_short = order_side == SignalDirection.SHORT
        if trade:
            is_short = trade.is_short
            if not self._freqtrade.strategy.position_adjustment_enable:
                raise RPCException(f"交易对 {pair} 已有开放头寸 - ID: {trade.id}")
            if trade.has_open_orders:
                raise RPCException(
                    f"交易对 {pair} 已有开放头寸 - ID: {trade.id} "
                    f"且有开放订单 {','.join(trade.open_orders_ids)}"
                )
        else:
            if Trade.get_open_trade_count() >= self._config["max_open_trades"]:
                raise RPCException("已达到最大交易数量。")

        if not stake_amount:
            # 生成抵押金额
            stake_amount = self._freqtrade.wallets.get_trade_stake_amount(
                pair, self._config["max_open_trades"]
            )

        # 执行买入
        if not order_type:
            order_type = self._freqtrade.strategy.order_types.get(
                "force_entry", self._freqtrade.strategy.order_types["entry"]
            )
        with self._freqtrade._exit_lock:
            if self._freqtrade.execute_entry(
                pair,
                stake_amount,
                price,
                ordertype=order_type,
                trade=trade,
                is_short=is_short,
                enter_tag=enter_tag,
                leverage_=leverage,
                mode="pos_adjust" if trade else "initial",
            ):
                Trade.commit()
                trade = Trade.get_trades([Trade.is_open.is_(True), Trade.pair == pair]).first()
                return trade
            else:
                raise RPCException(f"未能为交易对 {pair} 进入头寸。")

    def _rpc_cancel_open_order(self, trade_id: int):
        if self._freqtrade.state == State.STOPPED:
            raise RPCException("交易员未运行")
        with self._freqtrade._exit_lock:
            # 查询交易
            trade = Trade.get_trades(
                trade_filter=[
                    Trade.id == trade_id,
                    Trade.is_open.is_(True),
                ]
            ).first()
            if not trade:
                logger.warning("cancel_open_order: 收到无效的 trade_id。")
                raise RPCException("无效的 trade_id。")
            if not trade.has_open_orders:
                logger.warning("cancel_open_order: 没有 trade_id 的开放订单。")
                raise RPCException("没有 trade_id 的开放订单。")

            for open_order in trade.open_orders:
                try:
                    order = self._freqtrade.exchange.fetch_order(open_order.order_id, trade.pair)
                except ExchangeError as e:
                    logger.info(f"由于 {e} 无法查询交易 {trade} 的订单。", exc_info=True)
                    raise RPCException("未找到订单。")
                self._freqtrade.handle_cancel_order(
                    order, open_order, trade, CANCEL_REASON["USER_CANCEL"]
                )
            Trade.commit()

    def _rpc_delete(self, trade_id: int) -> dict[str, str | int]:
        """
        delete <id> 的处理程序。
        删除给定交易并关闭可能存在的开放订单。
        """
        with self._freqtrade._exit_lock:
            c_count = 0
            trade = Trade.get_trades(trade_filter=[Trade.id == trade_id]).first()
            if not trade:
                logger.warning("删除交易: 收到无效参数")
                raise RPCException("无效参数")

            # 尝试取消常规订单（如果存在）
            for open_order in trade.open_orders:
                try:
                    self._freqtrade.exchange.cancel_order(open_order.order_id, trade.pair)
                    c_count += 1
                except ExchangeError:
                    pass

            # 取消交易所止损订单...
            if (
                self._freqtrade.strategy.order_types.get("stoploss_on_exchange")
                and trade.has_open_sl_orders
            ):
                for oslo in trade.open_sl_orders:
                    try:
                        self._freqtrade.exchange.cancel_stoploss_order(oslo.order_id, trade.pair)
                        c_count += 1
                    except ExchangeError:
                        pass

            trade.delete()
            self._freqtrade.wallets.update()
            return {
                "result": "成功",
                "trade_id": trade_id,
                "result_msg": f"已删除交易 {trade_id}。已关闭 {c_count} 个开放订单。",
                "cancel_order_count": c_count,
            }

    @custom_data_rpc_wrapper
    def _rpc_list_custom_data(
        self, trade_id: int | None = None, key: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        """
        为特定交易或所有开放交易（如果未提供 `trade_id`）获取自定义数据。
        通过 `limit` 和 `offset` 应用分页。

        返回一个字典数组，每个字典包含：
        - "trade_id": 交易ID (int)
        - "custom_data": 自定义数据字典列表，每个字典包含以下字段：
                "id", "key", "type", "value", "created_at", "updated_at"
        """

        trades: Sequence[Trade]
        if trade_id is None:
            # 获取所有开放交易
            trades = Trade.session.scalars(
                Trade.get_trades_query([Trade.is_open.is_(True)])
                .order_by(Trade.id)
                .limit(limit)
                .offset(offset)
            ).all()
        else:
            trades = Trade.get_trades(trade_filter=[Trade.id == trade_id]).all()

        if not trades:
            raise RPCException(
                f"未找到 ID 为 {trade_id} 的交易。" if trade_id else "未找到任何开放交易。"
            )

        results = []
        for trade in trades:
            # 根据是否提供了特定键，相应地检索自定义数据。
            if key:
                data = trade.get_custom_data_entry(key=key)
                # 如果数据存在，将其封装在列表中，以便输出保持一致。
                custom_data = [data] if data else []
            else:
                custom_data = trade.get_all_custom_data()

            # 如果找到任何自定义数据，则格式化并追加交易结果。
            if custom_data:
                formatted_custom_data = [
                    {
                        "key": data_entry.cd_key,
                        "type": data_entry.cd_type,
                        "value": CustomDataWrapper._convert_custom_data(data_entry).value,
                        "created_at": data_entry.created_at,
                        "updated_at": data_entry.updated_at,
                    }
                    for data_entry in custom_data
                ]
                results.append({"trade_id": trade.id, "custom_data": formatted_custom_data})

            # 处理未在交易中找到自定义数据的情况。
            if not results:
                message_details = ""
                if key:
                    message_details += f"带有键 '{key}' "
                message_details += (
                    f"在交易ID: {trade_id} 中找到。" if trade_id else "在任何开放交易中找到。"
                )
                raise RPCException(f"未找到自定义数据 {message_details}")

        return results

    def _rpc_performance(self) -> list[dict[str, Any]]:
        """
        性能处理程序。
        显示已完成交易的性能统计数据
        """
        pair_rates = Trade.get_overall_performance()

        return pair_rates

    def _rpc_enter_tag_performance(self, pair: str | None) -> list[dict[str, Any]]:
        """
        买入标签性能处理程序。
        显示已完成交易的性能统计数据
        """
        return Trade.get_enter_tag_performance(pair)

    def _rpc_exit_reason_performance(self, pair: str | None) -> list[dict[str, Any]]:
        """
        退出原因性能处理程序。
        显示已完成交易的性能统计数据
        """
        return Trade.get_exit_reason_performance(pair)

    def _rpc_mix_tag_performance(self, pair: str | None) -> list[dict[str, Any]]:
        """
        混合标签（enter_tag + exit_reason）性能处理程序。
        显示已完成交易的性能统计数据
        """
        mix_tags = Trade.get_mix_tag_performance(pair)

        return mix_tags

    def _rpc_count(self) -> dict[str, float]:
        """返回正在运行的交易数量"""
        if self._freqtrade.state == State.STOPPED:
            raise RPCException("交易员未运行")

        trades = Trade.get_open_trades()
        return {
            "current": len(trades),
            "max": (
                int(self._freqtrade.config["max_open_trades"])
                if self._freqtrade.config["max_open_trades"] != float("inf")
                else -1
            ),
            "total_stake": sum((trade.open_rate * trade.amount) for trade in trades),
        }

    def _rpc_locks(self) -> dict[str, Any]:
        """返回当前的锁定"""

        locks = PairLocks.get_pair_locks(None)
        return {"lock_count": len(locks), "locks": [lock.to_json() for lock in locks]}

    def _rpc_delete_lock(
        self, lockid: int | None = None, pair: str | None = None
    ) -> dict[str, Any]:
        """删除特定锁定"""
        locks: Sequence[PairLock] = []

        if pair:
            locks = PairLocks.get_pair_locks(pair)
        if lockid:
            locks = PairLock.session.scalars(select(PairLock).filter(PairLock.id == lockid)).all()

        for lock in locks:
            lock.active = False
            lock.lock_end_time = datetime.now(timezone.utc)

        Trade.commit()

        return self._rpc_locks()

    def _rpc_add_lock(self, pair: str, until: datetime, reason: str | None, side: str) -> PairLock:
        lock = PairLocks.lock_pair(
            pair=pair,
            until=until,
            reason=reason,
            side=side,
        )
        return lock

    def _rpc_whitelist(self) -> dict:
        """返回当前活跃的白名单"""
        res = {
            "method": self._freqtrade.pairlists.name_list,
            "length": len(self._freqtrade.active_pair_whitelist),
            "whitelist": self._freqtrade.active_pair_whitelist,
        }
        return res

    def _rpc_blacklist_delete(self, delete: list[str]) -> dict:
        """从当前活跃的黑名单中移除交易对"""
        errors = {}
        for pair in delete:
            if pair in self._freqtrade.pairlists.blacklist:
                self._freqtrade.pairlists.blacklist.remove(pair)
            else:
                errors[pair] = {"error_msg": f"交易对 {pair} 不在当前黑名单中。"}
        resp = self._rpc_blacklist()
        resp["errors"] = errors
        return resp

    def _rpc_blacklist(self, add: list[str] | None = None) -> dict:
        """返回当前活跃的黑名单"""
        errors = {}
        if add:
            for pair in add:
                if pair not in self._freqtrade.pairlists.blacklist:
                    try:
                        expand_pairlist([pair], list(self._freqtrade.exchange.get_markets().keys()))
                        self._freqtrade.pairlists.blacklist.append(pair)

                    except ValueError:
                        errors[pair] = {"error_msg": f"交易对 {pair} 不是有效的通配符。"}
                else:
                    errors[pair] = {"error_msg": f"交易对 {pair} 已在交易对列表中。"}

        res = {
            "method": self._freqtrade.pairlists.name_list,
            "length": len(self._freqtrade.pairlists.blacklist),
            "blacklist": self._freqtrade.pairlists.blacklist,
            "blacklist_expanded": self._freqtrade.pairlists.expanded_blacklist,
            "errors": errors,
        }
        return res

    @staticmethod
    def _rpc_get_logs(limit: int | None) -> dict[str, Any]:
        """返回最近X条日志"""
        if limit:
            buffer = bufferHandler.buffer[-limit:]
        else:
            buffer = bufferHandler.buffer
        records = [
            [
                format_date(datetime.fromtimestamp(r.created)),
                r.created * 1000,
                r.name,
                r.levelname,
                r.message + ("\n" + r.exc_text if r.exc_text else ""),
            ]
            for r in buffer
            if hasattr(r, "message")
        ]

        # 日志格式：
        # [格式化后的日志时间, 日志纪元, 日志器名称, 日志级别, 消息 \n + 异常]
        # 例如：["2020-08-27 11:35:01", 1598520901097.9397,
        #       "freqtrade.worker", "INFO", "正在启动 worker develop"]

        return {"log_count": len(records), "logs": records}

    @staticmethod
    def _convert_dataframe_to_dict(
        strategy: str,
        pair: str,
        timeframe: str,
        dataframe: DataFrame,
        last_analyzed: datetime,
        selected_cols: list[str] | None,
        annotations: list[AnnotationType],
    ) -> dict[str, Any]:
        has_content = len(dataframe) != 0
        dataframe_columns = list(dataframe.columns)
        signals = {
            "enter_long": 0,
            "exit_long": 0,
            "enter_short": 0,
            "exit_short": 0,
        }
        if has_content:
            if selected_cols is not None:
                # 确保OHLCV列始终存在
                cols_set = set(DEFAULT_DATAFRAME_COLUMNS + list(signals.keys()) + selected_cols)
                df_cols = [col for col in dataframe_columns if col in cols_set]
                dataframe = dataframe.loc[:, df_cols]

            dataframe.loc[:, "__date_ts"] = dataframe.loc[:, "date"].astype(int64) // 1000 // 1000
            # 将信号收盘价移动到单独的列，以便于绘图
            for sig_type in signals.keys():
                if sig_type in dataframe.columns:
                    mask = dataframe[sig_type] == 1
                    signals[sig_type] = int(mask.sum())
                    dataframe.loc[mask, f"_{sig_type}_signal_close"] = dataframe.loc[mask, "close"]

            # 临时修复，直到此问题得到解决：
            # https://github.com/pandas-dev/pandas/issues/45836
            datetime_types = ["datetime", "datetime64", "datetime64[ns, UTC]"]
            date_columns = dataframe.select_dtypes(include=datetime_types)
            for date_column in date_columns:
                # 将 NaT 替换为 `None`
                dataframe[date_column] = dataframe[date_column].astype(object).replace({NaT: None})

            dataframe = dataframe.replace({inf: None, -inf: None, nan: None})

        res = {
            "pair": pair,
            "timeframe": timeframe,
            "timeframe_ms": timeframe_to_msecs(timeframe),
            "strategy": strategy,
            "all_columns": dataframe_columns,
            "columns": list(dataframe.columns),
            "data": dataframe.values.tolist(),
            "length": len(dataframe),
            "buy_signals": signals["enter_long"],  # 已弃用
            "sell_signals": signals["exit_long"],  # 已弃用
            "enter_long_signals": signals["enter_long"],
            "exit_long_signals": signals["exit_long"],
            "enter_short_signals": signals["enter_short"],
            "exit_short_signals": signals["exit_short"],
            "last_analyzed": last_analyzed,
            "last_analyzed_ts": int(last_analyzed.timestamp()),
            "data_start": "",
            "data_start_ts": 0,
            "data_stop": "",
            "data_stop_ts": 0,
            "annotations": annotations,
        }
        if has_content:
            res.update(
                {
                    "data_start": str(dataframe.iloc[0]["date"]),
                    "data_start_ts": int(dataframe.iloc[0]["__date_ts"]),
                    "data_stop": str(dataframe.iloc[-1]["date"]),
                    "data_stop_ts": int(dataframe.iloc[-1]["__date_ts"]),
                }
            )
        return res

    def _rpc_analysed_dataframe(
        self, pair: str, timeframe: str, limit: int | None, selected_cols: list[str] | None
    ) -> dict[str, Any]:
        """字典形式的分析数据框"""

        _data, last_analyzed = self.__rpc_analysed_dataframe_raw(pair, timeframe, limit)
        annotations = self._freqtrade.strategy.ft_plot_annotations(pair=pair, dataframe=_data)

        return RPC._convert_dataframe_to_dict(
            self._freqtrade.config["strategy"],
            pair,
            timeframe,
            _data,
            last_analyzed,
            selected_cols,
            annotations,
        )

    def __rpc_analysed_dataframe_raw(
        self, pair: str, timeframe: str, limit: int | None
    ) -> tuple[DataFrame, datetime]:
        """
        从数据提供者获取数据框和最后分析时间

        :param pair: 要获取的交易对
        :param timeframe: 要获取数据的时间框架
        :param limit: 数据框中的蜡烛图数量
        """
        _data, last_analyzed = self._freqtrade.dataprovider.get_analyzed_dataframe(pair, timeframe)
        _data = _data.copy()

        if limit:
            _data = _data.iloc[-limit:]

        return _data, last_analyzed

    def _ws_all_analysed_dataframes(
        self, pairlist: list[str], limit: int | None
    ) -> Generator[dict[str, Any], None, None]:
        """
        获取交易对列表中每个交易对的分析数据框。
        如果指定，只返回每个数据框最近 `limit` 根蜡烛图。

        :param pairlist: 要获取的交易对列表
        :param limit: 如果是整数，限制数据框的大小
                      如果是一个字符串日期时间列表，只返回这些蜡烛图
        :returns: 一个字典生成器，包含键、数据框和最后分析时间戳
        """
        timeframe = self._freqtrade.config["timeframe"]
        candle_type = self._freqtrade.config.get("candle_type_def", CandleType.SPOT)

        for pair in pairlist:
            dataframe, last_analyzed = self.__rpc_analysed_dataframe_raw(pair, timeframe, limit)

            yield {"key": (pair, timeframe, candle_type), "df": dataframe, "la": last_analyzed}

    def _ws_request_analyzed_df(self, limit: int | None = None, pair: str | None = None):
        """WebSocket的历史分析数据框"""
        pairlist = [pair] if pair else self._freqtrade.active_pair_whitelist

        return self._ws_all_analysed_dataframes(pairlist, limit)

    def _ws_request_whitelist(self):
        """WebSocket的白名单数据"""
        return self._freqtrade.active_pair_whitelist

    @staticmethod
    def _rpc_analysed_history_full(
        config: Config,
        pair: str,
        timeframe: str,
        exchange: Exchange,
        selected_cols: list[str] | None,
        live: bool,
    ) -> dict[str, Any]:
        timerange_parsed = TimeRange.parse_timerange(config.get("timerange"))

        from freqtrade.data.converter import trim_dataframe
        from freqtrade.data.dataprovider import DataProvider
        from freqtrade.resolvers.strategy_resolver import StrategyResolver

        strategy_name = ""
        startup_candles = 0
        if config.get("strategy"):
            strategy = StrategyResolver.load_strategy(config)
            startup_candles = strategy.startup_candle_count
            strategy_name = strategy.get_strategy_name()

        if live:
            data = exchange.get_historic_ohlcv(
                pair=pair,
                timeframe=timeframe,
                since_ms=timerange_parsed.startts * 1000
                if timerange_parsed.startts
                else dt_ts(dt_now() - timedelta(days=30)),
                is_new_pair=True,  # 历史数据从未可用 - 所以总是视为新交易对
                candle_type=config.get("candle_type_def", CandleType.SPOT),
                until_ms=timerange_parsed.stopts,
            )
        else:
            _data = load_data(
                datadir=config["datadir"],
                pairs=[pair],
                timeframe=timeframe,
                timerange=timerange_parsed,
                data_format=config["dataformat_ohlcv"],
                candle_type=config.get("candle_type_def", CandleType.SPOT),
                startup_candles=startup_candles,
            )
            if pair not in _data:
                raise RPCException(
                    f"未找到 {pair}, {timeframe} 在 {config.get('timerange')} 中的数据。"
                )
            data = _data[pair]

        annotations = []
        if config.get("strategy"):
            strategy.dp = DataProvider(config, exchange=exchange, pairlists=None)
            strategy.ft_bot_start()

            df_analyzed = strategy.analyze_ticker(data, {"pair": pair})
            df_analyzed = trim_dataframe(
                df_analyzed, timerange_parsed, startup_candles=startup_candles
            )
            annotations = strategy.ft_plot_annotations(pair=pair, dataframe=df_analyzed)

        else:
            df_analyzed = data

        return RPC._convert_dataframe_to_dict(
            strategy_name,
            pair,
            timeframe,
            df_analyzed.copy(),
            dt_now(),
            selected_cols,
            annotations,
        )

    def _rpc_plot_config(self) -> dict[str, Any]:
        if (
            self._freqtrade.strategy.plot_config
            and "subplots" not in self._freqtrade.strategy.plot_config
        ):
            self._freqtrade.strategy.plot_config["subplots"] = {}
        return self._freqtrade.strategy.plot_config

    @staticmethod
    def _rpc_plot_config_with_strategy(config: Config) -> dict[str, Any]:
        from freqtrade.resolvers.strategy_resolver import StrategyResolver

        strategy = StrategyResolver.load_strategy(config)
        # 手动加载超参数，因为我们不调用机器人启动回调。
        strategy.ft_load_hyper_params(False)

        if strategy.plot_config and "subplots" not in strategy.plot_config:
            strategy.plot_config["subplots"] = {}
        return strategy.plot_config

    @staticmethod
    def _rpc_sysinfo() -> dict[str, Any]:
        return {
            "cpu_pct": psutil.cpu_percent(interval=1, percpu=True),
            "ram_pct": psutil.virtual_memory().percent,
        }

    def health(self) -> dict[str, str | int | None]:
        last_p = self._freqtrade.last_process
        res: dict[str, None | str | int] = {
            "last_process": None,
            "last_process_loc": None,
            "last_process_ts": None,
            "bot_start": None,
            "bot_start_loc": None,
            "bot_start_ts": None,
            "bot_startup": None,
            "bot_startup_loc": None,
            "bot_startup_ts": None,
        }

        if last_p is not None:
            res.update(
                {
                    "last_process": str(last_p),
                    "last_process_loc": format_date(last_p.astimezone(tzlocal())),
                    "last_process_ts": int(last_p.timestamp()),
                }
            )

        if bot_start := KeyValueStore.get_datetime_value("bot_start_time"):
            res.update(
                {
                    "bot_start": str(bot_start),
                    "bot_start_loc": format_date(bot_start.astimezone(tzlocal())),
                    "bot_start_ts": int(bot_start.timestamp()),
                }
            )
        if bot_startup := KeyValueStore.get_datetime_value("startup_time"):
            res.update(
                {
                    "bot_startup": str(bot_startup),
                    "bot_startup_loc": format_date(bot_startup.astimezone(tzlocal())),
                    "bot_startup_ts": int(bot_startup.timestamp()),
                }
            )

        return res

    def _update_market_direction(self, direction: MarketDirection) -> None:
        self._freqtrade.strategy.market_direction = direction

    def _get_market_direction(self) -> MarketDirection:
        return self._freqtrade.strategy.market_direction
