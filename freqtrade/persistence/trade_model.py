"""
该模块包含将交易持久化到SQLite的类
"""

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isclose
from typing import Any, ClassVar, Optional, cast

from sqlalchemy import (
    Enum,
    Float,
    ForeignKey,
    Integer,
    ScalarResult,
    Select,
    String,
    UniqueConstraint,
    case,
    desc,
    func,
    select,
)
from sqlalchemy.orm import Mapped, lazyload, mapped_column, relationship, validates
from typing_extensions import Self

from freqtrade.constants import (
    CANCELED_EXCHANGE_STATES,
    CUSTOM_TAG_MAX_LENGTH,
    DATETIME_PRINT_FORMAT,
    MATH_CLOSE_PREC,
    NON_OPEN_EXCHANGE_STATES,
    BuySell,
    LongShort,
)
from freqtrade.enums import ExitType, TradingMode
from freqtrade.exceptions import DependencyException, OperationalException
from freqtrade.exchange import (
    ROUND_DOWN,
    ROUND_UP,
    amount_to_contract_precision,
    price_to_precision,
)
from freqtrade.exchange.exchange_types import CcxtOrder
from freqtrade.leverage import interest
from freqtrade.misc import safe_value_fallback
from freqtrade.persistence.base import ModelBase, SessionType
from freqtrade.persistence.custom_data import CustomDataWrapper, _CustomData
from freqtrade.util import FtPrecise, dt_from_ts, dt_now, dt_ts, dt_ts_none


logger = logging.getLogger(__name__)


@dataclass
class ProfitStruct:
    profit_abs: float
    profit_ratio: float
    total_profit: float
    total_profit_ratio: float


class Order(ModelBase):
    """
    订单数据库模型
    记录在交易所下的所有订单

    与交易的一对多关系：
      - 一笔交易可以有多个订单
      - 一个订单只能关联一笔交易

    镜像CCXT订单结构
    """

    __tablename__ = "orders"
    __allow_unmapped__ = True
    session: ClassVar[SessionType]

    # Uniqueness should be ensured over pair, order_id
    # its likely that order_id is unique per Pair on some exchanges.
    __table_args__ = (UniqueConstraint("ft_pair", "order_id", name="_order_pair_order_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ft_trade_id: Mapped[int] = mapped_column(Integer, ForeignKey("trades.id"), index=True)

    _trade_live: Mapped["Trade"] = relationship("Trade", back_populates="orders", lazy="immediate")
    _trade_bt: "LocalTrade" = None  # type: ignore

    # order_side can only be 'buy', 'sell' or 'stoploss'
    ft_order_side: Mapped[str] = mapped_column(String(25), nullable=False)
    ft_pair: Mapped[str] = mapped_column(String(25), nullable=False)
    ft_is_open: Mapped[bool] = mapped_column(nullable=False, default=True, index=True)
    ft_amount: Mapped[float] = mapped_column(Float(), nullable=False)
    ft_price: Mapped[float] = mapped_column(Float(), nullable=False)
    ft_cancel_reason: Mapped[str] = mapped_column(String(CUSTOM_TAG_MAX_LENGTH), nullable=True)

    order_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    status: Mapped[str | None] = mapped_column(String(255), nullable=True)
    symbol: Mapped[str | None] = mapped_column(String(25), nullable=True)
    order_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    side: Mapped[str] = mapped_column(String(25), nullable=True)
    price: Mapped[float | None] = mapped_column(Float(), nullable=True)
    average: Mapped[float | None] = mapped_column(Float(), nullable=True)
    amount: Mapped[float | None] = mapped_column(Float(), nullable=True)
    filled: Mapped[float | None] = mapped_column(Float(), nullable=True)
    remaining: Mapped[float | None] = mapped_column(Float(), nullable=True)
    cost: Mapped[float | None] = mapped_column(Float(), nullable=True)
    stop_price: Mapped[float | None] = mapped_column(Float(), nullable=True)
    order_date: Mapped[datetime] = mapped_column(nullable=True, default=dt_now)
    order_filled_date: Mapped[datetime | None] = mapped_column(nullable=True)
    order_update_date: Mapped[datetime | None] = mapped_column(nullable=True)
    funding_fee: Mapped[float | None] = mapped_column(Float(), nullable=True)

    # Fee if paid in base currency
    ft_fee_base: Mapped[float | None] = mapped_column(Float(), nullable=True)
    ft_order_tag: Mapped[str | None] = mapped_column(String(CUSTOM_TAG_MAX_LENGTH), nullable=True)

    @property
    def order_date_utc(self) -> datetime:
        """带有时区信息（UTC）的订单日期"""
        return self.order_date.replace(tzinfo=timezone.utc)

    @property
    def order_filled_utc(self) -> datetime | None:
        """带有时区信息（UTC）的最后订单日期"""
        return (
            self.order_filled_date.replace(tzinfo=timezone.utc) if self.order_filled_date else None
        )

    @property
    def safe_amount(self) -> float:
        return self.amount or self.ft_amount

    @property
    def safe_placement_price(self) -> float:
        """Price at which the order was placed"""
        return self.price or self.stop_price or self.ft_price

    @property
    def safe_price(self) -> float:
        return self.average or self.price or self.stop_price or self.ft_price

    @property
    def safe_filled(self) -> float:
        return self.filled if self.filled is not None else 0.0

    @property
    def safe_cost(self) -> float:
        return self.cost or 0.0

    @property
    def safe_remaining(self) -> float:
        return (
            self.remaining
            if self.remaining is not None
            else self.safe_amount - (self.filled or 0.0)
        )

    @property
    def safe_fee_base(self) -> float:
        return self.ft_fee_base or 0.0

    @property
    def safe_amount_after_fee(self) -> float:
        return self.safe_filled - self.safe_fee_base

    @property
    def trade(self) -> "LocalTrade":
        return self._trade_bt or self._trade_live

    @property
    def stake_amount(self) -> float:
        """此订单使用的质押货币金额"""
        return float(
            FtPrecise(self.safe_amount)
            * FtPrecise(self.safe_price)
            / FtPrecise(self.trade.leverage)
        )

    @property
    def stake_amount_filled(self) -> float:
        """此订单已填充的质押货币金额"""
        return float(
            FtPrecise(self.safe_filled)
            * FtPrecise(self.safe_price)
            / FtPrecise(self.trade.leverage)
        )

    def __repr__(self):
        return (
            f"订单(id={self.id}, 交易={self.ft_trade_id}, 订单ID={self.order_id}, "
            f"侧边={self.side}, 已填充={self.safe_filled}, 价格={self.safe_price}, "
            f"金额={self.amount}, "
            f"状态={self.status}, 日期={self.order_date_utc:{DATETIME_PRINT_FORMAT}})"
        )

    def update_from_ccxt_object(self, order):
        """
        从ccxt响应更新订单
        仅在ccxt响应中包含字段时才更新
        """
        if self.order_id != str(order["id"]):
            raise DependencyException("订单ID不匹配")

        self.status = safe_value_fallback(order, "status", default_value=self.status)
        self.symbol = safe_value_fallback(order, "symbol", default_value=self.symbol)
        self.order_type = safe_value_fallback(order, "type", default_value=self.order_type)
        self.side = safe_value_fallback(order, "side", default_value=self.side)
        self.price = safe_value_fallback(order, "price", default_value=self.price)
        self.amount = safe_value_fallback(order, "amount", default_value=self.amount)
        self.filled = safe_value_fallback(order, "filled", default_value=self.filled)
        self.average = safe_value_fallback(order, "average", default_value=self.average)
        self.remaining = safe_value_fallback(order, "remaining", default_value=self.remaining)
        self.cost = safe_value_fallback(order, "cost", default_value=self.cost)
        self.stop_price = safe_value_fallback(order, "stopPrice", default_value=self.stop_price)
        order_date = safe_value_fallback(order, "timestamp")
        if order_date:
            self.order_date = dt_from_ts(order_date)
        elif not self.order_date:
            self.order_date = dt_now()

        self.ft_is_open = True
        if self.status in NON_OPEN_EXCHANGE_STATES:
            self.ft_is_open = False
            if (order.get("filled", 0.0) or 0.0) > 0 and not self.order_filled_date:
                self.order_filled_date = dt_from_ts(
                    safe_value_fallback(order, "lastTradeTimestamp", default_value=dt_ts())
                )
        self.order_update_date = datetime.now(timezone.utc)

    def to_ccxt_object(self, stopPriceName: str = "stopPrice") -> dict[str, Any]:
        order: dict[str, Any] = {
            "id": self.order_id,
            "symbol": self.ft_pair,
            "price": self.price,
            "average": self.average,
            "amount": self.amount,
            "cost": self.cost,
            "type": self.order_type,
            "side": self.ft_order_side,
            "filled": self.filled,
            "remaining": self.remaining,
            "datetime": self.order_date_utc.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "timestamp": int(self.order_date_utc.timestamp() * 1000),
            "status": self.status,
            "fee": None,
            "info": {},
        }
        if self.ft_order_side == "stoploss":
            order.update(
                {
                    stopPriceName: self.stop_price,
                    "ft_order_type": "stoploss",
                }
            )

        return order

    def to_json(self, entry_side: str, minified: bool = False) -> dict[str, Any]:
        """
        :param minified: 如果为True，则仅返回数据的子集。
                         仅用于回测。
        """
        resp = {
            "amount": self.safe_amount,
            "safe_price": self.safe_price,
            "ft_order_side": self.ft_order_side,
            "order_filled_timestamp": dt_ts_none(self.order_filled_utc),
            "ft_is_entry": self.ft_order_side == entry_side,
            "ft_order_tag": self.ft_order_tag,
            "cost": self.cost if self.cost else 0,
        }
        if not minified:
            resp.update(
                {
                    "pair": self.ft_pair,
                    "order_id": self.order_id,
                    "status": self.status,
                    "average": round(self.average, 8) if self.average else 0,
                    "filled": self.filled,
                    "is_open": self.ft_is_open,
                    "order_date": (
                        self.order_date.strftime(DATETIME_PRINT_FORMAT) if self.order_date else None
                    ),
                    "order_timestamp": (
                        int(self.order_date.replace(tzinfo=timezone.utc).timestamp() * 1000)
                        if self.order_date
                        else None
                    ),
                    "order_filled_date": (
                        self.order_filled_date.strftime(DATETIME_PRINT_FORMAT)
                        if self.order_filled_date
                        else None
                    ),
                    "order_type": self.order_type,
                    "price": self.price,
                    "remaining": self.remaining,
                    "ft_fee_base": self.ft_fee_base,
                    "funding_fee": self.funding_fee,
                }
            )
        return resp

    def close_bt_order(self, close_date: datetime, trade: "LocalTrade"):
        self.order_filled_date = close_date
        self.filled = self.amount
        self.remaining = 0
        self.status = "closed"
        self.ft_is_open = False
        # Assign funding fees to Order.
        # Assumes backtesting will use date_last_filled_utc to calculate future funding fees.
        self.funding_fee = trade.funding_fee_running
        trade.funding_fee_running = 0.0

        if self.ft_order_side == trade.entry_side and self.price:
            trade.open_rate = self.price
            trade.recalc_trade_from_orders()
            if trade.nr_of_successful_entries == 1:
                trade.initial_stop_loss_pct = None
                trade.is_stop_loss_trailing = False
            trade.adjust_stop_loss(trade.open_rate, trade.stop_loss_pct)

    @staticmethod
    def update_orders(orders: list["Order"], order: CcxtOrder):
        """
        获取所有非关闭订单 - 在尝试批量更新订单时很有用
        """
        if not isinstance(order, dict):
            logger.warning(f"{order} 不是一个有效的响应对象。")
            return

        filtered_orders = [o for o in orders if o.order_id == order.get("id")]
        if filtered_orders:
            oobj = filtered_orders[0]
            oobj.update_from_ccxt_object(order)
            Trade.commit()
        else:
            logger.warning(f"未找到订单 {order}。")

    @classmethod
    def parse_from_ccxt_object(
        cls,
        order: CcxtOrder,
        pair: str,
        side: str,
        amount: float | None = None,
        price: float | None = None,
    ) -> Self:
        """
        从ccxt对象解析订单并返回一个新的订单对象。
        用于覆盖金额和价格的可选支持仅用于简化测试。
        """
        o = cls(
            order_id=str(order["id"]),
            ft_order_side=side,
            ft_pair=pair,
            ft_amount=amount or order.get("amount", None) or 0.0,
            ft_price=price or order.get("price", None),
        )

        o.update_from_ccxt_object(order)
        return o

    @staticmethod
    def get_open_orders() -> Sequence["Order"]:
        """
        从数据库检索未结订单
        :return: 未结订单列表
        """
        return Order.session.scalars(select(Order).filter(Order.ft_is_open.is_(True))).all()

    @staticmethod
    def order_by_id(order_id: str) -> Optional["Order"]:
        """
        根据order_id检索订单
        :return: 订单或None
        """
        return Order.session.scalars(select(Order).filter(Order.order_id == order_id)).first()


class LocalTrade:
    """
    交易数据库模型。
    用于回测 - 必须与Trade模型对齐！
    """

    use_db: bool = False
    # 回测的交易容器
    bt_trades: list["LocalTrade"] = []
    bt_trades_open: list["LocalTrade"] = []
    # trades_open 的副本 - 但按交易对索引
    bt_trades_open_pp: dict[str, list["LocalTrade"]] = defaultdict(list)
    bt_open_open_trade_count: int = 0
    bt_total_profit: float = 0
    realized_profit: float = 0

    id: int = 0

    orders: list[Order] = []

    exchange: str = ""
    pair: str = ""
    base_currency: str | None = ""
    stake_currency: str | None = ""
    is_open: bool = True
    fee_open: float = 0.0
    fee_open_cost: float | None = None
    fee_open_currency: str | None = ""
    fee_close: float | None = 0.0
    fee_close_cost: float | None = None
    fee_close_currency: str | None = ""
    open_rate: float = 0.0
    open_rate_requested: float | None = None
    # open_trade_value - 通过_calc_open_trade_value计算
    open_trade_value: float = 0.0
    close_rate: float | None = None
    close_rate_requested: float | None = None
    close_profit: float | None = None
    close_profit_abs: float | None = None
    stake_amount: float = 0.0
    max_stake_amount: float | None = 0.0
    amount: float = 0.0
    amount_requested: float | None = None
    open_date: datetime
    close_date: datetime | None = None
    # absolute value of the stop loss
    stop_loss: float = 0.0
    # percentage value of the stop loss
    stop_loss_pct: float | None = 0.0
    # absolute value of the initial stop loss
    initial_stop_loss: float | None = 0.0
    # percentage value of the initial stop loss
    initial_stop_loss_pct: float | None = None
    is_stop_loss_trailing: bool = False
    # absolute value of the highest reached price
    max_rate: float | None = None
    # Lowest price reached
    min_rate: float | None = None
    exit_reason: str | None = ""
    exit_order_status: str | None = ""
    strategy: str | None = ""
    enter_tag: str | None = None
    timeframe: int | None = None

    trading_mode: TradingMode = TradingMode.SPOT
    amount_precision: float | None = None
    price_precision: float | None = None
    precision_mode: int | None = None
    precision_mode_price: int | None = None
    contract_size: float | None = None

    # Leverage trading properties
    liquidation_price: float | None = None
    is_short: bool = False
    leverage: float = 1.0

    # Margin trading properties
    interest_rate: float = 0.0

    # Futures properties
    funding_fees: float | None = None
    # 用于跟踪持续的资金费 - 从最后一次填充订单到现在
    # 不应用于计算！
    funding_fee_running: float | None = None
    # v 2 -> 修正杠杆交易的max_stake_amount计算
    record_version: int = 2

    @property
    def stoploss_or_liquidation(self) -> float:
        if self.liquidation_price:
            if self.is_short:
                return min(self.stop_loss, self.liquidation_price)
            else:
                return max(self.stop_loss, self.liquidation_price)

        return self.stop_loss

    @property
    def buy_tag(self) -> str | None:
        """
        buy_tag（旧）和enter_tag（新）之间的兼容性
        buy_tag 已被视为废弃
        """
        return self.enter_tag

    @property
    def has_no_leverage(self) -> bool:
        """如果这是一笔非杠杆、非做空交易，则返回True"""
        return (self.leverage == 1.0 or self.leverage is None) and not self.is_short

    @property
    def borrowed(self) -> float:
        """
        杠杆交易从交易所借入的货币金额
        如果是多头交易，金额以基础货币计
        如果是空头交易，金额以被交易的另一种货币计
        """
        if self.has_no_leverage:
            return 0.0
        elif not self.is_short:
            return (self.amount * self.open_rate) * ((self.leverage - 1) / self.leverage)
        else:
            return self.amount

    @property
    def _date_last_filled_utc(self) -> datetime | None:
        """最后一次填充订单的日期"""
        orders = self.select_filled_orders()
        if orders:
            return max(o.order_filled_utc for o in orders if o.order_filled_utc)
        return None

    @property
    def date_last_filled_utc(self) -> datetime:
        """最后一次填充订单的日期 - 如果没有订单被填充，则为open_date"""
        dt_last_filled = self._date_last_filled_utc
        if not dt_last_filled:
            return self.open_date_utc
        return max([self.open_date_utc, dt_last_filled])

    @property
    def date_entry_fill_utc(self) -> datetime | None:
        """第一次填充订单的日期"""
        orders = self.select_filled_orders(self.entry_side)
        if orders and len(
            filled_date := [o.order_filled_utc for o in orders if o.order_filled_utc]
        ):
            return min(filled_date)
        return None

    @property
    def open_date_utc(self):
        return self.open_date.replace(tzinfo=timezone.utc)

    @property
    def stoploss_last_update_utc(self):
        if self.has_open_sl_orders:
            return max(o.order_date_utc for o in self.open_sl_orders)
        return None

    @property
    def close_date_utc(self):
        return self.close_date.replace(tzinfo=timezone.utc) if self.close_date else None

    @property
    def entry_side(self) -> str:
        if self.is_short:
            return "sell"
        else:
            return "buy"

    @property
    def exit_side(self) -> BuySell:
        if self.is_short:
            return "buy"
        else:
            return "sell"

    @property
    def trade_direction(self) -> LongShort:
        if self.is_short:
            return "short"
        else:
            return "long"

    @property
    def safe_base_currency(self) -> str:
        """
        Compatibility layer for asset - which can be empty for old trades.
        """
        try:
            return self.base_currency or self.pair.split("/")[0]
        except IndexError:
            return ""

    @property
    def safe_quote_currency(self) -> str:
        """
        Compatibility layer for asset - which can be empty for old trades.
        """
        try:
            return self.stake_currency or self.pair.split("/")[1].split(":")[0]
        except IndexError:
            return ""

    @property
    def open_orders(self) -> list[Order]:
        """
        此交易的所有未结订单（不包括止损订单）
        """
        return [o for o in self.orders if o.ft_is_open and o.ft_order_side != "stoploss"]

    @property
    def has_open_orders(self) -> bool:
        """
        如果此交易有未结订单（不包括止损订单），则为True
        """
        open_orders_wo_sl = [
            o for o in self.orders if o.ft_order_side not in ["stoploss"] and o.ft_is_open
        ]
        return len(open_orders_wo_sl) > 0

    @property
    def has_open_position(self) -> bool:
        """
        如果此交易有未平仓位，则为True
        """
        return self.amount > 0

    @property
    def open_sl_orders(self) -> list[Order]:
        """
        此交易的所有未结止损订单
        """
        return [o for o in self.orders if o.ft_order_side in ["stoploss"] and o.ft_is_open]

    @property
    def has_open_sl_orders(self) -> bool:
        """
        如果此交易有未结止损订单，则为True
        """
        open_sl_orders = [
            o for o in self.orders if o.ft_order_side in ["stoploss"] and o.ft_is_open
        ]
        return len(open_sl_orders) > 0

    @property
    def sl_orders(self) -> list[Order]:
        """
        此交易的所有止损订单
        """
        return [o for o in self.orders if o.ft_order_side in ["stoploss"]]

    @property
    def open_orders_ids(self) -> list[str]:
        open_orders_ids_wo_sl = [
            oo.order_id for oo in self.open_orders if oo.ft_order_side not in ["stoploss"]
        ]
        return open_orders_ids_wo_sl

    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.recalc_open_trade_value()
        self.orders = []
        if self.trading_mode == TradingMode.MARGIN and self.interest_rate is None:
            raise OperationalException(
                f"{self.trading_mode} 交易需要交易上的 interest_rate 参数"
            )

    def __repr__(self):
        open_since = (
            self.open_date_utc.strftime(DATETIME_PRINT_FORMAT) if self.is_open else "已关闭"
        )

        return (
            f"交易(id={self.id}, 交易对={self.pair}, 金额={self.amount:.8f}, "
            f"是否做空={self.is_short or False}, 杠杆={self.leverage or 1.0}, "
            f"开仓价={self.open_rate:.8f}, 开仓时间={open_since})"
        )

    def to_json(self, minified: bool = False) -> dict[str, Any]:
        """
        :param minified: 如果为True，则仅返回数据的子集。
                         仅用于回测。
        :return: 包含交易数据的字典
        """
        filled_or_open_orders = self.select_filled_or_open_orders()
        orders_json = [order.to_json(self.entry_side, minified) for order in filled_or_open_orders]

        return {
            "trade_id": self.id,
            "pair": self.pair,
            "base_currency": self.safe_base_currency,
            "quote_currency": self.safe_quote_currency,
            "is_open": self.is_open,
            "exchange": self.exchange,
            "amount": round(self.amount, 8),
            "amount_requested": round(self.amount_requested, 8) if self.amount_requested else None,
            "stake_amount": round(self.stake_amount, 8),
            "max_stake_amount": round(self.max_stake_amount, 8) if self.max_stake_amount else None,
            "strategy": self.strategy,
            "enter_tag": self.enter_tag,
            "timeframe": self.timeframe,
            "fee_open": self.fee_open,
            "fee_open_cost": self.fee_open_cost,
            "fee_open_currency": self.fee_open_currency,
            "fee_close": self.fee_close,
            "fee_close_cost": self.fee_close_cost,
            "fee_close_currency": self.fee_close_currency,
            "open_date": self.open_date.strftime(DATETIME_PRINT_FORMAT),
            "open_timestamp": dt_ts_none(self.open_date_utc),
            "open_fill_date": (
                self.date_entry_fill_utc.strftime(DATETIME_PRINT_FORMAT)
                if self.date_entry_fill_utc
                else None
            ),
            "open_fill_timestamp": dt_ts_none(self.date_entry_fill_utc),
            "open_rate": self.open_rate,
            "open_rate_requested": self.open_rate_requested,
            "open_trade_value": round(self.open_trade_value, 8),
            "close_date": (
                self.close_date.strftime(DATETIME_PRINT_FORMAT) if self.close_date else None
            ),
            "close_timestamp": dt_ts_none(self.close_date_utc),
            "realized_profit": self.realized_profit or 0.0,
            # close-profit 对应于相对的已实现利润率
            "realized_profit_ratio": self.close_profit or None,
            "close_rate": self.close_rate,
            "close_rate_requested": self.close_rate_requested,
            "close_profit": self.close_profit,  # 已废弃
            "close_profit_pct": round(self.close_profit * 100, 2) if self.close_profit else None,
            "close_profit_abs": self.close_profit_abs,  # 已废弃
            "trade_duration_s": (
                int((self.close_date_utc - self.open_date_utc).total_seconds())
                if self.close_date
                else None
            ),
            "trade_duration": (
                int((self.close_date_utc - self.open_date_utc).total_seconds() // 60)
                if self.close_date
                else None
            ),
            "profit_ratio": self.close_profit,
            "profit_pct": round(self.close_profit * 100, 2) if self.close_profit else None,
            "profit_abs": self.close_profit_abs,
            "exit_reason": self.exit_reason,
            "exit_order_status": self.exit_order_status,
            "stop_loss_abs": self.stop_loss,
            "stop_loss_ratio": self.stop_loss_pct if self.stop_loss_pct else None,
            "stop_loss_pct": (self.stop_loss_pct * 100) if self.stop_loss_pct else None,
            "stoploss_last_update": (
                self.stoploss_last_update_utc.strftime(DATETIME_PRINT_FORMAT)
                if self.stoploss_last_update_utc
                else None
            ),
            "stoploss_last_update_timestamp": dt_ts_none(self.stoploss_last_update_utc),
            "initial_stop_loss_abs": self.initial_stop_loss,
            "initial_stop_loss_ratio": (
                self.initial_stop_loss_pct if self.initial_stop_loss_pct else None
            ),
            "initial_stop_loss_pct": (
                self.initial_stop_loss_pct * 100 if self.initial_stop_loss_pct else None
            ),
            "min_rate": self.min_rate,
            "max_rate": self.max_rate,
            "leverage": self.leverage,
            "interest_rate": self.interest_rate,
            "liquidation_price": self.liquidation_price,
            "is_short": self.is_short,
            "trading_mode": self.trading_mode,
            "funding_fees": self.funding_fees,
            "amount_precision": self.amount_precision,
            "price_precision": self.price_precision,
            "precision_mode": self.precision_mode,
            "precision_mode_price": self.precision_mode_price,
            "contract_size": self.contract_size,
            "has_open_orders": self.has_open_orders,
            "orders": orders_json,
        }

    @staticmethod
    def reset_trades() -> None:
        """
        重置所有交易。仅在回测模式下有效。
        """
        LocalTrade.bt_trades = []
        LocalTrade.bt_trades_open = []
        LocalTrade.bt_trades_open_pp = defaultdict(list)
        LocalTrade.bt_open_open_trade_count = 0
        LocalTrade.bt_total_profit = 0

    def adjust_min_max_rates(self, current_price: float, current_price_low: float) -> None:
        """
        调整max_rate和min_rate。
        """
        self.max_rate = max(current_price, self.max_rate or self.open_rate)
        self.min_rate = min(current_price_low, self.min_rate or self.open_rate)

    def set_liquidation_price(self, liquidation_price: float | None):
        """
        设置self.liquidation_price应使用的方法。
        确保止损价不会超过爆仓价
        """
        if liquidation_price is None:
            return
        self.liquidation_price = price_to_precision(
            liquidation_price, self.price_precision, self.precision_mode_price
        )

    def set_funding_fees(self, funding_fee: float) -> None:
        """
        将资金费用分配给交易。
        """
        if funding_fee is None:
            return
        self.funding_fee_running = funding_fee
        prior_funding_fees = sum([o.funding_fee for o in self.orders if o.funding_fee])
        self.funding_fees = prior_funding_fees + funding_fee

    def __set_stop_loss(self, stop_loss: float, percent: float):
        """
        内部用于设置self.stop_loss的方法。
        """
        if not self.stop_loss:
            self.initial_stop_loss = stop_loss
        self.stop_loss = stop_loss

        self.stop_loss_pct = -1 * abs(percent)

    def adjust_stop_loss(
        self,
        current_price: float,
        stoploss: float | None,
        initial: bool = False,
        allow_refresh: bool = False,
    ) -> None:
        """
        这会调整止损到最近观察到的设置
        :param current_price: 资产当前的交易价格
        :param stoploss: 止损因子（例如 -0.05 -> 比当前价格低-5%）。
        :param initial: 用于初始化止损。
            如果self.stop_loss已设置，则跳过所有操作。
        :param refresh: 用于刷新止损，允许双向调整
        """
        if stoploss is None or (initial and not (self.stop_loss is None or self.stop_loss == 0)):
            # 如果使用 initial 参数调用且无事可做，则不修改
            return

        leverage = self.leverage or 1.0
        if self.is_short:
            new_loss = float(current_price * (1 + abs(stoploss / leverage)))
        else:
            new_loss = float(current_price * (1 - abs(stoploss / leverage)))

        stop_loss_norm = price_to_precision(
            new_loss,
            self.price_precision,
            self.precision_mode_price,
            rounding_mode=ROUND_DOWN if self.is_short else ROUND_UP,
        )
        # 尚未分配止损
        if self.initial_stop_loss_pct is None:
            self.__set_stop_loss(stop_loss_norm, stoploss)
            self.initial_stop_loss = price_to_precision(
                stop_loss_norm,
                self.price_precision,
                self.precision_mode_price,
                rounding_mode=ROUND_DOWN if self.is_short else ROUND_UP,
            )
            self.initial_stop_loss_pct = -1 * abs(stoploss)

        # 评估是否需要更新止损
        else:
            higher_stop = stop_loss_norm > self.stop_loss
            lower_stop = stop_loss_norm < self.stop_loss

            # 止损只向上移动，永不向下！
            #   ? 但在杠杆交易中增加金额会产生更低的爆仓价，
            #   ? 从而降低最小止损
            if (
                allow_refresh
                or (higher_stop and not self.is_short)
                or (lower_stop and self.is_short)
            ):
                logger.debug(f"{self.pair} - 正在调整止损...")
                if not allow_refresh:
                    self.is_stop_loss_trailing = True
                self.__set_stop_loss(stop_loss_norm, stoploss)
            else:
                logger.debug(f"{self.pair} - 保持当前止损...")

        logger.debug(
            f"{self.pair} - 止损已调整。当前价格={current_price:.8f}, "
            f"开仓价={self.open_rate:.8f}, 最高价={self.max_rate or self.open_rate:.8f}, "
            f"初始止损={self.initial_stop_loss:.8f}, "
            f"止损={self.stop_loss:.8f}。 "
            f"追踪止损为我们节省了: "
            f"{float(self.stop_loss) - float(self.initial_stop_loss or 0.0):.8f}。"
        )

    def update_trade(self, order: Order, recalculating: bool = False) -> None:
        """
        使用实际的开仓/平仓价格和数量更新此实体。
        :param order: 通过exchange.fetch_order()获取的订单
        :return: None
        """

        # 忽略未结和已取消的订单
        if order.status == "open" or order.safe_price is None:
            return

        logger.info(f"正在更新交易 (id={self.id}) ...")
        if order.ft_order_side != "stoploss":
            order.funding_fee = self.funding_fee_running
            # Reset running funding fees
            self.funding_fee_running = 0.0
        order_type = order.order_type.upper() if order.order_type else None

        if order.ft_order_side == self.entry_side:
            # Update open rate and actual amount
            self.open_rate = order.safe_price
            self.amount = order.safe_amount_after_fee
            if self.is_open:
                payment = "SELL" if self.is_short else "BUY"
                logger.info(f"{order_type}_{payment} 已为 {self} 完成。")

            self.recalc_trade_from_orders()
        elif order.ft_order_side == self.exit_side:
            if self.is_open:
                payment = "BUY" if self.is_short else "SELL"
                # * 在保证金空头交易中，您购买的金额会略高于实际金额（金额+利息）
                logger.info(f"{order_type}_{payment} 已为 {self} 完成。")

        elif order.ft_order_side == "stoploss" and order.status not in ("open",):
            self.close_rate_requested = self.stop_loss
            self.exit_reason = ExitType.STOPLOSS_ON_EXCHANGE.value
            if self.is_open and order.safe_filled > 0:
                logger.info(f"{order_type} 已为 {self} 触发。")
        else:
            raise ValueError(f"未知订单类型: {order.order_type}")

        if order.ft_order_side != self.entry_side:
            amount_tr = amount_to_contract_precision(
                self.amount, self.amount_precision, self.precision_mode, self.contract_size
            )
            if isclose(order.safe_amount_after_fee, amount_tr, abs_tol=MATH_CLOSE_PREC) or (
                not recalculating and order.safe_amount_after_fee > amount_tr
            ):
                # 在重新计算交易时，只有当金额变为0时才能强制关闭交易
                self.close(order.safe_price)
            else:
                self.recalc_trade_from_orders()

        Trade.commit()

    def close(self, rate: float, *, show_msg: bool = True) -> None:
        """
        将close_rate设置为给定价格，计算总利润
        并将交易标记为已关闭
        """
        self.close_rate = rate
        self.close_date = self.close_date or self._date_last_filled_utc or dt_now()
        self.is_open = False
        self.exit_order_status = "closed"
        self.recalc_trade_from_orders(is_closing=True)
        if show_msg:
            logger.info(
                f"将 {self} 标记为已关闭，因为交易已完成且未找到任何未结订单。"
            )

    def update_fee(
        self, fee_cost: float, fee_currency: str | None, fee_rate: float | None, side: str
    ) -> None:
        """
        更新费用参数。每边仅操作一次
        :param fee_cost: 以质押货币计的费用成本
        :param fee_currency: 支付费用的货币
        :param fee_rate: 费用比率（例如 0.001 代表 0.1%）
        :param side: 费用的方向（买入/卖出）
        """
        if self.entry_side == side and self.fee_open_currency is None:
            self.fee_open_cost = fee_cost
            self.fee_open_currency = fee_currency
            if fee_rate is not None:
                self.fee_open = fee_rate
                # Assume close-fee will fall into the same fee category and take an educated guess
                self.fee_close = fee_rate
        elif self.exit_side == side and self.fee_close_currency is None:
            self.fee_close_cost = fee_cost
            self.fee_close_currency = fee_currency
            if fee_rate is not None:
                self.fee_close = fee_rate

    def fee_updated(self, side: str) -> bool:
        """
        验证此方向（买入/卖出）是否已更新
        """
        if self.entry_side == side:
            return self.fee_open_currency is not None
        elif self.exit_side == side:
            return self.fee_close_currency is not None
        else:
            return False

    def update_order(self, order: CcxtOrder) -> None:
        Order.update_orders(self.orders, order)

    @property
    def fully_canceled_entry_order_count(self) -> int:
        """
        获取失败的出场订单数量
        假定是完全出场。
        """
        return len(
            [
                o
                for o in self.orders
                if o.ft_order_side == self.entry_side
                and o.status in CANCELED_EXCHANGE_STATES
                and o.filled == 0
            ]
        )

    @property
    def canceled_exit_order_count(self) -> int:
        """
        获取失败的出场订单数量
        假定是完全出场。
        """
        return len(
            [
                o
                for o in self.orders
                if o.ft_order_side == self.exit_side and o.status in CANCELED_EXCHANGE_STATES
            ]
        )

    def get_canceled_exit_order_count(self) -> int:
        """
        获取失败的出场订单数量
        假定是完全出场。
        """
        return self.canceled_exit_order_count

    def _calc_open_trade_value(self, amount: float, open_rate: float) -> float:
        """
        计算包含开仓费用的开仓价格。
        :return: 包含费用的开仓交易价格
        """
        open_value = FtPrecise(amount) * FtPrecise(open_rate)
        fees = open_value * FtPrecise(self.fee_open)
        if self.is_short:
            return float(open_value - fees)
        else:
            return float(open_value + fees)

    def recalc_open_trade_value(self) -> None:
        """
        重新计算open_trade_value。
        每当open_rate、fee_open发生变化时必须调用此方法。
        """
        self.open_trade_value = self._calc_open_trade_value(self.amount, self.open_rate)

    def calculate_interest(self) -> FtPrecise:
        """
        计算此交易的利息。仅适用于保证金交易。
        """
        zero = FtPrecise(0.0)
        # 如果未借入任何资金
        if self.trading_mode != TradingMode.MARGIN or self.has_no_leverage:
            return zero

        open_date = self.open_date.replace(tzinfo=None)
        now = (self.close_date or datetime.now(timezone.utc)).replace(tzinfo=None)
        sec_per_hour = FtPrecise(3600)
        total_seconds = FtPrecise((now - open_date).total_seconds())
        hours = total_seconds / sec_per_hour or zero

        rate = FtPrecise(self.interest_rate)
        borrowed = FtPrecise(self.borrowed)

        return interest(exchange_name=self.exchange, borrowed=borrowed, rate=rate, hours=hours)

    def _calc_base_close(self, amount: FtPrecise, rate: float, fee: float | None) -> FtPrecise:
        close_value = amount * FtPrecise(rate)
        fees = close_value * FtPrecise(fee or 0.0)

        if self.is_short:
            return close_value + fees
        else:
            return close_value - fees

    def calc_close_trade_value(self, rate: float, amount: float | None = None) -> float:
        """
        计算交易的平仓价值（包含费用）
        :param rate: 用于比较的价格。
        :return: 以质押货币计的开仓交易价值
        """
        if rate is None and not self.close_rate:
            return 0.0

        amount1 = FtPrecise(amount or self.amount)
        trading_mode = self.trading_mode or TradingMode.SPOT

        if trading_mode == TradingMode.SPOT:
            return float(self._calc_base_close(amount1, rate, self.fee_close))

        elif trading_mode == TradingMode.MARGIN:
            total_interest = self.calculate_interest()

            if self.is_short:
                amount1 = amount1 + total_interest
                return float(self._calc_base_close(amount1, rate, self.fee_close))
            else:
                # 对于多头，货币已持有，无需购买
                return float(self._calc_base_close(amount1, rate, self.fee_close) - total_interest)

        elif trading_mode == TradingMode.FUTURES:
            funding_fees = self.funding_fees or 0.0
            # 正向资金费 -> 交易从费用中获益。
            # 负向资金费 -> 交易需要支付费用。
            if self.is_short:
                return float(self._calc_base_close(amount1, rate, self.fee_close)) - funding_fees
            else:
                return float(self._calc_base_close(amount1, rate, self.fee_close)) + funding_fees
        else:
            raise OperationalException(
                f"{self.trading_mode} 交易模式尚未在freqtrade中可用"
            )

    def calc_profit(
        self, rate: float, amount: float | None = None, open_rate: float | None = None
    ) -> float:
        """
        计算平仓交易与开仓交易之间以质押货币计的绝对利润
        已废弃 - 仅为向后兼容性保留
        :param rate: 用于比较的平仓价格。
        :param amount: 用于计算的金额。如果未设置，则回退到trade.amount。
        :param open_rate: 要使用的开仓价格。如果未提供，则默认为self.open_rate。
        :return: 以质押货币计的利润（浮点数）
        """
        prof = self.calculate_profit(rate, amount, open_rate)
        return prof.profit_abs

    def calculate_profit(
        self, rate: float, amount: float | None = None, open_rate: float | None = None
    ) -> ProfitStruct:
        """
        计算利润指标（绝对值、比率、总利润、总利润率）。
        所有计算均包含费用。
        :param rate: 用于比较的平仓价格。
        :param amount: 用于计算的金额。如果未设置，则回退到trade.amount。
        :param open_rate: 要使用的开仓价格。如果未提供，则默认为self.open_rate。
        :return: 利润结构，包含绝对利润和相对利润。
        """

        close_trade_value = self.calc_close_trade_value(rate, amount)
        if amount is None or open_rate is None:
            open_trade_value = self.open_trade_value
        else:
            open_trade_value = self._calc_open_trade_value(amount, open_rate)

        if self.is_short:
            profit_abs = open_trade_value - close_trade_value
        else:
            profit_abs = close_trade_value - open_trade_value

        try:
            if self.is_short:
                profit_ratio = (1 - (close_trade_value / open_trade_value)) * self.leverage
            else:
                profit_ratio = ((close_trade_value / open_trade_value) - 1) * self.leverage
            profit_ratio = float(f"{profit_ratio:.8f}")
        except ZeroDivisionError:
            profit_ratio = 0.0

        total_profit_abs = profit_abs + self.realized_profit
        if self.max_stake_amount:
            max_stake = self.max_stake_amount * (
                (1 - self.fee_open) if self.is_short else (1 + self.fee_open)
            )
            total_profit_ratio = total_profit_abs / max_stake
            total_profit_ratio = float(f"{total_profit_ratio:.8f}")
        else:
            total_profit_ratio = 0.0
        profit_abs = float(f"{profit_abs:.8f}")
        total_profit_abs = float(f"{total_profit_abs:.8f}")

        return ProfitStruct(
            profit_abs=profit_abs,
            profit_ratio=profit_ratio,
            total_profit=total_profit_abs,
            total_profit_ratio=total_profit_ratio,
        )

    def calc_profit_ratio(
        self, rate: float, amount: float | None = None, open_rate: float | None = None
    ) -> float:
        """
        计算利润比率（包含费用）。
        :param rate: 用于比较的价格。
        :param amount: 用于计算的金额。如果未设置，则回退到trade.amount。
        :param open_rate: 要使用的开仓价格。如果未提供，则默认为self.open_rate。
        :return: 利润比率（浮点数）
        """
        close_trade_value = self.calc_close_trade_value(rate, amount)

        if amount is None or open_rate is None:
            open_trade_value = self.open_trade_value
        else:
            open_trade_value = self._calc_open_trade_value(amount, open_rate)

        if open_trade_value == 0.0:
            return 0.0
        else:
            if self.is_short:
                profit_ratio = (1 - (close_trade_value / open_trade_value)) * self.leverage
            else:
                profit_ratio = ((close_trade_value / open_trade_value) - 1) * self.leverage

        return float(f"{profit_ratio:.8f}")

    def recalc_trade_from_orders(self, *, is_closing: bool = False):
        ZERO = FtPrecise(0.0)
        current_amount = FtPrecise(0.0)
        current_stake = FtPrecise(0.0)
        max_stake_amount = FtPrecise(0.0)
        total_stake = 0.0  # Total stake after all buy orders (does not subtract!)
        avg_price = FtPrecise(0.0)
        close_profit = 0.0
        close_profit_abs = 0.0
        # 重置资金费用
        self.funding_fees = 0.0
        funding_fees = 0.0
        ordercount = len(self.orders) - 1
        for i, o in enumerate(self.orders):
            if o.ft_is_open or not o.filled:
                continue
            funding_fees += o.funding_fee or 0.0
            tmp_amount = FtPrecise(o.safe_amount_after_fee)
            tmp_price = FtPrecise(o.safe_price)

            is_exit = o.ft_order_side != self.entry_side
            side = FtPrecise(-1 if is_exit else 1)
            if tmp_amount > ZERO and tmp_price is not None:
                current_amount += tmp_amount * side
                price = avg_price if is_exit else tmp_price
                current_stake += price * tmp_amount * side

                if current_amount > ZERO and not is_exit:
                    avg_price = current_stake / current_amount

            if is_exit:
                # 处理出场
                if i == ordercount and is_closing:
                    # 仅将资金费用应用于最后一次平仓订单
                    self.funding_fees = funding_fees

                exit_rate = o.safe_price
                exit_amount = o.safe_amount_after_fee
                prof = self.calculate_profit(exit_rate, exit_amount, float(avg_price))
                close_profit_abs += prof.profit_abs
                if total_stake > 0:
                    # 这需要根据最后一次出场来计算，以保持一致
                    # 与已实现利润一致。
                    close_profit = (close_profit_abs / total_stake) * self.leverage
            else:
                total_stake += self._calc_open_trade_value(tmp_amount, price)
                max_stake_amount += tmp_amount * price
        self.funding_fees = funding_fees
        self.max_stake_amount = float(max_stake_amount) / (self.leverage or 1.0)

        if close_profit:
            self.close_profit = close_profit
            self.realized_profit = close_profit_abs
            self.close_profit_abs = prof.profit_abs

        current_amount_tr = amount_to_contract_precision(
            float(current_amount), self.amount_precision, self.precision_mode, self.contract_size
        )
        if current_amount_tr > 0.0:
            # 交易仍未关闭
            # 杠杆未更新，因为目前我们不允许通过DCA改变杠杆。
            self.open_rate = price_to_precision(
                float(current_stake / current_amount),
                self.price_precision,
                self.precision_mode_price,
            )
            self.amount = current_amount_tr
            self.stake_amount = float(current_stake) / (self.leverage or 1.0)
            self.fee_open_cost = self.fee_open * float(self.max_stake_amount)
            self.recalc_open_trade_value()
            if self.stop_loss_pct is not None and self.open_rate is not None:
                self.adjust_stop_loss(self.open_rate, self.stop_loss_pct)
        elif is_closing and total_stake > 0:
            # 绝对平仓利润 / 最大持有量
            # 费用已被考虑在内，因为它们是close_profit_abs的一部分
            self.close_profit = (close_profit_abs / total_stake) * self.leverage
            self.close_profit_abs = close_profit_abs

    def select_order_by_order_id(self, order_id: str) -> Order | None:
        """
        根据订单ID查找订单对象。
        :param order_id: 交易所订单ID
        """
        for o in self.orders:
            if o.order_id == order_id:
                return o
        return None

    def select_order(
        self,
        order_side: str | None = None,
        is_open: bool | None = None,
        only_filled: bool = False,
    ) -> Order | None:
        """
        查找此订单方向和状态的最新订单
        :param order_side: 订单的ft_order_side（'buy'、'sell'或'stoploss'）
        :param is_open: 只搜索未结订单吗？
        :param only_filled: 只搜索已填充订单（仅在is_open=False时有效）。
        :return: 如果存在，则为最新订单对象，否则为None
        """
        orders = self.orders
        if order_side:
            orders = [o for o in orders if o.ft_order_side == order_side]
        if is_open is not None:
            orders = [o for o in orders if o.ft_is_open == is_open]
        if is_open is False and only_filled:
            orders = [o for o in orders if o.filled and o.status in NON_OPEN_EXCHANGE_STATES]
        if len(orders) > 0:
            return orders[-1]
        else:
            return None

    def select_filled_orders(self, order_side: str | None = None) -> list["Order"]:
        """
        查找此订单方向的已填充订单。
        不会返回已部分填充的未结订单。
        :param order_side: 订单方向（'buy'、'sell'或None）
        :return: Order对象数组
        """
        return [
            o
            for o in self.orders
            if ((o.ft_order_side == order_side) or (order_side is None))
            and o.ft_is_open is False
            and o.filled
            and o.status in NON_OPEN_EXCHANGE_STATES
        ]

    def select_filled_or_open_orders(self) -> list["Order"]:
        """
        查找已填充或未结订单
        :param order_side: 订单方向（'buy'、'sell'或None）
        :return: Order对象数组
        """
        return [
            o
            for o in self.orders
            if (
                o.ft_is_open is False
                and (o.filled or 0) > 0
                and o.status in NON_OPEN_EXCHANGE_STATES
            )
            or (o.ft_is_open is True and o.status is not None)
        ]

    def set_custom_data(self, key: str, value: Any) -> None:
        """
        为此交易设置自定义数据
        :param key: 自定义数据的键
        :param value: 自定义数据的值（必须是JSON可序列化的）
        """
        CustomDataWrapper.set_custom_data(trade_id=self.id, key=key, value=value)

    def get_custom_data(self, key: str, default: Any = None) -> Any:
        """
        获取此交易的自定义数据。

        :param key: 自定义数据的键
        :param default: 如果未找到数据则返回的值
        """
        data = CustomDataWrapper.get_custom_data(trade_id=self.id, key=key)
        if data:
            return data[0].value
        return default

    def get_custom_data_entry(self, key: str) -> _CustomData | None:
        """
        获取此交易的自定义数据
        :param key: 自定义数据的键
        """
        data = CustomDataWrapper.get_custom_data(trade_id=self.id, key=key)
        if data:
            return data[0]
        return None

    def get_all_custom_data(self) -> list[_CustomData]:
        """
        获取此交易的所有自定义数据
        """
        return CustomDataWrapper.get_custom_data(trade_id=self.id)

    @property
    def nr_of_successful_entries(self) -> int:
        """
        辅助函数，用于计算已填充的入场订单数量。
        :return: 此交易已填充入场订单的整数计数。
        """

        return len(self.select_filled_orders(self.entry_side))

    @property
    def nr_of_successful_exits(self) -> int:
        """
        辅助函数，用于计算已填充的出场订单数量。
        :return: 此交易已填充出场订单的整数计数。
        """
        return len(self.select_filled_orders(self.exit_side))

    @property
    def nr_of_successful_buys(self) -> int:
        """
        辅助函数，用于计算已填充的买入订单数量。
        警告：请使用nr_of_successful_entries以支持做空。
        :return: 此交易已填充买入订单的整数计数。
        """

        return len(self.select_filled_orders("buy"))

    @property
    def nr_of_successful_sells(self) -> int:
        """
        辅助函数，用于计算已填充的卖出订单数量。
        警告：请使用nr_of_successful_exits以支持做空。
        :return: 此交易已填充卖出订单的整数计数。
        """
        return len(self.select_filled_orders("sell"))

    @property
    def sell_reason(self) -> str | None:
        """已废弃！请使用exit_reason代替。"""
        return self.exit_reason

    @property
    def safe_close_rate(self) -> float:
        return self.close_rate or self.close_rate_requested or 0.0

    @staticmethod
    def get_trades_proxy(
        *,
        pair: str | None = None,
        is_open: bool | None = None,
        open_date: datetime | None = None,
        close_date: datetime | None = None,
    ) -> list["LocalTrade"]:
        """
        查询交易的辅助函数。
        返回一个交易列表，根据给定参数进行过滤。
        在实盘模式下，将过滤器转换为数据库查询并返回所有行。
        在回测模式下，使用Trade.bt_trades上的过滤器获取结果。

        :param pair: 按交易对过滤
        :param is_open: 按未结/已关闭状态过滤
        :param open_date: 按open_date过滤（通过trade.open_date > 输入进行过滤）
        :param close_date: 按close_date过滤（通过trade.close_date > 输入进行过滤）
                           这将隐式地只返回已关闭的交易。
        :return: 未排序的List[Trade]
        """

        # 离线模式 - 无数据库
        if is_open is not None:
            if is_open:
                sel_trades = LocalTrade.bt_trades_open
            else:
                sel_trades = LocalTrade.bt_trades

        else:
            # Not used during backtesting, but might be used by a strategy
            sel_trades = list(LocalTrade.bt_trades + LocalTrade.bt_trades_open)

        if pair:
            sel_trades = [trade for trade in sel_trades if trade.pair == pair]
        if open_date:
            sel_trades = [trade for trade in sel_trades if trade.open_date > open_date]
        if close_date:
            sel_trades = [
                trade for trade in sel_trades if trade.close_date and trade.close_date > close_date
            ]

        return sel_trades

    @staticmethod
    def close_bt_trade(trade):
        LocalTrade.bt_trades_open.remove(trade)
        LocalTrade.bt_trades_open_pp[trade.pair].remove(trade)
        LocalTrade.bt_open_open_trade_count -= 1
        LocalTrade.bt_trades.append(trade)
        LocalTrade.bt_total_profit += trade.close_profit_abs

    @staticmethod
    def add_bt_trade(trade):
        if trade.is_open:
            LocalTrade.bt_trades_open.append(trade)
            LocalTrade.bt_trades_open_pp[trade.pair].append(trade)
            LocalTrade.bt_open_open_trade_count += 1
        else:
            LocalTrade.bt_trades.append(trade)

    @staticmethod
    def remove_bt_trade(trade):
        LocalTrade.bt_trades_open.remove(trade)
        LocalTrade.bt_trades_open_pp[trade.pair].remove(trade)
        LocalTrade.bt_open_open_trade_count -= 1

    @staticmethod
    def get_open_trades() -> list[Any]:
        """
        检索未结交易
        """
        return Trade.get_trades_proxy(is_open=True)

    @staticmethod
    def get_open_trade_count() -> int:
        """
        获取未结交易数量
        """
        if Trade.use_db:
            return Trade.session.execute(
                select(func.count(Trade.id)).filter(Trade.is_open.is_(True))
            ).scalar_one()
        else:
            return LocalTrade.bt_open_open_trade_count

    @staticmethod
    def stoploss_reinitialization(desired_stoploss: float):
        """
        将所有未结交易的初始止损调整到所需的止损。
        """
        trade: Trade
        for trade in Trade.get_open_trades():
            logger.info(f"找到未结交易: {trade}")

            # 如果追踪止损已改变止损，则跳过此情况。
            if not trade.is_stop_loss_trailing and trade.initial_stop_loss_pct != desired_stoploss:
                # 止损值已更改

                logger.info(f"{trade} 的止损需要调整...")
                # 强制重置止损
                trade.stop_loss = 0.0
                trade.initial_stop_loss_pct = None
                trade.adjust_stop_loss(trade.open_rate, desired_stoploss)
                logger.info(f"新止损: {trade.stop_loss}。")

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """
        从JSON字符串创建Trade实例。

        用于调试目的 - 请保留。
        :param json_str: 要解析的JSON字符串
        :return: Trade实例
        """
        from uuid import uuid4

        import rapidjson

        data = rapidjson.loads(json_str)
        trade = cls(
            __FROM_JSON=True,
            id=data.get("trade_id"),
            pair=data["pair"],
            base_currency=data.get("base_currency"),
            stake_currency=data.get("quote_currency"),
            is_open=data["is_open"],
            exchange=data.get("exchange", "import"),
            amount=data["amount"],
            amount_requested=data.get("amount_requested", data["amount"]),
            stake_amount=data["stake_amount"],
            strategy=data.get("strategy"),
            enter_tag=data["enter_tag"],
            timeframe=data.get("timeframe"),
            fee_open=data["fee_open"],
            fee_open_cost=data.get("fee_open_cost"),
            fee_open_currency=data.get("fee_open_currency"),
            fee_close=data["fee_close"],
            fee_close_cost=data.get("fee_close_cost"),
            fee_close_currency=data.get("fee_close_currency"),
            open_date=datetime.fromtimestamp(data["open_timestamp"] // 1000, tz=timezone.utc),
            open_rate=data["open_rate"],
            open_rate_requested=data.get("open_rate_requested", data["open_rate"]),
            open_trade_value=data.get("open_trade_value"),
            close_date=(
                datetime.fromtimestamp(data["close_timestamp"] // 1000, tz=timezone.utc)
                if data["close_timestamp"]
                else None
            ),
            realized_profit=data.get("realized_profit", 0),
            close_rate=data["close_rate"],
            close_rate_requested=data.get("close_rate_requested", data["close_rate"]),
            close_profit=data.get("close_profit", data.get("profit_ratio")),
            close_profit_abs=data.get("close_profit_abs", data.get("profit_abs")),
            exit_reason=data["exit_reason"],
            exit_order_status=data.get("exit_order_status"),
            stop_loss=data["stop_loss_abs"],
            stop_loss_pct=data["stop_loss_ratio"],
            initial_stop_loss=data["initial_stop_loss_abs"],
            initial_stop_loss_pct=data["initial_stop_loss_ratio"],
            min_rate=data["min_rate"],
            max_rate=data["max_rate"],
            leverage=data["leverage"],
            interest_rate=data.get("interest_rate"),
            liquidation_price=data.get("liquidation_price"),
            is_short=data["is_short"],
            trading_mode=data.get("trading_mode"),
            funding_fees=data.get("funding_fees"),
            amount_precision=data.get("amount_precision", None),
            price_precision=data.get("price_precision", None),
            precision_mode=data.get("precision_mode", None),
            precision_mode_price=data.get("precision_mode_price", data.get("precision_mode", None)),
            contract_size=data.get("contract_size", None),
        )
        for order in data["orders"]:
            order_obj = Order(
                amount=order["amount"],
                ft_amount=order["amount"],
                ft_order_side=order["ft_order_side"],
                ft_pair=order.get("pair", data["pair"]),
                ft_is_open=order.get("is_open", False),
                order_id=order.get("order_id", uuid4().hex),
                status=order.get("status"),
                average=order.get("average", order.get("safe_price")),
                cost=order["cost"],
                filled=order.get("filled", order["amount"]),
                order_date=datetime.strptime(order["order_date"], DATETIME_PRINT_FORMAT)
                if order.get("order_date")
                else None,
                order_filled_date=(
                    datetime.fromtimestamp(order["order_filled_timestamp"] // 1000, tz=timezone.utc)
                    if order["order_filled_timestamp"]
                    else None
                ),
                order_type=order.get("order_type"),
                price=order.get("price", order.get("safe_price")),
                ft_price=order.get("price", order.get("safe_price")),
                remaining=order.get("remaining", 0.0),
                funding_fee=order.get("funding_fee", None),
                ft_order_tag=order.get("ft_order_tag", None),
                ft_fee_base=order.get("ft_fee_base", None),
            )
            trade.orders.append(order_obj)

        return trade


class Trade(ModelBase, LocalTrade):
    """
    交易数据库模型。
    还处理交易的更新和查询

    注意：字段必须与LocalTrade类对齐
    """

    __tablename__ = "trades"
    session: ClassVar[SessionType]

    use_db: bool = True

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    orders: Mapped[list[Order]] = relationship(
        "Order",
        order_by="Order.id",
        cascade="all, delete-orphan",
        lazy="selectin",
        innerjoin=True,
        back_populates="_trade_live",
    )
    custom_data: Mapped[list[_CustomData]] = relationship(
        "_CustomData", cascade="all, delete-orphan", lazy="raise"
    )

    exchange: Mapped[str] = mapped_column(String(25), nullable=False)
    pair: Mapped[str] = mapped_column(String(25), nullable=False, index=True)
    base_currency: Mapped[str | None] = mapped_column(String(25), nullable=True)
    stake_currency: Mapped[str | None] = mapped_column(String(25), nullable=True)
    is_open: Mapped[bool] = mapped_column(nullable=False, default=True, index=True)
    fee_open: Mapped[float] = mapped_column(Float(), nullable=False, default=0.0)
    # 入场交易以计价货币计的费用成本
    fee_open_cost: Mapped[float | None] = mapped_column(Float(), nullable=True)
    # 支付费用的货币。与fee_open_cost无关。
    fee_open_currency: Mapped[str | None] = mapped_column(String(25), nullable=True)
    fee_close: Mapped[float | None] = mapped_column(Float(), nullable=False, default=0.0)
    # 出场订单以计价货币计的费用成本
    fee_close_cost: Mapped[float | None] = mapped_column(Float(), nullable=True)
    fee_close_currency: Mapped[str | None] = mapped_column(String(25), nullable=True)
    open_rate: Mapped[float] = mapped_column(Float())
    open_rate_requested: Mapped[float | None] = mapped_column(Float(), nullable=True)
    # open_trade_value - calculated via _calc_open_trade_value
    open_trade_value: Mapped[float] = mapped_column(Float(), nullable=True)
    close_rate: Mapped[float | None] = mapped_column(Float())
    close_rate_requested: Mapped[float | None] = mapped_column(Float())
    realized_profit: Mapped[float] = mapped_column(Float(), default=0.0, nullable=True)
    close_profit: Mapped[float | None] = mapped_column(Float())
    close_profit_abs: Mapped[float | None] = mapped_column(Float())
    stake_amount: Mapped[float] = mapped_column(Float())
    max_stake_amount: Mapped[float | None] = mapped_column(Float())
    amount: Mapped[float] = mapped_column(Float())
    amount_requested: Mapped[float | None] = mapped_column(Float())
    open_date: Mapped[datetime] = mapped_column(nullable=False, default=datetime.now)
    close_date: Mapped[datetime | None] = mapped_column()
    # 止损的绝对值
    stop_loss: Mapped[float] = mapped_column(Float(), nullable=True, default=0.0)
    # 止损的百分比值
    stop_loss_pct: Mapped[float | None] = mapped_column(Float(), nullable=True)
    # 初始止损的绝对值
    initial_stop_loss: Mapped[float | None] = mapped_column(Float(), nullable=True, default=0.0)
    # 初始止损的百分比值
    initial_stop_loss_pct: Mapped[float | None] = mapped_column(Float(), nullable=True)
    is_stop_loss_trailing: Mapped[bool] = mapped_column(nullable=False, default=False)
    # 达到的最高价格的绝对值
    max_rate: Mapped[float | None] = mapped_column(Float(), nullable=True, default=0.0)
    # 达到的最低价格
    min_rate: Mapped[float | None] = mapped_column(Float(), nullable=True)
    exit_reason: Mapped[str | None] = mapped_column(String(CUSTOM_TAG_MAX_LENGTH), nullable=True)
    exit_order_status: Mapped[str | None] = mapped_column(String(100), nullable=True)
    strategy: Mapped[str | None] = mapped_column(String(100), nullable=True)
    enter_tag: Mapped[str | None] = mapped_column(String(CUSTOM_TAG_MAX_LENGTH), nullable=True)
    timeframe: Mapped[int | None] = mapped_column(Integer, nullable=True)

    trading_mode: Mapped[TradingMode] = mapped_column(Enum(TradingMode), nullable=True)
    amount_precision: Mapped[float | None] = mapped_column(Float(), nullable=True)
    price_precision: Mapped[float | None] = mapped_column(Float(), nullable=True)
    precision_mode: Mapped[int | None] = mapped_column(Integer, nullable=True)
    precision_mode_price: Mapped[int | None] = mapped_column(Integer, nullable=True)
    contract_size: Mapped[float | None] = mapped_column(Float(), nullable=True)

    # Leverage trading properties
    leverage: Mapped[float] = mapped_column(Float(), nullable=True, default=1.0)
    is_short: Mapped[bool] = mapped_column(nullable=False, default=False)
    liquidation_price: Mapped[float | None] = mapped_column(Float(), nullable=True)

    # Margin Trading Properties
    interest_rate: Mapped[float] = mapped_column(Float(), nullable=False, default=0.0)

    # Futures properties
    funding_fees: Mapped[float | None] = mapped_column(Float(), nullable=True, default=None)
    funding_fee_running: Mapped[float | None] = mapped_column(Float(), nullable=True, default=None)

    record_version: Mapped[int] = mapped_column(Integer, nullable=False, default=2)

    def __init__(self, **kwargs):
        from_json = kwargs.pop("__FROM_JSON", None)
        super().__init__(**kwargs)
        if not from_json:
            # 从json加载时跳过重新计算
            self.realized_profit = 0
            self.recalc_open_trade_value()

    @validates("enter_tag", "exit_reason")
    def validate_string_len(self, key, value):
        max_len = getattr(self.__class__, key).prop.columns[0].type.length
        if value and len(value) > max_len:
            return value[:max_len]
        return value

    def delete(self) -> None:
        for order in self.orders:
            Order.session.delete(order)

        CustomDataWrapper.delete_custom_data(trade_id=self.id)

        Trade.session.delete(self)
        Trade.commit()

    @staticmethod
    def commit():
        Trade.session.commit()

    @staticmethod
    def rollback():
        Trade.session.rollback()

    @staticmethod
    def get_trades_proxy(
        *,
        pair: str | None = None,
        is_open: bool | None = None,
        open_date: datetime | None = None,
        close_date: datetime | None = None,
    ) -> list["LocalTrade"]:
        """
        查询交易的辅助函数。
        返回一个交易列表，根据给定参数进行过滤。
        在实盘模式下，将过滤器转换为数据库查询并返回所有行。
        在回测模式下，使用Trade.bt_trades上的过滤器获取结果。
        :param pair: 按交易对过滤
        :param is_open: 按未结/已关闭状态过滤
        :param open_date: 按open_date过滤（通过trade.open_date > 输入进行过滤）
        :param close_date: 按close_date过滤（通过trade.close_date > 输入进行过滤）
                           并将隐式地只返回已关闭的交易。
        :return: 未排序的List[Trade]
        """
        if Trade.use_db:
            trade_filter = []
            if pair:
                trade_filter.append(Trade.pair == pair)
            if open_date:
                trade_filter.append(Trade.open_date > open_date)
            if close_date:
                trade_filter.append(Trade.close_date > close_date)
            if is_open is not None:
                trade_filter.append(Trade.is_open.is_(is_open))
            return cast(list[LocalTrade], Trade.get_trades(trade_filter).all())
        else:
            return LocalTrade.get_trades_proxy(
                pair=pair, is_open=is_open, open_date=open_date, close_date=close_date
            )

    @staticmethod
    def get_trades_query(trade_filter=None, include_orders: bool = True) -> Select:
        """
        使用过滤器查询交易的辅助函数。
        注意：回测中不支持。
        :param trade_filter: 可选的过滤器，应用于交易
                             可以是Filter对象，也可以是Filter列表
                             例如 `(trade_filter=[Trade.id == trade_id, Trade.is_open.is_(True),])`
                             例如 `(trade_filter=Trade.id == trade_id)`
        :return: 未排序的查询对象
        """
        if not Trade.use_db:
            raise NotImplementedError("`Trade.get_trades()` 在回测模式下不支持。")
        if trade_filter is not None:
            if not isinstance(trade_filter, list):
                trade_filter = [trade_filter]
            this_query = select(Trade).filter(*trade_filter)
        else:
            this_query = select(Trade)
        if not include_orders:
            # 不加载订单关系
            # 考虑使用noload或raiseload代替lazyload
            this_query = this_query.options(lazyload(Trade.orders))
        return this_query

    @staticmethod
    def get_trades(trade_filter=None, include_orders: bool = True) -> ScalarResult["Trade"]:
        """
        使用过滤器查询交易的辅助函数。
        注意：回测中不支持。
        :param trade_filter: 可选的过滤器，应用于交易
                             可以是Filter对象，也可以是Filter列表
                             例如 `(trade_filter=[Trade.id == trade_id, Trade.is_open.is_(True),])`
                             例如 `(trade_filter=Trade.id == trade_id)`
        :return: 未排序的查询对象
        """
        query = Trade.get_trades_query(trade_filter, include_orders)
        # 这应该保持分离。如果use_db为False，则session不可用，上面的代码将
        # 抛出异常。
        return Trade.session.scalars(query)

    @staticmethod
    def get_open_trades_without_assigned_fees():
        """
        返回所有未正确设置开仓费用的未结交易
        注意：回测中不支持。
        """
        return Trade.get_trades(
            [
                Trade.fee_open_currency.is_(None),
                Trade.orders.any(),
                Trade.is_open.is_(True),
            ]
        ).all()

    @staticmethod
    def get_closed_trades_without_assigned_fees():
        """
        返回所有未正确设置费用的已关闭交易
        注意：回测中不支持。
        """
        return Trade.get_trades(
            [
                Trade.fee_close_currency.is_(None),
                Trade.orders.any(),
                Trade.is_open.is_(False),
            ]
        ).all()

    @staticmethod
    def get_total_closed_profit() -> float:
        """
        检索已实现的总利润
        """
        if Trade.use_db:
            total_profit = Trade.session.execute(
                select(func.sum(Trade.close_profit_abs)).filter(Trade.is_open.is_(False))
            ).scalar_one()
        else:
            total_profit = sum(
                t.close_profit_abs  # type: ignore
                for t in LocalTrade.get_trades_proxy(is_open=False)
            )
        return total_profit or 0

    @staticmethod
    def total_open_trades_stakes() -> float:
        """
        计算未结交易中以质押货币计的总投资金额
        """
        if Trade.use_db:
            total_open_stake_amount = Trade.session.scalar(
                select(func.sum(Trade.stake_amount)).filter(Trade.is_open.is_(True))
            )
        else:
            total_open_stake_amount = sum(
                t.stake_amount for t in LocalTrade.get_trades_proxy(is_open=True)
            )
        return total_open_stake_amount or 0

    @staticmethod
    def _generic_performance_query(columns: list, filters: list, fallback: str = "") -> Select:
        """
        检索一个通用选择对象，以根据`columns`分组计算绩效。
        返回以下列：
        - columns
        - profit_ratio
        - profit_sum_abs
        - count
        注意：回测中不支持。
        """
        columns_coal = [func.coalesce(c, fallback).label(c.name) for c in columns]
        pair_costs = (
            select(
                *columns_coal,
                func.sum(
                    (
                        func.coalesce(Order.filled, Order.amount)
                        * func.coalesce(Order.average, Order.price, Order.ft_price)
                    )
                    / func.coalesce(Trade.leverage, 1)
                ).label("cost_per_pair"),
            )
            .join(Order, Trade.id == Order.ft_trade_id)
            .filter(
                *filters,
                Order.ft_order_side == case((Trade.is_short.is_(True), "sell"), else_="buy"),
                Order.filled > 0,
            )
            .group_by(*columns)
            .cte("pair_costs")
        )
        trades_grouped = (
            select(
                *columns_coal,
                func.sum(Trade.close_profit_abs).label("profit_sum_abs"),
                func.count(*columns_coal).label("count"),
            )
            .filter(*filters)
            .group_by(*columns_coal)
            .cte("trades_grouped")
        )
        q = (
            select(
                *[trades_grouped.c[x.name] for x in columns],
                (trades_grouped.c.profit_sum_abs / pair_costs.c.cost_per_pair).label(
                    "profit_ratio"
                ),
                trades_grouped.c.profit_sum_abs,
                trades_grouped.c.count,
            )
            .join(pair_costs, *[trades_grouped.c[x.name] == pair_costs.c[x.name] for x in columns])
            .order_by(desc("profit_sum_abs"))
        )
        return q

    @staticmethod
    def get_overall_performance(start_date: datetime | None = None) -> list[dict[str, Any]]:
        """
        返回包含所有交易的字典列表，包括利润和交易数量
        注意：回测中不支持。
        """
        filters: list = [Trade.is_open.is_(False)]
        if start_date:
            filters.append(Trade.close_date >= start_date)
        pair_rates_query = Trade._generic_performance_query([Trade.pair], filters)
        pair_rates = Trade.session.execute(pair_rates_query).all()
        return [
            {
                "pair": pair,
                "profit_ratio": profit,
                "profit": round(profit * 100, 2),  # 兼容模式
                "profit_pct": round(profit * 100, 2),
                "profit_abs": profit_abs,
                "count": count,
            }
            for pair, profit, profit_abs, count in pair_rates
        ]

    @staticmethod
    def get_enter_tag_performance(pair: str | None) -> list[dict[str, Any]]:
        """
        返回包含所有交易的字典列表，基于买入标签表现
        可以是所有交易对的平均值，也可以是提供的特定交易对的平均值
        注意：回测中不支持。
        """

        filters: list = [Trade.is_open.is_(False)]
        if pair is not None:
            filters.append(Trade.pair == pair)

        pair_rates_query = Trade._generic_performance_query([Trade.enter_tag], filters, "Other")
        enter_tag_perf = Trade.session.execute(pair_rates_query).all()

        return [
            {
                "enter_tag": enter_tag if enter_tag is not None else "Other",
                "profit_ratio": profit,
                "profit_pct": round(profit * 100, 2),
                "profit_abs": profit_abs,
                "count": count,
            }
            for enter_tag, profit, profit_abs, count in enter_tag_perf
        ]

    @staticmethod
    def get_exit_reason_performance(pair: str | None) -> list[dict[str, Any]]:
        """
        返回包含所有交易的字典列表，基于出场原因表现
        可以是所有交易对的平均值，也可以是提供的特定交易对的平均值
        注意：回测中不支持。
        """

        filters: list = [Trade.is_open.is_(False)]
        if pair is not None:
            filters.append(Trade.pair == pair)

        pair_rates_query = Trade._generic_performance_query([Trade.exit_reason], filters, "Other")
        sell_tag_perf = Trade.session.execute(pair_rates_query).all()

        return [
            {
                "exit_reason": exit_reason if exit_reason is not None else "Other",
                "profit_ratio": profit,
                "profit_pct": round(profit * 100, 2),
                "profit_abs": profit_abs,
                "count": count,
            }
            for exit_reason, profit, profit_abs, count in sell_tag_perf
        ]

    @staticmethod
    def get_mix_tag_performance(pair: str | None) -> list[dict[str, Any]]:
        """
        返回包含所有交易的字典列表，基于entry_tag + exit_reason表现
        可以是所有交易对的平均值，也可以是提供的特定交易对的平均值
        注意：回测中不支持。
        """

        filters: list = [Trade.is_open.is_(False)]
        if pair is not None:
            filters.append(Trade.pair == pair)
        mix_tag_perf = Trade.session.execute(
            select(
                Trade.id,
                Trade.enter_tag,
                Trade.exit_reason,
                func.sum(Trade.close_profit).label("profit_sum"),
                func.sum(Trade.close_profit_abs).label("profit_sum_abs"),
                func.count(Trade.pair).label("count"),
            )
            .filter(*filters)
            .group_by(Trade.id)
            .order_by(desc("profit_sum_abs"))
        ).all()

        resp: list[dict] = []
        for _, enter_tag, exit_reason, profit, profit_abs, count in mix_tag_perf:
            enter_tag = enter_tag if enter_tag is not None else "Other"
            exit_reason = exit_reason if exit_reason is not None else "Other"

            if exit_reason is not None and enter_tag is not None:
                mix_tag = enter_tag + " " + exit_reason
                i = 0
                if not any(item["mix_tag"] == mix_tag for item in resp):
                    resp.append(
                        {
                            "mix_tag": mix_tag,
                            "profit_ratio": profit,
                            "profit_pct": round(profit * 100, 2),
                            "profit_abs": profit_abs,
                            "count": count,
                        }
                    )
                else:
                    while i < len(resp):
                        if resp[i]["mix_tag"] == mix_tag:
                            resp[i] = {
                                "mix_tag": mix_tag,
                                "profit_ratio": profit + resp[i]["profit_ratio"],
                                "profit_pct": round(profit + resp[i]["profit_ratio"] * 100, 2),
                                "profit_abs": profit_abs + resp[i]["profit_abs"],
                                "count": 1 + resp[i]["count"],
                            }
                        i += 1

        return resp

    @staticmethod
    def get_best_pair(start_date: datetime | None = None):
        """
        获取已关闭交易的最佳交易对。
        注意：回测中不支持。
        :returns: 包含(交易对, 利润总和)的元组
        """
        filters: list = [Trade.is_open.is_(False)]
        if start_date:
            filters.append(Trade.close_date >= start_date)

        pair_rates_query = Trade._generic_performance_query([Trade.pair], filters)
        best_pair = Trade.session.execute(pair_rates_query).first()
        # 返回交易对、利润率、绝对利润、计数
        return best_pair

    @staticmethod
    def get_trading_volume(start_date: datetime | None = None) -> float:
        """
        根据订单获取交易量
        注意：回测中不支持。
        :returns: 包含(交易对, 利润总和)的元组
        """
        filters = [Order.status == "closed"]
        if start_date:
            filters.append(Order.order_filled_date >= start_date)
        trading_volume = Trade.session.execute(
            select(func.sum(Order.cost).label("volume")).filter(*filters)
        ).scalar_one()
        return trading_volume or 0.0

