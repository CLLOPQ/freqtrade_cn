"""Okx 交易所子类。"""

import logging
from datetime import timedelta

import ccxt

from freqtrade.constants import BuySell
from freqtrade.enums import CandleType, MarginMode, PriceType, TradingMode
from freqtrade.exceptions import (
    DDosProtection,
    OperationalException,
    RetryableOrderError,
    TemporaryError,
)
from freqtrade.exchange import Exchange, date_minus_candles
from freqtrade.exchange.common import API_RETRY_COUNT, retrier
from freqtrade.exchange.exchange_types import CcxtOrder, FtHas
from freqtrade.misc import safe_value_fallback2
from freqtrade.util import dt_now, dt_ts


logger = logging.getLogger(__name__)


class Okx(Exchange):
    """Okx 交易所类。

    包含使 Freqtrade 能够与该交易所正常工作所需的调整。
    """

    _ft_has: FtHas = {
        "ohlcv_candle_limit": 100,  # 警告，X个月前的数据有特殊情况
        "mark_ohlcv_timeframe": "4h",  # 标记价格K线时间框架
        "funding_fee_timeframe": "8h",  # 资金费用时间框架
        "stoploss_order_types": {"limit": "limit"},  # 止损订单类型映射
        "stoploss_on_exchange": True,  # 支持交易所止损
        "trades_has_history": False,  # 交易历史不支持分页（端点没有"since"参数）
        "ws_enabled": True,  # 支持WebSocket
    }
    _ft_has_futures: FtHas = {
        "tickers_have_quoteVolume": False,  # Ticker不包含报价量
        "stop_price_type_field": "slTriggerPxType",  # 止损价格类型字段
        "stop_price_type_value_mapping": {
            PriceType.LAST: "last",  # 最新价格对应值
            PriceType.MARK: "index",  # 标记价格对应值
            PriceType.INDEX: "mark",  # 指数价格对应值
        },
        "stoploss_blocks_assets": False,  # 止损订单不锁定资产
        "ws_enabled": True,  # 期货支持WebSocket
    }

    _supported_trading_mode_margin_pairs: list[tuple[TradingMode, MarginMode]] = [
        # TradingMode.SPOT始终受支持，无需在此列表中
        # (TradingMode.MARGIN, MarginMode.CROSS),
        # (TradingMode.FUTURES, MarginMode.CROSS),
        (TradingMode.FUTURES, MarginMode.ISOLATED),  # 期货逐仓模式
    ]

    net_only = True  # 净额模式

    _ccxt_params: dict = {"options": {"brokerId": "ffb5405ad327SUDE"}}  # ccxt参数

    def ohlcv_candle_limit(
        self, timeframe: str, candle_type: CandleType, since_ms: int | None = None
    ) -> int:
        """
        交易所OHLCV蜡烛图限制
        OKX有以下行为：
        * 最新数据最多300根蜡烛
        * 历史数据最多100根蜡烛
        * 其他类型蜡烛（非期货或现货）最多100根
        :param timeframe: 要检查的时间框架
        :param candle_type: 蜡烛图类型
        :param since_ms: 起始时间戳
        :return: 蜡烛图限制数量（整数）
        """
        if candle_type in (CandleType.FUTURES, CandleType.SPOT) and (
            not since_ms or since_ms > (date_minus_candles(timeframe, 300).timestamp() * 1000)
        ):
            return 300

        return super().ohlcv_candle_limit(timeframe, candle_type, since_ms)

    @retrier
    def additional_exchange_init(self) -> None:
        """
        额外的交易所初始化逻辑。
        此时.api已可用。
        如果需要，必须在子方法中重写。
        """
        try:
            if self.trading_mode == TradingMode.FUTURES and not self._config["dry_run"]:
                accounts = self._api.fetch_accounts()
                self._log_exchange_response("fetch_accounts", accounts)
                if len(accounts) > 0:
                    self.net_only = accounts[0].get("info", {}).get("posMode") == "net_mode"
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于 {e.__class__.__name__}，在 additional_exchange_init 中出错。消息: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def _get_posSide(self, side: BuySell, reduceOnly: bool):
        """获取持仓方向"""
        if self.net_only:
            return "net"
        if not reduceOnly:
            # 开仓
            return "long" if side == "buy" else "short"
        else:
            # 平仓
            return "long" if side == "sell" else "short"

    def _get_params(
        self,
        side: BuySell,
        ordertype: str,
        leverage: float,
        reduceOnly: bool,
        time_in_force: str = "GTC",
    ) -> dict:
        params = super()._get_params(
            side=side,
            ordertype=ordertype,
            leverage=leverage,
            reduceOnly=reduceOnly,
            time_in_force=time_in_force,
        )
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode:
            params["tdMode"] = self.margin_mode.value  # 交易模式
            params["posSide"] = self._get_posSide(side, reduceOnly)  # 持仓方向
        return params

    def __fetch_leverage_already_set(self, pair: str, leverage: float, side: BuySell) -> bool:
        """检查杠杆是否已设置"""
        try:
            res_lev = self._api.fetch_leverage(
                symbol=pair,
                params={
                    "mgnMode": self.margin_mode.value,
                    "posSide": self._get_posSide(side, False),
                },
            )
            self._log_exchange_response("get_leverage", res_lev)
            already_set = all(float(x["lever"]) == leverage for x in res_lev["data"])
            return already_set

        except ccxt.BaseError:
            # 假设所有错误都表示"尚未设置"
            return False

    @retrier
    def _lev_prep(self, pair: str, leverage: float, side: BuySell, accept_fail: bool = False):
        """准备杠杆设置"""
        if self.trading_mode != TradingMode.SPOT and self.margin_mode is not None:
            try:
                res = self._api.set_leverage(
                    leverage=leverage,
                    symbol=pair,
                    params={
                        "mgnMode": self.margin_mode.value,
                        "posSide": self._get_posSide(side, False),
                    },
                )
                self._log_exchange_response("set_leverage", res)

            except ccxt.DDoSProtection as e:
                raise DDosProtection(e) from e
            except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
                already_set = self.__fetch_leverage_already_set(pair, leverage, side)
                if not already_set:
                    raise TemporaryError(
                        f"由于 {e.__class__.__name__}，无法设置杠杆。消息: {e}"
                    ) from e
            except ccxt.BaseError as e:
                raise OperationalException(e) from e

    def get_max_pair_stake_amount(self, pair: str, price: float, leverage: float = 1.0) -> float:
        """获取交易对的最大持仓金额"""
        if self.trading_mode == TradingMode.SPOT:
            return float("inf")  # 实际上不是无限，但对于现货可能无关紧要

        if pair not in self._leverage_tiers:
            return float("inf")

        pair_tiers = self._leverage_tiers[pair]
        return pair_tiers[-1]["maxNotional"] / leverage

    def _get_stop_params(self, side: BuySell, ordertype: str, stop_price: float) -> dict:
        """获取止损订单参数"""
        params = super()._get_stop_params(side, ordertype, stop_price)
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode:
            params["tdMode"] = self.margin_mode.value  # 交易模式
            params["posSide"] = self._get_posSide(side, True)  # 持仓方向
        return params

    def _convert_stop_order(self, pair: str, order_id: str, order: CcxtOrder) -> CcxtOrder:
        """转换止损订单"""
        if (
            order.get("status", "open") == "closed"
            and (real_order_id := order.get("info", {}).get("ordId")) is not None
        ):
            # 一旦订单触发，我们获取常规的后续订单
            order_reg = self.fetch_order(real_order_id, pair)
            self._log_exchange_response("fetch_stoploss_order1", order_reg)
            order_reg["id_stop"] = order_reg["id"]
            order_reg["id"] = order_id
            order_reg["type"] = "stoploss"
            order_reg["status_stop"] = "triggered"
            return order_reg
        order = self._order_contracts_to_amount(order)
        order["type"] = "stoploss"
        return order

    @retrier(retries=API_RETRY_COUNT)
    def fetch_stoploss_order(
        self, order_id: str, pair: str, params: dict | None = None
    ) -> CcxtOrder:
        """获取止损订单"""
        if self._config["dry_run"]:
            return self.fetch_dry_run_order(order_id)

        try:
            params1 = {"stop": True}
            order_reg = self._api.fetch_order(order_id, pair, params=params1)
            self._log_exchange_response("fetch_stoploss_order", order_reg)
            return self._convert_stop_order(pair, order_id, order_reg)
        except (ccxt.OrderNotFound, ccxt.InvalidOrder):
            pass
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于 {e.__class__.__name__}，无法获取订单。消息: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

        return self._fetch_stop_order_fallback(order_id, pair)

    def _fetch_stop_order_fallback(self, order_id: str, pair: str) -> CcxtOrder:
        """获取止损订单的备选方法"""
        params2 = {"stop": True, "ordType": "conditional"}
        for method in (
            self._api.fetch_open_orders,
            self._api.fetch_closed_orders,
            self._api.fetch_canceled_orders,
        ):
            try:
                orders = method(pair, params=params2)
                orders_f = [order for order in orders if order["id"] == order_id]
                if orders_f:
                    order = orders_f[0]
                    return self._convert_stop_order(pair, order_id, order)
            except (ccxt.OrderNotFound, ccxt.InvalidOrder):
                pass
            except ccxt.DDoSProtection as e:
                raise DDosProtection(e) from e
            except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
                raise TemporaryError(
                    f"由于 {e.__class__.__name__}，无法获取订单。消息: {e}"
                ) from e
            except ccxt.BaseError as e:
                raise OperationalException(e) from e
        raise RetryableOrderError(f"未找到止损订单 (交易对: {pair} ID: {order_id})。")

    def get_order_id_conditional(self, order: CcxtOrder) -> str:
        """获取条件订单ID"""
        if order.get("type", "") == "stop":
            return safe_value_fallback2(order, order, "id_stop", "id")
        return order["id"]

    def cancel_stoploss_order(self, order_id: str, pair: str, params: dict | None = None) -> dict:
        """取消止损订单"""
        params1 = {"stop": True}
        return self.cancel_order(
            order_id=order_id,
            pair=pair,
            params=params1,
        )

    def _fetch_orders_emulate(self, pair: str, since_ms: int) -> list[CcxtOrder]:
        """模拟获取订单"""
        orders = []

        orders = self._api.fetch_closed_orders(pair, since=since_ms)
        if since_ms < dt_ts(dt_now() - timedelta(days=6, hours=23)):
            # 常规的fetch_closed_orders只返回7天的数据
            # 强制使用"archive"端点，返回3个月的数据
            params = {"method": "privateGetTradeOrdersHistoryArchive"}
            orders_hist = self._api.fetch_closed_orders(pair, since=since_ms, params=params)
            orders.extend(orders_hist)

        orders_open = self._api.fetch_open_orders(pair, since=since_ms)
        orders.extend(orders_open)
        return orders