"""Gate.io 交易所子类"""

import logging
from datetime import datetime

import ccxt

from freqtrade.constants import BuySell
from freqtrade.enums import MarginMode, PriceType, TradingMode
from freqtrade.exceptions import DDosProtection, OperationalException, TemporaryError
from freqtrade.exchange import Exchange
from freqtrade.exchange.common import retrier
from freqtrade.exchange.exchange_types import CcxtOrder, FtHas
from freqtrade.misc import safe_value_fallback2


logger = logging.getLogger(__name__)


class Gate(Exchange):
    """
    Gate.io 交易所类。包含使 Freqtrade 能够与该交易所正常工作所需的调整。

    请注意，该交易所未包含在 Freqtrade 开发团队官方支持的交易所列表中。
    因此，某些功能可能仍无法按预期工作。
    """

    unified_account = False

    _ft_has: FtHas = {
        "order_time_in_force": ["GTC", "IOC"],  # 订单有效期类型
        "stoploss_on_exchange": True,  # 支持交易所止损
        "stoploss_order_types": {"limit": "limit"},  # 止损订单类型
        "stop_price_param": "stopPrice",  # 止损价格参数名
        "stop_price_prop": "stopPrice",  # 止损价格属性名
        "l2_limit_upper": 1000,  # 订单簿最大深度
        "marketOrderRequiresPrice": True,  # 市价单需要价格
        "trades_has_history": False,  # 交易历史不支持分页（端点支持但ccxt不支持）
    }

    _ft_has_futures: FtHas = {
        "needs_trading_fees": True,  # 需要交易费用
        "marketOrderRequiresPrice": False,  # 期货市价单不需要价格
        "funding_fee_candle_limit": 90,  # 资金费用蜡烛图限制
        "stop_price_type_field": "price_type",  # 止损价格类型字段
        "l2_limit_upper": 300,  # 期货订单簿最大深度
        "stoploss_blocks_assets": False,  # 止损订单不锁定资产
        "stop_price_type_value_mapping": {
            PriceType.LAST: 0,  # 最新价格对应值
            PriceType.MARK: 1,  # 标记价格对应值
            PriceType.INDEX: 2,  # 指数价格对应值
        },
    }

    _supported_trading_mode_margin_pairs: list[tuple[TradingMode, MarginMode]] = [
        # TradingMode.SPOT始终受支持，无需在此列表中
        # (TradingMode.MARGIN, MarginMode.CROSS),
        # (TradingMode.FUTURES, MarginMode.CROSS),
        (TradingMode.FUTURES, MarginMode.ISOLATED)  # 期货逐仓模式
    ]

    @retrier
    def additional_exchange_init(self) -> None:
        """
        额外的交易所初始化逻辑。
        此时.api已可用。
        如果需要，必须在子方法中重写。
        """
        try:
            if not self._config["dry_run"]:
                self._api.load_unified_status()
                is_unified = self._api.options.get("unifiedAccount")

                # 返回布尔值，指示是否为统一账户
                if is_unified:
                    self.unified_account = True
                    logger.info("Gate: 统一账户。")
                else:
                    self.unified_account = False
                    logger.info("Gate: 经典账户。")
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于 {e.__class__.__name__}，在 additional_exchange_init 中出错。消息: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

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
        if ordertype == "market" and self.trading_mode == TradingMode.FUTURES:
            params["type"] = "market"
            params.update({"timeInForce": "IOC"})
        return params

    def get_trades_for_order(
        self, order_id: str, pair: str, since: datetime, params: dict | None = None
    ) -> list:
        trades = super().get_trades_for_order(order_id, pair, since, params)

        if self.trading_mode == TradingMode.FUTURES:
            # 期货响应通常不包含费用。
            # 因此，Gate的期货订单不会包含费用，这会导致
            # 重复的"更新费用"循环和错误的计算。
            # 因此，如果费用不可用，我们会用费用修补响应。
            # 另一个包含费用的方法是
            # privateFuturesGetSettleAccountBook({"settle": "usdt"})
            pair_fees = self._trading_fees.get(pair, {})
            if pair_fees:
                for idx, trade in enumerate(trades):
                    fee = trade.get("fee", {})
                    if fee and fee.get("cost") is None:
                        takerOrMaker = trade.get("takerOrMaker", "taker")
                        if pair_fees.get(takerOrMaker) is not None:
                            trades[idx]["fee"] = {
                                "currency": self.get_pair_quote_currency(pair),
                                "cost": trade["cost"] * pair_fees[takerOrMaker],
                                "rate": pair_fees[takerOrMaker],
                            }
        return trades

    def get_order_id_conditional(self, order: CcxtOrder) -> str:
        return safe_value_fallback2(order, order, "id_stop", "id")

    def fetch_stoploss_order(
        self, order_id: str, pair: str, params: dict | None = None
    ) -> CcxtOrder:
        order = self.fetch_order(order_id=order_id, pair=pair, params={"stop": True})
        if order.get("status", "open") == "closed":
            # 下达真实订单 - 我们需要显式获取。
            val = "trade_id" if self.trading_mode == TradingMode.FUTURES else "fired_order_id"

            if new_orderid := order.get("info", {}).get(val):
                order1 = self.fetch_order(order_id=new_orderid, pair=pair, params=params)
                order1["id_stop"] = order1["id"]
                order1["id"] = order_id
                order1["type"] = "stoploss"
                order1["stopPrice"] = order.get("stopPrice")
                order1["status_stop"] = "triggered"

                return order1
        return order

    def cancel_stoploss_order(self, order_id: str, pair: str, params: dict | None = None) -> dict:
        return self.cancel_order(order_id=order_id, pair=pair, params={"stop": True})