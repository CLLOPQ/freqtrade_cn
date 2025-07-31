"""Kucoin 交易所子类。"""

import logging

from freqtrade.constants import BuySell
from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import CcxtOrder, FtHas


logger = logging.getLogger(__name__)


class Kucoin(Exchange):
    """Kucoin 交易所类。

    包含使 Freqtrade 能够与该交易所正常工作所需的调整。

    请注意，该交易所未包含在 Freqtrade 开发团队官方支持的交易所列表中。
    因此，某些功能可能仍无法按预期工作。
    """

    _ft_has: FtHas = {
        "stoploss_on_exchange": True,  # 支持交易所止损
        "stop_price_param": "stopPrice",  # 止损价格参数名
        "stop_price_prop": "stopPrice",  # 止损价格属性名
        "stoploss_order_types": {"limit": "limit", "market": "market"},  # 止损订单类型映射
        "l2_limit_range": [20, 100],  # L2订单簿深度范围
        "l2_limit_range_required": False,  # 不需要指定订单簿深度
        "order_time_in_force": ["GTC", "FOK", "IOC"],  # 订单有效期类型
    }

    def _get_stop_params(self, side: BuySell, ordertype: str, stop_price: float) -> dict:
        """
        获取止损订单的参数
        :param side: 交易方向（买入/卖出）
        :param ordertype: 订单类型
        :param stop_price: 止损价格
        :return: 包含止损参数的字典
        """
        params = self._params.copy()
        params.update({"stopPrice": stop_price, "stop": "loss"})  # 止损价格和止损类型
        return params

    def create_order(
        self,
        *,
        pair: str,
        ordertype: str,
        side: BuySell,
        amount: float,
        rate: float,
        leverage: float,
        reduceOnly: bool = False,
        time_in_force: str = "GTC",
    ) -> CcxtOrder:
        res = super().create_order(
            pair=pair,
            ordertype=ordertype,
            side=side,
            amount=amount,
            rate=rate,
            leverage=leverage,
            reduceOnly=reduceOnly,
            time_in_force=time_in_force,
        )
        # Kucoin 只返回订单ID。
        # 目前ccxt返回状态为'closed' - 这是ccxt自行添加的信息。
        # 由于我们严重依赖状态信息，因此必须在此处将其设置为'open'。
        # 参考: https://github.com/ccxt/ccxt/pull/16674, (https://github.com/ccxt/ccxt/pull/16553)
        if not self._config["dry_run"]:
            res["type"] = ordertype
            res["status"] = "open"
        return res