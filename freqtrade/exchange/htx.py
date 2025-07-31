"""HTX 交易所子类"""

import logging

from freqtrade.constants import BuySell
from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Htx(Exchange):
    """
    HTX 交易所类。包含使 Freqtrade 能够与该交易所正常工作所需的调整。
    """

    _ft_has: FtHas = {
        "stoploss_on_exchange": True,  # 支持交易所止损
        "stop_price_param": "stopPrice",  # 止损价格参数名
        "stop_price_prop": "stopPrice",  # 止损价格属性名
        "stoploss_order_types": {"limit": "stop-limit"},  # 止损订单类型映射
        "l2_limit_range": [5, 10, 20],  # L2订单簿深度范围
        "l2_limit_range_required": False,  # 不需要指定订单簿深度
        "ohlcv_candle_limit_per_timeframe": {
            "1w": 500,  # 1周时间框架的蜡烛图限制
            "1M": 500,  # 1月时间框架的蜡烛图限制
        },
        "trades_has_history": False,  # 交易历史不支持分页（端点没有"since"参数）
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
        params.update(
            {
                "stopPrice": stop_price,  # 止损价格
                "operator": "lte",  # 操作符，lte表示小于等于
            }
        )
        return params