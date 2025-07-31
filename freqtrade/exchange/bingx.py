"""Bingx 交易所子类"""

import logging

from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Bingx(Exchange):
    """
    Bingx 交易所类。包含使 Freqtrade 能够与该交易所正常工作所需的调整。
    """

    _ft_has: FtHas = {
        "ohlcv_candle_limit": 1000,  # OHLCV 蜡烛图数据的最大限制
        "stoploss_on_exchange": True,  # 支持交易所止损
        "stoploss_order_types": {"limit": "limit", "market": "market"},  # 止损订单类型
        "order_time_in_force": ["GTC", "IOC", "PO"],  # 订单有效期类型
        "trades_has_history": False,  # 交易历史不支持分页
    }