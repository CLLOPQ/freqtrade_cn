"""Bitmart 交易所子类"""

import logging

from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Bitmart(Exchange):
    """
    Bitmart 交易所类。包含使 Freqtrade 能够与该交易所正常工作所需的调整。
    """

    _ft_has: FtHas = {
        "stoploss_on_exchange": False,  # Bitmart API 不支持止损订单
        "ohlcv_candle_limit": 200,  # OHLCV 蜡烛图数据的最大限制
        "trades_has_history": False,  # 交易历史不支持分页
    }