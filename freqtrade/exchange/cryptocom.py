"""Crypto.com 交易所子类"""

import logging

from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Cryptocom(Exchange):
    """Crypto.com 交易所类。
    包含使 Freqtrade 能够与该交易所正常工作所需的调整。
    """

    _ft_has: FtHas = {
        "ohlcv_candle_limit": 300,  # OHLCV 蜡烛图数据的最大限制
    }