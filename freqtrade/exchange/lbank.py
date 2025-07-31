"""Lbank 交易所子类"""

import logging

from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Lbank(Exchange):
    """
    Lbank 交易所类。包含使 Freqtrade 能够与该交易所正常工作所需的调整。
    """

    _ft_has: FtHas = {
        "ohlcv_candle_limit": 1998,  # 低于允许的2000以避免当前蜡烛图问题
        "trades_has_history": False,  # 交易历史不支持分页
    }