"""Hitbtc 交易所子类"""

import logging

from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Hitbtc(Exchange):
    """
    Hitbtc 交易所类。包含使 Freqtrade 能够与该交易所正常工作所需的调整。

    请注意，该交易所未包含在 Freqtrade 开发团队官方支持的交易所列表中。
    因此，某些功能可能仍无法按预期工作。
    """

    _ft_has: FtHas = {
        "ohlcv_candle_limit": 1000,  # OHLCV 蜡烛图数据的最大限制
    }