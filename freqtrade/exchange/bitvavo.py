"""Bitvavo 交易所子类。"""

import logging

from ccxt import DECIMAL_PLACES

from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Bitvavo(Exchange):
    """Bitvavo 交易所类。

    包含使 Freqtrade 能够与该交易所正常工作所需的调整。

    请注意，该交易所未包含在 Freqtrade 开发团队官方支持的交易所列表中。
    因此，某些功能可能仍仍无法按预期工作。
    """

    _ft_has: FtHas = {
        "ohlcv_candle_limit": 1440,  # OHLCV 蜡烛图数据的最大限制
    }

    @property
    def precisionMode(self) -> int:
        """
        交易所 ccxt 精度模式
        由于 https://github.com/ccxt/ccxt/issues/20408 而重写
        """
        return DECIMAL_PLACES