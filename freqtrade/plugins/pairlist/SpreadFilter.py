"""
价差交易对列表过滤器
"""

import logging

from freqtrade.exceptions import OperationalException
from freqtrade.exchange.exchange_types import Ticker
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting


logger = logging.getLogger(__name__)


class SpreadFilter(IPairList):
    supports_backtesting = SupportsBacktesting.NO

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._max_spread_ratio = self._pairlistconfig.get("max_spread_ratio", 0.005)
        self._enabled = self._max_spread_ratio != 0

        if not self._exchange.get_option("tickers_have_bid_ask"):
            raise OperationalException(
                f"{self.name} requires exchange to have bid/ask data for tickers, "
                "which is not available for the selected exchange / trading mode."
            )

    @property
    def needstickers(self) -> bool:
        """
        定义是否需要交易对数据的布尔属性。如果没有交易对列表需要交易对数据，则将空字典作为交易对数据参数传递给filter_pairlist方法
        """
        return True

    def short_desc(self) -> str:
        """
        简短的白名单方法描述 - 用于启动消息
        """
        return (
            f"{self.name} - 过滤买卖价差超过{self._max_spread_ratio:.2%}的交易对。"
        )

    @staticmethod
    def description() -> str:
        return "根据买卖价差进行过滤。"

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "max_spread_ratio": {
                "type": "number",
                "default": 0.005,
                "description": "最大价差比率",
                "help": "交易对被考虑的最大价差比率。",
            },
        }

    def _validate_pair(self, pair: str, ticker: Ticker | None) -> bool:
        """
        验证交易对的价差
        :param pair: 当前正在验证的交易对
        :param ticker: 从ccxt.fetch_ticker返回的交易对数据字典
        :return: 如果交易对可以保留则返回True，否则返回False
        """
        if ticker and "bid" in ticker and "ask" in ticker and ticker["ask"] and ticker["bid"]:
            spread = 1 - ticker["bid"] / ticker["ask"]
            if spread > self._max_spread_ratio:
                self.log_once(
                    f"从白名单中移除了{pair}，因为价差{spread:.3%} > {self._max_spread_ratio:.3%}",
                    logger.info,
                )
                return False
            else:
                return True
        self.log_once(
            f"由于交易对数据无效，从白名单中移除了{pair}：{ticker}", logger.info
        )
        return False