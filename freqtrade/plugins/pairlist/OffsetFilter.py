"""
偏移交易对列表过滤器
"""

import logging

from freqtrade.exceptions import OperationalException
from freqtrade.exchange.exchange_types import Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting


logger = logging.getLogger(__name__)


class OffsetFilter(IPairList):
    supports_backtesting = SupportsBacktesting.YES

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._offset = self._pairlistconfig.get("offset", 0)
        self._number_pairs = self._pairlistconfig.get("number_assets", 0)

        if self._offset < 0:
            raise OperationalException("OffsetFilter requires offset to be >= 0")

    @property
    def needstickers(self) -> bool:
        """
        定义是否需要行情数据的布尔属性。
        如果没有交易对列表需要行情数据，则将空字典作为行情数据参数传递给filter_pairlist方法
        """
        return False

    def short_desc(self) -> str:
        """
        简短的白名单方法描述 - 用于启动消息
        """
        if self._number_pairs:
            return f"{self.name} - 选取{self._number_pairs}个交易对，从{self._offset}开始。"
        return f"{self.name} - 按{self._offset}偏移交易对。"

    @staticmethod
    def description() -> str:
        return "偏移交易对列表过滤器。"

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "offset": {
                "type": "number",
                "default": 0,
                "description": "偏移量",
                "help": "交易对列表的偏移量。",
            },
            "number_assets": {
                "type": "number",
                "default": 0,
                "description": "资产数量",
                "help": "从偏移量开始，从交易对列表中使用的资产数量。",
            },
        }

    def filter_pairlist(self, pairlist: list[str], tickers: Tickers) -> list[str]:
        """
        过滤并排序交易对列表，然后再次返回白名单。
        在每个机器人迭代时调用 - 如果需要，请使用内部缓存
        :param pairlist: 要过滤或排序的交易对列表
        :param tickers: 行情数据（来自 exchange.get_tickers）。可能已缓存。
        :return: 新的白名单
        """
        if self._offset > len(pairlist):
            self.log_once(
                f"偏移量{self._offset}大于交易对数量{len(pairlist)}",
                logger.warning,
            )
        pairs = pairlist[self._offset :]
        if self._number_pairs:
            pairs = pairs[: self._number_pairs]

        return pairs