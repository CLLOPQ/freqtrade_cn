"""
完整交易仓位对列表过滤器
"""

import logging

from freqtrade.exchange.exchange_types import Tickers
from freqtrade.persistence import Trade
from freqtrade.plugins.pairlist.IPairList import IPairList, SupportsBacktesting


logger = logging.getLogger(__name__)


class FullTradesFilter(IPairList):
    supports_backtesting = SupportsBacktesting.NO_ACTION

    @property
    def needstickers(self) -> bool:
        """
        定义是否需要行情数据的布尔属性。如果没有交易对列表需要行情数据，则将空列表作为行情数据参数传递给filter_pairlist方法
        """
        return False

    def short_desc(self) -> str:
        """
        简短的允许列表方法描述 - 用于启动消息
        """
        return f"{self.name} - 当交易仓位满时缩小允许列表"

    @staticmethod
    def description() -> str:
        return "当交易仓位满时缩小允许列表"

    def filter_pairlist(self, pairlist: list[str], tickers: Tickers) -> list[str]:
        """
        过滤并排序交易对列表，然后再次返回允许列表。在每个机器人迭代时调用 - 如果有必要，请使用内部缓存
        :param pairlist: 要过滤或排序的交易对列表
        :param tickers: 行情数据（来自exchange.get_tickers）。可能已缓存。
        :return: 新的允许列表
        """
        # 获取未平仓交易数量和最大未平仓交易配置
        num_open = Trade.get_open_trade_count()
        max_trades = self._config["max_open_trades"]

        if (num_open >= max_trades) and (max_trades > 0):
            return []

        return pairlist