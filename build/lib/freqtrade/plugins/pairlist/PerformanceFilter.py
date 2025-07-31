"""
性能交易对列表过滤器
"""

import logging
from datetime import timedelta

import pandas as pd

from freqtrade.exchange.exchange_types import Tickers
from freqtrade.persistence import Trade
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting
from freqtrade.util.datetime_helpers import dt_now


logger = logging.getLogger(__name__)


class PerformanceFilter(IPairList):
    supports_backtesting = SupportsBacktesting.NO_ACTION

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._minutes = self._pairlistconfig.get("minutes", 0)
        self._min_profit = self._pairlistconfig.get("min_profit")

    @property
    def needstickers(self) -> bool:
        """
        定义是否需要行情数据的布尔属性。
        如果没有交易对列表需要行情数据，则将空列表作为行情数据参数传递给filter_pairlist方法
        """
        return False

    def short_desc(self) -> str:
        """
        简短的允许列表方法描述 - 用于启动消息
        """
        return f"{self.name} - 按性能对交易对进行排序。"

    @staticmethod
    def description() -> str:
        return "按性能过滤交易对。"

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "minutes": {
                "type": "number",
                "default": 0,
                "description": "分钟数",
                "help": "考虑过去X分钟内的交易。0表示所有交易。",
            },
            "min_profit": {
                "type": "number",
                "default": None,
                "description": "最低利润",
                "help": "以百分比表示的最低利润。利润低于此值的交易对将被移除。",
            },
        }

    def filter_pairlist(self, pairlist: list[str], tickers: Tickers) -> list[str]:
        """
        过滤并排序交易对列表，然后再次返回允许列表。
        在每个机器人迭代时调用 - 如果需要，请使用内部缓存
        :param pairlist: 要过滤或排序的交易对列表
        :param tickers: 行情数据（来自 exchange.get_tickers）。可能已缓存。
        :return: 新的允许列表
        """
        # 获取数据库中交易对的交易性能
        try:
            start_date = dt_now() - timedelta(minutes=self._minutes)
            performance = pd.DataFrame(Trade.get_overall_performance(start_date))
        except AttributeError:
            # 性能过滤器在回测模式下不可用。
            self.log_once("性能过滤器在这种模式下不可用。", logger.warning)
            return pairlist

        # 如果没有性能数据可用，则跳过基于性能的排序
        if len(performance) == 0:
            return pairlist

        # 从性能数据框值获取交易对列表
        list_df = pd.DataFrame({"pair": pairlist})
        list_df["prior_idx"] = list_df.index

        # 将无交易的交易对的初始值设为0
        # 使用以下方式对列表进行排序：
        #  - 主要按性能（从高到低）
        #  - 然后按数量（从低到高，以便在相同性能下优先选择交易次数少的）
        #  - 最后按原始索引，保持原始排序顺序
        sorted_df = (
            list_df.merge(performance, on="pair", how="left")
            .fillna(0)
            .sort_values(by=["profit_ratio", "count", "prior_idx"], ascending=[False, True, True])
        )
        if self._min_profit is not None:
            removed = sorted_df[sorted_df["profit_ratio"] < self._min_profit]
            for _, row in removed.iterrows():
                self.log_once(
                    f"移除交易对 {row['pair']}，因为其 {row['profit_ratio']} 低于 {self._min_profit}",
                    logger.info,
                )
            sorted_df = sorted_df[sorted_df["profit_ratio"] >= self._min_profit]

        pairlist = sorted_df["pair"].tolist()

        return pairlist