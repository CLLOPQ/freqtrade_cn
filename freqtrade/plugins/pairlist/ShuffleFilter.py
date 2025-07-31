"""
随机打乱交易对列表过滤器
"""

import logging
import random
from typing import Literal

from freqtrade.enums import RunMode
from freqtrade.exchange import timeframe_to_seconds
from freqtrade.exchange.exchange_types import Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting
from freqtrade.util.periodic_cache import PeriodicCache


logger = logging.getLogger(__name__)

ShuffleValues = Literal["candle", "iteration"]


class ShuffleFilter(IPairList):
    supports_backtesting = SupportsBacktesting.YES

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # 在回测模式下应用种子以获得可比较的结果，
        # 但在实盘模式下不应用以确保实盘时交易对顺序不重复。
        if self._config.get("runmode") in (RunMode.LIVE, RunMode.DRY_RUN):
            self._seed = None
            logger.info("检测到实盘模式，不应用种子。")
        else:
            self._seed = self._pairlistconfig.get("seed")
            logger.info(f"检测到回测模式，应用种子值：{self._seed}")

        self._random = random.Random(self._seed)  # noqa: S311
        self._shuffle_freq: ShuffleValues = self._pairlistconfig.get("shuffle_frequency", "candle")
        self.__pairlist_cache = PeriodicCache(
            maxsize=1000, ttl=timeframe_to_seconds(self._config["timeframe"])
        )

    @property
    def needstickers(self) -> bool:
        """
        定义是否需要行情数据的布尔属性。
        如果没有交易对列表需要行情数据，则会将空字典作为行情数据参数传递给filter_pairlist方法
        """
        return False

    def short_desc(self) -> str:
        """
        简短的白名单方法描述 - 用于启动消息
        """
        return f"{self.name} - 每{self._shuffle_freq}打乱交易对" + (
            f"，种子 = {self._seed}。" if self._seed is not None else "。"
        )

    @staticmethod
    def description() -> str:
        return "随机打乱交易对列表顺序。"

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "shuffle_frequency": {
                "type": "option",
                "default": "candle",
                "options": ["candle", "iteration"],
                "description": "打乱频率",
                "help": "打乱频率。可以是'candle'（蜡烛图周期）或'iteration'（迭代周期）。",
            },
            "seed": {
                "type": "number",
                "default": None,
                "description": "随机种子",
                "help": "随机数生成器的种子。在实盘模式下不使用。",
            },
        }

    def filter_pairlist(self, pairlist: list[str], tickers: Tickers) -> list[str]:
        """
        过滤并排序交易对列表，然后返回更新后的白名单。
        在每个机器人迭代时调用 - 如果需要，请使用内部缓存。
        :param pairlist: 要过滤或排序的交易对列表
        :param tickers: 行情数据（来自exchange.get_tickers）。可能已缓存。
        :return: 新的白名单
        """
        pairlist_bef = tuple(pairlist)
        pairlist_new = self.__pairlist_cache.get(pairlist_bef)
        if pairlist_new and self._shuffle_freq == "candle":
            # 使用缓存的交易对列表。
            return pairlist_new
        # 在原地打乱交易对列表
        self._random.shuffle(pairlist)
        self.__pairlist_cache[pairlist_bef] = pairlist

        return pairlist