"""
静态交易对列表提供器

提供在配置中指定的交易对白名单
"""

import logging
from copy import deepcopy

from freqtrade.exchange.exchange_types import Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting


logger = logging.getLogger(__name__)


class StaticPairList(IPairList):
    is_pairlist_generator = True
    supports_backtesting = SupportsBacktesting.YES

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._allow_inactive = self._pairlistconfig.get("allow_inactive", False)

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
        -> 请在子类中重写
        """
        return f"{self.name}"

    @staticmethod
    def description() -> str:
        return "使用配置中指定的交易对列表。"

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "allow_inactive": {
                "type": "boolean",
                "default": False,
                "description": "允许非活跃交易对",
                "help": "允许非活跃交易对出现在白名单中。",
            },
        }

    def gen_pairlist(self, tickers: Tickers) -> list[str]:
        """
        生成交易对列表
        :param tickers: 行情数据（来自交易所的get_tickers方法）。可能已缓存。
        :return: 交易对列表
        """
        wl = self.verify_whitelist(
            self._config["exchange"]["pair_whitelist"], logger.info, keep_invalid=True
        )
        if self._allow_inactive:
            return wl
        else:
            # 避免隐式过滤"verify_whitelist"以保留日志中的正确警告
            return self._whitelist_for_active_markets(wl)

    def filter_pairlist(self, pairlist: list[str], tickers: Tickers) -> list[str]:
        """
        过滤并排序交易对列表，再次返回白名单。
        在每个机器人迭代时调用 - 如果需要，请使用内部缓存
        :param pairlist: 要过滤或排序的交易对列表
        :param tickers: 行情数据（来自交易所的get_tickers方法）。可能已缓存。
        :return: 新的白名单
        """
        pairlist_ = deepcopy(pairlist)
        for pair in self._config["exchange"]["pair_whitelist"]:
            if pair not in pairlist_:
                pairlist_.append(pair)
        return pairlist_