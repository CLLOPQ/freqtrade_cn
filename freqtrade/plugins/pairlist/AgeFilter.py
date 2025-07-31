"""
最小上市天数（天）交易对列表过滤器
"""

import logging
from copy import deepcopy
from datetime import timedelta

from pandas import DataFrame

from freqtrade.constants import ListPairsWithTimeframes
from freqtrade.exceptions import OperationalException
from freqtrade.exchange.exchange_types import Tickers
from freqtrade.misc import plural
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting
from freqtrade.util import PeriodicCache, dt_floor_day, dt_now, dt_ts


logger = logging.getLogger(__name__)


class AgeFilter(IPairList):
    supports_backtesting = SupportsBacktesting.NO

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # 已检查交易对缓存（交易对符号 => 时间戳的字典）
        self._symbolsChecked: dict[str, int] = {}
        self._symbolsCheckFailed = PeriodicCache(maxsize=1000, ttl=86_400)

        self._min_days_listed = self._pairlistconfig.get("min_days_listed", 10)
        self._max_days_listed = self._pairlistconfig.get("max_days_listed")

        candle_limit = self._exchange.ohlcv_candle_limit("1d", self._config["candle_type_def"])
        if self._min_days_listed < 1:
            raise OperationalException("AgeFilter要求min_days_listed至少为1")
        if self._min_days_listed > candle_limit:
            raise OperationalException(
                "AgeFilter要求min_days_listed不超过交易所最大请求大小"
                f"({candle_limit})"
            )
        if self._max_days_listed and self._max_days_listed <= self._min_days_listed:
            raise OperationalException("AgeFilter不允许max_days_listed小于等于min_days_listed")
        if self._max_days_listed and self._max_days_listed > candle_limit:
            raise OperationalException(
                "AgeFilter要求max_days_listed不超过交易所最大请求大小"
                f"({candle_limit})"
            )

    @property
    def needstickers(self) -> bool:
        """
        定义是否需要行情数据的布尔属性。
        如果没有交易对列表需要行情数据，则传递空字典作为tickers参数给filter_pairlist方法
        """
        return False

    def short_desc(self) -> str:
        """
        简短的白名单方法描述 - 用于启动消息
        """
        return (
            f"{self.name} - 过滤上市年龄小于"
            f"{self._min_days_listed} {plural(self._min_days_listed, '天')}"
        ) + (
            ("或超过{self._max_days_listed} {plural(self._max_days_listed, '天')}")
            if self._max_days_listed
            else ""
        )

    @staticmethod
    def description() -> str:
        return "按上市天数（天）过滤交易对。"

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "min_days_listed": {
                "type": "number",
                "default": 10,
                "description": "最小上市天数",
                "help": "交易对必须在交易所上市的最小天数。",
            },
            "max_days_listed": {
                "type": "number",
                "default": None,
                "description": "最大上市天数",
                "help": "交易对必须在交易所上市的最大天数。",
            },
        }

    def filter_pairlist(self, pairlist: list[str], tickers: Tickers) -> list[str]:
        """
        :param pairlist: 要过滤或排序的交易对列表
        :param tickers: 行情数据（来自交易所.get_tickers）。可能已缓存。
        :return: 新的允许列表
        """
        needed_pairs: ListPairsWithTimeframes = [
            (p, "1d", self._config["candle_type_def"])
            for p in pairlist
            if p not in self._symbolsChecked and p not in self._symbolsCheckFailed
        ]
        if not needed_pairs:
            # 移除之前已被排除的交易对
            return [p for p in pairlist if p not in self._symbolsCheckFailed]

        since_days = (
            -(self._max_days_listed if self._max_days_listed else self._min_days_listed) - 1
        )
        since_ms = dt_ts(dt_floor_day(dt_now()) + timedelta(days=since_days))
        candles = self._exchange.refresh_latest_ohlcv(needed_pairs, since_ms=since_ms, cache=False)
        if self._enabled:
            for p in deepcopy(pairlist):
                daily_candles = (
                    candles[(p, "1d", self._config["candle_type_def"])]
                    if (p, "1d", self._config["candle_type_def"]) in candles
                    else None
                )
                if not self._validate_pair_loc(p, daily_candles):
                    pairlist.remove(p)
        self.log_once(f"验证了{len(pairlist)}个交易对。", logger.info)
        return pairlist

    def _validate_pair_loc(self, pair: str, daily_candles: DataFrame | None) -> bool:
        """
        验证交易对的上市年龄
        :param pair: 当前正在验证的交易对
        :param daily_candles: 下载的日K线数据
        :return: 如果交易对可以保留则返回True，否则返回False
        """
        # 检查交易对是否在缓存中
        if pair in self._symbolsChecked:
            return True

        if daily_candles is not None:
            if len(daily_candles) >= self._min_days_listed and (
                not self._max_days_listed or len(daily_candles) <= self._max_days_listed
            ):
                # 已获取至少所需的最小数量的日K线
                # 添加到缓存，存储上次检查该交易对的时间
                self._symbolsChecked[pair] = dt_ts()
                return True
            else:
                self.log_once(
                    (
                        f"从白名单中移除了{pair}，因为其上市天数"
                        f"{len(daily_candles)}小于{self._min_days_listed}"
                        f"{plural(self._min_days_listed, '天')}"
                    )
                    + (
                        (
                            "或超过"
                            f"{self._max_days_listed} {plural(self._max_days_listed, '天')}"
                        )
                        if self._max_days_listed
                        else ""
                    ),
                    logger.info,
                )
                self._symbolsCheckFailed[pair] = dt_ts()
                return False
        return False