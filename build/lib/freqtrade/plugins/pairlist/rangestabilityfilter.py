"""
变化率交易对列表过滤器
"""

import logging
from datetime import timedelta

from cachetools import TTLCache
from pandas import DataFrame

from freqtrade.constants import ListPairsWithTimeframes
from freqtrade.exceptions import OperationalException
from freqtrade.exchange.exchange_types import Tickers
from freqtrade.misc import plural
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting
from freqtrade.util import dt_floor_day, dt_now, dt_ts


logger = logging.getLogger(__name__)


class RangeStabilityFilter(IPairList):
    supports_backtesting = SupportsBacktesting.NO

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._days = self._pairlistconfig.get("lookback_days", 10)
        self._min_rate_of_change = self._pairlistconfig.get("min_rate_of_change", 0.01)
        self._max_rate_of_change = self._pairlistconfig.get("max_rate_of_change")
        self._refresh_period = self._pairlistconfig.get("refresh_period", 86400)
        self._def_candletype = self._config["candle_type_def"]
        self._sort_direction: str | None = self._pairlistconfig.get("sort_direction", None)

        self._pair_cache: TTLCache = TTLCache(maxsize=1000, ttl=self._refresh_period)

        candle_limit = self._exchange.ohlcv_candle_limit("1d", self._config["candle_type_def"])
        if self._days < 1:
            raise OperationalException("RangeStabilityFilter要求回溯天数至少为1")
        if self._days > candle_limit:
            raise OperationalException(
                "RangeStabilityFilter要求回溯天数不超过交易所最大请求数量（"
                f"{candle_limit}）"
            )
        if self._sort_direction not in [None, "asc", "desc"]:
            raise OperationalException(
                "RangeStabilityFilter要求排序方向为None（未定义）、'asc'或'desc'"
            )

    @property
    def needstickers(self) -> bool:
        """
        定义是否需要行情数据的布尔属性。如果没有交易对列表需要行情数据，则将空列表作为行情数据参数传递给filter_pairlist方法
        """
        return False

    def short_desc(self) -> str:
        """
        简短的白名单方法描述 - 用于启动消息
        """
        max_rate_desc = ""
        if self._max_rate_of_change:
            max_rate_desc = f"且不超过 {self._max_rate_of_change}"
        return (
            f"{self.name} - 过滤变化率低于 {self._min_rate_of_change} {max_rate_desc} 的交易对，"
            f"过滤周期为过去 {plural(self._days, '天')}。"
        )

    @staticmethod
    def description() -> str:
        return "通过交易对的变化率过滤交易对。"

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "lookback_days": {
                "type": "number",
                "default": 10,
                "description": "回溯天数",
                "help": "回溯的天数。",
            },
            "min_rate_of_change": {
                "type": "number",
                "default": 0.01,
                "description": "最小变化率",
                "help": "用于过滤交易对的最小变化率。",
            },
            "max_rate_of_change": {
                "type": "number",
                "default": None,
                "description": "最大变化率",
                "help": "用于过滤交易对的最大变化率。",
            },
            "sort_direction": {
                "type": "option",
                "default": None,
                "options": ["", "asc", "desc"],
                "description": "对交易对列表排序",
                "help": "按变化率升序或降序对交易对列表排序。",
            },
            **IPairList.refresh_period_parameter(),
        }

    def filter_pairlist(self, pairlist: list[str], tickers: Tickers) -> list[str]:
        """
        验证交易范围：param pairlist：要过滤或排序的交易对列表：param tickers：行情数据（来自exchange.get_tickers）。可能已缓存。：return：新的允许列表
        """
        needed_pairs: ListPairsWithTimeframes = [
            (p, "1d", self._def_candletype) for p in pairlist if p not in self._pair_cache
        ]

        since_ms = dt_ts(dt_floor_day(dt_now()) - timedelta(days=self._days + 1))
        candles = self._exchange.refresh_ohlcv_with_cache(needed_pairs, since_ms=since_ms)

        resulting_pairlist: list[str] = []
        pct_changes: dict[str, float] = {}

        for p in pairlist:
            daily_candles = candles.get((p, "1d", self._def_candletype), None)

            pct_change = self._calculate_rate_of_change(p, daily_candles)

            if pct_change is not None:
                if self._validate_pair_loc(p, pct_change):
                    resulting_pairlist.append(p)
                    pct_changes[p] = pct_change
            else:
                self.log_once(f"从白名单中移除 {p}，未找到蜡烛数据。", logger.info)

        if self._sort_direction:
            resulting_pairlist = sorted(
                resulting_pairlist,
                key=lambda p: pct_changes[p],
                reverse=self._sort_direction == "desc",
            )
        return resulting_pairlist

    def _calculate_rate_of_change(self, pair: str, daily_candles: DataFrame) -> float | None:
        # 检查交易对是否在缓存中
        if (pct_change := self._pair_cache.get(pair, None)) is not None:
            return pct_change
        if daily_candles is not None and not daily_candles.empty:
            highest_high = daily_candles["high"].max()
            lowest_low = daily_candles["low"].min()
            # 计算变化率：（最高最高价 - 最低最低价）/ 最低最低价
            pct_change = ((highest_high - lowest_low) / lowest_low) if lowest_low > 0 else 0
            self._pair_cache[pair] = pct_change
            return pct_change
        else:
            return None

    def _validate_pair_loc(self, pair: str, pct_change: float) -> bool:
        """
        验证交易范围：param pair：当前正在验证的交易对：param pct_change：变化率：return：如果交易对可以保留则返回True，否则返回False
        """

        result = True
        if pct_change < self._min_rate_of_change:
            self.log_once(
                f"从白名单中移除 {pair}，因为在过去 {self._days} {plural(self._days, '天')} 内的变化率为 {pct_change:.3f}，"
                f"低于阈值 {self._min_rate_of_change}。",
                logger.info,
            )
            result = False
        if self._max_rate_of_change:
            if pct_change > self._max_rate_of_change:
                self.log_once(
                    f"从白名单中移除 {pair}，因为在过去 {self._days} {plural(self._days, '天')} 内的变化率为 {pct_change:.3f}，"
                    f"高于阈值 {self._max_rate_of_change}。",
                    logger.info,
                )
                result = False
        return result