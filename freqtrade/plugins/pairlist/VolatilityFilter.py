"""
波动率交易对列表过滤器
"""

import logging
import sys
from datetime import timedelta

import numpy as np
from cachetools import TTLCache
from pandas import DataFrame

from freqtrade.constants import ListPairsWithTimeframes
from freqtrade.exceptions import OperationalException
from freqtrade.exchange.exchange_types import Tickers
from freqtrade.misc import plural
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting
from freqtrade.util import dt_floor_day, dt_now, dt_ts


logger = logging.getLogger(__name__)


class VolatilityFilter(IPairList):
    """
    按波动率过滤交易对
    """

    supports_backtesting = SupportsBacktesting.NO

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._days = self._pairlistconfig.get("lookback_days", 10)
        self._min_volatility = self._pairlistconfig.get("min_volatility", 0)
        self._max_volatility = self._pairlistconfig.get("max_volatility", sys.maxsize)
        self._refresh_period = self._pairlistconfig.get("refresh_period", 1440)
        self._def_candletype = self._config["candle_type_def"]
        self._sort_direction: str | None = self._pairlistconfig.get("sort_direction", None)

        self._pair_cache: TTLCache = TTLCache(maxsize=1000, ttl=self._refresh_period)

        candle_limit = self._exchange.ohlcv_candle_limit("1d", self._config["candle_type_def"])
        if self._days < 1:
            raise OperationalException("波动率过滤器要求回溯天数至少为1")
        if self._days > candle_limit:
            raise OperationalException(
                "波动率过滤器要求回溯天数不超过交易所最大请求限制（{candle_limit}）"
            )
        if self._sort_direction not in [None, "asc", "desc"]:
            raise OperationalException(
                "波动率过滤器要求排序方向为None（未定义）、'asc'或'desc'中的一个"
            )

    @property
    def needstickers(self) -> bool:
        """
        定义是否需要行情数据的布尔属性。
        如果没有交易对列表需要行情数据，则以空列表作为tickers参数传递给filter_pairlist方法
        """
        return False

    def short_desc(self) -> str:
        """
        交易对列表过滤方法的简短描述 - 用于启动消息
        """
        return (
            f"{self.name} - 过滤波动率范围在{self._min_volatility}-{self._max_volatility}之间，"
            f"过去{self._days} {plural(self._days, '天')}的交易对。"
        )

    @staticmethod
    def description() -> str:
        return "按近期波动率过滤交易对。"

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "lookback_days": {
                "type": "number",
                "default": 10,
                "description": "回溯天数",
                "help": "查看回溯的天数。",
            },
            "min_volatility": {
                "type": "number",
                "default": 0,
                "description": "最小波动率",
                "help": "交易对必须具备的最小波动率才能被考虑。",
            },
            "max_volatility": {
                "type": "number",
                "default": None,
                "description": "最大波动率",
                "help": "交易对必须具备的最大波动率才能被考虑。",
            },
            "sort_direction": {
                "type": "option",
                "default": None,
                "options": ["", "asc", "desc"],
                "description": "排序交易对列表",
                "help": "按波动率升序或降序排序交易对列表。",
            },
            **IPairList.refresh_period_parameter(),
        }

    def filter_pairlist(self, pairlist: list[str], tickers: Tickers) -> list[str]:
        """
        验证交易范围
        :param pairlist: 要过滤或排序的交易对列表
        :param tickers: 行情数据（来自exchange.get_tickers）。可能已缓存。
        :return: 新的允许列表
        """
        needed_pairs: ListPairsWithTimeframes = [
            (p, "1d", self._def_candletype) for p in pairlist if p not in self._pair_cache
        ]

        since_ms = dt_ts(dt_floor_day(dt_now()) - timedelta(days=self._days))
        candles = self._exchange.refresh_ohlcv_with_cache(needed_pairs, since_ms=since_ms)

        resulting_pairlist: list[str] = []
        volatilitys: dict[str, float] = {}
        for p in pairlist:
            daily_candles = candles.get((p, "1d", self._def_candletype), None)

            volatility_avg = self._calculate_volatility(p, daily_candles)

            if volatility_avg is not None:
                if self._validate_pair_loc(p, volatility_avg):
                    resulting_pairlist.append(p)
                    volatilitys[p] = (
                        volatility_avg if volatility_avg and not np.isnan(volatility_avg) else 0
                    )
            else:
                self.log_once(f"从白名单中移除{p}，未找到K线数据。", logger.info)

        if self._sort_direction:
            resulting_pairlist = sorted(
                resulting_pairlist,
                key=lambda p: volatilitys[p],
                reverse=self._sort_direction == "desc",
            )
        return resulting_pairlist

    def _calculate_volatility(self, pair: str, daily_candles: DataFrame) -> float | None:
        # 检查缓存中的波动率
        if (volatility_avg := self._pair_cache.get(pair, None)) is not None:
            return volatility_avg

        if daily_candles is not None and not daily_candles.empty:
            returns = np.log(daily_candles["close"].shift(1) / daily_candles["close"])
            returns.fillna(0, inplace=True)

            volatility_series = returns.rolling(window=self._days).std() * np.sqrt(self._days)
            volatility_avg = volatility_series.mean()
            self._pair_cache[pair] = volatility_avg

            return volatility_avg
        else:
            return None

    def _validate_pair_loc(self, pair: str, volatility_avg: float) -> bool:
        """
        验证交易范围
        :param pair: 当前正在验证的交易对
        :param volatility_avg: 平均波动率
        :return: 如果交易对可以保留则返回True，否则返回False
        """

        if self._min_volatility <= volatility_avg <= self._max_volatility:
            result = True
        else:
            self.log_once(
                f"从白名单中移除{pair}，因为过去{self._days} {plural(self._days, '天')}的波动率为：{volatility_avg:.3f}，"
                f"不在配置的{self._min_volatility}-{self._max_volatility}范围内。",
                logger.info,
            )
            result = False
        return result