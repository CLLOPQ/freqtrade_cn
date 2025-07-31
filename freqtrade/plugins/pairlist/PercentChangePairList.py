"""
百分比变化交易对列表提供器

基于交易变化提供动态交易对列表，根据价格在定义的时间段内的百分比变化或从行情中获取的变化进行排序
"""

import logging
from datetime import timedelta
from typing import TypedDict

from cachetools import TTLCache
from pandas import DataFrame

from freqtrade.constants import ListPairsWithTimeframes, PairWithTimeframe
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.exchange.exchange_types import Ticker, Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting
from freqtrade.util import dt_now, format_ms_time


logger = logging.getLogger(__name__)


class SymbolWithPercentage(TypedDict):
    symbol: str
    percentage: float | None


class PercentChangePairList(IPairList):
    is_pairlist_generator = True
    supports_backtesting = SupportsBacktesting.NO

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if "number_assets" not in self._pairlistconfig:
            raise OperationalException(
                "`number_assets`未指定。请检查您的配置中是否包含'pairlist.config.number_assets'"
            )

        self._stake_currency = self._config["stake_currency"]
        self._number_pairs = self._pairlistconfig["number_assets"]
        self._min_value = self._pairlistconfig.get("min_value", None)
        self._max_value = self._pairlistconfig.get("max_value", None)
        self._refresh_period = self._pairlistconfig.get("refresh_period", 1800)
        self._pair_cache: TTLCache = TTLCache(maxsize=1, ttl=self._refresh_period)
        self._lookback_days = self._pairlistconfig.get("lookback_days", 0)
        self._lookback_timeframe = self._pairlistconfig.get("lookback_timeframe", "1d")
        self._lookback_period = self._pairlistconfig.get("lookback_period", 0)
        self._sort_direction: str | None = self._pairlistconfig.get("sort_direction", "desc")
        self._def_candletype = self._config["candle_type_def"]

        if (self._lookback_days > 0) & (self._lookback_period > 0):
            raise OperationalException(
                "配置不明确：交易对列表配置中同时设置了lookback_days和lookback_period。请仅设置lookback_days或同时设置lookback_period和lookback_timeframe，然后重启机器人。"
            )

        # 当设置了lookback_days时，覆盖回溯时间周期和天数
        if self._lookback_days > 0:
            self._lookback_timeframe = "1d"
            self._lookback_period = self._lookback_days

        # 获取时间周期（分钟和秒）
        self._tf_in_min = timeframe_to_minutes(self._lookback_timeframe)
        _tf_in_sec = self._tf_in_min * 60

        # 是否使用范围回溯
        self._use_range = (self._tf_in_min > 0) & (self._lookback_period > 0)

        if self._use_range & (self._refresh_period < _tf_in_sec):
            raise OperationalException(
                f"刷新周期{self._refresh_period}秒小于一个{self._lookback_timeframe}时间周期。请将刷新周期调整为至少{_tf_in_sec}秒，然后重启机器人。"
            )

        if not self._use_range and not (
            self._exchange.exchange_has("fetchTickers")
            and self._exchange.get_option("tickers_have_percentage")
        ):
            raise OperationalException(
                "交易所不支持此配置下的动态白名单。请编辑配置文件，移除PercentChangePairList，或切换为使用K线数据，然后重启机器人。"
            )

        candle_limit = self._exchange.ohlcv_candle_limit(
            self._lookback_timeframe, self._config["candle_type_def"]
        )

        if self._lookback_period > candle_limit:
            raise OperationalException(
                "变化过滤需要回溯周期不超过交易所最大请求数量（{candle_limit}）"
            )

    @property
    def needstickers(self) -> bool:
        """
        定义是否需要行情数据的布尔属性。
        如果没有交易对列表需要行情数据，则将空字典作为行情参数传递给filter_pairlist方法
        """
        return not self._use_range

    def short_desc(self) -> str:
        """
        简短的白名单方法描述 - 用于启动消息
        """
        return f"{self.name} - 按百分比变化排序的前{self._pairlistconfig['number_assets']}个交易对。"

    @staticmethod
    def description() -> str:
        return "基于百分比变化提供动态交易对列表。"

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "number_assets": {
                "type": "number",
                "default": 30,
                "description": "资产数量",
                "help": "从交易对列表中使用的资产数量",
            },
            "min_value": {
                "type": "number",
                "default": None,
                "description": "最小值",
                "help": "用于过滤交易对列表的最小值。",
            },
            "max_value": {
                "type": "number",
                "default": None,
                "description": "最大值",
                "help": "用于过滤交易对列表的最大值。",
            },
            "sort_direction": {
                "type": "option",
                "default": "desc",
                "options": ["", "asc", "desc"],
                "description": "排序方式",
                "help": "按变化率升序或降序对交易对列表进行排序。",
            },
            **IPairList.refresh_period_parameter(),
            "lookback_days": {
                "type": "number",
                "default": 0,
                "description": "回溯天数",
                "help": "回溯的天数。",
            },
            "lookback_timeframe": {
                "type": "string",
                "default": "1d",
                "description": "回溯时间周期",
                "help": "回溯使用的时间周期。",
            },
            "lookback_period": {
                "type": "number",
                "default": 0,
                "description": "回溯周期数",
                "help": "回溯的周期数。",
            },
        }

    def gen_pairlist(self, tickers: Tickers) -> list[str]:
        """
        生成交易对列表
        :param tickers: 行情数据（来自交易所的get_tickers方法）。可能已缓存。
        :return: 交易对列表
        """
        pairlist = self._pair_cache.get("pairlist")
        if pairlist:
            # 找到数据 - 无需刷新
            return pairlist.copy()
        else:
            # 使用新数据
            # 检查交易对的计价货币是否等于持仓货币
            _pairlist = [
                k
                for k in self._exchange.get_markets(
                    quote_currencies=[self._stake_currency], tradable_only=True, active_only=True
                ).keys()
            ]

            # 无需测试黑名单...
            _pairlist = self.verify_blacklist(_pairlist, logger.info)
            if not self._use_range:
                filtered_tickers = [
                    v
                    for k, v in tickers.items()
                    if (
                        self._exchange.get_pair_quote_currency(k) == self._stake_currency
                        and v["symbol"] in _pairlist
                    )
                ]
                pairlist = [s["symbol"] for s in filtered_tickers]
            else:
                pairlist = _pairlist

            pairlist = self.filter_pairlist(pairlist, tickers)
            self._pair_cache["pairlist"] = pairlist.copy()

        return pairlist

    def filter_pairlist(self, pairlist: list[str], tickers: dict) -> list[str]:
        """
        过滤并排序交易对列表，然后返回新的白名单。
        在每个机器人迭代时调用 - 如果需要，请使用内部缓存
        :param pairlist: 待过滤或排序的交易对列表
        :param tickers: 行情数据（来自交易所的get_tickers方法）。可能已缓存。
        :return: 新的白名单
        """
        filtered_tickers: list[SymbolWithPercentage] = [
            {"symbol": k, "percentage": None} for k in pairlist
        ]
        if self._use_range:
            # 使用回溯周期计算百分比变化
            filtered_tickers = self.fetch_percent_change_from_lookback_period(filtered_tickers)
        else:
            # 默认从支持的交易所行情中获取24小时变化
            filtered_tickers = self.fetch_percent_change_from_tickers(filtered_tickers, tickers)

        if self._min_value is not None:
            filtered_tickers = [v for v in filtered_tickers if v["percentage"] > self._min_value]
        if self._max_value is not None:
            filtered_tickers = [v for v in filtered_tickers if v["percentage"] < self._max_value]

        sorted_tickers = sorted(
            filtered_tickers,
            reverse=self._sort_direction == "desc",
            key=lambda t: t["percentage"],  # type: ignore
        )

        # 验证白名单仅包含活跃市场交易对
        pairs = self._whitelist_for_active_markets([s["symbol"] for s in sorted_tickers])
        pairs = self.verify_blacklist(pairs, logmethod=logger.info)
        # 将交易对列表限制为请求的数量
        pairs = pairs[: self._number_pairs]

        return pairs

    def fetch_candles_for_lookback_period(
        self, filtered_tickers: list[SymbolWithPercentage]
    ) -> dict[PairWithTimeframe, DataFrame]:
        since_ms = (
            int(
                timeframe_to_prev_date(
                    self._lookback_timeframe,
                    dt_now()
                    + timedelta(
                        minutes=-(self._lookback_period * self._tf_in_min) - self._tf_in_min
                    ),
                ).timestamp()
            )
            * 1000
        )
        to_ms = (
            int(
                timeframe_to_prev_date(
                    self._lookback_timeframe, dt_now() - timedelta(minutes=self._tf_in_min)
                ).timestamp()
            )
            * 1000
        )
        self.log_once(
            f"使用{self._lookback_period}根K线的变化范围，时间周期：{self._lookback_timeframe}，起始时间{format_ms_time(since_ms)}，结束时间{format_ms_time(to_ms)}",
            logger.info,
        )
        needed_pairs: ListPairsWithTimeframes = [
            (p, self._lookback_timeframe, self._def_candletype)
            for p in [s["symbol"] for s in filtered_tickers]
            if p not in self._pair_cache
        ]
        candles = self._exchange.refresh_ohlcv_with_cache(needed_pairs, since_ms)
        return candles

    def fetch_percent_change_from_lookback_period(
        self, filtered_tickers: list[SymbolWithPercentage]
    ) -> list[SymbolWithPercentage]:
        # 获取回溯周期（毫秒），用于交易所K线数据获取
        candles = self.fetch_candles_for_lookback_period(filtered_tickers)

        for i, p in enumerate(filtered_tickers):
            pair_candles = (
                candles[(p["symbol"], self._lookback_timeframe, self._def_candletype)]
                if (p["symbol"], self._lookback_timeframe, self._def_candletype) in candles
                else None
            )

            # 如果有K线数据，计算典型价格和K线变化
            if pair_candles is not None and not pair_candles.empty:
                current_close = pair_candles["close"].iloc[-1]
                previous_close = pair_candles["close"].shift(self._lookback_period).iloc[-1]
                pct_change = (
                    ((current_close - previous_close) / previous_close) * 100
                    if previous_close > 0
                    else 0
                )

                # 用上面计算的范围变化和替换变化值
                filtered_tickers[i]["percentage"] = pct_change
            else:
                filtered_tickers[i]["percentage"] = 0
        return filtered_tickers

    def fetch_percent_change_from_tickers(
        self, filtered_tickers: list[SymbolWithPercentage], tickers
    ) -> list[SymbolWithPercentage]:
        valid_tickers: list[SymbolWithPercentage] = []
        for p in filtered_tickers:
            # 过滤资产
            if (
                self._validate_pair(
                    p["symbol"], tickers[p["symbol"]] if p["symbol"] in tickers else None
                )
                and p["symbol"] != "UNI/USDT"
            ):
                p["percentage"] = tickers[p["symbol"]]["percentage"]
                valid_tickers.append(p)
        return valid_tickers

    def _validate_pair(self, pair: str, ticker: Ticker | None) -> bool:
        """
        检查一个价格步长（点）是否大于某个阈值。
        :param pair: 当前正在验证的交易对
        :param ticker: 从ccxt.fetch_ticker返回的ticker字典
        :return: 如果交易对可以保留则返回True，否则返回False
        """
        if not ticker or "percentage" not in ticker or ticker["percentage"] is None:
            self.log_once(
                f"将{pair}从白名单中移除，因为ticker['percentage']为空（通常是过去24小时内无交易）。",
                logger.info,
            )
            return False

        return True