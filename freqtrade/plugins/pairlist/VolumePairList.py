"""
成交量交易对列表提供器 基于交易量提供动态交易对列表
"""

import logging
from datetime import timedelta
from typing import Any, Literal

from cachetools import TTLCache

from freqtrade.constants import ListPairsWithTimeframes
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.exchange.exchange_types import Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting
from freqtrade.util import dt_now, format_ms_time


logger = logging.getLogger(__name__)


SORT_VALUES = ["quoteVolume"]


class VolumePairList(IPairList):
    is_pairlist_generator = True
    supports_backtesting = SupportsBacktesting.NO

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if "number_assets" not in self._pairlistconfig:
            raise OperationalException(
                "`number_assets`未指定。请检查您的配置中“pairlist.config.number_assets”项"
            )

        self._stake_currency = self._config["stake_currency"]
        self._number_pairs = self._pairlistconfig["number_assets"]
        self._sort_key: Literal["quoteVolume"] = self._pairlistconfig.get("sort_key", "quoteVolume")
        self._min_value = self._pairlistconfig.get("min_value", 0)
        self._max_value = self._pairlistconfig.get("max_value", None)
        self._refresh_period = self._pairlistconfig.get("refresh_period", 1800)
        self._pair_cache: TTLCache = TTLCache(maxsize=1, ttl=self._refresh_period)
        self._lookback_days = self._pairlistconfig.get("lookback_days", 0)
        self._lookback_timeframe = self._pairlistconfig.get("lookback_timeframe", "1d")
        self._lookback_period = self._pairlistconfig.get("lookback_period", 0)
        self._def_candletype = self._config["candle_type_def"]

        if (self._lookback_days > 0) & (self._lookback_period > 0):
            raise OperationalException(
                "配置不明确：交易对列表配置中同时设置了lookback_days和lookback_period。请仅设置lookback_days，或设置lookback_period和lookback_timeframe，然后重启机器人。"
            )

        # 当设置了lookback_days时，覆盖lookback_timeframe和lookback_period
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
                f"刷新周期为{self._refresh_period}秒，小于一个{self._lookback_timeframe}周期。请将刷新周期调整为至少{_tf_in_sec}秒，然后重启机器人。"
            )

        if not self._use_range and not (
            self._exchange.exchange_has("fetchTickers")
            and self._exchange.get_option("tickers_have_quoteVolume")
        ):
            raise OperationalException(
                "交易所不支持此配置下的动态白名单。请编辑配置文件，移除Volumepairlist，或切换到使用candles模式，然后重启机器人。"
            )

        if not self._validate_keys(self._sort_key):
            raise OperationalException(f"键{self._sort_key}不在{SORT_VALUES}中")

        candle_limit = self._exchange.ohlcv_candle_limit(
            self._lookback_timeframe, self._config["candle_type_def"]
        )
        if self._lookback_period < 0:
            raise OperationalException("成交量过滤要求lookback_period >= 0")
        if self._lookback_period > candle_limit:
            raise OperationalException(
                "成交量过滤要求lookback_period不超过交易所最大请求数量({candle_limit})"
            )

    @property
    def needstickers(self) -> bool:
        """
        定义是否需要行情数据的布尔属性。如果没有交易对列表需要行情数据，则将空字典作为行情数据参数传递给filter_pairlist方法
        """
        return not self._use_range

    def _validate_keys(self, key):
        return key in SORT_VALUES

    def short_desc(self) -> str:
        """
        简短的交易对列表方法描述 - 用于启动消息
        """
        return f"{self.name} - 前{self._pairlistconfig['number_assets']}高成交量交易对。"

    @staticmethod
    def description() -> str:
        return "基于交易量提供动态交易对列表。"

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "number_assets": {
                "type": "number",
                "default": 30,
                "description": "资产数量",
                "help": "从交易对列表中使用的资产数量",
            },
            "sort_key": {
                "type": "option",
                "default": "quoteVolume",
                "options": SORT_VALUES,
                "description": "排序键",
                "help": "用于对交易对列表排序的排序键。",
            },
            "min_value": {
                "type": "number",
                "default": 0,
                "description": "最小值",
                "help": "用于过滤交易对列表的最小值。",
            },
            "max_value": {
                "type": "number",
                "default": None,
                "description": "最大值",
                "help": "用于过滤交易对列表的最大值。",
            },
            **IPairList.refresh_period_parameter(),
            "lookback_days": {
                "type": "number",
                "default": 0,
                "description": "回溯天数",
                "help": "查看的天数。",
            },
            "lookback_timeframe": {
                "type": "string",
                "default": "",
                "description": "回溯时间周期",
                "help": "用于回溯的时间周期。",
            },
            "lookback_period": {
                "type": "number",
                "default": 0,
                "description": "回溯周期数",
                "help": "查看的周期数。",
            },
        }

    def gen_pairlist(self, tickers: Tickers) -> list[str]:
        """
        生成交易对列表
        :param tickers: 行情数据（来自exchange.get_tickers）。可能已缓存。
        :return: 交易对列表
        """
        # 生成动态交易对白名单
        # 如果此交易对列表不是第一个，则必须始终运行。
        pairlist = self._pair_cache.get("pairlist")
        if pairlist:
            # 找到缓存项 - 无需刷新
            return pairlist.copy()
        else:
            # 使用新的交易对列表
            # 检查交易对的计价货币是否等于 stake_currency
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
                        and (self._use_range or v.get(self._sort_key) is not None)
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
        过滤并排序交易对列表，然后返回新的交易对白名单。在每个机器人迭代时调用 - 如果需要，请使用内部缓存
        :param pairlist: 待过滤或排序的交易对列表
        :param tickers: 行情数据（来自exchange.get_tickers）。可能已缓存。
        :return: 新的交易对白名单
        """
        if self._use_range:
            # 从行情数据结构中创建基本数据。
            filtered_tickers: list[dict[str, Any]] = [{"symbol": k} for k in pairlist]

            # 获取回溯周期（毫秒），用于交易所K线数据获取
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

            # todo: 起始日期的UTC日期输出
            self.log_once(
                f"使用{self._lookback_period}根K线的成交量范围，时间周期：{self._lookback_timeframe}，起始时间为{format_ms_time(since_ms)}，结束时间为{format_ms_time(to_ms)}",
                logger.info,
            )
            needed_pairs: ListPairsWithTimeframes = [
                (p, self._lookback_timeframe, self._def_candletype)
                for p in [s["symbol"] for s in filtered_tickers]
                if p not in self._pair_cache
            ]

            candles = self._exchange.refresh_ohlcv_with_cache(needed_pairs, since_ms)

            for i, p in enumerate(filtered_tickers):
                contract_size = self._exchange.markets[p["symbol"]].get("contractSize", 1.0) or 1.0
                pair_candles = (
                    candles[(p["symbol"], self._lookback_timeframe, self._def_candletype)]
                    if (p["symbol"], self._lookback_timeframe, self._def_candletype) in candles
                    else None
                )
                # 如果有K线数据，计算典型价格和成交量
                if pair_candles is not None and not pair_candles.empty:
                    if self._exchange.get_option("ohlcv_volume_currency") == "base":
                        pair_candles["typical_price"] = (
                            pair_candles["high"] + pair_candles["low"] + pair_candles["close"]
                        ) / 3

                        pair_candles["quoteVolume"] = (
                            pair_candles["volume"] * pair_candles["typical_price"] * contract_size
                        )
                    else:
                        # 交易所K线数据已包含成交量（以计价货币计）
                        pair_candles["quoteVolume"] = pair_candles["volume"]
                    # 确保对lookback_period周期内的成交量进行滚动求和
                    # 如果pair_candles包含的K线数超过lookback_period，取最后一个周期的滚动和
                    quoteVolume = (
                        pair_candles["quoteVolume"]
                        .rolling(self._lookback_period)
                        .sum()
                        .fillna(0)
                        .iloc[-1]
                    )

                    # 用上面计算的周期成交量总和替换quoteVolume
                    filtered_tickers[i]["quoteVolume"] = quoteVolume
                else:
                    filtered_tickers[i]["quoteVolume"] = 0
        else:
            # 行情数据模式 - 基于传入的交易对列表进行过滤。
            filtered_tickers = [v for k, v in tickers.items() if k in pairlist]

        if self._min_value > 0:
            filtered_tickers = [v for v in filtered_tickers if v[self._sort_key] > self._min_value]
        if self._max_value is not None:
            filtered_tickers = [v for v in filtered_tickers if v[self._sort_key] < self._max_value]

        sorted_tickers = sorted(filtered_tickers, reverse=True, key=lambda t: t[self._sort_key])

        # 验证白名单仅包含活跃市场的交易对
        pairs = self._whitelist_for_active_markets([s["symbol"] for s in sorted_tickers])
        pairs = self.verify_blacklist(pairs, logmethod=logger.info)
        # 将交易对列表限制为请求的数量
        pairs = pairs[: self._number_pairs]

        return pairs