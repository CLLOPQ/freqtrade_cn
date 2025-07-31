"""
数据提供器（Dataprovider）
负责向机器人提供数据
包括 ticker 和订单簿数据、实时和历史蜡烛图（OHLCV）数据
为机器人和策略提供访问数据的通用接口
"""

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any

from pandas import DataFrame, Timedelta, Timestamp, to_timedelta

from freqtrade.configuration import TimeRange
from freqtrade.constants import (
    FULL_DATAFRAME_THRESHOLD,
    Config,
    ListPairsWithTimeframes,
    PairWithTimeframe,
)
from freqtrade.data.history import get_datahandler, load_pair_history
from freqtrade.enums import CandleType, RPCMessageType, RunMode, TradingMode
from freqtrade.exceptions import ExchangeError, OperationalException
from freqtrade.exchange import Exchange, timeframe_to_prev_date, timeframe_to_seconds
from freqtrade.exchange.exchange_types import OrderBook
from freqtrade.misc import append_candles_to_dataframe
from freqtrade.rpc import RPCManager
from freqtrade.rpc.rpc_types import RPCAnalyzedDFMsg
from freqtrade.util import PeriodicCache


logger = logging.getLogger(__name__)

NO_EXCHANGE_EXCEPTION = "数据提供器无法访问交易所。"
MAX_DATAFRAME_CANDLES = 1000


class DataProvider:
    def __init__(
        self,
        config: Config,
        exchange: Exchange | None,
        pairlists=None,
        rpc: RPCManager | None = None,
    ) -> None:
        self._config = config
        self._exchange = exchange
        self._pairlists = pairlists
        self.__rpc = rpc
        self.__cached_pairs: dict[PairWithTimeframe, tuple[DataFrame, datetime]] = {}
        self.__slice_index: dict[str, int] = {}
        self.__slice_date: datetime | None = None

        self.__cached_pairs_backtesting: dict[PairWithTimeframe, DataFrame] = {}
        self.__producer_pairs_df: dict[
            str, dict[PairWithTimeframe, tuple[DataFrame, datetime]]
        ] = {}
        self.__producer_pairs: dict[str, list[str]] = {}
        self._msg_queue: deque = deque()

        self._default_candle_type = self._config.get("candle_type_def", CandleType.SPOT)
        self._default_timeframe = self._config.get("timeframe", "1h")

        self.__msg_cache = PeriodicCache(
            maxsize=1000, ttl=timeframe_to_seconds(self._default_timeframe)
        )

        self.producers = self._config.get("external_message_consumer", {}).get("producers", [])
        self.external_data_enabled = len(self.producers) > 0

    def _set_dataframe_max_index(self, pair: str, limit_index: int):
        """
        将分析数据框限制为指定的最大索引。
        仅在回测中相关。
        :param limit_index: 数据框索引。
        """
        self.__slice_index[pair] = limit_index

    def _set_dataframe_max_date(self, limit_date: datetime):
        """
        将信息性数据框限制为指定的最大索引。
        仅在回测中相关。
        :param limit_date: "当前日期"
        """
        self.__slice_date = limit_date

    def _set_cached_df(
        self, pair: str, timeframe: str, dataframe: DataFrame, candle_type: CandleType
    ) -> None:
        """
        存储缓存的数据框。
        使用私有方法，因为用户不应使用此方法
        （但该类通过 `self.dp` 暴露给策略）
        :param pair: 要获取数据的交易对
        :param timeframe: 要获取数据的时间框架
        :param dataframe: 分析后的数据框
        :param candle_type: CandleType 枚举值之一（必须与交易模式匹配！）
        """
        pair_key = (pair, timeframe, candle_type)
        self.__cached_pairs[pair_key] = (dataframe, datetime.now(timezone.utc))

    # 对于多个生产者，我们希望合并交易对列表而不是覆盖
    def _set_producer_pairs(self, pairlist: list[str], producer_name: str = "default"):
        """
        设置接收的交易对以供后续使用。

        :param pairlist: 交易对列表
        """
        self.__producer_pairs[producer_name] = pairlist

    def get_producer_pairs(self, producer_name: str = "default") -> list[str]:
        """
        获取从生产者缓存的交易对

        :returns: 交易对列表
        """
        return self.__producer_pairs.get(producer_name, []).copy()

    def _emit_df(self, pair_key: PairWithTimeframe, dataframe: DataFrame, new_candle: bool) -> None:
        """
        将此数据框作为 ANALYZED_DF 消息发送到 RPC

        :param pair_key: PairWithTimeframe 元组
        :param dataframe: 要发送的数据框
        :param new_candle: 这是一个新蜡烛
        """
        if self.__rpc:
            msg: RPCAnalyzedDFMsg = {
                "type": RPCMessageType.ANALYZED_DF,
                "data": {
                    "key": pair_key,
                    "df": dataframe.tail(1),
                    "la": datetime.now(timezone.utc),
                },
            }
            self.__rpc.send_msg(msg)
            if new_candle:
                self.__rpc.send_msg(
                    {
                        "type": RPCMessageType.NEW_CANDLE,
                        "data": pair_key,
                    }
                )

    def _replace_external_df(
        self,
        pair: str,
        dataframe: DataFrame,
        last_analyzed: datetime,
        timeframe: str,
        candle_type: CandleType,
        producer_name: str = "default",
    ) -> None:
        """
        从外部源向此类添加交易对数据。

        :param pair: 要获取数据的交易对
        :param timeframe: 要获取数据的时间框架
        :param candle_type: CandleType 枚举值之一（必须与交易模式匹配！）
        """
        pair_key = (pair, timeframe, candle_type)

        if producer_name not in self.__producer_pairs_df:
            self.__producer_pairs_df[producer_name] = {}

        _last_analyzed = datetime.now(timezone.utc) if not last_analyzed else last_analyzed

        self.__producer_pairs_df[producer_name][pair_key] = (dataframe, _last_analyzed)
        logger.debug(f"已添加来自 {producer_name} 的 {pair_key} 外部数据框。")

    def _add_external_df(
        self,
        pair: str,
        dataframe: DataFrame,
        last_analyzed: datetime,
        timeframe: str,
        candle_type: CandleType,
        producer_name: str = "default",
    ) -> tuple[bool, int]:
        """
        将蜡烛图附加到现有的外部数据框。传入的数据框
        必须至少有 1 根蜡烛。

        :param pair: 要获取数据的交易对
        :param timeframe: 要获取数据的时间框架
        :param candle_type: CandleType 枚举值之一（必须与交易模式匹配！）
        :returns: 如果无法附加蜡烛则返回 False，或返回缺失蜡烛的数量（整数）。
        """
        pair_key = (pair, timeframe, candle_type)

        if dataframe.empty:
            # 传入的数据框必须至少有 1 根蜡烛
            return (False, 0)

        if len(dataframe) >= FULL_DATAFRAME_THRESHOLD:
            # 这可能是一个完整的数据框
            # 将数据框添加到数据提供器
            self._replace_external_df(
                pair,
                dataframe,
                last_analyzed=last_analyzed,
                timeframe=timeframe,
                candle_type=candle_type,
                producer_name=producer_name,
            )
            return (True, 0)

        if (
            producer_name not in self.__producer_pairs_df
            or pair_key not in self.__producer_pairs_df[producer_name]
        ):
            # 我们还没有来自此生产者的数据，
            # 或者我们没有此 pair_key 的数据
            # 返回 False 和 1000 表示需要完整数据框
            return (False, 1000)

        existing_df, _ = self.__producer_pairs_df[producer_name][pair_key]

        # 检查缺失的蜡烛
        # 将时间框架转换为 pandas 的时间增量
        timeframe_delta: Timedelta = to_timedelta(timeframe)
        local_last: Timestamp = existing_df.iloc[-1]["date"]  # 我们需要从副本中获取最后一个日期
        # 我们需要从传入的数据中获取第一个日期
        incoming_first: Timestamp = dataframe.iloc[0]["date"]

        # 移除比传入的第一个蜡烛更新的现有蜡烛
        existing_df1 = existing_df[existing_df["date"] < incoming_first]

        candle_difference = (incoming_first - local_last) / timeframe_delta

        # 如果差值除以时间框架等于 1，则这
        # 是我们想要的蜡烛，并且传入的数据没有缺失任何内容。
        # 如果 candle_difference 大于 1，则意味着
        # 我们在我们的数据和传入的数据之间错过了一些蜡烛
        # 所以返回 False 和 candle_difference。
        if candle_difference > 1:
            return (False, int(candle_difference))
        if existing_df1.empty:
            appended_df = dataframe
        else:
            appended_df = append_candles_to_dataframe(existing_df1, dataframe)

        # 一切正常，我们已附加
        self._replace_external_df(
            pair,
            appended_df,
            last_analyzed=last_analyzed,
            timeframe=timeframe,
            candle_type=candle_type,
            producer_name=producer_name,
        )
        return (True, 0)

    def get_producer_df(
        self,
        pair: str,
        timeframe: str | None = None,
        candle_type: CandleType | None = None,
        producer_name: str = "default",
    ) -> tuple[DataFrame, datetime]:
        """
        从生产者获取交易对数据。

        :param pair: 要获取数据的交易对
        :param timeframe: 要获取数据的时间框架
        :param candle_type: CandleType 枚举值之一（必须与交易模式匹配！）
        :returns: 数据框和最后分析时间戳的元组
        """
        _timeframe = self._default_timeframe if not timeframe else timeframe
        _candle_type = self._default_candle_type if not candle_type else candle_type

        pair_key = (pair, _timeframe, _candle_type)

        # 如果我们还没有来自此生产者的数据
        if producer_name not in self.__producer_pairs_df:
            # 我们还没有此数据，返回空数据框和 datetime (01-01-1970)
            return (DataFrame(), datetime.fromtimestamp(0, tz=timezone.utc))

        # 如果我们有来自该生产者的数据，但没有此 pair_key 的数据
        if pair_key not in self.__producer_pairs_df[producer_name]:
            # 我们还没有此数据，返回空数据框和 datetime (01-01-1970)
            return (DataFrame(), datetime.fromtimestamp(0, tz=timezone.utc))

        # 我们有数据，返回此数据
        df, la = self.__producer_pairs_df[producer_name][pair_key]
        return (df.copy(), la)

    def add_pairlisthandler(self, pairlists) -> None:
        """
        允许在初始化后添加交易对列表处理器
        """
        self._pairlists = pairlists

    def historic_ohlcv(self, pair: str, timeframe: str, candle_type: str = "") -> DataFrame:
        """
        获取存储的历史蜡烛图（OHLCV）数据
        :param pair: 要获取数据的交易对
        :param timeframe: 要获取数据的时间框架
        :param candle_type: '', mark, index, premiumIndex, 或 funding_rate
        """
        _candle_type = (
            CandleType.from_string(candle_type)
            if candle_type != ""
            else self._config["candle_type_def"]
        )
        saved_pair: PairWithTimeframe = (pair, str(timeframe), _candle_type)
        if saved_pair not in self.__cached_pairs_backtesting:
            timerange = TimeRange.parse_timerange(
                None
                if self._config.get("timerange") is None
                else str(self._config.get("timerange"))
            )

            startup_candles = self.get_required_startup(str(timeframe))
            tf_seconds = timeframe_to_seconds(str(timeframe))
            timerange.subtract_start(tf_seconds * startup_candles)

            logger.info(
                f"加载 {pair} {timeframe} 的数据，从 {timerange.start_fmt} 到 {timerange.stop_fmt}"
            )

            self.__cached_pairs_backtesting[saved_pair] = load_pair_history(
                pair=pair,
                timeframe=timeframe,
                datadir=self._config["datadir"],
                timerange=timerange,
                data_format=self._config["dataformat_ohlcv"],
                candle_type=_candle_type,
            )
        return self.__cached_pairs_backtesting[saved_pair].copy()

    def get_required_startup(self, timeframe: str) -> int:
        freqai_config = self._config.get("freqai", {})
        if not freqai_config.get("enabled", False):
            return self._config.get("startup_candle_count", 0)
        else:
            startup_candles = self._config.get("startup_candle_count", 0)
            indicator_periods = freqai_config["feature_parameters"]["indicator_periods_candles"]
            # 确保启动蜡烛数至少是设置的最大指标周期
            self._config["startup_candle_count"] = max(startup_candles, max(indicator_periods))
            tf_seconds = timeframe_to_seconds(timeframe)
            train_candles = freqai_config["train_period_days"] * 86400 / tf_seconds
            total_candles = int(self._config["startup_candle_count"] + train_candles)
            logger.info(
                f"为 freqai 在 {timeframe} 上将 startup_candle_count 增加到 {total_candles}"
            )
        return total_candles

    def get_pair_dataframe(
        self, pair: str, timeframe: str | None = None, candle_type: str = ""
    ) -> DataFrame:
        """
        返回交易对蜡烛图（OHLCV）数据，根据运行模式可以是实时数据或缓存的历史数据。
        只有交易对列表中的组合或已指定为信息性交易对的组合
        才会可用。
        :param pair: 要获取数据的交易对
        :param timeframe: 要获取数据的时间框架
        :return: 此交易对的数据框
        :param candle_type: '', mark, index, premiumIndex, 或 funding_rate
        """
        if self.runmode in (RunMode.DRY_RUN, RunMode.LIVE):
            # 获取实时 OHLCV 数据
            data = self.ohlcv(pair=pair, timeframe=timeframe, candle_type=candle_type)
        else:
            # 获取历史 OHLCV 数据（缓存在磁盘上）
            timeframe = timeframe or self._config["timeframe"]
            data = self.historic_ohlcv(pair=pair, timeframe=timeframe, candle_type=candle_type)
            # 将日期截断到特定时间框架的日期
            # 这对于通过信息性交易对在回调中防止未来数据偏差是必要的
            if self.__slice_date:
                cutoff_date = timeframe_to_prev_date(timeframe, self.__slice_date)
                data = data.loc[data["date"] < cutoff_date]
        if len(data) == 0:
            logger.warning(f"未找到 ({pair}, {timeframe}, {candle_type}) 的数据。")
        return data

    def get_analyzed_dataframe(self, pair: str, timeframe: str) -> tuple[DataFrame, datetime]:
        """
        检索分析后的数据框。在交易模式（实时 / 模拟）下返回完整数据框，
        在所有其他模式下返回最后 1000 根蜡烛（截至此时评估的时间）。
        :param pair: 要获取数据的交易对
        :param timeframe: 要获取数据的时间框架
        :return: 所请求交易对 / 时间框架组合的（分析后的数据框，最后刷新时间）元组。
            如果没有缓存的数据框，则返回空数据框和纪元 0（1970-01-01）。
        """
        pair_key = (pair, timeframe, self._config.get("candle_type_def", CandleType.SPOT))
        if pair_key in self.__cached_pairs:
            if self.runmode in (RunMode.DRY_RUN, RunMode.LIVE):
                df, date = self.__cached_pairs[pair_key]
            else:
                df, date = self.__cached_pairs[pair_key]
                if (max_index := self.__slice_index.get(pair)) is not None:
                                    df = df.iloc[max(0, max_index - MAX_DATAFRAME_CANDLES) : max_index]
                else:
                    return (DataFrame(), datetime.fromtimestamp(0, tz=timezone.utc))
            return df, date
        else:
            return (DataFrame(), datetime.fromtimestamp(0, tz=timezone.utc))

@property
def runmode(self) -> RunMode:
    """
    获取机器人的运行模式
    可以是 "live"、"dry-run"、"backtest"、"hyperopt" 或 "other"。
    """
    return RunMode(self._config.get("runmode", RunMode.OTHER))

def current_whitelist(self) -> list[str]:
    """
    获取最新的可用白名单。

    当您有一个大型白名单并且需要将每个交易对作为信息性交易对调用时非常有用。
    因为可用交易对在信息性交易对被缓存之前不会显示白名单。
    :return: 白名单中的交易对列表
    """

    if self._pairlists:
        return self._pairlists.whitelist.copy()
    else:
        raise OperationalException("数据提供器未使用交易对列表提供器进行初始化。")

def clear_cache(self):
    """
    清除交易对数据框缓存。
    """
    self.__cached_pairs = {}
    # 不要重置回测交易对 -
    # 否则在超参数优化期间，由于 with analyze_per_epoch，它们会每次重新加载
    # self.__cached_pairs_backtesting = {}
    self.__slice_index = {}

# 交易所函数

def refresh(
    self,
    pairlist: ListPairsWithTimeframes,
    helping_pairs: ListPairsWithTimeframes | None = None,
) -> None:
    """
    刷新数据，每个周期调用一次
    """
    if self._exchange is None:
        raise OperationalException(NO_EXCHANGE_EXCEPTION)
    final_pairs = (pairlist + helping_pairs) if helping_pairs else pairlist
    # 刷新最新的 ohlcv 数据
    self._exchange.refresh_latest_ohlcv(final_pairs)
    # 刷新最新的交易数据
    self.refresh_latest_trades(pairlist)

def refresh_latest_trades(self, pairlist: ListPairsWithTimeframes) -> None:
    """
    刷新最新的交易数据（如果在配置中启用）
    """

    use_public_trades = self._config.get("exchange", {}).get("use_public_trades", False)
    if use_public_trades:
        if self._exchange:
            self._exchange.refresh_latest_trades(pairlist)

@property
def available_pairs(self) -> ListPairsWithTimeframes:
    """
    返回包含当前缓存数据的（交易对，时间框架）元组列表。
    应该是白名单 + 未平仓交易。
    """
    if self._exchange is None:
        raise OperationalException(NO_EXCHANGE_EXCEPTION)
    return list(self._exchange._klines.keys())

def ohlcv(
    self, pair: str, timeframe: str | None = None, copy: bool = True, candle_type: str = ""
) -> DataFrame:
    """
    获取给定交易对的蜡烛图（OHLCV）数据作为数据框
    请使用 `available_pairs` 方法验证当前缓存了哪些交易对。
    :param pair: 要获取数据的交易对
    :param timeframe: 要获取数据的时间框架
    :param candle_type: '', mark, index, premiumIndex, 或 funding_rate
    :param copy: 如果为 True，则在返回前复制数据框。
                 仅在只读操作（不修改数据框）时使用 False
    """
    if self._exchange is None:
        raise OperationalException(NO_EXCHANGE_EXCEPTION)
    if self.runmode in (RunMode.DRY_RUN, RunMode.LIVE):
        _candle_type = (
            CandleType.from_string(candle_type)
            if candle_type != ""
            else self._config["candle_type_def"]
        )
        return self._exchange.klines(
            (pair, timeframe or self._config["timeframe"], _candle_type), copy=copy
        )
    else:
        return DataFrame()

def trades(
    self, pair: str, timeframe: str | None = None, copy: bool = True, candle_type: str = ""
) -> DataFrame:
    """
    获取给定交易对的蜡烛图（TRADES）数据作为数据框
    请使用 `available_pairs` 方法验证当前缓存了哪些交易对。
    由于未来数据偏差，不建议在回调中使用。
    :param pair: 要获取数据的交易对
    :param timeframe: 要获取数据的时间框架
    :param candle_type: '', mark, index, premiumIndex, 或 funding_rate
    :param copy: 如果为 True，则在返回前复制数据框。
                 仅在只读操作（不修改数据框）时使用 False
    """
    if self.runmode in (RunMode.DRY_RUN, RunMode.LIVE):
        if self._exchange is None:
            raise OperationalException(NO_EXCHANGE_EXCEPTION)
        _candle_type = (
            CandleType.from_string(candle_type)
            if candle_type != ""
            else self._config["candle_type_def"]
        )
        return self._exchange.trades(
            (pair, timeframe or self._config["timeframe"], _candle_type), copy=copy
        )
    else:
        data_handler = get_datahandler(
            self._config["datadir"], data_format=self._config["dataformat_trades"]
        )
        trades_df = data_handler.trades_load(
            pair, self._config.get("trading_mode", TradingMode.SPOT)
        )
        return trades_df

def market(self, pair: str) -> dict[str, Any] | None:
    """
    返回交易对的市场数据
    :param pair: 要获取数据的交易对
    :return: 来自 ccxt 的市场数据字典，如果该交易对的市场信息不可用则返回 None
    """
    if self._exchange is None:
        raise OperationalException(NO_EXCHANGE_EXCEPTION)
    return self._exchange.markets.get(pair)

def ticker(self, pair: str):
    """
    从交易所返回最新的 ticker 数据
    :param pair: 要获取数据的交易对
    :return: 来自交易所的 ticker 字典，如果该交易对的 ticker 不可用则返回空字典
    """
    if self._exchange is None:
        raise OperationalException(NO_EXCHANGE_EXCEPTION)
    try:
        return self._exchange.fetch_ticker(pair)
    except ExchangeError:
        return {}

def orderbook(self, pair: str, maximum: int) -> OrderBook:
    """
    获取最新的 l2 订单簿数据
    警告：会发起网络请求 - 请合理使用。
    :param pair: 要获取数据的交易对
    :param maximum: 要查询的订单簿条目最大数量
    :return: 包含买单/卖单的字典，总共 `maximum` 个条目。
    """
    if self._exchange is None:
        raise OperationalException(NO_EXCHANGE_EXCEPTION)
    return self._exchange.fetch_l2_order_book(pair, maximum)

def send_msg(self, message: str, *, always_send: bool = False) -> None:
    """
    从您的机器人发送自定义 RPC 通知。
    在模拟或实时模式以外的模式下不会发送任何消息。
    :param message: 要发送的消息。必须少于 4096 个字符。
    :param always_send: 如果为 False，则每个蜡烛只发送一次消息，并抑制
                        相同的消息。
                        请注意，这可能会导致您的聊天收到垃圾信息。
                        默认为 False
    """
    if self.runmode not in (RunMode.DRY_RUN, RunMode.LIVE):
        return

    if always_send or message not in self.__msg_cache:
        self._msg_queue.append(message)
    self.__msg_cache[message] = True
