"""
用于将数据从一种格式转换为另一种格式的函数
"""

import logging

import numpy as np
import pandas as pd
from pandas import DataFrame, to_datetime

from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS, Config
from freqtrade.enums import CandleType, TradingMode


logger = logging.getLogger(__name__)


def ohlcv_to_dataframe(
    ohlcv: list,
    timeframe: str,
    pair: str,
    *,
    fill_missing: bool = True,
    drop_incomplete: bool = True,
) -> DataFrame:
    """
    将包含蜡烛图（OHLCV）数据的列表（以 ccxt.fetch_ohlcv 返回的格式）转换为数据框
    :param ohlcv: 包含蜡烛图（OHLCV）数据的列表，如交易所的 async_get_candle_history 返回的格式
    :param timeframe: 时间框架（例如 5m）。用于填充可能缺失的数据
    :param pair: 此数据对应的交易对（用于在需要填充时发出警告）
    :param fill_missing: 用 0 蜡烛填充缺失的蜡烛
                         （详见 ohlcv_fill_up_missing_data）
    :param drop_incomplete: 丢弃数据框的最后一根蜡烛，假设它是不完整的
    :return: 数据框
    """
    logger.debug(f"正在将蜡烛图（OHLCV）数据转换为 {pair} 的数据框。")
    cols = DEFAULT_DATAFRAME_COLUMNS
    df = DataFrame(ohlcv, columns=cols)

    df["date"] = to_datetime(df["date"], unit="ms", utc=True)

    # 有些交易所返回的成交量甚至 OHLC 是整数。
    # 转换它们，因为策略中使用的 TA-LIB 指标假设是浮点数，否则会抛出异常...
    df = df.astype(
        dtype={
            "open": "float",
            "high": "float",
            "low": "float",
            "close": "float",
            "volume": "float",
        }
    )
    return clean_ohlcv_dataframe(
        df, timeframe, pair, fill_missing=fill_missing, drop_incomplete=drop_incomplete
    )


def clean_ohlcv_dataframe(
    data: DataFrame, timeframe: str, pair: str, *, fill_missing: bool, drop_incomplete: bool
) -> DataFrame:
    """
    通过以下方式清理 OHLCV 数据框：
      * 按日期分组（移除重复的时间点）
      * 如请求的那样丢弃最后一根蜡烛
      * 填充缺失的数据（如请求的那样）
    :param data: 包含蜡烛图（OHLCV）数据的数据框。
    :param timeframe: 时间框架（例如 5m）。用于填充可能缺失的数据
    :param pair: 此数据对应的交易对（用于在需要填充时发出警告）
    :param fill_missing: 用 0 蜡烛填充缺失的蜡烛
                         （详见 ohlcv_fill_up_missing_data）
    :param drop_incomplete: 丢弃数据框的最后一根蜡烛，假设它是不完整的
    :return: 数据框
    """
    # 按索引分组并聚合结果以消除重复的时间点
    data = data.groupby(by="date", as_index=False, sort=True).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "max",
        }
    )
    # 消除部分蜡烛
    if drop_incomplete:
        data.drop(data.tail(1).index, inplace=True)
        logger.debug("正在丢弃最后一根蜡烛")

    if fill_missing:
        return ohlcv_fill_up_missing_data(data, timeframe, pair)
    else:
        return data


def ohlcv_fill_up_missing_data(dataframe: DataFrame, timeframe: str, pair: str) -> DataFrame:
    """
    用零成交量的行填充缺失的数据，
    使用前一根的收盘价作为 "开盘价"、"最高价"、"最低价" 和 "收盘价" 的价格，成交量设为 0

    """
    from freqtrade.exchange import timeframe_to_resample_freq

    ohlcv_dict = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    resample_interval = timeframe_to_resample_freq(timeframe)
    # 重采样以创建 "NAN" 值
    df = dataframe.resample(resample_interval, on="date").agg(ohlcv_dict)

    # 向前填充缺失列的收盘价
    df["close"] = df["close"].ffill()
    # 使用收盘价填充 "开盘价、最高价、最低价"
    df.loc[:, ["open", "high", "low"]] = df[["open", "high", "low"]].fillna(
        value={
            "open": df["close"],
            "high": df["close"],
            "low": df["close"],
        }
    )
    df.reset_index(inplace=True)
    填充前长度 = len(dataframe)
    填充后长度 = len(df)
    缺失百分比 = (填充后长度 - 填充前长度) / 填充前长度 if 填充前长度 > 0 else 0
    if 填充前长度 != 填充后长度:
        消息 = (
            f"{pair} 的缺失数据填充，{timeframe}："
            f"填充前：{填充前长度} - 填充后：{填充后长度} - {缺失百分比:.2%}"
        )
        if 缺失百分比 > 0.01:
            logger.info(消息)
        else:
            # 如果只有少量缺失，不详细输出
            logger.debug(消息)
    return df


def trim_dataframe(
    df: DataFrame, timerange, *, df_date_col: str = "date", startup_candles: int = 0
) -> DataFrame:
    """
    根据给定的时间范围修剪数据框
    :param df: 要修剪的数据框
    :param timerange: 时间范围（如果可用，使用开始和结束日期）
    :param df_date_col: 数据框中用作日期列的列
    :param startup_candles: 非零时，用于替代时间范围的开始日期
    :return: 修剪后的数据框
    """
    if startup_candles:
        # 在给定启动蜡烛数量的情况下修剪蜡烛而不是时间范围
        df = df.iloc[startup_candles:, :]
    else:
        if timerange.starttype == "date":
            df = df.loc[df[df_date_col] >= timerange.startdt, :]
    if timerange.stoptype == "date":
        df = df.loc[df[df_date_col] <= timerange.stopdt, :]
    return df


def trim_dataframes(
    preprocessed: dict[str, DataFrame], timerange, startup_candles: int
) -> dict[str, DataFrame]:
    """
    从分析后的数据框中修剪启动期
    :param preprocessed: 交易对:数据框的字典
    :param timerange: 时间范围（如果可用，使用开始和结束日期）
    :param startup_candles: 应移除的启动蜡烛数量
    :return: 修剪后的数据框字典
    """
    processed: dict[str, DataFrame] = {}

    for pair, df in preprocessed.items():
        修剪后的_df = trim_dataframe(df, timerange, startup_candles=startup_candles)
        if not 修剪后的_df.empty:
            processed[pair] = 修剪后的_df
        else:
            logger.warning(
                f"{pair} 在调整启动蜡烛后没有剩余数据，将跳过。"
            )
    return processed


def order_book_to_dataframe(bids: list, asks: list) -> DataFrame:
    """
    TODO: 这应该有专门的测试
    获取订单簿列表，返回按 creslin 建议的格式的数据框
    -------------------------------------------------------------------
     b_sum       b_size       bids       asks       a_size       a_sum
    -------------------------------------------------------------------
    """
    cols = ["bids", "b_size"]

    bids_frame = DataFrame(bids, columns=cols)
    # 添加累积和列
    bids_frame["b_sum"] = bids_frame["b_size"].cumsum()
    cols2 = ["asks", "a_size"]
    asks_frame = DataFrame(asks, columns=cols2)
    # 添加累积和列
    asks_frame["a_sum"] = asks_frame["a_size"].cumsum()

    frame = pd.concat(
        [
            bids_frame["b_sum"],
            bids_frame["b_size"],
            bids_frame["bids"],
            asks_frame["asks"],
            asks_frame["a_size"],
            asks_frame["a_sum"],
        ],
        axis=1,
        keys=["b_sum", "b_size", "bids", "asks", "a_size", "a_sum"],
    )
    # logger.info('order book %s', frame )
    return frame


def convert_ohlcv_format(
    config: Config,
    convert_from: str,
    convert_to: str,
    erase: bool,
):
    """
    将 OHLCV 从一种格式转换为另一种格式
    :param config: 配置字典
    :param convert_from: 源格式
    :param convert_to: 目标格式
    :param erase: 擦除源数据（如果源格式和目标格式相同则不适用）
    """
    from freqtrade.data.history import get_datahandler

    src = get_datahandler(config["datadir"], convert_from)
    trg = get_datahandler(config["datadir"], convert_to)
    timeframes = config.get("timeframes", [config.get("timeframe")])
    logger.info(f"正在转换 {timeframes} 时间框架的蜡烛图（OHLCV）")

    candle_types = [
        CandleType.from_string(ct)
        for ct in config.get("candle_types", [c.value for c in CandleType])
    ]
    logger.info(candle_types)
    paircombs = src.ohlcv_get_available_data(config["datadir"], TradingMode.SPOT)
    paircombs.extend(src.ohlcv_get_available_data(config["datadir"], TradingMode.FUTURES))

    if "pairs" in config:
        # 过滤交易对
        paircombs = [comb for comb in paircombs if comb[0] in config["pairs"]]

    if "timeframes" in config:
        paircombs = [comb for comb in paircombs if comb[1] in config["timeframes"]]
    paircombs = [comb for comb in paircombs if comb[2] in candle_types]

    paircombs = sorted(paircombs, key=lambda x: (x[0], x[1], x[2].value))

    formatted_paircombs = "\n".join(
        [f"{pair}, {timeframe}, {candle_type}" for pair, timeframe, candle_type in paircombs]
    )

    logger.info(
        f"正在转换以下交易对组合的蜡烛图（OHLCV）数据：\n"
        f"{formatted_paircombs}"
    )
    for pair, timeframe, candle_type in paircombs:
        data = src.ohlcv_load(
            pair=pair,
            timeframe=timeframe,
            timerange=None,
            fill_missing=False,
            drop_incomplete=False,
            startup_candles=0,
            candle_type=candle_type,
        )
        logger.info(f"正在转换 {pair} 的 {len(data)} 根 {timeframe} {candle_type} 蜡烛")
        if len(data) > 0:
            trg.ohlcv_store(pair=pair, timeframe=timeframe, data=data, candle_type=candle_type)
            if erase and convert_from != convert_to:
                logger.info(f"正在删除 {pair} / {timeframe} 的源数据")
                src.ohlcv_purge(pair=pair, timeframe=timeframe, candle_type=candle_type)


def reduce_dataframe_footprint(df: DataFrame) -> DataFrame:
    """
    确保传入数据框中的所有值都是 float32。
    :param df: 要转换为 float/int 32s 的数据框
    :return: 转换为 float/int 32s 的数据框
    """

    logger.debug(f"数据框的内存使用量为 {df.memory_usage().sum() / 1024**2:.2f} MB")

    df_dtypes = df.dtypes
    for column, dtype in df_dtypes.items():
        if column in ["open", "high", "low", "close", "volume"]:
            continue
        if dtype == np.float64:
            df_dtypes[column] = np.float32
        elif dtype == np.int64:
            df_dtypes[column] = np.int32
    df = df.astype(df_dtypes)

    logger.debug(f"优化后的内存使用量为：{df.memory_usage().sum() / 1024**2:.2f} MB")

    return df