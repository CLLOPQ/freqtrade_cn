"""
用于从公开交易中转换订单流数据的函数
"""

import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd

from freqtrade.constants import DEFAULT_ORDERFLOW_COLUMNS, ORDERFLOW_ADDED_COLUMNS, Config
from freqtrade.exceptions import DependencyException


logger = logging.getLogger(__name__)


def _init_dataframe_with_trades_columns(dataframe: pd.DataFrame):
    """
    为数据框填充交易相关列
    :param dataframe: 要填充的数据框
    """
    # 用适当的数据类型初始化列
    for column in ORDERFLOW_ADDED_COLUMNS:
        dataframe[column] = np.nan

    # 将特定列设置为对象类型
    for column in (
        "trades",
        "orderflow",
        "imbalances",
        "stacked_imbalances_bid",
        "stacked_imbalances_ask",
    ):
        dataframe[column] = dataframe[column].astype(object)


def timeframe_to_DateOffset(timeframe: str) -> pd.DateOffset:
    """
    将人类可读形式（'1m'、'5m'、'1h'、'1d'、'1w' 等）表示的时间框架间隔值转换为
    一个时间框架间隔的秒数。
    """
    from freqtrade.exchange import timeframe_to_seconds

    timeframe_seconds = timeframe_to_seconds(timeframe)
    timeframe_minutes = timeframe_seconds // 60
    if timeframe_minutes < 1:
        return pd.DateOffset(seconds=timeframe_seconds)
    elif 59 < timeframe_minutes < 1440:
        return pd.DateOffset(hours=timeframe_minutes // 60)
    elif 1440 <= timeframe_minutes < 10080:
        return pd.DateOffset(days=timeframe_minutes // 1440)
    elif 10000 < timeframe_minutes < 43200:
        return pd.DateOffset(weeks=1)
    elif timeframe_minutes >= 43200 and timeframe_minutes < 525600:
        return pd.DateOffset(months=1)
    elif timeframe == "1y":
        return pd.DateOffset(years=1)
    else:
        return pd.DateOffset(minutes=timeframe_minutes)


def _calculate_ohlcv_candle_start_and_end(df: pd.DataFrame, timeframe: str):
    from freqtrade.exchange import timeframe_to_resample_freq

    if df is not None and not df.empty:
        timeframe_frequency = timeframe_to_resample_freq(timeframe)
        dofs = timeframe_to_DateOffset(timeframe)
        # 计算 OHLCV 蜡烛的开始和结束时间
        df["datetime"] = pd.to_datetime(df["date"], unit="ms")
        df["candle_start"] = df["datetime"].dt.floor(timeframe_frequency)
        # 在 _now_is_time_to_refresh_trades 中使用
        df["candle_end"] = df["candle_start"] + dofs
        df.drop(columns=["datetime"], inplace=True)


def populate_dataframe_with_trades(
    cached_grouped_trades: pd.DataFrame | None,
    config: Config,
    dataframe: pd.DataFrame,
    trades: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    为数据框填充交易数据
    :param dataframe: 要填充的数据框
    :param trades: 用于填充的交易数据
    :return: 填充了交易数据的数据框
    """

    timeframe = config["timeframe"]
    config_orderflow = config["orderflow"]

    # 创建交易相关列
    _init_dataframe_with_trades_columns(dataframe)
    if trades is None or trades.empty:
        return dataframe, cached_grouped_trades

    try:
        start_time = time.time()
        # 计算 OHLCV 蜡烛的开始和结束时间
        _calculate_ohlcv_candle_start_and_end(trades, timeframe)

        # 获取最早的 max_candles 蜡烛的日期
        max_candles = config_orderflow["max_candles"]
        start_date = dataframe.tail(max_candles).date.iat[0]
        # 截取当前 OHLCV 蜡烛之前的交易，使分组操作更快
        trades = trades.loc[trades["candle_start"] >= start_date]
        trades.reset_index(inplace=True, drop=True)

        # 按蜡烛开始时间分组交易
        trades_grouped_by_candle_start = trades.groupby("candle_start", group_keys=False)

        candle_start: datetime
        for candle_start, trades_grouped_df in trades_grouped_by_candle_start:
            is_between = candle_start == dataframe["date"]
            if is_between.any():
                # 同一日期只能有一行
                index = dataframe.index[is_between][0]

                if (
                    cached_grouped_trades is not None
                    and (candle_start == cached_grouped_trades["date"]).any()
                ):
                    # 检查交易是否已在缓存中
                    cache_idx = cached_grouped_trades.index[
                        cached_grouped_trades["date"] == candle_start
                    ][0]
                    for col in ORDERFLOW_ADDED_COLUMNS:
                        dataframe.at[index, col] = cached_grouped_trades.at[cache_idx, col]
                    continue

                dataframe.at[index, "trades"] = trades_grouped_df.drop(
                    columns=["candle_start", "candle_end"]
                ).to_dict(orient="records")

                # 计算每个蜡烛的订单流
                orderflow = trades_to_volumeprofile_with_total_delta_bid_ask(
                    trades_grouped_df, scale=config_orderflow["scale"]
                )
                dataframe.at[index, "orderflow"] = orderflow.to_dict(orient="index")
                # orderflow_series.loc[[index]] = [orderflow.to_dict(orient="index")]
                # 计算每个蜡烛订单流的不平衡
                imbalances = trades_orderflow_to_imbalances(
                    orderflow,
                    imbalance_ratio=config_orderflow["imbalance_ratio"],
                    imbalance_volume=config_orderflow["imbalance_volume"],
                )
                dataframe.at[index, "imbalances"] = imbalances.to_dict(orient="index")

                stacked_imbalance_range = config_orderflow["stacked_imbalance_range"]
                dataframe.at[index, "stacked_imbalances_bid"] = stacked_imbalance(
                    imbalances, label="bid", stacked_imbalance_range=stacked_imbalance_range
                )

                dataframe.at[index, "stacked_imbalances_ask"] = stacked_imbalance(
                    imbalances, label="ask", stacked_imbalance_range=stacked_imbalance_range
                )

                bid = np.where(
                    trades_grouped_df["side"].str.contains("sell"), trades_grouped_df["amount"], 0
                )

                ask = np.where(
                    trades_grouped_df["side"].str.contains("buy"), trades_grouped_df["amount"], 0
                )
                deltas_per_trade = ask - bid
                dataframe.at[index, "max_delta"] = deltas_per_trade.cumsum().max()
                dataframe.at[index, "min_delta"] = deltas_per_trade.cumsum().min()

                dataframe.at[index, "bid"] = bid.sum()
                dataframe.at[index, "ask"] = ask.sum()
                dataframe.at[index, "delta"] = (
                    dataframe.at[index, "ask"] - dataframe.at[index, "bid"]
                )
                dataframe.at[index, "total_trades"] = len(trades_grouped_df)

        logger.debug(f"trades.groups_keys 耗时 {time.time() - start_time} 秒")

        # 缓存整个数据框
        cached_grouped_trades = dataframe.tail(config_orderflow["cache_size"]).copy()

    except Exception as e:
        logger.exception("填充交易数据到数据框时出错")
        raise DependencyException(e)

    return dataframe, cached_grouped_trades


def trades_to_volumeprofile_with_total_delta_bid_ask(
    trades: pd.DataFrame, scale: float
) -> pd.DataFrame:
    """
    :param trades: 数据框
    :param scale: 尺度，即区间大小，例如 0.5
    :return: 按尺度分箱到不同级别的交易，也称为订单流
    """
    df = pd.DataFrame([], columns=DEFAULT_ORDERFLOW_COLUMNS)
    # 创建买单、卖单列，根据交易方向判断
    df["bid_amount"] = np.where(trades["side"].str.contains("sell"), trades["amount"], 0)
    df["ask_amount"] = np.where(trades["side"].str.contains("buy"), trades["amount"], 0)
    df["bid"] = np.where(trades["side"].str.contains("sell"), 1, 0)
    df["ask"] = np.where(trades["side"].str.contains("buy"), 1, 0)
    # 将价格四舍五入到尺度的最接近倍数
    df["price"] = ((trades["price"] / scale).round() * scale).astype("float64").values
    if df.empty:
        df["total"] = np.nan
        df["delta"] = np.nan
        return df

    df["delta"] = df["ask_amount"] - df["bid_amount"]
    df["total_volume"] = df["ask_amount"] + df["bid_amount"]
    df["total_trades"] = df["ask"] + df["bid"]

    # 分组到区间，即应用尺度
    df = df.groupby("price").sum(numeric_only=True)
    return df


def trades_orderflow_to_imbalances(df: pd.DataFrame, imbalance_ratio: int, imbalance_volume: int):
    """
    :param df: 包含买单和卖单的数据框
    :param imbalance_ratio: 不平衡比率，例如 3
    :param imbalance_volume: 不平衡成交量，例如 10
    :return: 包含买单和卖单不平衡的数据框
    """
    bid = df.bid
    # 对角线比较买单和卖单
    ask = df.ask.shift(-1)
    bid_imbalance = (bid / ask) > (imbalance_ratio)
    # 如果成交量不够大，将买单不平衡设为 False
    bid_imbalance_filtered = np.where(df.total_volume < imbalance_volume, False, bid_imbalance)
    ask_imbalance = (ask / bid) > (imbalance_ratio)
    # 如果成交量不够大，将卖单不平衡设为 False
    ask_imbalance_filtered = np.where(df.total_volume < imbalance_volume, False, ask_imbalance)
    dataframe = pd.DataFrame(
        {"bid_imbalance": bid_imbalance_filtered, "ask_imbalance": ask_imbalance_filtered},
        index=df.index,
    )

    return dataframe


def stacked_imbalance(df: pd.DataFrame, label: str, stacked_imbalance_range: int):
    """
    y * (y.groupby((y != y.shift()).cumsum()).cumcount() + 1)
    https://stackoverflow.com/questions/27626542/counting-consecutive-positive-values-in-python-pandas-array
    """
    imbalance = df[f"{label}_imbalance"]
    int_series = pd.Series(np.where(imbalance, 1, 0))
    # 对连续的 True 值分组并获取其计数
    groups = (int_series != int_series.shift()).cumsum()
    counts = int_series.groupby(groups).cumsum()

    # 找到计数达到或超过范围要求的索引
    valid_indices = counts[counts >= stacked_imbalance_range].index

    stacked_imbalance_prices = []
    if not valid_indices.empty:
        # 从范围的开始获取所有有效索引的价格
        stacked_imbalance_prices = [
            imbalance.index.values[idx - (stacked_imbalance_range - 1)] for idx in valid_indices
        ]
    return stacked_imbalance_prices