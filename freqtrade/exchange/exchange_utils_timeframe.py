from datetime import datetime, timezone

import ccxt
from ccxt import ROUND_DOWN, ROUND_UP

from freqtrade.util.datetime_helpers import dt_from_ts, dt_ts


def timeframe_to_seconds(timeframe: str) -> int:
    """
    将人类可读形式的时间框架间隔值（'1m'、'5m'、'1h'、'1d'、'1w'等）
    转换为一个时间框架间隔的秒数。
    """
    return ccxt.Exchange.parse_timeframe(timeframe)


def timeframe_to_minutes(timeframe: str) -> int:
    """
    与timeframe_to_seconds相同，但返回分钟数。
    """
    return ccxt.Exchange.parse_timeframe(timeframe) // 60


def timeframe_to_msecs(timeframe: str) -> int:
    """
    与timeframe_to_seconds相同，但返回毫秒数。
    """
    return ccxt.Exchange.parse_timeframe(timeframe) * 1000


def timeframe_to_resample_freq(timeframe: str) -> str:
    """
    将人类可读形式的时间框架间隔值（'1m'、'5m'、'1h'、'1d'、'1w'等）
    转换为pandas使用的重采样频率（'1T'、'5T'、'1H'、'1D'、'1W'等）。
    """
    if timeframe == "1y":
        return "1YS"
    timeframe_seconds = timeframe_to_seconds(timeframe)
    timeframe_minutes = timeframe_seconds // 60
    resample_interval = f"{timeframe_seconds}s"
    if 10000 < timeframe_minutes < 43200:
        resample_interval = "1W-MON"
    elif timeframe_minutes >= 43200 and timeframe_minutes < 525600:
        # 月度蜡烛需要特殊处理以固定在每月1日
        resample_interval = f"{timeframe}S"
    elif timeframe_minutes > 43200:
        resample_interval = timeframe
    return resample_interval


def timeframe_to_prev_date(timeframe: str, date: datetime | None = None) -> datetime:
    """
    使用时间框架并确定该日期对应的蜡烛图开始日期。
    当给定蜡烛图开始日期时不进行四舍五入。
    :param timeframe: 字符串格式的时间框架（例如"5m"）
    :param date: 要使用的日期。默认为当前时间(utc)
    :returns: 前一根蜡烛图的日期（带utc时区）
    """
    if not date:
        date = datetime.now(timezone.utc)

    new_timestamp = ccxt.Exchange.round_timeframe(timeframe, dt_ts(date), ROUND_DOWN) // 1000
    return dt_from_ts(new_timestamp)


def timeframe_to_next_date(timeframe: str, date: datetime | None = None) -> datetime:
    """
    使用时间框架并确定下一根蜡烛图的日期。
    :param timeframe: 字符串格式的时间框架（例如"5m"）
    :param date: 要使用的日期。默认为当前时间(utc)
    :returns: 下一根蜡烛图的日期（带utc时区）
    """
    if not date:
        date = datetime.now(timezone.utc)
    new_timestamp = ccxt.Exchange.round_timeframe(timeframe, dt_ts(date), ROUND_UP) // 1000
    return dt_from_ts(new_timestamp)