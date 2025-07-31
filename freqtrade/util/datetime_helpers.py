import re
from datetime import datetime, timezone
from time import time

import humanize

from freqtrade.constants import DATETIME_PRINT_FORMAT


def dt_now() -> datetime:
    """返回当前UTC时间。"""
    return datetime.now(timezone.utc)


def dt_utc(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    microsecond: int = 0,
) -> datetime:
    """返回UTC时间的datetime对象。"""
    return datetime(year, month, day, hour, minute, second, microsecond, tzinfo=timezone.utc)


def dt_ts(dt: datetime | None = None) -> int:
    """
    以毫秒为单位返回dt的UTC时间戳。
    如果dt为None，则返回当前UTC时间。
    """
    if dt:
        return int(dt.timestamp() * 1000)
    return int(time() * 1000)


def dt_ts_def(dt: datetime | None, default: int = 0) -> int:
    """
    以毫秒为单位返回dt的UTC时间戳。
    如果dt为None，则返回给定的默认值。
    """
    if dt:
        return int(dt.timestamp() * 1000)
    return default


def dt_ts_none(dt: datetime | None) -> int | None:
    """
    以毫秒为单位返回dt的UTC时间戳。
    如果dt为None，则返回None。
    """
    if dt:
        return int(dt.timestamp() * 1000)
    return None


def dt_floor_day(dt: datetime) -> datetime:
    """返回给定datetime的当天起始时间（即日期的零点）。"""
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def dt_from_ts(timestamp: float) -> datetime:
    """
    从时间戳返回datetime对象。
    :param timestamp: 以秒或毫秒为单位的时间戳
    """
    if timestamp > 1e10:
        # 时间戳为毫秒级 - 转换为秒
        timestamp /= 1000
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def shorten_date(_date: str) -> str:
    """
    修剪日期字符串，使其适合在小屏幕上显示
    """
    new_date = re.sub("seconds?", "sec", _date)
    new_date = re.sub("minutes?", "min", new_date)
    new_date = re.sub("hours?", "h", new_date)
    new_date = re.sub("days?", "d", new_date)
    new_date = re.sub("^an?", "1", new_date)
    return new_date


def dt_humanize_delta(dt: datetime):
    """
    为给定的时间差返回一个人性化的字符串。
    """
    return humanize.naturaltime(dt)


def format_date(date: datetime | None) -> str:
    """
    返回格式化的日期字符串。
    如果date为None，则返回空字符串。
    :param date: 要格式化的datetime对象
    """
    if date:
        return date.strftime(DATETIME_PRINT_FORMAT)
    return ""


def format_ms_time(date: int | float) -> str:
    """
    将毫秒级日期转换为可读格式。
    : 以毫秒为单位的时间戳字符串
    """
    return dt_from_ts(date).strftime("%Y-%m-%dT%H:%M:%S")


def format_ms_time_det(date: int | float) -> str:
    """
    将毫秒级日期转换为可读格式（详细版）。
    : 以毫秒为单位的时间戳字符串
    """
    # return dt_from_ts(date).isoformat(timespec="milliseconds")
    return dt_from_ts(date).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]