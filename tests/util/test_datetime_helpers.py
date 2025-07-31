from datetime import datetime, timedelta, timezone

import pytest
import time_machine

from freqtrade.util import (
    dt_floor_day,
    dt_from_ts,
    dt_now,
    dt_ts,
    dt_ts_def,
    dt_ts_none,
    dt_utc,
    format_date,
    format_ms_time,
    format_ms_time_det,
    shorten_date,
)
from freqtrade.util.datetime_helpers import dt_humanize_delta


def test_dt_now():
    """测试获取当前UTC时间的函数"""
    with time_machine.travel("2021-09-01 05:01:00 +00:00", tick=False) as t:
        now = datetime.now(timezone.utc)
        # 验证当前时间是否正确
        assert dt_now() == now
        # 验证时间戳转换是否正确
        assert dt_ts() == int(now.timestamp() * 1000)
        assert dt_ts(now) == int(now.timestamp() * 1000)

        # 向前推移5小时
        t.shift(timedelta(hours=5))
        assert dt_now() >= now
        assert dt_now() == datetime.now(timezone.utc)
        assert dt_ts() == int(dt_now().timestamp() * 1000)
        # 验证使用指定时间的时间戳转换
        assert dt_ts(now) == int(now.timestamp() * 1000)


def test_dt_ts_def():
    """测试带默认值的时间戳转换函数"""
    assert dt_ts_def(None) == 0
    assert dt_ts_def(None, 123) == 123
    # 验证特定日期的时间戳
    assert dt_ts_def(datetime(2023, 5, 5, tzinfo=timezone.utc)) == 1683244800000
    assert dt_ts_def(datetime(2023, 5, 5, tzinfo=timezone.utc), 123) == 1683244800000


def test_dt_ts_none():
    """测试时间戳转换为None的情况"""
    assert dt_ts_none(None) is None
    assert dt_ts_none(None) is None
    # 验证特定日期的时间戳
    assert dt_ts_none(datetime(2023, 5, 5, tzinfo=timezone.utc)) == 1683244800000
    assert dt_ts_none(datetime(2023, 5, 5, tzinfo=timezone.utc)) == 1683244800000


def test_dt_utc():
    """测试创建UTC时间的函数"""
    assert dt_utc(2023, 5, 5) == datetime(2023, 5, 5, tzinfo=timezone.utc)
    assert dt_utc(2023, 5, 5, 0, 0, 0, 555500) == datetime(
        2023, 5, 5, 0, 0, 0, 555500, tzinfo=timezone.utc
    )


@pytest.mark.parametrize("as_ms", [True, False])
def test_dt_from_ts(as_ms):
    """测试从时间戳转换为datetime对象"""
    multi = 1000 if as_ms else 1
    # 测试不同格式的时间戳转换
    assert dt_from_ts(1683244800.0 * multi) == datetime(2023, 5, 5, tzinfo=timezone.utc)
    assert dt_from_ts(1683244800.5555 * multi) == datetime(
        2023, 5, 5, 0, 0, 0, 555500, tzinfo=timezone.utc
    )
    # 测试整数时间戳
    assert dt_from_ts(1683244800 * multi) == datetime(2023, 5, 5, tzinfo=timezone.utc)
    # 测试毫秒级时间戳
    assert dt_from_ts(1683244800 * multi) == datetime(2023, 5, 5, tzinfo=timezone.utc)
    assert dt_from_ts(1683242400 * multi) == datetime(2023, 5, 4, 23, 20, tzinfo=timezone.utc)


def test_dt_floor_day():
    """测试将时间向下取整到当天开始的函数"""
    now = datetime(2023, 9, 1, 5, 2, 3, 455555, tzinfo=timezone.utc)
    # 验证取整后的时间是否为当天0点
    assert dt_floor_day(now) == datetime(2023, 9, 1, tzinfo=timezone.utc)


def test_shorten_date() -> None:
    """测试缩短日期字符串的函数"""
    str_data = "1 day, 2 hours, 3 minutes, 4 seconds ago"
    str_shorten_data = "1 d, 2 h, 3 min, 4 sec ago"
    assert shorten_date(str_data) == str_shorten_data


def test_dt_humanize() -> None:
    """测试将时间差转换为人类可读格式的函数"""
    assert dt_humanize_delta(dt_now()) == "now"
    assert dt_humanize_delta(dt_now() - timedelta(minutes=50)) == "50 minutes ago"
    assert dt_humanize_delta(dt_now() - timedelta(hours=16)) == "16 hours ago"
    assert dt_humanize_delta(dt_now() - timedelta(hours=16, minutes=30)) == "16 hours ago"
    assert dt_humanize_delta(dt_now() - timedelta(days=16, hours=10, minutes=25)) == "16 days ago"
    assert dt_humanize_delta(dt_now() - timedelta(minutes=50)) == "50 minutes ago"


def test_format_ms_time() -> None:
    """测试格式化毫秒级时间戳的函数"""
    # 2018-04-10 18:02:01的时间戳
    date_in_epoch_ms = 1523383321132
    date = format_ms_time(date_in_epoch_ms)
    assert isinstance(date, str)
    res = datetime(2018, 4, 10, 18, 2, 1, tzinfo=timezone.utc)
    assert date == res.strftime("%Y-%m-%dT%H:%M:%S")
    assert date == "2018-04-10T18:02:01"
    
    # 2017-12-13 08:02:01的时间戳
    res = datetime(2017, 12, 13, 8, 2, 1, tzinfo=timezone.utc)
    date_in_epoch_ms = 1513152121000
    assert format_ms_time(date_in_epoch_ms) == res.strftime("%Y-%m-%dT%H:%M:%S")


def test_format_date() -> None:
    """测试格式化日期对象的函数"""
    date = datetime(2023, 9, 1, 5, 2, 3, 455555, tzinfo=timezone.utc)
    assert format_date(date) == "2023-09-01 05:02:03"
    assert format_date(None) == ""  # 测试None的情况

    date = datetime(2021, 9, 30, 22, 59, 3, 455555, tzinfo=timezone.utc)
    assert format_date(date) == "2021-09-30 22:59:03"
    assert format_date(None) == ""


def test_format_ms_time_detailed() -> None:
    """测试格式化毫秒级时间戳的详细版本函数"""
    # 2018-04-10 18:02:01.132的时间戳
    date_in_epoch_ms = 1523383321132
    date = format_ms_time_det(date_in_epoch_ms)
    assert isinstance(date, str)
    res = datetime(2018, 4, 10, 18, 2, 1, 132145, tzinfo=timezone.utc)
    # 验证到毫秒级的格式化
    assert date == res.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    assert date == "2018-04-10T18:02:01.132"
    
    # 2017-12-13 08:02:01.512的时间戳
    res = datetime(2017, 12, 13, 8, 2, 1, 512321, tzinfo=timezone.utc)
    date_in_epoch_ms = 1513152121512
    assert format_ms_time_det(date_in_epoch_ms) == res.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]