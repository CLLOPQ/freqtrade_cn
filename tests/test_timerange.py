# pragma pylint: disable=missing-docstring, C0103
from datetime import datetime, timezone

import pytest

from freqtrade.configuration import TimeRange
from freqtrade.exceptions import OperationalException


def test_parse_timerange_incorrect():
    """测试时间范围解析功能，包括各种格式和边界情况"""
    # 空字符串解析
    timerange = TimeRange.parse_timerange("")
    assert timerange == TimeRange(None, None, 0, 0)
    
    # 仅开始日期
    timerange = TimeRange.parse_timerange("20100522-")
    assert TimeRange("date", None, 1274486400, 0) == timerange
    assert timerange.timerange_str == "20100522-"
    
    # 仅结束日期
    timerange = TimeRange.parse_timerange("-20100522")
    assert TimeRange(None, "date", 0, 1274486400) == timerange
    assert timerange.timerange_str == "-20100522"
    
    # 完整日期范围
    timerange = TimeRange.parse_timerange("20100522-20150730")
    assert timerange == TimeRange("date", "date", 1274486400, 1438214400)
    assert timerange.timerange_str == "20100522-20150730"
    assert timerange.start_fmt == "2010-05-22 00:00:00"
    assert timerange.stop_fmt == "2015-07-30 00:00:00"

    # 测试Unix时间戳 - 比特币创世日期
    assert TimeRange("date", None, 1231006505, 0) == TimeRange.parse_timerange("1231006505-")
    assert TimeRange(None, "date", 0, 1233360000) == TimeRange.parse_timerange("-1233360000")
    
    # 完整Unix时间戳范围
    timerange = TimeRange.parse_timerange("1231006505-1233360000")
    assert TimeRange("date", "date", 1231006505, 1233360000) == timerange
    assert isinstance(timerange.startdt, datetime)
    assert isinstance(timerange.stopdt, datetime)
    assert timerange.startdt == datetime.fromtimestamp(1231006505, tz=timezone.utc)
    assert timerange.stopdt == datetime.fromtimestamp(1233360000, tz=timezone.utc)
    assert timerange.timerange_str == "20090103-20090131"

    # 毫秒级Unix时间戳
    timerange = TimeRange.parse_timerange("1231006505000-1233360000000")
    assert TimeRange("date", "date", 1231006505, 1233360000) == timerange

    timerange = TimeRange.parse_timerange("1231006505000-")
    assert TimeRange("date", None, 1231006505, 0) == timerange

    timerange = TimeRange.parse_timerange("-1231006505000")
    assert TimeRange(None, "date", 0, 1231006505) == timerange

    # 测试错误格式
    with pytest.raises(OperationalException, match=r"Incorrect syntax.*"):
        TimeRange.parse_timerange("-")

    # 测试开始日期晚于结束日期的情况
    with pytest.raises(
        OperationalException, match=r"Start date is after stop date for timerange.*"
    ):
        TimeRange.parse_timerange("20100523-20100522")


def test_subtract_start():
    """测试从开始时间减去指定秒数的功能"""
    # 正常情况：有开始和结束时间
    x = TimeRange("date", "date", 1274486400, 1438214400)
    x.subtract_start(300)
    assert x.startts == 1274486400 - 300

    # 边界情况：没有开始时间
    x = TimeRange(None, "date", 0, 1438214400)
    x.subtract_start(300)
    assert not x.startts
    assert not x.startdt

    # 边界情况：没有结束时间
    x = TimeRange("date", None, 1274486400, 0)
    x.subtract_start(300)
    assert x.startts == 1274486400 - 300


def test_adjust_start_if_necessary():
    """测试根据需要调整开始时间的功能"""
    min_date = datetime(2017, 11, 14, 21, 15, 00, tzinfo=timezone.utc)
    # 5分钟 = 300秒

    # 情况1：调整开始时间增加20个蜡烛的时间
    x = TimeRange("date", "date", 1510694100, 1510780500)
    x.adjust_start_if_necessary(300, 20, min_date)
    assert x.startts == 1510694100 + (20 * 300)

    # 情况2：已经调整过，不再改变
    x = TimeRange("date", "date", 1510700100, 1510780500)
    x.adjust_start_if_necessary(300, 20, min_date)
    assert x.startts == 1510694100 + (20 * 300)

    # 情况3：没有开始时间，根据结束时间和蜡烛数量计算开始时间
    x = TimeRange(None, "date", 0, 1510780500)
    x.adjust_start_if_necessary(300, 20, min_date)
    assert x.startts == 1510694100 + (20 * 300)