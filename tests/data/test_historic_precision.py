# pragma pylint: disable=missing-docstring, C0103

from datetime import timezone

import pandas as pd
from numpy import nan
from pandas import DataFrame, Timestamp

from freqtrade.data.btanalysis.historic_precision import get_tick_size_over_time


def test_get_tick_size_over_time():
    """
    使用预定义数据测试get_tick_size_over_time函数
    """
    # 创建具有不同精度级别的测试数据框
    data = {
        "date": [
            Timestamp("2020-01-01 00:00:00", tz=timezone.utc),
            Timestamp("2020-01-02 00:00:00", tz=timezone.utc),
            Timestamp("2020-01-03 00:00:00", tz=timezone.utc),
            Timestamp("2020-01-15 00:00:00", tz=timezone.utc),
            Timestamp("2020-01-16 00:00:00", tz=timezone.utc),
            Timestamp("2020-01-31 00:00:00", tz=timezone.utc),
            Timestamp("2020-02-01 00:00:00", tz=timezone.utc),
            Timestamp("2020-02-15 00:00:00", tz=timezone.utc),
            Timestamp("2020-03-15 00:00:00", tz=timezone.utc),
        ],
        "open": [1.23456, 1.234, 1.23, 1.2, 1.23456, 1.234, 2.3456, 2.34, 2.34],
        "high": [1.23457, 1.235, 1.24, 1.3, 1.23456, 1.235, 2.3457, 2.34, 2.34],
        "low": [1.23455, 1.233, 1.22, 1.1, 1.23456, 1.233, 2.3455, 2.34, 2.34],
        "close": [1.23456, 1.234, 1.23, 1.2, 1.23456, 1.234, 2.3456, 2.34, 2.34],
        "volume": [100, 200, 300, 400, 500, 600, 700, 800, 900],
    }

    candles = DataFrame(data)

    # 计算有效数字
    result = get_tick_size_over_time(candles)

    # 检查结果是否为pandas Series
    assert isinstance(result, pd.Series)

    # 检查我们有三个月的数据（2020年1月、2月和3月）
    assert len(result) == 3

    # 之前
    assert result.asof("2019-01-01 00:00:00+00:00") is nan
    # 1月应该有5位有效数字（基于1.23456789是最精确的值）
    # 应该转换为0.00001

    assert result.asof("2020-01-01 00:00:00+00:00") == 0.00001
    assert result.asof("2020-01-01 00:00:00+00:00") == 0.00001
    assert result.asof("2020-02-25 00:00:00+00:00") == 0.0001
    assert result.asof("2020-03-25 00:00:00+00:00") == 0.01
    assert result.asof("2020-04-01 00:00:00+00:00") == 0.01
    # 远超过最后日期的值应该是最后一个值
    assert result.asof("2025-04-01 00:00:00+00:00") == 0.01

    assert result.iloc[0] == 0.00001


def test_get_tick_size_over_time_real_data(testdatadir):
    """
    使用testdatadir中的真实数据测试get_tick_size_over_time函数
    """
    from freqtrade.data.history import load_pair_history

    # 从测试数据目录加载一些测试数据
    pair = "UNITTEST/BTC"
    timeframe = "1m"

    candles = load_pair_history(
        datadir=testdatadir,
        pair=pair,
        timeframe=timeframe,
    )

    # 确保我们有测试数据
    assert not candles.empty, "未找到测试数据，无法运行测试"

    # 计算有效数字
    result = get_tick_size_over_time(candles)

    assert isinstance(result, pd.Series)

    # 验证所有值都在0到1之间（有效的精度值）
    assert all(result > 0)
    assert all(result < 1)

    assert all(result <= 0.0001)
    assert all(result >= 0.00000001)