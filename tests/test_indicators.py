import numpy as np
import pandas as pd

import freqtrade.vendor.qtpylib.indicators as qtpylib


def test_crossed_numpy_types():
    """
    本测试的存在是因为当前方法与qtpylib的实现存在差异。
    当我们从原始来源进行更新时，必须确保不会再次破坏此功能。
    """
    # 创建测试用的序列数据
    series = pd.Series([56, 97, 19, 76, 65, 25, 87, 91, 79, 79])
    # 预期的结果序列：当数值从下方突破60时为True，否则为False
    expected_result = pd.Series([False, True, False, True, False, False, True, False, False, False])

    # 测试不同数据类型的阈值参数是否都能得到正确结果
    # 整数类型阈值
    assert qtpylib.crossed_above(series, 60).equals(expected_result)
    # 浮点数类型阈值
    assert qtpylib.crossed_above(series, 60.0).equals(expected_result)
    # numpy整数类型阈值（32位）
    assert qtpylib.crossed_above(series, np.int32(60)).equals(expected_result)
    # numpy整数类型阈值（64位）
    assert qtpylib.crossed_above(series, np.int64(60)).equals(expected_result)
    # numpy浮点数类型阈值（64位）
    assert qtpylib.crossed_above(series, np.float64(60.0)).equals(expected_result)