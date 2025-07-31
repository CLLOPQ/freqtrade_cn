from unittest.mock import MagicMock

import time_machine

from freqtrade.util import MeasureTime


def test_measure_time():
    """测试MeasureTime类的功能"""
    # 创建一个模拟回调函数
    callback = MagicMock()
    
    # 使用time_machine控制时间
    with time_machine.travel("2021-09-01 05:00:00 +00:00", tick=False) as t:
        # 初始化MeasureTime，阈值为5，TTL为60
        measure = MeasureTime(callback, 5, ttl=60)
        
        # 第一次进入上下文管理器，不耗时
        with measure:
            pass
        # 未达到阈值，回调不应被调用
        assert callback.call_count == 0

        # 第二次进入上下文管理器，耗时10单位
        with measure:
            t.shift(10)
        # 超过阈值，回调应被调用一次
        assert callback.call_count == 1
        
        # 重置回调并第三次进入上下文管理器，再次耗时10单位
        callback.reset_mock()
        with measure:
            t.shift(10)
        # 在TTL内，回调不应被调用
        assert callback.call_count == 0

        # 重置回调并将时间推移45单位（超过TTL）
        callback.reset_mock()
        t.shift(45)

        # 第四次进入上下文管理器，耗时10单位
        with measure:
            t.shift(10)
        # TTL已过，再次超过阈值，回调应被调用一次
        assert callback.call_count == 1