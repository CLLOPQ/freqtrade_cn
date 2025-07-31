from datetime import timedelta
from typing import NamedTuple

from pandas import DataFrame

from freqtrade.enums import ExitType
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.util.datetime_helpers import dt_utc


tests_start_time = dt_utc(2018, 10, 3)
tests_timeframe = "1h"


class BTrade(NamedTuple):
    """
    用于功能性回测的简化交易结果
    """
    exit_reason: ExitType
    open_tick: int
    close_tick: int
    enter_tag: str | None = None
    is_short: bool = False


class BTContainer(NamedTuple):
    """
    简化的回测容器，定义回测输入和结果
    """
    data: list[list[float]]
    stop_loss: float
    roi: dict[str, float]
    trades: list[BTrade]
    profit_perc: float
    trailing_stop: bool = False
    trailing_only_offset_is_reached: bool = False
    trailing_stop_positive: float | None = None
    trailing_stop_positive_offset: float = 0.0
    use_exit_signal: bool = False
    use_custom_stoploss: bool = False
    custom_entry_price: float | None = None
    custom_exit_price: float | None = None
    leverage: float = 1.0
    timeout: int | None = None
    adjust_entry_price: float | None = None
    adjust_exit_price: float | None = None
    adjust_trade_position: list[float] | None = None


def _get_frame_time_from_offset(offset):
    """根据偏移量计算对应的时间"""
    minutes = offset * timeframe_to_minutes(tests_timeframe)
    return tests_start_time + timedelta(minutes=minutes)


def _build_backtest_dataframe(data):
    """构建回测用的DataFrame数据"""
    columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "enter_long",
        "exit_long",
        "enter_short",
        "exit_short",
    ]
    if len(data[0]) == 8:
        # 没有做空相关列，添加默认值
        data = [[*d, 0, 0] for d in data]
    # 如果有入场标签列则添加
    columns = [*columns, "enter_tag"] if len(data[0]) == 11 else columns

    frame = DataFrame.from_records(data, columns=columns)
    frame["date"] = frame["date"].apply(_get_frame_time_from_offset)
    # 确保数值列为浮点型
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = frame[column].astype("float64")

    # 确保所有K线数据合理
    assert all(frame["low"] <= frame["close"])
    assert all(frame["low"] <= frame["open"])
    assert all(frame["high"] >= frame["close"])
    assert all(frame["high"] >= frame["open"])
    return frame