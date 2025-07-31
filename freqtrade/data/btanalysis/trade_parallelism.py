import logging

import numpy as np
import pandas as pd

from freqtrade.constants import IntOrInf


logger = logging.getLogger(__name__)


def analyze_trade_parallelism(trades: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    通过为每个交易在其开放的每个时间段扩展一次，然后计算重叠来查找重叠交易。
    :param trades: 交易数据框 - 可以从回测加载，或通过 trade_list_to_dataframe 创建
    :param timeframe: 回测使用的时间框架
    :return: 每个时间段内的开仓计数数据框
    """
    from freqtrade.exchange import timeframe_to_resample_freq

    timeframe_freq = timeframe_to_resample_freq(timeframe)
    dates = [
        pd.Series(
            pd.date_range(
                row[1]["open_date"],
                row[1]["close_date"],
                freq=timeframe_freq,
                # 排除右边界 - 日期是蜡烛图的开盘日期。
                inclusive="left",
            )
        )
        for row in trades[["open_date", "close_date"]].iterrows()
    ]
    deltas = [len(x) for x in dates]
    dates = pd.Series(pd.concat(dates).values, name="date")
    df2 = pd.DataFrame(np.repeat(trades.values, deltas, axis=0), columns=trades.columns)

    df2 = pd.concat([dates, df2], axis=1)
    df2 = df2.set_index("date")
    df_final = df2.resample(timeframe_freq)[["pair"]].count()
    df_final = df_final.rename({"pair": "open_trades"}, axis=1)
    return df_final


def evaluate_result_multi(
    trades: pd.DataFrame, timeframe: str, max_open_trades: IntOrInf
) -> pd.DataFrame:
    """
    通过为每个交易在其开放的每个时间段扩展一次，然后计算重叠来查找重叠交易
    :param trades: 交易数据框 - 可以从回测加载，或通过 trade_list_to_dataframe 创建
    :param timeframe: 回测使用的频率
    :param max_open_trades: 回测运行期间使用的 max_open_trades 参数
    :return: 每个时间段内的开仓计数数据框
    """
    df_final = analyze_trade_parallelism(trades, timeframe)
    return df_final[df_final["open_trades"] > max_open_trades]