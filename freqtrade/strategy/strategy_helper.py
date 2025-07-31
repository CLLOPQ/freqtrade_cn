import pandas as pd

from freqtrade.exchange import timeframe_to_minutes


def merge_informative_pair(
    dataframe: pd.DataFrame,
    informative: pd.DataFrame,
    timeframe: str,
    timeframe_inf: str,
    ffill: bool = True,
    append_timeframe: bool = True,
    date_column: str = "date",
    suffix: str | None = None,
) -> pd.DataFrame:
    """
    将信息性样本正确合并到原始数据框，避免前瞻偏差。

    由于日期是蜡烛图的开盘日期，合并15分钟周期蜡烛图（从15:00开始）和
    1小时周期蜡烛图（从15:00开始）会导致所有蜡烛图都知道16:00的收盘价，
    而这是它们不应该知道的。

    将信息性交易对的日期向前移动1个时间间隔。
    这样，14:00的1小时周期蜡烛图会合并到15:00的15分钟周期蜡烛图，因为14:00的1小时周期蜡烛图是
    最后一个在15:00、15:15、15:30或15:45时已闭合的蜡烛图。

    假设信息性时间周期为'1d'，则生成的列将是：
    date_1d、open_1d、high_1d、low_1d、close_1d、rsi_1d

    :param dataframe: 原始数据框
    :param informative: 信息性交易对，很可能通过dp.get_pair_dataframe加载
    :param timeframe: 原始交易对样本的时间周期
    :param timeframe_inf: 信息性交易对样本的时间周期
    :param ffill: 前向填充缺失值——可选但通常是必需的
    :param append_timeframe: 通过追加时间周期重命名列
    :param date_column: 自定义的日期列名
    :param suffix: 在信息性列末尾添加的字符串后缀。如果指定，则append_timeframe必须为false
    :return: 合并后的数据框
    :raise: 如果辅助时间周期短于数据框时间周期，则抛出ValueError
    """
    informative = informative.copy()
    minutes_inf = timeframe_to_minutes(timeframe_inf)
    minutes = timeframe_to_minutes(timeframe)
    if minutes == minutes_inf:
        # 如果时间周期相同，则无需前向移位
        informative["date_merge"] = informative[date_column]
    elif minutes < minutes_inf:
        # 减去"较小"时间周期，避免合并因1个小蜡烛图而延迟
        # 详细说明请参见https://github.com/freqtrade/freqtrade/issues/4073
        if not informative.empty:
            if timeframe_inf == "1M":
                informative["date_merge"] = (
                    informative[date_column] + pd.offsets.MonthBegin(1)
                ) - pd.to_timedelta(minutes, "m")
            else:
                informative["date_merge"] = (
                    informative[date_column]
                    + pd.to_timedelta(minutes_inf, "m")
                    - pd.to_timedelta(minutes, "m")
                )
        else:
            informative["date_merge"] = informative[date_column]
    else:
        raise ValueError(
            "尝试将更快的时间周期合并到更慢的时间周期。"
            "这会创建新行，可能会影响您的常规指标。"
        )

    # 将列重命名以确保唯一性
    date_merge = "date_merge"
    if suffix and append_timeframe:
        raise ValueError("不能同时指定`append_timeframe`为True和`suffix`。")
    elif append_timeframe:
        date_merge = f"date_merge_{timeframe_inf}"
        informative.columns = [f"{col}_{timeframe_inf}" for col in informative.columns]

    elif suffix:
        date_merge = f"date_merge_{suffix}"
        informative.columns = [f"{col}_{suffix}" for col in informative.columns]

    # 合并两个数据框
    # 信息性样本上的所有指标必须在此之前计算完成
    if ffill:
        # https://pandas.pydata.org/docs/user_guide/merging.html#timeseries-friendly-merging
        # merge_ordered - 前向填充方法比单独使用ffill()快2.5倍
        dataframe = pd.merge_ordered(
            dataframe,
            informative,
            fill_method="ffill",
            left_on="date",
            right_on=date_merge,
            how="left",
        )
    else:
        dataframe = pd.merge(
            dataframe, informative, left_on="date", right_on=date_merge, how="left"
        )
    dataframe = dataframe.drop(date_merge, axis=1)

    return dataframe


def stoploss_from_open(
    open_relative_stop: float, current_profit: float, is_short: bool = False, leverage: float = 1.0
) -> float:
    """
    给定当前利润，以及相对于交易入场价的期望止损值，返回一个相对于当前价格的止损值，该值可从`custom_stoploss`返回。

    请求的止损可以为正数（表示在开盘价上方设置止损）或负数（表示在开盘价下方设置止损）。返回值始终≥0。
    如果提供了杠杆，则`open_relative_stop`将被视为已根据杠杆调整。

    如果计算出的止损价高于/低于（多单/空单）当前价格，则返回0。

    :param open_relative_stop: 相对于开盘价的期望止损百分比，已根据杠杆调整
    :param current_profit: 当前利润百分比
    :param is_short: 当为True时，执行空单计算而非多单计算
    :param leverage: 计算时使用的杠杆
    :return: 相对于当前价格的止损值
    """

    # 当current_profit为-1（多单）或1（空单）时，公式无定义，返回最大值
    _current_profit = current_profit / leverage
    if (_current_profit == -1 and not is_short) or (is_short and _current_profit == 1):
        return 1

    if is_short is True:
        stoploss = -1 + ((1 - open_relative_stop / leverage) / (1 - _current_profit))
    else:
        stoploss = 1 - ((1 + open_relative_stop / leverage) / (1 + _current_profit))

    # 负的止损值表示请求的止损价高于/低于（多单/空单）当前价格
    return max(stoploss * leverage, 0.0)


def stoploss_from_absolute(
    stop_rate: float, current_rate: float, is_short: bool = False, leverage: float = 1.0
) -> float:
    """
    给定当前价格和期望的止损价格，返回一个相对于当前价格的止损值。

    请求的止损可以为正数（表示在开盘价上方设置止损）或负数（表示在开盘价下方设置止损）。返回值始终≥0。

    如果计算出的止损价高于当前价格，则返回0。

    :param stop_rate: 止损价格
    :param current_rate: 当前资产价格
    :param is_short: 当为True时，执行空单计算而非多单计算
    :param leverage: 计算时使用的杠杆
    :return: 相对于当前价格的正止损值
    """

    # 当current_rate为0时，公式无定义，返回最大值
    if current_rate == 0:
        return 1

    stoploss = 1 - (stop_rate / current_rate)
    if is_short:
        stoploss = -stoploss

    # 负的止损值表示请求的止损价高于/低于（多单/空单）当前价格
    # 空单可能产生高于1的止损值，因此也限制该值
    return max(min(stoploss, 1.0), 0.0) * leverage