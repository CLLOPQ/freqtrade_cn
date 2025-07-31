# QTPyLib: 量化交易Python库
# https://github.com/ranaroussi/qtpylib
#
# 版权所有 2016-2018 Ran Aroussi
#
# 根据Apache许可证2.0版（"许可证"）授权；
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律要求或书面同意，否则根据许可证分发的软件按"原样"基础提供，
# 不附带任何明示或暗示的担保或条件。请参阅许可证了解特定语言下的权限和
# 限制。
#

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pandas.core.base import PandasObject


# =============================================
warnings.simplefilter(action="ignore", category=RuntimeWarning)

# =============================================


def numpy_rolling_window(data, window):
    """
    使用numpy创建滚动窗口

    参数:
        data: 输入数据数组
        window: 窗口大小

    返回:
        包含滚动窗口数据的数组
    """
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = (*data.strides, data.strides[-1])
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def numpy_rolling_series(func):
    """
    滚动计算装饰器，将numpy滚动计算应用于pandas Series

    参数:
        func: 要应用的滚动计算函数
    """
    def func_wrapper(data, window, as_source=False):
        """
        滚动计算函数包装器

        参数:
            data: 输入数据（pandas Series或numpy数组）
            window: 窗口大小
            as_source: 是否返回pandas Series（默认返回numpy数组）

        返回:
            滚动计算结果（pandas Series或numpy数组）
        """
        series = data.values if isinstance(data, pd.Series) else data

        new_series = np.empty(len(series)) * np.nan
        calculated = func(series, window)
        new_series[-len(calculated) :] = calculated

        if as_source and isinstance(data, pd.Series):
            return pd.Series(index=data.index, data=new_series)

        return new_series

    return func_wrapper


@numpy_rolling_series
def numpy_rolling_mean(data, window, as_source=False):
    """
    计算滚动窗口均值

    参数:
        data: 输入数据（pandas Series或numpy数组）
        window: 窗口大小
        as_source: 是否返回pandas Series（默认返回numpy数组）

    返回:
        滚动窗口均值结果
    """
    return np.mean(numpy_rolling_window(data, window), axis=-1)


@numpy_rolling_series
def numpy_rolling_std(data, window, as_source=False):
    """
    计算滚动窗口标准差（样本标准差）

    参数:
        data: 输入数据（pandas Series或numpy数组）
        window: 窗口大小
        as_source: 是否返回pandas Series（默认返回numpy数组）

    返回:
        滚动窗口标准差结果
    """
    return np.std(numpy_rolling_window(data, window), axis=-1, ddof=1)


# ---------------------------------------------


def session(df, start="17:00", end="16:00"):
    """从df中移除前一个Globex时段的日期"""
    if df.empty:
        return df

    # 将开始/结束/当前时间转换为小数形式
    int_start = list(map(int, start.split(":")))
    int_start = (int_start[0] + int_start[1] - 1 / 100) - 0.0001
    int_end = list(map(int, end.split(":")))
    int_end = int_end[0] + int_end[1] / 100
    int_now = df[-1:].index.hour[0] + (df[:1].index.minute[0]) / 100

    # 是否为同一交易日？
    is_same_day = int_end > int_start

    # 设置指针
    curr = prev = df[-1:].index[0].strftime("%Y-%m-%d")

    # Globex/外汇时段
    if not is_same_day:
        prev = (datetime.strptime(curr, "%Y-%m-%d") - timedelta(1)).strftime("%Y-%m-%d")

    # 切片
    if int_now >= int_start:
        df = df[df.index >= curr + " " + start]
    else:
        df = df[df.index >= prev + " " + start]

    return df.copy()


# ---------------------------------------------


def heikinashi(bars):
    """生成海克阿希蜡烛图（Heikin Ashi）"""
    bars = bars.copy()
    bars["ha_close"] = (bars["open"] + bars["high"] + bars["low"] + bars["close"]) / 4

    # 海克阿希开盘价
    bars.at[0, "ha_open"] = (bars.at[0, "open"] + bars.at[0, "close"]) / 2
    for i in range(1, len(bars)):
        bars.at[i, "ha_open"] = (bars.at[i - 1, "ha_open"] + bars.at[i - 1, "ha_close"]) / 2

    bars["ha_high"] = bars.loc[:, ["high", "ha_open", "ha_close"]].max(axis=1)
    bars["ha_low"] = bars.loc[:, ["low", "ha_open", "ha_close"]].min(axis=1)

    return pd.DataFrame(
        index=bars.index,
        data={
            "open": bars["ha_open"],
            "high": bars["ha_high"],
            "low": bars["ha_low"],
            "close": bars["ha_close"],
        },
    )


# ---------------------------------------------


def tdi(series, rsi_lookback=13, rsi_smooth_len=2, rsi_signal_len=7, bb_lookback=34, bb_std=1.6185):
    """计算TDI指标（技术指标组合）"""
    rsi_data = rsi(series, rsi_lookback)
    rsi_smooth = sma(rsi_data, rsi_smooth_len)
    rsi_signal = sma(rsi_data, rsi_signal_len)

    bb_series = bollinger_bands(rsi_data, bb_lookback, bb_std)

    return pd.DataFrame(
        index=series.index,
        data={
            "rsi": rsi_data,
            "rsi_signal": rsi_signal,
            "rsi_smooth": rsi_smooth,
            "rsi_bb_upper": bb_series["upper"],
            "rsi_bb_lower": bb_series["lower"],
            "rsi_bb_mid": bb_series["mid"],
        },
    )


# ---------------------------------------------


def awesome_oscillator(df, weighted=False, fast=5, slow=34):
    """计算震憾振荡器（Awesome Oscillator）"""
    midprice = (df["high"] + df["low"]) / 2

    if weighted:
        ao = (midprice.ewm(fast).mean() - midprice.ewm(slow).mean()).values
    else:
        ao = numpy_rolling_mean(midprice, fast) - numpy_rolling_mean(midprice, slow)

    return pd.Series(index=df.index, data=ao)


# ---------------------------------------------


def nans(length=1):
    """创建指定长度的NaN数组"""
    mtx = np.empty(length)
    mtx[:] = np.nan
    return mtx


# ---------------------------------------------


def typical_price(bars):
    """计算典型价格（Typical Price）"""
    res = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    return pd.Series(index=bars.index, data=res)


# ---------------------------------------------


def mid_price(bars):
    """计算中间价格（Mid Price）"""
    res = (bars["high"] + bars["low"]) / 2.0
    return pd.Series(index=bars.index, data=res)


# ---------------------------------------------


def ibs(bars):
    """内部柱体强度（Internal Bar Strength）"""
    res = np.round((bars["close"] - bars["low"]) / (bars["high"] - bars["low"]), 2)
    return pd.Series(index=bars.index, data=res)


# ---------------------------------------------


def true_range(bars):
    """计算真实波幅（True Range）"""
    return pd.DataFrame(
        {
            "hl": bars["high"] - bars["low"],
            "hc": abs(bars["high"] - bars["close"].shift(1)),
            "lc": abs(bars["low"] - bars["close"].shift(1)),
        }
    ).max(axis=1)


# ---------------------------------------------


def atr(bars, window=14, exp=False):
    """计算平均真实波幅（Average True Range）"""
    tr = true_range(bars)

    if exp:
        res = rolling_weighted_mean(tr, window)
    else:
        res = rolling_mean(tr, window)

    return pd.Series(res)


# ---------------------------------------------


def crossed(series1, series2, direction=None):
    """判断两个序列是否交叉"""
    if isinstance(series1, np.ndarray):
        series1 = pd.Series(series1)

    if isinstance(series2, float | int | np.ndarray | np.integer | np.floating):
        series2 = pd.Series(index=series1.index, data=series2)

    if direction is None or direction == "above":
        above = pd.Series((series1 > series2) & (series1.shift(1) <= series2.shift(1)))

    if direction is None or direction == "below":
        below = pd.Series((series1 < series2) & (series1.shift(1) >= series2.shift(1)))

    if direction is None:
        return above | below

    return above if direction == "above" else below


def crossed_above(series1, series2):
    """判断series1是否从下方穿过series2（金叉）"""
    return crossed(series1, series2, "above")


def crossed_below(series1, series2):
    """判断series1是否从上方穿过series2（死叉）"""
    return crossed(series1, series2, "below")


# ---------------------------------------------


def rolling_std(series, window=200, min_periods=None):
    """计算滚动窗口标准差"""
    min_periods = window if min_periods is None else min_periods
    if min_periods == window and len(series) > window:
        return numpy_rolling_std(series, window, True)
    else:
        try:
            return series.rolling(window=window, min_periods=min_periods).std()
        except Exception as e:  # noqa: F841
            return pd.Series(series).rolling(window=window, min_periods=min_periods).std()


# ---------------------------------------------


def rolling_mean(series, window=200, min_periods=None):
    """计算滚动窗口均值"""
    min_periods = window if min_periods is None else min_periods
    if min_periods == window and len(series) > window:
        return numpy_rolling_mean(series, window, True)
    else:
        try:
            return series.rolling(window=window, min_periods=min_periods).mean()
        except Exception as e:  # noqa: F841
            return pd.Series(series).rolling(window=window, min_periods=min_periods).mean()


# ---------------------------------------------


def rolling_min(series, window=14, min_periods=None):
    """计算滚动窗口最小值"""
    min_periods = window if min_periods is None else min_periods
    try:
        return series.rolling(window=window, min_periods=min_periods).min()
    except Exception as e:  # noqa: F841
        return pd.Series(series).rolling(window=window, min_periods=min_periods).min()


# ---------------------------------------------


def rolling_max(series, window=14, min_periods=None):
    """计算滚动窗口最大值"""
    min_periods = window if min_periods is None else min_periods
    try:
        return series.rolling(window=window, min_periods=min_periods).max()
    except Exception as e:  # noqa: F841
        return pd.Series(series).rolling(window=window, min_periods=min_periods).max()


# ---------------------------------------------


def rolling_weighted_mean(series, window=200, min_periods=None):
    """计算滚动窗口加权均值（指数移动平均）"""
    min_periods = window if min_periods is None else min_periods
    try:
        return series.ewm(span=window, min_periods=min_periods).mean()
    except Exception as e:  # noqa: F841
        return pd.ewma(series, span=window, min_periods=min_periods)


# ---------------------------------------------


def hull_moving_average(series, window=200, min_periods=None):
    """计算赫尔移动平均线（Hull Moving Average）"""
    min_periods = window if min_periods is None else min_periods
    ma = (2 * rolling_weighted_mean(series, window / 2, min_periods)) - rolling_weighted_mean(
        series, window, min_periods
    )
    return rolling_weighted_mean(ma, np.sqrt(window), min_periods)


# ---------------------------------------------


def sma(series, window=200, min_periods=None):
    """计算简单移动平均线（Simple Moving Average）"""
    return rolling_mean(series, window=window, min_periods=min_periods)


# ---------------------------------------------


def wma(series, window=200, min_periods=None):
    """计算加权移动平均线（Weighted Moving Average）"""
    return rolling_weighted_mean(series, window=window, min_periods=min_periods)


# ---------------------------------------------


def hma(series, window=200, min_periods=None):
    """计算赫尔移动平均线（Hull Moving Average）"""
    return hull_moving_average(series, window=window, min_periods=min_periods)


# ---------------------------------------------


def vwap(bars):
    """
    计算成交量加权平均价（Volume-Weighted Average Price）
    （输入可以是pandas Series或numpy数组）
    通常使用中间价[(h+l)/2]或典型价[(h+l+c)/3]
    """
    raise ValueError(
        "使用`qtpylib.vwap`会引入前瞻偏差。请使用`qtpylib.rolling_vwap`替代，"
        "它以滚动方式计算成交量加权平均价。"
    )
    # typical = ((bars['high'] + bars['low'] + bars['close']) / 3).values
    # volume = bars['volume'].values

    # return pd.Series(index=bars.index,
    #                  data=np.cumsum(volume * typical) / np.cumsum(volume))


# ---------------------------------------------


def rolling_vwap(bars, window=200, min_periods=None):
    """
    计算滚动窗口成交量加权平均价（Rolling VWAP）
    （输入可以是pandas Series或numpy数组）
    通常使用中间价[(h+l)/2]或典型价[(h+l+c)/3]
    """
    min_periods = window if min_periods is None else min_periods

    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    volume = bars["volume"]

    left = (volume * typical).rolling(window=window, min_periods=min_periods).sum()
    right = volume.rolling(window=window, min_periods=min_periods).sum()

    return (
        pd.Series(index=bars.index, data=(left / right))
        .replace([np.inf, -np.inf], float("NaN"))
        .ffill()
    )


# ---------------------------------------------


def rsi(series, window=14):
    """
    计算n周期相对强弱指数（Relative Strength Index）
    """

    # 100-(100/相对强弱值)
    deltas = np.diff(series)
    seed = deltas[: window + 1]

    # 默认值
    ups = seed[seed > 0].sum() / window
    downs = -seed[seed < 0].sum() / window
    rsival = np.zeros_like(series)
    rsival[:window] = 100.0 - 100.0 / (1.0 + ups / downs)

    # 周期值
    for i in range(window, len(series)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0
        else:
            upval = 0
            downval = -delta

        ups = (ups * (window - 1) + upval) / window
        downs = (downs * (window - 1.0) + downval) / window
        rsival[i] = 100.0 - 100.0 / (1.0 + ups / downs)

    # 返回RSI值
    return pd.Series(index=series.index, data=rsival)


# ---------------------------------------------


def macd(series, fast=3, slow=10, smooth=16):
    """
    计算MACD（指数平滑异同平均线）
    使用快、慢指数移动平均线
    返回值为ema慢线、ema快线、MACD线
    """
    macd_line = rolling_weighted_mean(series, window=fast) - rolling_weighted_mean(
        series, window=slow
    )
    signal = rolling_weighted_mean(macd_line, window=smooth)
    histogram = macd_line - signal
    # 返回MACD线、信号线、柱状图
    return pd.DataFrame(
        index=series.index,
        data={"macd": macd_line.values, "signal": signal.values, "histogram": histogram.values},
    )


# ---------------------------------------------


def bollinger_bands(series, window=20, stds=2):
    """计算布林带（Bollinger Bands）"""
    ma = rolling_mean(series, window=window, min_periods=1)
    std = rolling_std(series, window=window, min_periods=1)
    upper = ma + std * stds
    lower = ma - std * stds

    return pd.DataFrame(index=series.index, data={"upper": upper, "mid": ma, "lower": lower})


# ---------------------------------------------


def weighted_bollinger_bands(series, window=20, stds=2):
    """计算加权布林带（Weighted Bollinger Bands）"""
    ema = rolling_weighted_mean(series, window=window)
    std = rolling_std(series, window=window)
    upper = ema + std * stds
    lower = ema - std * stds

    return pd.DataFrame(
        index=series.index, data={"upper": upper.values, "mid": ema.values, "lower": lower.values}
    )


# ---------------------------------------------


def returns(series):
    """计算收益率（Returns）"""
    try:
        res = (series / series.shift(1) - 1).replace([np.inf, -np.inf], float("NaN"))
    except Exception as e:  # noqa: F841
        res = nans(len(series))

    return pd.Series(index=series.index, data=res)


# ---------------------------------------------


def log_returns(series):
    """计算对数收益率（Log Returns）"""
    try:
        res = np.log(series / series.shift(1)).replace([np.inf, -np.inf], float("NaN"))
    except Exception as e:  # noqa: F841
        res = nans(len(series))

    return pd.Series(index=series.index, data=res)


# ---------------------------------------------


def implied_volatility(series, window=252):
    """计算隐含波动率（Implied Volatility）"""
    try:
        logret = np.log(series / series.shift(1)).replace([np.inf, -np.inf], float("NaN"))
        res = numpy_rolling_std(logret, window) * np.sqrt(window)
    except Exception as e:  # noqa: F841
        res = nans(len(series))

    return pd.Series(index=series.index, data=res)


# ---------------------------------------------


def keltner_channel(bars, window=14, atrs=2):
    """计算肯特纳通道（Keltner Channel）"""
    typical_mean = rolling_mean(typical_price(bars), window)
    atrval = atr(bars, window) * atrs

    upper = typical_mean + atrval
    lower = typical_mean - atrval

    return pd.DataFrame(
        index=bars.index,
        data={"upper": upper.values, "mid": typical_mean.values, "lower": lower.values},
    )


# ---------------------------------------------


def roc(series, window=14):
    """
    计算变动率指标（Rate of Change）
    """
    res = (series - series.shift(window)) / series.shift(window)
    return pd.Series(index=series.index, data=res)


# ---------------------------------------------


def cci(series, window=14):
    """
    计算商品通道指数（Commodity Channel Index）
    """
    price = typical_price(series)
    typical_mean = rolling_mean(price, window)
    res = (price - typical_mean) / (0.015 * np.std(typical_mean))
    return pd.Series(index=series.index, data=res)


# ---------------------------------------------


def stoch(df, window=14, d=3, k=3, fast=False):
    """
    计算随机指标（Stochastic Oscillator）
    http://excelta.blogspot.co.il/2013/09/stochastic-oscillator-technical.html
    """

    my_df = pd.DataFrame(index=df.index)

    my_df["rolling_max"] = df["high"].rolling(window).max()
    my_df["rolling_min"] = df["low"].rolling(window).min()

    my_df["fast_k"] = (
        100 * (df["close"] - my_df["rolling_min"]) / (my_df["rolling_max"] - my_df["rolling_min"])
    )
    my_df["fast_d"] = my_df["fast_k"].rolling(d).mean()

    if fast:
        return my_df.loc[:, ["fast_k", "fast_d"]]

    my_df["slow_k"] = my_df["fast_k"].rolling(k).mean()
    my_df["slow_d"] = my_df["slow_k"].rolling(d).mean()

    return my_df.loc[:, ["slow_k", "slow_d"]]


# ---------------------------------------------


def zlma(series, window=20, min_periods=None, kind="ema"):
    """
    约翰·艾略特的零滞后（指数）移动平均线（Zero lag EMA）
    https://en.wikipedia.org/wiki/Zero_lag_exponential_moving_average
    """
    min_periods = window if min_periods is None else min_periods

    lag = (window - 1) // 2
    series = 2 * series - series.shift(lag)
    if kind in ["ewm", "ema"]:
        return wma(series, lag, min_periods)
    elif kind == "hma":
        return hma(series, lag, min_periods)
    return sma(series, lag, min_periods)


def zlema(series, window, min_periods=None):
    """零滞后指数移动平均线（Zero Lag EMA）"""
    return zlma(series, window, min_periods, kind="ema")


def zlsma(series, window, min_periods=None):
    """零滞后简单移动平均线（Zero Lag SMA）"""
    return zlma(series, window, min_periods, kind="sma")


def zlhma(series, window, min_periods=None):
    """零滞后赫尔移动平均线（Zero Lag HMA）"""
    return zlma(series, window, min_periods, kind="hma")


# ---------------------------------------------


def zscore(bars, window=20, stds=1, col="close"):
    """计算价格的Z分数（Z-Score）"""
    std = numpy_rolling_std(bars[col], window)
    mean = numpy_rolling_mean(bars[col], window)
    return (bars[col] - mean) / (std * stds)


# ---------------------------------------------


def pvt(bars):
    """计算价格动量线（Price Volume Trend）"""
    trend = ((bars["close"] - bars["close"].shift(1)) / bars["close"].shift(1)) * bars["volume"]
    return trend.cumsum()


def chopiness(bars, window=14):
    """计算波动性指标（Chopiness Index）"""
    atrsum = true_range(bars).rolling(window).sum()
    highs = bars["high"].rolling(window).max()
    lows = bars["low"].rolling(window).min()
    return 100 * np.log10(atrsum / (highs - lows)) / np.log10(window)


# =============================================


PandasObject.session = session
PandasObject.atr = atr
PandasObject.bollinger_bands = bollinger_bands
PandasObject.cci = cci
PandasObject.crossed = crossed
PandasObject.crossed_above = crossed_above
PandasObject.crossed_below = crossed_below
PandasObject.heikinashi = heikinashi
PandasObject.hull_moving_average = hull_moving_average
PandasObject.ibs = ibs
PandasObject.implied_volatility = implied_volatility
PandasObject.keltner_channel = keltner_channel
PandasObject.log_returns = log_returns
PandasObject.macd = macd
PandasObject.returns = returns
PandasObject.roc = roc
PandasObject.rolling_max = rolling_max
PandasObject.rolling_min = rolling_min
PandasObject.rolling_mean = rolling_mean
PandasObject.rolling_std = rolling_std
PandasObject.rsi = rsi
PandasObject.stoch = stoch
PandasObject.zscore = zscore
PandasObject.pvt = pvt
PandasObject.chopiness = chopiness
PandasObject.tdi = tdi
PandasObject.true_range = true_range
PandasObject.mid_price = mid_price
PandasObject.typical_price = typical_price
PandasObject.vwap = vwap
PandasObject.rolling_vwap = rolling_vwap
PandasObject.weighted_bollinger_bands = weighted_bollinger_bands
PandasObject.rolling_weighted_mean = rolling_weighted_mean

PandasObject.sma = sma
PandasObject.wma = wma
PandasObject.ema = wma
PandasObject.hma = hma

PandasObject.zlsma = zlsma
PandasObject.zlwma = zlema
PandasObject.zlema = zlema
PandasObject.zlhma = zlhma
PandasObject.zlma = zlma