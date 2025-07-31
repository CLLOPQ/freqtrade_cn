import logging
import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def calculate_market_change(
    data: dict[str, pd.DataFrame], column: str = "close", min_date: datetime | None = None
) -> float:
    """
    基于"column"计算市场变化。
    计算方法是取每列的第一个非空值和最后一个非空值，计算百分比变化"(last - first) / first"。
    然后取所有交易对结果的平均值作为最终结果。

    :param data: 数据帧字典，字典键应为交易对。
    :param column: 要使用的原始数据帧中的列
    :param min_date: 计算时考虑的最小日期。市场变化应仅基于实际回测的数据计算，排除启动阶段。
    :return: 市场变化平均值
    """
    tmp_means = []
    for pair, df in data.items():
        df1 = df
        if min_date is not None:
            df1 = df1[df1["date"] >= min_date]
        if df1.empty:
            logger.warning(f"交易对 {pair} 在 {min_date} 之后没有数据。")
            continue
        start = df1[column].dropna().iloc[0]
        end = df1[column].dropna().iloc[-1]
        tmp_means.append((end - start) / start)

    if not tmp_means:
        return 0.0
    return float(np.mean(tmp_means))


def combine_dataframes_by_column(
    data: dict[str, pd.DataFrame], column: str = "close"
) -> pd.DataFrame:
    """
    合并多个数据帧的"column"列
    :param data: 数据帧字典，字典键应为交易对。
    :param column: 要使用的原始数据帧中的列
    :return: 列名重命名为字典键的数据帧。
    :raise: 如果未提供数据则抛出ValueError。
    """
    if not data:
        raise ValueError("未提供数据。")
    df_comb = pd.concat(
        [data[pair].set_index("date").rename({column: pair}, axis=1)[pair] for pair in data], axis=1
    )
    return df_comb


def combined_dataframes_with_rel_mean(
    data: dict[str, pd.DataFrame], fromdt: datetime, todt: datetime, column: str = "close"
) -> pd.DataFrame:
    """
    合并多个数据帧的"column"列
    :param data: 数据帧字典，字典键应为交易对。
    :param column: 要使用的原始数据帧中的列
    :return: 列名重命名为字典键的数据帧，包含一个名为mean的列，包含所有交易对的平均值。
    :raise: 如果未提供数据则抛出ValueError。
    """
    df_comb = combine_dataframes_by_column(data, column)
    # 将数据帧修剪到给定的时间范围
    df_comb = df_comb.iloc[(df_comb.index >= fromdt) & (df_comb.index < todt)]
    df_comb["count"] = df_comb.count(axis=1)
    df_comb["mean"] = df_comb.mean(axis=1)
    df_comb["rel_mean"] = df_comb["mean"].pct_change().fillna(0).cumsum()
    return df_comb[["mean", "rel_mean", "count"]]


def combine_dataframes_with_mean(
    data: dict[str, pd.DataFrame], column: str = "close"
) -> pd.DataFrame:
    """
    合并多个数据帧的"column"列
    :param data: 数据帧字典，字典键应为交易对。
    :param column: 要使用的原始数据帧中的列
    :return: 列名重命名为字典键的数据帧，包含一个名为mean的列，包含所有交易对的平均值。
    :raise: 如果未提供数据则抛出ValueError。
    """
    df_comb = combine_dataframes_by_column(data, column)

    df_comb["mean"] = df_comb.mean(axis=1)

    return df_comb


def create_cum_profit(
    df: pd.DataFrame, trades: pd.DataFrame, col_name: str, timeframe: str
) -> pd.DataFrame:
    """
    添加一个名为`col_name`的列，包含给定交易数组的累积利润。
    :param df: 带日期索引的数据帧
    :param trades: 包含交易的数据帧（需要close_date和profit_abs列）
    :param col_name: 将被分配结果的列名
    :param timeframe: 操作期间使用的时间框架
    :return: 返回带有一个附加列col_name的数据帧，包含累积利润。
    :raise: 如果交易数据帧为空则抛出ValueError。
    """
    if len(trades) == 0:
        raise ValueError("交易数据帧为空。")
    from freqtrade.exchange import timeframe_to_resample_freq

    timeframe_freq = timeframe_to_resample_freq(timeframe)
    # 重采样到时间框架，确保交易与K线匹配
    _trades_sum = trades.resample(timeframe_freq, on="close_date")[["profit_abs"]].sum()
    df.loc[:, col_name] = _trades_sum["profit_abs"].cumsum()
    # 将第一个值设为0
    df.loc[df.iloc[0].name, col_name] = 0
    # 向前填充以获得连续数据
    df[col_name] = df[col_name].ffill()
    return df


def _calc_drawdown_series(
    profit_results: pd.DataFrame, *, date_col: str, value_col: str, starting_balance: float
) -> pd.DataFrame:
    max_drawdown_df = pd.DataFrame()
    max_drawdown_df["cumulative"] = profit_results[value_col].cumsum()
    max_drawdown_df["high_value"] = np.maximum(0, max_drawdown_df["cumulative"].cummax())
    max_drawdown_df["drawdown"] = max_drawdown_df["cumulative"] - max_drawdown_df["high_value"]
    max_drawdown_df["date"] = profit_results.loc[:, date_col]
    if starting_balance:
        cumulative_balance = starting_balance + max_drawdown_df["cumulative"]
        max_balance = starting_balance + max_drawdown_df["high_value"]
        max_drawdown_df["drawdown_relative"] = (max_balance - cumulative_balance) / max_balance
    else:
        # 注意：这并不完全准确，
        # 但如果没有可用的starting_balance，可能已经足够好
        max_drawdown_df["drawdown_relative"] = (
            max_drawdown_df["high_value"] - max_drawdown_df["cumulative"]
        ) / max_drawdown_df["high_value"]
    return max_drawdown_df


def calculate_underwater(
    trades: pd.DataFrame,
    *,
    date_col: str = "close_date",
    value_col: str = "profit_ratio",
    starting_balance: float = 0.0,
):
    """
    计算最大回撤和相应的平仓日期
    :param trades: 包含交易的数据帧（需要close_date和profit_ratio列）
    :param date_col: 数据帧中用于日期的列（默认为'close_date'）
    :param value_col: 数据帧中用于值的列（默认为'profit_ratio'）
    :return: 包含累积收益、高值、回撤和相对回撤的数据帧
    :raise: 如果交易数据帧为空则抛出ValueError。
    """
    if len(trades) == 0:
        raise ValueError("交易数据帧为空。")
    profit_results = trades.sort_values(date_col).reset_index(drop=True)
    max_drawdown_df = _calc_drawdown_series(
        profit_results, date_col=date_col, value_col=value_col, starting_balance=starting_balance
    )

    return max_drawdown_df


@dataclass()
class DrawDownResult:
    drawdown_abs: float = 0.0
    high_date: pd.Timestamp = None
    low_date: pd.Timestamp = None
    high_value: float = 0.0
    low_value: float = 0.0
    relative_account_drawdown: float = 0.0


def calculate_max_drawdown(
    trades: pd.DataFrame,
    *,
    date_col: str = "close_date",
    value_col: str = "profit_abs",
    starting_balance: float = 0,
    relative: bool = False,
) -> DrawDownResult:
    """
    计算最大回撤和相应的平仓日期
    :param trades: 包含交易的数据帧（需要close_date和profit_ratio列）
    :param date_col: 数据帧中用于日期的列（默认为'close_date'）
    :param value_col: 数据帧中用于值的列（默认为'profit_abs'）
    :param starting_balance: 投资组合起始资金 - 用于正确计算相对回撤。
    :return: DrawDownResult对象
             包含绝对最大回撤、高低点时间、高低点价值，
             以及相对账户回撤
    :raise: 如果交易数据帧为空则抛出ValueError。
    """
    if len(trades) == 0:
        raise ValueError("交易数据帧为空。")
    profit_results = trades.sort_values(date_col).reset_index(drop=True)
    max_drawdown_df = _calc_drawdown_series(
        profit_results, date_col=date_col, value_col=value_col, starting_balance=starting_balance
    )

    idxmin = (
        max_drawdown_df["drawdown_relative"].idxmax()
        if relative
        else max_drawdown_df["drawdown"].idxmin()
    )

    high_idx = max_drawdown_df.iloc[: idxmin + 1]["high_value"].idxmax()
    high_date = profit_results.loc[high_idx, date_col]
    low_date = profit_results.loc[idxmin, date_col]
    high_val = max_drawdown_df.loc[high_idx, "cumulative"]
    low_val = max_drawdown_df.loc[idxmin, "cumulative"]
    max_drawdown_rel = max_drawdown_df.loc[idxmin, "drawdown_relative"]

    return DrawDownResult(
        drawdown_abs=abs(max_drawdown_df.loc[idxmin, "drawdown"]),
        high_date=high_date,
        low_date=low_date,
        high_value=high_val,
        low_value=low_val,
        relative_account_drawdown=max_drawdown_rel,
    )


def calculate_csum(trades: pd.DataFrame, starting_balance: float = 0) -> tuple[float, float]:
    """
    计算交易的累计和的最小值/最大值，以显示钱包/赌注金额比率是否合理
    :param trades: 包含交易的数据帧（需要close_date和profit_percent列）
    :param starting_balance: 将起始资金添加到结果中，以显示钱包的高点/低点
    :return: 包含profit_abs累计和的元组（float, float）
    :raise: 如果交易数据帧为空则抛出ValueError。
    """
    if len(trades) == 0:
        raise ValueError("交易数据帧为空。")

    csum_df = pd.DataFrame()
    csum_df["sum"] = trades["profit_abs"].cumsum()
    csum_min = csum_df["sum"].min() + starting_balance
    csum_max = csum_df["sum"].max() + starting_balance

    return csum_min, csum_max


def calculate_cagr(days_passed: int, starting_balance: float, final_balance: float) -> float:
    """
    计算复合年增长率（CAGR）
    :param days_passed: 起始和结束资金之间的天数
    :param starting_balance: 起始资金
    :param final_balance: 用于计算CAGR的最终资金
    :return: 复合年增长率
    """
    if final_balance < 0:
        # 对于杠杆交易，最终资金可能变为负数。
        return 0
    return (final_balance / starting_balance) ** (1 / (days_passed / 365)) - 1


def calculate_expectancy(trades: pd.DataFrame) -> tuple[float, float]:
    """
    计算期望值
    :param trades: 包含交易的数据帧（需要close_date和profit_abs列）
    :return: 期望值，期望值比率
    """

    expectancy = 0.0
    expectancy_ratio = 100.0

    if len(trades) > 0:
        winning_trades = trades.loc[trades["profit_abs"] > 0]
        losing_trades = trades.loc[trades["profit_abs"] < 0]
        profit_sum = winning_trades["profit_abs"].sum()
        loss_sum = abs(losing_trades["profit_abs"].sum())
        nb_win_trades = len(winning_trades)
        nb_loss_trades = len(losing_trades)

        average_win = (profit_sum / nb_win_trades) if nb_win_trades > 0 else 0
        average_loss = (loss_sum / nb_loss_trades) if nb_loss_trades > 0 else 0
        winrate = nb_win_trades / len(trades)
        loserate = nb_loss_trades / len(trades)

        expectancy = (winrate * average_win) - (loserate * average_loss)
        if average_loss > 0:
            risk_reward_ratio = average_win / average_loss
            expectancy_ratio = ((1 + risk_reward_ratio) * winrate) - 1

    return expectancy, expectancy_ratio


def calculate_sortino(
    trades: pd.DataFrame, min_date: datetime, max_date: datetime, starting_balance: float
) -> float:
    """
    计算索提诺比率
    :param trades: 包含交易的数据帧（需要profit_abs列）
    :return: 索提诺比率
    """
    if (len(trades) == 0) or (min_date is None) or (max_date is None) or (min_date == max_date):
        return 0

    total_profit = trades["profit_abs"] / starting_balance
    days_period = max(1, (max_date - min_date).days)

    expected_returns_mean = total_profit.sum() / days_period

    down_stdev = np.std(trades.loc[trades["profit_abs"] < 0, "profit_abs"] / starting_balance)

    if down_stdev != 0 and not np.isnan(down_stdev):
        sortino_ratio = expected_returns_mean / down_stdev * np.sqrt(365)
    else:
        # 定义较高（负）的索提诺比率以明确表示这不是最佳的。
        sortino_ratio = -100

    # print(expected_returns_mean, down_stdev, sortino_ratio)
    return sortino_ratio


def calculate_sharpe(
    trades: pd.DataFrame, min_date: datetime, max_date: datetime, starting_balance: float
) -> float:
    """
    计算夏普比率
    :param trades: 包含交易的数据帧（需要profit_abs列）
    :return: 夏普比率
    """
    if (len(trades) == 0) or (min_date is None) or (max_date is None) or (min_date == max_date):
        return 0

    total_profit = trades["profit_abs"] / starting_balance
    days_period = max(1, (max_date - min_date).days)

    expected_returns_mean = total_profit.sum() / days_period
    up_stdev = np.std(total_profit)

    if up_stdev != 0:
        sharp_ratio = expected_returns_mean / up_stdev * np.sqrt(365)
    else:
        # 定义较高（负）的夏普比率以明确表示这不是最佳的。
        sharp_ratio = -100

    # print(expected_returns_mean, up_stdev, sharp_ratio)
    return sharp_ratio


def calculate_calmar(
    trades: pd.DataFrame, min_date: datetime, max_date: datetime, starting_balance: float
) -> float:
    """
    计算卡尔玛比率
    :param trades: 包含交易的数据帧（需要close_date和profit_abs列）
    :return: 卡尔玛比率
    """
    if (len(trades) == 0) or (min_date is None) or (max_date is None) or (min_date == max_date):
        return 0

    total_profit = trades["profit_abs"].sum() / starting_balance
    days_period = max(1, (max_date - min_date).days)

    # 添加每次交易0.1%的滑点
    # total_profit = total_profit - 0.0005
    expected_returns_mean = total_profit / days_period * 100

    # 计算最大回撤
    try:
        drawdown = calculate_max_drawdown(
            trades, value_col="profit_abs", starting_balance=starting_balance
        )
        max_drawdown = drawdown.relative_account_drawdown
    except ValueError:
        max_drawdown = 0

    if max_drawdown != 0:
        calmar_ratio = expected_returns_mean / max_drawdown * math.sqrt(365)
    else:
        # 定义较高（负）的卡尔玛比率以明确表示这不是最佳的。
        calmar_ratio = -100

    # print(expected_returns_mean, max_drawdown, calmar_ratio)
    return calmar_ratio


def calculate_sqn(trades: pd.DataFrame, starting_balance: float) -> float:
    """
    计算系统质量数(SQN) - Van K. Tharp。
    SQN衡量系统性交易质量，并同时考虑交易数量及其标准差。

    :param trades: 包含交易的数据帧（需要profit_abs列）
    :param starting_balance: 交易系统的起始资金
    :return: SQN值
    """
    if len(trades) == 0:
        return 0.0

    total_profit = trades["profit_abs"] / starting_balance
    number_of_trades = len(trades)

    # 计算平均交易和标准差
    average_profits = total_profit.mean()
    profits_std = total_profit.std()

    if profits_std != 0 and not np.isnan(profits_std):
        sqn = math.sqrt(number_of_trades) * (average_profits / profits_std)
    else:
        # 定义负的SQN以表明这不是最佳的
        sqn = -100.0

    return round(sqn, 4)