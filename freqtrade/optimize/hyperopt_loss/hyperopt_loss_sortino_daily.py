"""
SortinoHyperOptLossDaily

此模块定义了可用于超参数优化的替代 HyperOptLoss 类。
"""

import math
from datetime import datetime

from pandas import DataFrame, date_range

from freqtrade.optimize.hyperopt import IHyperOptLoss


class SortinoHyperOptLossDaily(IHyperOptLoss):
    """
    定义超参数优化的损失函数。

    此实现使用 Sortino 比率计算。
    """

    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        *args,
        **kwargs,
    ) -> float:
        """
        目标函数，返回更小的数值表示更优的结果。

        使用 Sortino 比率计算。

        Sortino 比率的计算方法如
        http://www.redrockcapital.com/Sortino__A__Sharper__Ratio_Red_Rock_Capital.pdf 中所述
        """
        resample_freq = "1D"
        slippage_per_trade_ratio = 0.0005
        days_in_year = 365
        minimum_acceptable_return = 0.0

        # 将每笔交易的滑点应用到利润比率
        results.loc[:, "profit_ratio_after_slippage"] = (
            results["profit_ratio"] - slippage_per_trade_ratio
        )

        # 在 min_date 和 max_date 之间创建索引
        t_index = date_range(start=min_date, end=max_date, freq=resample_freq, normalize=True)

        sum_daily = (
            results.resample(resample_freq, on="close_date")
            .agg({"profit_ratio_after_slippage": "sum"})
            .reindex(t_index)
            .fillna(0)
        )

        total_profit = sum_daily["profit_ratio_after_slippage"] - minimum_acceptable_return
        expected_returns_mean = total_profit.mean()

        sum_daily["downside_returns"] = 0.0
        sum_daily.loc[total_profit < 0, "downside_returns"] = total_profit
        total_downside = sum_daily["downside_returns"]
        # 这里 total_downside 包含 min(0, P - MAR) 值，其中 P = sum_daily["profit_ratio_after_slippage"]
        down_stdev = math.sqrt((total_downside**2).sum() / len(total_downside))

        if down_stdev != 0:
            sortino_ratio = expected_returns_mean / down_stdev * math.sqrt(days_in_year)
        else:
            # 定义高（负）的索丁诺比率以明确这不是最优的。
            sortino_ratio = -20.0

        # 打印(t_index, sum_daily, total_profit)
        # 打印(minimum_acceptable_return, expected_returns_mean, down_stdev, sortino_ratio)
        return -sortino_ratio