"""
夏普比率超参数优化损失（每日）

此模块定义了可用于超参数优化的替代HyperOptLoss类。
"""

import math
from datetime import datetime

from pandas import DataFrame, date_range

from freqtrade.optimize.hyperopt import IHyperOptLoss


class SharpeHyperOptLossDaily(IHyperOptLoss):
    """
    定义超参数优化的损失函数。

    此实现使用夏普比率计算。
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
        目标函数，返回较小的数值表示更优的结果。

        使用夏普比率计算。
        """
        resample_freq = "1D"
        slippage_per_trade_ratio = 0.0005
        days_in_year = 365
        annual_risk_free_rate = 0.0
        risk_free_rate = annual_risk_free_rate / days_in_year

        # 对每笔交易的利润比率应用滑点
        results.loc[:, "profit_ratio_after_slippage"] = (
            results["profit_ratio"] - slippage_per_trade_ratio
        )

        # 在min_date和max_date之间创建索引
        t_index = date_range(start=min_date, end=max_date, freq=resample_freq, normalize=True)

        sum_daily = (
            results.resample(resample_freq, on="close_date")
            .agg({"profit_ratio_after_slippage": "sum"})
            .reindex(t_index)
            .fillna(0)
        )

        total_profit = sum_daily["profit_ratio_after_slippage"] - risk_free_rate
        expected_returns_mean = total_profit.mean()
        up_stdev = total_profit.std()

        if up_stdev != 0:
            sharp_ratio = expected_returns_mean / up_stdev * math.sqrt(days_in_year)
        else:
            # 定义高（负）夏普比率以明确这不是最优解
            sharp_ratio = -20.0

        # print(t_index, sum_daily, total_profit)
        # print(risk_free_rate, expected_returns_mean, up_stdev, sharp_ratio)
        return -sharp_ratio