"""
夏普比率超参数优化损失
该模块定义了可用于超参数优化的替代HyperOptLoss类。
"""

from datetime import datetime

from pandas import DataFrame

from freqtrade.data.metrics import calculate_sharpe
from freqtrade.optimize.hyperopt import IHyperOptLoss


class SharpeHyperOptLoss(IHyperOptLoss):
    """
    定义超参数优化的损失函数。
    此实现使用夏普比率计算。
    """

    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        min_date: datetime,
        max_date: datetime,
        starting_balance: float,
        *args,
        **kwargs,
    ) -> float:
        """
        目标函数，返回较小的数值表示结果更优。
        使用夏普比率计算。
        """
        sharp_ratio = calculate_sharpe(results, min_date, max_date, starting_balance)
        # print(expected_returns_mean, up_stdev, sharp_ratio)
        return -sharp_ratio