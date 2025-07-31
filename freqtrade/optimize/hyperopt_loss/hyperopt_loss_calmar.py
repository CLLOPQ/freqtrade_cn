"""
Calmar超参数优化损失函数
这个模块定义了可用于超参数优化的替代HyperOptLoss类。
"""

from datetime import datetime

from pandas import DataFrame

from freqtrade.data.metrics import calculate_calmar
from freqtrade.optimize.hyperopt import IHyperOptLoss


class CalmarHyperOptLoss(IHyperOptLoss):
    """
    定义超参数优化的损失函数。
    此实现使用Calmar比率计算。
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
        目标函数，返回较小的数值表示更优的结果。
        使用Calmar比率计算。
        """
        calmar_ratio = calculate_calmar(results, min_date, max_date, starting_balance)
        # print(expected_returns_mean, max_drawdown, calmar_ratio)
        return -calmar_ratio