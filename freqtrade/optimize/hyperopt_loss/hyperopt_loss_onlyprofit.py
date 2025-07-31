"""
OnlyProfitHyperOptLoss

此模块定义了可用于超参数优化的替代HyperOptLoss类。
"""

from pandas import DataFrame

from freqtrade.optimize.hyperopt import IHyperOptLoss


class OnlyProfitHyperOptLoss(IHyperOptLoss):
    """
    定义超参数优化的损失函数。

    此实现仅考虑绝对收益，不查看任何其他指标。
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int, *args, **kwargs) -> float:
        """
        目标函数，返回较小的数值表示结果更好。
        """
        total_profit = results["profit_abs"].sum()
        return -1 * total_profit