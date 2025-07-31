"""
最大回撤超参数优化损失函数（MaxDrawDownHyperOptLoss）

该模块定义了可用于超参数优化的替代HyperOptLoss类。
"""

from datetime import datetime

from pandas import DataFrame

from freqtrade.data.metrics import calculate_max_drawdown
from freqtrade.optimize.hyperopt import IHyperOptLoss


class MaxDrawDownHyperOptLoss(IHyperOptLoss):
    """
    定义超参数优化的损失函数。

    此实现优化最大回撤和收益，较小的最大回撤和较高的收益对应较低的返回值。
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
        目标函数。

        当存在回撤时，使用利润比率加权的最大回撤；否则直接优化利润比率。
        """
        total_profit = results["profit_abs"].sum()
        try:
            max_drawdown = calculate_max_drawdown(results, value_col="profit_abs")
        except ValueError:
            # 没有亏损交易，因此没有回撤。
            return -total_profit
        return -total_profit / max_drawdown.drawdown_abs