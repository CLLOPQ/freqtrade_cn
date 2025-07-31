"""
最大回撤相对超参数优化损失函数（MaxDrawDownRelativeHyperOptLoss）

该模块定义了可用于超参数优化的替代HyperOptLoss类。
"""

from pandas import DataFrame

from freqtrade.data.metrics import calculate_underwater
from freqtrade.optimize.hyperopt import IHyperOptLoss


class MaxDrawDownRelativeHyperOptLoss(IHyperOptLoss):
    """
    定义超参数优化的损失函数。

    此实现优化最大回撤和收益，目标是：最大回撤越小、收益越高，返回值越低。
    """

    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame, starting_balance: float, *args, **kwargs
    ) -> float:
        """
        目标函数。

        当存在回撤数据时，使用收益比率加权的最大回撤；否则直接优化收益比率。
        """
        total_profit = results["profit_abs"].sum()
        try:
            drawdown_df = calculate_underwater(
                results, value_col="profit_abs", starting_balance=starting_balance
            )
            max_drawdown = abs(min(drawdown_df["drawdown"]))
            relative_drawdown = max(drawdown_df["drawdown_relative"])
            if max_drawdown == 0:
                return -total_profit
            return -total_profit / max_drawdown / relative_drawdown
        except (Exception, ValueError):
            return -total_profit