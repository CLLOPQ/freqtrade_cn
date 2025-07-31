"""
ProfitDrawDownHyperOptLoss
该模块定义了基于利润和回撤目标的替代 HyperOptLoss 类，可用于超参数优化。
可以通过修改 `DRAWDOWN_MULT` 来根据个人需求调整对回撤目标的惩罚力度。
"""

from pandas import DataFrame

from freqtrade.data.metrics import calculate_max_drawdown
from freqtrade.optimize.hyperopt import IHyperOptLoss


# 数值越小，对回撤的惩罚越严厉
DRAWDOWN_MULT = 0.075


class ProfitDrawDownHyperOptLoss(IHyperOptLoss):
    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame, starting_balance: float, *args, **kwargs
    ) -> float:
        total_profit = results["绝对利润"].sum()

        try:
            drawdown = calculate_max_drawdown(
                results, starting_balance=starting_balance, value_col="绝对利润"
            )
            relative_account_drawdown = drawdown.relative_account_drawdown
        except ValueError:
            relative_account_drawdown = 0

        return -1 * (
            total_profit - (relative_account_drawdown * total_profit) * (1 - DRAWDOWN_MULT)
        )