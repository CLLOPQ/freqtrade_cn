"""
多指标HyperOpt损失

该模块基于以下指标定义了替代的HyperOptLoss类：
  - 利润
  - 回撤
  - 盈亏比
  - 期望比率
  - 胜率
  - 交易数量

可调整的参数：
  - `DRAWDOWN_MULT`：根据个人需求调整对回撤目标的惩罚程度；
  - `TARGET_TRADE_AMOUNT`：调整交易数量的影响程度。
  - `EXPECTANCY_CONST`：调整期望比率的影响程度。
  - `PF_CONST`：调整盈亏比的影响程度。
  - `WINRATE_CONST`：调整胜率的影响程度。


hyperoptloss文件中的DRAWDOWN_MULT变量可根据回撤目的调整为更严格或更灵活。数值越小，对回撤的惩罚越严厉。
PF_CONST变量调整盈亏比对优化的影响程度。
EXPECTANCY_CONST变量控制期望比率的影响。
WINRATE_CONST变量可调整以增加或减少胜率的影响。

PF_CONST、EXPECTANCY_CONST、WINRATE_CONST的作用方式类似：
        较高的值意味着该指标对目标的影响较小，
        而较低的值意味着该指标对目标的影响较大。
TARGET_TRADE_AMOUNT变量设置避免惩罚所需的最小交易数量。
            如果交易数量低于此阈值，则会应用惩罚。
"""

import numpy as np
from pandas import DataFrame

from freqtrade.data.metrics import calculate_expectancy, calculate_max_drawdown
from freqtrade.optimize.hyperopt import IHyperOptLoss


# 较小的数值会更严厉地惩罚回撤
DRAWDOWN_MULT = 0.055
# 用作无穷大替代值的非常大的数
LARGE_NUMBER = 1e6
# 目标交易数量，若交易数量高于此值则不施加惩罚
TARGET_TRADE_AMOUNT = 50
# 调整期望比率影响的系数
EXPECTANCY_CONST = 2.0
# 调整盈亏比影响的系数
PF_CONST = 1.0
# 调整胜率影响的系数
WINRATE_CONST = 1.2


class MultiMetricHyperOptLoss(IHyperOptLoss):
    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        trade_count: int,
        starting_balance: float,
        **kwargs,
    ) -> float:
        total_profit = results["profit_abs"].sum()

        # 计算盈亏比
        winning_profit = results.loc[results["profit_abs"] > 0, "profit_abs"].sum()
        losing_profit = results.loc[results["profit_abs"] < 0, "profit_abs"].sum()
        profit_factor = winning_profit / (abs(losing_profit) + 1e-6)
        log_profit_factor = np.log(profit_factor + PF_CONST)

        # 计算期望比率
        _, expectancy_ratio = calculate_expectancy(results)
        log_expectancy_ratio = np.log(min(10, expectancy_ratio) + EXPECTANCY_CONST)

        # 计算胜率
        winning_trades = results.loc[results["profit_abs"] > 0]
        winrate = len(winning_trades) / len(results)
        log_winrate_coef = np.log(WINRATE_CONST + winrate)

        # 计算回撤
        try:
            drawdown = calculate_max_drawdown(
                results, starting_balance=starting_balance, value_col="profit_abs"
            )
            relative_account_drawdown = drawdown.relative_account_drawdown
        except ValueError:
            relative_account_drawdown = 0

        # 交易数量惩罚
        trade_count_penalty = 1.0  # 默认：无惩罚
        if trade_count < TARGET_TRADE_AMOUNT:
            trade_count_penalty = 1 - (abs(trade_count - TARGET_TRADE_AMOUNT) / TARGET_TRADE_AMOUNT)
            trade_count_penalty = max(trade_count_penalty, 0.1)

        profit_draw_function = total_profit - (relative_account_drawdown * total_profit) * (
            1 - DRAWDOWN_MULT
        )

        return -1 * (
            profit_draw_function
            * log_profit_factor
            * log_expectancy_ratio
            * log_winrate_coef
            * trade_count_penalty
        )