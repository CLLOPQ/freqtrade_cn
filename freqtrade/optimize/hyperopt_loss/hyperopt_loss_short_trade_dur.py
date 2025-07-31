"""
ShortTradeDurHyperOptLoss 该模块定义了默认的HyperoptLoss类，用于超参数优化。
"""

from math import exp

from pandas import DataFrame

from freqtrade.optimize.hyperopt import IHyperOptLoss


# 将TARGET_TRADES设置为适合您的并发交易数量，使其与天数相匹配
TARGET_TRADES = 600

# 这假设为预期平均利润乘以预期交易数量。例如，每笔交易平均0.35%（或比率0.0035）和1100笔交易，
# 预期最大利润=3.85 检查报告的Σ%值是否不超过此值！注意，这是比率。上述3.85表示385Σ%。
EXPECTED_MAX_PROFIT = 3.0

# 最大平均交易持续时间（分钟）。如果评估结束时该值更高，则认为评估失败。
MAX_ACCEPTED_TRADE_DURATION = 300


class ShortTradeDurHyperOptLoss(IHyperOptLoss):
    """定义超参数优化的默认损失函数"""

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int, *args, **kwargs) -> float:
        """
        目标函数，返回较小的数值表示结果更好 这是默认算法 权重分布如下：
        * 0.4 用于交易持续时间 * 0.25：避免交易亏损 * 1.0 用于总利润，与上述定义的预期值（`EXPECTED_MAX_PROFIT`）相比
        """
        total_profit = results["profit_ratio"].sum()
        trade_duration = results["trade_duration"].mean()

        trade_loss = 1 - 0.25 * exp(-((trade_count - TARGET_TRADES) ** 2) / 10**5.8)
        profit_loss = max(0, 1 - total_profit / EXPECTED_MAX_PROFIT)
        duration_loss = 0.4 * min(trade_duration / MAX_ACCEPTED_TRADE_DURATION, 1)
        result = trade_loss + profit_loss + duration_loss
        return result


# 创建该类的别名，以确保旧方法也能正常工作。
class DefaultHyperOptLoss(ShortTradeDurHyperOptLoss):
    pass