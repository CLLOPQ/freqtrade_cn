"""
每对最大回撤超参数损失 此模块定义了可用于超参数优化的替代 HyperOptLoss 类。
"""

from typing import Any

from freqtrade.optimize.hyperopt import IHyperOptLoss


class MaxDrawDownPerPairHyperOptLoss(IHyperOptLoss):
    """
    定义超参数优化的损失函数。此实现计算每对的利润/回撤比率，并将最差结果作为目标返回，
    迫使超参数优化器为交易对列表中的所有交易对优化参数。通过这种方式，我们防止一个或多个
    表现良好的交易对夸大指标，而其余表现不佳的交易对未被代表，因此未被优化。
    """

    @staticmethod
    def hyperopt_loss_function(backtest_stats: dict[str, Any], *args, **kwargs) -> float:
        """
        目标函数，返回较小的数值表示结果更好。
        """

        ##############################################
        # 可配置参数
        ##############################################
        # 每对可接受的最小利润/回撤
        min_acceptable_profit_dd = 1.0
        # 未达到可接受最小值时的惩罚值
        penalty = 20
        ##############################################

        score_per_pair = []
        for p in backtest_stats["results_per_pair"]:
            if p["key"] != "TOTAL":
                profit = p.get("profit_total_abs", 0)
                drawdown = p.get("max_drawdown_abs", 0)

                if drawdown != 0 and profit != 0:
                    profit_dd = profit / drawdown
                else:
                    profit_dd = profit

                if profit_dd < min_acceptable_profit_dd:
                    score = profit_dd - penalty
                else:
                    score = profit_dd

                score_per_pair.append(score)

        return -min(score_per_pair)