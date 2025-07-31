from datetime import datetime
from math import exp

from pandas import DataFrame

from freqtrade.constants import Config
from freqtrade.optimize.hyperopt import IHyperOptLoss


# 定义一些常量：

# 将TARGET_TRADES设置为适合您的并发交易数量，使其与天数相符
# 这被认为是预期平均利润乘以预期交易数量。例如，对于每笔交易0.35%的平均利润（或作为比率为0.0035）和1100笔交易，
# self.expected_max_profit = 3.85
# 检查报告的Σ%值是否不超过此值！
# 注意，这是比率。上面提到的3.85表示385Σ%。
TARGET_TRADES = 600
EXPECTED_MAX_PROFIT = 3.0

# （以分钟为单位）最大平均交易时长
# 如果评估结束时（交易时长）更高，我们认为这是一次失败的评估
MAX_ACCEPTED_TRADE_DURATION = 300


class SampleHyperOptLoss(IHyperOptLoss):
    """
    定义超参数优化的默认损失函数
    这旨在为您自己的损失函数提供一些灵感。

    该函数需要返回一个数字（浮点数）——对于更好的回测结果，该数字会变小。
    """

    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: Config,
        processed: dict[str, DataFrame],
        *args,
        **kwargs,
    ) -> float:
        """
        目标函数，返回更小的数字表示更好的结果
        """
        total_profit = results["profit_ratio"].sum()
        trade_duration = results["trade_duration"].mean()

        trade_loss = 1 - 0.25 * exp(-((trade_count - TARGET_TRADES) ** 2) / 10**5.8)
        profit_loss = max(0, 1 - total_profit / EXPECTED_MAX_PROFIT)
        duration_loss = 0.4 * min(trade_duration / MAX_ACCEPTED_TRADE_DURATION, 1)
        result = trade_loss + profit_loss + duration_loss
        return result