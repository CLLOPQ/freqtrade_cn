"""
IHyperOptLoss 接口
此模块定义了超参数优化的损失函数接口
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pandas import DataFrame

from freqtrade.constants import Config


class IHyperOptLoss(ABC):
    """
    Freqtrade超参数优化损失函数的接口。
    定义了自定义损失函数（`hyperopt_loss_function()`，该函数在每个周期进行评估。）
    """

    timeframe: str

    @staticmethod
    @abstractmethod
    def hyperopt_loss_function(
        *,
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: Config,
        processed: dict[str, DataFrame],
        backtest_stats: dict[str, Any],
        starting_balance: float,
        **kwargs,
    ) -> float:
        """
        目标函数，返回较小的数值表示结果更好
        """