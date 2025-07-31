# pragma pylint: disable=attribute-defined-outside-init

"""
此模块加载自定义超参数优化器
"""

import logging
from pathlib import Path

from freqtrade.constants import HYPEROPT_LOSS_BUILTIN, USERPATH_HYPEROPTS, Config
from freqtrade.exceptions import OperationalException
from freqtrade.optimize.hyperopt_loss.hyperopt_loss_interface import IHyperOptLoss
from freqtrade.resolvers import IResolver


logger = logging.getLogger(__name__)


class HyperOptLossResolver(IResolver):
    """
    此类包含加载自定义超参数优化损失类的所有逻辑
    """

    object_type = IHyperOptLoss
    object_type_str = "超参数优化损失"
    user_subdir = USERPATH_HYPEROPTS
    initial_search_path = Path(__file__).parent.parent.joinpath("optimize/hyperopt_loss").resolve()

    @staticmethod
    def load_hyperoptloss(config: Config) -> IHyperOptLoss:
        """
        从配置参数加载自定义类
        :param config: 配置字典
        """

        hyperoptloss_name = config.get("超参数优化损失")
        if not hyperoptloss_name:
            raise OperationalException(
                "未设置超参数优化损失。请使用 `--hyperopt-loss` 指定要使用的超参数优化损失类。\n"
                f"内置的超参数优化损失函数有：{', '.join(HYPEROPT_LOSS_BUILTIN)}"
            )
        hyperoptloss = HyperOptLossResolver.load_object(
            hyperoptloss_name, config, kwargs={}, extra_dir=config.get("超参数优化路径")
        )

        # 为超参数优化分配时间周期
        hyperoptloss.__class__.timeframe = str(config["timeframe"])

        return hyperoptloss