# pragma pylint: disable=attribute-defined-outside-init

"""
此模块为freqai加载自定义模型
"""

import logging
from pathlib import Path

from freqtrade.constants import USERPATH_FREQAIMODELS, Config
from freqtrade.exceptions import OperationalException
from freqtrade.freqai.freqai_interface import IFreqaiModel
from freqtrade.resolvers import IResolver


logger = logging.getLogger(__name__)


class FreqaiModelResolver(IResolver):
    """
    此类包含加载自定义超参数优化损失类的所有逻辑
    """

    object_type = IFreqaiModel
    object_type_str = "FreqAI模型"
    user_subdir = USERPATH_FREQAIMODELS
    initial_search_path = (
        Path(__file__).parent.parent.joinpath("freqai/prediction_models").resolve()
    )
    extra_path = "freqai模型路径"

    @staticmethod
    def load_freqaimodel(config: Config) -> IFreqaiModel:
        """
        从配置参数加载自定义类
        :param config: 配置字典
        """
        disallowed_models = ["BaseRegressionModel"]

        freqaimodel_name = config.get("freqaimodel")
        if not freqaimodel_name:
            raise OperationalException(
                "未设置freqai模型。请使用`--freqaimodel`指定要使用的FreqAI模型类。\n"
            )
        if freqaimodel_name in disallowed_models:
            raise OperationalException(
                f"{freqaimodel_name}是基类，不能直接使用。请选择现有的子类或继承自此类。\n"
            )
        freqaimodel = FreqaiModelResolver.load_object(
            freqaimodel_name,
            config,
            kwargs={"config": config},
        )

        return freqaimodel