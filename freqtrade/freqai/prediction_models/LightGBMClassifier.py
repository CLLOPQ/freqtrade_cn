"""LightGBM分类器模型"""

import logging
from typing import Any

from lightgbm import LGBMClassifier

from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)


class LightGBMClassifier(BaseClassifierModel):
    """
    用户创建的预测模型。该类继承自IFreqaiModel，
    这意味着它可以完全访问所有Frequency AI功能。通常，
    用户会使用它来重写通用的`fit()`、`train()`或
    `predict()`方法，以添加自定义数据处理工具或更改
    无法通过顶级config.json文件配置的训练的各个方面。
    """

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        用户在此处设置训练和测试数据以拟合其所需的模型
        :param data_dictionary: 包含所有训练、测试、标签、权重数据的字典
        :param dk: 当前币种/模型的数据处理工具对象
        """

        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) == 0:
            eval_set = None
            test_weights = None
        else:
            eval_set = [
                (
                    data_dictionary["test_features"].to_numpy(),
                    data_dictionary["test_labels"].to_numpy()[:, 0],
                )
            ]
            test_weights = data_dictionary["test_weights"]
        X = data_dictionary["train_features"].to_numpy()
        y = data_dictionary["train_labels"].to_numpy()[:, 0]
        train_weights = data_dictionary["train_weights"]

        init_model = self.get_init_model(dk.pair)

        model = LGBMClassifier(** self.model_training_parameters)
        model.fit(
            X=X,
            y=y,
            eval_set=eval_set,
            sample_weight=train_weights,
            eval_sample_weight=[test_weights],
            init_model=init_model,
        )

        return model