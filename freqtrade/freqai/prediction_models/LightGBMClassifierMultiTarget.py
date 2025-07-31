"""LightGBM多目标分类器模型"""

import logging
from typing import Any

from lightgbm import LGBMClassifier

from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.base_models.FreqaiMultiOutputClassifier import FreqaiMultiOutputClassifier
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)


class LightGBMClassifierMultiMultiTarget(BaseClassifierModel):
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

        lgb = LGBMClassifier(** self.model_training_parameters)

        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        sample_weight = data_dictionary["train_weights"]

        eval_weights = None
        eval_sets = [None] * y.shape[1]

        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) != 0:
            eval_weights = [data_dictionary["test_weights"]]
            eval_sets = [(None, None)] * data_dictionary["test_labels"].shape[1]  # type: ignore
            for i in range(data_dictionary["test_labels"].shape[1]):
                eval_sets[i] = (  # type: ignore
                    data_dictionary["test_features"],
                    data_dictionary["test_labels"].iloc[:, i],
                )

        init_model = self.get_init_model(dk.pair)
        if init_model:
            init_models = init_model.estimators_
        else:
            init_models = [None] * y.shape[1]

        fit_params = []
        for i in range(len(eval_sets)):
            fit_params.append(
                {
                    "eval_set": eval_sets[i],
                    "eval_sample_weight": eval_weights,
                    "init_model": init_models[i],
                }
            )

        model = FreqaiMultiOutputClassifier(estimator=lgb)
        thread_training = self.freqai_info.get("multitarget_parallel_training", False)
        if thread_training:
            model.n_jobs = y.shape[1]
        model.fit(X=X, y=y, sample_weight=sample_weight, fit_params=fit_params)

        return model