"""XGBoost多目标回归器模型"""

import logging
from typing import Any

from xgboost import XGBRegressor

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.tensorboard import TBCallback


logger = logging.getLogger(__name__)


class XGBoostRegressorMultiTarget(BaseRegressionModel):
    """
    用户创建的预测模型。该类继承自IFreqaiModel，
    这意味着它可以完全访问所有Frequency AI功能。通常，
    用户会使用它来重写通用的`fit()`、`train()`或
    `predict()`方法，以添加自定义数据处理工具或更改
    无法通过顶级config.json文件配置的训练的各个方面。
    这是XGBoostRegressor的精确副本，为了兼容性而保留。
    """

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        用户在此处设置训练和测试数据以拟合其所需的模型
        :param data_dictionary: 包含所有训练、测试、标签、权重数据的字典
        :param dk: 当前币种/模型的数据处理工具对象
        """

        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]

        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) == 0:
            eval_set = None
            eval_weights = None
        else:
            eval_set = [(data_dictionary["test_features"], data_dictionary["test_labels"]), (X, y)]
            eval_weights = [data_dictionary["test_weights"], data_dictionary["train_weights"]]

        sample_weight = data_dictionary["train_weights"]

        xgb_model = self.get_init_model(dk.pair)

        model = XGBRegressor(** self.model_training_parameters)

        model.set_params(callbacks=[TBCallback(dk.data_path)])
        model.fit(
            X=X,
            y=y,
            sample_weight=sample_weight,
            eval_set=eval_set,
            sample_weight_eval_set=eval_weights,
            xgb_model=xgb_model,
        )
        # 将回调设置为空，以便稍后可以序列化到磁盘
        model.set_params(callbacks=[])

        return model