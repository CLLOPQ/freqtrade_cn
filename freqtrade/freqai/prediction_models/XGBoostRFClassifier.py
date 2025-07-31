"""XGBoost随机森林分类器模型"""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame
from pandas.api.types import is_integer_dtype
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRFClassifier

from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)


class XGBoostRFClassifier(BaseClassifierModel):
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

        X = data_dictionary["train_features"].to_numpy()
        y = data_dictionary["train_labels"].to_numpy()[:, 0]

        le = LabelEncoder()
        if not is_integer_dtype(y):
            y = pd.Series(le.fit_transform(y), dtype="int64")

        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) == 0:
            eval_set = None
        else:
            test_features = data_dictionary["test_features"].to_numpy()
            test_labels = data_dictionary["test_labels"].to_numpy()[:, 0]

            if not is_integer_dtype(test_labels):
                test_labels = pd.Series(le.transform(test_labels), dtype="int64")

            eval_set = [(test_features, test_labels)]

        train_weights = data_dictionary["train_weights"]

        init_model = self.get_init_model(dk.pair)

        model = XGBRFClassifier(** self.model_training_parameters)

        model.fit(X=X, y=y, eval_set=eval_set, sample_weight=train_weights, xgb_model=init_model)

        return model

    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> tuple[DataFrame, npt.NDArray[np.int_]]:
        """
        过滤预测特征数据并进行预测。
        :param  unfiltered_df: 当前回测周期的完整数据帧。
        :return:
        :pred_df: 包含预测结果的数据帧
        :do_predict: 由1和0组成的np数组，表示Freqai需要移除的数据点（NaN）
                    或对数据不确定的地方（PCA和DI指数）
        """

        (pred_df, dk.do_predict) = super().predict(unfiltered_df, dk, **kwargs)

        le = LabelEncoder()
        label = dk.label_list[0]
        labels_before = list(dk.data["labels_std"].keys())
        labels_after = le.fit_transform(labels_before).tolist()
        pred_df[label] = le.inverse_transform(pred_df[label])
        pred_df = pred_df.rename(
            columns={labels_after[i]: labels_before[i] for i in range(len(labels_before))}
        )

        return (pred_df, dk.do_predict)