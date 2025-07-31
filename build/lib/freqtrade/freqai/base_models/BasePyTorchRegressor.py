"""PyTorch回归器基类"""

import logging
from time import time
from typing import Any

import numpy as np
import numpy.typing as npt
from pandas import DataFrame

from freqtrade.freqai.base_models.BasePyTorchModel import BasePyTorchModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)


class BasePyTorchRegressor(BasePyTorchModel):
    """
    回归器的PyTorch实现。
    用户必须实现fit方法
    """

    def __init__(self,** kwargs):
        super().__init__(**kwargs)

    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> tuple[DataFrame, npt.NDArray[np.int_]]:
        """
        过滤预测特征数据并进行预测。
        :param unfiltered_df: 当前回测周期的完整数据帧
        :return:
        :pred_df: 包含预测结果的数据帧
        :do_predict: 由1和0组成的np数组，表示Freqai需要移除的数据点（NaN）
                    或对数据不确定的地方（PCA和DI指数）
        """

        dk.find_features(unfiltered_df)
        filtered_df, _ = dk.filter_features(
            unfiltered_df, dk.training_features_list, training_filter=False
        )
        dk.data_dictionary["prediction_features"] = filtered_df

        dk.data_dictionary["prediction_features"], outliers, _ = dk.feature_pipeline.transform(
            dk.data_dictionary["prediction_features"], outlier_check=True
        )

        x = self.data_convertor.convert_x(
            dk.data_dictionary["prediction_features"], device=self.device
        )
        self.model.model.eval()
        y = self.model.model(x)
        pred_df = DataFrame(y.detach().tolist(), columns=[dk.label_list[0]])
        pred_df, _, _ = dk.label_pipeline.inverse_transform(pred_df)

        if dk.feature_pipeline["di"]:
            dk.DI_values = dk.feature_pipeline["di"].di_values
        else:
            dk.DI_values = np.zeros(outliers.shape[0])
        dk.do_predict = outliers
        return (pred_df, dk.do_predict)

    def train(self, unfiltered_df: DataFrame, pair: str, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        过滤训练数据并训练模型。训练过程大量使用datakitchen
        来存储、保存、加载和分析数据。
        :param unfiltered_df: 当前训练周期的完整数据帧
        :return:
        :model: 训练好的模型，可用于推理（self.predict）
        """

        logger.info(f"-------------------- 开始训练 {pair} --------------------")

        start_time = time()

        features_filtered, labels_filtered = dk.filter_features(
            unfiltered_df,
            dk.training_features_list,
            dk.label_list,
            training_filter=True,
        )

        # 将数据拆分为训练/测试数据集
        dd = dk.make_train_test_datasets(features_filtered, labels_filtered)
        if not self.freqai_info.get("fit_live_predictions_candles", 0) or not self.live:
            dk.fit_labels()
        dk.feature_pipeline = self.define_data_pipeline(threads=dk.thread_count)
        dk.label_pipeline = self.define_label_pipeline(threads=dk.thread_count)

        (dd["train_features"], dd["train_labels"], dd["train_weights"]) = (
            dk.feature_pipeline.fit_transform(
                dd["train_features"], dd["train_labels"], dd["train_weights"]
            )
        )
        dd["train_labels"], _, _ = dk.label_pipeline.fit_transform(dd["train_labels"])

        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) != 0:
            (dd["test_features"], dd["test_labels"], dd["test_weights"]) = (
                dk.feature_pipeline.transform(
                    dd["test_features"], dd["test_labels"], dd["test_weights"]
                )
            )
            dd["test_labels"], _, _ = dk.label_pipeline.transform(dd["test_labels"])

        logger.info(
            f"使用 {len(dk.data_dictionary['train_features'].columns)} 个特征训练模型"
        )
        logger.info(f"使用 {len(dd['train_features'])} 个数据点训练模型")

        model = self.fit(dd, dk)
        end_time = time()

        logger.info(
            f"-------------------- {pair} 训练完成 "
            f"({end_time - start_time:.2f} 秒) --------------------"
        )

        return model