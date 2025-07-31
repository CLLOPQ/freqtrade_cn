"""基础分类器模型类"""

import logging
from time import time
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel


logger = logging.getLogger(__name__)


class BaseClassifierModel(IFreqaiModel):
    """
    回归类型模型的基类（例如Catboost、LightGBM、XGboost等）。
    用户*必须*继承此类并实现fit()方法。参考示例脚本
    如prediction_models/CatboostClassifier.py。
    """

    def train(self, unfiltered_df: DataFrame, pair: str, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        过滤训练数据并训练模型。训练过程大量使用datakitchen
        来存储、保存、加载和分析数据。
        :param unfiltered_df: 当前训练周期的完整数据帧
        :param pair: 交易对
        :param dk: 数据处理工具类实例
        :return: 训练好的模型，可用于推理（self.predict）
        """

        logger.info(f"-------------------- 开始训练 {pair} --------------------")

        开始时间 = time()

        # 过滤配置文件中用户请求的特征，并优雅地处理NaN值
        features_filtered, labels_filtered = dk.filter_features(
            unfiltered_df,
            dk.training_features_list,
            dk.label_list,
            training_filter=True,
        )

        开始日期 = unfiltered_df["date"].iloc[0].strftime("%Y-%m-%d")
        结束日期 = unfiltered_df["date"].iloc[-1].strftime("%Y-%m-%d")
        logger.info(
            f"-------------------- 训练数据时间范围: {开始日期} 至 "
            f"{结束日期} --------------------"
        )
        # 将数据拆分为训练/测试数据集
        dd = dk.make_train_test_datasets(features_filtered, labels_filtered)
        if not self.freqai_info.get("fit_live_predictions_candles", 0) or not self.live:
            dk.fit_labels()
        dk.feature_pipeline = self.define_data_pipeline(threads=dk.thread_count)

        (dd["train_features"], dd["train_labels"], dd["train_weights"]) = (
            dk.feature_pipeline.fit_transform(
                dd["train_features"], dd["train_labels"], dd["train_weights"]
            )
        )

        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) != 0:
            (dd["test_features"], dd["test_labels"], dd["test_weights"]) = (
                dk.feature_pipeline.transform(
                    dd["test_features"], dd["test_labels"], dd["test_weights"]
                )
            )

        logger.info(
            f"使用 {len(dk.data_dictionary['train_features'].columns)} 个特征训练模型"
        )
        logger.info(f"使用 {len(dd['train_features'])} 个数据点训练模型")

        model = self.fit(dd, dk)

        结束时间 = time()

        logger.info(
            f"-------------------- {pair} 训练完成 "
            f"({结束时间 - 开始时间:.2f} 秒) --------------------"
        )

        return model

    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> tuple[DataFrame, npt.NDArray[np.int_]]:
        """
        过滤预测特征数据并进行预测。
        :param unfiltered_df: 当前回测周期的完整数据帧
        :param dk: 数据处理工具类实例
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

        predictions = self.model.predict(dk.data_dictionary["prediction_features"])
        if self.CONV_WIDTH == 1:
            predictions = np.reshape(predictions, (-1, len(dk.label_list)))

        pred_df = DataFrame(predictions, columns=dk.label_list)

        predictions_prob = self.model.predict_proba(dk.data_dictionary["prediction_features"])
        if self.CONV_WIDTH == 1:
            predictions_prob = np.reshape(predictions_prob, (-1, len(self.model.classes_)))
        pred_df_prob = DataFrame(predictions_prob, columns=self.model.classes_)

        pred_df = pd.concat([pred_df, pred_df_prob], axis=1)

        if dk.feature_pipeline["di"]:
            dk.DI_values = dk.feature_pipeline["di"].di_values
        else:
            dk.DI_values = np.zeros(outliers.shape[0])
        dk.do_predict = outliers

        return (pred_df, dk.do_predict)