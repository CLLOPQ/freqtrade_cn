"""PyTorch分类器基类"""

import logging
from time import time
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from pandas import DataFrame
from torch.nn import functional as F

from freqtrade.exceptions import OperationalException
from freqtrade.freqai.base_models.BasePyTorchModel import BasePyTorchModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)


class BasePyTorchClassifier(BasePyTorchModel):
    """
    分类器的PyTorch实现。
    用户必须实现fit方法

    重要提示!

    - 用户必须在策略中声明目标类名称，
    在IStrategy.set_freqai_targets方法中。

    例如，在你的策略中:
    ```
        def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs):
            self.freqai.class_names = ["down", "up"]
            dataframe['&s-up_or_down'] = np.where(dataframe["close"].shift(-100) >
                                                  dataframe["close"], 'up', 'down')

            return dataframe
    ```
    """

    def __init__(self,** kwargs):
        super().__init__(**kwargs)
        self.class_name_to_index = {}  # 类名到索引的映射
        self.index_to_class_name = {}  # 索引到类名的映射

    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> tuple[DataFrame, npt.NDArray[np.int_]]:
        """
        过滤预测特征数据并进行预测。
        :param dk: 数据处理工具类实例
        :param unfiltered_df: 当前回测周期的完整数据帧
        :return:
        :pred_df: 包含预测结果的数据帧
        :do_predict: 由1和0组成的np数组，表示Freqai需要移除的数据点（NaN）
                    或对数据不确定的地方（PCA和DI指数）
        :raises ValueError: 如果模型元数据中不存在'class_names'
        """

        class_names = self.model.model_meta_data.get("class_names", None)
        if not class_names:
            raise ValueError(
                "缺少类名称。self.model.model_meta_data['class_names']为None。"
            )

        if not self.class_name_to_index:
            self.init_class_names_to_index_mapping(class_names)

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
        logits = self.model.model(x)
        probs = F.softmax(logits, dim=-1)
        predicted_classes = torch.argmax(probs, dim=-1)
        predicted_classes_str = self.decode_class_names(predicted_classes)
        # 使用.tolist()将概率转换为可迭代对象，这样张量
        # 会自动移至CPU（如有必要）
        pred_df_prob = DataFrame(probs.detach().tolist(), columns=class_names)
        pred_df = DataFrame(predicted_classes_str, columns=[dk.label_list[0]])
        pred_df = pd.concat([pred_df, pred_df_prob], axis=1)

        if dk.feature_pipeline["di"]:
            dk.DI_values = dk.feature_pipeline["di"].di_values
        else:
            dk.DI_values = np.zeros(outliers.shape[0])
        dk.do_predict = outliers

        return (pred_df, dk.do_predict)

    def encode_class_names(
        self,
        data_dictionary: dict[str, pd.DataFrame],
        dk: FreqaiDataKitchen,
        class_names: list[str],
    ):
        """
        编码类名称，字符串 -> 整数
        假设*_labels数据帧的第一列是包含类名称的目标列
        """

        target_column_name = dk.label_list[0]
        for split in self.splits:
            label_df = data_dictionary[f"{split}_labels"]
            self.assert_valid_class_names(label_df[target_column_name], class_names)
            label_df[target_column_name] = list(
                map(lambda x: self.class_name_to_index[x], label_df[target_column_name])
            )

    @staticmethod
    def assert_valid_class_names(target_column: pd.Series, class_names: list[str]):
        """验证目标列中的类名称是否都在定义的类列表中"""
        non_defined_labels = set(target_column) - set(class_names)
        if len(non_defined_labels) != 0:
            raise OperationalException(
                f"发现未定义的标签: {non_defined_labels}, ",
                f"预期的标签: {class_names}",
            )

    def decode_class_names(self, class_ints: torch.Tensor) -> list[str]:
        """
        解码类名称，整数 -> 字符串
        """

        return list(map(lambda x: self.index_to_class_name[x.item()], class_ints))

    def init_class_names_to_index_mapping(self, class_names):
        """初始化类名称到索引的映射"""
        self.class_name_to_index = {s: i for i, s in enumerate(class_names)}
        self.index_to_class_name = {i: s for i, s in enumerate(class_names)}
        logger.info(f"编码的类名称到索引的映射: {self.class_name_to_index}")

    def convert_label_column_to_int(
        self,
        data_dictionary: dict[str, pd.DataFrame],
        dk: FreqaiDataKitchen,
        class_names: list[str],
    ):
        """将标签列转换为整数"""
        self.init_class_names_to_index_mapping(class_names)
        self.encode_class_names(data_dictionary, dk, class_names)

    def get_class_names(self) -> list[str]:
        """获取类名称列表"""
        if not self.class_names:
            raise ValueError(
                "self.class_names为空，"
                "请在IStrategy.set_freqai_targets方法中设置self.freqai.class_names = ['class a', 'class b', 'class c']。"
            )

        return self.class_names

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
        end_time = time()

        logger.info(
            f"-------------------- {pair} 训练完成 "
            f"({end_time - start_time:.2f} 秒) --------------------"
        )

        return model