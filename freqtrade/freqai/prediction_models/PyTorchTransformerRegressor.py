"""PyTorch Transformer回归器"""

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from freqtrade.freqai.base_models.BasePyTorchRegressor import BasePyTorchRegressor
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.torch.PyTorchDataConvertor import (
    DefaultPyTorchDataConvertor,
    PyTorchDataConvertor,
)
from freqtrade.freqai.torch.PyTorchModelTrainer import PyTorchTransformerTrainer
from freqtrade.freqai.torch.PyTorchTransformerModel import PyTorchTransformerModel


class PyTorchTransformerRegressor(BasePyTorchRegressor):
    """
    该类实现了IFreqaiModel的fit方法。
    在fit方法中，我们初始化模型和训练器对象。
    对模型的唯一要求是与PyTorchRegressor的predict方法对齐，
    该方法期望模型预测float类型的张量。
    训练器定义了训练循环。

    参数通过配置文件中freqai部分下的`model_training_parameters`传递。例如：
    {
        ...
        "freqai": {
            ...
            "conv_width": 30,  // PyTorchTransformer基于窗口机制
            "feature_parameters": {
                ...
                "include_shifted_candles": 0,  // 不需要偏移蜡烛图
                ...
            },
            "model_training_parameters" : {
                "learning_rate": 3e-4,
                "trainer_kwargs": {
                    "n_steps": 5000,
                    "batch_size": 64,
                    "n_epochs": null
                },
                "model_kwargs": {
                    "hidden_dim": 512,
                    "dropout_percent": 0.2,
                    "n_layer": 1,
                },
            }
        }
    }
    """

    @property
    def data_convertor(self) -> PyTorchDataConvertor:
        return DefaultPyTorchDataConvertor(target_tensor_type=torch.float)

    def __init__(self,** kwargs) -> None:
        super().__init__(**kwargs)
        config = self.freqai_info.get("model_training_parameters", {})
        self.learning_rate: float = config.get("learning_rate", 3e-4)
        self.model_kwargs: dict[str, Any] = config.get("model_kwargs", {})
        self.trainer_kwargs: dict[str, Any] = config.get("trainer_kwargs", {})

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        用户在此处设置训练和测试数据以拟合其所需的模型
        :param data_dictionary: 包含所有训练、测试、标签、权重数据的字典
        :param dk: 当前币种/模型的数据处理工具对象
        """

        n_features = data_dictionary["train_features"].shape[-1]
        n_labels = data_dictionary["train_labels"].shape[-1]
        model = PyTorchTransformerModel(
            input_dim=n_features,
            output_dim=n_labels,
            time_window=self.window_size,** self.model_kwargs,
        )
        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss()
        # 检查是否启用了持续学习，并检索模型以继续训练
        trainer = self.get_init_model(dk.pair)
        if trainer is None:
            trainer = PyTorchTransformerTrainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=self.device,
                data_convertor=self.data_convertor,
                window_size=self.window_size,
                tb_logger=self.tb_logger,
                **self.trainer_kwargs,
            )
        trainer.fit(data_dictionary, self.splits)
        return trainer

    def predict(
        self, unfiltered_df: pd.DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> tuple[pd.DataFrame, npt.NDArray[np.int_]]:
        """
        过滤预测特征数据并进行预测。
        :param unfiltered_df: 当前回测周期的完整数据帧
        :return:
        :pred_df: 包含预测结果的数据帧
        :do_predict: 由1和0组成的np数组，表示Freqai需要移除的数据点（NaN）
                    或对数据不确定的地方（PCA和DI指数）
        """

        dk.find_features(unfiltered_df)
        dk.data_dictionary["prediction_features"], _ = dk.filter_features(
            unfiltered_df, dk.training_features_list, training_filter=False
        )

        dk.data_dictionary["prediction_features"], outliers, _ = dk.feature_pipeline.transform(
            dk.data_dictionary["prediction_features"], outlier_check=True
        )

        x = self.data_convertor.convert_x(
            dk.data_dictionary["prediction_features"], device=self.device
        )
        # 如果用户需要多个预测，沿着张量滑动窗口
        x = x.unsqueeze(0)
        # 创建空的torch张量
        self.model.model.eval()
        yb = torch.empty(0).to(self.device)
        if x.shape[1] > self.window_size:
            ws = self.window_size
            for i in range(0, x.shape[1] - ws):
                xb = x[:, i : i + ws, :].to(self.device)
                y = self.model.model(xb)
                yb = torch.cat((yb, y), dim=1)
        else:
            yb = self.model.model(x)

        yb = yb.cpu().squeeze(0)
        pred_df = pd.DataFrame(yb.detach().numpy(), columns=dk.label_list)
        pred_df, _, _ = dk.label_pipeline.inverse_transform(pred_df)

        if self.ft_params.get("DI_threshold", 0) > 0:
            dk.DI_values = dk.feature_pipeline["di"].di_values
        else:
            dk.DI_values = np.zeros(outliers.shape[0])
        dk.do_predict = outliers

        if x.shape[1] > 1:
            zeros_df = pd.DataFrame(
                np.zeros((x.shape[1] - len(pred_df), len(pred_df.columns))), columns=pred_df.columns
            )
            pred_df = pd.concat([zeros_df, pred_df], axis=0, ignore_index=True)
        return (pred_df, dk.do_predict)