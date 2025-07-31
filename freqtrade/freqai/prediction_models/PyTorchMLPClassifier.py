"""PyTorch MLP分类器"""

from typing import Any

import torch

from freqtrade.freqai.base_models.BasePyTorchClassifier import BasePyTorchClassifier
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.torch.PyTorchDataConvertor import (
    DefaultPyTorchDataConvertor,
    PyTorchDataConvertor,
)
from freqtrade.freqai.torch.PyTorchMLPModel import PyTorchMLPModel
from freqtrade.freqai.torch.PyTorchModelTrainer import PyTorchModelTrainer


class PyTorchMLPClassifier(BasePyTorchClassifier):
    """
    该类实现了IFreqaiModel的fit方法。
    在fit方法中，我们初始化模型和训练器对象。
    对模型的唯一要求是与PyTorchClassifier的predict方法对齐，
    该方法期望模型预测long类型的张量。

    参数通过配置文件中freqai部分下的`model_training_parameters`传递。例如：
    {
        ...
        "freqai": {
            ...
            "model_training_parameters" : {
                "learning_rate": 3e-4,
                "trainer_kwargs": {
                    "n_steps": 5000,
                    "batch_size": 64,
                    "n_epochs": null,
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
        return DefaultPyTorchDataConvertor(
            target_tensor_type=torch.long, squeeze_target_tensor=True
        )

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
        :raises ValueError: 如果父类中未定义self.class_names
        """

        class_names = self.get_class_names()
        self.convert_label_column_to_int(data_dictionary, dk, class_names)
        n_features = data_dictionary["train_features"].shape[-1]
        model = PyTorchMLPModel(
            input_dim=n_features, output_dim=len(class_names),** self.model_kwargs
        )
        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        # 检查是否启用了持续学习，并检索模型以继续训练
        trainer = self.get_init_model(dk.pair)
        if trainer is None:
            trainer = PyTorchModelTrainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                model_meta_data={"class_names": class_names},
                device=self.device,
                data_convertor=self.data_convertor,
                tb_logger=self.tb_logger,
                **self.trainer_kwargs,
            )
        trainer.fit(data_dictionary, self.splits)
        return trainer