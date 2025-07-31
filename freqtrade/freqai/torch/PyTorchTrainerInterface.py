from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import torch
from torch import nn


class PyTorchTrainerInterface(ABC):
    """
    PyTorch训练器接口类，定义了模型训练、保存和加载的标准接口
    所有PyTorch训练器类都应实现该接口，确保统一的使用方式
    """

    @abstractmethod
    def fit(self, data_dictionary: dict[str, pd.DataFrame], splits: list[str]) -> None:
        """
        训练模型的核心方法，实现完整的模型训练流程
        :param data_dictionary: 由DataHandler构建的字典，包含所有训练和测试数据/标签
        :param splits: 用于训练的数据集分割列表，必须包含"train"，
            可选"test"可通过在配置文件中设置freqai.data_split_parameters.test_size > 0添加
        
        训练流程包括：
         - 使用PyTorch模型计算批次数据的预测输出
         - 使用损失函数计算预测输出与实际输出之间的损失
         - 通过反向传播计算损失相对于模型参数的梯度
         - 使用优化器更新模型参数
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """
        保存模型状态的方法
        - 保存所有nn.Module的state_dict
        - 保存model_meta_data，该字典应包含用户需要存储的任何附加数据
          例如分类模型的class_names
        :param path: 保存模型的文件路径
        """

    def load(self, path: Path) -> nn.Module:
        """
        从文件加载模型的默认实现
        :param path: 模型zip文件的路径
        :returns: 加载后的pytorch模型
        """
        checkpoint = torch.load(path)
        return self.load_from_checkpoint(checkpoint)

    @abstractmethod
    def load_from_checkpoint(self, checkpoint: dict) -> nn.Module:
        """
        从检查点字典加载模型的方法
        当使用continual_learning时，DataDrawer将通过调用torch.load(path)
        加载包含状态字典和model_meta_data的字典
        任何继承自IFreqaiModel的类都可以通过调用get_init_model方法访问这个字典
        :param checkpoint: 包含模型和优化器状态字典、model_meta_data等的字典
        :returns: 加载后的pytorch模型
        """