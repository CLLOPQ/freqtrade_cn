from abc import ABC, abstractmethod

import pandas as pd
import torch


class PyTorchDataConvertor(ABC):
    """
    该类负责将`*_features`和`*_labels` pandas数据帧转换为PyTorch张量
    """

    @abstractmethod
    def convert_x(self, df: pd.DataFrame, device: str) -> torch.Tensor:
        """
        将特征数据帧转换为PyTorch张量
        :param df: "*_features"数据帧
        :param device: 用于训练的设备（例如'cpu'，'cuda'）
        """

    @abstractmethod
    def convert_y(self, df: pd.DataFrame, device: str) -> torch.Tensor:
        """
        将标签数据帧转换为PyTorch张量
        :param df: "*_labels"数据帧
        :param device: 用于训练的设备（例如'cpu'，'cuda'）
        """


class DefaultPyTorchDataConvertor(PyTorchDataConvertor):
    """
    默认的数据转换类，保持特征数据帧的形状
    """

    def __init__(
        self,
        target_tensor_type: torch.dtype = torch.float32,
        squeeze_target_tensor: bool = False,
    ):
        """
        :param target_tensor_type: 目标张量的数据类型，分类任务使用torch.long，
            回归任务使用torch.float或torch.double
        :param squeeze_target_tensor: 控制目标张量的形状，用于需要0维或1维输入的损失函数
        """
        self._target_tensor_type = target_tensor_type
        self._squeeze_target_tensor = squeeze_target_tensor

    def convert_x(self, df: pd.DataFrame, device: str) -> torch.Tensor:
        """
        将特征数据帧转换为float32类型的PyTorch张量
        """
        numpy_arrays = df.values  # 将DataFrame转换为NumPy数组
        # 转换为PyTorch张量并移动到指定设备
        x = torch.tensor(numpy_arrays, device=device, dtype=torch.float32)
        return x

    def convert_y(self, df: pd.DataFrame, device: str) -> torch.Tensor:
        """
        将标签数据帧转换为指定类型的PyTorch张量，可选压缩维度
        """
        numpy_arrays = df.values  # 将DataFrame转换为NumPy数组
        # 转换为指定类型的PyTorch张量并移动到指定设备
        y = torch.tensor(numpy_arrays, device=device, dtype=self._target_tensor_type)
        # 如果需要，压缩目标张量的维度
        if self._squeeze_target_tensor:
            y = y.squeeze()
        return y