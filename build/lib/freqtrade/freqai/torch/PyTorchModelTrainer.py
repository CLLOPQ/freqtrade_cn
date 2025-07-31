import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from freqtrade.freqai.torch.PyTorchDataConvertor import PyTorchDataConvertor
from freqtrade.freqai.torch.PyTorchTrainerInterface import PyTorchTrainerInterface

from .datasets import WindowDataset


logger = logging.getLogger(__name__)


class PyTorchModelTrainer(PyTorchTrainerInterface):
    """
    PyTorch模型训练器基类，负责模型训练、评估、保存和加载的核心功能
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str,
        data_convertor: PyTorchDataConvertor,
        model_meta_data: dict[str, Any] | None = None,
        window_size: int = 1,
        tb_logger: Any = None,** kwargs,
    ):
        """
        初始化模型训练器
        :param model: 待训练的PyTorch模型
        :param optimizer: 训练使用的优化器
        :param criterion: 训练使用的损失函数
        :param device: 训练设备（如'cpu'或'cuda'）
        :param data_convertor: 用于将pandas数据帧转换为PyTorch张量的转换器
        :param model_meta_data: 模型的附加元数据（可选）
        :param window_size: 时间窗口大小，用于时序数据处理
        :param tb_logger: TensorBoard日志记录器
        :param n_steps: 训练迭代次数（每次迭代指调用一次optimizer.step()）
                        如果设置了n_epochs则忽略此参数
        :param n_epochs: 训练轮数
        :param batch_size: 训练批次大小
        """
        if model_meta_data is None:
            model_meta_data = {}
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.model_meta_data = model_meta_data
        self.device = device
        self.n_epochs: int | None = kwargs.get("n_epochs", 10)
        self.n_steps: int | None = kwargs.get("n_steps", None)
        if self.n_steps is None and not self.n_epochs:
            raise Exception("必须设置`n_steps`或`n_epochs`参数")

        self.batch_size: int = kwargs.get("batch_size", 64)
        self.data_convertor = data_convertor
        self.window_size: int = window_size
        self.tb_logger = tb_logger
        self.test_batch_counter = 0

    def fit(self, data_dictionary: dict[str, pd.DataFrame], splits: list[str]):
        """
        训练模型的主方法
        :param data_dictionary: 包含训练和测试数据/标签的字典
        :param splits: 数据集分割列表，必须包含"train"，可选包含"test"
        """
        self.model.train()  # 设置模型为训练模式

        # 创建数据加载器字典
        data_loaders_dictionary = self.create_data_loaders_dictionary(data_dictionary, splits)
        n_obs = len(data_dictionary["train_features"])
        # 计算训练轮数
        n_epochs = self.n_epochs or self.calc_n_epochs(n_obs=n_obs)
        batch_counter = 0
        
        # 训练循环
        for _ in range(n_epochs):
            # 遍历训练数据批次
            for _, batch_data in enumerate(data_loaders_dictionary["train"]):
                xb, yb = batch_data
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                
                # 前向传播：计算预测值
                yb_pred = self.model(xb)
                # 计算损失
                loss = self.criterion(yb_pred, yb)

                # 反向传播与参数更新
                self.optimizer.zero_grad(set_to_none=True)  # 清空梯度
                loss.backward()  # 计算梯度
                self.optimizer.step()  # 更新参数
                
                # 记录训练损失
                if self.tb_logger:
                    self.tb_logger.log_scalar("train_loss", loss.item(), batch_counter)
                batch_counter += 1

            # 测试集评估
            if "test" in splits:
                self.estimate_loss(data_loaders_dictionary, "test")

    @torch.no_grad()  # 禁用梯度计算，加速评估并节省内存
    def estimate_loss(
        self,
        data_loader_dictionary: dict[str, DataLoader],
        split: str,
    ) -> None:
        """
        在指定数据集上评估模型损失
        :param data_loader_dictionary: 数据加载器字典
        :param split: 数据集分割名称（如"test"）
        """
        self.model.eval()  # 设置模型为评估模式
        for _, batch_data in enumerate(data_loader_dictionary[split]):
            xb, yb = batch_data
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            # 前向传播计算预测值
            yb_pred = self.model(xb)
            loss = self.criterion(yb_pred, yb)
            
            # 记录评估损失
            if self.tb_logger:
                self.tb_logger.log_scalar(f"{split}_loss", loss.item(), self.test_batch_counter)
            self.test_batch_counter += 1

        self.model.train()  # 恢复训练模式

    def create_data_loaders_dictionary(
        self, data_dictionary: dict[str, pd.DataFrame], splits: list[str]
    ) -> dict[str, DataLoader]:
        """
        创建数据加载器字典，将数据转换为PyTorch张量并进行批处理
        :param data_dictionary: 包含各分割数据集的字典
        :param splits: 数据集分割列表
        :return: 包含各分割数据加载器的字典
        """
        data_loader_dictionary = {}
        for split in splits:
            # 转换特征和标签为张量
            x = self.data_convertor.convert_x(data_dictionary[f"{split}_features"], self.device)
            y = self.data_convertor.convert_y(data_dictionary[f"{split}_labels"], self.device)
            
            # 创建数据集和数据加载器
            dataset = TensorDataset(x, y)
            data_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0,
            )
            data_loader_dictionary[split] = data_loader

        return data_loader_dictionary

    def calc_n_epochs(self, n_obs: int) -> int:
        """
        根据数据量和批次大小计算所需训练轮数
        :param n_obs: 观测数据总量
        :return: 计算得到的训练轮数
        """
        if not isinstance(self.n_steps, int):
            raise ValueError("必须设置`n_steps`或`n_epochs`参数")
        n_batches = n_obs // self.batch_size
        n_epochs = max(self.n_steps // n_batches, 1)
        if n_epochs <= 10:
            logger.warning(
                f"设置的n_epochs较小: {n_epochs}。"
                f"请考虑增加`n_steps`超参数。"
            )

        return n_epochs

    def save(self, path: Path):
        """
        保存模型状态
        :param path: 保存路径
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_meta_data": self.model_meta_data,
                "pytrainer": self,
            },
            path,
        )

    def load(self, path: Path):
        """
        从路径加载模型
        :param path: 模型文件路径
        :return: 加载后的训练器实例
        """
        checkpoint = torch.load(path)
        return self.load_from_checkpoint(checkpoint)

    def load_from_checkpoint(self, checkpoint: dict):
        """
        从检查点字典加载模型状态
        :param checkpoint: 包含模型状态的字典
        :return: 加载后的训练器实例
        """
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model_meta_data = checkpoint["model_meta_data"]
        return self


class PyTorchTransformerTrainer(PyTorchModelTrainer):
    """
    专为Transformer模型设计的训练器，处理时序数据的窗口化
    """

    def create_data_loaders_dictionary(
        self, data_dictionary: dict[str, pd.DataFrame], splits: list[str]
    ) -> dict[str, DataLoader]:
        """
        创建适用于Transformer的时序数据加载器
        使用WindowDataset处理时序数据，保持时间连续性
        """
        data_loader_dictionary = {}
        for split in splits:
            # 转换特征和标签为张量
            x = self.data_convertor.convert_x(data_dictionary[f"{split}_features"], self.device)
            y = self.data_convertor.convert_y(data_dictionary[f"{split}_labels"], self.device)
            
            # 使用窗口数据集创建时序样本
            dataset = WindowDataset(x, y, self.window_size)
            data_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,  # 时序数据不打乱顺序
                drop_last=True,
                num_workers=0,
            )
            data_loader_dictionary[split] = data_loader

        return data_loader_dictionary