import logging

import torch
from torch import nn


logger = logging.getLogger(__name__)


class PyTorchMLPModel(nn.Module):
    """
    使用PyTorch实现的多层感知器(MLP)模型

    该类主要作为PyTorch模型与freqai集成的简单示例。它没有经过任何优化，不应用于生产环境。

    :param input_dim: 输入特征的数量。该参数指定MLP将用于进行预测的输入数据中的特征数量。
    :param output_dim: 输出类别的数量。该参数指定MLP将预测的类别数量。
    :param hidden_dim: 每层中的隐藏单元数量。该参数控制MLP的复杂度，并决定MLP可以表示多少非线性关系。
        增加隐藏单元的数量可以提高MLP建模复杂模式的能力，但也会增加过拟合训练数据的风险。默认值：256
    :param dropout_percent: 用于正则化的dropout率。该参数指定训练期间丢弃神经元的概率，以防止过拟合。
        dropout率应仔细调整，以平衡欠拟合和过拟合。默认值：0.2
    :param n_layer: MLP中的层数。该参数指定MLP架构中的层数。向MLP添加更多层可以提高其建模复杂模式的能力，
        但也会增加过拟合训练数据的风险。默认值：1

    :returns: MLP的输出，形状为(batch_size, output_dim)
    """

    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super().__init__()
        hidden_dim: int = kwargs.get("hidden_dim", 256)
        dropout_percent: float = kwargs.get("dropout_percent", 0.2)
        n_layer: int = kwargs.get("n_layer", 1)
        
        # 输入层：将输入特征映射到隐藏维度
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        # 隐藏层块：由n_layer个Block组成的序列
        self.blocks = nn.Sequential(*[Block(hidden_dim, dropout_percent) for _ in range(n_layer)])
        # 输出层：将隐藏维度映射到输出维度
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        # 激活函数和dropout层
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_percent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播函数"""
        # 输入层处理并应用激活函数
        x = self.relu(self.input_layer(x))
        # 应用dropout防止过拟合
        x = self.dropout(x)
        # 通过隐藏层块
        x = self.blocks(x)
        # 输出层得到最终结果
        x = self.output_layer(x)
        return x


class Block(nn.Module):
    """
    多层感知器(MLP)的构建块

    :param hidden_dim: 前馈网络中的隐藏单元数量
    :param dropout_percent: 用于正则化的dropout率

    :returns: torch.Tensor，形状为(batch_size, hidden_dim)
    """

    def __init__(self, hidden_dim: int, dropout_percent: float):
        super().__init__()
        self.ff = FeedForward(hidden_dim)  # 前馈网络
        self.dropout = nn.Dropout(p=dropout_percent)  # dropout层
        self.ln = nn.LayerNorm(hidden_dim)  # 层归一化

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播函数"""
        # 先进行层归一化，再通过前馈网络，实现残差连接
        x = x + self.ff(self.ln(x))
        # 应用dropout
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    """
    简单的全连接前馈神经网络块

    :param hidden_dim: 块中的隐藏单元数量
    :return: torch.Tensor，形状为(batch_size, hidden_dim)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # 线性变换
            nn.ReLU(),  # 激活函数
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播函数"""
        return self.net(x)