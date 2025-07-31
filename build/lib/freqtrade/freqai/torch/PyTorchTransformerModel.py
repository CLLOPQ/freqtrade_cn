import math

import torch
from torch import nn


"""
该架构基于论文“Attention Is All You Need”。
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Lukasz Kaiser, 和 Illia Polosukhin. 2017.
"""


class PyTorchTransformerModel(nn.Module):
    """
    一种使用位置编码的时序建模Transformer方法。
    架构基于论文“Attention Is All You Need”。
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, 和 Illia Polosukhin. 2017.
    """

    def __init__(
        self,
        input_dim: int = 7,
        output_dim: int = 7,
        hidden_dim=1024,
        n_layer=2,
        dropout_percent=0.1,
        time_window=10,
        nhead=8,
    ):
        super().__init__()
        self.time_window = time_window
        # 确保Transformer的输入维度可被nhead整除
        self.dim_val = input_dim - (input_dim % nhead)
        # 输入网络：将输入特征映射到合适的维度并应用dropout
        self.input_net = nn.Sequential(
            nn.Dropout(dropout_percent), 
            nn.Linear(input_dim, self.dim_val)
        )

        # 使用位置编码对时序进行编码
        self.positional_encoding = PositionalEncoding(d_model=self.dim_val, max_len=self.dim_val)

        # 定义Transformer的编码器块
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim_val, 
            nhead=nhead, 
            dropout=dropout_percent, 
            batch_first=True  # 使批次维度作为第一个维度
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layer)

        # 伪解码全连接网络
        self.output_net = nn.Sequential(
            nn.Linear(self.dim_val * time_window, int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(dropout_percent),
            nn.Linear(int(hidden_dim), int(hidden_dim / 2)),
            nn.ReLU(),
            nn.Dropout(dropout_percent),
            nn.Linear(int(hidden_dim / 2), int(hidden_dim / 4)),
            nn.ReLU(),
            nn.Dropout(dropout_percent),
            nn.Linear(int(hidden_dim / 4), output_dim),
        )

    def forward(self, x, mask=None, add_positional_encoding=True):
        """
        前向传播函数
        Args:
            x: 输入特征，形状为 [批次大小, 序列长度, 输入维度]
            mask: 要应用于注意力输出的掩码（可选）
            add_positional_encoding: 如果为True，我们将位置编码添加到输入中
                                     某些任务可能不需要此操作
        """
        # 输入特征映射
        x = self.input_net(x)
        # 添加位置编码
        if add_positional_encoding:
            x = self.positional_encoding(x)
        # Transformer编码器处理
        x = self.transformer(x, mask=mask)
        # 重塑特征以适应全连接网络
        x = x.reshape(-1, 1, self.time_window * x.shape[-1])
        # 输出网络得到最终结果
        x = self.output_net(x)
        return x


class PositionalEncoding(nn.Module):
    """
    位置编码模块，为序列中的每个位置添加固定的位置信息
    使模型能够理解输入序列的时序关系
    """
    def __init__(self, d_model, max_len=5000):
        """
        Args
            d_model: 输入的隐藏维度
            max_len: 预期的最大序列长度
        """
        super().__init__()

        # 创建[序列长度, 隐藏维度]的矩阵，表示max_len输入的位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算位置编码的除数项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 偶数索引使用正弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数索引使用余弦函数
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 增加批次维度

        # 注册为非持久化缓冲区，不会被视为模型参数
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        # 将位置编码添加到输入中
        x = x + self.pe[:, : x.size(1)]
        return x