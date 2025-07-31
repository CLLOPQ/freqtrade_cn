import torch


class WindowDataset(torch.utils.data.Dataset):
    """
    用于处理时序数据的窗口数据集类
    将时序数据转换为固定窗口大小的样本，适用于时序预测任务
    """
    def __init__(self, xs, ys, window_size):
        """
        初始化窗口数据集
        :param xs: 输入特征数据，形状为 [样本数, 特征数]
        :param ys: 目标标签数据，形状为 [样本数, 标签数]
        :param window_size: 时间窗口大小，即每个样本包含的时序步数
        """
        self.xs = xs
        self.ys = ys
        self.window_size = window_size

    def __len__(self):
        """
        返回数据集中样本的数量
        由于每个样本需要window_size个连续数据点，因此总样本数为总数据量减去窗口大小
        """
        return len(self.xs) - self.window_size

    def __getitem__(self, index):
        """
        根据索引获取一个样本
        样本由窗口大小的输入特征和对应的目标标签组成
        
        :param index: 样本索引
        :return: 元组 (window_x, window_y)，其中：
                 - window_x: 窗口内的输入特征，形状为 [window_size, 特征数]
                 - window_y: 窗口对应的目标标签，形状为 [1, 标签数]
        """
        # 计算反向索引，使数据集按时间倒序排列
        idx_rev = len(self.xs) - self.window_size - index - 1
        
        # 提取窗口内的输入特征
        window_x = self.xs[idx_rev : idx_rev + self.window_size, :]
        
        # 提取窗口对应的目标标签（注意索引对齐）
        # 这里取窗口最后一个位置对应的标签作为目标
        window_y = self.ys[idx_rev + self.window_size - 1, :].unsqueeze(0)
        
        return window_x, window_y