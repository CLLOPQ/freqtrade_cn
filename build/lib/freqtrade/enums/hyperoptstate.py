from enum import Enum


class HyperoptState(Enum):
    """超参数优化状态"""

    STARTUP = 1    # 启动阶段
    DATALOAD = 2   # 数据加载阶段
    INDICATORS = 3 # 指标计算阶段
    OPTIMIZE = 4   # 优化阶段

    def __str__(self):
        return f"{self.name.lower()}"