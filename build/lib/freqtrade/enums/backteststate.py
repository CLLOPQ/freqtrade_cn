from enum import Enum


class BacktestState(Enum):
    """
    机器人应用状态
    """

    STARTUP = 1  # 启动阶段
    DATALOAD = 2  # 数据加载阶段
    ANALYZE = 3   # 分析阶段
    CONVERT = 4   # 转换阶段
    BACKTEST = 5  # 回测阶段

    def __str__(self):
        return f"{self.name.lower()}"