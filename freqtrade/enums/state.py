from enum import Enum


class State(Enum):
    """
    机器人应用状态
    """

    RUNNING = 1        # 运行中
    PAUSED = 2         # 已暂停
    STOPPED = 3        # 已停止
    RELOAD_CONFIG = 4  # 重新加载配置

    def __str__(self):
        return f"{self.name.lower()}"