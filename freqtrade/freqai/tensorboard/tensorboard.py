import logging
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter
from xgboost import callback

from freqtrade.freqai.tensorboard.base_tensorboard import (
    BaseTensorBoardCallback,
    BaseTensorboardLogger,
)


logger = logging.getLogger(__name__)


class TensorboardLogger(BaseTensorboardLogger):
    """
    Tensorboard日志记录器实现类
    基于PyTorch的SummaryWriter提供实际的日志记录功能
    """
    def __init__(self, logdir: Path, activate: bool = True):
        """
        初始化Tensorboard日志记录器
        :param logdir: 日志存储目录
        :param activate: 是否激活日志记录
        """
        super().__init__(logdir, activate)
        self.activate = activate
        self.writer: SummaryWriter | None = None
        if self.activate:
            # 创建SummaryWriter实例，用于写入Tensorboard日志
            self.writer = SummaryWriter(f"{str(logdir)}/tensorboard")
            logger.info(f"Tensorboard日志记录器已初始化，日志目录: {logdir}/tensorboard")

    def log_scalar(self, tag: str, scalar_value: Any, step: int):
        """
        记录标量值到Tensorboard
        :param tag: 标量的标签/名称
        :param scalar_value: 要记录的标量值
        :param step: 对应的步骤数（通常是epoch或iteration）
        """
        if self.activate and self.writer:
            self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        """关闭日志记录器，释放资源"""
        if self.activate and self.writer:
            self.writer.flush()  # 确保所有日志都被写入磁盘
            self.writer.close()
            logger.info("Tensorboard日志记录器已关闭")


class TensorBoardCallback(BaseTensorBoardCallback):
    """
    TensorBoard回调实现类
    用于在XGBoost训练过程中记录指标到Tensorboard
    """
    def __init__(self, logdir: Path, activate: bool = True):
        """
        初始化TensorBoard回调
        :param logdir: 日志存储目录
        :param activate: 是否激活回调
        """
        super().__init__(logdir, activate)
        self.activate = activate
        self.writer: SummaryWriter | None = None
        if self.activate:
            self.writer = SummaryWriter(f"{str(logdir)}/tensorboard")
            logger.info(f"TensorBoard回调已初始化，日志目录: {logdir}/tensorboard")

    def after_iteration(
        self, model, epoch: int, evals_log: callback.TrainingCallback.EvalsLog
    ) -> bool:
        """
        每次迭代（epoch）结束后调用，记录评估指标
        :param model: XGBoost模型实例
        :param epoch: 当前epoch数
        :param evals_log: 评估日志字典
        :return: 是否停止训练（False表示继续）
        """
        if not self.activate or not self.writer or not evals_log:
            return False

        # 定义评估数据集的名称列表
        evals = ["validation", "train"]
        # 遍历评估日志，记录每个指标
        for metric, eval_name in zip(evals_log.items(), evals, strict=False):
            for metric_name, log in metric[1].items():
                # 提取最新的评估分数
                score = log[-1][0] if isinstance(log[-1], tuple) else log[-1]
                # 记录标量到Tensorboard
                self.writer.add_scalar(f"{eval_name}-{metric_name}", score, epoch)

        return False  # 不停止训练

    def after_training(self, model):
        """
        训练结束后调用，清理资源
        :param model: XGBoost模型实例
        :return: 原模型实例
        """
        if self.activate and self.writer:
            self.writer.flush()  # 确保所有日志都被写入磁盘
            self.writer.close()
            logger.info("TensorBoard回调已关闭日志写入器")

        return model