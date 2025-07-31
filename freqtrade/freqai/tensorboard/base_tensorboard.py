import logging
from pathlib import Path
from typing import Any

from xgboost.callback import TrainingCallback


logger = logging.getLogger(__name__)


class BaseTensorboardLogger:
    """
    基础Tensorboard日志记录器类
    提供Tensorboard日志记录的基础接口，可被具体实现类继承
    """
    def __init__(self, logdir: Path, activate: bool = True):
        """
        初始化日志记录器
        :param logdir: 日志存储目录
        :param activate: 是否激活日志记录
        """
        self.logdir = logdir
        self.activate = activate
        # 确保日志目录存在
        if self.activate and not self.logdir.exists():
            self.logdir.mkdir(parents=True, exist_ok=True)
        logger.info(f"初始化基础Tensorboard日志记录器，日志目录: {logdir}")

    def log_scalar(self, tag: str, scalar_value: Any, step: int):
        """
        记录标量值到日志
        :param tag: 标量的标签/名称
        :param scalar_value: 要记录的标量值
        :param step: 对应的步骤数（通常是epoch或iteration）
        """
        if self.activate:
            logger.debug(f"记录标量 - 标签: {tag}, 值: {scalar_value}, 步骤: {step}")

    def close(self):
        """关闭日志记录器，释放资源"""
        if self.activate:
            logger.info("关闭基础Tensorboard日志记录器")


class BaseTensorBoardCallback(TrainingCallback):
    """
    基础TensorBoard回调类
    用于XGBoost训练过程中的TensorBoard日志记录回调
    """
    def __init__(self, logdir: Path, activate: bool = True):
        """
        初始化回调函数
        :param logdir: 日志存储目录
        :param activate: 是否激活回调
        """
        self.logdir = logdir
        self.activate = activate
        self.logger = BaseTensorboardLogger(logdir, activate)
        logger.info(f"初始化基础TensorBoard回调，日志目录: {logdir}")

    def after_iteration(self, model, epoch: int, evals_log: TrainingCallback.EvalsLog) -> bool:
        """
        每次迭代（epoch）结束后调用
        :param model: XGBoost模型实例
        :param epoch: 当前 epoch 数
        :param evals_log: 评估日志
        :return: 是否停止训练（False表示继续）
        """
        if self.activate and evals_log:
            # 记录评估指标
            for data_name, metric_logs in evals_log.items():
                for metric_name, log in metric_logs.items():
                    if log:
                        # 获取最新的评估值
                        value = log[-1]
                        tag = f"{data_name}/{metric_name}"
                        self.logger.log_scalar(tag, value, epoch)
        return False  # 不停止训练

    def after_training(self, model):
        """
        训练结束后调用
        :param model: XGBoost模型实例
        :return: 处理后的模型实例
        """
        self.logger.close()
        logger.info("训练结束，关闭TensorBoard回调日志记录器")
        return model