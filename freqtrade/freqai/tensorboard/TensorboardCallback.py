from enum import Enum
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

from freqtrade.freqai.RL.BaseEnvironment import BaseActions


class TensorboardCallback(BaseCallback):
    """
    自定义回调类，用于在TensorBoard中绘制额外的值
    以及生成 episodic 摘要报告
    """

    def __init__(self, verbose=1, actions: type[Enum] = BaseActions):
        super().__init__(verbose)
        self.model: Any = None  # 训练的模型实例
        self.actions: type[Enum] = actions  # 动作空间枚举类

    def _on_training_start(self) -> None:
        """
        训练开始时调用，记录超参数信息
        """
        # 收集超参数信息
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,  # 算法名称
            "learning_rate": self.model.learning_rate,   # 学习率
            # 可以根据需要添加更多超参数
            # "gamma": self.model.gamma,
            # "gae_lambda": self.model.gae_lambda,
            # "batch_size": self.model.batch_size,
            # "n_steps": self.model.n_steps,
        }
        
        # 定义要跟踪的指标
        metric_dict: dict[str, float | int] = {
            "eval/mean_reward": 0,           # 评估平均奖励
            "rollout/ep_rew_mean": 0,        # 滚动平均奖励
            "rollout/ep_len_mean": 0,        # 平均 episode 长度
            "train/value_loss": 0,           # 价值损失
            "train/explained_variance": 0,   # 解释方差
        }
        
        # 记录超参数到TensorBoard
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        """
        每个训练步骤结束时调用，记录额外的指标到TensorBoard
        """
        # 获取当前步骤的本地信息
        local_info = self.locals["infos"][0]

        # 获取环境中的TensorBoard指标
        if hasattr(self.training_env, "envs"):
            # 处理多环境情况
            tensorboard_metrics = self.training_env.envs[0].unwrapped.tensorboard_metrics
        else:
            # 处理RL多进程情况
            tensorboard_metrics = self.training_env.get_attr("tensorboard_metrics")[0]

        # 记录info中的指标
        for metric in local_info:
            if metric not in ["episode", "terminal_observation"]:
                self.logger.record(f"info/{metric}", local_info[metric])

        # 记录自定义TensorBoard指标
        for category in tensorboard_metrics:
            for metric in tensorboard_metrics[category]:
                self.logger.record(
                    f"{category}/{metric}", 
                    tensorboard_metrics[category][metric]
                )

        return True  # 继续训练