"""强化学习器模型"""

import logging
from pathlib import Path
from typing import Any

import torch as th
from stable_baselines3.common.callbacks import ProgressBarCallback

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Base5ActionRLEnv, Positions
from freqtrade.freqai.RL.BaseEnvironment import BaseEnvironment
from freqtrade.freqai.RL.BaseReinforcementLearningModel import BaseReinforcementLearningModel


logger = logging.getLogger(__name__)


class ReinforcementLearner(BaseReinforcementLearningModel):
    """
    强化学习模型预测模型。

    用户可以继承这个类来创建自己的RL模型，带有自定义的
    环境/训练控制。定义文件如下：

    ```
    from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner

    class MyCoolRLModel(ReinforcementLearner):
    ```

    将文件保存到`user_data/freqaimodels`，然后运行：

    freqtrade trade --freqaimodel MyCoolRLModel --config config.json --strategy SomeCoolStrat

    在这里，用户可以覆盖`IFreqaiModel`继承树中可用的任何函数。对于RL来说最重要的是，
    这里用户可以覆盖`MyRLEnv`（见下文），以定义自定义的`calculate_reward()`函数，
    或者覆盖环境的任何其他部分。

    这个类还允许用户覆盖IFreqaiModel树的任何其他部分。
    例如，用户可以覆盖`def fit()`或`def train()`或`def predict()`
    来获得对这些过程的精细控制。

    另一个常见的覆盖可能是`def data_cleaning_predict()`，用户可以在其中
    获得对数据处理管道的精细控制。
    """

    def fit(self, data_dictionary: dict[str, Any], dk: FreqaiDataKitchen, **kwargs):
        """
        用户可自定义的fit方法
        :param data_dictionary: 包含所有训练/测试特征/标签/权重的通用数据字典
        :param dk: 当前交易对的数据处理工具
        :return:
        model Any = 训练好的模型，用于模拟/实盘/回测中的推理
        """
        train_df = data_dictionary["train_features"]
        total_timesteps = self.freqai_info["rl_config"]["train_cycles"] * len(train_df)

        policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=self.net_arch)

        if self.activate_tensorboard:
            tb_path = Path(dk.full_path / "tensorboard" / dk.pair.split("/")[0])
        else:
            tb_path = None

        if dk.pair not in self.dd.model_dictionary or not self.continual_learning:
            model = self.MODELCLASS(
                self.policy_type,
                self.train_env,
                policy_kwargs=policy_kwargs,
                tensorboard_log=tb_path,** self.freqai_info.get("model_training_parameters", {}),
            )
        else:
            logger.info(
                "启用持续训练 - 从之前训练的智能体开始训练。"
            )
            model = self.dd.model_dictionary[dk.pair]
            model.set_env(self.train_env)
        callbacks: list[Any] = [self.eval_callback, self.tensorboard_callback]
        progressbar_callback: ProgressBarCallback | None = None
        if self.rl_config.get("progress_bar", False):
            progressbar_callback = ProgressBarCallback()
            callbacks.insert(0, progressbar_callback)

        try:
            model.learn(
                total_timesteps=int(total_timesteps),
                callback=callbacks,
            )
        finally:
            if progressbar_callback:
                progressbar_callback.on_training_end()

        if Path(dk.data_path / "best_model.zip").is_file():
            logger.info("回调发现最佳模型。")
            best_model = self.MODELCLASS.load(dk.data_path / "best_model")
            return best_model

        logger.info("找不到最佳模型，使用最终模型替代。")

        return model

    MyRLEnv: type[BaseEnvironment]  # type: ignore[assignment, unused-ignore]

    class MyRLEnv(Base5ActionRLEnv):  # type: ignore[no-redef]
        """
        用户可以覆盖BaseRLEnv和gym.Env中的任何函数。这里用户
        基于利润和交易持续时间设置自定义奖励。
        """

        def calculate_reward(self, action: int) -> float:
            """
            示例奖励函数。这是用户可能希望注入自己创意的一个函数。

                        警告！
            这个函数展示了旨在展示尽可能多的可能环境控制功能。
            它也设计用于在小型计算机上快速运行。这是一个基准，*不*用于实际生产。

            :param action: int = 智能体对当前K线的操作
            :return:
            float = 给智能体当前步骤的奖励（用于神经网络权重优化）
            """
            # 首先，如果操作无效则惩罚
            if not self._is_valid(action):
                self.tensorboard_log("invalid", category="actions")
                return -2

            pnl = self.get_unrealized_profit()
            factor = 100.0

            # 奖励进入交易的智能体
            if action == Actions.Long_enter.value and self._position == Positions.Neutral:
                return 25
            if action == Actions.Short_enter.value and self._position == Positions.Neutral:
                return 25
            # 不鼓励智能体不进入交易
            if action == Actions.Neutral.value and self._position == Positions.Neutral:
                return -1

            max_trade_duration = self.rl_config.get("max_trade_duration_candles", 300)
            trade_duration = self._current_tick - self._last_trade_tick  # type: ignore

            if trade_duration <= max_trade_duration:
                factor *= 1.5
            elif trade_duration > max_trade_duration:
                factor *= 0.5

            # 不鼓励持仓不动
            if (
                self._position in (Positions.Short, Positions.Long)
                and action == Actions.Neutral.value
            ):
                return -1 * trade_duration / max_trade_duration

            # 平多仓
            if action == Actions.Long_exit.value and self._position == Positions.Long:
                if pnl > self.profit_aim * self.rr:
                    factor *= self.rl_config["model_reward_parameters"].get("win_reward_factor", 2)
                return float(pnl * factor)

            # 平空仓
            if action == Actions.Short_exit.value and self._position == Positions.Short:
                if pnl > self.profit_aim * self.rr:
                    factor *= self.rl_config["model_reward_parameters"].get("win_reward_factor", 2)
                return float(pnl * factor)

            return 0.0