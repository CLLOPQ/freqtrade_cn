"""多进程强化学习器"""

import logging
from typing import Any

from pandas import DataFrame
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import is_masking_supported
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner
from freqtrade.freqai.RL.BaseReinforcementLearningModel import make_env
from freqtrade.freqai.tensorboard.TensorboardCallback import TensorboardCallback


logger = logging.getLogger(__name__)


class ReinforcementLearner_multiproc(ReinforcementLearner):
    """
    演示如何构建向量化环境
    """

    def set_train_and_eval_environments(
        self,
        data_dictionary: dict[str, Any],
        prices_train: DataFrame,
        prices_test: DataFrame,
        dk: FreqaiDataKitchen,
    ):
        """
        用户可以重写此方法，如果他们使用自定义的MyRLEnv
        :param data_dictionary: 包含训练和测试特征/标签/权重的通用数据字典
        :param prices_train/test: 用于训练或测试期间环境中的价格数据帧
        :param dk: 当前交易对的数据处理工具
        """
        train_df = data_dictionary["train_features"]
        test_df = data_dictionary["test_features"]

        if self.train_env:
            self.train_env.close()
        if self.eval_env:
            self.eval_env.close()

        env_info = self.pack_env_dict(dk.pair)

        eval_freq = len(train_df) // self.max_threads

        env_id = "train_env"
        self.train_env = VecMonitor(
            SubprocVecEnv(
                [
                    make_env(self.MyRLEnv, env_id, i, 1, train_df, prices_train, env_info=env_info)
                    for i in range(self.max_threads)
                ]
            )
        )

        eval_env_id = "eval_env"
        self.eval_env = VecMonitor(
            SubprocVecEnv(
                [
                    make_env(
                        self.MyRLEnv, eval_env_id, i, 1, test_df, prices_test, env_info=env_info
                    )
                    for i in range(self.max_threads)
                ]
            )
        )

        self.eval_callback = MaskableEvalCallback(
            self.eval_env,
            deterministic=True,
            render=False,
            eval_freq=eval_freq,
            best_model_save_path=str(dk.data_path),
            use_masking=(self.model_type == "MaskablePPO" and is_masking_supported(self.eval_env)),
        )

        # 不建议在多环境下使用TENSORBOARD CALLBACK，
        # 它会返回错误信息，并且在SB3中不是线程安全的！！！
        actions = self.train_env.env_method("get_actions")[0]
        self.tensorboard_callback = TensorboardCallback(verbose=1, actions=actions)