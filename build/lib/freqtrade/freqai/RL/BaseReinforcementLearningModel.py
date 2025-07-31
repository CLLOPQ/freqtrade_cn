"""强化学习模型基类"""

import copy
import importlib
import logging
from abc import abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch as th
import torch.multiprocessing
from pandas import DataFrame
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import is_masking_supported
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from freqtrade.exceptions import OperationalException
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Base5ActionRLEnv
from freqtrade.freqai.RL.BaseEnvironment import BaseActions, BaseEnvironment, Positions
from freqtrade.freqai.tensorboard.TensorboardCallback import TensorboardCallback
from freqtrade.persistence import Trade


logger = logging.getLogger(__name__)

torch.multiprocessing.set_sharing_strategy("file_system")

SB3_MODELS = ["PPO", "A2C", "DQN"]
SB3_CONTRIB_MODELS = ["TRPO", "ARS", "RecurrentPPO", "MaskablePPO", "QRDQN"]


class BaseReinforcementLearningModel(IFreqaiModel):
    """
    用户创建的强化学习模型预测类
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(config=kwargs["config"])
        self.max_threads = min(
            self.freqai_info["rl_config"].get("cpu_count", 1),
            max(int(self.max_system_threads / 2), 1),
        )
        th.set_num_threads(self.max_threads)
        self.reward_params = self.freqai_info["rl_config"]["model_reward_parameters"]
        self.train_env: VecMonitor | SubprocVecEnv | gym.Env = gym.Env()
        self.eval_env: VecMonitor | SubprocVecEnv | gym.Env = gym.Env()
        self.eval_callback: MaskableEvalCallback | None = None
        self.model_type = self.freqai_info["rl_config"]["model_type"]
        self.rl_config = self.freqai_info["rl_config"]
        self.df_raw: DataFrame = DataFrame()
        self.continual_learning = self.freqai_info.get("continual_learning", False)
        if self.model_type in SB3_MODELS:
            import_str = "stable_baselines3"
        elif self.model_type in SB3_CONTRIB_MODELS:
            import_str = "sb3_contrib"
        else:
            raise OperationalException(
                f"{self.model_type}在stable_baselines3或sb3_contrib中不可用。"
                f"请选择{SB3_MODELS}或{SB3_CONTRIB_MODELS}中的一个"
            )

        mod = importlib.import_module(import_str, self.model_type)
        self.MODELCLASS = getattr(mod, self.model_type)
        self.policy_type = self.freqai_info["rl_config"]["policy_type"]
        self.unset_outlier_removal()
        self.net_arch = self.rl_config.get("net_arch", [128, 128])
        self.dd.model_type = import_str
        self.tensorboard_callback: TensorboardCallback = TensorboardCallback(
            verbose=1, actions=BaseActions
        )

    def unset_outlier_removal(self):
        """
        如果用户激活了任何可能移除训练点的功能，此函数将把它们设置为false并警告用户
        """
        if self.ft_params.get("use_SVM_to_remove_outliers", False):
            self.ft_params.update({"use_SVM_to_remove_outliers": False})
            logger.warning("用户尝试在RL中使用SVM。正在停用SVM。")
        if self.ft_params.get("use_DBSCAN_to_remove_outliers", False):
            self.ft_params.update({"use_DBSCAN_to_remove_outliers": False})
            logger.warning("用户尝试在RL中使用DBSCAN。正在停用DBSCAN。")
        if self.ft_params.get("DI_threshold", False):
            self.ft_params.update({"DI_threshold": False})
            logger.warning("用户尝试在RL中使用DI_threshold。正在停用DI_threshold。")
        if self.freqai_info["data_split_parameters"].get("shuffle", False):
            self.freqai_info["data_split_parameters"].update({"shuffle": False})
            logger.warning("用户尝试打乱训练数据。将shuffle设置为False")

    def train(self, unfiltered_df: DataFrame, pair: str, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        过滤训练数据并训练模型。train大量使用datakitchen来存储、保存、加载和分析数据。
        :param unfiltered_df: 当前训练周期的完整数据帧
        :param metadata: 来自策略的交易对元数据
        :returns:
        :model: 训练好的模型，可用于推理（self.predict）
        """

        logger.info(f"--------------------开始训练 {pair} --------------------")

        features_filtered, labels_filtered = dk.filter_features(
            unfiltered_df,
            dk.training_features_list,
            dk.label_list,
            training_filter=True,
        )

        dd: dict[str, Any] = dk.make_train_test_datasets(features_filtered, labels_filtered)
        self.df_raw = copy.deepcopy(dd["train_features"])
        dk.fit_labels()  # FIXME 目前无用，只是为了满足append方法

        # 仅基于训练数据集标准化所有数据
        prices_train, prices_test = self.build_ohlc_price_dataframes(dk.data_dictionary, pair, dk)

        dk.feature_pipeline = self.define_data_pipeline(threads=dk.thread_count)

        (dd["train_features"], dd["train_labels"], dd["train_weights"]) = (
            dk.feature_pipeline.fit_transform(
                dd["train_features"], dd["train_labels"], dd["train_weights"]
            )
        )

        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) != 0:
            (dd["test_features"], dd["test_labels"], dd["test_weights"]) = (
                dk.feature_pipeline.transform(
                    dd["test_features"], dd["test_labels"], dd["test_weights"]
                )
            )

        logger.info(
            f"在{len(dk.data_dictionary['train_features'].columns)}个特征和"
            f"{len(dd['train_features'])}个数据点上训练模型"
        )

        self.set_train_and_eval_environments(dd, prices_train, prices_test, dk)

        model = self.fit(dd, dk)

        logger.info(f"--------------------完成训练 {pair}--------------------")

        return model

    def set_train_and_eval_environments(
        self,
        data_dictionary: dict[str, DataFrame],
        prices_train: DataFrame,
        prices_test: DataFrame,
        dk: FreqaiDataKitchen,
    ):
        """
        如果用户使用自定义的MyRLEnv，可以重写此方法
        :param data_dictionary: dict = 包含训练和测试特征/标签/权重的通用数据字典
        :param prices_train/test: DataFrame = 包含训练或测试期间环境中使用的价格的数据帧
        :param dk: FreqaiDataKitchen = 当前交易对的数据处理工具
        """
        train_df = data_dictionary["train_features"]
        test_df = data_dictionary["test_features"]

        env_info = self.pack_env_dict(dk.pair)

        self.train_env = self.MyRLEnv(df=train_df, prices=prices_train,** env_info)
        self.eval_env = Monitor(self.MyRLEnv(df=test_df, prices=prices_test, **env_info))
        self.eval_callback = MaskableEvalCallback(
            self.eval_env,
            deterministic=True,
            render=False,
            eval_freq=len(train_df),
            best_model_save_path=str(dk.data_path),
            use_masking=(self.model_type == "MaskablePPO" and is_masking_supported(self.eval_env)),
        )

        actions = self.train_env.get_actions()
        self.tensorboard_callback = TensorboardCallback(verbose=1, actions=actions)

    def pack_env_dict(self, pair: str) -> dict[str, Any]:
        """
        创建环境参数字典
        """
        env_info = {
            "window_size": self.CONV_WIDTH,
            "reward_kwargs": self.reward_params,
            "config": self.config,
            "live": self.live,
            "can_short": self.can_short,
            "pair": pair,
            "df_raw": self.df_raw,
        }
        if self.data_provider:
            env_info["fee"] = self.data_provider._exchange.get_fee(  # type: ignore
                symbol=self.data_provider.current_whitelist()[0]
            )

        return env_info

    @abstractmethod
    def fit(self, data_dictionary: dict[str, Any], dk: FreqaiDataKitchen, **kwargs):
        """
        智能体自定义和抽象强化学习自定义在这里进行。抽象方法，因此必须由用户类重写。
        """
        return

    def get_state_info(self, pair: str) -> tuple[float, float, int]:
        """
        模拟/实盘（非回测）期间的状态信息，反馈给模型
        :param pair: str = 要获取环境信息的交易对（COIN/STAKE）
        :return:
        :market_side: float = 表示交易对的空头、多头或中性状态
        :current_profit: float = 当前交易的未实现利润
        :trade_duration: int = 交易已开仓的K线数量
        """
        open_trades = Trade.get_trades_proxy(is_open=True)
        market_side = 0.5
        current_profit: float = 0
        trade_duration = 0
        for trade in open_trades:
            if trade.pair == pair:
                if self.data_provider._exchange is None:  # type: ignore
                    logger.error("没有可用的交易所。")
                    return 0, 0, 0
                else:
                    current_rate = self.data_provider._exchange.get_rate(  # type: ignore
                        pair, refresh=False, side="exit", is_short=trade.is_short
                    )

                now = datetime.now(timezone.utc).timestamp()
                trade_duration = int((now - trade.open_date_utc.timestamp()) / self.base_tf_seconds)
                current_profit = trade.calc_profit_ratio(current_rate)
                if trade.is_short:
                    market_side = 0
                else:
                    market_side = 1

        return market_side, current_profit, int(trade_duration)

    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> tuple[DataFrame, npt.NDArray[np.int_]]:
        """
        过滤预测特征数据并进行预测
        :param unfiltered_dataframe: 当前回测周期的完整数据帧
        :return:
        :pred_df: 包含预测结果的数据帧
        :do_predict: 由1和0组成的np数组，表示Freqai需要移除的数据点（NaN）
                    或对数据不确定的地方（PCA和DI指数）
        """

        dk.find_features(unfiltered_df)
        filtered_dataframe, _ = dk.filter_features(
            unfiltered_df, dk.training_features_list, training_filter=False
        )

        dk.data_dictionary["prediction_features"] = self.drop_ohlc_from_df(filtered_dataframe, dk)

        dk.data_dictionary["prediction_features"], _, _ = dk.feature_pipeline.transform(
            dk.data_dictionary["prediction_features"], outlier_check=True
        )

        pred_df = self.rl_model_predict(dk.data_dictionary["prediction_features"], dk, self.model)
        pred_df.fillna(0, inplace=True)

        return (pred_df, dk.do_predict)

    def rl_model_predict(
        self, dataframe: DataFrame, dk: FreqaiDataKitchen, model: Any
    ) -> DataFrame:
        """
        强化学习模块中用于预测的辅助函数
        :param dataframe: DataFrame = 用于预测的特征数据帧
        :param dk: FreqaiDatakitchen = 当前交易对的数据处理工具
        :param model: Any = 用于特征推理的训练好的模型
        """
        output = pd.DataFrame(np.zeros(len(dataframe)), columns=dk.label_list)

        def _predict(window):
            observations = dataframe.iloc[window.index]
            if self.live and self.rl_config.get("add_state_info", False):
                market_side, current_profit, trade_duration = self.get_state_info(dk.pair)
                observations["current_profit_pct"] = current_profit
                observations["position"] = market_side
                observations["trade_duration"] = trade_duration
            res, _ = model.predict(observations, deterministic=True)
            return res

        output = output.rolling(window=self.CONV_WIDTH).apply(_predict)

        return output

    def build_ohlc_price_dataframes(
        self, data_dictionary: dict, pair: str, dk: FreqaiDataKitchen
    ) -> tuple[DataFrame, DataFrame]:
        """
        为环境构建训练价格和测试价格
        """

        pair = pair.replace(":", "")
        train_df = data_dictionary["train_features"]
        test_df = data_dictionary["test_features"]

        # 模型训练和评估的价格数据
        tf = self.config["timeframe"]
        rename_dict = {
            "%-raw_open": "open",
            "%-raw_low": "low",
            "%-raw_high": "high",
            "%-raw_close": "close",
        }
        rename_dict_old = {
            f"%-{pair}raw_open_{tf}": "open",
            f"%-{pair}raw_low_{tf}": "low",
            f"%-{pair}raw_high_{tf}": "high",
            f"%-{pair}raw_close_{tf}": "close",
        }

        prices_train = train_df.filter(rename_dict.keys(), axis=1)
        prices_train_old = train_df.filter(rename_dict_old.keys(), axis=1)
        if prices_train.empty or not prices_train_old.empty:
            if not prices_train_old.empty:
                prices_train = prices_train_old
                rename_dict = rename_dict_old
            logger.warning(
                "强化学习模块未在feature_engineering_standard()中找到正确的原始价格分配。"
                "请使用以下代码分配：\n"
                'dataframe["%-raw_close"] = dataframe["close"]\n'
                'dataframe["%-raw_open"] = dataframe["open"]\n'
                'dataframe["%-raw_high"] = dataframe["high"]\n'
                'dataframe["%-raw_low"] = dataframe["low"]\n'
                "在feature_engineering_standard()中"
            )
        elif prices_train.empty:
            raise OperationalException(
                "未找到价格，请按照日志警告说明更正策略。"
            )

        prices_train.rename(columns=rename_dict, inplace=True)
        prices_train.reset_index(drop=True)

        prices_test = test_df.filter(rename_dict.keys(), axis=1)
        prices_test.rename(columns=rename_dict, inplace=True)
        prices_test.reset_index(drop=True)

        train_df = self.drop_ohlc_from_df(train_df, dk)
        test_df = self.drop_ohlc_from_df(test_df, dk)

        return prices_train, prices_test

    def drop_ohlc_from_df(self, df: DataFrame, dk: FreqaiDataKitchen):
        """
        给定一个数据帧，删除ohlc数据
        """
        drop_list = ["%-raw_open", "%-raw_low", "%-raw_high", "%-raw_close"]

        if self.rl_config["drop_ohlc_from_features"]:
            df.drop(drop_list, axis=1, inplace=True)
            feature_list = dk.training_features_list
            dk.training_features_list = [e for e in feature_list if e not in drop_list]

        return df

    def load_model_from_disk(self, dk: FreqaiDataKitchen) -> Any:
        """
        用户可以在尝试限制内存使用量*和*执行持续学习时使用
        目前未使用
        """
        exists = Path(dk.data_path / f"{dk.model_filename}_model").is_file()
        if exists:
            model = self.MODELCLASS.load(dk.data_path / f"{dk.model_filename}_model")
        else:
            logger.info("磁盘上没有模型文件可继续学习。")

        return model

    def _on_stop(self):
        """
        机器人关闭时调用的钩子。关闭SubprocVecEnv子进程以干净地关闭。
        """

        if self.train_env:
            self.train_env.close()

        if self.eval_env:
            self.eval_env.close()

    # 可以由用户重写以进一步自定义的嵌套类
    class MyRLEnv(Base5ActionRLEnv):
        """
        用户可以重写BaseRLEnv和gym.Env中的任何函数。这里用户
        基于利润和交易持续时间设置自定义奖励。
        """

        def calculate_reward(self, action: int) -> float:  # noqa: C901
            """
            示例奖励函数。这是用户可能希望注入自己创意的一个函数。

            警告！
            这个函数展示了旨在展示尽可能多的可能环境控制功能。
            它也设计用于在小型计算机上快速运行。这是一个基准，*不*用于实际生产。

            :param action: int = 智能体对当前K线的操作
            :return:
            float = 给智能体当前步骤的奖励（用于神经网络权重优化）
            """
            # 首先，如果动作无效则惩罚
            if not self._is_valid(action):
                return -2

            pnl = self.get_unrealized_profit()
            factor = 100.0

            # 可以使用数据帧中的特征值
            rsi_now = self.raw_features[
                f"%-rsi-period-10_shift-1_{self.pair}_{self.config['timeframe']}"
            ].iloc[self._current_tick]

            # 奖励智能体进入交易
            if (
                action in (Actions.Long_enter.value, Actions.Short_enter.value)
                and self._position == Positions.Neutral
            ):
                if rsi_now < 40:
                    factor = 40 / rsi_now
                else:
                    factor = 1
                return 25 * factor

            # 不鼓励智能体不进入交易
            if action == Actions.Neutral.value and self._position == Positions.Neutral:
                return -1

            max_trade_duration = self.rl_config.get("max_trade_duration_candles", 300)
            if self._last_trade_tick:
                trade_duration = self._current_tick - self._last_trade_tick
            else:
                trade_duration = 0

            if trade_duration <= max_trade_duration:
                factor *= 1.5
            elif trade_duration > max_trade_duration:
                factor *= 0.5

            # 不鼓励持有头寸
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


def make_env(
    MyRLEnv: type[BaseEnvironment],
    env_id: str,
    rank: int,
    seed: int,
    train_df: DataFrame,
    price: DataFrame,
    env_info: dict[str, Any],
) -> Callable:
    """
    多进程环境的实用函数

    :param env_id: (str) 环境ID
    :param num_env: (int) 希望在子进程中拥有的环境数量
    :param seed: (int) RNG的初始种子
    :param rank: (int) 子进程的索引
    :param env_info: (dict) 实例化环境所需的所有参数
    :return: (Callable)
    """

    def _init() -> gym.Env:
        env = MyRLEnv(df=train_df, prices=price, id=env_id, seed=seed + rank,** env_info)

        return env

    set_random_seed(seed)
    return _init