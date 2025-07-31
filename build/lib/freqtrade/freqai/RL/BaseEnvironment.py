"""强化学习环境基类"""

import logging
import random
from abc import abstractmethod
from enum import Enum

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from pandas import DataFrame

from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


class BaseActions(Enum):
    """
    默认动作空间，主要用于类型处理
    """

    Neutral = 0         # 中性（不操作）
    Long_enter = 1      # 开多仓
    Long_exit = 2       # 平多仓
    Short_enter = 3     # 开空仓
    Short_exit = 4      # 平空仓


class Positions(Enum):
    Short = 0           # 空头
    Long = 1            # 多头
    Neutral = 0.5       # 中性

    def opposite(self):
        """返回相反的头寸"""
        return Positions.Short if self == Positions.Long else Positions.Long


class BaseEnvironment(gym.Env):
    """
    环境基类。该类与动作数量无关。
    继承类可自定义以包含不同的动作数量/类型，
    参见RL/Base5ActionRLEnv.py和RL/Base4ActionRLEnv.py
    """

    def __init__(
        self,
        *,
        df: DataFrame,
        prices: DataFrame,
        reward_kwargs: dict,
        window_size=10,
        starting_point=True,
        id: str = "baseenv-1",  # noqa: A002
        seed: int = 1,
        config: dict,
        live: bool = False,
        fee: float = 0.0015,
        can_short: bool = False,
        pair: str = "",
        df_raw: DataFrame,
    ):
        """
        初始化训练/评估环境
        :param df: 特征数据帧
        :param prices: 用于训练环境的价格数据帧
        :param window_size: 传递给智能体的窗口大小（时间序列）
        :param reward_kwargs: 用户在`rl_config`中指定的额外配置设置
        :param starting_point: 是否从窗口边缘开始
        :param id: 环境的字符串ID（用于多进程环境的后端）
        :param seed: 设置gym.Env对象中环境的种子
        :param config: 典型的用户配置文件
        :param live: 该环境是否在模拟/实盘/回测中激活
        :param fee: 用于环境交互的手续费
        :param can_short: 环境是否允许做空
        """
        self.config: dict = config
        self.rl_config: dict = config["freqai"]["rl_config"]
        self.add_state_info: bool = self.rl_config.get("add_state_info", False)
        self.id: str = id
        self.max_drawdown: float = 1 - self.rl_config.get("max_training_drawdown_pct", 0.8)
        self.compound_trades: bool = config["stake_amount"] == "unlimited"
        self.pair: str = pair
        self.raw_features: DataFrame = df_raw
        if self.config.get("fee", None) is not None:
            self.fee = self.config["fee"]
        else:
            self.fee = fee

        # 默认为5动作，所有子环境可以覆盖此设置
        self.actions: type[Enum] = BaseActions
        self.tensorboard_metrics: dict = {}
        self.can_short: bool = can_short
        self.live: bool = live
        if not self.live and self.add_state_info:
            raise OperationalException(
                "`add_state_info`在回测中不可用。请在rl_config中将该参数更改为false。"
                "有关更多信息，请参见`add_state_info`文档。"
            )
        self.seed(seed)
        self.reset_env(df, prices, window_size, reward_kwargs, starting_point)

    def reset_env(
        self,
        df: DataFrame,
        prices: DataFrame,
        window_size: int,
        reward_kwargs: dict,
        starting_point=True,
    ):
        """
        当智能体失败时重置环境（在我们的情况下，如果回撤超过用户设置的max_training_drawdown_pct）
        :param df: 特征数据帧
        :param prices: 用于训练环境的价格数据帧
        :param window_size: 传递给智能体的窗口大小（时间序列）
        :param reward_kwargs: 用户在`rl_config`中指定的额外配置设置
        :param starting_point: 是否从窗口边缘开始
        """
        self.signal_features: DataFrame = df
        self.prices: DataFrame = prices
        self.window_size: int = window_size
        self.starting_point: bool = starting_point
        self.rr: float = reward_kwargs["rr"]
        self.profit_aim: float = reward_kwargs["profit_aim"]

        # 状态空间
        if self.add_state_info:
            self.total_features = self.signal_features.shape[1] + 3
        else:
            self.total_features = self.signal_features.shape[1]
        self.shape = (window_size, self.total_features)
        self.set_action_space()
        self.observation_space = spaces.Box(low=-1, high=1, shape=self.shape, dtype=np.float32)

        # 回合
        self._start_tick: int = self.window_size
        self._end_tick: int = len(self.prices) - 1
        self._done: bool = False
        self._current_tick: int = self._start_tick
        self._last_trade_tick: int | None = None
        self._position = Positions.Neutral
        self._position_history: list = [None]
        self.total_reward: float = 0
        self._total_profit: float = 1
        self._total_unrealized_profit: float = 1
        self.history: dict = {}
        self.trade_history: list = []

    def get_attr(self, attr: str):
        """
        返回环境的属性
        :param attr: 要返回的属性
        :return: 属性值
        """
        return getattr(self, attr)

    @abstractmethod
    def set_action_space(self):
        """
        特定于环境动作数量的设置。必须被继承实现。
        """

    def action_masks(self) -> list[bool]:
        """返回动作掩码，指示哪些动作是有效的"""
        return [self._is_valid(action.value) for action in self.actions]

    def seed(self, seed: int = 1):
        """设置环境的随机种子"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def tensorboard_log(
        self,
        metric: str,
        value: int | float | None = None,
        inc: bool | None = None,
        category: str = "custom",
    ):
        """
        构建tensorboard_metrics字典，供TensorboardCallback解析。
        此函数用于跟踪训练环境中的增量对象、事件和动作。
        例如，用户可以调用此函数来跟踪`calculate_reward()`中`is_valid`调用的发生频率：

        def calculate_reward(self, action: int) -> float:
            if not self._is_valid(action):
                self.tensorboard_log("invalid")
                return -2

        :param metric: 要跟踪和增量的指标
        :param value: `metric`的值
        :param inc: （已弃用）设置是否增加值
        :param category: `metric`的类别
        """
        increment = True if value is None else False
        value = 1 if increment else value

        if category not in self.tensorboard_metrics:
            self.tensorboard_metrics[category] = {}

        if not increment or metric not in self.tensorboard_metrics[category]:
            self.tensorboard_metrics[category][metric] = value
        else:
            self.tensorboard_metrics[category][metric] += value

    def reset_tensorboard_log(self):
        """重置tensorboard日志"""
        self.tensorboard_metrics = {}

    def reset(self, seed=None):
        """
        在每个回合开始时调用重置
        """
        self.reset_tensorboard_log()

        self._done = False

        if self.starting_point is True:
            if self.rl_config.get("randomize_starting_position", False):
                length_of_data = int(self._end_tick / 4)
                start_tick = random.randint(self.window_size + 1, length_of_data)
                self._start_tick = start_tick
            self._position_history = (self._start_tick * [None]) + [self._position]
        else:
            self._position_history = (self.window_size * [None]) + [self._position]

        self._current_tick = self._start_tick
        self._last_trade_tick = None
        self._position = Positions.Neutral

        self.total_reward = 0.0
        self._total_profit = 1.0  # 单位
        self.history = {}
        self.trade_history = []
        self.portfolio_log_returns = np.zeros(len(self.prices))

        self._profits = [(self._start_tick, 1)]
        self.close_trade_profit = []
        self._total_unrealized_profit = 1

        return self._get_observation(), self.history

    @abstractmethod
    def step(self, action: int):
        """
        步骤取决于动作类型，必须被继承实现。
        """
        return

    def _get_observation(self):
        """
        这可能与动作类型相关或无关，用户可以在其自定义的"MyRLEnv"中继承此方法
        """
        features_window = self.signal_features[
            (self._current_tick - self.window_size) : self._current_tick
        ]
        if self.add_state_info:
            features_and_state = DataFrame(
                np.zeros((len(features_window), 3)),
                columns=["current_profit_pct", "position", "trade_duration"],
                index=features_window.index,
            )

            features_and_state["current_profit_pct"] = self.get_unrealized_profit()
            features_and_state["position"] = self._position.value
            features_and_state["trade_duration"] = self.get_trade_duration()
            features_and_state = pd.concat([features_window, features_and_state], axis=1)
            return features_and_state
        else:
            return features_window

    def get_trade_duration(self):
        """
        如果智能体处于交易中，获取交易持续时间
        """
        if self._last_trade_tick is None:
            return 0
        else:
            return self._current_tick - self._last_trade_tick

    def get_unrealized_profit(self):
        """
        如果智能体处于交易中，获取未实现利润
        """
        if self._last_trade_tick is None:
            return 0.0

        if self._position == Positions.Neutral:
            return 0.0
        elif self._position == Positions.Short:
            current_price = self.add_entry_fee(self.prices.iloc[self._current_tick].open)
            last_trade_price = self.add_exit_fee(self.prices.iloc[self._last_trade_tick].open)
            return (last_trade_price - current_price) / last_trade_price
        elif self._position == Positions.Long:
            current_price = self.add_exit_fee(self.prices.iloc[self._current_tick].open)
            last_trade_price = self.add_entry_fee(self.prices.iloc[self._last_trade_tick].open)
            return (current_price - last_trade_price) / last_trade_price
        else:
            return 0.0

    @abstractmethod
    def is_tradesignal(self, action: int) -> bool:
        """
        确定信号是否为交易信号。这取决于环境中的动作，因此必须被继承实现。
        """
        return True

    def _is_valid(self, action: int) -> bool:
        """
        确定信号是否有效。这取决于环境中的动作，因此必须被继承实现。
        """
        return True

    def add_entry_fee(self, price):
        """添加入场手续费"""
        return price * (1 + self.fee)

    def add_exit_fee(self, price):
        """添加出场手续费"""
        return price / (1 + self.fee)

    def _update_history(self, info):
        """更新历史记录"""
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    @abstractmethod
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

    def _update_unrealized_total_profit(self):
        """
        在回合结束时更新未实现总利润
        """
        if self._position in (Positions.Long, Positions.Short):
            pnl = self.get_unrealized_profit()
            if self.compound_trades:
                # 假设单位赌注和复利
                unrl_profit = self._total_profit * (1 + pnl)
            else:
                # 假设单位赌注和不复利
                unrl_profit = self._total_profit + pnl
            self._total_unrealized_profit = unrl_profit

    def _update_total_profit(self):
        """更新总利润"""
        pnl = self.get_unrealized_profit()
        if self.compound_trades:
            # 假设单位赌注和复利
            self._total_profit = self._total_profit * (1 + pnl)
        else:
            # 假设单位赌注和不复利
            self._total_profit += pnl

    def current_price(self) -> float:
        """获取当前价格"""
        return self.prices.iloc[self._current_tick].open

    def get_actions(self) -> type[Enum]:
        """
        用于SubprocVecEnv从初始化的环境中获取动作，以用于tensorboard回调
        """
        return self.actions

    # 保留以备将来可能构建更复杂的环境模板
    # def most_recent_return(self):
    #     """
    #     计算交易中的逐笔回报。
    #     回报来自多头头寸的价格上涨和空头头寸的价格下跌。
    #     多头头寸期间的卖出/买入或持有动作会触发卖出/买入手续费。
    #     """
    #     # 多头头寸
    #     if self._position == Positions.Long:
    #         current_price = self.prices.iloc[self._current_tick].open
    #         previous_price = self.prices.iloc[self._current_tick - 1].open

    #         if (self._position_history[self._current_tick - 1] == Positions.Short
    #                 or self._position_history[self._current_tick - 1] == Positions.Neutral):
    #             previous_price = self.add_entry_fee(previous_price)

    #         return np.log(current_price) - np.log(previous_price)

    #     # 空头头寸
    #     if self._position == Positions.Short:
    #         current_price = self.prices.iloc[self._current_tick].open
    #         previous_price = self.prices.iloc[self._current_tick - 1].open
    #         if (self._position_history[self._current_tick - 1] == Positions.Long
    #                 or self._position_history[self._current_tick - 1] == Positions.Neutral):
    #             previous_price = self.add_exit_fee(previous_price)

    #         return np.log(previous_price) - np.log(current_price)

    #     return 0

    # def update_portfolio_log_returns(self, action):
    #     self.portfolio_log_returns[self._current_tick] = self.most_recent_return(action)