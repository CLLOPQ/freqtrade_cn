"""3动作强化学习环境基类"""

import logging
from enum import Enum

from gymnasium import spaces

from freqtrade.freqai.RL.BaseEnvironment import BaseEnvironment, Positions


logger = logging.getLogger(__name__)


class Actions(Enum):
    Neutral = 0  # 中性（不操作）
    Buy = 1      # 买入
    Sell = 2     # 卖出


class Base3ActionRLEnv(BaseEnvironment):
    """
    3动作环境的基类
    """

    def __init__(self, **kwargs):
        super().__init__(** kwargs)
        self.actions = Actions

    def set_action_space(self):
        self.action_space = spaces.Discrete(len(Actions))

    def step(self, action: int):
        """
        智能体单步（时间上增加一个K线）的逻辑
        :param: action: int = 智能体计划在当前步骤采取的动作类型
        :returns:
            observation = 环境的当前状态
            step_reward = 从`calculate_reward()`获得的奖励
            _done = 智能体是否"死亡"或K线是否结束
            info = 传递回openai gym库的字典
        """
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        self._update_unrealized_total_profit()
        step_reward = self.calculate_reward(action)
        self.total_reward += step_reward
        self.tensorboard_log(self.actions._member_names_[action], category="actions")

        trade_type = None
        if self.is_tradesignal(action):
            if action == Actions.Buy.value:
                if self._position == Positions.Short:
                    self._update_total_profit()
                self._position = Positions.Long
                trade_type = "long"  # 多头
                self._last_trade_tick = self._current_tick
            elif action == Actions.Sell.value and self.can_short:
                if self._position == Positions.Long:
                    self._update_total_profit()
                self._position = Positions.Short
                trade_type = "short"  # 空头
                self._last_trade_tick = self._current_tick
            elif action == Actions.Sell.value and not self.can_short:
                self._update_total_profit()
                self._position = Positions.Neutral
                trade_type = "exit"   # 平仓
                self._last_trade_tick = None
            else:
                print("未定义的情况")

            if trade_type is not None:
                self.trade_history.append(
                    {
                        "price": self.current_price(),
                        "index": self._current_tick,
                        "type": trade_type,
                        "profit": self.get_unrealized_profit(),
                    }
                )

        # 如果总利润或未实现总利润低于最大回撤，结束交易
        if (
            self._total_profit < self.max_drawdown
            or self._total_unrealized_profit < self.max_drawdown
        ):
            self._done = True

        self._position_history.append(self._position)

        info = dict(
            tick=self._current_tick,
            action=action,
            total_reward=self.total_reward,
            total_profit=self._total_profit,
            position=self._position.value,
            trade_duration=self.get_trade_duration(),
            current_profit_pct=self.get_unrealized_profit(),
        )

        observation = self._get_observation()

        # 用户可以根据需要调整时间
        truncated = False

        self._update_history(info)

        return observation, step_reward, self._done, truncated, info

    def is_tradesignal(self, action: int) -> bool:
        """
        确定信号是否为交易信号
        例如：当智能体处于空头头寸时想要执行买入动作
        """
        return (
            (action == Actions.Buy.value and self._position == Positions.Neutral)  # 中性时买入
            or (action == Actions.Sell.value and self._position == Positions.Long)  # 多头时卖出
            or (
                action == Actions.Sell.value
                and self._position == Positions.Neutral
                and self.can_short  # 允许做空时，中性状态下卖出（开空）
            )
            or (
                action == Actions.Buy.value and self._position == Positions.Short and self.can_short  # 允许做空时，空头状态下买入（平空）
            )
        )

    def _is_valid(self, action: int) -> bool:
        """
        确定信号是否有效
        例如：当智能体处于多头头寸时想要执行卖出动作
        """
        if self.can_short:
            return action in [Actions.Buy.value, Actions.Sell.value, Actions.Neutral.value]
        else:
            # 不允许做空时，只有在多头状态下才能卖出
            if action == Actions.Sell.value and self._position != Positions.Long:
                return False
            return True