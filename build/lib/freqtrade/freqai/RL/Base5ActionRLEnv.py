"""5动作强化学习环境基类"""

import logging
from enum import Enum

from gymnasium import spaces

from freqtrade.freqai.RL.BaseEnvironment import BaseEnvironment, Positions


logger = logging.getLogger(__name__)


class Actions(Enum):
    Neutral = 0         # 中性（不操作）
    Long_enter = 1      # 开多仓
    Long_exit = 2       # 平多仓
    Short_enter = 3     # 开空仓
    Short_exit = 4      # 平空仓


class Base5ActionRLEnv(BaseEnvironment):
    """
    5动作环境的基类
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
            if action == Actions.Neutral.value:
                self._position = Positions.Neutral
                trade_type = "neutral"  # 中性
                self._last_trade_tick = None
            elif action == Actions.Long_enter.value:
                self._position = Positions.Long
                trade_type = "enter_long"  # 开多
                self._last_trade_tick = self._current_tick
            elif action == Actions.Short_enter.value:
                self._position = Positions.Short
                trade_type = "enter_short"  # 开空
                self._last_trade_tick = self._current_tick
            elif action == Actions.Long_exit.value:
                self._update_total_profit()
                self._position = Positions.Neutral
                trade_type = "exit_long"  # 平多
                self._last_trade_tick = None
            elif action == Actions.Short_exit.value:
                self._update_total_profit()
                self._position = Positions.Neutral
                trade_type = "exit_short"  # 平空
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
        例如：当智能体处于空头头寸时想要执行多头平仓动作
        """
        return not (
            (action == Actions.Neutral.value and self._position == Positions.Neutral)  # 中性时不操作
            or (action == Actions.Neutral.value and self._position == Positions.Short)  # 空头时不操作
            or (action == Actions.Neutral.value and self._position == Positions.Long)  # 多头时不操作
            or (action == Actions.Short_enter.value and self._position == Positions.Short)  # 空头时开空
            or (action == Actions.Short_enter.value and self._position == Positions.Long)  # 多头时开空
            or (action == Actions.Short_exit.value and self._position == Positions.Long)  # 多头时平空
            or (action == Actions.Short_exit.value and self._position == Positions.Neutral)  # 中性时平空
            or (action == Actions.Long_enter.value and self._position == Positions.Long)  # 多头时开多
            or (action == Actions.Long_enter.value and self._position == Positions.Short)  # 空头时开多
            or (action == Actions.Long_exit.value and self._position == Positions.Short)  # 空头时平多
            or (action == Actions.Long_exit.value and self._position == Positions.Neutral)  # 中性时平多
        )

    def _is_valid(self, action: int) -> bool:
        """
        确定信号是否有效
        例如：当智能体处于空头头寸时想要执行多头平仓动作
        """
        # 只有在对应持仓状态下才能平仓
        if action == Actions.Short_exit.value:
            if self._position != Positions.Short:
                return False
        elif action == Actions.Long_exit.value:
            if self._position != Positions.Long:
                return False

        # 只有在未持仓状态下才能开仓
        if action in (Actions.Short_enter.value, Actions.Long_enter.value):
            if self._position != Positions.Neutral:
                return False

        return True