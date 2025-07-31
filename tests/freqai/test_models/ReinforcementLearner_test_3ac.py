import logging

import numpy as np

from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner
from freqtrade.freqai.RL.Base3ActionRLEnv import Actions, Base3ActionRLEnv, Positions


logger = logging.getLogger(__name__)


class ReinforcementLearner_test_3ac(ReinforcementLearner):
    """
    用户创建的强化学习模型预测模型。
    """

    class MyRLEnv(Base3ActionRLEnv):
        """
        用户可以重写BaseRLEnv和gym.Env中的任何函数。这里用户
        基于利润和交易持续时间设置自定义奖励。

        警告！
        此函数展示了多种可能的环境控制功能。它也设计用于在小型计算机上快速运行。
        这只是一个基准示例，*不适合*实际生产环境使用。
        """

        def calculate_reward(self, action: int) -> float:
            # 首先，如果动作无效则进行惩罚
            if not self._is_valid(action):
                return -2

            pnl = self.get_unrealized_profit()
            rew = np.sign(pnl) * (pnl + 1)
            factor = 100.0

            # 为进入交易的智能体提供奖励
            if (
                action in (Actions.Buy.value, Actions.Sell.value)
                and self._position == Positions.Neutral
            ):
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

            # 不鼓励长时间持有仓位
            if self._position in (Positions.Short, Positions.Long) and (
                action == Actions.Neutral.value
                or (action == Actions.Sell.value and self._position == Positions.Short)
                or (action == Actions.Buy.value and self._position == Positions.Long)
            ):
                return -1 * trade_duration / max_trade_duration

            # 平仓操作
            if (action == Actions.Buy.value and self._position == Positions.Short) or (
                action == Actions.Sell.value and self._position == Positions.Long
            ):
                if pnl > self.profit_aim * self.rr:
                    factor *= self.rl_config["model_reward_parameters"].get("win_reward_factor", 2)
                return float(rew * factor)

            return 0.0