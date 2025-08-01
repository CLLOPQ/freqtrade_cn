from freqtrade.enums import BacktestState


class BTProgress:
    _action: BacktestState = BacktestState.STARTUP
    _progress: float = 0
    _max_steps: float = 0

    def __init__(self):
        pass

    def init_step(self, action: BacktestState, max_steps: float):
        self._action = action
        self._max_steps = max_steps
        self._progress = 0

    def set_new_value(self, new_value: float):
        self._progress = new_value

    def increment(self):
        self._progress += 1

    @property
    def progress(self):
        """
        获取进度作为比率，限制在0到1之间（以避免小的计算误差）。
        """
        return max(
            min(round(self._progress / self._max_steps, 5) if self._max_steps > 0 else 0, 1), 0
        )

    @property
    def action(self):
        return str(self._action)