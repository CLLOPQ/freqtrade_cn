# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

from strategy_test_v2 import StrategyTestV2

from freqtrade.strategy import BooleanParameter, DecimalParameter, IntParameter, RealParameter


class HyperoptableStrategyV2(StrategyTestV2):
    """
    freqtrade机器人提供的默认策略。
    请不要修改此策略，它仅用于内部使用。
    请查看user_data/strategy目录中的SampleStrategy
    或策略仓库https://github.com/freqtrade/freqtrade-strategies
    获取示例和灵感。
    """

    buy_params = {
        "buy_rsi": 35,
        # 故意不指定，以便测试"默认值"
        # 'buy_plusdi': 0.4
    }

    sell_params = {
        # 卖出参数
        "sell_rsi": 74,
        "sell_minusdi": 0.4,
    }

    buy_plusdi = RealParameter(low=0, high=1, default=0.5, space="buy")
    sell_rsi = IntParameter(low=50, high=100, default=70, space="sell")
    sell_minusdi = DecimalParameter(
        low=0, high=1, default=0.5001, decimals=3, space="sell", load=False
    )
    protection_enabled = BooleanParameter(default=True)
    protection_cooldown_lookback = IntParameter([0, 50], default=30)

    @property
    def protections(self):
        prot = []
        if self.protection_enabled.value:
            prot.append(
                {
                    "method": "CooldownPeriod",
                    "stop_duration_candles": self.protection_cooldown_lookback.value,
                }
            )
        return prot

    bot_loop_started = False

    def bot_loop_start(self, **kwargs):
        self.bot_loop_started = True

    def bot_start(self, **kwargs) -> None:
        """
        参数也可以在这里定义...
        """
        self.buy_rsi = IntParameter([0, 50], default=30, space="buy")