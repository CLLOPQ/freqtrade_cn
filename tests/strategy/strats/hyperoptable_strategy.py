# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

from pandas import DataFrame
from strategy_test_v3 import StrategyTestV3

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import BooleanParameter, DecimalParameter, IntParameter, RealParameter


class HyperoptableStrategy(StrategyTestV3):
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

    sell_params = {"sell_rsi": 74, "sell_minusdi": 0.4}

    buy_plusdi = RealParameter(low=0, high=1, default=0.5, space="buy")
    sell_rsi = IntParameter(low=50, high=100, default=70, space="sell")
    sell_minusdi = DecimalParameter(
        low=0, high=1, default=0.5001, decimals=3, space="sell", load=False
    )
    protection_enabled = BooleanParameter(default=True)
    protection_cooldown_lookback = IntParameter([0, 50], default=30)

    # 无效的绘图配置...
    plot_config = {
        "main_plot": {},
    }

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
    bot_started = False

    def bot_loop_start(self, **kwargs):
        self.bot_loop_started = True

    def bot_start(self,** kwargs) -> None:
        """
        参数也可以在这里定义...
        """
        self.bot_started = True
        self.buy_rsi = IntParameter([0, 50], default=30, space="buy")

    def informative_pairs(self):
        """
        定义额外的、用于从交易所缓存的信息性交易对/时间间隔组合。
        这些交易对/时间间隔组合不可交易，除非它们也在白名单中。
        有关更多信息，请参考文档
        :return: 元组列表，格式为 (交易对, 时间间隔)
            示例: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        基于技术指标，为给定的数据帧填充买入信号
        :param dataframe: 数据帧
        :param metadata: 附加信息，如当前交易的交易对
        :return: 带有买入列的数据帧
        """
        dataframe.loc[
            (
                (dataframe["rsi"] < self.buy_rsi.value)
                & (dataframe["fastd"] < 35)
                & (dataframe["adx"] > 30)
                & (dataframe["plus_di"] > self.buy_plusdi.value)
            )
            | ((dataframe["adx"] > 65) & (dataframe["plus_di"] > self.buy_plusdi.value)),
            "buy",
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        基于技术指标，为给定的数据帧填充卖出信号
        :param dataframe: 数据帧
        :param metadata: 附加信息，如当前交易的交易对
        :return: 带有卖出列的数据帧
        """
        dataframe.loc[
            (
                (
                    (qtpylib.crossed_above(dataframe["rsi"], self.sell_rsi.value))
                    | (qtpylib.crossed_above(dataframe["fastd"], 70))
                )
                & (dataframe["adx"] > 10)
                & (dataframe["minus_di"] > 0)
            )
            | ((dataframe["adx"] > 70) & (dataframe["minus_di"] > self.sell_minusdi.value)),
            "sell",
        ] = 1
        return dataframe