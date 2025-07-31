# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

from datetime import datetime

import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from freqtrade.strategy import (
    BooleanParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
    RealParameter,
)


class StrategyTestV3(IStrategy):
    """
    供freqtrade机器人测试使用的策略。
    请不要修改此策略，它仅用于内部使用。
    请查看user_data/strategy目录中的SampleStrategy
    或策略仓库https://github.com/freqtrade/freqtrade-strategies
    获取示例和灵感。
    """

    INTERFACE_VERSION = 3

    # 为策略设计的最小ROI
    minimal_roi = {"40": 0.0, "30": 0.01, "20": 0.02, "0": 0.04}

    # 策略的最佳最大开仓交易数
    max_open_trades = -1

    # 为策略设计的最佳止损
    stoploss = -0.10

    # 策略的最佳时间框架
    timeframe = "5m"

    # 可选的订单类型映射
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "limit",
        "stoploss_on_exchange": False,
    }

    # 策略产生有效信号所需的K线数量
    startup_candle_count: int = 20

    # 订单的可选有效时间
    order_time_in_force = {
        "entry": "gtc",
        "exit": "gtc",
    }

    buy_params = {
        "buy_rsi": 35,
        # 故意不指定，以便测试"默认值"
        # 'buy_plusdi': 0.4
    }

    sell_params = {"sell_rsi": 74, "sell_minusdi": 0.4}

    buy_rsi = IntParameter([0, 50], default=30, space="buy")
    buy_plusdi = RealParameter(low=0, high=1, default=0.5, space="buy")
    sell_rsi = IntParameter(low=50, high=100, default=70, space="sell")
    sell_minusdi = DecimalParameter(
        low=0, high=1, default=0.5001, decimals=3, space="sell", load=False
    )
    protection_enabled = BooleanParameter(default=True)
    protection_cooldown_lookback = IntParameter([0, 50], default=30)

    # TODO: 这能用于保护测试吗？（隐式替换HyperoptableStrategy...）
    @property
    def protections(self):
        prot = []
        if self.protection_enabled.value:
            # 简化测试的变通方法。这在实际场景中不起作用。
            prot = self.config.get("_strategy_protections", {})
        return prot

    bot_started = False

    def bot_start(self):
        self.bot_started = True

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 动量指标
        # ------------------------------------

        # ADX指标
        dataframe["adx"] = ta.ADX(dataframe)

        # MACD指标
        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        # 负向趋向指标
        dataframe["minus_di"] = ta.MINUS_DI(dataframe)

        # 正向趋向指标
        dataframe["plus_di"] = ta.PLUS_DI(dataframe)

        # RSI指标
        dataframe["rsi"] = ta.RSI(dataframe)

        # 快速随机指标
        stoch_fast = ta.STOCHF(dataframe)
        dataframe["fastd"] = stoch_fast["fastd"]
        dataframe["fastk"] = stoch_fast["fastk"]

        # 布林带
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]

        # EMA - 指数移动平均线
        dataframe["ema10"] = ta.EMA(dataframe, timeperiod=10)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["rsi"] < self.buy_rsi.value)
                & (dataframe["fastd"] < 35)
                & (dataframe["adx"] > 30)
                & (dataframe["plus_di"] > self.buy_plusdi.value)
            )
            | ((dataframe["adx"] > 65) & (dataframe["plus_di"] > self.buy_plusdi.value)),
            "enter_long",
        ] = 1
        dataframe.loc[
            (qtpylib.crossed_below(dataframe["rsi"], self.sell_rsi.value)),
            ("enter_short", "enter_tag"),
        ] = (1, "short_Tag")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
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
            "exit_long",
        ] = 1

        dataframe.loc[
            (qtpylib.crossed_above(dataframe["rsi"], self.buy_rsi.value)),
            ("exit_short", "exit_tag"),
        ] = (1, "short_Tag")

        return dataframe

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        # 在所有情况下返回3.0。
        # 机器人逻辑必须确保这是一个允许的杠杆，并最终进行相应调整。

        return 3.0

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,** kwargs,
    ) -> float | None:
        if current_profit < -0.0075:
            orders = trade.select_filled_orders(trade.entry_side)
            return round(orders[0].stake_amount, 0)

        return None


class StrategyTestV3Futures(StrategyTestV3):
    can_short = True