# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy
from freqtrade.strategy.parameters import CategoricalParameter


class strategy_test_v3_recursive_issue(IStrategy):
    INTERFACE_VERSION = 3

    # 为策略设计的最小ROI
    minimal_roi = {"0": 0.04}

    # 为策略设计的最佳止损
    stoploss = -0.10

    # 策略的最佳时间框架
    timeframe = "5m"
    scenario = CategoricalParameter(["no_bias", "bias1", "bias2"], default="bias1", space="buy")

    # 策略产生有效信号所需的K线数量
    startup_candle_count: int = 100

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 此处引入偏差
        if self.scenario.value == "no_bias":
            dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        else:
            dataframe["rsi"] = ta.RSI(dataframe, timeperiod=50)

        if self.scenario.value == "bias2":
            # 同时包含bias1和bias2
            dataframe["rsi_lookahead"] = ta.RSI(dataframe, timeperiod=50).shift(-1)

        # 字符串列不应引起问题
        dataframe["test_string_column"] = f"a{len(dataframe)}"

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe