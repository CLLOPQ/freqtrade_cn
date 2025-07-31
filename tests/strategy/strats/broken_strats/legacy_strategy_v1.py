# 类型: 忽略
from pandas import DataFrame

from freqtrade.strategy import IStrategy


# 虚拟策略 - 不再加载但会引发异常。
class TestStrategyLegacyV1(IStrategy):
    minimal_roi = {"40": 0.0, "30": 0.01, "20": 0.02, "0": 0.04}
    stoploss = -0.10

    timeframe = "5m"

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
        return dataframe

