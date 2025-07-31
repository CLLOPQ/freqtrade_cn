# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

from datetime import datetime

from pandas import DataFrame
from strategy_test_v3 import StrategyTestV3

from freqtrade.persistence import Trade


class StrategyTestV3CustomEntryPrice(StrategyTestV3):
    """
    供freqtrade机器人测试使用的策略。
    请不要修改此策略，它仅用于内部使用。
    请查看user_data/strategy目录中的SampleStrategy
    或策略仓库https://github.com/freqtrade/freqtrade-strategies
    获取示例和灵感。
    """

    new_entry_price: float = 0.001

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["volume"] > 0, "enter_long"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def custom_entry_price(
        self,
        pair: str,
        trade: Trade | None,
        current_time: datetime,
        proposed_rate: float,
        entry_tag: str | None,
        side: str,** kwargs,
    ) -> float:
        return self.new_entry_price