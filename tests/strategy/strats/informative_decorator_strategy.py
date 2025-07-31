# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

from pandas import DataFrame

from freqtrade.strategy import IStrategy, informative, merge_informative_pair


class InformativeDecoratorTest(IStrategy):
    """
    供freqtrade机器人测试使用的策略。
    请不要修改此策略，它仅用于内部使用。
    请查看user_data/strategy目录中的SampleStrategy
    或策略仓库https://github.com/freqtrade/freqtrade-strategies
    获取示例和灵感。
    """

    INTERFACE_VERSION = 2
    stoploss = -0.10
    timeframe = "5m"
    startup_candle_count: int = 20

    def informative_pairs(self):
        # 故意返回2个元组，必须在兼容性代码中转换为3个
        return [
            ("NEO/USDT", "5m"),
            ("NEO/USDT", "15m", ""),
            ("NEO/USDT", "2h", "futures"),
        ]

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["buy"] = 0
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["sell"] = 0
        return dataframe

    # 装饰器堆叠测试
    @informative("30m")
    @informative("1h")
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = 14
        return dataframe

    # 简单信息对测试
    @informative("1h", "NEO/{stake}")
    def populate_indicators_neo_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = 14
        return dataframe

    @informative("1h", "{base}/BTC")
    def populate_indicators_base_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = 14
        return dataframe

    # 报价货币与基础货币不同的测试
    @informative("1h", "ETH/BTC", candle_type="spot")
    def populate_indicators_eth_btc_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = 14
        return dataframe

    # 格式化测试
    @informative("30m", "NEO/{stake}", "{column}_{BASE}_{QUOTE}_{base}_{quote}_{asset}_{timeframe}")
    def populate_indicators_btc_1h_2(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = 14
        return dataframe

    # 自定义格式化器测试
    @informative("30m", "ETH/{stake}", fmt=lambda column, **kwargs: column + "_from_callable")
    def populate_indicators_eth_30m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = 14
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 当前交易对的策略时间框架指标
        dataframe["rsi"] = 14
        # 此方法中可使用信息对
        dataframe["rsi_less"] = dataframe["rsi"] < dataframe["rsi_1h"]

        # 将手动信息对与装饰器混合使用
        informative = self.dp.get_pair_dataframe("NEO/USDT", "5m", "")
        informative["rsi"] = 14
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, "5m", ffill=True)

        return dataframe