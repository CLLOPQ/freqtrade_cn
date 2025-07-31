# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy


class StrategyTestV2(IStrategy):
    """
    供freqtrade机器人测试使用的策略。
    请不要修改此策略，它仅用于内部使用。
    请查看user_data/strategy目录中的SampleStrategy
    或策略仓库https://github.com/freqtrade/freqtrade-strategies
    获取示例和灵感。
    """

    INTERFACE_VERSION = 2

    # 为策略设计的最小ROI
    minimal_roi = {"40": 0.0, "30": 0.01, "20": 0.02, "0": 0.04}

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
    # 测试传统的use_sell_signal定义
    use_sell_signal = False

    # 默认情况下，此策略不使用仓位调整
    position_adjustment_enable = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        向给定的DataFrame添加多个不同的技术指标

        性能注意：为获得最佳性能，请谨慎使用指标数量
        只取消您在策略中使用的指标或您的超参数优化配置的注释，
        否则会浪费内存和CPU使用率。
        :param dataframe: 包含交易所数据的Dataframe
        :param metadata: 附加信息，如当前交易的交易对
        :return: 包含策略所需所有必要指标的Dataframe
        """

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

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        基于技术指标，为给定的数据帧填充买入信号
        :param dataframe: 数据帧
        :param metadata: 附加信息，如当前交易的交易对
        :return: 带有买入列的数据帧
        """
        dataframe.loc[
            (
                (dataframe["rsi"] < 35)
                & (dataframe["fastd"] < 35)
                & (dataframe["adx"] > 30)
                & (dataframe["plus_di"] > 0.5)
            )
            | ((dataframe["adx"] > 65) & (dataframe["plus_di"] > 0.5)),
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
                    (qtpylib.crossed_above(dataframe["rsi"], 70))
                    | (qtpylib.crossed_above(dataframe["fastd"], 70))
                )
                & (dataframe["adx"] > 10)
                & (dataframe["minus_di"] > 0)
            )
            | ((dataframe["adx"] > 70) & (dataframe["minus_di"] > 0.5)),
            "sell",
        ] = 1
        return dataframe