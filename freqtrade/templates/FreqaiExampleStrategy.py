import logging
from functools import reduce

import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import IStrategy


logger = logging.getLogger(__name__)


class FreqaiExampleStrategy(IStrategy):
    """
    示例策略，展示用户如何将自己的IFreqaiModel连接到策略。

    警告！这是功能展示，
    这意味着它旨在展示FreqAI的各种功能，并且在所有计算机上都能运行。我们使用此展示来帮助用户
    了解如何构建策略，并将其作为基准来帮助调试可能的问题。

    这意味着它不适合在生产环境中实时运行。
    """

    minimal_roi = {"0": 0.1, "240": -1}

    plot_config = {
        "main_plot": {},
        "subplots": {
            "&-s_close": {"&-s_close": {"color": "blue"}},
            "do_predict": {
                "do_predict": {"color": "brown"},
            },
        },
    }

    process_only_new_candles = True
    stoploss = -0.05
    use_exit_signal = True
    # 这是提供给TA-Lib的最大周期（与时间框架无关）
    startup_candle_count: int = 40
    can_short = True

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        *仅在启用FreqAI的策略中起作用*
        此函数将根据配置中定义的`indicator_periods_candles`、`include_timeframes`、`include_shifted_candles`和
        `include_corr_pairs`自动扩展定义的特征。换句话说，此函数中定义的单个特征将自动扩展为总数为
        `indicator_periods_candles` * `include_timeframes` * `include_shifted_candles` *
        `include_corr_pairs`个添加到模型的特征。

        所有特征必须以`%`为前缀才能被FreqAI内部识别。

        可通过以下方式访问当前交易对/时间框架等元数据：

        `metadata["pair"]` `metadata["tf"]`

        有关这些配置定义的参数如何加速特征工程的更多详细信息，请参见文档：

        https://www.freqtrade.io/en/latest/freqai-parameter-table/#feature-parameters

        https://www.freqtrade.io/en/latest/freqai-feature-engineering/#defining-the-features

        :param dataframe: 将接收特征的策略数据帧
        :param period: 指标周期 - 使用示例：
        :param metadata: 当前交易对的元数据
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        """

        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-sma-period"] = ta.SMA(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=period, stds=2.2
        )
        dataframe["bb_lowerband-period"] = bollinger["lower"]
        dataframe["bb_middleband-period"] = bollinger["mid"]
        dataframe["bb_upperband-period"] = bollinger["upper"]

        dataframe["%-bb_width-period"] = (
            dataframe["bb_upperband-period"] - dataframe["bb_lowerband-period"]
        ) / dataframe["bb_middleband-period"]
        dataframe["%-close-bb_lower-period"] = dataframe["close"] / dataframe["bb_lowerband-period"]

        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)

        dataframe["%-relative_volume-period"] = (
            dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        )

        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        *仅在启用FreqAI的策略中起作用*
        此函数将根据配置中定义的`include_timeframes`、`include_shifted_candles`和`include_corr_pairs`自动扩展定义的特征。
        换句话说，此函数中定义的单个特征将自动扩展为总数为
        `include_timeframes` * `include_shifted_candles` * `include_corr_pairs`个添加到模型的特征。

        此处定义的特征不会在用户定义的`indicator_periods_candles`上自动复制

        所有特征必须以`%`为前缀才能被FreqAI内部识别。

        可通过以下方式访问当前交易对/时间框架等元数据：

        `metadata["pair"]` `metadata["tf"]`

        有关这些配置定义的参数如何加速特征工程的更多详细信息，请参见文档：

        https://www.freqtrade.io/en/latest/freqai-parameter-table/#feature-parameters

        https://www.freqtrade.io/en/latest/freqai-feature-engineering/#defining-the-features

        :param dataframe: 将接收特征的策略数据帧
        :param metadata: 当前交易对的元数据
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-ema-200"] = ta.EMA(dataframe, timeperiod=200)
        """
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        *仅在启用FreqAI的策略中起作用*
        此可选函数将使用基础时间框架的数据帧调用一次。
        这是要调用的最后一个函数，这意味着进入此函数的数据帧将包含所有其他
        freqai_feature_engineering_*函数创建的特征和列。

        此函数是进行自定义特殊特征提取（例如tsfresh）的理想场所。
        此函数是任何不应自动扩展的特征（例如星期几）的理想场所。

        所有特征必须以`%`为前缀才能被FreqAI内部识别。

        可通过以下方式访问当前交易对等元数据：

        `metadata["pair"]`

        有关特征工程的更多详细信息：

        https://www.freqtrade.io/en/latest/freqai-feature-engineering

        :param dataframe: 将接收特征的策略数据帧
        :param metadata: 当前交易对的元数据
        usage example: dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        """
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        *仅在启用FreqAI的策略中起作用*
        设置模型目标的必需函数。
        所有目标必须以`&`为前缀才能被FreqAI内部识别。

        可通过以下方式访问当前交易对等元数据：

        `metadata["pair"]`

        有关特征工程的更多详细信息：

        https://www.freqtrade.io/en/latest/freqai-feature-engineering

        :param dataframe: 将接收目标的策略数据帧
        :param metadata: 当前交易对的元数据
        usage example: dataframe["&-target"] = dataframe["close"].shift(-1) / dataframe["close"]
        """
        dataframe["&-s_close"] = (
            dataframe["close"]
            .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
            .mean()
            / dataframe["close"]
            - 1
        )

        # 分类器通常设置为字符串作为目标：
        # df['&s-up_or_down'] = np.where( df["close"].shift(-100) >
        #                                 df["close"], 'up', 'down')

        # 如果用户希望使用多个目标，可以通过
        # 添加更多以'&'开头的列来实现。用户应记住，多目标
        # 需要多输出预测模型，例如
        # freqai/prediction_models/CatboostRegressorMultiTarget.py，
        # 运行命令：freqtrade trade --freqaimodel CatboostRegressorMultiTarget

        # df["&-s_range"] = (
        #     df["close"]
        #     .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
        #     .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
        #     .max()
        #     -
        #     df["close"]
        #     .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
        #     .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
        #     .min()
        # )

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 所有指标必须由feature_engineering_*()函数填充

        # 模型将返回用户在`set_freqai_targets()`中创建的所有标签
        # （以'&'开头的目标），指示是否应接受预测，
        # 用户在`set_freqai_targets()`中为每个训练周期创建的每个标签的目标均值/标准差。

        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        enter_long_conditions = [
            df["do_predict"] == 1,
            df["&-s_close"] > 0.01,
        ]

        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")

        enter_short_conditions = [
            df["do_predict"] == 1,
            df["&-s_close"] < -0.01,
        ]

        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
            ] = (1, "short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = [df["do_predict"] == 1, df["&-s_close"] < 0]
        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        exit_short_conditions = [df["do_predict"] == 1, df["&-s_close"] > 0]
        if exit_short_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1

        return df

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time,
        entry_tag,
        side: str,
        **kwargs,
    ) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        if side == "long":
            if rate > (last_candle["close"] * (1 + 0.0025)):
                return False
        else:
            if rate < (last_candle["close"] * (1 - 0.0025)):
                return False

        return True