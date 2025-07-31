import logging

import numpy as np  # noqa
import pandas as pd  # noqa
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import IntParameter, IStrategy, merge_informative_pair  # noqa


logger = logging.getLogger(__name__)


class FreqaiExampleHybridStrategy(IStrategy):
    """
    混合FreqAI策略的示例，旨在说明用户如何使用FreqAI来增强典型的Freqtrade策略。

    启动此策略的命令为：

    freqtrade trade --strategy FreqaiExampleHybridStrategy --strategy-path freqtrade/templates
    --freqaimodel CatboostClassifier --config config_examples/config_freqai.example.json

    或者用户可以在配置文件中添加以下内容：

    "freqai": {
        "enabled": true,
        "purge_old_models": 2,
        "train_period_days": 15,
        "identifier": "unique-id",
        "feature_parameters": {
            "include_timeframes": [
                "3m",
                "15m",
                "1h"
            ],
            "include_corr_pairlist": [
                "BTC/USDT",
                "ETH/USDT"
            ],
            "label_period_candles": 20,
            "include_shifted_candles": 2,
            "DI_threshold": 0.9,
            "weight_factor": 0.9,
            "principal_component_analysis": false,
            "use_SVM_to_remove_outliers": true,
            "indicator_periods_candles": [10, 20]
        },
        "data_split_parameters": {
            "test_size": 0,
            "random_state": 1
        },
        "model_training_parameters": {
            "n_estimators": 800
        }
    },

    感谢@smarmau和@johanvulgt开发并分享此策略。
    """

    minimal_roi = {
        # "120": 0.0,  # 120分钟后以盈亏平衡退出
        "60": 0.01,
        "30": 0.02,
        "0": 0.04,
    }

    plot_config = {
        "main_plot": {
            "tema": {},
        },
        "subplots": {
            "MACD": {
                "macd": {"color": "blue"},
                "macdsignal": {"color": "orange"},
            },
            "RSI": {
                "rsi": {"color": "red"},
            },
            "涨跌状态": {
                "&s-up_or_down": {"color": "green"},
            },
        },
    }

    process_only_new_candles = True
    stoploss = -0.05
    use_exit_signal = True
    startup_candle_count: int = 30
    can_short = True

    # 可超参数优化参数
    buy_rsi = IntParameter(low=1, high=50, default=30, space="buy", optimize=True, load=True)
    sell_rsi = IntParameter(low=50, high=100, default=70, space="sell", optimize=True, load=True)
    short_rsi = IntParameter(low=51, high=100, default=70, space="sell", optimize=True, load=True)
    exit_short_rsi = IntParameter(low=1, high=50, default=30, space="buy", optimize=True, load=True)

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        *仅在启用FreqAI的策略中有效*
        此函数将根据配置中定义的`indicator_periods_candles`、`include_timeframes`、`include_shifted_candles`和`include_corr_pairs`自动扩展已定义的特征。换句话说，此函数中定义的单个特征将自动扩展为总共`indicator_periods_candles` * `include_timeframes` * `include_shifted_candles` * `include_corr_pairs`个特征，添加到模型中。

        所有特征必须以`%`开头才能被FreqAI内部识别。

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
        *仅在启用FreqAI的策略中有效*
        此函数将根据配置中定义的`include_timeframes`、`include_shifted_candles`和`include_corr_pairs`自动扩展已定义的特征。换句话说，此函数中定义的单个特征将自动扩展为总共`include_timeframes` * `include_shifted_candles` * `include_corr_pairs`个特征，添加到模型中。

        此处定义的特征不会在用户定义的`indicator_periods_candles`上自动复制

        所有特征必须以`%`开头才能被FreqAI内部识别。

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
        *仅在启用FreqAI的策略中有效*
        此可选函数将使用基础时间帧的数据帧调用一次。这是最后一个被调用的函数，这意味着进入此函数的数据帧将包含所有其他freqai_feature_engineering_*函数创建的特征和列。

        此函数是进行自定义特殊特征提取（例如tsfresh）的理想位置。此函数是任何不应自动扩展的特征（例如星期几）的理想位置。

        所有特征必须以`%`开头才能被FreqAI内部识别。

        有关特征工程的更多详细信息：

        https://www.freqtrade.io/en/latest/freqai-feature-engineering

        :param dataframe: 将接收特征的策略数据帧
        :param metadata: 当前交易对的元数据
        示例用法：dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        """
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        *仅在启用FreqAI的策略中有效*
        设置模型目标的必需函数。
        所有目标必须以`&`开头才能被FreqAI内部识别。

        有关特征工程的更多详细信息：

        https://www.freqtrade.io/en/latest/freqai-feature-engineering

        :param dataframe: 将接收目标的策略数据帧
        :param metadata: 当前交易对的元数据
        示例用法：dataframe["&-target"] = dataframe["close"].shift(-1) / dataframe["close"]
        """
        self.freqai.class_names = ["down", "up"]
        dataframe["&s-up_or_down"] = np.where(
            dataframe["close"].shift(-50) > dataframe["close"], "up", "down"
        )

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:  # noqa: C901
        # 用户在此处创建自己的自定义策略。本示例是基于超趋势的策略。

        dataframe = self.freqai.start(dataframe, metadata, self)

        # TA指标，用于与Freqai目标结合
        # RSI
        dataframe["rsi"] = ta.RSI(dataframe)

        # 布林带
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]
        dataframe["bb_percent"] = (dataframe["close"] - dataframe["bb_lowerband"]) / (
            dataframe["bb_upperband"] - dataframe["bb_lowerband"]
        )
        dataframe["bb_width"] = (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe[
            "bb_middleband"
        ]

        # TEMA - 三重指数移动平均线
        dataframe["tema"] = ta.TEMA(dataframe, timeperiod=9)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (
                # 信号：RSI上穿30
                (qtpylib.crossed_above(df["rsi"], self.buy_rsi.value))
                & (df["tema"] <= df["bb_middleband"])  # 保护：tema低于布林带中轨
                & (df["tema"] > df["tema"].shift(1))  # 保护：tema上升
                & (df["volume"] > 0)  # 确保成交量不为0
                & (df["do_predict"] == 1)  # 确保Freqai对预测有信心
                &
                # 只有当Freqai认为趋势朝此方向时才入场
                (df["&s-up_or_down"] == "up")
            ),
            "enter_long",
        ] = 1

        df.loc[
            (
                # 信号：RSI上穿70
                (qtpylib.crossed_above(df["rsi"], self.short_rsi.value))
                & (df["tema"] > df["bb_middleband"])  # 保护：tema高于布林带中轨
                & (df["tema"] < df["tema"].shift(1))  # 保护：tema下降
                & (df["volume"] > 0)  # 确保成交量不为0
                & (df["do_predict"] == 1)  # 确保Freqai对预测有信心
                &
                # 只有当Freqai认为趋势朝此方向时才入场
                (df["&s-up_or_down"] == "down")
            ),
            "enter_short",
        ] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (
                # 信号：RSI上穿70
                (qtpylib.crossed_above(df["rsi"], self.sell_rsi.value))
                & (df["tema"] > df["bb_middleband"])  # 保护：tema高于布林带中轨
                & (df["tema"] < df["tema"].shift(1))  # 保护：tema下降
                & (df["volume"] > 0)  # 确保成交量不为0
            ),
            "exit_long",
        ] = 1

        df.loc[
            (
                # 信号：RSI上穿30
                (qtpylib.crossed_above(df["rsi"], self.exit_short_rsi.value))
                &
                # 保护：tema低于布林带中轨
                (df["tema"] <= df["bb_middleband"])
                & (df["tema"] > df["tema"].shift(1))  # 保护：tema上升
                & (df["volume"] > 0)  # 确保成交量不为0
            ),
            "exit_short",
        ] = 1

        return df