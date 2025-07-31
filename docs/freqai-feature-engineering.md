def feature_engineering_expand_all(self, dataframe: DataFrame, period, metadata, **kwargs) -> DataFrame:
        """
        *仅在启用FreqAI的策略中生效*
        此函数将根据配置中定义的 `indicator_periods_candles`、`include_timeframes`、`include_shifted_candles` 和 
        `include_corr_pairs` 自动扩展已定义的特征。换句话说，在此函数中定义的单个特征将自动扩展为 
        `indicator_periods_candles` * `include_timeframes` * `include_shifted_candles` * 
        `include_corr_pairs` 个特征添加到模型中。

        所有特征必须以 `%` 为前缀，才能被 FreqAI 内部识别。

        使用以下方式访问元数据，例如当前交易对/时间框架/周期：

        `metadata["pair"]` `metadata["tf"]`  `metadata["period"]`

        :param df: 将接收特征的策略数据框
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
            dataframe["bb_upperband-period"]
            - dataframe["bb_lowerband-period"]
        ) / dataframe["bb_middleband-period"]
        dataframe["%-close-bb_lower-period"] = (
            dataframe["close"] / dataframe["bb_lowerband-period"]
        )

        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)

        dataframe["%-relative_volume-period"] = (
            dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        )

        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata, **kwargs) -> DataFrame:
        """
        *仅在启用FreqAI的策略中生效*
        此函数将根据配置中定义的 `include_timeframes`、`include_shifted_candles` 和 `include_corr_pairs` 自动扩展已定义的特征。
        换句话说，在此函数中定义的单个特征将自动扩展为 
        `include_timeframes` * `include_shifted_candles` * `include_corr_pairs` 个特征添加到模型中。

        此处定义的特征**不会**根据用户定义的 `indicator_periods_candles` 自动复制。

        使用以下方式访问元数据，例如当前交易对/时间框架：

        `metadata["pair"]` `metadata["tf"]`

        所有特征必须以 `%` 为前缀，才能被 FreqAI 内部识别。

        :param df: 将接收特征的策略数据框
        :param metadata: 当前交易对的元数据
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-ema-200"] = ta.EMA(dataframe, timeperiod=200)
        """
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata, **kwargs) -> DataFrame:
        """
        *仅在启用FreqAI的策略中生效*
        此可选函数将使用基础时间框架的数据框调用一次。这是最后被调用的函数，意味着进入此函数的数据框将包含由其他 
        `feature_engineering_expand` 函数创建的基础资产的所有特征和列。此函数是进行自定义复杂特征提取（例如 tsfresh）的理想位置。
        此函数也是放置不应自动扩展的特征（例如星期几）的理想位置。

        使用以下方式访问元数据，例如当前交易对：

        `metadata["pair"]`

        所有特征必须以 `%` 为前缀，才能被 FreqAI 内部识别。

        :param df: 将接收特征的策略数据框
        :param metadata: 当前交易对的元数据
        使用示例：dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        """
        dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        dataframe["%-hour_of_day"] = (dataframe["date"].dt.hour + 1) / 25
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata, **kwargs) -> DataFrame:
        """
        *仅在启用FreqAI的策略中生效*
        设置模型目标的必需函数。所有目标必须以 `&` 为前缀，才能被 FreqAI 内部识别。

        使用以下方式访问元数据，例如当前交易对：

        `metadata["pair"]`

        :param df: 将接收目标的策略数据框
        :param metadata: 当前交易对的元数据
        使用示例：dataframe["&-target"] = dataframe["close"].shift(-1) / dataframe["close"]
        """
        dataframe["&-s_close"] = (
            dataframe["close"]
            .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
            .mean()
            / dataframe["close"]
            - 1
            )
        
        return dataframe