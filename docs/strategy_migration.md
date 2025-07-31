修改后：
class AwesomeStrategy(IStrategy):
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:
    return True
for t in self.freqai_info["feature_parameters"]["indicator_periods_candles"]:

            t = int(t)
            informative[f"%-{pair}rsi-period_{t}"] = ta.RSI(informative, timeperiod=t)
            informative[f"%-{pair}mfi-period_{t}"] = ta.MFI(informative, timeperiod=t)
            informative[f"%-{pair}adx-period_{t}"] = ta.ADX(informative, timeperiod=t)
            informative[f"%-{pair}sma-period_{t}"] = ta.SMA(informative, timeperiod=t)
            informative[f"%-{pair}ema-period_{t}"] = ta.EMA(informative, timeperiod=t)

            bollinger = qtpylib.bollinger_bands(
                qtpylib.typical_price(informative), window=t, stds=2.2
            )
            informative[f"{pair}bb_lowerband-period_{t}"] = bollinger["lower"]
            informative[f"{pair}bb_middleband-period_{t}"] = bollinger["mid"]
            informative[f"{pair}bb_upperband-period_{t}"] = bollinger["upper"]

            informative[f"%-{pair}bb_width-period_{t}"] = (
                informative[f"{pair}bb_upperband-period_{t}"]
                - informative[f"{pair}bb_lowerband-period_{t}"]
            ) / informative[f"{pair}bb_middleband-period_{t}"]
            informative[f"%-{pair}close-bb_lower-period_{t}"] = (
                informative["close"] / informative[f"{pair}bb_lowerband-period_{t}"]
            )

            informative[f"%-{pair}roc-period_{t}"] = ta.ROC(informative, timeperiod=t)

            informative[f"%-{pair}relative_volume-period_{t}"] = (
                informative["volume"] / informative["volume"].rolling(t).mean()
            ) # (1)

        informative[f"%-{pair}pct-change"] = informative["close"].pct_change()
        informative[f"%-{pair}raw_volume"] = informative["volume"]
        informative[f"%-{pair}raw_price"] = informative["close"]
        # (2)

        indicators = [col for col in informative if col.startswith("%")]
        # 此循环复制并平移所有指标，为数据添加时效性特征
        for n in range(self.freqai_info["feature_parameters"]["include_shifted_candles"] + 1):
            if n == 0:
                continue
            informative_shift = informative[indicators].shift(n)
            informative_shift = informative_shift.add_suffix("_shift-" + str(n))
            informative = pd.concat((informative, informative_shift), axis=1)

        df = merge_informative_pair(df, informative, self.config["timeframe"], tf, ffill=True)
        skip_columns = [
            (s + "_" + tf) for s in ["date", "open", "high", "low", "close", "volume"]
        ]
        df = df.drop(columns=skip_columns)

        # 在此处添加通用指标（因为在实盘模式下，训练期间会调用此函数来生成指标）。注意确保不要重复添加
        if set_generalized_indicators:
            df["%-day_of_week"] = (df["date"].dt.dayofweek + 1) / 7
            df["%-hour_of_day"] = (df["date"].dt.hour + 1) / 25
            # (3)

            # 用户在此处添加目标，需以&-为前缀（见下方约定）
            df["&-s_close"] = (
                df["close"]
                .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
                .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
                .mean()
                / df["close"]
                - 1
            )  # (4)

        return df
# 删除这些行
        # data_dictionary = dk.make_train_test_datasets(features_filtered, labels_filtered)
        # self.data_cleaning_train(dk)
        # data_dictionary = dk.normalize_data(data_dictionary)
        # (1)

        # 添加这些行。现在我们自己控制管道的fit/transform过程
        dd = dk.make_train_test_datasets(features_filtered, labels_filtered)
        dk.feature_pipeline = self.define_data_pipeline(threads=dk.thread_count)
        dk.label_pipeline = self.define_label_pipeline(threads=dk.thread_count)

        (dd["train_features"],
         dd["train_labels"],
         dd["train_weights"]) = dk.feature_pipeline.fit_transform(dd["train_features"],
                                                                  dd["train_labels"],
                                                                  dd["train_weights"])

        (dd["test_features"],
         dd["test_labels"],
         dd["test_weights"]) = dk.feature_pipeline.transform(dd["test_features"],
                                                             dd["test_labels"],
                                                             dd["test_weights"])

        dd["train_labels"], _, _ = dk.label_pipeline.fit_transform(dd["train_labels"])
        dd["test_labels"], _, _ = dk.label_pipeline.transform(dd["test_labels"])

        # ... 你的自定义代码

        return model

    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> tuple[DataFrame, npt.NDArray[np.int_]]:

        # ... 你的自定义内容

        # 删除这些行:
        # self.data_cleaning_predict(dk)
        # (2)

        # 添加这些行:
        dk.data_dictionary["prediction_features"], outliers, _ = dk.feature_pipeline.transform(
            dk.data_dictionary["prediction_features"], outlier_check=True)

        # 删除这行
        # pred_df = dk.denormalize_labels_from_metadata(pred_df)
        # (3)

        # 替换为这些行
        pred_df, _, _ = dk.label_pipeline.inverse_transform(pred_df)
        if self.freqai_info.get("DI_threshold", 0) > 0:
            dk.DI_values = dk.feature_pipeline["di"].di_values
        else:
            dk.DI_values = np.zeros(outliers.shape[0])
        dk.do_predict = outliers

        # ... 你的自定义代码
        return (pred_df, dk.do_predict)