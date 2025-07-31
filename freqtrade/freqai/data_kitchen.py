import copy
import inspect
import logging
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import psutil
from datasieve.pipeline import Pipeline
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from freqtrade.configuration import TimeRange
from freqtrade.constants import DOCS_LINK, ORDERFLOW_ADDED_COLUMNS, Config
from freqtrade.data.converter import reduce_dataframe_footprint
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_seconds
from freqtrade.strategy import merge_informative_pair
from freqtrade.strategy.interface import IStrategy


pd.set_option("future.no_silent_downcasting", True)

SECONDS_IN_DAY = 86400
SECONDS_IN_HOUR = 3600

logger = logging.getLogger(__name__)


class FreqaiDataKitchen:
    """
    为单个交易对分析数据的类。由IFreqaiModel类使用。
    功能包括保存、加载和分析数据。

    该对象不是持久化的，每次需要推理或训练交易对模型时都会重新实例化。

    贡献记录:
    FreqAI是由一群人开发的，他们都为这个项目贡献了特定的技能。

    概念和软件开发:
    Robert Caulk @robcaulk

    理论构思:
    Elin Törnquist @th0rntwig

    代码审查，软件架构构思:
    @xmatthias

    Beta测试和错误报告:
    @bloodhunter4rc, Salah Lamkadem @ikonx, @ken11o2, @longyu, @paranoidandy, @smidelis, @smarm
    Juha Nykänen @suikula, Wagner Costa @wagnercosta, Johan Vlugt @Jooopieeert
    """

    def __init__(
        self,
        config: Config,
        live: bool = False,
        pair: str = "",
    ):
        self.data: dict[str, Any] = {}                  # 存储各种数据和元数据
        self.data_dictionary: dict[str, DataFrame] = {} # 存储训练和测试数据
        self.config = config
        self.freqai_config: dict[str, Any] = config["freqai"]
        self.full_df: DataFrame = DataFrame()           # 完整的数据帧
        self.append_df: DataFrame = DataFrame()         # 待追加的数据帧
        self.data_path = Path()                         # 数据路径
        self.label_list: list = []                      # 标签列表
        self.training_features_list: list = []          # 训练特征列表
        self.model_filename: str = ""                   # 模型文件名
        self.backtesting_results_path = Path()          # 回测结果路径
        self.backtest_predictions_folder: str = "backtesting_predictions"  # 回测预测文件夹
        self.live = live                                # 是否为实盘模式
        self.pair = pair                                # 当前交易对
        self.keras: bool = self.freqai_config.get("keras", False)  # 是否使用Keras
        self.set_all_pairs()                            # 设置        self.backtest_live_models = config.get("freqai_backtest_live_models", False)  # 是否使用实盘模型回测
        self.feature_pipeline = Pipeline()              # 特征处理管道
        self.label_pipeline = Pipeline()                # 标签处理管道
        self.DI_values: npt.NDArray = np.array([])      # DI值数组

        if not self.live:
            self.full_path = self.get_full_models_path(self.config)

            if not self.backtest_live_models:
                self.full_timerange = self.create_fulltimerange(
                    self.config["timerange"], self.freqai_config.get("train_period_days", 0)
                )
                (self.training_timeranges, self.backtesting_timeranges) = self.split_timerange(
                    self.full_timerange,
                    config["freqai"]["train_period_days"],
                    config["freqai"]["backtest_period_days"],
                )

        self.data["extra_returns_per_train"] = self.freqai_config.get("extra_returns_per_train", {})
        if not self.freqai_config.get("data_kitchen_thread_count", 0):
            self.thread_count = max(int(psutil.cpu_count() * 2 - 2), 1)
        else:
            self.thread_count = self.freqai_config["data_kitchen_thread_count"]
        self.train_dates: DataFrame = pd.DataFrame()     # 训练日期
        self.unique_classes: dict[str, list] = {}        # 唯一类别
        self.unique_class_list: list = []                # 唯一类别列表
        self.backtest_live_models_data: dict[str, Any] = {}  # 实盘模型回测数据

    def set_paths(
        self,
        pair: str,
        trained_timestamp: int | None = None,
    ) -> None:
        """
        设置当前交易对/机器人循环的数据路径
        :param metadata: 策略提供的交易对元数据
        :param trained_timestamp: 最近训练的时间戳
        """
        self.full_path = self.get_full_models_path(self.config)
        self.data_path = Path(
            self.full_path / f"sub-train-{pair.split('/')[0]}_{trained_timestamp}"
        )

        return

    def make_train_test_datasets(
        self, filtered_dataframe: DataFrame, labels: DataFrame
    ) -> dict[Any, Any]:
        """
        给定用于训练的完整历史数据帧，根据配置文件中用户指定的参数将数据拆分为训练和测试数据。
        :param filtered_dataframe: 准备好拆分的清洗后的数据帧。
        :param labels: 准备好拆分的清洗后标签。
        """
        feat_dict = self.freqai_config["feature_parameters"]

        if "shuffle" not in self.freqai_config["data_split_parameters"]:
            self.freqai_config["data_split_parameters"].update({"shuffle": False})

        weights: npt.ArrayLike
        if feat_dict.get("weight_factor", 0) > 0:
            weights = self.set_weights_higher_recent(len(filtered_dataframe))
        else:
            weights = np.ones(len(filtered_dataframe))

        if self.freqai_config.get("data_split_parameters", {}).get("test_size", 0.1) != 0:
            (
                train_features,
                test_features,
                train_labels,
                test_labels,
                train_weights,
                test_weights,
            ) = train_test_split(
                filtered_dataframe[: filtered_dataframe.shape[0]],
                labels,
                weights,** self.config["freqai"]["data_split_parameters"],
            )
        else:
            test_labels = np.zeros(2)
            test_features = pd.DataFrame()
            test_weights = np.zeros(2)
            train_features = filtered_dataframe
            train_labels = labels
            train_weights = weights

        if feat_dict["shuffle_after_split"]:
            rint1 = random.randint(0, 100)
            rint2 = random.randint(0, 100)
            train_features = train_features.sample(frac=1, random_state=rint1).reset_index(
                drop=True
            )
            train_labels = train_labels.sample(frac=1, random_state=rint1).reset_index(drop=True)
            train_weights = (
                pd.DataFrame(train_weights)
                .sample(frac=1, random_state=rint1)
                .reset_index(drop=True)
                .to_numpy()[:, 0]
            )
            test_features = test_features.sample(frac=1, random_state=rint2).reset_index(drop=True)
            test_labels = test_labels.sample(frac=1, random_state=rint2).reset_index(drop=True)
            test_weights = (
                pd.DataFrame(test_weights)
                .sample(frac=1, random_state=rint2)
                .reset_index(drop=True)
                .to_numpy()[:, 0]
            )

        # 反转训练和测试数据顺序的最简单方法:
        if self.freqai_config["feature_parameters"].get("reverse_train_test_order", False):
            return self.build_data_dictionary(
                test_features,
                train_features,
                test_labels,
                train_labels,
                test_weights,
                train_weights,
            )
        else:
            return self.build_data_dictionary(
                train_features,
                test_features,
                train_labels,
                test_labels,
                train_weights,
                test_weights,
            )

    def filter_features(
        self,
        unfiltered_df: DataFrame,
        training_feature_list: list,
        label_list: list | None = None,
        training_filter: bool = True,
    ) -> tuple[DataFrame, DataFrame]:
        """
        过滤未过滤的数据帧以提取用户请求的特征/标签，并适当移除所有NaN值。
        任何包含NaN的行都会从训练数据集中移除，或在预测数据集中替换为0。
        然而，预测数据集的do_predict将反映任何有NaN的行，并将用户与该预测隔离。

        :param unfiltered_df: 当前训练期间的完整数据帧
        :param training_feature_list: 列表，由self.build_feature_list()根据配置文件中用户指定的参数构建的训练特征列表。
        :param labels: 数据集的标签
        :param training_filter: 布尔值，让函数知道是要过滤训练数据还是预测数据。
        :returns:
        :filtered_df: 清理了NaN且只包含用户请求的特征集的数据帧。
        :labels: 清理了NaN的标签。
        """
        filtered_df = unfiltered_df.filter(training_feature_list, axis=1)
        filtered_df = filtered_df.replace([np.inf, -np.inf], np.nan)

        drop_index = pd.isnull(filtered_df).any(axis=1)  # 获取有NaN的行，
        drop_index = drop_index.replace(True, 1).replace(False, 0).infer_objects(copy=False)
        if training_filter:
            # 我们不关心训练中的总行数（数据点总数），只关心移除任何有NaN的行
            # 如果标签有多个列（用户想训练多个模型），我们在这里检测
            labels = unfiltered_df.filter(label_list or [], axis=1)
            drop_index_labels = pd.isnull(labels).any(axis=1)
            drop_index_labels = (
                drop_index_labels.replace(True, 1).replace(False, 0).infer_objects(copy=False)
            )
            dates = unfiltered_df["date"]
            filtered_df = filtered_df[
                (drop_index == 0) & (drop_index_labels == 0)
            ]  # 丢弃值
            labels = labels[
                (drop_index == 0) & (drop_index_labels == 0)
            ]  # 假设标签完全依赖于这里的数据帧。
            self.train_dates = dates[(drop_index == 0) & (drop_index_labels == 0)]
            logger.info(
                f"{self.pair}: 由于填充数据集中的NaN，丢弃了 {len(unfiltered_df) - len(filtered_df)} 个训练点 {len(unfiltered_df)}。"
            )
            if len(filtered_df) == 0 and not self.live:
                raise OperationalException(
                    f"{self.pair}: 所有训练数据因NaN被丢弃。"
                    " 您可能在回测时间范围之前没有下载足够的训练数据。提示:\n"
                    f"{DOCS_LINK}/freqai-running/"
                    "#downloading-data-to-cover-the-full-backtest-period"
                )
            if (1 - len(filtered_df) / len(unfiltered_df)) > 0.1 and self.live:
                worst_indicator = str(unfiltered_df.count().idxmin())
                logger.warning(
                    f"  {(1 - len(filtered_df) / len(unfiltered_df)) * 100:.0f}% "
                    " 的训练数据因NaN被丢弃，模型性能可能与预期不一致 "
                    f"请验证 {worst_indicator}"
                )
            self.data["filter_drop_index_training"] = drop_index

        else:
            # 我们正在回测，所以需要保留行数以发送回策略，
            # 所以现在我们使用do_predict来避免任何基于NaN的预测
            drop_index = pd.isnull(filtered_df).any(axis=1)
            self.data["filter_drop_index_prediction"] = drop_index
            filtered_df.fillna(0, inplace=True)
            # 将所有NaN替换为零，以避免'prediction'中的问题，但任何基于单个NaN的预测最终都会通过do_predict保护不被买入
            drop_index = ~drop_index
            self.do_predict = np.array(drop_index.replace(True, 1).replace(False, 0))
            if (len(self.do_predict) - self.do_predict.sum()) > 0:
                logger.info(
                    "由于NaN，丢弃了 %s/%s 个预测数据点。",
                    len(self.do_predict) - self.do_predict.sum(),
                    len(filtered_df),
                )
            labels = []

        return filtered_df, labels

    def build_data_dictionary(
        self,
        train_df: DataFrame,
        test_df: DataFrame,
        train_labels: DataFrame,
        test_labels: DataFrame,
        train_weights: Any,
        test_weights: Any,
    ) -> dict:
        """构建包含训练和测试数据的数据字典"""
        self.data_dictionary = {
            "train_features": train_df,
            "test_features": test_df,
            "train_labels": train_labels,
            "test_labels": test_labels,
            "train_weights": train_weights,
            "test_weights": test_weights,
            "train_dates": self.train_dates,
        }

        return self.data_dictionary

    def split_timerange(
        self, tr: str, train_split: int = 28, bt_split: float = 7
    ) -> tuple[list, list]:
        """
        将单个时间范围(tr)根据用户输入拆分为用于训练和回测的子时间范围
        tr: str，用于训练的完整时间范围
        train_split: 每个训练的周期长度（天）。在用户配置文件中指定
        bt_split: 回测长度（天）。在用户配置文件中指定
        """

        if not isinstance(train_split, int) or train_split < 1:
            raise OperationalException(
                f"train_period_days必须是大于0的整数。得到 {train_split}。"
            )
        train_period_days = train_split * SECONDS_IN_DAY
        bt_period = bt_split * SECONDS_IN_DAY

        full_timerange = TimeRange.parse_timerange(tr)
        config_timerange = TimeRange.parse_timerange(self.config["timerange"])
        if config_timerange.stopts == 0:
            config_timerange.stopts = int(datetime.now(tz=timezone.utc).timestamp())
        timerange_train = copy.deepcopy(full_timerange)
        timerange_backtest = copy.deepcopy(full_timerange)

        tr_training_list = []
        tr_backtesting_list = []
        tr_training_list_timerange = []
        tr_backtesting_list_timerange = []
        first = True

        while True:
            if not first:
                timerange_train.startts = timerange_train.startts + int(bt_period)
            timerange_train.stopts = timerange_train.startts + train_period_days

            first = False
            tr_training_list.append(timerange_train.timerange_str)
            tr_training_list_timerange.append(copy.deepcopy(timerange_train))

            # 相关的回测周期
            timerange_backtest.startts = timerange_train.stopts
            timerange_backtest.stopts = timerange_backtest.startts + int(bt_period)

            if timerange_backtest.stopts > config_timerange.stopts:
                timerange_backtest.stopts = config_timerange.stopts

            tr_backtesting_list.append(timerange_backtest.timerange_str)
            tr_backtesting_list_timerange.append(copy.deepcopy(timerange_backtest))

            # 确保我们预测的数据量与用户定义的--timerange所请求的完全相同
            if timerange_backtest.stopts == config_timerange.stopts:
                break

        return tr_training_list_timerange, tr_backtesting_list_timerange

    def slice_dataframe(self, timerange: TimeRange, df: DataFrame) -> DataFrame:
        """
        给定完整的数据帧，提取用户所需的窗口
        :param tr: 我们希望从df中提取的时间范围字符串
        :param df: 包含运行整个回测的所有K线的数据帧。这里它被切片到当前训练周期。
        """
        if not self.live:
            df = df.loc[(df["date"] >= timerange.startdt) & (df["date"] < timerange.stopdt), :]
        else:
            df = df.loc[df["date"] >= timerange.startdt, :]

        return df

    def find_features(self, dataframe: DataFrame) -> None:
        """
        在策略提供的数据帧中查找特征
        :param dataframe: 策略提供的数据帧
        :return:
        features: 用于训练/预测的特征列表
        """
        column_names = dataframe.columns
        features = [c for c in column_names if "%" in c]

        if not features:
            raise OperationalException("找不到任何特征!")

        self.training_features_list = features

    def find_labels(self, dataframe: DataFrame) -> None:
        """在数据帧中查找标签（以&开头的列）"""
        column_names = dataframe.columns
        labels = [c for c in column_names if "&" in c]
        self.label_list = labels

    def set_weights_higher_recent(self, num_weights: int) -> npt.ArrayLike:
        """
        设置权重，使最近的数据在训练期间比旧数据具有更高的权重。
        """
        wfactor = self.config["freqai"]["feature_parameters"]["weight_factor"]
        weights = np.exp(-np.arange(num_weights) / (wfactor * num_weights))[::-1]
        return weights

    def get_predictions_to_append(
        self, predictions: DataFrame, do_predict: npt.ArrayLike, dataframe_backtest: DataFrame
    ) -> DataFrame:
        """获取当前回测周期的回测预测"""

        append_df = DataFrame()
        for label in predictions.columns:
            append_df[label] = predictions[label]
            if append_df[label].dtype == object:
                continue
            if "labels_mean" in self.data:
                append_df[f"{label}_mean"] = self.data["labels_mean"][label]
            if "labels_std" in self.data:
                append_df[f"{label}_std"] = self.data["labels_std"][label]

        for extra_col in self.data["extra_returns_per_train"]:
            append_df[f"{extra_col}"] = self.data["extra_returns_per_train"][extra_col]

        append_df["do_predict"] = do_predict
        if self.freqai_config["feature_parameters"].get("DI_threshold", 0) > 0:
            append_df["DI_values"] = self.DI_values

        user_cols = [col for col in dataframe_backtest.columns if col.startswith("%%")]
        cols = ["date"]
        cols.extend(user_cols)

        dataframe_backtest.reset_index(drop=True, inplace=True)
        merged_df = pd.concat([dataframe_backtest[cols], append_df], axis=1)
        return merged_df

    def append_predictions(self, append_df: DataFrame) -> None:
        """将当前回测周期的回测预测追加到所有先前的周期"""

        if self.full_df.empty:
            self.full_df = append_df
        else:
            self.full_df = pd.concat([self.full_df, append_df], axis=0, ignore_index=True)

    def fill_predictions(self, dataframe):
        """
        回填回测范围之前的值，以便数据帧返回策略时大小匹配。这些行不包括在回测中。
        """
        to_keep = [
            col for col in dataframe.columns if not col.startswith("&") and not col.startswith("%%")
        ]
        self.return_dataframe = pd.merge(dataframe[to_keep], self.full_df, how="left", on="date")
        self.return_dataframe[self.full_df.columns] = self.return_dataframe[
            self.full_df.columns
        ].fillna(value=0)
        self.full_df = DataFrame()

        return

    def create_fulltimerange(self, backtest_tr: str, backtest_period_days: int) -> str:
        """创建完整的时间范围，包括训练和回测期"""
        if not isinstance(backtest_period_days, int):
            raise OperationalException("backtest_period_days必须是整数")

        if backtest_period_days < 0:
            raise OperationalException("backtest_period_days必须为正数")

        backtest_timerange = TimeRange.parse_timerange(backtest_tr)

        if backtest_timerange.stopts == 0:
            # 通常开放式时间范围是有效的，但是，在某些边缘情况下它不是。
            # 仅仅为了允许开放式时间范围而适应这些类型的边缘情况的优先级不足以保证付出努力。
            # 现在，简单地要求用户添加他们的结束日期更安全
            raise OperationalException(
                "FreqAI回测不允许开放式时间范围。"
                " 请指明您期望的回测结束日期。"
                "时间范围。"
            )

        backtest_timerange.startts = (
            backtest_timerange.startts - backtest_period_days * SECONDS_IN_DAY
        )
        full_timerange = backtest_timerange.timerange_str
        config_path = Path(self.config["config_files"][0])

        if not self.full_path.is_dir():
            self.full_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(
                config_path.resolve(),
                Path(self.full_path / config_path.parts[-1]),
            )

        return full_timerange

    def check_if_model_expired(self, trained_timestamp: int) -> bool:
        """
        模型年龄检查器，根据用户在配置文件中定义的`expiration_hours`确定模型是否可信。
        :param trained_timestamp: 最近模型的训练时间。
        :return:
            bool = 模型是否过期。
        """
        time = datetime.now(tz=timezone.utc).timestamp()
        elapsed_time = (time - trained_timestamp) / 3600  # 小时
        max_time = self.freqai_config.get("expiration_hours", 0)
        if max_time > 0:
            return elapsed_time > max_time
        else:
            return False

    def check_if_new_training_required(
        self, trained_timestamp: int
    ) -> tuple[bool, TimeRange, TimeRange]:
        """检查是否需要新的训练"""
        time = datetime.now(tz=timezone.utc).timestamp()
        trained_timerange = TimeRange()
        data_load_timerange = TimeRange()

        timeframes = self.freqai_config["feature_parameters"].get("include_timeframes")

        max_tf_seconds = 0
        for tf in timeframes:
            secs = timeframe_to_seconds(tf)
            if secs > max_tf_seconds:
                max_tf_seconds = secs

        # 我们注意到用户喜欢使用他们不知道所需时间周期的特殊指标。
        # 这里我们通过将用户认为的"最大"乘以2来包含安全系数。
        max_period = self.config.get("startup_candle_count", 20) * 2
        additional_seconds = max_period * max_tf_seconds

        if trained_timestamp != 0:
            elapsed_time = (time - trained_timestamp) / SECONDS_IN_HOUR
            retrain = elapsed_time > self.freqai_config.get("live_retrain_hours", 0)
            if retrain:
                trained_timerange.startts = int(
                    time - self.freqai_config.get("train_period_days", 0) * SECONDS_IN_DAY
                )
                trained_timerange.stopts = int(time)
                # 我们希望加载/填充比我们计划训练的数据更多的数据，因为大多数指标都有滚动时间周期，
                # 因此除非在训练周期开始之前有更早的数据，否则它们将是NaN
                data_load_timerange.startts = int(
                    time
                    - self.freqai_config.get("train_period_days", 0) * SECONDS_IN_DAY
                    - additional_seconds
                )
                data_load_timerange.stopts = int(time)
        else:  # 用户在配置中没有传递live_trained_timerange
            trained_timerange.startts = int(
                time - self.freqai_config.get("train_period_days", 0) * SECONDS_IN_DAY
            )
            trained_timerange.stopts = int(time)

            data_load_timerange.startts = int(
                time
                - self.freqai_config.get("train_period_days", 0) * SECONDS_IN_DAY
                - additional_seconds
            )
            data_load_timerange.stopts = int(time)
            retrain = True

        return retrain, trained_timerange, data_load_timerange

    def set_new_model_names(self, pair: str, timestamp_id: int):
        """设置新模型的名称和路径"""
        coin, _ = pair.split("/")
        self.data_path = Path(self.full_path / f"sub-train-{pair.split('/')[0]}_{timestamp_id}")

        self.model_filename = f"cb_{coin.lower()}_{timestamp_id}"

    def set_all_pairs(self) -> None:
        """设置所有相关交易对"""
        self.all_pairs = copy.deepcopy(
            self.freqai_config["feature_parameters"].get("include_corr_pairlist", [])
        )
        for pair in self.config.get("exchange", "").get("pair_whitelist"):
            if pair not in self.all_pairs:
                self.all_pairs.append(pair)

    def extract_corr_pair_columns_from_populated_indicators(
        self, dataframe: DataFrame
    ) -> dict[str, DataFrame]:
        """
        查找与corr_pairlist对应的dataframe列，将它们保存在字典中以便重用并附加到其他交易对。

        :param dataframe: 完全填充的数据帧（当前交易对+相关交易对）
        :return: corr_dataframes，要附加到同一K线中其他交易对的数据帧字典。
        """
        corr_dataframes: dict[str, DataFrame] = {}
        pairs = self.freqai_config["feature_parameters"].get("include_corr_pairlist", [])

        for pair in pairs:
            pair = pair.replace(":", "")  # lightgbm不喜欢冒号
            pair_cols = [
                col for col in dataframe.columns if col.startswith("%") and f"{pair}_" in col
            ]

            if pair_cols:
                pair_cols.insert(0, "date")
                corr_dataframes[pair] = dataframe.filter(pair_cols, axis=1)

        return corr_dataframes

    def attach_corr_pair_columns(
        self, dataframe: DataFrame, corr_dataframes: dict[str, DataFrame], current_pair: str
    ) -> DataFrame:
        """
        在训练前将现有的相关交易对数据帧附加到当前交易对数据帧

        :param dataframe: 当前交易对策略数据帧，指标已填充
        :param corr_dataframes: 同一K线中早期保存的数据帧字典
        :param current_pair: 要附加相关交易对数据帧的当前交易对
        :return:
        :dataframe: 已填充指标的当前交易对数据帧，与相关交易对连接，准备训练
        """
        pairs = self.freqai_config["feature_parameters"].get("include_corr_pairlist", [])
        current_pair = current_pair.replace(":", "")
        for pair in pairs:
            pair = pair.replace(":", "")  # lightgbm不支持冒号
            if current_pair != pair:
                dataframe = dataframe.merge(corr_dataframes[pair], how="left", on="date")

        return dataframe

    def get_pair_data_for_features(
        self,
        pair: str,
        tf: str,
        strategy: IStrategy,
        corr_dataframes: dict,
        base_dataframes: dict,
        is_corr_pairs: bool = False,
    ) -> DataFrame:
        """
        获取交易对的数据。如果不在字典中，则从数据提供者获取
        :param pair: 要获取数据的交易对
        :param tf: 要获取数据的时间框架
        :param strategy: 用户定义的策略对象
        :param corr_dataframes: 包含df交易对数据帧的字典（对于用户定义的时间框架）
        :param base_dataframes: 包含当前交易对数据帧的字典（对于用户定义的时间框架）
        :param is_corr_pairs: 该交易对是否为相关交易对
        :return: 包含交易对数据的数据帧
        """
        if is_corr_pairs:
            dataframe = corr_dataframes[pair][tf]
            if not dataframe.empty:
                return dataframe
            else:
                dataframe = strategy.dp.get_pair_dataframe(pair=pair, timeframe=tf)
                return dataframe
        else:
            dataframe = base_dataframes[tf]
            if not dataframe.empty:
                return dataframe
            else:
                dataframe = strategy.dp.get_pair_dataframe(pair=pair, timeframe=tf)
                return dataframe

    def merge_features(
        self, df_main: DataFrame, df_to_merge: DataFrame, tf: str, timeframe_inf: str, suffix: str
    ) -> DataFrame:
        """
        合并数据帧的特征并移除HLCV和添加的日期列
        :param df_main: 主数据帧
        :param df_to_merge: 要合并的数据帧
        :param tf: 主数据帧的时间框架
        :param timeframe_inf: 要合并的数据帧的时间框架
        :param suffix: 要添加到要合并的数据帧列的后缀
        :return: 合并后的数据帧
        """
        dataframe = merge_informative_pair(
            df_main,
            df_to_merge,
            tf,
            timeframe_inf=timeframe_inf,
            append_timeframe=False,
            suffix=suffix,
            ffill=True,
        )
        skip_columns = [
            (f"{s}_{suffix}") for s in ["date", "open", "high", "low", "close", "volume"]
        ]

        for s in ORDERFLOW_ADDED_COLUMNS:
            if s in dataframe.columns and f"{s}_{suffix}" in dataframe.columns:
                skip_columns.append(f"{s}_{suffix}")

        dataframe = dataframe.drop(columns=skip_columns)
        return dataframe

    def populate_features(
        self,
        dataframe: DataFrame,
        pair: str,
        strategy: IStrategy,
        corr_dataframes: dict,
        base_dataframes: dict,
        is_corr_pairs: bool = False,
    ) -> DataFrame:
        """
        使用用户定义的策略函数填充特征
        :param dataframe: 要填充的数dataframe
        :param pair: 要填充的交易对
        :param strategy: 用户定义的策略对象
        :param corr_dataframes: 包含df交易对数据帧的字典
        :param base_dataframes: 包含当前交易对数据帧的字典
        :param is_corr_pairs: 该交易对是否为相关交易对
        :return: 已填充特征的数据帧
        """
        tfs: list[str] = self.freqai_config["feature_parameters"].get("include_timeframes")

        for tf in tfs:
            metadata = {"pair": pair, "tf": tf}
            informative_df = self.get_pair_data_for_features(
                pair, tf, strategy, corr_dataframes, base_dataframes, is_corr_pairs
            )
            informative_copy = informative_df.copy()

            logger.debug(f"为 {pair} {tf} 填充特征")

            for t in self.freqai_config["feature_parameters"]["indicator_periods_candles"]:
                df_features = strategy.feature_engineering_expand_all(
                    informative_copy.copy(), t, metadata=metadata
                )
                suffix = f"{t}"
                informative_df = self.merge_features(informative_df, df_features, tf, tf, suffix)

            generic_df = strategy.feature_engineering_expand_basic(
                informative_copy.copy(), metadata=metadata
            )
            suffix = "gen"

            informative_df = self.merge_features(informative_df, generic_df, tf, tf, suffix)

            indicators = [col for col in informative_df if col.startswith("%")]
            for n in range(self.freqai_config["feature_parameters"]["include_shifted_candles"] + 1):
                if n == 0:
                    continue
                df_shift = informative_df[indicators].shift(n)
                df_shift = df_shift.add_suffix("_shift-" + str(n))
                informative_df = pd.concat((informative_df, df_shift), axis=1)

            dataframe = self.merge_features(
                dataframe.copy(), informative_df, self.config["timeframe"], tf, f"{pair}_{tf}"
            )

        return dataframe

    def use_strategy_to_populate_indicators(  # noqa: C901
        self,
        strategy: IStrategy,
        corr_dataframes: dict[str, DataFrame] | None = None,
        base_dataframes: dict[str, dict[str, DataFrame]] | None = None,
        pair: str = "",
        prediction_dataframe: DataFrame | None = None,
        do_corr_pairs: bool = True,
    ) -> DataFrame:
        """
        在重新训练期间使用用户定义的策略填充指标
        :param strategy: 用户定义的策略对象
        :param corr_dataframes: 包含df交易对数据帧的字典（对于用户定义的时间框架）
        :param base_dataframes: 包含当前交易对数据帧的字典（对于用户定义的时间框架）
        :param pair: 要填充的交易对
        :param prediction_dataframe: 包含用于预测的交易对数据的数据帧
        :param do_corr_pairs: 是否填充相关交易对
        :return:
        dataframe: 包含已填充指标的数据帧
        """
        if not corr_dataframes:
            corr_dataframes = {}
        if not base_dataframes:
            base_dataframes = {}

        # 检查用户是否使用已弃用的populate_any_indicators函数
        new_version = inspect.getsource(strategy.populate_any_indicators) == (
            inspect.getsource(IStrategy.populate_any_indicators)
        )

        if not new_version:
            raise OperationalException(
                "您正在使用`populate_any_indicators()`函数，该函数已于2023年3月1日弃用。"
                " 请参考策略迁移指南使用新的feature_engineering_*方法：\n"
                f"{DOCS_LINK}/strategy_migration/#freqai-strategy \n"
                "以及feature_engineering_*文档：\n"
                f"{DOCS_LINK}/freqai-feature-engineering/"
            )

        tfs: list[str] = self.freqai_config["feature_parameters"].get("include_timeframes")
        pairs: list[str] = self.freqai_config["feature_parameters"].get("include_corr_pairlist", [])

        for tf in tfs:
            if tf not in base_dataframes:
                base_dataframes[tf] = pd.DataFrame()
            for p in pairs:
                if p not in corr_dataframes:
                    corr_dataframes[p] = {}
                if tf not in corr_dataframes[p]:
                    corr_dataframes[p][tf] = pd.DataFrame()

        if prediction_dataframe is not None and not prediction_dataframe.empty:
            dataframe = prediction_dataframe.copy()
            base_dataframes[self.config["timeframe"]] = dataframe.copy()
        else:
            dataframe = base_dataframes[self.config["timeframe"]].copy()

        corr_pairs: list[str] = self.freqai_config["feature_parameters"].get(
            "include_corr_pairlist", []
        )
        dataframe = self.populate_features(
            dataframe.copy(), pair, strategy, corr_dataframes, base_dataframes
        )
        metadata = {"pair": pair}
        dataframe = strategy.feature_engineering_standard(dataframe.copy(), metadata=metadata)
        # 确保相关交易对总是最后
        for corr_pair in corr_pairs:
            if pair == corr_pair:
                continue  # 不要重复白名单中的任何内容
            if corr_pairs and do_corr_pairs:
                dataframe = self.populate_features(
                    dataframe.copy(), corr_pair, strategy, corr_dataframes, base_dataframes, True
                )

        if self.live:
            dataframe = strategy.set_freqai_targets(dataframe.copy(), metadata=metadata)
            dataframe = self.remove_special_chars_from_feature_names(dataframe)

            self.get_unique_classes_from_labels(dataframe)

        if self.config.get("reduce_df_footprint", False):
            dataframe = reduce_dataframe_footprint(dataframe)

        return dataframe

    def fit_labels(self) -> None:
        """用高斯分布拟合标签"""
        import scipy as spy

        self.data["labels_mean"], self.data["labels_std"] = {}, {}
        for label in self.data_dictionary["train_labels"].columns:
            if self.data_dictionary["train_labels"][label].dtype == object:
                continue
            f = spy.stats.norm.fit(self.data_dictionary["train_labels"][label])
            self.data["labels_mean"][label], self.data["labels_std"][label] = f[0], f[1]

        # 以防目标是分类
        for label in self.unique_class_list:
            self.data["labels_mean"][label], self.data["labels_std"][label] = 0, 0

        return

    def remove_features_from_df(self, dataframe: DataFrame) -> DataFrame:
        """
        在将数据帧返回给策略之前从数据帧中移除特征。这使其在Frequi中保持紧凑。
        """
        to_keep = [
            col for col in dataframe.columns if not col.startswith("%") or col.startswith("%%")
        ]
        return dataframe[to_keep]

    def get_unique_classes_from_labels(self, dataframe: DataFrame) -> None:
        """从标签中获取唯一类别"""
        # self.find_features(dataframe)
        self.find_labels(dataframe)

        for key in self.label_list:
            if dataframe[key].dtype == object:
                self.unique_classes[key] = dataframe[key].dropna().unique()

        if self.unique_classes:
            for label in self.unique_classes:
                self.unique_class_list += list(self.unique_classes[label])

    def save_backtesting_prediction(self, append_df: DataFrame) -> None:
        """
        将回测的预测数据帧保存为feather文件格式
        :param append_df: 回测期间的数据帧
        """
        full_predictions_folder = Path(self.full_path / self.backtest_predictions_folder)
        if not full_predictions_folder.is_dir():
            full_predictions_folder.mkdir(parents=True, exist_ok=True)

        append_df.to_feather(self.backtesting_results_path)

    def get_backtesting_prediction(self) -> DataFrame:
        """从feather文件格式获取预测数据帧"""
        append_df = pd.read_feather(self.backtesting_results_path)
        return append_df

    def check_if_backtest_prediction_is_valid(self, len_backtest_df: int) -> bool:
        """
        检查回测预测是否已存在，以及要追加的预测是否与回测数据帧切片的大小相同
        :param length_backtesting_dataframe: 回测数据帧切片的长度
        :return:
        :boolean: 预测文件是否有效。
        """
        path_to_predictionfile = Path(
            self.full_path
            / self.backtest_predictions_folder
            / f"{self.model_filename}_prediction.feather"
        )
        self.backtesting_results_path = path_to_predictionfile

        file_exists = path_to_predictionfile.is_file()

        if file_exists:
            append_df = self.get_backtesting_prediction()
            if len(append_df) == len_backtest_df and "date" in append_df:
                logger.info(f"在 {path_to_predictionfile} 找到回测预测文件")
                return True
            else:
                logger.info(
                    "需要新的回测预测文件。"
                    "（预测数量与数据帧长度不同或旧预测文件版本）。"
                )
                return False
        else:
            logger.info(f"在 {path_to_predictionfile} 找不到回测预测文件")
            return False

    def get_full_models_path(self, config: Config) -> Path:
        """
        返回默认的FreqAI模型路径
        :param config: 配置字典
        """
        freqai_config: dict[str, Any] = config["freqai"]
        return Path(config["user_data_dir"] / "models" / str(freqai_config.get("identifier")))

    def remove_special_chars_from_feature_names(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        从特征字符串中移除所有特殊字符(:)
        :param dataframe: 刚刚完成指标填充的数据帧。(未过滤)
        :return: 具有清理后的特征名称的数据帧
        """

        spec_chars = [":"]
        for c in spec_chars:
            dataframe.columns = dataframe.columns.str.replace(c, "")

        return dataframe

    def buffer_timerange(self, timerange: TimeRange):
        """
        缓冲时间范围的开始和结束。这在指标填充*之后*使用。

        主要示例用途是在预测最大值和最小值时，argrelextrema函数无法知道时间范围边缘的最大值/最小值。
        为了提高模型准确性，最好在完整时间范围内计算argrelextrema，然后使用此函数按内核截断边缘（缓冲区）。

        在另一种情况下，如果目标设置为偏移的价格变动，则此缓冲区是不必要的，因为时间范围末尾的偏移K线将是NaN，
        FreqAI会自动从训练数据集中删除这些K线。
        """
        buffer = self.freqai_config["feature_parameters"]["buffer_train_data_candles"]
        if buffer:
            timerange.stopts -= buffer * timeframe_to_seconds(self.config["timeframe"])
            timerange.startts += buffer * timeframe_to_seconds(self.config["timeframe"])

        return timerange

    # 已弃用的函数
    def normalize_data(self, data_dictionary: dict) -> dict[Any, Any]:
        """
        弃用警告，迁移帮助
        """
        logger.warning(
            f"您的自定义IFreqaiModel依赖于已弃用的数据管道。请更新您的模型以使用新的数据管道。"
            " 这可以通过遵循迁移指南来实现 "
            f"{DOCS_LINK}/strategy_migration/#freqai-new-data-pipeline "
            "我们为您添加了一个基本管道，但这将在未来版本中移除。"
        )

        return data_dictionary

    def denormalize_labels_from_metadata(self, df: DataFrame) -> DataFrame:
        """
        弃用警告，迁移帮助
        """
        logger.warning(
            f"您的自定义IFreqaiModel依赖于已弃用的数据管道。请更新您的模型以使用新的数据管道。"
            " 这可以通过遵循迁移指南来实现 "
            f"{DOCS_LINK}/strategy_migration/#freqai-new-data-pipeline "
            "我们为您添加了一个基本管道，但这将在未来版本中移除。"
        )

        pred_df, _, _ = self.label_pipeline.inverse_transform(df)

        return pred_df