import collections
import importlib
import logging
import re
import shutil
import threading
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import psutil
import rapidjson
from joblib.externals import cloudpickle
from numpy.typing import NDArray
from pandas import DataFrame

from freqtrade.configuration import TimeRange
from freqtrade.constants import Config
from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType
from freqtrade.exceptions import OperationalException
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.strategy.interface import IStrategy


logger = logging.getLogger(__name__)

# 常量定义
FEATURE_PIPELINE = "feature_pipeline"
LABEL_PIPELINE = "label_pipeline"
TRAINDF = "trained_df"
METADATA = "metadata"

METADATA_NUMBER_MODE = rapidjson.NM_NATIVE | rapidjson.NM_NAN


class pair_info(TypedDict):
    """交易对信息的数据结构类型"""
    model_filename: str          # 模型文件名
    trained_timestamp: int       # 训练时间戳
    data_path: str               # 数据路径
    extras: dict                 # 额外信息


class FreqaiDataDrawer:
    """
    用于在内存中存储所有交易对模型/信息的类，以便更好地进行推理、再训练、保存和加载。
    该对象在实盘/模拟交易期间保持持久化。

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

    def __init__(self, full_path: Path, config: Config):
        """初始化数据抽屉"""
        self.config = config
        self.freqai_info = config.get("freqai", {})
        # 存储所有交易对元数据的字典，用于从磁盘加载
        self.pair_dict: dict[str, pair_info] = {}
        # 存储所有活跃推理模型的字典，以模型文件名为键
        self.model_dictionary: dict[str, Any] = {}
        # 存储所有需要保存在内存中的附加元数据
        self.meta_data_dictionary: dict[str, dict[str, Any]] = {}
        self.model_return_values: dict[str, DataFrame] = {}
        self.historic_data: dict[str, dict[str, DataFrame]] = {}  # 历史数据
        self.historic_predictions: dict[str, DataFrame] = {}      # 历史预测
        self.full_path = full_path
        # 路径定义
        self.historic_predictions_path = Path(self.full_path / "historic_predictions.pkl")
        self.historic_predictions_bkp_path = Path(
            self.full_path / "historic_predictions.backup.pkl"
        )
        self.pair_dictionary_path = Path(self.full_path / "pair_dictionary.json")
        self.global_metadata_path = Path(self.full_path / "global_metadata.json")
        self.metric_tracker_path = Path(self.full_path / "metric_tracker.json")
        
        # 从磁盘加载数据
        self.load_drawer_from_disk()
        self.load_historic_predictions_from_disk()
        self.metric_tracker: dict[str, dict[str, dict[str, list]]] = {}
        self.load_metric_tracker_from_disk()
        
        # 训练队列和锁机制
        self.training_queue: dict[str, int] = {}
        self.history_lock = threading.Lock()
        self.save_lock = threading.Lock()
        self.pair_dict_lock = threading.Lock()
        self.metric_tracker_lock = threading.Lock()
        self.old_DBSCAN_eps: dict[str, float] = {}
        
        # 空交易对字典模板
        self.empty_pair_dict: pair_info = {
            "model_filename": "",
            "trained_timestamp": 0,
            "data_path": "",
            "extras": {},
        }
        self.model_type = self.freqai_info.get("model_save_type", "joblib")

    def update_metric_tracker(self, metric: str, value: float, pair: str) -> None:
        """
        添加和更新自定义指标的通用工具。通常用于添加训练性能、训练时间、推理时间、CPU负载等。
        """
        with self.metric_tracker_lock:
            if pair not in self.metric_tracker:
                self.metric_tracker[pair] = {}
            if metric not in self.metric_tracker[pair]:
                self.metric_tracker[pair][metric] = {"timestamp": [], "value": []}

            timestamp = int(datetime.now(timezone.utc).timestamp())
            self.metric_tracker[pair][metric]["value"].append(value)
            self.metric_tracker[pair][metric]["timestamp"].append(timestamp)

    def collect_metrics(self, time_spent: float, pair: str):
        """收集并添加指标到指标跟踪器字典"""
        load1, load5, load15 = psutil.getloadavg()
        cpus = psutil.cpu_count()
        self.update_metric_tracker("train_time", time_spent, pair)
        self.update_metric_tracker("cpu_load1min", load1 / cpus, pair)
        self.update_metric_tracker("cpu_load5min", load5 / cpus, pair)
        self.update_metric_tracker("cpu_load15min", load15 / cpus, pair)

    def load_global_metadata_from_disk(self):
        """从当前模型文件夹中查找并加载之前保存的全局元数据"""
        exists = self.global_metadata_path.is_file()
        if exists:
            with self.global_metadata_path.open("r") as fp:
                metatada_dict = rapidjson.load(fp, number_mode=rapidjson.NM_NATIVE)
                return metatada_dict
        return {}

    def load_drawer_from_disk(self):
        """
        从当前模型文件夹中查找并加载之前保存的包含所有交易对模型元数据的数据抽屉。
        加载可能存在的任何现有指标跟踪器。
        """
        exists = self.pair_dictionary_path.is_file()
        if exists:
            with self.pair_dictionary_path.open("r") as fp:
                self.pair_dict = rapidjson.load(fp, number_mode=rapidjson.NM_NATIVE)
        else:
            logger.info("找不到现有的数据抽屉，从头开始")

    def load_metric_tracker_from_disk(self):
        """如果用户希望收集指标，则尝试加载现有的指标字典"""
        if self.freqai_info.get("write_metrics_to_disk", False):
            exists = self.metric_tracker_path.is_file()
            if exists:
                with self.metric_tracker_path.open("r") as fp:
                    self.metric_tracker = rapidjson.load(fp, number_mode=rapidjson.NM_NATIVE)
                logger.info("从磁盘加载现有的指标跟踪器。")
            else:
                logger.info("找不到现有的指标跟踪器，从头开始")

    def load_historic_predictions_from_disk(self):
        """
        查找并加载之前保存的历史预测。
        :return: bool - 是否找到并加载了抽屉
        """
        exists = self.historic_predictions_path.is_file()
        if exists:
            try:
                with self.historic_predictions_path.open("rb") as fp:
                    self.historic_predictions = cloudpickle.load(fp)
                logger.info(
                    f"在 {self.full_path} 找到现有的历史预测，但请注意 "
                    "如果机器人离线时间过长，统计数据可能不准确。"
                )
            except EOFError:
                logger.warning("历史预测文件已损坏。尝试加载备份文件。")
                with self.historic_predictions_bkp_path.open("rb") as fp:
                    self.historic_predictions = cloudpickle.load(fp)
                logger.warning("FreqAI成功加载了备份的历史预测文件。")

        else:
            logger.info("找不到现有的historic_predictions，从头开始")

        return exists

    def save_historic_predictions_to_disk(self):
        """将历史预测pickle保存到磁盘"""
        with self.historic_predictions_path.open("wb") as fp:
            cloudpickle.dump(self.historic_predictions, fp, protocol=cloudpickle.DEFAULT_PROTOCOL)

        # 创建备份
        shutil.copy(self.historic_predictions_path, self.historic_predictions_bkp_path)

    def save_metric_tracker_to_disk(self):
        """保存所有交易对收集的指标跟踪器"""
        with self.save_lock:
            with self.metric_tracker_path.open("w") as fp:
                rapidjson.dump(
                    self.metric_tracker,
                    fp,
                    default=self.np_encoder,
                    number_mode=rapidjson.NM_NATIVE,
                )

    def save_drawer_to_disk(self) -> None:
        """将包含所有交易对模型元数据的数据抽屉保存到当前模型文件夹"""
        with self.save_lock:
            with self.pair_dictionary_path.open("w") as fp:
                rapidjson.dump(
                    self.pair_dict, fp, default=self.np_encoder, number_mode=rapidjson.NM_NATIVE
                )

    def save_global_metadata_to_disk(self, metadata: dict[str, Any]):
        """将全局元数据json保存到磁盘"""
        with self.save_lock:
            with self.global_metadata_path.open("w") as fp:
                rapidjson.dump(
                    metadata, fp, default=self.np_encoder, number_mode=rapidjson.NM_NATIVE
                )

    def np_encoder(self, obj):
        """numpy类型编码器，用于JSON序列化"""
        if isinstance(obj, np.generic):
            return obj.item()

    def get_pair_dict_info(self, pair: str) -> tuple[str, int]:
        """
        从持久化存储中查找并加载现有模型元数据。如果未找到，
        创建一个新的并将当前交易对添加到其中，为第一次训练做准备
        :param pair: 要查找的交易对
        :return:
            model_filename: 用于从磁盘加载持久化对象的唯一文件名
            trained_timestamp: 硬币上次训练的时间
        """

        pair_dict = self.pair_dict.get(pair)

        if pair_dict:
            model_filename = pair_dict["model_filename"]
            trained_timestamp = pair_dict["trained_timestamp"]
        else:
            self.pair_dict[pair] = self.empty_pair_dict.copy()
            model_filename = ""
            trained_timestamp = 0

        return model_filename, trained_timestamp

    def set_pair_dict_info(self, metadata: dict) -> None:
        """设置交易对字典信息"""
        pair_in_dict = self.pair_dict.get(metadata["pair"])
        if pair_in_dict:
            return
        else:
            self.pair_dict[metadata["pair"]] = self.empty_pair_dict.copy()
            return

    def set_initial_return_values(
        self, pair: str, pred_df: DataFrame, dataframe: DataFrame
    ) -> None:
        """
        将初始返回值设置到历史预测数据帧。这避免了需要对历史K线重新预测，
        并且还存储了尽管有再训练的历史预测（因此存储的预测是真实预测，而不仅仅是对训练数据的推理）。

        我们还旨在保留历史预测中的日期，以便FreqUI在任何停机期间（在FreqAI重新加载之间）显示零。
        """

        new_pred = pred_df.copy()
        # 将new_pred值设置为nans（我们想向用户表明在停机期间没有任何历史记录。最新的预测将在后面的append_model_predictions中添加）

        new_pred["date_pred"] = dataframe["date"]
        # 将除date_pred之外的所有内容设置为nan
        columns_to_nan = new_pred.columns.difference(["date_pred", "date"])
        new_pred[columns_to_nan] = None

        hist_preds = self.historic_predictions[pair].copy()

        # 确保两个数据帧具有相同的日期格式，以便它们可以合并
        new_pred["date_pred"] = pd.to_datetime(new_pred["date_pred"])
        hist_preds["date_pred"] = pd.to_datetime(hist_preds["date_pred"])

        # 找到new_pred和历史预测之间最接近的公共日期，并在该日期截断new_pred数据帧
        common_dates = pd.merge(new_pred, hist_preds, on="date_pred", how="inner")
        if len(common_dates.index) > 0:
            new_pred = new_pred.iloc[len(common_dates) :]
        else:
            logger.warning(
                "在新预测和历史预测之间没有找到共同日期。您可能让FreqAI实例离线 "
                f"超过 {len(dataframe.index)} 根K线。"
            )

        # Pandas警告说它保留非NaN列的数据类型...
        # 是的，我们知道，我们已经想要这种行为。忽略。
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            # 重新索引new_pred列以匹配历史预测数据帧
            new_pred_reindexed = new_pred.reindex(columns=hist_preds.columns)
            df_concat = pd.concat([hist_preds, new_pred_reindexed], ignore_index=True)

        # 任何缺失值都将被归零，以便用户可以在FreqUI中看到确切的停机时间
        df_concat = df_concat.fillna(0)
        self.historic_predictions[pair] = df_concat
        self.model_return_values[pair] = df_concat.tail(len(dataframe.index)).reset_index(drop=True)

    def append_model_predictions(
        self,
        pair: str,
        predictions: DataFrame,
        do_preds: NDArray[np.int_],
        dk: FreqaiDataKitchen,
        strat_df: DataFrame,
    ) -> None:
        """
        将模型预测附加到历史预测数据帧，然后将策略返回数据帧设置为历史预测的尾部。
        尾部的长度等于最初从策略进入FreqAI的数据帧的长度。这样做可以使FreqUI始终显示正确的历史预测。
        """

        len_df = len(strat_df)
        index = self.historic_predictions[pair].index[-1:]
        columns = self.historic_predictions[pair].columns

        zeros_df = pd.DataFrame(np.zeros((1, len(columns))), index=index, columns=columns)
        self.historic_predictions[pair] = pd.concat(
            [self.historic_predictions[pair], zeros_df], ignore_index=True, axis=0
        )
        df = self.historic_predictions[pair]

        # 模型输出和相关统计
        for label in predictions.columns:
            label_loc = df.columns.get_loc(label)
            pred_label_loc = predictions.columns.get_loc(label)
            df.iloc[-1, label_loc] = predictions.iloc[-1, pred_label_loc]
            if df[label].dtype == object:
                continue
            label_mean_loc = df.columns.get_loc(f"{label}_mean")
            label_std_loc = df.columns.get_loc(f"{label}_std")
            df.iloc[-1, label_mean_loc] = dk.data["labels_mean"][label]
            df.iloc[-1, label_std_loc] = dk.data["labels_std"][label]

        # 异常值指示器
        do_predict_loc = df.columns.get_loc("do_predict")
        df.iloc[-1, do_predict_loc] = do_preds[-1]
        if self.freqai_info["feature_parameters"].get("DI_threshold", 0) > 0:
            DI_values_loc = df.columns.get_loc("DI_values")
            df.iloc[-1, DI_values_loc] = dk.DI_values[-1]

        # 用户在自定义预测模型中添加的额外值
        if dk.data["extra_returns_per_train"]:
            rets = dk.data["extra_returns_per_train"]
            for return_str in rets:
                return_loc = df.columns.get_loc(return_str)
                df.iloc[-1, return_loc] = rets[return_str]

        # 添加价格信息
        high_price_loc = df.columns.get_loc("high_price")
        high_loc = strat_df.columns.get_loc("high")
        df.iloc[-1, high_price_loc] = strat_df.iloc[-1, high_loc]
        low_price_loc = df.columns.get_loc("low_price")
        low_loc = strat_df.columns.get_loc("low")
        df.iloc[-1, low_price_loc] = strat_df.iloc[-1, low_loc]
        close_price_loc = df.columns.get_loc("close_price")
        close_loc = strat_df.columns.get_loc("close")
        df.iloc[-1, close_price_loc] = strat_df.iloc[-1, close_loc]
        date_pred_loc = df.columns.get_loc("date_pred")
        date_loc = strat_df.columns.get_loc("date")
        df.iloc[-1, date_pred_loc] = strat_df.iloc[-1, date_loc]

        self.model_return_values[pair] = df.tail(len_df).reset_index(drop=True)

    def attach_return_values_to_return_dataframe(
        self, pair: str, dataframe: DataFrame
    ) -> DataFrame:
        """
        将返回值附加到策略数据帧
        :param dataframe: 策略数据帧
        :return: 附加了返回值的策略数据帧
        """
        df = self.model_return_values[pair]
        to_keep = [col for col in dataframe.columns if not col.startswith("&")]
        dataframe = pd.concat([dataframe[to_keep], df], axis=1)
        return dataframe

    def return_null_values_to_strategy(self, dataframe: DataFrame, dk: FreqaiDataKitchen) -> None:
        """构建填充0的数据帧返回给策略"""

        dk.find_features(dataframe)
        dk.find_labels(dataframe)

        full_labels = dk.label_list + dk.unique_class_list

        for label in full_labels:
            dataframe[label] = 0
            dataframe[f"{label}_mean"] = 0
            dataframe[f"{label}_std"] = 0

        dataframe["do_predict"] = 0

        if self.freqai_info["feature_parameters"].get("DI_threshold", 0) > 0:
            dataframe["DI_values"] = 0

        if dk.data["extra_returns_per_train"]:
            rets = dk.data["extra_returns_per_train"]
            for return_str in rets:
                dataframe[return_str] = 0

        dk.return_dataframe = dataframe

    def purge_old_models(self) -> None:
        """清理旧模型文件"""
        num_keep = self.freqai_info["purge_old_models"]
        if not num_keep:
            return
        elif isinstance(num_keep, bool):
            num_keep = 2

        model_folders = [x for x in self.full_path.iterdir() if x.is_dir()]

        pattern = re.compile(r"sub-train-(\w+)_(\d{10})")

        delete_dict: dict[str, Any] = {}

        for directory in model_folders:
            result = pattern.match(str(directory.name))
            if result is None:
                continue
            coin = result.group(1)
            timestamp = result.group(2)

            if coin not in delete_dict:
                delete_dict[coin] = {}
                delete_dict[coin]["num_folders"] = 1
                delete_dict[coin]["timestamps"] = {int(timestamp): directory}
            else:
                delete_dict[coin]["num_folders"] += 1
                delete_dict[coin]["timestamps"][int(timestamp)] = directory

        for coin in delete_dict:
            if delete_dict[coin]["num_folders"] > num_keep:
                sorted_dict = collections.OrderedDict(
                    sorted(delete_dict[coin]["timestamps"].items())
                )
                num_delete = len(sorted_dict) - num_keep
                deleted = 0
                for k, v in sorted_dict.items():
                    if deleted >= num_delete:
                        break
                    logger.info(f"Freqai清理旧模型文件 {v}")
                    shutil.rmtree(v)
                    deleted += 1

    def save_metadata(self, dk: FreqaiDataKitchen) -> None:
        """
        如果用户倾向于不保存模型数据，则仅保存用于回测研究的元数据。
        这为生成大量研究的用户节省了大量空间。
        仅当`save_backtest_models`为false时激活（非默认）
        """
        if not dk.data_path.is_dir():
            dk.data_path.mkdir(parents=True, exist_ok=True)

        save_path = Path(dk.data_path)

        dk.data["data_path"] = str(dk.data_path)
        dk.data["model_filename"] = str(dk.model_filename)
        dk.data["training_features_list"] = list(dk.data_dictionary["train_features"].columns)
        dk.data["label_list"] = dk.label_list

        with (save_path / f"{dk.model_filename}_{METADATA}.json").open("w") as fp:
            rapidjson.dump(dk.data, fp, default=self.np_encoder, number_mode=METADATA_NUMBER_MODE)

        return

    def save_data(self, model: Any, coin: str, dk: FreqaiDataKitchen) -> None:
        """
        保存与单个子训练时间范围内的模型相关的所有数据
        :param model: 用户训练的模型，可重用于推理以生成预测
        """

        if not dk.data_path.is_dir():
            dk.data_path.mkdir(parents=True, exist_ok=True)

        save_path = Path(dk.data_path)

        # 保存训练好的模型
        if self.model_type == "joblib":
            with (save_path / f"{dk.model_filename}_model.joblib").open("wb") as fp:
                cloudpickle.dump(model, fp)
        elif self.model_type == "keras":
            model.save(save_path / f"{dk.model_filename}_model.h5")
        elif self.model_type in ["stable_baselines3", "sb3_contrib", "pytorch"]:
            model.save(save_path / f"{dk.model_filename}_model.zip")

        dk.data["data_path"] = str(dk.data_path)
        dk.data["model_filename"] = str(dk.model_filename)
        dk.data["training_features_list"] = dk.training_features_list
        dk.data["label_list"] = dk.label_list
        # 存储元数据
        with (save_path / f"{dk.model_filename}_{METADATA}.json").open("w") as fp:
            rapidjson.dump(dk.data, fp, default=self.np_encoder, number_mode=METADATA_NUMBER_MODE)

        # 将管道保存到pickle文件
        with (save_path / f"{dk.model_filename}_{FEATURE_PIPELINE}.pkl").open("wb") as fp:
            cloudpickle.dump(dk.feature_pipeline, fp)

        with (save_path / f"{dk.model_filename}_{LABEL_PIPELINE}.pkl").open("wb") as fp:
            cloudpickle.dump(dk.label_pipeline, fp)

        # 将训练数据保存到文件，以便需要时进行后处理
        dk.data_dictionary["train_features"].to_pickle(
            save_path / f"{dk.model_filename}_{TRAINDF}.pkl"
        )

        dk.data_dictionary["train_dates"].to_pickle(
            save_path / f"{dk.model_filename}_trained_dates_df.pkl"
        )

        self.model_dictionary[coin] = model
        self.pair_dict[coin]["model_filename"] = dk.model_filename
        self.pair_dict[coin]["data_path"] = str(dk.data_path)

        if coin not in self.meta_data_dictionary:
            self.meta_data_dictionary[coin] = {}
        self.meta_data_dictionary[coin][METADATA] = dk.data
        self.meta_data_dictionary[coin][FEATURE_PIPELINE] = dk.feature_pipeline
        self.meta_data_dictionary[coin][LABEL_PIPELINE] = dk.label_pipeline
        self.save_drawer_to_disk()

        return

    def load_metadata(self, dk: FreqaiDataKitchen) -> None:
        """
        将仅元数据加载到数据厨房中，以提高预保存回测期间的性能（预测文件加载）。
        """
        with (dk.data_path / f"{dk.model_filename}_{METADATA}.json").open("r") as fp:
            dk.data = rapidjson.load(fp, number_mode=METADATA_NUMBER_MODE)
            dk.training_features_list = dk.data["training_features_list"]
            dk.label_list = dk.data["label_list"]

    def load_data(self, coin: str, dk: FreqaiDataKitchen) -> Any:
        """
        加载在子训练时间范围内进行预测所需的所有数据
        :returns:
        :model: 可用于推理新预测的用户训练模型
        """

        if not self.pair_dict[coin]["model_filename"]:
            return None

        if dk.live:
            dk.model_filename = self.pair_dict[coin]["model_filename"]
            dk.data_path = Path(self.pair_dict[coin]["data_path"])

        if coin in self.meta_data_dictionary:
            dk.data = self.meta_data_dictionary[coin][METADATA]
            dk.feature_pipeline = self.meta_data_dictionary[coin][FEATURE_PIPELINE]
            dk.label_pipeline = self.meta_data_dictionary[coin][LABEL_PIPELINE]
        else:
            with (dk.data_path / f"{dk.model_filename}_{METADATA}.json").open("r") as fp:
                dk.data = rapidjson.load(fp, number_mode=METADATA_NUMBER_MODE)

            with (dk.data_path / f"{dk.model_filename}_{FEATURE_PIPELINE}.pkl").open("rb") as fp:
                dk.feature_pipeline = cloudpickle.load(fp)
            with (dk.data_path / f"{dk.model_filename}_{LABEL_PIPELINE}.pkl").open("rb") as fp:
                dk.label_pipeline = cloudpickle.load(fp)

        dk.training_features_list = dk.data["training_features_list"]
        dk.label_list = dk.data["label_list"]

        # 尝试访问内存中的模型，而不是从磁盘加载对象以节省时间
        if dk.live and coin in self.model_dictionary:
            model = self.model_dictionary[coin]
        elif self.model_type == "joblib":
            with (dk.data_path / f"{dk.model_filename}_model.joblib").open("rb") as fp:
                model = cloudpickle.load(fp)
        elif "stable_baselines" in self.model_type or "sb3_contrib" == self.model_type:
            mod = importlib.import_module(
                self.model_type, self.freqai_info["rl_config"]["model_type"]
            )
            MODELCLASS = getattr(mod, self.freqai_info["rl_config"]["model_type"])
            model = MODELCLASS.load(dk.data_path / f"{dk.model_filename}_model")
        elif self.model_type == "pytorch":
            import torch

            zipfile = torch.load(dk.data_path / f"{dk.model_filename}_model.zip")
            model = zipfile["pytrainer"]
            model = model.load_from_checkpoint(zipfile)

        if not model:
            raise OperationalException(
                f"无法加载模型，请确保模型存在于 {dk.data_path} "
            )

        # 如果从磁盘加载，将其加载到内存中
        if coin not in self.model_dictionary:
            self.model_dictionary[coin] = model

        return model

    def update_historic_data(self, strategy: IStrategy, dk: FreqaiDataKitchen) -> None:
        """
        将新K线附加到我们存储的历史数据（内存中），这样我们就不需要从磁盘加载K线历史，
        也不需要多次 ping 交易所获取相同的K线。
        :param dataframe: 策略提供的数据帧
        """
        feat_params = self.freqai_info["feature_parameters"]
        with self.history_lock:
            history_data = self.historic_data

            for pair in dk.all_pairs:
                for tf in feat_params.get("include_timeframes"):
                    hist_df = history_data[pair][tf]
                    # 检查最新的K线是否已附加
                    df_dp = strategy.dp.get_pair_dataframe(pair, tf)
                    if len(df_dp.index) == 0:
                        continue
                    if str(hist_df.iloc[-1]["date"]) == str(df_dp.iloc[-1:]["date"].iloc[-1]):
                        continue

                    try:
                        index = df_dp.loc[df_dp["date"] == hist_df.iloc[-1]["date"]].index[0] + 1
                    except IndexError:
                        if hist_df.iloc[-1]["date"] < df_dp["date"].iloc[0]:
                            raise OperationalException(
                                "内存中的历史数据早于 "
                                f"{pair} 在时间框架 {tf} 上的最旧DataProvider K线"
                            )
                        else:
                            index = -1
                            logger.warning(
                                f"{pair}的历史数据和数据提供者中没有共同日期。 "
                                f"将最新的数据提供者K线附加到历史数据 "
                                "但请注意，历史数据中可能存在差距。 \n"
                                f"历史数据结束于 {hist_df.iloc[-1]['date']} "
                                f"而数据提供者开始于 {df_dp['date'].iloc[0]} 并且"
                                f"结束于 {df_dp['date'].iloc[0]}。"
                            )

                    history_data[pair][tf] = pd.concat(
                        [
                            hist_df,
                            df_dp.iloc[index:],
                        ],
                        ignore_index=True,
                        axis=0,
                    )

            self.current_candle = history_data[dk.pair][self.config["timeframe"]].iloc[-1]["date"]

    def load_all_pair_histories(self, timerange: TimeRange, dk: FreqaiDataKitchen) -> None:
        """
        为所有白名单和相关交易对列表加载交易对历史。
        仅在机器人启动时调用一次。
        :param timerange: 根据用户定义的train_period_days，填充所有指标进行训练所需的完整时间范围
        """
        history_data = self.historic_data

        for pair in dk.all_pairs:
            if pair not in history_data:
                history_data[pair] = {}
            for tf in self.freqai_info["feature_parameters"].get("include_timeframes"):
                history_data[pair][tf] = load_pair_history(
                    datadir=self.config["datadir"],
                    timeframe=tf,
                    pair=pair,
                    timerange=timerange,
                    data_format=self.config.get("dataformat_ohlcv", "feather"),
                    candle_type=self.config.get("candle_type_def", CandleType.SPOT),
                )

    def get_base_and_corr_dataframes(
        self, timerange: TimeRange, pair: str, dk: FreqaiDataKitchen
    ) -> tuple[dict[Any, Any], dict[Any, Any]]:
        """
        在内存中搜索我们的历史数据，并返回与当前交易对相关的数据帧。
        :param timerange: 根据用户定义的train_period_days，填充所有指标进行训练所需的完整时间范围
        :param metadata: 策略提供的交易对元数据
        """
        with self.history_lock:
            corr_dataframes: dict[Any, Any] = {}
            base_dataframes: dict[Any, Any] = {}
            historic_data = self.historic_data
            pairs = self.freqai_info["feature_parameters"].get("include_corr_pairlist", [])

            for tf in self.freqai_info["feature_parameters"].get("include_timeframes"):
                base_dataframes[tf] = dk.slice_dataframe(
                    timerange, historic_data[pair][tf]
                ).reset_index(drop=True)
                if pairs:
                    for p in pairs:
                        if pair in p:
                            continue  # 不要重复白名单中的任何内容
                        if p not in corr_dataframes:
                            corr_dataframes[p] = {}
                        corr_dataframes[p][tf] = dk.slice_dataframe(
                            timerange, historic_data[p][tf]
                        ).reset_index(drop=True)

        return corr_dataframes, base_dataframes

    def get_timerange_from_live_historic_predictions(self) -> TimeRange:
        """
        基于历史预测文件返回时间范围信息
        :return: 从保存的实时数据计算的时间范围
        """
        if not self.historic_predictions_path.is_file():
            raise OperationalException(
                "未找到历史预测。运行带有freqai-backtest-live-models选项的回测需要历史预测数据 "
            )

        self.load_historic_predictions_from_disk()

        all_pairs_end_dates = []
        for pair in self.historic_predictions:
            pair_historic_data = self.historic_predictions[pair]
            all_pairs_end_dates.append(pair_historic_data.date_pred.max())

        global_metadata = self.load_global_metadata_from_disk()
        start_date = datetime.fromtimestamp(int(global_metadata["start_dry_live_date"]))
        end_date = max(all_pairs_end_dates)
        # 向字符串时间范围添加1天，确保BT模块将加载所有数据帧数据
        end_date = end_date + timedelta(days=1)
        backtesting_timerange = TimeRange(
            "date", "date", int(start_date.timestamp()), int(end_date.timestamp())
        )
        return backtesting_timerange