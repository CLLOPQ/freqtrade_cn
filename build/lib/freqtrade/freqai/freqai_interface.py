import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import datasieve.transforms as ds
import numpy as np
import pandas as pd
import psutil
from datasieve.pipeline import Pipeline
from datasieve.transforms import SKLearnWrapper
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

from freqtrade.configuration import TimeRange
from freqtrade.constants import DOCS_LINK, Config
from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_seconds
from freqtrade.freqai.data_drawer import FreqaiDataDrawer
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.utils import get_tb_logger, plot_feature_importance, record_params
from freqtrade.strategy.interface import IStrategy


pd.options.mode.chained_assignment = None
logger = logging.getLogger(__name__)


class IFreqaiModel(ABC):
    """
    包含策略中所有训练和预测工具的类。
    Base*PredictionModels 继承自此类。

    贡献记录：
    FreqAI 是由一群人开发的，他们都为该项目贡献了特定的技能。

    构思和软件开发：
    Robert Caulk @robcaulk

    理论研讨：
    Elin Törnquist @th0rntwig

    代码审查、软件架构研讨：
    @xmatthias

    Beta 测试和错误报告：
    @bloodhunter4rc, Salah Lamkadem @ikonx, @ken11o2, @longyu, @paranoidandy, @smidelis, @smarm
    Juha Nykänen @suikula, Wagner Costa @wagnercosta, Johan Vlugt @Jooopieeert
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.assert_config(self.config)
        self.freqai_info: dict[str, Any] = config["freqai"]
        self.data_split_parameters: dict[str, Any] = config.get("freqai", {}).get(
            "data_split_parameters", {}
        )
        self.model_training_parameters: dict[str, Any] = config.get("freqai", {}).get(
            "model_training_parameters", {}
        )
        self.identifier: str = self.freqai_info.get("identifier", "no_id_provided")
        self.retrain = False
        self.first = True
        self.set_full_path()
        self.save_backtest_models: bool = self.freqai_info.get("save_backtest_models", True)
        if self.save_backtest_models:
            logger.info("回测模块已配置为保存所有模型。")

        self.dd = FreqaiDataDrawer(Path(self.full_path), self.config)
        # 将当前蜡烛设置为任意历史日期
        self.current_candle: datetime = datetime.fromtimestamp(637887600, tz=timezone.utc)
        self.dd.current_candle = self.current_candle
        self.scanning = False
        self.ft_params = self.freqai_info["feature_parameters"]
        self.corr_pairlist: list[str] = self.ft_params.get("include_corr_pairlist", [])
        self.keras: bool = self.freqai_info.get("keras", False)
        if self.keras and self.ft_params.get("DI_threshold", 0):
            self.ft_params["DI_threshold"] = 0
            logger.warning("Keras 模型尚未配置 DI 阈值。正在禁用。")

        self.CONV_WIDTH = self.freqai_info.get("conv_width", 1)
        self.class_names: list[str] = []  # 在分类子类中使用
        self.pair_it = 0
        self.pair_it_train = 0
        self.total_pairs = len(self.config.get("exchange", {}).get("pair_whitelist"))
        self.train_queue = self._set_train_queue()
        self.inference_time: float = 0
        self.train_time: float = 0
        self.begin_time: float = 0
        self.begin_time_train: float = 0
        self.base_tf_seconds = timeframe_to_seconds(self.config["timeframe"])
        self.continual_learning = self.freqai_info.get("continual_learning", False)
        self.plot_features = self.ft_params.get("plot_feature_importances", 0)
        self.corr_dataframes: dict[str, DataFrame] = {}
        # get_corr_dataframes 控制 corr_dataframes 的缓存以提高性能。小心使用这个布尔值。
        self.get_corr_dataframes: bool = True
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()
        self.metadata: dict[str, Any] = self.dd.load_global_metadata_from_disk()
        self.data_provider: DataProvider | None = None
        self.max_system_threads = max(int(psutil.cpu_count() * 2 - 2), 1)
        self.can_short = True  # 在 start() 中被 strategy.can_short 覆盖
        self.model: Any = None
        if self.ft_params.get("principal_component_analysis", False) and self.continual_learning:
            self.ft_params.update({"principal_component_analysis": False})
            logger.warning("用户尝试将 PCA 与持续学习一起使用。正在禁用 PCA。")
        self.activate_tensorboard: bool = self.freqai_info.get("activate_tensorboard", True)

        record_params(config, self.full_path)

    def __getstate__(self):
        """
        返回一个空状态以便在超参数优化中被序列化
        """
        return {}

    def assert_config(self, config: Config) -> None:
        if not config.get("freqai", {}):
            raise OperationalException("在配置文件中未找到 freqai 参数。")

    def start(self, dataframe: DataFrame, metadata: dict, strategy: IStrategy) -> DataFrame:
        """
        从特定交易对进入 FreqaiModel 的入口点，在进行预测之前，如有必要，它将训练一个新模型。

        :param dataframe: 来自策略的完整数据框 - 它包含整个回测时间范围 + 训练模型所需的额外历史数据。
        :param metadata: 来自策略的交易对元数据。
        :param strategy: 要训练的策略
        """
        self.live = strategy.dp.runmode in (RunMode.DRY_RUN, RunMode.LIVE)
        self.dd.set_pair_dict_info(metadata)
        self.data_provider = strategy.dp
        self.can_short = strategy.can_short

        if self.live:
            self.inference_timer("start")
            self.dk = FreqaiDataKitchen(self.config, self.live, metadata["pair"])
            dk = self.start_live(dataframe, metadata, strategy, self.dk)
            dataframe = dk.remove_features_from_df(dk.return_dataframe)

        # 对于回测，每个交易对进入并在由 "train_period_days"（训练窗口）和 "live_retrain_hours"（回测窗口，即紧随训练窗口之后的窗口）定义的滑动窗口上进行训练。
        # FreqAI 滑动窗口并顺序构建回测结果，然后将整个回测期间的拼接结果返回给策略。
        else:
            self.dk = FreqaiDataKitchen(self.config, self.live, metadata["pair"])
            if not self.config.get("freqai_backtest_live_models", False):
                logger.info(f"训练 {len(self.dk.training_timeranges)} 个时间范围")
                dk = self.start_backtesting(dataframe, metadata, self.dk, strategy)
                dataframe = dk.remove_features_from_df(dk.return_dataframe)
            else:
                logger.info("使用历史预测（实时模型）进行回测")
                dk = self.start_backtesting_from_historic_predictions(dataframe, metadata, self.dk)
                dataframe = dk.return_dataframe

        self.clean_up()
        if self.live:
            self.inference_timer("stop", metadata["pair"])

        return dataframe

    def clean_up(self):
        """
        硬币之间的对象应该已经由 GC 处理，但在此明确显示以帮助说明这些对象的非持久性。
        """
        self.model = None
        self.dk = None

    def _on_stop(self):
        """
        子类的回调函数，用于在发送 SIGINT 时包含关闭资源的逻辑。
        """
        self.dd.save_historic_predictions_to_disk()
        return

    def shutdown(self):
        """
        在关闭时清理线程，设置停止事件。连接线程以等待当前训练迭代。
        """
        logger.info("停止 FreqAI")
        self._stop_event.set()

        self.data_provider = None
        self._on_stop()

        if self.freqai_info.get("wait_for_training_iteration_on_reload", True):
            logger.info("等待训练迭代")
            for _thread in self._threads:
                _thread.join()
        else:
            logger.warning(
                "中断当前训练迭代，因为您将 wait_for_training_iteration_on_reload 设置为 False。"
            )

    def start_scanning(self, *args, **kwargs) -> None:
        """
        在单独的线程中启动 `self._start_scanning`
        """
        _thread = threading.Thread(target=self._start_scanning, args=args, kwargs=kwargs)
        self._threads.append(_thread)
        _thread.start()

    def _start_scanning(self, strategy: IStrategy) -> None:
        """
        旨在不断扫描交易对以在单独的线程（蜡烛内）上重新训练，以提高模型时效性的函数。此函数与数据准备/收集/存储无关，它只是根据 self.dd 中可用的数据进行训练。
        :param strategy: IStrategy = 用户定义的策略类
        """
        while not self._stop_event.is_set():
            time.sleep(1)
            pair = self.train_queue[0]

            # 确保交易对在 dp 中可用
            if pair not in strategy.dp.current_whitelist():
                self.train_queue.popleft()
                logger.warning(f"{pair} 不在当前白名单中，从训练队列中移除。")
                continue

            (_, trained_timestamp) = self.dd.get_pair_dict_info(pair)

            dk = FreqaiDataKitchen(self.config, self.live, pair)
            (
                retrain,
                new_trained_timerange,
                data_load_timerange,
            ) = dk.check_if_new_training_required(trained_timestamp)

            if retrain:
                self.train_timer("start")
                dk.set_paths(pair, new_trained_timerange.stopts)
                try:
                    self.extract_data_and_train_model(
                        new_trained_timerange, pair, strategy, dk, data_load_timerange
                    )
                except Exception as msg:
                    logger.exception(
                        f"训练 {pair} 时引发异常 {msg.__class__.__name__}。"
                        f"消息: {msg}，正在跳过。"
                    )

                self.train_timer("stop", pair)

                # 只有在第一个训练完成后才旋转队列。
                self.train_queue.rotate(-1)

                self.dd.save_historic_predictions_to_disk()
                if self.freqai_info.get("write_metrics_to_disk", False):
                    self.dd.save_metric_tracker_to_disk()

    def start_backtesting(
        self, dataframe: DataFrame, metadata: dict, dk: FreqaiDataKitchen, strategy: IStrategy
    ) -> FreqaiDataKitchen:
        """
        回测的主要执行流程。对于回测，每个交易对进入并在由 "train_period_days"（训练窗口）和 "backtest_period_days"（回测窗口，即紧随训练窗口之后的窗口）定义的滑动窗口上进行训练。FreqAI 滑动窗口并顺序构建回测结果，然后将整个回测期间的拼接结果返回给策略。
        :param dataframe: DataFrame = 策略传递的数据框
        :param metadata: Dict = 交易对元数据
        :param dk: FreqaiDataKitchen = 仅与当前交易对相关联的数据管理/分析工具
        :param strategy: 要训练的策略
        :return:
            FreqaiDataKitchen = 仅与当前交易对相关联的数据管理/分析工具
        """

        self.pair_it += 1
        train_it = 0
        pair = metadata["pair"]
        populate_indicators = True
        check_features = True
        # 强制滑动窗口训练/回测范式的循环
        # tr_train 是训练时间范围，例如 1 个历史月
        # tr_backtest 是回测时间范围，例如紧随 tr_train 之后的一周。这两个窗口都在整个回测中滑动
        for tr_train, tr_backtest in zip(
            dk.training_timeranges, dk.backtesting_timeranges, strict=False
        ):
            (_, _) = self.dd.get_pair_dict_info(pair)
            train_it += 1
            total_trains = len(dk.backtesting_timeranges)
            self.training_timerange = tr_train
            len_backtest_df = len(
                dataframe.loc[
                    (dataframe["date"] >= tr_backtest.startdt)
                    & (dataframe["date"] < tr_backtest.stopdt),
                    :,
                ]
            )

            if not self.ensure_data_exists(len_backtest_df, tr_backtest, pair):
                continue

            self.log_backtesting_progress(tr_train, pair, train_it, total_trains)

            timestamp_model_id = int(tr_train.stopts)
            if dk.backtest_live_models:
                timestamp_model_id = int(tr_backtest.startts)

            dk.set_paths(pair, timestamp_model_id)

            dk.set_new_model_names(pair, timestamp_model_id)

            if dk.check_if_backtest_prediction_is_valid(len_backtest_df):
                if check_features:
                    self.dd.load_metadata(dk)
                    df_fts = self.dk.use_strategy_to_populate_indicators(
                        strategy, prediction_dataframe=dataframe.tail(1), pair=pair
                    )
                    df_fts = dk.remove_special_chars_from_feature_names(df_fts)
                    dk.find_features(df_fts)
                    self.check_if_feature_list_matches_strategy(dk)
                    check_features = False
                append_df = dk.get_backtesting_prediction()
                dk.append_predictions(append_df)
            else:
                if populate_indicators:
                    dataframe = self.dk.use_strategy_to_populate_indicators(
                        strategy, prediction_dataframe=dataframe, pair=pair
                    )
                    populate_indicators = False

                dataframe_base_train = dataframe.loc[dataframe["date"] < tr_train.stopdt, :]
                dataframe_base_train = strategy.set_freqai_targets(
                    dataframe_base_train, metadata=metadata
                )
                dataframe_base_backtest = dataframe.loc[dataframe["date"] < tr_backtest.stopdt, :]
                dataframe_base_backtest = strategy.set_freqai_targets(
                    dataframe_base_backtest, metadata=metadata
                )

                tr_train = dk.buffer_timerange(tr_train)

                dataframe_train = dk.slice_dataframe(tr_train, dataframe_base_train)
                dataframe_backtest = dk.slice_dataframe(tr_backtest, dataframe_base_backtest)

                dataframe_train = dk.remove_special_chars_from_feature_names(dataframe_train)
                dataframe_backtest = dk.remove_special_chars_from_feature_names(dataframe_backtest)
                dk.get_unique_classes_from_labels(dataframe_train)

                if not self.model_exists(dk):
                    dk.find_features(dataframe_train)
                    dk.find_labels(dataframe_train)

                    try:
                        self.tb_logger = get_tb_logger(
                            self.dd.model_type, dk.data_path, self.activate_tensorboard
                        )
                        self.model = self.train(dataframe_train, pair, dk)
                        self.tb_logger.close()
                    except Exception as msg:
                        logger.warning(
                            f"训练 {pair} 时引发异常 {msg.__class__.__name__}。"
                            f"消息: {msg}，正在跳过。",
                            exc_info=True,
                        )
                        self.model = None

                    self.dd.pair_dict[pair]["trained_timestamp"] = int(tr_train.stopts)
                    if self.plot_features and self.model is not None:
                        plot_feature_importance(self.model, pair, dk, self.plot_features)
                    if self.save_backtest_models and self.model is not None:
                        logger.info("将回测模型保存到磁盘。")
                        self.dd.save_data(self.model, pair, dk)
                    else:
                        logger.info("将元数据保存到磁盘。")
                        self.dd.save_metadata(dk)
                else:
                    self.model = self.dd.load_data(pair, dk)

                pred_df, do_preds = self.predict(dataframe_backtest, dk)
                append_df = dk.get_predictions_to_append(pred_df, do_preds, dataframe_backtest)
                dk.append_predictions(append_df)
                dk.save_backtesting_prediction(append_df)

        self.backtesting_fit_live_predictions(dk)
        dk.fill_predictions(dataframe)

        return dk

    def start_live(
        self, dataframe: DataFrame, metadata: dict, strategy: IStrategy, dk: FreqaiDataKitchen
    ) -> FreqaiDataKitchen:
        """
        模拟/实盘的主要执行流程。此函数将检查是否应执行重新训练，如果是，则重新训练并重置模型。
        :param dataframe: DataFrame = 策略传递的数据框
        :param metadata: Dict = 交易对元数据
        :param strategy: IStrategy = 当前使用的策略
        dk: FreqaiDataKitchen = 仅与当前交易对相关联的数据管理/分析工具
        :returns:
        dk: FreqaiDataKitchen = 仅与当前交易对相关联的数据管理/分析工具
        """

        if not strategy.process_only_new_candles:
            raise OperationalException(
                "您正尝试使用 FreqAI 策略，同时设置 process_only_new_candles = False。这不受 FreqAI 支持，因此将中止。"
            )

        # 获取与当前交易对相关联的模型元数据
        (_, trained_timestamp) = self.dd.get_pair_dict_info(metadata["pair"])

        # 每轮仅附加一次历史数据
        if self.dd.historic_data:
            self.dd.update_historic_data(strategy, dk)
            logger.debug(f"更新交易对 {metadata['pair']} 的历史数据")
            self.track_current_candle()

        (_, new_trained_timerange, data_load_timerange) = dk.check_if_new_training_required(
            trained_timestamp
        )
        dk.set_paths(metadata["pair"], new_trained_timerange.stopts)

        # 如果尚未将蜡烛历史加载到内存中，则加载。
        if not self.dd.historic_data:
            self.dd.load_all_pair_histories(data_load_timerange, dk)

        if not self.scanning:
            self.scanning = True
            self.start_scanning(strategy)

        # 将模型和相关数据加载到数据工具中
        self.model = self.dd.load_data(metadata["pair"], dk)

        dataframe = dk.use_strategy_to_populate_indicators(
            strategy,
            prediction_dataframe=dataframe,
            pair=metadata["pair"],
            do_corr_pairs=self.get_corr_dataframes,
        )

        if not self.model:
            logger.warning(
                f"{metadata['pair']} 没有可用模型，向策略返回空值。"
            )
            self.dd.return_null_values_to_strategy(dataframe, dk)
            return dk

        if self.corr_pairlist:
            dataframe = self.cache_corr_pairlist_dfs(dataframe, dk)

        dk.find_labels(dataframe)

        self.build_strategy_return_arrays(dataframe, dk, metadata["pair"], trained_timestamp)

        return dk

    def build_strategy_return_arrays(
        self, dataframe: DataFrame, dk: FreqaiDataKitchen, pair: str, trained_timestamp: int
    ) -> None:
        # 在内存中保存历史预测，以便我们向策略返回正确的数组

        if pair not in self.dd.model_return_values:
            # 第一次预测是对来自策略的整个历史蜡烛集进行的。这允许 FreqUI 显示完整的返回值。
            pred_df, do_preds = self.predict(dataframe, dk)
            if pair not in self.dd.historic_predictions:
                self.set_initial_historic_predictions(pred_df, dk, pair, dataframe)
            self.dd.set_initial_return_values(pair, pred_df, dataframe)

            dk.return_dataframe = self.dd.attach_return_values_to_return_dataframe(pair, dataframe)
            return
        elif self.dk.check_if_model_expired(trained_timestamp):
            pred_df = DataFrame(np.zeros((2, len(dk.label_list))), columns=dk.label_list)
            do_preds = np.ones(2, dtype=np.int_) * 2
            dk.DI_values = np.zeros(2)
            logger.warning(
                f"{pair} 的模型已过期，向策略返回空值。策略构建应考虑到这种情况，即 prediction == 0 且 do_predict == 2"
            )
        else:
            # 为了性能和历史准确性，其余预测仅在最近的蜡烛上进行。
            pred_df, do_preds = self.predict(dataframe.iloc[-self.CONV_WIDTH :], dk, first=False)

        if self.freqai_info.get("fit_live_predictions_candles", 0) and self.live:
            self.fit_live_predictions(dk, pair)
        self.dd.append_model_predictions(pair, pred_df, do_preds, dk, dataframe)
        dk.return_dataframe = self.dd.attach_return_values_to_return_dataframe(pair, dataframe)

        return

    def check_if_feature_list_matches_strategy(self, dk: FreqaiDataKitchen) -> None:
        """
        如果用户重用指向持有现有模型的文件夹的 `identifier`，确保用户传递正确的特征集。
        :param dataframe: DataFrame = 策略提供的数据框
        :param dk: FreqaiDataKitchen = 当前币种/机器人循环的非持久性数据容器/分析器
        """

        if "training_features_list_raw" in dk.data:
            feature_list = dk.data["training_features_list_raw"]
        else:
            feature_list = dk.data["training_features_list"]

        if dk.training_features_list != feature_list:
            raise OperationalException(
                "尝试使用 `identifier` 访问预训练模型，但发现当前策略提供的特征不同。"
                "更改 `identifier` 从头开始训练，或确保策略提供与预训练模型相同的特征。"
                "对于 --strategy-list，请注意 FreqAI 要求所有策略保持相同的 feature_engineering_* 函数"
            )

    def define_data_pipeline(self, threads=-1) -> Pipeline:
        ft_params = self.freqai_info["feature_parameters"]
        pipe_steps = [
            ("const", ds.VarianceThreshold(threshold=0)),
            ("scaler", SKLearnWrapper(MinMaxScaler(feature_range=(-1, 1)))),
        ]

        if ft_params.get("principal_component_analysis", False):
            pipe_steps.append(("pca", ds.PCA(n_components=0.999)))
            pipe_steps.append(
                ("post-pca-scaler", SKLearnWrapper(MinMaxScaler(feature_range=(-1, 1))))
            )

        if ft_params.get("use_SVM_to_remove_outliers", False):
            svm_params = ft_params.get("svm_params", {"shuffle": False, "nu": 0.01})
            pipe_steps.append(("svm", ds.SVMOutlierExtractor(**svm_params)))

        di = ft_params.get("DI_threshold", 0)
        if di:
            pipe_steps.append(("di", ds.DissimilarityIndex(di_threshold=di, n_jobs=threads)))

        if ft_params.get("use_DBSCAN_to_remove_outliers", False):
            pipe_steps.append(("dbscan", ds.DBSCAN(n_jobs=threads)))

        sigma = self.freqai_info["feature_parameters"].get("noise_standard_deviation", 0)
        if sigma:
            pipe_steps.append(("noise", ds.Noise(sigma=sigma)))

        return Pipeline(pipe_steps)

    def define_label_pipeline(self, threads=-1) -> Pipeline:
        label_pipeline = Pipeline([("scaler", SKLearnWrapper(MinMaxScaler(feature_range=(-1, 1))))])

        return label_pipeline

    def model_exists(self, dk: FreqaiDataKitchen) -> bool:
        """
        给定交易对和路径，检查模型是否已存在
        :param pair: 交易对，例如 BTC/USD
        :param path: 模型路径
        :return:
        :boolean: 模型文件是否存在。
        """
        if self.dd.model_type == "joblib":
            file_type = ".joblib"
        elif self.dd.model_type in ["stable_baselines3", "sb3_contrib", "pytorch"]:
            file_type = ".zip"

        path_to_modelfile = Path(dk.data_path / f"{dk.model_filename}_model{file_type}")
        file_exists = path_to_modelfile.is_file()
        if file_exists:
            logger.info("在 %s 找到模型", dk.data_path / dk.model_filename)
        else:
            logger.info("在 %s 未找到模型", dk.data_path / dk.model_filename)
        return file_exists

    def set_full_path(self) -> None:
        """
        为标识符创建并设置完整路径
        """
        self.full_path = Path(self.config["user_data_dir"] / "models" / f"{self.identifier}")
        self.full_path.mkdir(parents=True, exist_ok=True)

    def extract_data_and_train_model(
        self,
        new_trained_timerange: TimeRange,
        pair: str,
        strategy: IStrategy,
        dk: FreqaiDataKitchen,
        data_load_timerange: TimeRange,
    ):
        """
        检索数据并训练模型。
        :param new_trained_timerange: TimeRange = 训练模型的时间范围
        :param metadata: dict = 策略提供的元数据
        :param strategy: IStrategy = 用户定义的策略对象
        :param dk: FreqaiDataKitchen = 当前币种/循环的非持久性数据容器
        :param data_load_timerange: TimeRange = 要加载的数据量，用于填充指标（大于 new_trained_timerange，以便 new_trained_timerange 不包含任何 NaN）
        """

        corr_dataframes, base_dataframes = self.dd.get_base_and_corr_dataframes(
            data_load_timerange, pair, dk
        )

        unfiltered_dataframe = dk.use_strategy_to_populate_indicators(
            strategy, corr_dataframes=corr_dataframes, base_dataframes=base_dataframes, pair=pair
        )

        trained_timestamp = new_trained_timerange.stopts

        buffered_timerange = dk.buffer_timerange(new_trained_timerange)

        unfiltered_dataframe = dk.slice_dataframe(buffered_timerange, unfiltered_dataframe)

        # 找到策略指示的特征并存储在数据工具中
        dk.find_features(unfiltered_dataframe)
        dk.find_labels(unfiltered_dataframe)

        self.tb_logger = get_tb_logger(self.dd.model_type, dk.data_path, self.activate_tensorboard)
        model = self.train(unfiltered_dataframe, pair, dk)
        self.tb_logger.close()

        self.dd.pair_dict[pair]["trained_timestamp"] = trained_timestamp
        dk.set_new_model_names(pair, trained_timestamp)
        self.dd.save_data(model, pair, dk)

        if self.plot_features:
            plot_feature_importance(model, pair, dk, self.plot_features)

        self.dd.purge_old_models()

    def set_initial_historic_predictions(
        self, pred_df: DataFrame, dk: FreqaiDataKitchen, pair: str, strat_df: DataFrame
    ) -> None:
        """
        仅当数据抽屉未能加载现有历史预测集时才调用此函数。在这种情况下，它构建结构并基于第一个训练数据设置假预测。之后，FreqAI 将新的真实预测附加到历史预测集。

        这些值用于生成可在策略中用于自适应值的实时统计信息。例如，&*_mean/std 是可以基于历史预测集中的实时预测计算的量。这些值可用于用户策略，以更好地评估预测的稀有性，从而等待相对于实时历史预测概率有利的入场点。

        如果用户在后续实例中重用标识符，则不会调用此函数。在这种情况下，“真实”预测将附加到加载的历史预测集。
        :param pred_df: DataFrame = 包含从模型输出的预测的数据框
        :param dk: FreqaiDataKitchen = 包含数据分析方法的对象
        :param pair: str = 当前交易对
        :param strat_df: DataFrame = 来自策略的数据框
        """

        self.dd.historic_predictions[pair] = pred_df
        hist_preds_df = self.dd.historic_predictions[pair]

        self.set_start_dry_live_date(strat_df)

        for label in hist_preds_df.columns:
            if hist_preds_df[label].dtype == object:
                continue
            hist_preds_df[f"{label}_mean"] = 0
            hist_preds_df[f"{label}_std"] = 0

        hist_preds_df["do_predict"] = 0

        if self.freqai_info["feature_parameters"].get("DI_threshold", 0) > 0:
            hist_preds_df["DI_values"] = 0

        for return_str in dk.data["extra_returns_per_train"]:
            hist_preds_df[return_str] = dk.data["extra_returns_per_train"][return_str]

        hist_preds_df["high_price"] = strat_df["high"]
        hist_preds_df["low_price"] = strat_df["low"]
        hist_preds_df["close_price"] = strat_df["close"]
        hist_preds_df["date_pred"] = strat_df["date"]

    def fit_live_predictions(self, dk: FreqaiDataKitchen, pair: str) -> None:
        """
        用高斯分布拟合标签
        """
        import scipy as spy

        # 如果使用分类器标签类型，则添加类别
        full_labels = dk.label_list + dk.unique_class_list

        num_candles = self.freqai_info.get("fit_live_predictions_candles", 100)
        dk.data["labels_mean"], dk.data["labels_std"] = {}, {}
        for label in full_labels:
            if self.dd.historic_predictions[dk.pair][label].dtype == object:
                continue
            f = spy.stats.norm.fit(self.dd.historic_predictions[dk.pair][label].tail(num_candles))
            dk.data["labels_mean"][label], dk.data["labels_std"][label] = f[0], f[1]

        return

    def inference_timer(self, do: Literal["start", "stop"] = "start", pair: str = ""):
        """
        旨在跟踪 FreqAI 在一次遍历白名单中花费的累积时间的计时器。这将检查花费的时间是否超过单个蜡烛时间的 1/4，如果是，它将警告用户性能下降
        """
        if do == "start":
            self.pair_it += 1
            self.begin_time = time.time()
        elif do == "stop":
            end = time.time()
            time_spent = end - self.begin_time
            if self.freqai_info.get("write_metrics_to_disk", False):
                self.dd.update_metric_tracker("inference_time", time_spent, pair)
            self.inference_time += time_spent
            if self.pair_it == self.total_pairs:
                logger.info(
                    f"遍历交易对白名单总共花费的推理时间 {self.inference_time:.2f} 秒"
                )
                self.pair_it = 0
                self.inference_time = 0
        return

    def train_timer(self, do: Literal["start", "stop"] = "start", pair: str = ""):
        """
        旨在跟踪在 FreqAI 中训练整个交易对白名单所花费的累积时间的计时器。
        """
        if do == "start":
            self.pair_it_train += 1
            self.begin_time_train = time.time()
        elif do == "stop":
            end = time.time()
            time_spent = end - self.begin_time_train
            if self.freqai_info.get("write_metrics_to_disk", False):
                self.dd.collect_metrics(time_spent, pair)

            self.train_time += time_spent
            if self.pair_it_train == self.total_pairs:
                logger.info(f"训练交易对白名单总共花费的时间 {self.train_time:.2f} 秒")
                self.pair_it_train = 0
                self.train_time = 0
        return

    def get_init_model(self, pair: str) -> Any:
        if pair not in self.dd.model_dictionary or not self.continual_learning:
            init_model = None
        else:
            init_model = self.dd.model_dictionary[pair]
            # 设置 "新的" tb_logger - model_dictionary 中的那个已经关闭了写入器。
            init_model.tb_logger = self.tb_logger

        return init_model

    def _set_train_queue(self):
        """
        如果存在现有训练时间戳，则从它们设置训练队列，否则根据提供的白名单设置训练队列。
        """
        current_pairlist = self.config.get("exchange", {}).get("pair_whitelist")
        if not self.dd.pair_dict:
            logger.info(f"从白名单设置新的训练队列。队列: {current_pairlist}")
            return deque(current_pairlist)

        best_queue = deque()

        pair_dict_sorted = sorted(
            self.dd.pair_dict.items(), key=lambda k: k[1]["trained_timestamp"]
        )
        for pair in pair_dict_sorted:
            if pair[0] in current_pairlist:
                best_queue.append(pair[0])
        for pair in current_pairlist:
            if pair not in best_queue:
                best_queue.appendleft(pair)

        logger.info(
            f"从训练时间戳设置现有队列。最佳近似队列: {best_queue}"
        )
        return best_queue

    def cache_corr_pairlist_dfs(self, dataframe: DataFrame, dk: FreqaiDataKitchen) -> DataFrame:
        """
        缓存相关交易对列表数据框，以加快当前蜡烛期间后续交易对的性能。
        :param dataframe: 策略提供的数据框
        :param dk: 当前资产的数据工具对象
        :return: 要附加/从中提取缓存的相关交易对数据框的数据框。
        """

        if self.get_corr_dataframes:
            self.corr_dataframes = dk.extract_corr_pair_columns_from_populated_indicators(dataframe)
            if not self.corr_dataframes:
                logger.warning(
                    "无法缓存相关交易对数据框以提高性能。"
                    "考虑确保在 `feature_engineering_*` 函数中创建特征时，完整的币种/基础货币（例如 XYZ/USD）包含在列名中。"
                )
            self.get_corr_dataframes = not bool(self.corr_dataframes)
        elif self.corr_dataframes:
            dataframe = dk.attach_corr_pair_columns(dataframe, self.corr_dataframes, dk.pair)

        return dataframe

    def track_current_candle(self):
        """
        检查数据抽屉附加的最新蜡烛是否与 FreqAI 看到的最新蜡烛相同。如果不是，它要求刷新缓存的 corr_dfs，并重置交易对计数器。
        """
        if self.dd.current_candle > self.current_candle:
            self.get_corr_dataframes = True
            self.pair_it = 1
            self.current_candle = self.dd.current_candle

    def ensure_data_exists(
        self, len_dataframe_backtest: int, tr_backtest: TimeRange, pair: str
    ) -> bool:
        """
        检查数据框是否为空，如果为空，向用户报告有用信息。
        :param len_dataframe_backtest: 回测数据框的长度
        :param tr_backtest: 当前回测时间范围。
        :param pair: 当前交易对
        :return: 数据是否存在
        """
        if self.config.get("freqai_backtest_live_models", False) and len_dataframe_backtest == 0:
            logger.info(
                f"在交易对 {pair} 上从 {tr_backtest.start_fmt} 到 {tr_backtest.stop_fmt} 未找到数据。"
                "可能在同一蜡烛周期内有多次训练。"
            )
            return False
        return True

    def log_backtesting_progress(
        self, tr_train: TimeRange, pair: str, train_it: int, total_trains: int
    ):
        """
        记录回测进度，以便用户知道已训练了多少交易对以及还剩多少交易对/训练。
        :param tr_train: 训练时间范围
        :param train_it: 当前交易对的训练迭代（滑动窗口进度）
        :param pair: 当前交易对
        :param total_trains: 总训练次数（滑动窗口的总滑动次数）
        """
        if not self.config.get("freqai_backtest_live_models", False):
            logger.info(
                f"训练 {pair}，{self.pair_it}/{self.total_pairs} 个交易对，从 {tr_train.start_fmt} "
                f"到 {tr_train.stop_fmt}，{train_it}/{total_trains} 次训练"
            )

    def backtesting_fit_live_predictions(self, dk: FreqaiDataKitchen):
        """
        使用虚拟的 historic_predictions 在回测中应用 fit_live_predictions 函数
        需要循环来模拟模拟/实盘操作，因为无法预测用户实现的逻辑类型。
        :param dk: 数据工具对象
        """
        fit_live_predictions_candles = self.freqai_info.get("fit_live_predictions_candles", 0)
        if fit_live_predictions_candles:
            logger.info("在回测中应用 fit_live_predictions")
            label_columns = [
                col
                for col in dk.full_df.columns
                if (
                    col.startswith("&")
                    and not (col.startswith("&") and col.endswith("_mean"))
                    and not (col.startswith("&") and col.endswith("_std"))
                    and col not in self.dk.data["extra_returns_per_train"]
                )
            ]

            for index in range(len(dk.full_df)):
                if index >= fit_live_predictions_candles:
                    self.dd.historic_predictions[self.dk.pair] = dk.full_df.iloc[
                        index - fit_live_predictions_candles : index
                    ]
                    self.fit_live_predictions(self.dk, self.dk.pair)
                    for label in label_columns:
                        if dk.full_df[label].dtype == object:
                            continue
                        if "labels_mean" in self.dk.data:
                            dk.full_df.at[index, f"{label}_mean"] = self.dk.data["labels_mean"][
                                label
                            ]
                        if "labels_std" in self.dk.data:
                            dk.full_df.at[index, f"{label}_std"] = self.dk.data["labels_std"][label]

                    for extra_col in self.dk.data["extra_returns_per_train"]:
                        dk.full_df.at[index, f"{extra_col}"] = self.dk.data[
                            "extra_returns_per_train"
                        ][extra_col]

        return

    def update_metadata(self, metadata: dict[str, Any]):
        """
        更新全局元数据并保存更新的 json 文件
        :param metadata: 新的全局元数据字典
        """
        self.dd.save_global_metadata_to_disk(metadata)
        self.metadata = metadata

    def set_start_dry_live_date(self, live_dataframe: DataFrame):
        key_name = "start_dry_live_date"
        if key_name not in self.metadata:
            metadata = self.metadata
            metadata[key_name] = int(
                pd.to_datetime(live_dataframe.tail(1)["date"].values[0]).timestamp()
            )
            self.update_metadata(metadata)

    def start_backtesting_from_historic_predictions(
        self, dataframe: DataFrame, metadata: dict, dk: FreqaiDataKitchen
    ) -> FreqaiDataKitchen:
        """
        :param dataframe: DataFrame = 策略传递的数据框
        :param metadata: Dict = 交易对元数据
        :param dk: FreqaiDataKitchen = 仅与当前交易对相关联的数据管理/分析工具
        :return:
            FreqaiDataKitchen = 仅与当前交易对相关联的数据管理/分析工具
        """
        pair = metadata["pair"]
        dk.return_dataframe = dataframe
        saved_dataframe = self.dd.historic_predictions[pair]
        columns_to_drop = list(
            set(saved_dataframe.columns).intersection(dk.return_dataframe.columns)
        )
        dk.return_dataframe = dk.return_dataframe.drop(columns=list(columns_to_drop))
        dk.return_dataframe = pd.merge(
            dk.return_dataframe, saved_dataframe, how="left", left_on="date", right_on="date_pred"
        )
        return dk

    # 以下方法由用户创建的预测模型覆盖。
    # 参见 freqai/prediction_models/CatboostPredictionModel.py 中的示例。

    @abstractmethod
    def train(self, unfiltered_df: DataFrame, pair: str, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        过滤训练数据并为其训练模型。Train 大量使用数据处理程序来存储、保存、加载和分析数据。
        :param unfiltered_df: 当前训练期间的完整数据框
        :param metadata: 来自策略的交易对元数据。
        :return: 可用于推理的训练模型（self.predict）
        """

    @abstractmethod
    def fit(self, data_dictionary: dict[str, Any], dk: FreqaiDataKitchen,** kwargs) -> Any:
        """
        大多数回归器使用相同的函数名称和参数，例如用户可以用 LGBMRegressor 代替 CatBoostRegressor，所有数据管理都将由 Freqai 正确处理。
        :param data_dictionary: Dict = 由 DataHandler 构建的字典，用于保存所有训练和测试数据/标签。
        """

        return

    @abstractmethod
    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> tuple[DataFrame, NDArray[np.int_]]:
        """
        过滤预测特征数据并进行预测。
        :param unfiltered_df: 当前回测期间的完整数据框。
        :param dk: FreqaiDataKitchen = 仅与当前交易对相关联的数据管理/分析工具
        :param first: boolean = 这是否是第一次预测。
        :return:
        :predictions: 预测的 np.array
        :do_predict: 1 和 0 的 np.array，用于指示 freqai 需要删除数据（NaNs）或对数据不确定的地方（即 SVM 和/或 DI 指数）
        """

    # 已弃用的函数
    def data_cleaning_train(self, dk: FreqaiDataKitchen, pair: str):
        """
        如果调用此函数，则抛出弃用警告
        """
        logger.warning(
            f"您的模型 {self.__class__.__name__} 依赖于已弃用的数据管道。请更新您的模型以使用新的数据管道。"
            " 这可以通过遵循迁移指南来实现，网址为 "
            f"{DOCS_LINK}/strategy_migration/#freqai-new-data-pipeline"
        )
        dk.feature_pipeline = self.define_data_pipeline(threads=dk.thread_count)
        dd = dk.data_dictionary
        (dd["train_features"], dd["train_labels"], dd["train_weights"]) = (
            dk.feature_pipeline.fit_transform(
                dd["train_features"], dd["train_labels"], dd["train_weights"]
            )
        )

        (dd["test_features"], dd["test_labels"], dd["test_weights"]) = (
            dk.feature_pipeline.transform(
                dd["test_features"], dd["test_labels"], dd["test_weights"]
            )
        )

        dk.label_pipeline = self.define_label_pipeline(threads=dk.thread_count)

        dd["train_labels"], _, _ = dk.label_pipeline.fit_transform(dd["train_labels"])
        dd["test_labels"], _, _ = dk.label_pipeline.transform(dd["test_labels"])
        return

    def data_cleaning_predict(self, dk: FreqaiDataKitchen, pair: str):
        """
        如果调用此函数，则抛出弃用警告
        """
        logger.warning(
            f"您的模型 {self.__class__.__name__} 依赖于已弃用的数据管道。请更新您的模型以使用新的数据管道。"
            " 这可以通过遵循迁移指南来实现，网址为 "
            f"{DOCS_LINK}/strategy_migration/#freqai-new-data-pipeline"
        )
        dd = dk.data_dictionary
        dd["predict_features"], outliers, _ = dk.feature_pipeline.transform(
            dd["predict_features"], outlier_check=True
        )
        if self.freqai_info.get("DI_threshold", 0) > 0:
            dk.DI_values = dk.feature_pipeline["di"].di_values
        else:
            dk.DI_values = np.zeros(outliers.shape[0])
        dk.do_predict = outliers
        return