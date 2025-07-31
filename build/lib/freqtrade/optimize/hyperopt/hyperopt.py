# pragma pylint: disable=too-many-instance-attributes, pointless-string-statement

"""
此模块包含超参数优化逻辑
"""

import gc
import logging
import random
from datetime import datetime
from math import ceil
from multiprocessing import Manager
from pathlib import Path
from typing import Any

import rapidjson
from joblib import Parallel, cpu_count
from optuna.trial import FrozenTrial, Trial, TrialState

from freqtrade.constants import FTHYPT_FILEVERSION, LAST_BT_RESULT_FN, Config
from freqtrade.enums import HyperoptState
from freqtrade.exceptions import OperationalException
from freqtrade.misc import file_dump_json, plural
from freqtrade.optimize.hyperopt.hyperopt_logger import logging_mp_handle, logging_mp_setup
from freqtrade.optimize.hyperopt.hyperopt_optimizer import INITIAL_POINTS, HyperOptimizer
from freqtrade.optimize.hyperopt.hyperopt_output import HyperoptOutput
from freqtrade.optimize.hyperopt_tools import (
    HyperoptStateContainer,
    HyperoptTools,
    hyperopt_serializer,
)
from freqtrade.util import get_progress_tracker


logger = logging.getLogger(__name__)


log_queue: Any


class Hyperopt:
    """
    超参数优化类，包含运行超参数优化模拟的所有逻辑

    启动超参数优化运行：
    hyperopt = Hyperopt(config)
    hyperopt.start()
    """

    def __init__(self, config: Config) -> None:
        self._hyper_out: HyperoptOutput = HyperoptOutput(streaming=True)  # 超参数优化输出

        self.config = config  # 配置

        self.analyze_per_epoch = self.config.get("analyze_per_epoch", False)  # 是否每轮分析
        HyperoptStateContainer.set_state(HyperoptState.STARTUP)  # 设置初始状态

        if self.config.get("hyperopt"):
            raise OperationalException(
                "在2021.9版本中已移除使用单独的Hyperopt文件。请将您现有的Hyperopt文件转换为新的可超参数优化策略接口"
            )

        time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 当前时间
        strategy = str(self.config["strategy"])  # 策略名称
        self.results_file: Path = (
            self.config["user_data_dir"]
            / "hyperopt_results"
            / f"strategy_{strategy}_{time_now}.fthypt"
        )  # 结果文件路径
        self.data_pickle_file = (
            self.config["user_data_dir"] / "hyperopt_results" / "hyperopt_tickerdata.pkl"
        )  # 数据序列化文件路径
        self.total_epochs = config.get("epochs", 0)  # 总轮数

        self.current_best_loss = 100  # 当前最佳损失值

        self.clean_hyperopt()  # 清理超参数优化文件

        self.num_epochs_saved = 0  # 已保存的轮数
        self.current_best_epoch: dict[str, Any] | None = None  # 当前最佳轮次

        if HyperoptTools.has_space(self.config, "sell"):
            # 确保使用出场信号
            self.config["use_exit_signal"] = True

        self.print_all = self.config.get("print_all", False)  # 是否打印所有结果
        self.hyperopt_table_header = 0  # 超参数优化表格表头
        self.print_json = self.config.get("print_json", False)  # 是否以JSON格式打印

        self.hyperopter = HyperOptimizer(self.config, self.data_pickle_file)  # 超参数优化器实例
        self.count_skipped_epochs = 0  # 跳过的轮数计数

    @staticmethod
    def get_lock_filename(config: Config) -> str:
        """获取锁文件名称"""
        return str(config["user_data_dir"] / "hyperopt.lock")

    def clean_hyperopt(self) -> None:
        """
        删除超参数优化序列化文件以重新开始超参数优化。
        """
        for f in [self.data_pickle_file, self.results_file]:
            p = Path(f)
            if p.is_file():
                logger.info(f"正在删除 `{p}`。")
                p.unlink()

    def _save_result(self, epoch: dict) -> None:
        """
        将超参数优化结果保存到文件
        每轮存储一行。
        虽然不是有效的json对象，但这允许轻松追加。
        :param epoch: 本轮的结果字典。
        """
        epoch[FTHYPT_FILEVERSION] = 2
        with self.results_file.open("a") as f:
            rapidjson.dump(
                epoch,
                f,
                default=hyperopt_serializer,
                number_mode=rapidjson.NM_NATIVE | rapidjson.NM_NAN,
            )
            f.write("\n")

        self.num_epochs_saved += 1
        logger.debug(
            f"{self.num_epochs_saved} {plural(self.num_epochs_saved, '轮')} "
            f"已保存到 '{self.results_file}'。"
        )
        # 存储超参数优化文件名
        latest_filename = Path.joinpath(self.results_file.parent, LAST_BT_RESULT_FN)
        file_dump_json(latest_filename, {"latest_hyperopt": str(self.results_file.name)}, log=False)

    def print_results(self, results: dict[str, Any]) -> None:
        """
        如果结果比任何先前的评估更好，则记录结果
        TODO: 这也应该移到HyperoptTools
        """
        is_best = results["is_best"]  # 是否为最佳结果

        if self.print_all or is_best:
            self._hyper_out.add_data(
                self.config,
                [results],
                self.total_epochs,
                self.print_all,
            )

    def run_optimizer_parallel(self, parallel: Parallel, asked: list[list]) -> list[dict[str, Any]]:
        """以并行方式启动优化器"""

        def optimizer_wrapper(*args, **kwargs):
            # 全局日志队列。这必须发生在初始化Parallel的文件中
            logging_mp_setup(
                log_queue, logging.INFO if self.config["verbosity"] < 1 else logging.DEBUG
            )

            return self.hyperopter.generate_optimizer_wrapped(*args, **kwargs)

        return parallel(optimizer_wrapper(v) for v in asked)

    def _set_random_state(self, random_state: int | None) -> int:
        """设置随机状态"""
        return random_state or random.randint(1, 2**16 - 1)  # noqa: S311

    def get_optuna_asked_points(self, n_points: int, dimensions: dict) -> list[Any]:
        """获取Optuna询问的点"""
        asked: list[list[Any]] = []
        for i in range(n_points):
            asked.append(self.opt.ask(dimensions))
        return asked

    def duplicate_optuna_asked_points(self, trial: Trial, asked_trials: list[FrozenTrial]) -> bool:
        """检查Optuna询问的点是否重复"""
        asked_trials_no_dups: list[FrozenTrial] = []
        trials_to_consider = trial.study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        # 检查我们是否已经评估了采样的`params`
        for t in reversed(trials_to_consider):
            if trial.params == t.params:
                return True
        # 检查同一批次中是否有相同的`params`（asked_trials）。自动采样器正在做这个
        for t in asked_trials:
            if t.params not in asked_trials_no_dups:
                asked_trials_no_dups.append(t)
        if len(asked_trials_no_dups) != len(asked_trials):
            return True
        return False

    def get_asked_points(self, n_points: int, dimensions: dict) -> tuple[list[Any], list[bool]]:
        """
        确保从`self.opt.ask`返回的点尚未被评估

        步骤：
        1. 首先尝试使用`self.opt.ask`获取点
        2. 丢弃已经评估过的点
        3. 最多使用`self.opt.ask`重试`n_points`次
        """
        asked_non_tried: list[FrozenTrial] = []
        optuna_asked_trials = self.get_optuna_asked_points(n_points=n_points, dimensions=dimensions)
        asked_non_tried += [
            x
            for x in optuna_asked_trials
            if not self.duplicate_optuna_asked_points(x, optuna_asked_trials)
        ]
        i = 0
        while i < 2 * n_points and len(asked_non_tried) < n_points:
            asked_new = self.get_optuna_asked_points(n_points=1, dimensions=dimensions)[0]
            if not self.duplicate_optuna_asked_points(asked_new, asked_non_tried):
                asked_non_tried.append(asked_new)
            i += 1
        if len(asked_non_tried) < n_points:
            if self.count_skipped_epochs == 0:
                logger.warning("检测到重复参数。可能您的搜索空间太小？")
            self.count_skipped_epochs += n_points - len(asked_non_tried)

        return asked_non_tried, [False for _ in range(len(asked_non_tried))]

    def evaluate_result(self, val: dict[str, Any], current: int, is_random: bool):
        """
        评估从generate_optimizer返回的结果
        """
        val["current_epoch"] = current  # 当前轮次
        val["is_initial_point"] = current <= INITIAL_POINTS  # 是否为初始点

        logger.debug("已评估优化器轮次: %s", val)

        is_best = HyperoptTools.is_best_loss(val, self.current_best_loss)  # 是否为最佳结果
        # 这个值在这里分配，而不是在优化方法中
        # 以保持结果列表中的正确顺序。这是因为
        # 评估可能需要不同的时间。这里它们按顺序排列
        # 它们将向用户显示的顺序。
        val["is_best"] = is_best
        val["is_random"] = is_random
        self.print_results(val)  # 打印结果

        if is_best:
            self.current_best_loss = val["loss"]  # 更新最佳损失值
            self.current_best_epoch = val  # 更新最佳轮次

        self._save_result(val)  # 保存结果

    def _setup_logging_mp_workaround(self) -> None:
        """
        子进程中日志记录的解决方法。
        local_queue必须是初始化Parallel的文件中的全局变量。
        """
        global log_queue
        m = Manager()
        log_queue = m.Queue()

    def start(self) -> None:
        """启动超参数优化"""
        self.random_state = self._set_random_state(self.config.get("hyperopt_random_state"))
        logger.info(f"使用优化器随机状态: {self.random_state}")
        self.hyperopt_table_header = -1
        self.hyperopter.prepare_hyperopt()  # 准备超参数优化

        cpus = cpu_count()  # CPU核心数
        logger.info(f"发现 {cpus} 个CPU核心。让它们全速运转吧！")
        config_jobs = self.config.get("hyperopt_jobs", -1)  # 配置的并行作业数
        logger.info(f"设置的并行作业数为: {config_jobs}")

        self.opt = self.hyperopter.get_optimizer(self.random_state)  # 获取优化器
        self._setup_logging_mp_workaround()  # 设置多进程日志
        try:
            with Parallel(n_jobs=config_jobs) as parallel:
                jobs = parallel._effective_n_jobs()  # 有效并行工作数
                logger.info(f"使用的有效并行工作数: {jobs}")

                # 定义进度条
                with get_progress_tracker(cust_callables=[self._hyper_out]) as pbar:
                    task = pbar.add_task("轮次", total=self.total_epochs)  # 添加任务

                    start = 0  # 开始轮次

                    if self.analyze_per_epoch:
                        # 使用--analyze-per-epoch时，第一次分析不在并行模式下。
                        # 这允许数据提供程序加载其信息缓存。
                        asked, is_random = self.get_asked_points(
                            n_points=1, dimensions=self.hyperopter.o_dimensions
                        )
                        f_val0 = self.hyperopter.generate_optimizer(asked[0].params)
                        self.opt.tell(asked[0], [f_val0["loss"]])
                        self.evaluate_result(f_val0, 1, is_random[0])
                        pbar.update(task, advance=1)
                        start += 1

                    evals = ceil((self.total_epochs - start) / jobs)  # 评估次数
                    for i in range(evals):
                        # 纠正最后一次迭代要处理的轮数（总共不应超过self.total_epochs）
                        n_rest = (i + 1) * jobs - (self.total_epochs - start)
                        current_jobs = jobs - n_rest if n_rest > 0 else jobs

                        asked, is_random = self.get_asked_points(
                            n_points=current_jobs, dimensions=self.hyperopter.o_dimensions
                        )

                        f_val = self.run_optimizer_parallel(
                            parallel,
                            [asked1.params for asked1 in asked],
                        )

                        f_val_loss = [v["loss"] for v in f_val]
                        for o_ask, v in zip(asked, f_val_loss, strict=False):
                            self.opt.tell(o_ask, v)

                        for j, val in enumerate(f_val):
                            # 这里使用人类友好的索引（从1开始）
                            current = i * jobs + j + 1 + start

                            self.evaluate_result(val, current, is_random[j])
                            pbar.update(task, advance=1)
                        logging_mp_handle(log_queue)  # 处理日志
                        gc.collect()  # 垃圾回收

                        if (
                            self.hyperopter.es_epochs > 0
                            and self.hyperopter.es_terminator.should_terminate(self.opt)
                        ):
                            logger.info(f"在 {(i + 1) * jobs} 轮后早停")
                            break

        except KeyboardInterrupt:
            print("用户中断..")

        if self.count_skipped_epochs > 0:
            logger.info(
                f"{self.count_skipped_epochs} {plural(self.count_skipped_epochs, '轮')} "
                f"因重复参数而被跳过。"
            )

        logger.info(
            f"{self.num_epochs_saved} {plural(self.num_epochs_saved, '轮')} "
            f"已保存到 '{self.results_file}'。"
        )

        if self.current_best_epoch:
            HyperoptTools.try_export_params(
                self.config,
                self.hyperopter.get_strategy_name(),
                self.current_best_epoch,
            )

            HyperoptTools.show_epoch_details(
                self.current_best_epoch, self.total_epochs, self.print_json
            )
        elif self.num_epochs_saved > 0:
            print(
                f"在 {self.num_epochs_saved} {plural(self.num_epochs_saved, '轮')} 中，"
                f"未找到给定优化函数的良好结果。"
            )
        else:
            # 当快速按下Ctrl+C时打印，在第一轮有机会评估之前
            print("尚未评估任何轮次，没有最佳结果。")