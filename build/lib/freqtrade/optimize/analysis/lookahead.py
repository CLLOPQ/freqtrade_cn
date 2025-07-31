import logging
import shutil
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pandas import DataFrame

from freqtrade.data.history import get_timerange
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.loggers.set_log_levels import (
    reduce_verbosity_for_bias_tester,
    restore_verbosity_for_bias_tester,
)
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.optimize.base_analysis import BaseAnalysis, VarHolder


logger = logging.getLogger(__name__)


class Analysis:
    def __init__(self) -> None:
        self.total_signals = 0  # 总信号数
        self.false_entry_signals = 0  # 有偏差的入场信号数
        self.false_exit_signals = 0  # 有偏差的出场信号数
        self.false_indicators: list[str] = []  # 有偏差的指标列表
        self.has_bias = False  # 是否存在偏差


class LookaheadAnalysis(BaseAnalysis):
    def __init__(self, config: dict[str, Any], strategy_obj: dict):
        super().__init__(config, strategy_obj)

        self.entry_varHolders: list[VarHolder] = []  # 入场变量持有者列表
        self.exit_varHolders: list[VarHolder] = []  # 出场变量持有者列表

        self.current_analysis = Analysis()
        self.minimum_trade_amount = config["minimum_trade_amount"]  # 最小交易数量
        self.targeted_trade_amount = config["targeted_trade_amount"]  # 目标交易数量

    @staticmethod
    def get_result(backtesting: Backtesting, processed: DataFrame):
        min_date, max_date = get_timerange(processed)

        result = backtesting.backtest(
            processed=deepcopy(processed), start_date=min_date, end_date=max_date
        )
        return result

    @staticmethod
    def report_signal(result: dict, column_name: str, checked_timestamp: datetime):
        df = result["results"]
        row_count = df[column_name].shape[0]

        if row_count == 0:
            return False
        else:
            df_cut = df[(df[column_name] == checked_timestamp)]
            if df_cut[column_name].shape[0] == 0:
                return False
            else:
                return True
        return False

    # 分析两个带有已处理指标的数据框，并显示它们之间的差异
    def analyze_indicators(self, full_vars: VarHolder, cut_vars: VarHolder, current_pair: str):
        # 提取数据框
        cut_df: DataFrame = cut_vars.indicators[current_pair]
        full_df: DataFrame = full_vars.indicators[current_pair]

        # 将较长的数据框裁剪为较短数据框的长度
        full_df_cut = full_df[(full_df.date == cut_vars.compared_dt)].reset_index(drop=True)
        cut_df_cut = cut_df[(cut_df.date == cut_vars.compared_dt)].reset_index(drop=True)

        # 检查数据框是否非空
        if full_df_cut.shape[0] != 0 and cut_df_cut.shape[0] != 0:
            # 比较数据框
            compare_df = full_df_cut.compare(cut_df_cut)

            if compare_df.shape[0] > 0:
                for col_name, values in compare_df.items():
                    col_idx = compare_df.columns.get_loc(col_name)
                    compare_df_row = compare_df.iloc[0]
                    # compare_df现在包含元组，其中[1]要么是'self'要么是'other'
                    if "other" in col_name[1]:
                        continue
                    self_value = compare_df_row.iloc[col_idx]
                    other_value = compare_df_row.iloc[col_idx + 1]

                    # 输出差异
                    if self_value != other_value:
                        if not self.current_analysis.false_indicators.__contains__(col_name[0]):
                            self.current_analysis.false_indicators.append(col_name[0])
                            logger.info(
                                f"=> 在指标{col_name[0]}中发现前瞻偏差。"
                                f"{str(self_value)} != {str(other_value)}"
                            )

    def prepare_data(self, varholder: VarHolder, pairs_to_load: list[DataFrame]):
        if "freqai" in self.local_config and "identifier" in self.local_config["freqai"]:
            # 如果定义了freqai模型，则清除先前的数据
            # （确保没有从较旧的回测中遗留任何内容）
            path_to_current_identifier = Path(
                f"{self.local_config['user_data_dir']}/models/"
                f"{self.local_config['freqai']['identifier']}"
            ).resolve()
            # 删除文件夹及其内容
            if Path.exists(path_to_current_identifier):
                shutil.rmtree(path_to_current_identifier)

        prepare_data_config = deepcopy(self.local_config)
        prepare_data_config["timerange"] = (
            str(self.dt_to_timestamp(varholder.from_dt))
            + "-"
            + str(self.dt_to_timestamp(varholder.to_dt))
        )
        prepare_data_config["exchange"]["pair_whitelist"] = pairs_to_load

        if self._fee is not None:
            # 不要按交易对重新计算费用，因为费用可能因交易对而异
            prepare_data_config["fee"] = self._fee

        backtesting = Backtesting(prepare_data_config, self.exchange)
        self.exchange = backtesting.exchange
        self._fee = backtesting.fee
        backtesting._set_strategy(backtesting.strategylist[0])

        varholder.data, varholder.timerange = backtesting.load_bt_data()
        varholder.timeframe = backtesting.timeframe

        varholder.indicators = backtesting.strategy.advise_all_indicators(varholder.data)
        varholder.result = self.get_result(backtesting, varholder.indicators)

    def fill_entry_and_exit_varHolders(self, result_row):
        # 入场变量持有者
        entry_varHolder = VarHolder()
        self.entry_varHolders.append(entry_varHolder)
        entry_varHolder.from_dt = self.full_varHolder.from_dt
        entry_varHolder.compared_dt = result_row["open_date"]
        # to_dt需要+1根K线，因为不会在最后一根K线买入
        entry_varHolder.to_dt = result_row["open_date"] + timedelta(
            minutes=timeframe_to_minutes(self.full_varHolder.timeframe)
        )
        self.prepare_data(entry_varHolder, [result_row["pair"]])

        # 出场变量持有者
        exit_varHolder = VarHolder()
        self.exit_varHolders.append(exit_varHolder)
        # to_dt需要+1根K线，因为总是在最后一根K线平仓/强制平仓
        exit_varHolder.from_dt = self.full_varHolder.from_dt
        exit_varHolder.to_dt = result_row["close_date"] + timedelta(
            minutes=timeframe_to_minutes(self.full_varHolder.timeframe)
        )
        exit_varHolder.compared_dt = result_row["close_date"]
        self.prepare_data(exit_varHolder, [result_row["pair"]])

    # 现在我们分析full_varholder的完整交易并寻找其偏差
    def analyze_row(self, idx: int, result_row):
        # 如果是强制平仓，忽略这个信号，因为这里会无条件平仓
        if result_row.close_date == self.dt_to_timestamp(self.full_varHolder.to_dt):
            return

        # 跟踪总共处理了多少信号
        self.current_analysis.total_signals += 1

        # 填充入场和出场变量持有者
        self.fill_entry_and_exit_varHolders(result_row)

        # 这将触发一个日志消息
        buy_or_sell_biased: bool = False

        # 记录买入信号是否失效
        if not self.report_signal(
            self.entry_varHolders[idx].result, "open_date", self.entry_varHolders[idx].compared_dt
        ):
            self.current_analysis.false_entry_signals += 1
            buy_or_sell_biased = True

        # 记录买入或卖出信号是否失效
        if not self.report_signal(
            self.exit_varHolders[idx].result, "close_date", self.exit_varHolders[idx].compared_dt
        ):
            self.current_analysis.false_exit_signals += 1
            buy_or_sell_biased = True

        if buy_or_sell_biased:
            logger.info(
                f"在交易中发现前瞻偏差 "
                f"交易对: {result_row['pair']}, "
                f"时间范围: {result_row['open_date']} - {result_row['close_date']}, "
                f"索引: {idx}"
            )

        # 检查指标本身是否包含有偏差的数据
        self.analyze_indicators(self.full_varHolder, self.entry_varHolders[idx], result_row["pair"])
        self.analyze_indicators(self.full_varHolder, self.exit_varHolders[idx], result_row["pair"])

    def start(self) -> None:
        super().start()

        reduce_verbosity_for_bias_tester()

        # 检查full_varholder的要求是否满足
        found_signals: int = self.full_varHolder.result["results"].shape[0] + 1
        if found_signals >= self.targeted_trade_amount:
            logger.info(
                f"发现{found_signals}笔交易，计算{self.targeted_trade_amount}笔交易。"
            )
        elif self.targeted_trade_amount >= found_signals >= self.minimum_trade_amount:
            logger.info(f"只发现{found_signals}笔交易。计算所有可用交易。")
        else:
            logger.info(
                f"发现{found_signals}笔交易 "
                f"少于最小交易数量{self.minimum_trade_amount}。"
                f"取消这次回测前瞻偏差测试。"
            )
            return

        # 现在我们遍历所有信号
        # 从相同的日期时间开始，以避免偏差的错误报告
        for idx, result_row in self.full_varHolder.result["results"].iterrows():
            if self.current_analysis.total_signals == self.targeted_trade_amount:
                logger.info(f"找到目标交易数量 = {self.targeted_trade_amount}个信号。")
                break
            if found_signals < self.minimum_trade_amount:
                logger.info(
                    f"只发现{found_signals}笔交易 "
                    f"少于 "
                    f"最小交易数量 = {self.minimum_trade_amount}。"
                    f"退出这次前瞻分析"
                )
                return None
            if "force_exit" in result_row["exit_reason"]:
                logger.info(
                    f"在交易对: {result_row['pair']}中发现强制平仓，"
                    f"时间范围: {result_row['open_date']}-{result_row['close_date']}, "
                    f"索引: {idx}，跳过这个以避免误报。"
                )

                # 只是为了保持full、entry和exit varholders的ID相同
                # 以获得更好的调试体验
                self.entry_varHolders.append(VarHolder())
                self.exit_varHolders.append(VarHolder())
                continue

            self.analyze_row(idx, result_row)

        if len(self.entry_varHolders) < self.minimum_trade_amount:
            logger.info(
                f"跳过强制平仓后只发现{found_signals}笔交易 "
                f"少于 "
                f"最小交易数量 = {self.minimum_trade_amount}。"
                f"退出这次前瞻分析"
            )

        # 恢复详细程度，这样下一个策略就不会太安静
        restore_verbosity_for_bias_tester()
        # 检查并报告信号
        if self.current_analysis.total_signals < self.local_config["minimum_trade_amount"]:
            logger.info(
                f" -> {self.local_config['strategy']} : 交易太少。"
                f"我们只发现{self.current_analysis.total_signals}笔交易。"
                f"提示: 延长时间范围 "
                f"以获得至少{self.local_config['minimum_trade_amount']}笔交易 "
                f"或降低minimum_trade_amount的值。"
            )
            self.failed_bias_check = True
        elif (
            self.current_analysis.false_entry_signals > 0
            or self.current_analysis.false_exit_signals > 0
            or len(self.current_analysis.false_indicators) > 0
        ):
            logger.info(f" => {self.local_config['strategy']} : 检测到偏差!")
            self.current_analysis.has_bias = True
            self.failed_bias_check = False
        else:
            logger.info(self.local_config["strategy"] + ": 未检测到偏差")
            self.failed_bias_check = False