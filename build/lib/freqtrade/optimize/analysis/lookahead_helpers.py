import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
from rich.text import Text

from freqtrade.constants import Config
from freqtrade.exceptions import OperationalException
from freqtrade.optimize.analysis.lookahead import LookaheadAnalysis
from freqtrade.resolvers import StrategyResolver
from freqtrade.util import get_dry_run_wallet, print_rich_table


logger = logging.getLogger(__name__)


class LookaheadAnalysisSubFunctions:
    @staticmethod
    def text_table_lookahead_analysis_instances(
        config: dict[str, Any],
        lookahead_instances: list[LookaheadAnalysis],
        caption: str | None = None,
    ):
        headers = [
            "文件名",
            "策略",
            "存在偏差",
            "总信号数",
            "有偏差的入场信号",
            "有偏差的出场信号",
            "有偏差的指标",
        ]
        data = []
        for inst in lookahead_instances:
            if config["minimum_trade_amount"] > inst.current_analysis.total_signals:
                data.append(
                    [
                        inst.strategy_obj["location"].parts[-1],
                        inst.strategy_obj["name"],
                        f"捕获的交易太少 "
                        f"({inst.current_analysis.total_signals}/{config['minimum_trade_amount']})。"
                        f"测试失败。",
                    ]
                )
            elif inst.failed_bias_check:
                data.append(
                    [
                        inst.strategy_obj["location"].parts[-1],
                        inst.strategy_obj["name"],
                        "检查时出错",
                    ]
                )
            else:
                data.append(
                    [
                        inst.strategy_obj["location"].parts[-1],
                        inst.strategy_obj["name"],
                        Text("是", style="bold red")
                        if inst.current_analysis.has_bias
                        else Text("否", style="bold green"),
                        inst.current_analysis.total_signals,
                        inst.current_analysis.false_entry_signals,
                        inst.current_analysis.false_exit_signals,
                        ", ".join(inst.current_analysis.false_indicators),
                    ]
                )

        print_rich_table(
            data, headers, summary="前瞻分析", table_kwargs={"caption": caption}
        )
        return data

    @staticmethod
    def export_to_csv(config: dict[str, Any], lookahead_analysis: list[LookaheadAnalysis]):
        def add_or_update_row(df, row_data):
            if (
                (df["文件名"] == row_data["文件名"]) & (df["策略"] == row_data["策略"])
            ).any():
                # 更新现有行
                pd_series = pd.DataFrame([row_data])
                df.loc[
                    (df["文件名"] == row_data["文件名"])
                    & (df["策略"] == row_data["策略"])
                ] = pd_series
            else:
                # 添加新行
                df = pd.concat([df, pd.DataFrame([row_data], columns=df.columns)])

            return df

        if Path(config["lookahead_analysis_exportfilename"]).exists():
            # 将CSV文件读入pandas数据框
            csv_df = pd.read_csv(config["lookahead_analysis_exportfilename"])
        else:
            # 创建一个新的空数据框，包含所需的列名并设置索引
            csv_df = pd.DataFrame(
                columns=[
                    "文件名",
                    "策略",
                    "存在偏差",
                    "总信号数",
                    "有偏差的入场信号",
                    "有偏差的出场信号",
                    "有偏差的指标",
                ],
                index=None,
            )

        for inst in lookahead_analysis:
            # 仅在满足条件时更新
            if (
                inst.current_analysis.total_signals > config["minimum_trade_amount"]
                and inst.failed_bias_check is not True
            ):
                new_row_data = {
                    "文件名": inst.strategy_obj["location"].parts[-1],
                    "策略": inst.strategy_obj["name"],
                    "存在偏差": inst.current_analysis.has_bias,
                    "总信号数": int(inst.current_analysis.total_signals),
                    "有偏差的入场信号": int(inst.current_analysis.false_entry_signals),
                    "有偏差的出场信号": int(inst.current_analysis.false_exit_signals),
                    "有偏差的指标": ",".join(inst.current_analysis.false_indicators),
                }
                csv_df = add_or_update_row(csv_df, new_row_data)

        # 用默认值填充NaN值（例如0）
        csv_df["总信号数"] = csv_df["总信号数"].astype(int).fillna(0)
        csv_df["有偏差的入场信号"] = csv_df["有偏差的入场信号"].astype(int).fillna(0)
        csv_df["有偏差的出场信号"] = csv_df["有偏差的出场信号"].astype(int).fillna(0)

        # 将列转换为整数
        csv_df["总信号数"] = csv_df["总信号数"].astype(int)
        csv_df["有偏差的入场信号"] = csv_df["有偏差的入场信号"].astype(int)
        csv_df["有偏差的出场信号"] = csv_df["有偏差的出场信号"].astype(int)

        logger.info(f"保存 {config['lookahead_analysis_exportfilename']}")
        csv_df.to_csv(config["lookahead_analysis_exportfilename"], index=False)

    @staticmethod
    def calculate_config_overrides(config: Config):
        if config.get("enable_protections", False):
            # 如果全局启用了保护机制，可能会产生误报
            config["enable_protections"] = False
            logger.info(
                "保护机制已启用。"
                "现在正在禁用保护机制 "
                "因为它们可能会产生误报。"
            )
        if config["targeted_trade_amount"] < config["minimum_trade_amount"]:
            # 这种组合没有任何意义
            raise OperationalException(
                "目标交易数量不能小于最小交易数量。"
            )
        config["max_open_trades"] = -1
        logger.info("强制将max_open_trades设置为-1（与交易对数量相同）")

        min_dry_run_wallet = 1000000000
        if get_dry_run_wallet(config) < min_dry_run_wallet:
            logger.info(
                "模拟交易钱包未设置为10亿，将其提高到该值 "
                "只是为了避免误报"
            )
            config["dry_run_wallet"] = min_dry_run_wallet

        if "timerange" not in config:
            # 这里强制要求设置时间范围
            raise OperationalException(
                "请设置时间范围。"
                "通常几个月就足够了，具体取决于您的需求和策略。"
            )
        # 固定stake_amount为10k
        # 结合10亿的钱包大小，无论他们使用自定义的stake_amount作为钱包大小的一小部分
        # 还是将自定义stake_amount固定为某个值，都应该能够进行交易
        logger.info("将stake_amount固定为10k")
        config["stake_amount"] = 10000

        # 强制缓存为'none'，如果不是则将其改为'none'
        # （因为默认值是'day'）
        if config.get("backtest_cache") is None:
            config["backtest_cache"] = "none"
        elif config["backtest_cache"] != "none":
            logger.info(
                f"检测到backtest_cache = {config['backtest_cache']}。"
                f"在前瞻分析中，它被强制设置为'none'。"
                f"已将其更改为'none'"
            )
            config["backtest_cache"] = "none"
        return config

    @staticmethod
    def initialize_single_lookahead_analysis(config: Config, strategy_obj: dict[str, Any]):
        logger.info(f"{Path(strategy_obj['location']).name}的偏差测试已开始。")
        start = time.perf_counter()
        current_instance = LookaheadAnalysis(config, strategy_obj)
        current_instance.start()
        elapsed = time.perf_counter() - start
        logger.info(
            f"通过回测检查{Path(strategy_obj['location']).name}的前瞻偏差 "
            f"耗时{elapsed:.0f}秒。"
        )
        return current_instance

    @staticmethod
    def start(config: Config):
        config = LookaheadAnalysisSubFunctions.calculate_config_overrides(config)

        strategy_objs = StrategyResolver.search_all_objects(
            config, enum_failed=False, recursive=config.get("recursive_strategy_search", False)
        )

        lookaheadAnalysis_instances = []

        # 将--strategy和--strategy-list统一到一个列表中
        if not (strategy_list := config.get("strategy_list", [])):
            if config.get("strategy") is None:
                raise OperationalException(
                    "未指定策略。请通过--strategy或--strategy-list指定策略"
                )
            strategy_list = [config["strategy"]]

        # 检查策略是否可以正确加载，只检查可以加载的策略
        for strat in strategy_list:
            for strategy_obj in strategy_objs:
                if strategy_obj["name"] == strat and strategy_obj not in strategy_list:
                    lookaheadAnalysis_instances.append(
                        LookaheadAnalysisSubFunctions.initialize_single_lookahead_analysis(
                            config, strategy_obj
                        )
                    )
                    break

        # 报告结果
        if lookaheadAnalysis_instances:
            caption: str | None = None
            if any(
                [
                    any(
                        [
                            indicator.startswith("&")
                            for indicator in inst.current_analysis.false_indicators
                        ]
                    )
                    for inst in lookaheadAnalysis_instances
                ]
            ):
                caption = (
                    "在'有偏差的指标'中，任何在set_freqai_targets()中使用的指标都可以忽略。"
                )
            LookaheadAnalysisSubFunctions.text_table_lookahead_analysis_instances(
                config, lookaheadAnalysis_instances, caption=caption
            )
            if config.get("lookahead_analysis_exportfilename") is not None:
                LookaheadAnalysisSubFunctions.export_to_csv(config, lookaheadAnalysis_instances)
        else:
            logger.error(
                "既没有通过--strategy也没有通过--strategy-list指定策略 "
                "或者没有指定时间范围。"
            )