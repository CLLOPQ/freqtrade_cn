import logging
import time
from pathlib import Path
from typing import Any

from freqtrade.constants import Config
from freqtrade.exceptions import OperationalException
from freqtrade.optimize.analysis.recursive import RecursiveAnalysis
from freqtrade.resolvers import StrategyResolver
from freqtrade.util import print_rich_table


logger = logging.getLogger(__name__)


class RecursiveAnalysisSubFunctions:
    @staticmethod
    def text_table_recursive_analysis_instances(recursive_instances: list[RecursiveAnalysis]):
        startups = recursive_instances[0]._startup_candle
        strat_scc = recursive_instances[0]._strat_scc
        headers = ["指标"]
        for candle in startups:
            if candle == strat_scc:
                headers.append(f"{candle} (来自策略)")
            else:
                headers.append(str(candle))

        data = []
        for inst in recursive_instances:
            if len(inst.dict_recursive) > 0:
                for indicator, values in inst.dict_recursive.items():
                    temp_data = [indicator]
                    for candle in startups:
                        temp_data.append(values.get(int(candle), "-"))
                    data.append(temp_data)

        if len(data) > 0:
            print_rich_table(data, headers, summary="递归分析")

            return data

        return data

    @staticmethod
    def calculate_config_overrides(config: Config):
        if "timerange" not in config:
            # 这里强制要求设置时间范围
            raise OperationalException(
                "请设置时间范围。"
                "递归分析5000根K线的时间范围就足够了。"
            )

        if config.get("backtest_cache") is None:
            config["backtest_cache"] = "none"
        elif config["backtest_cache"] != "none":
            logger.info(
                f"检测到backtest_cache = {config['backtest_cache']}。"
                f"在递归分析中，它被强制设置为'none'。"
                f"已将其更改为'none'"
            )
            config["backtest_cache"] = "none"
        return config

    @staticmethod
    def initialize_single_recursive_analysis(config: Config, strategy_obj: dict[str, Any]):
        logger.info(f"{Path(strategy_obj['location']).name}的递归测试已开始。")
        start = time.perf_counter()
        current_instance = RecursiveAnalysis(config, strategy_obj)
        current_instance.start()
        elapsed = time.perf_counter() - start
        logger.info(
            f"检查{Path(strategy_obj['location']).name}指标的递归和仅指标前瞻偏差 "
            f"耗时{elapsed:.0f}秒。"
        )
        return current_instance

    @staticmethod
    def start(config: Config):
        config = RecursiveAnalysisSubFunctions.calculate_config_overrides(config)

        strategy_objs = StrategyResolver.search_all_objects(
            config, enum_failed=False, recursive=config.get("recursive_strategy_search", False)
        )

        RecursiveAnalysis_instances = []

        # 将--strategy和--strategy-list统一到一个列表中
        if not (strategy_list := config.get("strategy_list", [])):
            if config.get("strategy") is None:
                raise OperationalException(
                    "未指定策略。请通过--strategy指定策略"
                )
            strategy_list = [config["strategy"]]

        # 检查策略是否可以正确加载，只检查可以加载的策略
        for strat in strategy_list:
            for strategy_obj in strategy_objs:
                if strategy_obj["name"] == strat and strategy_obj not in strategy_list:
                    RecursiveAnalysis_instances.append(
                        RecursiveAnalysisSubFunctions.initialize_single_recursive_analysis(
                            config, strategy_obj
                        )
                    )
                    break

        # 报告结果
        if RecursiveAnalysis_instances:
            RecursiveAnalysisSubFunctions.text_table_recursive_analysis_instances(
                RecursiveAnalysis_instances
            )
        else:
            logger.error(
                "没有通过--strategy指定策略或者没有指定时间范围。"
            )