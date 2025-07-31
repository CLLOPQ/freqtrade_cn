import asyncio
import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.exceptions import HTTPException

from freqtrade.configuration import remove_exchange_credentials
from freqtrade.configuration.config_validation import validate_config_consistency
from freqtrade.constants import Config
from freqtrade.data.btanalysis import (
    delete_backtest_result,
    get_backtest_market_change,
    get_backtest_result,
    get_backtest_resultlist,
    load_and_merge_backtest_result,
    update_backtest_metadata,
)
from freqtrade.enums import BacktestState
from freqtrade.exceptions import ConfigurationError, DependencyException, OperationalException
from freqtrade.ft_types import get_BacktestResultType_default
from freqtrade.misc import deep_merge_dicts, is_file_in_dir
from freqtrade.rpc.api_server.api_schemas import (
    BacktestHistoryEntry,
    BacktestMarketChange,
    BacktestMetadataUpdate,
    BacktestRequest,
    BacktestResponse,
)
from freqtrade.rpc.api_server.deps import get_config
from freqtrade.rpc.api_server.webserver_bgwork import ApiBG
from freqtrade.rpc.rpc import RPCException


logger = logging.getLogger(__name__)

# 私有API，受身份验证和webserver_mode依赖保护
router = APIRouter()


def __run_backtest_bg(btconfig: Config):
    from freqtrade.data.metrics import combined_dataframes_with_rel_mean
    from freqtrade.optimize.optimize_reports import generate_backtest_stats, store_backtest_results
    from freqtrade.resolvers import StrategyResolver

    asyncio.set_event_loop(asyncio.new_event_loop())
    try:
        # 重新加载策略
        lastconfig = ApiBG.bt["last_config"]
        strat = StrategyResolver.load_strategy(btconfig)
        validate_config_consistency(btconfig)

        if (
            not ApiBG.bt["bt"]
            or lastconfig.get("timeframe") != strat.timeframe
            or lastconfig.get("timeframe_detail") != btconfig.get("timeframe_detail")
            or lastconfig.get("timerange") != btconfig["timerange"]
        ):
            from freqtrade.optimize.backtesting import Backtesting

            ApiBG.bt["bt"] = Backtesting(btconfig)
        else:
            ApiBG.bt["bt"].config = btconfig
            ApiBG.bt["bt"].init_backtest()
        # 仅在时间范围变化时重新加载数据
        if (
            not ApiBG.bt["data"]
            or not ApiBG.bt["timerange"]
            or lastconfig.get("timeframe") != strat.timeframe
            or lastconfig.get("timerange") != btconfig["timerange"]
        ):
            ApiBG.bt["data"], ApiBG.bt["timerange"] = ApiBG.bt["bt"].load_bt_data()

        lastconfig["timerange"] = btconfig["timerange"]
        lastconfig["timeframe"] = strat.timeframe
        lastconfig["enable_protections"] = btconfig.get("enable_protections")
        lastconfig["dry_run_wallet"] = btconfig.get("dry_run_wallet")

        ApiBG.bt["bt"].enable_protections = btconfig.get("enable_protections", False)
        ApiBG.bt["bt"].strategylist = [strat]
        ApiBG.bt["bt"].results = get_BacktestResultType_default()
        ApiBG.bt["bt"].load_prior_backtest()

        ApiBG.bt["bt"].abort = False
        strategy_name = strat.get_strategy_name()
        if ApiBG.bt["bt"].results and strategy_name in ApiBG.bt["bt"].results["strategy"]:
            # 如果之前的结果哈希匹配 - 复用该结果并跳过回测
            logger.info(f"正在为{strategy_name}复用之前回测的结果")
        else:
            min_date, max_date = ApiBG.bt["bt"].backtest_one_strategy(
                strat, ApiBG.bt["data"], ApiBG.bt["timerange"]
            )

            ApiBG.bt["bt"].results = generate_backtest_stats(
                ApiBG.bt["data"],
                ApiBG.bt["bt"].all_bt_content,
                min_date=min_date,
                max_date=max_date,
            )

            if btconfig.get("export", "none") == "trades":
                combined_res = combined_dataframes_with_rel_mean(
                    ApiBG.bt["data"], min_date, max_date
                )
                fn = store_backtest_results(
                    btconfig,
                    ApiBG.bt["bt"].results,
                    datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    market_change_data=combined_res,
                    strategy_files={
                        s.get_strategy_name(): s.__file__ for s in ApiBG.bt["bt"].strategylist
                    },
                )
                ApiBG.bt["bt"].results["metadata"][strategy_name]["filename"] = str(fn.stem)
                ApiBG.bt["bt"].results["metadata"][strategy_name]["strategy"] = strategy_name

        logger.info("回测完成。")

    except ConfigurationError as e:
        logger.error(f"回测遇到配置错误：{e}")

    except (Exception, OperationalException, DependencyException) as e:
        logger.exception(f"回测发生错误：{e}")
        ApiBG.bt["bt_error"] = str(e)
    finally:
        ApiBG.bgtask_running = False


@router.post("/backtest", response_model=BacktestResponse, tags=["webserver", "backtest"])
async def api_start_backtest(
    bt_settings: BacktestRequest, background_tasks: BackgroundTasks, config=Depends(get_config)
):
    ApiBG.bt["bt_error"] = None
    """如果尚未开始回测，则启动回测"""
    if ApiBG.bgtask_running:
        raise RPCException("Bot后台任务已在运行")

    if ":" in bt_settings.strategy:
        raise HTTPException(status_code=500, detail="不允许使用base64编码的策略。")

    btconfig = deepcopy(config)
    remove_exchange_credentials(btconfig["exchange"], True)
    settings = dict(bt_settings)
    if settings.get("freqai", None) is not None:
        settings["freqai"] = dict(settings["freqai"])
    # Pydantic模型将包含所有键，但未提供的键为None

    btconfig = deep_merge_dicts(settings, btconfig, allow_null_overrides=False)
    try:
        btconfig["stake_amount"] = float(btconfig["stake_amount"])
    except ValueError:
        pass

    # 为回测强制启用干运行
    btconfig["dry_run"] = True

    # 启动回测
    # 初始化回测对象

    background_tasks.add_task(__run_backtest_bg, btconfig=btconfig)
    ApiBG.bgtask_running = True

    return {
        "status": "running",
        "running": True,
        "progress": 0,
        "step": str(BacktestState.STARTUP),
        "status_msg": "回测已启动",
    }


@router.get("/backtest", response_model=BacktestResponse, tags=["webserver", "backtest"])
def api_get_backtest():
    """
    获取回测结果。
    在回测完成后返回结果。
    """
    from freqtrade.persistence import LocalTrade

    if ApiBG.bgtask_running:
        return {
            "status": "running",
            "running": True,
            "step": (
                ApiBG.bt["bt"].progress.action if ApiBG.bt["bt"] else str(BacktestState.STARTUP)
            ),
            "progress": ApiBG.bt["bt"].progress.progress if ApiBG.bt["bt"] else 0,
            "trade_count": len(LocalTrade.bt_trades),
            "status_msg": "回测正在运行",
        }

    if not ApiBG.bt["bt"]:
        return {
            "status": "not_started",
            "running": False,
            "step": "",
            "progress": 0,
            "status_msg": "回测尚未执行",
        }
    if ApiBG.bt["bt_error"]:
        return {
            "status": "error",
            "running": False,
            "step": "",
            "progress": 0,
            "status_msg": f"回测失败：{ApiBG.bt['bt_error']}",
        }

    return {
        "status": "ended",
        "running": False,
        "status_msg": "回测已结束",
        "step": "finished",
        "progress": 1,
        "backtest_result": ApiBG.bt["bt"].results,
    }


@router.delete("/backtest", response_model=BacktestResponse, tags=["webserver", "backtest"])
def api_delete_backtest():
    """重置回测"""
    if ApiBG.bgtask_running:
        return {
            "status": "running",
            "running": True,
            "step": "",
            "progress": 0,
            "status_msg": "回测正在运行",
        }
    if ApiBG.bt["bt"]:
        ApiBG.bt["bt"].cleanup()
        del ApiBG.bt["bt"]
        ApiBG.bt["bt"] = None
        del ApiBG.bt["data"]
        ApiBG.bt["data"] = None
        logger.info("回测已重置")
    return {
        "status": "reset",
        "running": False,
        "step": "",
        "progress": 0,
        "status_msg": "回测已重置",
    }


@router.get("/backtest/abort", response_model=BacktestResponse, tags=["webserver", "backtest"])
def api_backtest_abort():
    if not ApiBG.bgtask_running:
        return {
            "status": "not_running",
            "running": False,
            "step": "",
            "progress": 0,
            "status_msg": "回测已结束",
        }
    ApiBG.bt["bt"].abort = True
    return {
        "status": "stopping",
        "running": False,
        "step": "",
        "progress": 0,
        "status_msg": "回测已结束",
    }


@router.get(
    "/backtest/history", response_model=list[BacktestHistoryEntry], tags=["webserver", "backtest"]
)
def api_backtest_history(config=Depends(get_config)):
    # 获取回测结果历史，从元数据文件读取
    return get_backtest_resultlist(config["user_data_dir"] / "backtest_results")


@router.get(
    "/backtest/history/result", response_model=BacktestResponse, tags=["webserver", "backtest"]
)
def api_backtest_history_result(filename: str, strategy: str, config=Depends(get_config)):
    # 获取回测结果历史，从元数据文件读取
    bt_results_base: Path = config["user_data_dir"] / "backtest_results"
    for ext in [".zip", ".json"]:
        fn = (bt_results_base / filename).with_suffix(ext)
        if is_file_in_dir(fn, bt_results_base):
            break
    else:
        raise HTTPException(status_code=404, detail="文件未找到。")

    results: dict[str, Any] = {
        "metadata": {},
        "strategy": {},
        "strategy_comparison": [],
    }
    load_and_merge_backtest_result(strategy, fn, results)
    return {
        "status": "ended",
        "running": False,
        "step": "",
        "progress": 1,
        "status_msg": "历史结果",
        "backtest_result": results,
    }


@router.delete(
    "/backtest/history/{file}",
    response_model=list[BacktestHistoryEntry],
    tags=["webserver", "backtest"],
)
def api_delete_backtest_history_entry(file: str, config=Depends(get_config)):
    # 获取回测结果历史，从元数据文件读取
    bt_results_base: Path = config["user_data_dir"] / "backtest_results"
    for ext in [".zip", ".json"]:
        file_abs = (bt_results_base / file).with_suffix(ext)
        # 确保文件位于回测结果目录中
        if is_file_in_dir(file_abs, bt_results_base):
            break
    else:
        raise HTTPException(status_code=404, detail="文件未找到。")

    delete_backtest_result(file_abs)
    return get_backtest_resultlist(config["user_data_dir"] / "backtest_results")


@router.patch(
    "/backtest/history/{file}",
    response_model=list[BacktestHistoryEntry],
    tags=["webserver", "backtest"],
)
def api_update_backtest_history_entry(
    file: str, body: BacktestMetadataUpdate, config=Depends(get_config)
):
    # 获取回测结果历史，从元数据文件读取
    bt_results_base: Path = config["user_data_dir"] / "backtest_results"
    for ext in [".zip", ".json"]:
        file_abs = (bt_results_base / file).with_suffix(ext)
        # 确保文件位于回测结果目录中
        if is_file_in_dir(file_abs, bt_results_base):
            break
    else:
        raise HTTPException(status_code=404, detail="文件未找到。")

    content = {"notes": body.notes}
    try:
        update_backtest_metadata(file_abs, body.strategy, content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return get_backtest_result(file_abs)


@router.get(
    "/backtest/history/{file}/market_change",
    response_model=BacktestMarketChange,
    tags=["webserver", "backtest"],
)
def api_get_backtest_market_change(file: str, config=Depends(get_config)):
    bt_results_base: Path = config["user_data_dir"] / "backtest_results"
    for fn in (
        Path(file).with_suffix(".zip"),
        Path(f"{file}_market_change").with_suffix(".feather"),
    ):
        file_abs = bt_results_base / fn
        # 确保文件位于回测结果目录中
        if is_file_in_dir(file_abs, bt_results_base):
            break
    else:
        raise HTTPException(status_code=404, detail="文件未找到。")

    df = get_backtest_market_change(file_abs)

    return {
        "columns": df.columns.tolist(),
        "data": df.values.tolist(),
        "length": len(df),
    }