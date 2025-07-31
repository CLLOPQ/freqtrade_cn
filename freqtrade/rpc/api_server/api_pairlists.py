import logging
from copy import deepcopy

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.exceptions import HTTPException

from freqtrade.constants import Config
from freqtrade.enums import CandleType
from freqtrade.exceptions import OperationalException
from freqtrade.persistence import FtNoDBContext
from freqtrade.rpc.api_server.api_schemas import (
    BgJobStarted,
    ExchangeModePayloadMixin,
    PairListsPayload,
    PairListsResponse,
    WhitelistEvaluateResponse,
)
from freqtrade.rpc.api_server.deps import get_config, get_exchange
from freqtrade.rpc.api_server.webserver_bgwork import ApiBG


logger = logging.getLogger(__name__)

# 私有API，受认证和webserver_mode依赖保护
router = APIRouter()


@router.get(
    "/pairlists/available", response_model=PairListsResponse, tags=["pairlists", "webserver"]
)
def list_pairlists(config=Depends(get_config)):
    """
    获取所有可用的交易对列表生成器
    """
    from freqtrade.resolvers import PairListResolver

    pairlists = PairListResolver.search_all_objects(config, False)
    pairlists = sorted(pairlists, key=lambda x: x["name"])

    return {
        "pairlists": [
            {
                "name": x["name"],
                "is_pairlist_generator": x["class"].is_pairlist_generator,
                "params": x["class"].available_parameters(),
                "description": x["class"].description(),  # 交易对列表生成器的描述
            }
            for x in pairlists
        ]
    }


def __run_pairlist(job_id: str, config_loc: Config):
    """
    在后台执行交易对列表刷新操作
    """
    try:
        ApiBG.jobs[job_id]["is_running"] = True
        from freqtrade.plugins.pairlistmanager import PairListManager

        with FtNoDBContext():
            exchange = get_exchange(config_loc)
            pairlists = PairListManager(exchange, config_loc)
            pairlists.refresh_pairlist()
            ApiBG.jobs[job_id]["result"] = {
                "method": pairlists.name_list,
                "length": len(pairlists.whitelist),
                "whitelist": pairlists.whitelist,
            }
            ApiBG.jobs[job_id]["status"] = "success"
    except (OperationalException, Exception) as e:
        logger.exception(e)
        ApiBG.jobs[job_id]["error"] = str(e)
        ApiBG.jobs[job_id]["status"] = "failed"
    finally:
        ApiBG.jobs[job_id]["is_running"] = False
        ApiBG.pairlist_running = False


@router.post("/pairlists/evaluate", response_model=BgJobStarted, tags=["pairlists", "webserver"])
def pairlists_evaluate(
    payload: PairListsPayload, background_tasks: BackgroundTasks, config=Depends(get_config)
):
    """
    启动交易对列表评估作业
    """
    if ApiBG.pairlist_running:
        raise HTTPException(status_code=400, detail="交易对列表评估已在运行中。")

    config_loc = deepcopy(config)
    config_loc["stake_currency"] = payload.stake_currency
    config_loc["pairlists"] = payload.pairlists
    handleExchangePayload(payload, config_loc)
    # TODO: 覆盖黑名单？使其可选并回退到配置中的黑名单？
    # 结果取决于UI的实现方式。
    config_loc["exchange"]["pair_blacklist"] = payload.blacklist
    # 随机生成作业ID
    job_id = ApiBG.get_job_id()

    ApiBG.jobs[job_id] = {
        "category": "pairlist",
        "status": "pending",
        "progress": None,
        "is_running": False,
        "result": {},
        "error": None,
    }
    background_tasks.add_task(__run_pairlist, job_id, config_loc)
    ApiBG.pairlist_running = True

    return {
        "status": "交易对列表评估已在后台启动。",
        "job_id": job_id,
    }


def handleExchangePayload(payload: ExchangeModePayloadMixin, config_loc: Config):
    """
    处理交易所和交易模式的有效负载。使用有效负载值更新配置。
    """
    from freqtrade.configuration.directory_operations import create_datadir

    if payload.exchange:
        config_loc["exchange"]["name"] = payload.exchange
        config_loc.update({"datadir": create_datadir(config_loc, None)})
    if payload.trading_mode:
        config_loc["trading_mode"] = payload.trading_mode
        config_loc["candle_type_def"] = CandleType.get_default(
            config_loc.get("trading_mode", "spot") or "spot"
        )

    if payload.margin_mode:
        config_loc["margin_mode"] = payload.margin_mode


@router.get(
    "/pairlists/evaluate/{jobid}",
    response_model=WhitelistEvaluateResponse,
    tags=["pairlists", "webserver"],
)
def pairlists_evaluate_get(jobid: str):
    """
    获取交易对列表评估作业结果
    """
    if not (job := ApiBG.jobs.get(jobid)):
        raise HTTPException(status_code=404, detail="作业不存在。")

    if job["is_running"]:
        raise HTTPException(status_code=400, detail="作业尚未完成。")

    if error := job["error"]:
        return {
            "status": "failed",
            "error": error,
        }

    return {
        "status": "success",
        "result": job["result"],
    }