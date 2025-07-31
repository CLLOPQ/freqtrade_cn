import logging
from copy import deepcopy

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.exceptions import HTTPException

from freqtrade.constants import Config
from freqtrade.exceptions import OperationalException
from freqtrade.persistence import FtNoDBContext
from freqtrade.rpc.api_server.api_pairlists import handleExchangePayload
from freqtrade.rpc.api_server.api_schemas import BgJobStarted, DownloadDataPayload
from freqtrade.rpc.api_server.deps import get_config, get_exchange
from freqtrade.rpc.api_server.webserver_bgwork import ApiBG
from freqtrade.util.progress_tracker import get_progress_tracker


logger = logging.getLogger(__name__)

# 私有API，受认证和webserver_mode依赖保护
router = APIRouter(tags=["下载数据", "Web服务器"])


def __run_download(job_id: str, config_loc: Config):
    try:
        ApiBG.jobs[job_id]["is_running"] = True
        from freqtrade.data.history.history_utils import download_data

        with FtNoDBContext():
            exchange = get_exchange(config_loc)

            def ft_callback(task) -> None:
                ApiBG.jobs[job_id]["progress_tasks"][str(task.id)] = {
                    "progress": task.completed,
                    "total": task.total,
                    "description": task.description,
                }

            pt = get_progress_tracker(ft_callback=ft_callback)

            download_data(config_loc, exchange, progress_tracker=pt)
            ApiBG.jobs[job_id]["status"] = "成功"
    except (OperationalException, Exception) as e:
        logger.exception(e)
        ApiBG.jobs[job_id]["error"] = str(e)
        ApiBG.jobs[job_id]["status"] = "失败"
    finally:
        ApiBG.jobs[job_id]["is_running"] = False
        ApiBG.download_data_running = False


@router.post("/下载数据", response_model=BgJobStarted)
def pairlists_evaluate(
    payload: DownloadDataPayload, background_tasks: BackgroundTasks, config=Depends(get_config)
):
    if ApiBG.download_data_running:
        raise HTTPException(status_code=400, detail="数据下载已在运行中。")
    config_loc = deepcopy(config)
    config_loc["stake_currency"] = ""
    config_loc["pairs"] = payload.pairs
    config_loc["timerange"] = payload.timerange
    config_loc["days"] = payload.days
    config_loc["timeframes"] = payload.timeframes
    config_loc["erase"] = payload.erase
    config_loc["download_trades"] = payload.download_trades

    handleExchangePayload(payload, config_loc)

    job_id = ApiBG.get_job_id()

    ApiBG.jobs[job_id] = {
        "category": "download_data",
        "status": "待处理",
        "progress": None,
        "progress_tasks": {},
        "is_running": False,
        "result": {},
        "error": None,
    }
    background_tasks.add_task(__run_download, job_id, config_loc)
    ApiBG.download_data_running = True

    return {
        "status": "数据下载已在后台启动。",
        "job_id": job_id,
    }