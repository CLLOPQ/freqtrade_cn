import logging

from fastapi import APIRouter
from fastapi.exceptions import HTTPException

from freqtrade.rpc.api_server.api_schemas import BackgroundTaskStatus
from freqtrade.rpc.api_server.webserver_bgwork import ApiBG


logger = logging.getLogger(__name__)

# 私有API，受认证和webserver_mode依赖保护
router = APIRouter()


@router.get("/background", response_model=list[BackgroundTaskStatus], tags=["webserver"])
def background_job_list():
    return [
        {
            "job_id": jobid,
            "job_category": job["category"],
            "status": job["status"],
            "running": job["is_running"],
            "progress": job.get("progress"),
            "progress_tasks": job.get("progress_tasks"),
            "error": job.get("error", None),
        }
        for jobid, job in ApiBG.jobs.items()
    ]


@router.get("/background/{jobid}", response_model=BackgroundTaskStatus, tags=["webserver"])
def background_job(jobid: str):
    if not (job := ApiBG.jobs.get(jobid)):
        raise HTTPException(status_code=404, detail="未找到任务。")

    return {
        "job_id": jobid,
        "job_category": job["category"],
        "status": job["status"],
        "running": job["is_running"],
        "progress": job.get("progress"),
        "progress_tasks": job.get("progress_tasks"),
        "error": job.get("error", None),
    }