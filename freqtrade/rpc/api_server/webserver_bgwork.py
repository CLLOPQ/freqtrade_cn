from typing import Any, Literal
from uuid import uuid4

from typing_extensions import NotRequired, TypedDict

from freqtrade.exchange.exchange import Exchange


class ProgressTask(TypedDict):
    progress: float
    total: float
    description: str


class JobsContainer(TypedDict):
    category: Literal["pairlist", "download_data"]
    is_running: bool
    status: str
    progress: float | None
    progress_tasks: NotRequired[dict[str, ProgressTask]]
    result: Any
    error: str | None


class ApiBG:
    # 回测类型：回测
    bt: dict[str, Any] = {
        "bt": None,
        "data": None,
        "timerange": None,
        "last_config": {},
        "bt_error": None,
    }
    bgtask_running: bool = False
    # 交易所 - 仅在Web服务器模式下可用。
    exchanges: dict[str, Exchange] = {}

    # 通用后台任务

    # TODO: 将此更改为TTL缓存
    jobs: dict[str, JobsContainer] = {}
    # 交易对列表评估相关内容
    pairlist_running: bool = False
    download_data_running: bool = False

    @staticmethod
    def get_job_id() -> str:
        return str(uuid4())