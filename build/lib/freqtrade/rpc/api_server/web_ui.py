from pathlib import Path

from fastapi import APIRouter
from fastapi.exceptions import HTTPException
from starlette.responses import FileResponse


router_ui = APIRouter()


@router_ui.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(str(Path(__file__).parent / "ui/favicon.ico"))


@router_ui.get("/fallback_file.html", include_in_schema=False)
async def fallback():
    return FileResponse(str(Path(__file__).parent / "ui/fallback_file.html"))


@router_ui.get("/ui_version", include_in_schema=False)
async def ui_version():
    """
    获取UI版本信息。
    """
    from freqtrade.commands.deploy_ui import read_ui_version

    uibase = Path(__file__).parent / "ui/installed/"
    version = read_ui_version(uibase)

    return {
        "version": version if version else "未安装",
    }


@router_ui.get("/{rest_of_path:path}", include_in_schema=False)
async def index_html(rest_of_path: str):
    """
    模拟路径回退到index.html。
    """
    if rest_of_path.startswith("api") or rest_of_path.startswith("."):
        raise HTTPException(status_code=404, detail="未找到")
    uibase = Path(__file__).parent / "ui/installed/"
    filename = uibase / rest_of_path
    # 检查"relative_to"与安全相关。没有这个，可能会导致目录遍历攻击。
    # 为.js文件强制使用text/javascript - 规避有问题的系统配置
    media_type: str | None = None
    if filename.suffix == ".js":
        media_type = "application/javascript"
    if filename.is_file() and filename.is_relative_to(uibase):
        return FileResponse(str(filename), media_type=media_type)

    index_file = uibase / "index.html"
    if not index_file.is_file():
        return FileResponse(str(uibase.parent / "fallback_file.html"))
    # 如Vue Router文档所述，回退到index.html
    return FileResponse(str(index_file))