import logging
from pathlib import Path

import requests


logger = logging.getLogger(__name__)

# 请求超时时间
req_timeout = 30


def clean_ui_subdir(directory: Path):
    """清理UI子目录内容"""
    if directory.is_dir():
        logger.info("正在移除UI目录内容。")

        # 从叶子到根迭代内容
        for p in reversed(list(directory.glob("**/*"))):
            if p.name in (".gitkeep", "fallback_file.html"):
                continue
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                p.rmdir()


def read_ui_version(dest_folder: Path) -> str | None:
    """读取UI版本信息"""
    file = dest_folder / ".uiversion"
    if not file.is_file():
        return None

    with file.open("r") as f:
        return f.read()


def download_and_install_ui(dest_folder: Path, dl_url: str, version: str):
    """下载并安装UI"""
    from io import BytesIO
    from zipfile import ZipFile

    logger.info(f"正在下载 {dl_url}")
    resp = requests.get(dl_url, timeout=req_timeout).content
    dest_folder.mkdir(parents=True, exist_ok=True)
    with ZipFile(BytesIO(resp)) as zf:
        for fn in zf.filelist:
            with zf.open(fn) as x:
                destfile = dest_folder / fn.filename
                if fn.is_dir():
                    destfile.mkdir(exist_ok=True)
                else:
                    destfile.write_bytes(x.read())
    with (dest_folder / ".uiversion").open("w") as f:
        f.write(version)


def get_ui_download_url(version: str | None, prerelease: bool) -> tuple[str, str]:
    """获取UI下载URL和版本信息"""
    base_url = "https://api.github.com/repos/freqtrade/frequi/"
    # 获取基础UI仓库路径

    resp = requests.get(f"{base_url}releases", timeout=req_timeout)
    resp.raise_for_status()
    r = resp.json()

    if version:
        tmp = [x for x in r if x["name"] == version]
    else:
        tmp = [x for x in r if prerelease or not x.get("prerelease")]

    if tmp:
        # 确保我们有最新版本
        if version is None:
            tmp.sort(key=lambda x: x["created_at"], reverse=True)
        latest_version = tmp[0]["name"]
        assets = tmp[0].get("assets", [])
    else:
        raise ValueError("未找到UI版本。")

    dl_url = ""
    if assets and len(assets) > 0:
        dl_url = assets[0]["browser_download_url"]

    # 未找到URL - 尝试资产URL
    if not dl_url:
        assets = r[0]["assets_url"]
        resp = requests.get(assets, timeout=req_timeout)
        r = resp.json()
        dl_url = r[0]["browser_download_url"]

    return dl_url, latest_version