import os


def running_in_docker() -> bool:
    """
    检查是否在docker容器中运行
    """
    return os.environ.get("FT_APP_ENV") == "docker"