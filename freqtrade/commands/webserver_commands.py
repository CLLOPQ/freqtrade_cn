from typing import Any

from freqtrade.enums import RunMode


def start_webserver(args: dict[str, Any]) -> None:
    """
    Web服务器服务器模式的主要入口点
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.rpc.api_server import ApiServer

    # 初始化配置

    config = setup_utils_configuration(args, RunMode.WEBSERVER)
    ApiServer(config, standalone=True)