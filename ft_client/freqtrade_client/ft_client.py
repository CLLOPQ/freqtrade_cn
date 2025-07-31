import argparse
import inspect
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import rapidjson

from freqtrade_client import __version__
from freqtrade_client.ft_rest_client import FtRestClient


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ft_rest_client")


def add_arguments(args: Any = None):
    parser = argparse.ArgumentParser(
        prog="freqtrade-client",
        description="Freqtrade REST API的客户端",
    )
    parser.add_argument(
        "command", help="定义要执行的命令的位置参数。", nargs="?"
    )
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--show",
        help="显示此客户端可用的方法",
        dest="show",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-c",
        "--config",
        help="指定配置文件（默认：%(default)s）。",
        dest="config",
        type=str,
        metavar="路径",
        default="config.json",
    )

    parser.add_argument(
        "command_arguments",
        help="用于[command]参数的位置参数",
        nargs="*",
        default=[],
    )

    pargs = parser.parse_args(args)
    return vars(pargs)


def load_config(configfile):
    file = Path(configfile)
    if file.is_file():
        with file.open("r") as f:
            config = rapidjson.load(
                f, parse_mode=rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS
            )
        return config
    else:
        logger.warning(f"无法加载配置文件 {file}。")
        sys.exit(1)


def print_commands():
    # 使用命令的文档字符串打印不同命令的动态帮助
    client = FtRestClient(None)
    print("可用命令：\n")
    for x, _ in inspect.getmembers(client):
        if not x.startswith("_"):
            # 移除返回值描述，只保留命令说明
            doc = re.sub(":return:.*", "", getattr(client, x).__doc__, flags=re.MULTILINE).rstrip()
            print(f"{x}\n\t{doc}\n")


def main_exec(parsed: dict[str, Any]):
    if parsed.get("show"):
        print_commands()
        sys.exit()

    config = load_config(parsed["config"])
    url = config.get("api_server", {}).get("listen_ip_address", "127.0.0.1")
    port = config.get("api_server", {}).get("listen_port", "8080")
    username = config.get("api_server", {}).get("username")
    password = config.get("api_server", {}).get("password")

    server_url = f"http://{url}:{port}"
    client = FtRestClient(server_url, username, password)

    # 获取所有可用命令（排除私有方法）
    methods = [x for x, y in inspect.getmembers(client) if not x.startswith("_")]
    command = parsed["command"]
    if command not in methods:
        logger.error(f"命令 {command} 未定义")
        print_commands()
        return

    # 将带有=的参数拆分为键/值对
    kwargs = {x.split("=")[0]: x.split("=")[1] for x in parsed["command_arguments"] if "=" in x}
    args = [x for x in parsed["command_arguments"] if "=" not in x]
    try:
        result = getattr(client, command)(*args, **kwargs)
        print(json.dumps(result))
    except TypeError as e:
        logger.error(f"执行命令 {command} 时出错：{e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"执行命令 {command} 时发生致命错误：{e}")
        sys.exit(1)


def main():
    """
    客户端的主入口点
    """
    args = add_arguments()
    main_exec(args)