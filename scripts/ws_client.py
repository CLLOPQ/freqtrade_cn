#!/usr/bin/env python3
"""
用于测试/调试Freqtrade机器人消息WebSocket的简单命令行客户端

不应从freqtrade导入任何内容，
因此它可以用作独立脚本。
"""

import argparse
import asyncio
import logging
import socket
import sys
import time
from pathlib import Path

import orjson
import pandas
import rapidjson
import websockets


logger = logging.getLogger("WebSocketClient")


# ---------------------------------------------------------------------------


def setup_logging(filename: str):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(filename), logging.StreamHandler()],
    )


def parse_args():
    parser = argparse.ArgumentParser()
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
        "-l",
        "--logfile",
        help="日志文件名称。",
        dest="logfile",
        type=str,
        default="ws_client.log",
    )

    args = parser.parse_args()
    return vars(args)


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


def readable_timedelta(delta):
    """
    将毫秒级时间差转换为可读格式

    :param delta: 两个时间戳之间的毫秒级差值
    :returns: 可读的时间差字符串
    """
    seconds, milliseconds = divmod(delta, 1000)
    minutes, seconds = divmod(seconds, 60)

    return f"{int(minutes)}:{int(seconds)}.{int(milliseconds)}"


# ----------------------------------------------------------------------------


def json_serialize(message):
    """
    使用orjson将消息序列化为JSON
    :param message: 要序列化的消息
    """
    return str(orjson.dumps(message), "utf-8")


def json_deserialize(message):
    """
    将JSON反序列化为字典
    :param message: 要反序列化的消息
    """

    def json_to_dataframe(data: str) -> pandas.DataFrame:
        dataframe = pandas.read_json(data, orient="split")
        if "date" in dataframe.columns:
            dataframe["date"] = pandas.to_datetime(dataframe["date"], unit="ms", utc=True)

        return dataframe

    def _json_object_hook(z):
        if z.get("__type__") == "dataframe":
            return json_to_dataframe(z.get("__value__"))
        return z

    return rapidjson.loads(message, object_hook=_json_object_hook)


# ---------------------------------------------------------------------------


class ClientProtocol:
    logger = logging.getLogger("WebSocketClient.Protocol")
    _MESSAGE_COUNT = 0
    _LAST_RECEIVED_AT = 0  # 最近接收消息的时间戳（毫秒）

    async def on_connect(self, websocket):
        # 连接后发送初始请求
        initial_requests = [
            {
                "type": "subscribe",  # 订阅请求应始终放在首位
                "data": ["analyzed_df", "whitelist"],  # 我们需要的消息类型
            },
            {
                "type": "whitelist",
                "data": None,
            },
            {"type": "analyzed_df", "data": {"limit": 1500}},
        ]

        for request in initial_requests:
            await websocket.send(json_serialize(request))

    async def on_message(self, websocket, name, message):
        deserialized = json_deserialize(message)

        message_size = sys.getsizeof(message)
        message_type = deserialized.get("type")
        message_data = deserialized.get("data")

        self.logger.info(
            f"收到类型为 {message_type} 的消息 [{message_size} 字节] @ [{name}]"
        )

        time_difference = self._calculate_time_difference()

        if self._MESSAGE_COUNT > 0:
            self.logger.info(f"距上次消息的时间: {time_difference}")

        message_handler = getattr(self, f"_handle_{message_type}", None) or self._handle_default
        await message_handler(name, message_type, message_data)

        self._MESSAGE_COUNT += 1
        self.logger.info(f"共收到 [{self._MESSAGE_COUNT}] 条消息..")
        self.logger.info("-" * 80)

    def _calculate_time_difference(self):
        old_last_received_at = self._LAST_RECEIVED_AT
        self._LAST_RECEIVED_AT = time.time() * 1e3
        time_delta = self._LAST_RECEIVED_AT - old_last_received_at

        return readable_timedelta(time_delta)

    async def _handle_whitelist(self, name, msgtype, data):
        self.logger.info(data)

    async def _handle_analyzed_df(self, name, msgtype, data):
        key, la, df = data["key"], data["la"], data["df"]

        if not df.empty:
            columns = ", ".join([str(column) for column in df.columns])

            self.logger.info(key)
            self.logger.info(f"最后分析时间: {la}")
            self.logger.info(f"最新K线时间: {df.iloc[-1]['date']}")
            self.logger.info(f"数据框长度: {len(df)}")
            self.logger.info(f"数据框列: {columns}")
        else:
            self.logger.info("空数据框")

    async def _handle_default(self, name, msgtype, data):
        self.logger.info(f"收到未知类型 {msgtype} 的消息...")
        self.logger.info(data)


async def create_client(
    host,
    port,
    token,
    scheme="ws",
    name="default",
    protocol=None,
    sleep_time=10,
    ping_timeout=10,
    wait_timeout=30,** kwargs,
):
    """
    创建WebSocket客户端并监听消息
    :param host: 主机地址
    :param port: 端口号
    :param token: WebSocket认证令牌
    :param scheme: 大多数连接使用`ws`，SSL连接使用`wss`
    :param name: 生产者名称
    :param **kwargs: 传递给websockets.connect的额外参数
    """
    if not protocol:
        protocol = ClientProtocol()

    while 1:
        try:
            websocket_url = f"{scheme}://{host}:{port}/api/v1/message/ws?token={token}"
            logger.info(f"尝试连接到 {name} @ {host}:{port}")

            async with websockets.connect(websocket_url,** kwargs) as ws:
                logger.info("连接成功...")
                await protocol.on_connect(ws)

                # 开始监听消息
                while 1:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=wait_timeout)

                        await protocol.on_message(ws, name, message)

                    except (asyncio.TimeoutError, websockets.exceptions.WebSocketException):
                        # 尝试发送ping
                        try:
                            pong = await ws.ping()
                            latency = await asyncio.wait_for(pong, timeout=ping_timeout) * 1000

                            logger.info(f"连接仍然活跃，延迟: {latency}ms")

                            continue

                        except asyncio.TimeoutError:
                            logger.error(f"Ping超时，{sleep_time}秒后重试")
                            await asyncio.sleep(sleep_time)

                            break

        except (
            socket.gaierror,
            ConnectionRefusedError,
            websockets.exceptions.InvalidHandshake,
        ) as e:
            logger.error(f"连接被拒绝 - {e} {sleep_time}秒后重试")
            await asyncio.sleep(sleep_time)

            continue

        except (
            websockets.exceptions.ConnectionClosedError,
            websockets.exceptions.ConnectionClosedOK,
        ):
            logger.info("连接已关闭")
            # 继续尝试重新连接
            await asyncio.sleep(sleep_time)

            continue

        except Exception as e:
            # 发生意外错误，记录日志并尝试重新连接
            logger.error("发生意外错误:")
            logger.exception(e)

            await asyncio.sleep(sleep_time)
            continue


# ---------------------------------------------------------------------------


async def _main(args):
    setup_logging(args["logfile"])
    config = load_config(args["config"])

    emc_config = config.get("external_message_consumer", {})

    producers = emc_config.get("producers", [])
    producer = producers[0]

    wait_timeout = emc_config.get("wait_timeout", 30)
    ping_timeout = emc_config.get("ping_timeout", 10)
    sleep_time = emc_config.get("sleep_time", 10)
    message_size_limit = emc_config.get("message_size_limit", 8) << 20

    await create_client(
        producer["host"],
        producer["port"],
        producer["ws_token"],
        "wss" if producer.get("secure", False) else "ws",
        producer["name"],
        sleep_time=sleep_time,
        ping_timeout=ping_timeout,
        wait_timeout=wait_timeout,
        max_size=message_size_limit,
        ping_interval=None,
    )


def main():
    args = parse_args()
    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        logger.info("正在退出...")


if __name__ == "__main__":
    main()