import asyncio
import time


class MessageStream:
    """
    一个消息流，供消费者订阅，
    供生产者发布。
    """

    def __init__(self):
        self._loop = asyncio.get_running_loop()
        self._waiter = self._loop.create_future()

    def publish(self, message):
        """
        向此消息流发布消息

        :param message: 要发布的消息
        """
        waiter, self._waiter = self._waiter, self._loop.create_future()
        waiter.set_result((message, time.time(), self._waiter))

    async def __aiter__(self):
        """
        迭代消息流中的消息
        """
        waiter = self._waiter
        while True:
            # 保护future不被等待它的任务取消
            message, ts, waiter = await asyncio.shield(waiter)
            yield message, ts