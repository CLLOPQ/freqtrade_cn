import threading
import time

import uvicorn


def asyncio_setup() -> None:  # pragma: no cover
    # 为Windows系统设置事件循环
    # 恢复uvicorn 0.15.0中所做的一项更改——该版本现在通过策略设置事件循环
    # TODO: 这个解决方法是否真的需要？
    import sys

    if sys.version_info >= (3, 8) and sys.platform == "win32":
        import asyncio
        import selectors

        selector = selectors.SelectSelector()
        loop = asyncio.SelectorEventLoop(selector)
        asyncio.set_event_loop(loop)


class UvicornServer(uvicorn.Server):
    """
    多线程服务器——如在https://github.com/encode/uvicorn/issues/742中所发现的

    基于此提交的更改移除了install_signal_handlers()重写：
        https://github.com/encode/uvicorn/commit/ce2ef45a9109df8eae038c0ec323eb63d644cbc6

    由于此检查，不能依赖asyncio.get_event_loop()来创建新的事件循环：
        https://github.com/python/cpython/blob/4d7f11e05731f67fd2c07ec2972c6cb9861d52be/Lib/asyncio/events.py#L638

    通过重写run()并在uvloop可用时强制创建新的事件循环来修复
    """

    def run(self, sockets=None):
        import asyncio

        """
        父类实现调用self.config.setup_event_loop()，但我们需要手动创建uvloop事件循环
        """
        try:
            import uvloop
        except ImportError:  # pragma: no cover
            asyncio_setup()
        else:
            asyncio.set_event_loop(uvloop.new_event_loop())
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 当在线程中运行时，我们还没有事件循环。
            loop = asyncio.new_event_loop()
        loop.run_until_complete(self.serve(sockets=sockets))

    def run_in_thread(self):
        self.thread = threading.Thread(target=self.run, name="FTUvicorn")
        self.thread.start()
        while not self.started:
            time.sleep(1e-3)

    def cleanup(self):
        self.should_exit = True
        self.thread.join()