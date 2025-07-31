import sys


def asyncio_setup() -> None:  # pragma: no cover
    # 为Windows系统设置事件循环

    if sys.platform == "win32":
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())