import logging
import signal
from typing import Any


logger = logging.getLogger(__name__)


def start_trading(args: dict[str, Any]) -> int:
    """
    交易模式的主要入口点
    """
    # 在这里导入以避免在不使用工作模块时加载它
    from freqtrade.worker import Worker

    def term_handler(signum, frame):
        # 触发KeyboardInterrupt - 这样我们可以像处理Ctrl+C一样处理它
        raise KeyboardInterrupt()

    # 创建并运行工作器
    worker = None
    try:
        signal.signal(signal.SIGTERM, term_handler)
        worker = Worker(args)
        worker.run()
    finally:
        if worker:
            logger.info("找到工作器... 调用退出")
            worker.exit()
    return 0