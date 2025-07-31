import logging
from logging.handlers import QueueHandler
from multiprocessing import Queue, current_process
from queue import Empty


logger = logging.getLogger(__name__)


def logging_mp_setup(log_queue: Queue, verbosity: int):
    """
    在子进程中设置日志。
    必须在子进程中进行日志记录之前调用。
    log_queue 必须通过继承传递给子进程
        这本质上意味着 log_queue 必须是一个全局变量，在初始化 Parallel 的同一个文件中创建。
    """
    current_proc = current_process().name
    if current_proc != "MainProcess":
        h = QueueHandler(log_queue)
        root = logging.getLogger()
        root.setLevel(verbosity)
        root.addHandler(h)


def logging_mp_handle(q: Queue):
    """
    处理来自子进程的日志。
    必须在父进程中调用，以处理来自子进程的日志消息。
    """

    try:
        while True:
            record = q.get(block=False)
            if record is None:
                break
            logger.handle(record)

    except Empty:
        pass