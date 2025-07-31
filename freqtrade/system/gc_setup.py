import gc
import logging
import platform


logger = logging.getLogger(__name__)


def gc_set_threshold():
    """
    减少GC运行次数以提高性能（解释视频）
    https://www.youtube.com/watch?v=p4Sn6UcFTOU

    """
    if platform.python_implementation() == "CPython":
        # 分配次数、阈值1、阈值2 = gc.get_threshold()
        gc.set_threshold(50_000, 500, 1000)
        logger.debug("调整Python分配以减少GC运行次数")