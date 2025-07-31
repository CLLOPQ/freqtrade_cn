"""系统特定和性能调优"""

from freqtrade.system.asyncio_config import asyncio_setup
from freqtrade.system.gc_setup import gc_set_threshold
from freqtrade.system.version_info import print_version_info


__all__ = ["asyncio_setup", "gc_set_threshold", "print_version_info"]