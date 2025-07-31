"""
用于 freqtrade 的数据操作模块
"""

from freqtrade.data import converter


# 限制使用 `from freqtrade.data import *` 时导入的内容
__all__ = ["converter"]