"""
ccxt的Precise（字符串数学运算）的轻量级包装器，以便从freqtrade导入并支持浮点数初始化器
"""

from ccxt import Precise


class FtPrecise(Precise):
    def __init__(self, number, decimals=None):
        if not isinstance(number, str):
            number = str(number)
        super().__init__(number, decimals)