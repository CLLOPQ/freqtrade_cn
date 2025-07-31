from freqtrade.persistence.custom_data import CustomDataWrapper
from freqtrade.persistence.pairlock_middleware import PairLocks
from freqtrade.persistence.trade_model import Trade


def disable_database_use(timeframe: str) -> None:
    """
    禁用PairLocks和Trade模型的数据库使用。
    用于回测以及其他一些工具命令。
    """
    PairLocks.use_db = False
    PairLocks.timeframe = timeframe
    Trade.use_db = False
    CustomDataWrapper.use_db = False


def enable_database_use() -> None:
    """
    清理函数，用于恢复数据库使用。
    """
    PairLocks.use_db = True
    PairLocks.timeframe = ""
    Trade.use_db = True
    CustomDataWrapper.use_db = True


class FtNoDBContext:
    def __init__(self, timeframe: str = ""):
        self.timeframe = timeframe

    def __enter__(self):
        disable_database_use(self.timeframe)

    def __exit__(self, exc_type, exc_val, exc_tb):
        enable_database_use()