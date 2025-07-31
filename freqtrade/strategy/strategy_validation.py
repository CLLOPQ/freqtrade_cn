import logging
from datetime import datetime

from pandas import DataFrame

from freqtrade.exceptions import StrategyError


logger = logging.getLogger(__name__)


class StrategyResultValidator:
    def __init__(self, dataframe: DataFrame, warn_only: bool = False):
        """
        初始化策略结果验证器。存储预期的数据框长度、最后收盘价和最后日期。
        """
        self._warn_only = warn_only
        self._length: int = len(dataframe)
        self._close: float = dataframe["close"].iloc[-1]
        self._date: datetime = dataframe["date"].iloc[-1]

    def assert_df(self, dataframe: DataFrame):
        """
        确保数据框（长度、最后一根K线）未被修改，且包含我们需要的所有元素。如果数据框与预期值不匹配，则引发StrategyError。如果设置了warn_only，则会记录警告而不是引发错误。
        :param dataframe: 要验证的数据框
        :raises StrategyError: 如果数据框与预期值不匹配。
        :logs Warning: 如果设置了warn_only且数据框与预期值不匹配。
        """
        message_template = "策略返回的数据框存在不匹配的 {}。"
        message = ""
        if dataframe is None:
            message = "未返回数据框（可能缺少return语句？）。"
        elif self._length != len(dataframe):
            message = message_template.format("length")
        elif self._close != dataframe["close"].iloc[-1]:
            message = message_template.format("last close price")
        elif self._date != dataframe["date"].iloc[-1]:
            message = message_template.format("last date")
        if message:
            if self._warn_only:
                logger.warning(message)
            else:
                raise StrategyError(message)