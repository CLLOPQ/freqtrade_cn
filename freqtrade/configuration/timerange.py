"""
此模块包含参数管理器类
"""

import logging
import re
from datetime import datetime, timezone

from typing_extensions import Self

from freqtrade.constants import DATETIME_PRINT_FORMAT
from freqtrade.exceptions import ConfigurationError
from freqtrade.util import dt_from_ts


logger = logging.getLogger(__name__)


class TimeRange:
    """
    定义时间范围输入的对象。
    [start/stop]type 定义是否应使用 [start/stop]ts。
    如果 *type 为 None，则不使用相应的起始值。
    """

    def __init__(
        self,
        starttype: str | None = None,
        stoptype: str | None = None,
        startts: int = 0,
        stopts: int = 0,
    ):
        self.starttype: str | None = starttype
        self.stoptype: str | None = stoptype
        self.startts: int = startts
        self.stopts: int = stopts

    @property
    def startdt(self) -> datetime | None:
        if self.startts:
            return dt_from_ts(self.startts)
        return None

    @property
    def stopdt(self) -> datetime | None:
        if self.stopts:
            return dt_from_ts(self.stopts)
        return None

    @property
    def timerange_str(self) -> str:
        """
        返回由 parse_timerange 使用的时间范围的字符串表示形式。
        遵循 yyyymmdd-yyyymmdd 格式 - 省略未设置的部分。
        """
        start = ""
        stop = ""
        if startdt := self.startdt:
            start = startdt.strftime("%Y%m%d")
        if stopdt := self.stopdt:
            stop = stopdt.strftime("%Y%m%d")
        return f"{start}-{stop}"

    @property
    def start_fmt(self) -> str:
        """
        返回起始日期的字符串表示形式
        """
        val = "无限制"
        if (startdt := self.startdt) is not None:
            val = startdt.strftime(DATETIME_PRINT_FORMAT)
        return val

    @property
    def stop_fmt(self) -> str:
        """
        返回结束日期的字符串表示形式
        """
        val = "无限制"
        if (stopdt := self.stopdt) is not None:
            val = stopdt.strftime(DATETIME_PRINT_FORMAT)
        return val

    def __eq__(self, other):
        """重写默认的等于行为"""
        return (
            self.starttype == other.starttype
            and self.stoptype == other.stoptype
            and self.startts == other.startts
            and self.stopts == other.stopts
        )

    def subtract_start(self, seconds: int) -> None:
        """
        如果设置了 startts，则从 startts 中减去 <seconds>。
        :param seconds: 要从开始时间减去的秒数
        :return: None（就地修改对象）
        """
        if self.startts:
            self.startts = self.startts - seconds

    def adjust_start_if_necessary(
        self, timeframe_secs: int, startup_candles: int, min_date: datetime
    ) -> None:
        """
        将 startts 调整 <startup_candles> 根蜡烛。
        仅在没有可用的启动蜡烛时适用。
        :param timeframe_secs: 以秒为单位的时间框架，例如 `timeframe_to_seconds('5m')`
        :param startup_candles: 要将开始日期向前移动的蜡烛数量
        :param min_date: 加载的最小数据日期。决定是否必须移动开始时间的关键标准
        :return: None（就地修改对象）
        """
        if not self.starttype or (startup_candles and min_date.timestamp() >= self.startts):
            # 如果未定义 startts，或者回测数据从定义的回测日期开始
            logger.warning(
                "将开始日期移动 %s 根蜡烛以考虑启动时间。", startup_candles
            )
            self.startts = int(min_date.timestamp() + timeframe_secs * startup_candles)
            self.starttype = "date"

    @classmethod
    def parse_timerange(cls, text: str | None) -> Self:
        """
        解析 --timerange 参数的值以确定所需的范围
        :param text: 来自 --timerange 的值
        :return: 开始和结束范围周期
        """
        if not text:
            return cls(None, None, 0, 0)
        syntax = [
            (r"^-(\d{8})$", (None, "date")),
            (r"^(\d{8})-$", ("date", None)),
            (r"^(\d{8})-(\d{8})$", ("date", "date")),
            (r"^-(\d{10})$", (None, "date")),
            (r"^(\d{10})-$", ("date", None)),
            (r"^(\d{10})-(\d{10})$", ("date", "date")),
            (r"^-(\d{13})$", (None, "date")),
            (r"^(\d{13})-$", ("date", None)),
            (r"^(\d{13})-(\d{13})$", ("date", "date")),
        ]
        for rex, stype in syntax:
            # 将正则表达式应用于文本
            match = re.match(rex, text)
            if match:  # 正则表达式已匹配
                rvals = match.groups()
                index = 0
                start: int = 0
                stop: int = 0
                if stype[0]:
                    starts = rvals[index]
                    if stype[0] == "date" and len(starts) == 8:
                        start = int(
                            datetime.strptime(starts, "%Y%m%d")
                            .replace(tzinfo=timezone.utc)
                            .timestamp()
                        )
                    elif len(starts) == 13:
                        start = int(starts) // 1000
                    else:
                        start = int(starts)
                    index += 1
                if stype[1]:
                    stops = rvals[index]
                    if stype[1] == "date" and len(stops) == 8:
                        stop = int(
                            datetime.strptime(stops, "%Y%m%d")
                            .replace(tzinfo=timezone.utc)
                            .timestamp()
                        )
                    elif len(stops) == 13:
                        stop = int(stops) // 1000
                    else:
                        stop = int(stops)
                if start > stop > 0:
                    raise ConfigurationError(
                        f'时间范围 "{text}" 的开始日期在结束日期之后'
                    )
                return cls(stype[0], stype[1], start, stop)
        raise ConfigurationError(f'时间范围 "{text}" 的语法不正确')