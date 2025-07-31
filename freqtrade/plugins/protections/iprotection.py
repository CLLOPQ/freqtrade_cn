import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from freqtrade.constants import Config, LongShort
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.misc import plural
from freqtrade.mixins import LoggingMixin
from freqtrade.persistence import LocalTrade


logger = logging.getLogger(__name__)


@dataclass
class ProtectionReturn:
    lock: bool
    until: datetime
    reason: str | None
    lock_side: str = "*"


class IProtection(LoggingMixin, ABC):
    # 是否可以全局停止交易
    has_global_stop: bool = False
    # 是否可以为单个交易对停止交易
    has_local_stop: bool = False

    def __init__(self, config: Config, protection_config: dict[str, Any]) -> None:
        self._config = config
        self._protection_config = protection_config
        self._stop_duration_candles: int | None = None
        self._stop_duration: int = 0
        self._lookback_period_candles: int | None = None
        self._unlock_at: str | None = None

        tf_in_min = timeframe_to_minutes(config["timeframe"])
        if "stop_duration_candles" in protection_config:
            self._stop_duration_candles = int(protection_config.get("stop_duration_candles", 1))
            self._stop_duration = tf_in_min * self._stop_duration_candles
        elif "unlock_at" in protection_config:
            self._unlock_at = protection_config.get("unlock_at")
        else:
            self._stop_duration = int(protection_config.get("stop_duration", 60))

        if "lookback_period_candles" in protection_config:
            self._lookback_period_candles = int(protection_config.get("lookback_period_candles", 1))
            self._lookback_period = tf_in_min * self._lookback_period_candles
        else:
            self._lookback_period_candles = None
            self._lookback_period = int(protection_config.get("lookback_period", 60))

        LoggingMixin.__init__(self, logger)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def stop_duration_str(self) -> str:
        """
        以蜡烛数或分钟数输出配置的停止时长
        """
        if self._stop_duration_candles:
            return (
                f"{self._stop_duration_candles} "
                f"{plural(self._stop_duration_candles, '蜡烛', '蜡烛')}"
            )
        else:
            return f"{self._stop_duration} {plural(self._stop_duration, '分钟', '分钟')}"

    @property
    def lookback_period_str(self) -> str:
        """
        以蜡烛数或分钟数输出配置的回溯周期
        """
        if self._lookback_period_candles:
            return (
                f"{self._lookback_period_candles} "
                f"{plural(self._lookback_period_candles, '蜡烛', '蜡烛')}"
            )
        else:
            return f"{self._lookback_period} {plural(self._lookback_period, '分钟', '分钟')}"

    @property
    def unlock_reason_time_element(self) -> str:
        """
        输出配置的解锁时间或停止时长
        """
        if self._unlock_at is not None:
            return f"until {self._unlock_at}"
        else:
            return f"for {self.stop_duration_str}"

    @abstractmethod
    def short_desc(self) -> str:
        """
        简短方法描述 - 用于启动消息 -> 请在子类中覆盖
        """

    @abstractmethod
    def global_stop(self, date_now: datetime, side: LongShort) -> ProtectionReturn | None:
        """
        停止所有交易对的交易（入场） 这必须在整个“冷却期”内评估为true。
        """

    @abstractmethod
    def stop_per_pair(
        self, pair: str, date_now: datetime, side: LongShort
    ) -> ProtectionReturn | None:
        """
        停止该交易对的交易（入场） 这必须在整个“冷却期”内评估为true。
        :return: 元组 [bool, until, reason]。 如果为true，该交易对将被锁定，原因是<reason>，直到<until>
        """

    def calculate_lock_end(self, trades: list[LocalTrade]) -> datetime:
        """
        获取锁定结束时间 隐式使用`self._stop_duration`或`self._unlock_at`，具体取决于配置。
        """
        max_date: datetime = max([trade.close_date for trade in trades if trade.close_date])
        # 来自数据库，未设置时区信息。
        if max_date.tzinfo is None:
            max_date = max_date.replace(tzinfo=timezone.utc)

        if self._unlock_at is not None:
            # 固定时间的解锁情况
            hour, minutes = self._unlock_at.split(":")
            unlock_at = max_date.replace(hour=int(hour), minute=int(minutes))
            if unlock_at < max_date:
                unlock_at += timedelta(days=1)
            return unlock_at

        until = max_date + timedelta(minutes=self._stop_duration)
        return until