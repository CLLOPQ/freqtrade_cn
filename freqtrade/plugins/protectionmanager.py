"""
保护管理器类
"""

import logging
from datetime import datetime, timezone
from typing import Any

from freqtrade.constants import Config, LongShort
from freqtrade.exceptions import ConfigurationError
from freqtrade.persistence import PairLocks
from freqtrade.persistence.models import PairLock
from freqtrade.plugins.protections import IProtection
from freqtrade.resolvers import ProtectionResolver


logger = logging.getLogger(__name__)


class ProtectionManager:
    def __init__(self, config: Config, protections: list) -> None:
        self._config = config

        self._protection_handlers: list[IProtection] = []
        self.validate_protections(protections)
        for protection_handler_config in protections:
            protection_handler = ProtectionResolver.load_protection(
                protection_handler_config["method"],
                config=config,
                protection_config=protection_handler_config,
            )
            self._protection_handlers.append(protection_handler)

        if not self._protection_handlers:
            logger.info("未定义保护处理器。")

    @property
    def name_list(self) -> list[str]:
        """
        获取已加载的保护处理器名称列表
        """
        return [p.name for p in self._protection_handlers]

    def short_desc(self) -> list[dict]:
        """
        每个保护处理器的short_desc列表
        """
        return [{p.name: p.short_desc()} for p in self._protection_handlers]

    def global_stop(self, now: datetime | None = None, side: LongShort = "long") -> PairLock | None:
        if not now:
            now = datetime.now(timezone.utc)
        result = None
        for protection_handler in self._protection_handlers:
            if protection_handler.has_global_stop:
                lock = protection_handler.global_stop(date_now=now, side=side)
                if lock and lock.until:
                    if not PairLocks.is_global_lock(lock.until, side=lock.lock_side):
                        result = PairLocks.lock_pair(
                            "*", lock.until, lock.reason, now=now, side=lock.lock_side
                        )
        return result

    def stop_per_pair(
        self, pair, now: datetime | None = None, side: LongShort = "long"
    ) -> PairLock | None:
        if not now:
            now = datetime.now(timezone.utc)
        result = None
        for protection_handler in self._protection_handlers:
            if protection_handler.has_local_stop:
                lock = protection_handler.stop_per_pair(pair=pair, date_now=now, side=side)
                if lock and lock.until:
                    if not PairLocks.is_pair_locked(pair, lock.until, lock.lock_side):
                        result = PairLocks.lock_pair(
                            pair, lock.until, lock.reason, now=now, side=lock.lock_side
                        )
        return result

    @staticmethod
    def validate_protections(protections: list[dict[str, Any]]) -> None:
        """
        验证保护设置的有效性
        """

        for prot in protections:
            parsed_unlock_at = None
            if (config_unlock_at := prot.get("unlock_at")) is not None:
                try:
                    parsed_unlock_at = datetime.strptime(config_unlock_at, "%H:%M")
                except ValueError:
                    raise ConfigurationError(
                        f"解锁时间的日期格式无效: {config_unlock_at}。"
                    )

            if "stop_duration" in prot and "stop_duration_candles" in prot:
                raise ConfigurationError(
                    "保护必须指定`stop_duration`或`stop_duration_candles`。\n"
                    f"请修复保护{prot.get('method')}。"
                )

            if "lookback_period" in prot and "lookback_period_candles" in prot:
                raise ConfigurationError(
                    "保护必须指定`lookback_period`或`lookback_period_candles`。\n"
                    f"请修复保护{prot.get('method')}。"
                )

            if parsed_unlock_at is not None and (
                "stop_duration" in prot or "stop_duration_candles" in prot
            ):
                raise ConfigurationError(
                    "保护必须指定`unlock_at`、`stop_duration`或`stop_duration_candles`。\n"
                    f"请修复保护{prot.get('method')}。"
                )