import logging
from datetime import datetime, timedelta

from freqtrade.constants import LongShort
from freqtrade.persistence import Trade
from freqtrade.plugins.protections import IProtection, ProtectionReturn


logger = logging.getLogger(__name__)


class CooldownPeriod(IProtection):
    has_global_stop: bool = False
    has_local_stop: bool = True

    def _reason(self) -> str:
        """
        锁定原因说明
        """
        return f"冷却期，针对{self.unlock_reason_time_element}。"

    def short_desc(self) -> str:
        """
        简短方法描述 - 用于启动消息
        """
        return f"{self.name} - 冷却期 {self.unlock_reason_time_element}。"

    def _cooldown_period(self, pair: str, date_now: datetime) -> ProtectionReturn | None:
        """
        获取该交易对的最后一笔交易
        """
        look_back_until = date_now - timedelta(minutes=self._lookback_period)
        # filters = [
        #     Trade.is_open.is_(False),
        #     Trade.close_date > look_back_until,
        #     Trade.pair == pair,
        # ]
        # trade = Trade.get_trades(filters).first()
        trades = Trade.get_trades_proxy(pair=pair, is_open=False, close_date=look_back_until)
        if trades:
            # Get latest trade
            # Ignore type error as we know we only get closed trades.
            trade = sorted(trades, key=lambda t: t.close_date)[-1]  # type: ignore
            self.log_once(f"冷却期，针对{pair} {self.unlock_reason_time_element}。", logger.info)
            until = self.calculate_lock_end([trade])

            return ProtectionReturn(
                lock=True,
                until=until,
                reason=self._reason(),
            )

        return None

    def global_stop(self, date_now: datetime, side: LongShort) -> ProtectionReturn | None:
        """
        停止所有交易对的交易（入场）。这必须在整个“冷却期”内评估为真。
        :return: [布尔值, 结束时间, 原因]的元组。如果为真，则所有交易对将被锁定，原因是<reason>，直到<until>
        """
        # Not implemented for cooldown period.
        return None

    def stop_per_pair(
        self, pair: str, date_now: datetime, side: LongShort
    ) -> ProtectionReturn | None:
        """
        停止该交易对的交易（入场）。这必须在整个“冷却期”内评估为真。
        :return: [布尔值, 结束时间, 原因]的元组。如果为真，则该交易对将被锁定，原因是<reason>，直到<until>
        """
        return self._cooldown_period(pair, date_now)