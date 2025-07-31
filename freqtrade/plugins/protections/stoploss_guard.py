import logging
from datetime import datetime, timedelta
from typing import Any

from freqtrade.constants import Config, LongShort
from freqtrade.enums import ExitType
from freqtrade.persistence import Trade
from freqtrade.plugins.protections import IProtection, ProtectionReturn


logger = logging.getLogger(__name__)


class StoplossGuard(IProtection):
    has_global_stop: bool = True
    has_local_stop: bool = True

    def __init__(self, config: Config, protection_config: dict[str, Any]) -> None:
        super().__init__(config, protection_config)

        self._trade_limit = protection_config.get("trade_limit", 10)
        self._disable_global_stop = protection_config.get("only_per_pair", False)
        self._only_per_side = protection_config.get("only_per_side", False)
        self._profit_limit = protection_config.get("required_profit", 0.0)

    def short_desc(self) -> str:
        """
        简短方法描述 - 用于启动消息
        """
        return (
            f"{self.name} - 频繁止损卫士，在{self.lookback_period_str}内发生{self._trade_limit}次止损"
            f"且利润低于{self._profit_limit:.2%}。"
        )

    def _reason(self) -> str:
        """
        使用的锁定原因
        """
        return (
            f"{self._trade_limit}次止损在{self._lookback_period}分钟内，"
            f"锁定{self.unlock_reason_time_element}。"
        )

    def _stoploss_guard(
        self, date_now: datetime, pair: str | None, side: LongShort
    ) -> ProtectionReturn | None:
        """
        评估近期交易
        """
        look_back_until = date_now - timedelta(minutes=self._lookback_period)

        trades1 = Trade.get_trades_proxy(pair=pair, is_open=False, close_date=look_back_until)
        trades = [
            trade
            for trade in trades1
            if (
                str(trade.exit_reason)
                in (
                    ExitType.TRAILING_STOP_LOSS.value,
                    ExitType.STOP_LOSS.value,
                    ExitType.STOPLOSS_ON_EXCHANGE.value,
                    ExitType.LIQUIDATION.value,
                )
                and trade.close_profit
                and trade.close_profit < self._profit_limit
            )
        ]

        if self._only_per_side:
            # 仅长仓或短仓交易
            trades = [trade for trade in trades if trade.trade_direction == side]

        if len(trades) < self._trade_limit:
            return None

        self.log_once(
            f"由于在{self._lookback_period}分钟内发生了{self._trade_limit}次止损，交易已停止。",
            logger.info,
        )
        until = self.calculate_lock_end(trades)
        return ProtectionReturn(
            lock=True,
            until=until,
            reason=self._reason(),
            lock_side=(side if self._only_per_side else "*"),
        )

    def global_stop(self, date_now: datetime, side: LongShort) -> ProtectionReturn | None:
        """
        停止所有交易对的交易（入场）
        这必须在整个“冷却期”内评估为真。
        :return: [bool, until, reason]的元组。
            如果为真，所有交易对将被锁定，原因是<reason>，直到<until>
        """
        if self._disable_global_stop:
            return None
        return self._stoploss_guard(date_now, None, side)

    def stop_per_pair(
        self, pair: str, date_now: datetime, side: LongShort
    ) -> ProtectionReturn | None:
        """
        停止该交易对的交易（入场）
        这必须在整个“冷却期”内评估为真。
        :return: [bool, until, reason]的元组。
            如果为真，该交易对将被锁定，原因是<reason>，直到<until>
        """
        return self._stoploss_guard(date_now, pair, side)