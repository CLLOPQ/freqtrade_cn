import logging
from datetime import datetime, timedelta
from typing import Any

from freqtrade.constants import Config, LongShort
from freqtrade.persistence import Trade
from freqtrade.plugins.protections import IProtection, ProtectionReturn


logger = logging.getLogger(__name__)


class LowProfitPairs(IProtection):
    has_global_stop: bool = False
    has_local_stop: bool = True

    def __init__(self, config: Config, protection_config: dict[str, Any]) -> None:
        super().__init__(config, protection_config)

        self._trade_limit = protection_config.get("trade_limit", 1)
        self._required_profit = protection_config.get("required_profit", 0.0)
        self._only_per_side = protection_config.get("only_per_side", False)

    def short_desc(self) -> str:
        """
        简短方法描述 - 用于启动消息
        """
        return (
            f"{self.name} - 低利润保护，锁定{self.lookback_period_str}内利润低于{self._required_profit}的交易对。"
        )

    def _reason(self, profit: float) -> str:
        """
        使用的锁定原因
        """
        return (
            f"{profit} < {self._required_profit}（在{self.lookback_period_str}内），锁定{self.unlock_reason_time_element}。"
        )

    def _low_profit(
        self, date_now: datetime, pair: str, side: LongShort
    ) -> ProtectionReturn | None:
        """
        评估交易对的近期交易
        """
        look_back_until = date_now - timedelta(minutes=self._lookback_period)
        # filters = [
        #     Trade.is_open.is_(False),
        #     Trade.close_date > look_back_until,
        # ]
        # if pair:
        #     filters.append(Trade.pair == pair)

        trades = Trade.get_trades_proxy(pair=pair, is_open=False, close_date=look_back_until)
        # trades = Trade.get_trades(filters).all()
        if len(trades) < self._trade_limit:
            # 相关时间段内交易数量不足
            return None

        profit = sum(
            trade.close_profit
            for trade in trades
            if trade.close_profit and (not self._only_per_side or trade.trade_direction == side)
        )
        if profit < self._required_profit:
            self.log_once(
                f"交易对{pair}因{profit:.2f} < {self._required_profit}（在{self._lookback_period}分钟内）停止交易。",
                logger.info,
            )
            until = self.calculate_lock_end(trades)

            return ProtectionReturn(
                lock=True,
                until=until,
                reason=self._reason(profit),
                lock_side=(side if self._only_per_side else "*"),
            )

        return None

    def global_stop(self, date_now: datetime, side: LongShort) -> ProtectionReturn | None:
        """
        停止所有交易对的交易（入场）
        这必须在整个“冷却期”内都为true。
        :return: 元组 [bool, until, reason]。
            如果为true，所有交易对将被锁定，原因是<reason>，直到<until>
        """
        return None

    def stop_per_pair(
        self, pair: str, date_now: datetime, side: LongShort
    ) -> ProtectionReturn | None:
        """
        停止该交易对的交易（入场）
        这必须在整个“冷却期”内都为true。
        :return: 元组 [bool, until, reason]。
            如果为true，该交易对将被锁定，原因是<reason>，直到<until>
        """
        return self._low_profit(date_now, pair=pair, side=side)