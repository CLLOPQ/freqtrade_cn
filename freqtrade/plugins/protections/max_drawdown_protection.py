import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from freqtrade.constants import Config, LongShort
from freqtrade.data.metrics import calculate_max_drawdown
from freqtrade.persistence import Trade
from freqtrade.plugins.protections import IProtection, ProtectionReturn


logger = logging.getLogger(__name__)


class MaxDrawdown(IProtection):
    has_global_stop: bool = True
    has_local_stop: bool = False

    def __init__(self, config: Config, protection_config: dict[str, Any]) -> None:
        super().__init__(config, protection_config)

        self._trade_limit = protection_config.get("trade_limit", 1)  # 交易限制，默认为1
        self._max_allowed_drawdown = protection_config.get("max_allowed_drawdown", 0.0)  # 最大允许回撤，默认为0.0
        # TODO: Implement checks to limit max_drawdown to sensible values

    def short_desc(self) -> str:
        """
        简短方法描述 - 用于启动消息
        """
        return (
            f"{self.name} - 最大回撤保护，若在 {self.lookback_period_str} 内回撤超过 {self._max_allowed_drawdown}，则停止交易。"
        )

    def _reason(self, drawdown: float) -> str:
        """
        使用的锁定原因
        """
        return (
            f"{drawdown} 在 {self.lookback_period_str} 内超过 {self._max_allowed_drawdown}，锁定 {self.unlock_reason_time_element}。"
        )

    def _max_drawdown(self, date_now: datetime) -> ProtectionReturn | None:
        """
        评估近期交易的回撤...
        """
        look_back_until = date_now - timedelta(minutes=self._lookback_period)  # 回溯截止时间

        trades = Trade.get_trades_proxy(is_open=False, close_date=look_back_until)  # 获取相关时间段内的非开仓交易

        trades_df = pd.DataFrame([trade.to_json() for trade in trades])  # 将交易数据转为DataFrame

        if len(trades) < self._trade_limit:
            # 相关时间段内交易数量不足
            return None

        # 回撤始终为正值
        try:
            # TODO: 应使用绝对收益计算，考虑账户余额。
            drawdown_obj = calculate_max_drawdown(trades_df, value_col="close_profit")  # 计算最大回撤
            drawdown = drawdown_obj.drawdown_abs  # 绝对回撤
        except ValueError:
            return None

        if drawdown > self._max_allowed_drawdown:
            self.log_once(
                f"交易因最大回撤 {drawdown:.2f} 超过 {self._max_allowed_drawdown} 而在 {self.lookback_period_str} 内停止。",
                logger.info,
            )

            until = self.calculate_lock_end(trades)  # 计算锁定结束时间

            return ProtectionReturn(
                lock=True,
                until=until,
                reason=self._reason(drawdown),
            )

        return None

    def global_stop(self, date_now: datetime, side: LongShort) -> ProtectionReturn | None:
        """
        停止所有交易对的交易（入场）
        这必须在整个“冷却期”内评估为true。
        :return: [布尔值, 结束时间, 原因]的元组。
            如果为true，所有交易对将被锁定，原因是<reason>，直到<until>
        """
        return self._max_drawdown(date_now)

    def stop_per_pair(
        self, pair: str, date_now: datetime, side: LongShort
    ) -> ProtectionReturn | None:
        """
        停止该交易对的交易（入场）
        这必须在整个“冷却期”内评估为true。
        :return: [布尔值, 结束时间, 原因]的元组。
            如果为true，该交易对将被锁定，原因是<reason>，直到<until>
        """
        return None