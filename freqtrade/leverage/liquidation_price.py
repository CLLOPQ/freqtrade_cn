import logging

from freqtrade.enums import MarginMode
from freqtrade.exceptions import DependencyException
from freqtrade.exchange import Exchange
from freqtrade.persistence import LocalTrade, Trade
from freqtrade.wallets import Wallets


logger = logging.getLogger(__name__)


def update_liquidation_prices(
    trade: LocalTrade | None = None,
    *,
    exchange: Exchange,
    wallets: Wallets,
    stake_currency: str,
    dry_run: bool = False,
):
    """
    更新逐仓模式下交易的强平价格。
    更新全仓模式下所有交易的强平价格。
    """
    try:
        if exchange.margin_mode == MarginMode.CROSS:
            total_wallet_stake = 0.0
            if dry_run:
                # 仅全仓模式需要的参数
                total_wallet_stake = wallets.get_collateral()
                logger.info(
                    "正在更新所有未平仓交易的强平价格。"
                    f"保证金 {total_wallet_stake} {stake_currency}。"
                )

            open_trades: list[Trade] = Trade.get_open_trades()
            for t in open_trades:
                if t.has_open_position:
                    # TODO: 应该批量更新
                    t.set_liquidation_price(
                        exchange.get_liquidation_price(
                            pair=t.pair,
                            open_rate=t.open_rate,
                            is_short=t.is_short,
                            amount=t.amount,
                            stake_amount=t.stake_amount,
                            leverage=t.leverage,
                            wallet_balance=total_wallet_stake,
                            open_trades=open_trades,
                        )
                    )
        elif trade:
            trade.set_liquidation_price(
                exchange.get_liquidation_price(
                    pair=trade.pair,
                    open_rate=trade.open_rate,
                    is_short=trade.is_short,
                    amount=trade.amount,
                    stake_amount=trade.stake_amount,
                    leverage=trade.leverage,
                    wallet_balance=trade.stake_amount,
                )
            )
        else:
            raise DependencyException(
                "在逐仓模式下更新强平价格需要交易对象。"
            )
    except DependencyException:
        logger.warning("无法计算强平价格")