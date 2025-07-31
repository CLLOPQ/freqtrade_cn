# pragma pylint: disable=W0603
"""钱包"""

import logging
from datetime import datetime, timedelta
from typing import Literal, NamedTuple

from freqtrade.constants import UNLIMITED_STAKE_AMOUNT, Config, IntOrInf
from freqtrade.enums import RunMode, TradingMode
from freqtrade.exceptions import DependencyException
from freqtrade.exchange import Exchange
from freqtrade.misc import safe_value_fallback
from freqtrade.persistence import LocalTrade, Trade
from freqtrade.util.datetime_helpers import dt_now


logger = logging.getLogger(__name__)


# 钱包数据结构
class Wallet(NamedTuple):
    currency: str
    free: float = 0
    used: float = 0
    total: float = 0


class PositionWallet(NamedTuple):
    symbol: str
    position: float = 0
    leverage: float | None = 0  # 不要使用此属性 - 不保证已设置
    collateral: float = 0
    side: str = "long"


class Wallets:
    def __init__(self, config: Config, exchange: Exchange, is_backtest: bool = False) -> None:
        self._config = config
        self._is_backtest = is_backtest
        self._exchange = exchange
        self._wallets: dict[str, Wallet] = {}
        self._positions: dict[str, PositionWallet] = {}
        self._start_cap: dict[str, float] = {}

        self._stake_currency = self._exchange.get_proxy_coin()

        if isinstance(_start_cap := config["dry_run_wallet"], float | int):
            self._start_cap[self._stake_currency] = _start_cap
        else:
            self._start_cap = _start_cap

        self._last_wallet_refresh: datetime | None = None
        self.update()

    def get_free(self, currency: str) -> float:
        """获取指定货币的自由余额"""
        balance = self._wallets.get(currency)
        if balance and balance.free:
            return balance.free
        else:
            return 0

    def get_used(self, currency: str) -> float:
        """获取指定货币的已用余额"""
        balance = self._wallets.get(currency)
        if balance and balance.used:
            return balance.used
        else:
            return 0

    def get_total(self, currency: str) -> float:
        """获取指定货币的总余额"""
        balance = self._wallets.get(currency)
        if balance and balance.total:
            return balance.total
        else:
            return 0

    def get_collateral(self) -> float:
        """
        获取用于清算价格计算的总抵押品。
        """
        if self._config.get("margin_mode") == "cross":
            # free 包含所有余额，并与持仓抵押品结合使用
            # 作为"钱包余额"。
            return self.get_free(self._stake_currency) + sum(
                pos.collateral for pos in self._positions.values()
            )
        return self.get_total(self._stake_currency)

    def get_owned(self, pair: str, base_currency: str) -> float:
        """
        获取当前持有的价值。
        设计用于现货和期货市场。
        """
        if self._config.get("trading_mode", "spot") != TradingMode.FUTURES:
            return self.get_total(base_currency) or 0
        if pos := self._positions.get(pair):
            return pos.position
        return 0

    def _update_dry(self) -> None:
        """
        在模拟模式下从数据库更新
        - 将已平仓交易的利润应用于本金
        - 减去未平仓交易中已占用的本金
        - 更新当前在交易中的货币余额
        """
        # 重新创建 _wallets 以重置已平仓交易的余额
        _wallets = {}
        _positions = {}
        open_trades = Trade.get_trades_proxy(is_open=True)
        if not self._is_backtest:
            # 实盘/模拟模式
            tot_profit = Trade.get_total_closed_profit()
        else:
            # 回测模式
            tot_profit = LocalTrade.bt_total_profit
        tot_profit += sum(trade.realized_profit for trade in open_trades)
        tot_in_trades = sum(trade.stake_amount for trade in open_trades)
        used_stake = 0.0

        if self._config.get("trading_mode", "spot") != TradingMode.FUTURES:
            for trade in open_trades:
                curr = self._exchange.get_pair_base_currency(trade.pair)
                used_stake += sum(
                    o.stake_amount for o in trade.open_orders if o.ft_order_side == trade.entry_side
                )
                pending = sum(
                    o.amount
                    for o in trade.open_orders
                    if o.amount and o.ft_order_side == trade.exit_side
                )
                curr_wallet_bal = self._start_cap.get(curr, 0)

                _wallets[curr] = Wallet(
                    curr,
                    curr_wallet_bal + trade.amount - pending,
                    pending,
                    trade.amount + curr_wallet_bal,
                )
        else:
            for position in open_trades:
                _positions[position.pair] = PositionWallet(
                    position.pair,
                    position=position.amount,
                    leverage=position.leverage,
                    collateral=position.stake_amount,
                    side=position.trade_direction,
                )

            used_stake = tot_in_trades

        cross_margin = 0.0
        if self._config.get("margin_mode") == "cross":
            # 在全仓保证金模式下，总余额用作抵押品。
            # 这作为"自由"资金转移到 stake 货币余额中。
            # 与 get_collateral() 实现紧密关联。
            for curr, bal in self._start_cap.items():
                if curr == self._stake_currency:
                    continue
                rate = self._exchange.get_conversion_rate(curr, self._stake_currency)
                if rate:
                    cross_margin += bal * rate

        current_stake = self._start_cap.get(self._stake_currency, 0) + tot_profit - tot_in_trades
        total_stake = current_stake + used_stake

        _wallets[self._stake_currency] = Wallet(
            currency=self._stake_currency,
            free=current_stake + cross_margin,
            used=used_stake,
            total=total_stake,
        )
        for currency, bal in self._start_cap.items():
            if currency not in _wallets:
                _wallets[currency] = Wallet(currency, bal, 0, bal)

        self._wallets = _wallets
        self._positions = _positions

    def _update_live(self) -> None:
        balances = self._exchange.get_balances()
        _wallets = {}

        for currency in balances:
            if isinstance(balances[currency], dict):
                _wallets[currency] = Wallet(
                    currency,
                    balances[currency].get("free", 0),
                    balances[currency].get("used", 0),
                    balances[currency].get("total", 0),
                )

        positions = self._exchange.fetch_positions()
        _parsed_positions = {}
        for position in positions:
            symbol = position["symbol"]
            if position["side"] is None or position["collateral"] == 0.0:
                # 持仓未平仓...
                continue
            size = self._exchange._contracts_to_amount(symbol, position["contracts"])
            collateral = safe_value_fallback(position, "initialMargin", "collateral", 0.0)
            leverage = position.get("leverage")
            _parsed_positions[symbol] = PositionWallet(
                symbol,
                position=size,
                leverage=leverage,
                collateral=collateral,
                side=position["side"],
            )
        self._positions = _parsed_positions
        self._wallets = _wallets

    def update(self, require_update: bool = True) -> None:
        """
        从配置的版本更新钱包。
        默认情况下，从交易所更新。
        更新跳过仅用于用户调用的/balance命令，因为
        对于交易操作，需要最新的余额。
        :param require_update: 如果最近刷新过余额，允许跳过更新
        """
        now = dt_now()
        if (
            require_update
            or self._last_wallet_refresh is None
            or (self._last_wallet_refresh + timedelta(seconds=3600) < now)
        ):
            if not self._config["dry_run"] or self._config.get("runmode") == RunMode.LIVE:
                self._update_live()
            else:
                self._update_dry()
            self._local_log("钱包已同步。")
            self._last_wallet_refresh = dt_now()

    def get_all_balances(self) -> dict[str, Wallet]:
        return self._wallets

    def get_all_positions(self) -> dict[str, PositionWallet]:
        return self._positions

    def _check_exit_amount(self, trade: Trade) -> bool:
        """检查退出金额是否在钱包中可用"""
        if trade.trading_mode != TradingMode.FUTURES:
            # 偏移量略高于 safe_exit_amount 中的偏移量。
            wallet_amount: float = self.get_total(trade.safe_base_currency) * (2 - 0.981)
        else:
            # wallet_amount: float = self.wallets.get_free(trade.safe_base_currency)
            position = self._positions.get(trade.pair)
            if position is None:
                # 我们没有持仓 :O
                return False
            wallet_amount = position.position

        if wallet_amount >= trade.amount:
            return True
        return False

    def check_exit_amount(self, trade: Trade) -> bool:
        """
        检查退出金额是否在钱包中可用。
        :param trade: 要检查的交易
        :return: 如果退出金额可用则返回 True，否则返回 False
        """
        if not self._check_exit_amount(trade):
            # 确保更新钱包
            self.update()
            return self._check_exit_amount(trade)

        return True

    def get_starting_balance(self) -> float:
        """
        检索初始余额 - 基于可用资金，或通过使用当前余额减去
        """
        if "available_capital" in self._config:
            return self._config["available_capital"]
        else:
            tot_profit = Trade.get_total_closed_profit()
            open_stakes = Trade.total_open_trades_stakes()
            available_balance = self.get_free(self._stake_currency)
            return (available_balance - tot_profit + open_stakes) * self._config[
                "tradable_balance_ratio"
            ]

    def get_total_stake_amount(self):
        """
        返回 stake 货币中当前可用的总余额，包括已占用的 stake 并考虑可交易余额比例。
        计算方式为
        (<未平仓交易 stake> + 自由金额) * 可交易余额比例
        """
        val_tied_up = Trade.total_open_trades_stakes()
        if "available_capital" in self._config:
            starting_balance = self._config["available_capital"]
            tot_profit = Trade.get_total_closed_profit()
            available_amount = starting_balance + tot_profit

        else:
            # 确保从总余额中使用 <tradable_balance_ratio>%
            # 否则可能会随着每个未平仓交易降低 stake。
            # (已占用 + 当前自由) * 比例) - 已占用
            available_amount = (val_tied_up + self.get_free(self._stake_currency)) * self._config[
                "tradable_balance_ratio"
            ]
        return available_amount

    def get_available_stake_amount(self) -> float:
        """
        返回 stake 货币中当前可用的总余额，考虑可交易余额比例。
        计算方式为
        (<未平仓交易 stake> + 自由金额) * 可交易余额比例 - <未平仓交易 stake>
        """

        free = self.get_free(self._stake_currency)
        return min(self.get_total_stake_amount() - Trade.total_open_trades_stakes(), free)

    def _calculate_unlimited_stake_amount(
        self, available_amount: float, val_tied_up: float, max_open_trades: IntOrInf
    ) -> float:
        """
        计算“无限” stake 金额的 stake 数量
        :return: 如果达到最大交易数量则为 0，否则为要使用的 stake 数量。
        """
        if max_open_trades == 0:
            return 0

        possible_stake = (available_amount + val_tied_up) / max_open_trades
        # 理论金额可能超过可用金额 - 因此限制为可用金额！
        return min(possible_stake, available_amount)

    def _check_available_stake_amount(self, stake_amount: float, available_amount: float) -> float:
        """
        检查可用余额是否能满足 stake 金额
        :return: float：stake 金额
        :raise: 如果余额低于 stake 金额则引发 DependencyException
        """

        if self._config["amend_last_stake_amount"]:
            # 剩余金额至少需要 stake_amount * last_stake_amount_min_ratio
            # 否则剩余金额太低无法交易。
            if available_amount > (stake_amount * self._config["last_stake_amount_min_ratio"]):
                stake_amount = min(stake_amount, available_amount)
            else:
                stake_amount = 0

        if available_amount < stake_amount:
            raise DependencyException(
                f"可用余额（{available_amount} {self._config['stake_currency']}）低于 stake 金额（{stake_amount} {self._config['stake_currency']}）"
            )

        return max(stake_amount, 0)

    def get_trade_stake_amount(
        self, pair: str, max_open_trades: IntOrInf, update: bool = True
    ) -> float:
        """
        计算交易的 stake 金额
        :return: float：stake 金额
        :raise: 如果可用 stake 金额太低则引发 DependencyException
        """
        stake_amount: float
        # 确保钱包是最新的。
        if update:
            self.update()
        val_tied_up = Trade.total_open_trades_stakes()
        available_amount = self.get_available_stake_amount()

        stake_amount = self._config["stake_amount"]
        if stake_amount == UNLIMITED_STAKE_AMOUNT:
            stake_amount = self._calculate_unlimited_stake_amount(
                available_amount, val_tied_up, max_open_trades
            )

        return self._check_available_stake_amount(stake_amount, available_amount)

    def validate_stake_amount(
        self,
        pair: str,
        stake_amount: float | None,
        min_stake_amount: float | None,
        max_stake_amount: float,
        trade_amount: float | None,
    ):
        """验证 stake 金额"""
        if not stake_amount:
            self._local_log(
                f"stake 金额为 {stake_amount}，忽略对 {pair} 的可能交易。",
                level="debug",
            )
            return 0

        max_allowed_stake = min(max_stake_amount, self.get_available_stake_amount())
        if trade_amount:
            # 如果处于交易中，则最终交易数量不能超过最大 stake
            # 否则可能无法平仓。
            max_allowed_stake = min(max_allowed_stake, max_stake_amount - trade_amount)

        if min_stake_amount is not None and min_stake_amount > max_allowed_stake:
            self._local_log(
                "最小 stake 金额 > 可用余额。"
                f"{min_stake_amount} > {max_allowed_stake}",
                level="warning",
            )
            return 0
        if min_stake_amount is not None and stake_amount < min_stake_amount:
            self._local_log(
                f"交易对 {pair} 的 stake 金额太小（{stake_amount} < {min_stake_amount}），调整为 {min_stake_amount}。"
            )
            if stake_amount * 1.3 < min_stake_amount:
                # 将 stake 金额调整上限设为 +30%。
                self._local_log(
                    f"交易对 {pair} 的调整后 stake 金额比期望的 stake 金额大 30% 以上（{stake_amount:.8f} * 1.3 = "
                    f"{stake_amount * 1.3:.8f}）< {min_stake_amount}），忽略交易。"
                )
                return 0
            stake_amount = min_stake_amount

        if stake_amount > max_allowed_stake:
            self._local_log(
                f"交易对 {pair} 的 stake 金额太大（{stake_amount} > {max_allowed_stake}），调整为 {max_allowed_stake}。"
            )
            stake_amount = max_allowed_stake
        return stake_amount

    def _local_log(self, msg: str, level: Literal["info", "warning", "debug"] = "info") -> None:
        """
        记录消息到本地日志。
        """
        if not self._is_backtest:
            if level == "warning":
                logger.warning(msg)
            elif level == "debug":
                logger.debug(msg)
            else:
                logger.info(msg)