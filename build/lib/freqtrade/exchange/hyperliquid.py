"""Hyperliquid 交易所子类"""

import logging
from copy import deepcopy
from datetime import datetime

from freqtrade.constants import BuySell
from freqtrade.enums import MarginMode, TradingMode
from freqtrade.exceptions import ExchangeError, OperationalException
from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import CcxtOrder, FtHas
from freqtrade.util.datetime_helpers import dt_from_ts


logger = logging.getLogger(__name__)


class Hyperliquid(Exchange):
    """Hyperliquid 交易所类。
    包含使 Freqtrade 能够与该交易所正常工作所需的调整。
    """

    _ft_has: FtHas = {
        "ohlcv_has_history": False,  # 不支持OHLCV历史数据
        "l2_limit_range": [20],  # L2订单簿深度范围
        "trades_has_history": False,  # 交易历史不支持分页
        "tickers_have_bid_ask": False,  # Ticker不包含买卖价
        "stoploss_on_exchange": False,  # 不支持交易所止损
        "exchange_has_overrides": {"fetchTrades": False},  # 覆盖fetchTrades方法
        "marketOrderRequiresPrice": True,  # 市价单需要价格
    }
    _ft_has_futures: FtHas = {
        "stoploss_on_exchange": True,  # 期货支持交易所止损
        "stoploss_order_types": {"limit": "limit"},  # 期货止损订单类型
        "stoploss_blocks_assets": False,  # 期货止损订单不锁定资产
        "stop_price_prop": "stopPrice",  # 期货止损价格属性名
        "funding_fee_timeframe": "1h",  # 资金费用时间框架
        "funding_fee_candle_limit": 500,  # 资金费用蜡烛图限制
        "uses_leverage_tiers": False,  # 不使用杠杆等级
    }

    _supported_trading_mode_margin_pairs: list[tuple[TradingMode, MarginMode]] = [
        (TradingMode.FUTURES, MarginMode.ISOLATED)  # 支持期货逐仓模式
    ]

    @property
    def _ccxt_config(self) -> dict:
        # ccxt Hyperliquid默认是swap
        config = {}
        if self.trading_mode == TradingMode.SPOT:
            config.update({"options": {"defaultType": "spot"}})
        config.update(super()._ccxt_config)
        return config

    def get_max_leverage(self, pair: str, stake_amount: float | None) -> float:
        # 没有杠杆等级
        if self.trading_mode == TradingMode.FUTURES:
            return self.markets[pair]["limits"]["leverage"]["max"]
        else:
            return 1.0

    def _lev_prep(self, pair: str, leverage: float, side: BuySell, accept_fail: bool = False):
        if self.trading_mode != TradingMode.SPOT:
            # Hyperliquid期望杠杆是整数
            leverage = int(leverage)
            # Hyperliquid需要leverage参数
            # 不使用_set_leverage()，因为这会将保证金设置回交叉模式
            self.set_margin_mode(pair, self.margin_mode, params={"leverage": leverage})

    def dry_run_liquidation_price(
        self,
        pair: str,
        open_rate: float,  # 持仓的入场价格
        is_short: bool,
        amount: float,
        stake_amount: float,
        leverage: float,
        wallet_balance: float,  # 或保证金余额
        open_trades: list,
    ) -> float | None:
        """
        优化版
        文档: https://hyperliquid.gitbook.io/hyperliquid-docs/trading/liquidations
        可以用更少的代码行完成，但这样更符合文档

        用196个独特的ccxt fetch_positions()持仓输出测试
        - 仅每个持仓中pnl=0.0的第一个输出
        - 与返回的强平价格比较
        持仓: 197个 平均偏差: 0.00028980% 最大偏差: 0.01309453%
        持仓信息:
        {'leverage': {1.0: 23, 2.0: 155, 3.0: 8, 4.0: 7, 5.0: 4},
        'side': {'long': 133, 'short': 64},
        'symbol': {'BTC/USDC:USDC': 81,
                   'DOGE/USDC:USDC': 20,
                   'ETH/USDC:USDC': 53,
                   'SOL/USDC:USDC': 43}}
        """
        # 定义/重命名变量以匹配文档
        逐仓保证金 = wallet_balance
        持仓数量 = amount
        价格 = open_rate
        持仓价值 = 价格 * 持仓数量
        最大杠杆 = self.markets[pair]["limits"]["leverage"]["max"]

        # 文档: 维持保证金是最大杠杆下初始保证金的一半，
        #       最大杠杆从3-50倍不等。换句话说，维持保证金在1%
        #       （对于50倍最大杠杆资产）到16.7%（对于3倍最大杠杆资产）之间
        #       取决于资产
        # 关键是"最大杠杆下初始保证金的一半"
        # 有点模糊，但这种解释会产生准确的结果:
        #       1. 从持仓价值开始
        #       2. 假设最大杠杆，通过将持仓价值除以最大杠杆计算初始保证金
        #       3. 再除以2
        所需维持保证金 = 持仓价值 / 最大杠杆 / 2

        # 文档: 可用保证金（逐仓）= 逐仓保证金 - 所需维持保证金
        可用保证金 = 逐仓保证金 - 所需维持保证金

        # 文档: 维持保证金是最大杠杆下初始保证金的一半
        # 文档没有明确指定维持杠杆，但这样有效
        # 乘以2是因为"最大杠杆下初始保证金的一半"的表述
        维持杠杆 = 最大杠杆 * 2

        # 文档: l = 1 / 维持杠杆 (使用'll'以符合PEP8: E741)
        ll = 1 / 维持杠杆

        # 文档: 方向 = 1表示多头，-1表示空头
        方向 = -1 if is_short else 1

        # 文档: 强平价格 = 价格 - 方向 * 可用保证金 / 持仓数量 / (1 - ll * 方向)
        强平价格 = 价格 - 方向 * 可用保证金 / 持仓数量 / (1 - ll * 方向)

        if self.trading_mode == TradingMode.FUTURES:
            return 强平价格
        else:
            raise OperationalException(
                "Freqtrade仅支持逐仓期货进行杠杆交易"
            )

    def get_funding_fees(
        self, pair: str, amount: float, is_short: bool, open_date: datetime
    ) -> float:
        """
        获取资金费用，要么从交易所（实盘）获取，要么根据资金费率/标记价格历史计算
        :param pair: 交易的报价/基础货币对
        :param is_short: 交易方向
        :param amount: 交易数量
        :param open_date: 交易开仓日期
        :return: 自开仓日期以来的资金费用
        :raises: 如果出现问题则抛出ExchangeError
        """
        # Hyperliquid没有fetchFundingHistory方法
        if self.trading_mode == TradingMode.FUTURES:
            try:
                return self._fetch_and_calculate_funding_fees(pair, amount, is_short, open_date)
            except ExchangeError:
                logger.warning(f"无法更新 {pair} 的资金费用。")
        return 0.0

    def _adjust_hyperliquid_order(
        self,
        order: dict,
    ) -> dict:
        """
        调整Hyperliquid的订单响应
        :param order: 来自Hyperliquid的订单响应
        :return: 调整后的订单响应
        """
        if (
            order["average"] is None
            and order["status"] in ("canceled", "closed")
            and order["filled"] > 0
        ):
            # Hyperliquid不会在订单响应中填写平均价格
            # 获取交易以计算平均价格，以获得订单的实际执行价格
            交易记录 = self.get_trades_for_order(
                order["id"], order["symbol"], since=dt_from_ts(order["timestamp"])
            )

            if 交易记录:
                总数量 = sum(t["amount"] for t in 交易记录)
                order["average"] = (
                    sum(t["price"] * t["amount"] for t in 交易记录) / 总数量
                    if 总数量
                    else None
                )
        return order

    def fetch_order(self, order_id: str, pair: str, params: dict | None = None) -> CcxtOrder:
        order = super().fetch_order(order_id, pair, params)

        order = self._adjust_hyperliquid_order(order)
        self._log_exchange_response("fetch_order2", order)

        return order

    def fetch_orders(
        self, pair: str, since: datetime, params: dict | None = None
    ) -> list[CcxtOrder]:
        orders = super().fetch_orders(pair, since, params)
        for idx, order in enumerate(deepcopy(orders)):
            order2 = self._adjust_hyperliquid_order(order)
            orders[idx] = order2

        self._log_exchange_response("fetch_orders2", orders)
        return orders