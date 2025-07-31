"""Bybit 交易所子类"""

import logging
from datetime import datetime, timedelta
from typing import Any

import ccxt

from freqtrade.constants import BuySell
from freqtrade.enums import MarginMode, PriceType, TradingMode
from freqtrade.exceptions import DDosProtection, ExchangeError, OperationalException, TemporaryError
from freqtrade.exchange import Exchange
from freqtrade.exchange.common import retrier
from freqtrade.exchange.exchange_types import CcxtOrder, FtHas


logger = logging.getLogger(__name__)


class Bybit(Exchange):
    """
    Bybit 交易所类。包含使 Freqtrade 能够与该交易所正常工作所需的调整。

    请注意，该交易所未包含在 Freqtrade 开发团队官方支持的交易所列表中。
    因此，某些功能可能仍无法按预期工作。
    """

    unified_account = False

    _ft_has: FtHas = {
        "ohlcv_has_history": True,  # 支持OHLCV历史数据
        "order_time_in_force": ["GTC", "FOK", "IOC", "PO"],  # 订单有效期类型
        "ws_enabled": True,  # 支持WebSocket
        "trades_has_history": False,  # 交易历史不支持分页
        "fetch_orders_limit_minutes": 7 * 1440,  # 7天
        "exchange_has_overrides": {
            # Bybit现货不支持fetch_order
            # 除非账户是统一账户模式
            # TODO: 当bybit完全强制所有账户使用统一模式后可移除
            "fetchOrder": False,
        },
    }
    _ft_has_futures: FtHas = {
        "ohlcv_has_history": True,  # 支持OHLCV历史数据
        "mark_ohlcv_timeframe": "4h",  # 标记价格OHLCV时间框架
        "funding_fee_timeframe": "8h",  # 资金费用时间框架
        "funding_fee_candle_limit": 200,  # 资金费用蜡烛图限制
        "stoploss_on_exchange": True,  # 支持交易所止损
        "stoploss_order_types": {"limit": "limit", "market": "market"},  # 止损订单类型
        "stoploss_blocks_assets": False,  # 止损订单不锁定资产
        # bybit响应解析无法填充stopLossPrice
        "stop_price_prop": "stopPrice",  # 止损价格属性
        "stop_price_type_field": "triggerBy",  # 止损价格类型字段
        "stop_price_type_value_mapping": {
            PriceType.LAST: "LastPrice",  # 最新价格
            PriceType.MARK: "MarkPrice",  # 标记价格
            PriceType.INDEX: "IndexPrice",  # 指数价格
        },
        "exchange_has_overrides": {
            "fetchOrder": True,  # 支持获取订单
        },
    }

    _supported_trading_mode_margin_pairs: list[tuple[TradingMode, MarginMode]] = [
        # TradingMode.SPOT始终受支持，无需在此列表中
        # (TradingMode.FUTURES, MarginMode.CROSS),
        (TradingMode.FUTURES, MarginMode.ISOLATED)  # 期货逐仓模式
    ]

    @property
    def _ccxt_config(self) -> dict:
        # 直接添加到ccxt同步/异步初始化的参数
        # ccxt默认使用swap模式
        config = {}
        if self.trading_mode == TradingMode.SPOT:
            config.update({"options": {"defaultType": "spot"}})
        config.update(super()._ccxt_config)
        return config

    def market_is_future(self, market: dict[str, Any]) -> bool:
        main = super().market_is_future(market)
        # 对于ByBit，我们目前只支持USDT市场
        return main and market["settle"] == "USDT"

    @retrier
    def additional_exchange_init(self) -> None:
        """
        额外的交易所初始化逻辑。
        此时.api已可用。
        如果需要，必须在子方法中重写。
        """
        try:
            if not self._config["dry_run"]:
                if self.trading_mode == TradingMode.FUTURES:
                    position_mode = self._api.set_position_mode(False)
                    self._log_exchange_response("set_position_mode", position_mode)
                is_unified = self._api.is_unified_enabled()
                # 返回布尔值元组，第一个是保证金，第二个是账户
                if is_unified and len(is_unified) > 1 and is_unified[1]:
                    self.unified_account = True
                    logger.info(
                        "Bybit: 统一账户。假设此机器人使用专用子账户。"
                    )
                else:
                    self.unified_account = False
                    logger.info("Bybit: 标准账户。")
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于 {e.__class__.__name__}，在 additional_exchange_init 中出错。消息: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def _lev_prep(self, pair: str, leverage: float, side: BuySell, accept_fail: bool = False):
        if self.trading_mode != TradingMode.SPOT:
            params = {"leverage": leverage}
            self.set_margin_mode(pair, self.margin_mode, accept_fail=True, params=params)
            self._set_leverage(leverage, pair, accept_fail=True)

    def _get_params(
        self,
        side: BuySell,
        ordertype: str,
        leverage: float,
        reduceOnly: bool,
        time_in_force: str = "GTC",
    ) -> dict:
        params = super()._get_params(
            side=side,
            ordertype=ordertype,
            leverage=leverage,
            reduceOnly=reduceOnly,
            time_in_force=time_in_force,
        )
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode:
            params["position_idx"] = 0
        return params

    def _get_stop_params(self, side: BuySell, ordertype: str, stop_price: float) -> dict:
        params = super()._get_stop_params(
            side=side,
            ordertype=ordertype,
            stop_price=stop_price,
        )
        # 解决ccxt在https://github.com/ccxt/ccxt/pull/25887中引入的bug
        # create_order不再返回ID
        params.update(
            {
                "method": "privatePostV5OrderCreate",
            }
        )
        return params

    def _order_needs_price(self, side: BuySell, ordertype: str) -> bool:
        # Bybit要求市价单提供价格 - 但仅适用于经典账户，且仅在现货模式下
        return (
            ordertype != "market"
            or (
                side == "buy" and not self.unified_account and self.trading_mode == TradingMode.SPOT
            )
            or self._ft_has.get("marketOrderRequiresPrice", False)
        )

    def dry_run_liquidation_price(
        self,
        pair: str,
        open_rate: float,  # 仓位的入场价格
        is_short: bool,
        amount: float,
        stake_amount: float,
        leverage: float,
        wallet_balance: float,  # 或保证金余额
        open_trades: list,
    ) -> float | None:
        """
        重要提示：由于此方法用于回测，必须从缓存值中获取数据！
        永续合约：
         bybit:
          https://www.bybithelp.com/HelpCenterKnowledge/bybitHC_Article?language=en_US&id=000001067
          https://www.bybit.com/en/help-center/article/Liquidation-Price-Calculation-under-Isolated-Mode-Unified-Trading-Account#b

        多头：
        强平价格 = (
            入场价格 - [(初始保证金 - 维持保证金)/合约数量]
            - (额外添加的保证金/合约数量))
        空头：
        强平价格 = (
            入场价格 + [(初始保证金 - 维持保证金)/合约数量]
            + (额外添加的保证金/合约数量))

        实现说明：目前不使用额外保证金。

        :param pair: 计算强平价格的交易对
        :param open_rate: 仓位的入场价格
        :param is_short: 如果交易是空头则为True，否则为False
        :param amount: 包含杠杆的仓位大小绝对值（以基础货币计）
        :param stake_amount: 保证金金额 - 结算货币的抵押品
        :param leverage: 此仓位使用的杠杆
        :param wallet_balance: 用于交易的钱包中的保证金金额
            全仓模式：crossWalletBalance
            逐仓模式：isolatedWalletBalance
        :param open_trades: 同一钱包中的其他未平仓交易列表
        """

        market = self.markets[pair]
        mm_ratio, _ = self.get_maintenance_ratio_and_amt(pair, stake_amount)

        if self.trading_mode == TradingMode.FUTURES and self.margin_mode == MarginMode.ISOLATED:
            if market["inverse"]:
                raise OperationalException("Freqtrade 尚不支持反向合约")
            position_value = amount * open_rate
            initial_margin = position_value / leverage
            maintenance_margin = position_value * mm_ratio
            每合约保证金差额 = (initial_margin - maintenance_margin) / amount

            # 见文档字符串 - 忽略额外保证金！
            if is_short:
                return open_rate + 每合约保证金差额
            else:
                return open_rate - 每合约保证金差额

        else:
            raise OperationalException(
                "Freqtrade 仅支持逐仓期货进行杠杆交易"
            )

    def get_funding_fees(
        self, pair: str, amount: float, is_short: bool, open_date: datetime
    ) -> float:
        """
        获取资金费用，从交易所（实盘）获取或根据资金费率/标记价格历史计算
        :param pair: 交易的报价/基础货币对
        :param is_short: 交易方向
        :param amount: 交易数量
        :param open_date: 交易的开仓日期
        :return: 自开仓日期以来的资金费用
        :raises: 如果出现问题则抛出ExchangeError
        """
        # Bybit不提供每个仓位的"已应用"资金费用
        if self.trading_mode == TradingMode.FUTURES:
            try:
                return self._fetch_and_calculate_funding_fees(pair, amount, is_short, open_date)
            except ExchangeError:
                logger.warning(f"无法更新 {pair} 的资金费用。")
        return 0.0

    def fetch_order(self, order_id: str, pair: str, params: dict | None = None) -> CcxtOrder:
        if self.exchange_has("fetchOrder"):
            # 设置acknowledged为True以避免ccxt异常
            params = {"acknowledged": True}

        order = super().fetch_order(order_id, pair, params)
        if not order:
            order = self.fetch_order_emulated(order_id, pair, {})
        if (
            order.get("status") == "canceled"
            and order.get("filled") == 0.0
            and order.get("remaining") == 0.0
        ):
            # 在bybit上，取消的订单将有"remaining=0"
            order["remaining"] = None
        return order

    @retrier
    def get_leverage_tiers(self) -> dict[str, list[dict]]:
        """
        将杠杆等级缓存1天，因为它们预计不会经常变化，并且
        bybit需要分页来获取所有等级
        """

        # 加载缓存的等级
        tiers_cached = self.load_cached_leverage_tiers(
            self._config["stake_currency"], timedelta(days=1)
        )
        if tiers_cached:
            return tiers_cached

        # 从交易所获取等级
        tiers = super().get_leverage_tiers()

        self.cache_leverage_tiers(tiers, self._config["stake_currency"])
        return tiers