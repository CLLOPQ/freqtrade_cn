"""Binance 交易所子类"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import ccxt
from pandas import DataFrame

from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS
from freqtrade.enums import CandleType, MarginMode, PriceType, TradingMode
from freqtrade.exceptions import DDosProtection, OperationalException, TemporaryError
from freqtrade.exchange import Exchange
from freqtrade.exchange.binance_public_data import (
    concat_safe,
    download_archive_ohlcv,
    download_archive_trades,
)
from freqtrade.exchange.common import retrier
from freqtrade.exchange.exchange_types import FtHas, Tickers
from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_msecs
from freqtrade.misc import deep_merge_dicts, json_load
from freqtrade.util.datetime_helpers import dt_from_ts, dt_ts


logger = logging.getLogger(__name__)


class Binance(Exchange):
    _ft_has: FtHas = {
        "stoploss_on_exchange": True,
        "stop_price_param": "stopPrice",
        "stop_price_prop": "stopPrice",
        "stoploss_order_types": {"limit": "stop_loss_limit"},
        "stoploss_blocks_assets": True,  # 默认情况下止损订单会锁定资产
        "order_time_in_force": ["GTC", "FOK", "IOC", "PO"],
        "trades_pagination": "id",
        "trades_pagination_arg": "fromId",
        "trades_has_history": True,
        "fetch_orders_limit_minutes": None,
        "l2_limit_range": [5, 10, 20, 50, 100, 500, 1000],
        "ws_enabled": True,
    }
    _ft_has_futures: FtHas = {
        "funding_fee_candle_limit": 1000,
        "stoploss_order_types": {"limit": "stop", "market": "stop_market"},
        "stoploss_blocks_assets": False,  # 期货止损订单不会锁定资产
        "order_time_in_force": ["GTC", "FOK", "IOC"],
        "tickers_have_price": False,
        "floor_leverage": True,
        "fetch_orders_limit_minutes": 7 * 1440,  # "fetch_orders" 限制为7天
        "stop_price_type_field": "workingType",
        "order_props_in_contracts": ["amount", "cost", "filled", "remaining"],
        "stop_price_type_value_mapping": {
            PriceType.LAST: "CONTRACT_PRICE",
            PriceType.MARK: "MARK_PRICE",
        },
        "ws_enabled": False,
        "proxy_coin_mapping": {
            "BNFCR": "USDC",
            "BFUSD": "USDT",
        },
    }

    _supported_trading_mode_margin_pairs: list[tuple[TradingMode, MarginMode]] = [
        # TradingMode.SPOT 始终受支持，无需在此列表中
        # (TradingMode.MARGIN, MarginMode.CROSS),
        (TradingMode.FUTURES, MarginMode.CROSS),
        (TradingMode.FUTURES, MarginMode.ISOLATED),
    ]

    def get_proxy_coin(self) -> str:
        """
        获取给定币种的代理币
        如果没有找到代理币，则回退到基础货币
        :return: 代理币或基础货币
        """
        if self.margin_mode == MarginMode.CROSS:
            return self._config.get(
                "proxy_coin",
                self._config["stake_currency"],
            )  # type: ignore[return-value]
        return self._config["stake_currency"]

    def get_tickers(
        self,
        symbols: list[str] | None = None,
        *,
        cached: bool = False,
        market_type: TradingMode | None = None,
    ) -> Tickers:
        tickers = super().get_tickers(symbols=symbols, cached=cached, market_type=market_type)
        if self.trading_mode == TradingMode.FUTURES:
            # Binance 的期货结果没有买卖价。
            # 因此我们必须从 fetch_bids_asks 获取并合并这两个结果。
            bidsasks = self.fetch_bids_asks(symbols, cached=cached)
            tickers = deep_merge_dicts(bidsasks, tickers, allow_null_overrides=False)
        return tickers

    @retrier
    def additional_exchange_init(self) -> None:
        """
        额外的交易所初始化逻辑。
        此时 .api 已可用。
        如果需要，必须在子方法中重写。
        """
        try:
            if self.trading_mode == TradingMode.FUTURES and not self._config["dry_run"]:
                position_side = self._api.fapiPrivateGetPositionSideDual()
                self._log_exchange_response("position_side_setting", position_side)
                assets_margin = self._api.fapiPrivateGetMultiAssetsMargin()
                self._log_exchange_response("multi_asset_margin", assets_margin)
                msg = ""
                if position_side.get("dualSidePosition") is True:
                    msg += (
                        "\nFreqtrade 不支持对冲模式。"
                        "请在您的币安期货账户上更改'仓位模式'。"
                    )
                if (
                    assets_margin.get("multiAssetsMargin") is True
                    and self.margin_mode != MarginMode.CROSS
                ):
                    msg += (
                        "\nFreqtrade 不支持多资产模式。"
                        "请在您的币安期货账户上更改'资产模式'。"
                    )
                if msg:
                    raise OperationalException(msg)
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于 {e.__class__.__name__}，在 additional_exchange_init 中出错。消息: {e}"
            ) from e

        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def get_historic_ohlcv(
        self,
        pair: str,
        timeframe: str,
        since_ms: int,
        candle_type: CandleType,
        is_new_pair: bool = False,
        until_ms: int | None = None,
    ) -> DataFrame:
        """
        重写以引入"快速新交易对"功能，通过检测交易对的上市日期
        不适用于其他交易所，这些交易所调用"0"时不会返回最早的数据
        :param candle_type: CandleType 枚举值（必须匹配交易模式！）
        """
        if is_new_pair and candle_type in (CandleType.SPOT, CandleType.FUTURES, CandleType.MARK):
            with self._loop_lock:
                x = self.loop.run_until_complete(
                    self._async_get_candle_history(pair, timeframe, candle_type, 0)
                )
            if x and x[3] and x[3][0] and x[3][0][0] > since_ms:
                # 将起始日期设置为第一个可用的K线。
                since_ms = x[3][0][0]
                logger.info(
                    f"{pair} 的K线数据从 "
                    f"{datetime.fromtimestamp(since_ms // 1000, tz=timezone.utc).isoformat()} 开始可用。"
                )
                if until_ms and since_ms >= until_ms:
                    logger.warning(
                        f"{pair} 在 {dt_from_ts(until_ms).isoformat()} 之前没有可用的K线数据"
                    )
                    return DataFrame(columns=DEFAULT_DATAFRAME_COLUMNS)

        if (
            self._config["exchange"].get("only_from_ccxt", False)
            or
            # 只下载有显著改进的时间框架，
            # 否则回退到REST API
            not (
                (candle_type == CandleType.SPOT and timeframe in ["1s", "1m", "3m", "5m"])
                or (
                    candle_type == CandleType.FUTURES
                    and timeframe in ["1m", "3m", "5m", "15m", "30m"]
                )
            )
        ):
            return super().get_historic_ohlcv(
                pair=pair,
                timeframe=timeframe,
                since_ms=since_ms,
                candle_type=candle_type,
                is_new_pair=is_new_pair,
                until_ms=until_ms,
            )
        else:
            # 从 data.binance.vision 下载
            return self.get_historic_ohlcv_fast(
                pair=pair,
                timeframe=timeframe,
                since_ms=since_ms,
                candle_type=candle_type,
                is_new_pair=is_new_pair,
                until_ms=until_ms,
            )

    def get_historic_ohlcv_fast(
        self,
        pair: str,
        timeframe: str,
        since_ms: int,
        candle_type: CandleType,
        is_new_pair: bool = False,
        until_ms: int | None = None,
    ) -> DataFrame:
        """
        通过利用 https://data.binance.vision 快速获取 OHLCV 数据。
        """
        with self._loop_lock:
            df = self.loop.run_until_complete(
                download_archive_ohlcv(
                    candle_type=candle_type,
                    pair=pair,
                    timeframe=timeframe,
                    since_ms=since_ms,
                    until_ms=until_ms,
                    markets=self.markets,
                )
            )

        # 从REST API下载剩余数据
        if df.empty:
            rest_since_ms = since_ms
        else:
            rest_since_ms = dt_ts(df.iloc[-1].date) + timeframe_to_msecs(timeframe)

        # 确保since <= until
        if until_ms and rest_since_ms > until_ms:
            rest_df = DataFrame()
        else:
            rest_df = super().get_historic_ohlcv(
                pair=pair,
                timeframe=timeframe,
                since_ms=rest_since_ms,
                candle_type=candle_type,
                is_new_pair=is_new_pair,
                until_ms=until_ms,
            )
        all_df = concat_safe([df, rest_df])
        return all_df

    def funding_fee_cutoff(self, open_date: datetime):
        """
        资金费用仅在整点收取（通常每4-8小时）。
        因此，在10:00:01开仓的交易将在下一小时才会收取资金费用。
        在币安上，这个截止时间是15秒。
        https://github.com/freqtrade/freqtrade/pull/5779#discussion_r740175931
        :param open_date: 交易的开仓日期
        :return: 如果日期落在整点则为True，否则为False
        """
        return open_date.minute == 0 and open_date.second < 15

    def fetch_funding_rates(self, symbols: list[str] | None = None) -> dict[str, dict[str, float]]:
        """
        获取给定交易对的资金费率。
        :param symbols: 要获取资金费率的交易对列表
        :return: 给定交易对的资金费率字典
        """
        try:
            if self.trading_mode == TradingMode.FUTURES:
                rates = self._api.fetch_funding_rates(symbols)
                return rates
            return {}
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于 {e.__class__.__name__}，在 additional_exchange_init 中出错。消息: {e}"
            ) from e

        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def dry_run_liquidation_price(
        self,
        pair: str,
        open_rate: float,
        is_short: bool,
        amount: float,
        stake_amount: float,
        leverage: float,
        wallet_balance: float,
        open_trades: list,
    ) -> float | None:
        """
        重要提示：由于此方法用于回测，必须从缓存值中获取数据！
        保证金：https://www.binance.com/en/support/faq/f6b010588e55413aa58b7d63ee0125ed
        永续合约：https://www.binance.com/en/support/faq/b3c689c1f50a44cabb3a84e663b81d93

        :param pair: 计算平仓价格的交易对
        :param open_rate: 仓位的入场价格
        :param is_short: 如果交易是空头则为True，否则为False
        :param amount: 包含杠杆的仓位大小绝对值（以基础货币计）
        :param stake_amount: 保证金金额 - 结算货币的抵押品。
        :param leverage: 此仓位使用的杠杆。
        :param wallet_balance: 用于交易的钱包中的保证金金额
            全仓模式：crossWalletBalance
            逐仓模式：isolatedWalletBalance
        :param open_trades: 同一钱包中的未平仓交易列表

        # * 仅全仓需要
        :param mm_ex_1: (TMM)
            全仓模式：所有其他合约的维持保证金，不包括合约1
            逐仓模式：0
        :param upnl_ex_1: (UPNL)
            全仓模式：所有其他合约的未实现盈亏，不包括合约1。
            逐仓模式：0
        :param other
        """
        cross_vars: float = 0.0

        # mm_ratio: 币安的公式指定维持保证金率，即 mm_ratio * 100%
        # maintenance_amt: (CUM) 仓位的维持保证金金额
        mm_ratio, maintenance_amt = self.get_maintenance_ratio_and_amt(pair, stake_amount)

        if self.margin_mode == MarginMode.CROSS:
            mm_ex_1: float = 0.0
            upnl_ex_1: float = 0.0
            pairs = [trade.pair for trade in open_trades]
            if self._config["runmode"] in ("live", "dry_run"):
                funding_rates = self.fetch_funding_rates(pairs)
            for trade in open_trades:
                if trade.pair == pair:
                    # 只考虑"其他"交易
                    continue
                if self._config["runmode"] in ("live", "dry_run"):
                    mark_price = funding_rates[trade.pair]["markPrice"]
                else:
                    # 回测时回退到开仓价
                    mark_price = trade.open_rate
                mm_ratio1, maint_amnt1 = self.get_maintenance_ratio_and_amt(
                    trade.pair, trade.stake_amount
                )
                maint_margin = trade.amount * mark_price * mm_ratio1 - maint_amnt1
                mm_ex_1 += maint_margin

                upnl_ex_1 += trade.amount * mark_price - trade.amount * trade.open_rate

            cross_vars = upnl_ex_1 - mm_ex_1

        side_1 = -1 if is_short else 1

        if maintenance_amt is None:
            raise OperationalException(
                "Binance.liquidation_price 需要参数 maintenance_amt "
                f"对于 {self.trading_mode}"
            )

        if self.trading_mode == TradingMode.FUTURES:
            return (
                (wallet_balance + cross_vars + maintenance_amt) - (side_1 * amount * open_rate)
            ) / ((amount * mm_ratio) - (side_1 * amount))
        else:
            raise OperationalException(
                "Freqtrade 仅支持逐仓期货进行杠杆交易"
            )

    def load_leverage_tiers(self) -> dict[str, list[dict]]:
        if self.trading_mode == TradingMode.FUTURES:
            if self._config["dry_run"]:
                leverage_tiers_path = Path(__file__).parent / "binance_leverage_tiers.json"
                with leverage_tiers_path.open() as json_file:
                    return json_load(json_file)
            else:
                return self.get_leverage_tiers()
        else:
            return {}

    async def _async_get_trade_history_id_startup(
        self, pair: str, since: int
    ) -> tuple[list[list], str]:
        """
        初始调用的重写

        币安只提供有限的历史交易数据。
        使用 from_id=0，我们可以获取最早可用的交易。
        因此，如果使用提供的"since"没有获取到任何数据，我们可以假设
        下载所有可用数据。
        """
        t, from_id = await self._async_fetch_trades(pair, since=since)
        if not t:
            return [], "0"
        return t, from_id

    async def _async_get_trade_history_id(
        self, pair: str, until: int, since: int, from_id: str | None = None
    ) -> tuple[str, list[list]]:
        logger.info(f"从 Binance 获取交易，{from_id=}, {since=}, {until=}")

        if not self._config["exchange"].get("only_from_ccxt", False):
            if from_id is None or not since:
                trades = await self._api_async.fetch_trades(
                    pair,
                    params={
                        self._trades_pagination_arg: "0",
                    },
                    limit=5,
                )
                listing_date: int = trades[0]["timestamp"]
                since = max(since, listing_date)

            _, res = await download_archive_trades(
                CandleType.FUTURES if self.trading_mode == "futures" else CandleType.SPOT,
                pair,
                since_ms=since,
                until_ms=until,
                markets=self.markets,
            )

            if not res:
                end_time = since
                end_id = from_id
            else:
                end_time = res[-1][0]
                end_id = res[-1][1]

            if end_time and end_time >= until:
                return pair, res
            else:
                _, res2 = await super()._async_get_trade_history_id(
                    pair, until=until, since=end_time, from_id=end_id
                )
                res.extend(res2)
                return pair, res

        return await super()._async_get_trade_history_id(
            pair, until=until, since=since, from_id=from_id
        )