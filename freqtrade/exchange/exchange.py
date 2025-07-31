# pragma pylint: disable=W0603
"""
加密货币交易所支持
"""

import asyncio
import inspect
import logging
import signal
from collections.abc import Coroutine, Generator
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from math import floor, isnan
from threading import Lock
from typing import Any, Literal, TypeGuard, TypeVar

import ccxt
import ccxt.pro as ccxt_pro
from cachetools import TTLCache
from ccxt import TICK_SIZE
from dateutil import parser
from pandas import DataFrame, concat

from freqtrade.configuration import remove_exchange_credentials
from freqtrade.constants import (
    DEFAULT_AMOUNT_RESERVE_PERCENT,
    DEFAULT_TRADES_COLUMNS,
    NON_OPEN_EXCHANGE_STATES,
    BidAsk,
    BuySell,
    Config,
    EntryExit,
    ExchangeConfig,
    ListPairsWithTimeframes,
    MakerTaker,
    OBLiteral,
    PairWithTimeframe,
)
from freqtrade.data.converter import (
    clean_ohlcv_dataframe,
    ohlcv_to_dataframe,
    trades_df_remove_duplicates,
    trades_dict_to_list,
    trades_list_to_df,
)
from freqtrade.enums import (
    OPTIMIZE_MODES,
    TRADE_MODES,
    CandleType,
    MarginMode,
    PriceType,
    RunMode,
    TradingMode,
)
from freqtrade.exceptions import (
    ConfigurationError,
    DDosProtection,
    ExchangeError,
    InsufficientFundsError,
    InvalidOrderException,
    OperationalException,
    PricingError,
    RetryableOrderError,
    TemporaryError,
)
from freqtrade.exchange.common import (
    API_FETCH_ORDER_RETRY_COUNT,
    retrier,
    retrier_async,
)
from freqtrade.exchange.exchange_types import (
    CcxtBalances,
    CcxtOrder,
    CcxtPosition,
    FtHas,
    OHLCVResponse,
    OrderBook,
    Ticker,
    Tickers,
)
from freqtrade.exchange.exchange_utils import (
    ROUND,
    ROUND_DOWN,
    ROUND_UP,
    amount_to_contract_precision,
    amount_to_contracts,
    amount_to_precision,
    contracts_to_amount,
    date_minus_candles,
    is_exchange_known_ccxt,
    market_is_active,
    price_to_precision,
)
from freqtrade.exchange.exchange_utils_timeframe import (
    timeframe_to_minutes,
    timeframe_to_msecs,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    timeframe_to_seconds,
)
from freqtrade.exchange.exchange_ws import ExchangeWS
from freqtrade.misc import (
    chunks,
    deep_merge_dicts,
    file_dump_json,
    file_load_json,
    safe_value_fallback2,
)
from freqtrade.util import dt_from_ts, dt_now
from freqtrade.util.datetime_helpers import dt_humanize_delta, dt_ts, format_ms_time
from freqtrade.util.periodic_cache import PeriodicCache


logger = logging.getLogger(__name__)

T = TypeVar("T")


class Exchange:
    # 直接添加到买入/卖出调用的参数（如同意交易协议）
    _params: dict = {}

    # 额外参数 - 添加到ccxt对象
    _ccxt_params: dict = {}

    # 指定每个交易所实现哪些选项的字典
    # 这定义了默认值，可以通过子类使用_ft_has或在配置中指定来选择性覆盖
    _ft_has_default: FtHas = {
        "stoploss_on_exchange": False,
        "stop_price_param": "stopLossPrice",  # 用于交易所止损请求
        "stop_price_prop": "stopLossPrice",  # 用于交易所止损响应解析
        "stoploss_order_types": {},
        "stoploss_blocks_assets": True,  # 默认止损订单会冻结资产
        "order_time_in_force": ["GTC"],
        "ohlcv_params": {},
        "ohlcv_has_history": True,  # 某些交易所(Kraken)不通过ohlcv提供历史数据
        "ohlcv_partial_candle": True,
        "ohlcv_require_since": False,
        # 查看 https://github.com/ccxt/ccxt/issues/10767 了解ohlcv_volume_currency的移除情况
        "ohlcv_volume_currency": "base",  # "base" 或 "quote"
        "tickers_have_quoteVolume": True,
        "tickers_have_percentage": True,
        "tickers_have_bid_ask": True,  # fetch_tickers时买价/卖价为空
        "tickers_have_price": True,
        "trades_limit": 1000,  # 单次fetch_trades调用的限制
        "trades_pagination": "time",  # 可能是 "time" 或 "id"
        "trades_pagination_arg": "since",
        "trades_has_history": False,
        "l2_limit_range": None,
        "l2_limit_range_required": True,  # 允许空的L2限制(kucoin)
        "l2_limit_upper": None,  # L2限制的上限
        "mark_ohlcv_price": "mark",
        "mark_ohlcv_timeframe": "8h",
        "funding_fee_timeframe": "8h",
        "ccxt_futures_name": "swap",
        "needs_trading_fees": False,  # 使用fetch_trading_fees缓存费用
        "order_props_in_contracts": ["amount", "filled", "remaining"],
        "fetch_orders_limit_minutes": None,  # "fetch_orders"默认没有时间限制
        # 在ccxt有误时覆盖createMarketBuyOrderRequiresPrice
        "marketOrderRequiresPrice": False,
        "exchange_has_overrides": {},  # 覆盖ccxt的"has"的字典
        "proxy_coin_mapping": {},  # 代理币映射
        # 预期格式为 {"fetchOHLCV": True} 或 {"fetchOHLCV": False}
        "ws_enabled": False,  # 对于已测试websocket支持的交易所设置为true
    }
    _ft_has: FtHas = {}
    _ft_has_futures: FtHas = {}

    _supported_trading_mode_margin_pairs: list[tuple[TradingMode, MarginMode]] = [
        # TradingMode.SPOT 始终支持，不需要在此列表中
    ]

    def __init__(
        self,
        config: Config,
        *,
        exchange_config: ExchangeConfig | None = None,
        validate: bool = True,
        load_leverage_tiers: bool = False,
    ) -> None:
        """
        使用给定的配置初始化此模块，
        执行基本验证，检查指定的交易所和交易对是否有效。
        :return: None
        """
        self._api: ccxt.Exchange
        self._api_async: ccxt_pro.Exchange
        self._ws_async: ccxt_pro.Exchange = None
        self._exchange_ws: ExchangeWS | None = None
        self._markets: dict = {}
        self._trading_fees: dict[str, Any] = {}
        self._leverage_tiers: dict[str, list[dict]] = {}
        # 锁定事件循环。这是必要的，以避免使用force*命令时的竞态条件
        # 由于资金费用获取
        self._loop_lock = Lock()
        self.loop = self._init_async_loop()
        self._config: Config = {}

        self._config.update(config)

        # 保存每个交易对最后刷新蜡烛图的时间
        self._pairs_last_refresh_time: dict[PairWithTimeframe, int] = {}
        # 最后市场刷新的时间戳
        self._last_markets_refresh: int = 0

        self._cache_lock = Lock()
        # 缓存10分钟...
        self._fetch_tickers_cache: TTLCache = TTLCache(maxsize=4, ttl=60 * 10)
        # 缓存值300秒，避免频繁轮询交易所价格
        # 缓存仅适用于RPC方法，因此开仓交易的价格仍然
        # 每次迭代刷新一次。
        # 也不应该太高，否则会在有开放订单的情况下冻结UI更新。
        self._exit_rate_cache: TTLCache = TTLCache(maxsize=100, ttl=300)
        self._entry_rate_cache: TTLCache = TTLCache(maxsize=100, ttl=300)

        # 保存蜡烛图
        self._klines: dict[PairWithTimeframe, DataFrame] = {}
        self._expiring_candle_cache: dict[tuple[str, int], PeriodicCache] = {}

        # 保存公开交易
        self._trades: dict[PairWithTimeframe, DataFrame] = {}

        # 保存dry_run的所有开放卖单
        self._dry_run_open_orders: dict[str, Any] = {}

        if config["dry_run"]:
            logger.info("实例正在以dry_run模式运行")
        logger.info(f"使用 CCXT {ccxt.__version__}")
        exchange_conf: dict[str, Any] = exchange_config if exchange_config else config["exchange"]
        remove_exchange_credentials(exchange_conf, config.get("dry_run", False))
        self.log_responses = exchange_conf.get("log_responses", False)

        # 杠杆属性
        self.trading_mode: TradingMode = config.get("trading_mode", TradingMode.SPOT)
        self.margin_mode: MarginMode = (
            MarginMode(config.get("margin_mode")) if config.get("margin_mode") else MarginMode.NONE
        )
        self.liquidation_buffer = config.get("liquidation_buffer", 0.05)

        # 深度合并ft_has和默认ft_has选项
        self._ft_has = deep_merge_dicts(self._ft_has, deepcopy(self._ft_has_default))
        if self.trading_mode == TradingMode.FUTURES:
            self._ft_has = deep_merge_dicts(self._ft_has_futures, self._ft_has)
        if exchange_conf.get("_ft_has_params"):
            self._ft_has = deep_merge_dicts(exchange_conf.get("_ft_has_params"), self._ft_has)
            logger.info("使用配置参数覆盖exchange._ft_has，结果: %s", self._ft_has)

        # 直接分配以便于访问
        self._ohlcv_partial_candle = self._ft_has["ohlcv_partial_candle"]

        self._max_trades_limit = self._ft_has["trades_limit"]

        self._trades_pagination = self._ft_has["trades_pagination"]
        self._trades_pagination_arg = self._ft_has["trades_pagination_arg"]

        # 初始化ccxt对象
        ccxt_config = self._ccxt_config
        ccxt_config = deep_merge_dicts(exchange_conf.get("ccxt_config", {}), ccxt_config)
        ccxt_config = deep_merge_dicts(exchange_conf.get("ccxt_sync_config", {}), ccxt_config)

        self._api = self._init_ccxt(exchange_conf, True, ccxt_config)

        ccxt_async_config = self._ccxt_config
        ccxt_async_config = deep_merge_dicts(
            exchange_conf.get("ccxt_config", {}), ccxt_async_config
        )
        ccxt_async_config = deep_merge_dicts(
            exchange_conf.get("ccxt_async_config", {}), ccxt_async_config
        )
        self._api_async = self._init_ccxt(exchange_conf, False, ccxt_async_config)
        _has_watch_ohlcv = self.exchange_has("watchOHLCV") and self._ft_has["ws_enabled"]
        if (
            self._config["runmode"] in TRADE_MODES
            and exchange_conf.get("enable_ws", True)
            and _has_watch_ohlcv
        ):
            self._ws_async = self._init_ccxt(exchange_conf, False, ccxt_async_config)
            self._exchange_ws = ExchangeWS(self._config, self._ws_async)

        logger.info(f'使用交易所 "{self.name}"')
        self.required_candle_call_count = 1
        # 将配置中以分钟为单位的间隔转换为秒
        self.markets_refresh_interval: int = (
            exchange_conf.get("markets_refresh_interval", 60) * 60 * 1000
        )

        if validate:
            # 初始市场加载
            self.reload_markets(True, load_leverage_tiers=False)
            self.validate_config(config)
            self._startup_candle_count: int = config.get("startup_candle_count", 0)
            self.required_candle_call_count = self.validate_required_startup_candles(
                self._startup_candle_count, config.get("timeframe", "")
            )

        if self.trading_mode != TradingMode.SPOT and load_leverage_tiers:
            self.fill_leverage_tiers()
        self.additional_exchange_init()

    def __del__(self):
        """
        析构函数 - 清理异步资源
        """
        self.close()

    def close(self):
        if self._exchange_ws:
            self._exchange_ws.cleanup()
        logger.debug("交易所对象已销毁，关闭异步循环")
        if (
            getattr(self, "_api_async", None)
            and inspect.iscoroutinefunction(self._api_async.close)
            and self._api_async.session
        ):
            logger.debug("关闭异步ccxt会话。")
            self.loop.run_until_complete(self._api_async.close())
        if (
            self._ws_async
            and inspect.iscoroutinefunction(self._ws_async.close)
            and self._ws_async.session
        ):
            logger.debug("关闭ws ccxt会话。")
            self.loop.run_until_complete(self._ws_async.close())

        if self.loop and not self.loop.is_closed():
            self.loop.close()

    def _init_async_loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

    def validate_config(self, config: Config) -> None:
        # 检查时间框架是否可用
        self.validate_timeframes(config.get("timeframe"))

        # 检查所有交易对是否可用
        self.validate_stakecurrency(config["stake_currency"])
        self.validate_ordertypes(config.get("order_types", {}))
        self.validate_order_time_in_force(config.get("order_time_in_force", {}))
        self.validate_trading_mode_and_margin_mode(self.trading_mode, self.margin_mode)
        self.validate_pricing(config["exit_pricing"])
        self.validate_pricing(config["entry_pricing"])
        self.validate_orderflow(config["exchange"])
        self.validate_freqai(config)

    def _init_ccxt(
        self, exchange_config: dict[str, Any], sync: bool, ccxt_kwargs: dict[str, Any]
    ) -> ccxt.Exchange:
        """
        使用给定配置初始化ccxt并返回有效的ccxt实例。
        """
        # 查找给定交易所名称的匹配类
        name = exchange_config["name"]
        if sync:
            ccxt_module = ccxt
        else:
            ccxt_module = ccxt_pro
            if not is_exchange_known_ccxt(name, ccxt_module):
                # 如果pro不支持该交易所，则回退到async
                import ccxt.async_support as ccxt_async

                ccxt_module = ccxt_async

        if not is_exchange_known_ccxt(name, ccxt_module):
            raise OperationalException(f"交易所 {name} 不被ccxt支持")

        ex_config = {
            "apiKey": exchange_config.get(
                "api_key", exchange_config.get("apiKey", exchange_config.get("key"))
            ),
            "secret": exchange_config.get("secret"),
            "password": exchange_config.get("password"),
            "uid": exchange_config.get("uid", ""),
            "accountId": exchange_config.get("account_id", exchange_config.get("accountId", "")),
            # DEX属性：
            "walletAddress": exchange_config.get(
                "wallet_address", exchange_config.get("walletAddress")
            ),
            "privateKey": exchange_config.get("private_key", exchange_config.get("privateKey")),
        }
        if ccxt_kwargs:
            logger.info("应用额外的ccxt配置: %s", ccxt_kwargs)
        if self._ccxt_params:
            # 在上述输出后注入静态选项，以免混淆用户。
            ccxt_kwargs = deep_merge_dicts(self._ccxt_params, deepcopy(ccxt_kwargs))
        if ccxt_kwargs:
            ex_config.update(ccxt_kwargs)
        try:
            api = getattr(ccxt_module, name.lower())(ex_config)
        except (KeyError, AttributeError) as e:
            raise OperationalException(f"交易所 {name} 不被支持") from e
        except ccxt.BaseError as e:
            raise OperationalException(f"ccxt初始化失败。原因: {e}") from e

        return api

    @property
    def _ccxt_config(self) -> dict:
        # 直接添加到ccxt同步/异步初始化的参数。
        if self.trading_mode == TradingMode.MARGIN:
            return {"options": {"defaultType": "margin"}}
        elif self.trading_mode == TradingMode.FUTURES:
            return {"options": {"defaultType": self._ft_has["ccxt_futures_name"]}}
        else:
            return {}

    @property
    def name(self) -> str:
        """交易所名称（来自ccxt）"""
        return self._api.name

    @property
    def id(self) -> str:
        """交易所ccxt id"""
        return self._api.id

    @property
    def timeframes(self) -> list[str]:
        return list((self._api.timeframes or {}).keys())

    @property
    def markets(self) -> dict[str, Any]:
        """交易所ccxt市场"""
        if not self._markets:
            logger.info("市场未加载。正在加载...")
            self.reload_markets(True)
        return self._markets

    @property
    def precisionMode(self) -> int:
        """交易所ccxt精度模式"""
        return self._api.precisionMode

    @property
    def precision_mode_price(self) -> int:
        """
        用于价格的交易所ccxt精度模式
        ccxt限制的解决方法，如果交易所的价格精度模式不同
        如果 https://github.com/ccxt/ccxt/issues/20408 修复，可能需要更新。
        """
        return self._api.precisionMode

    def additional_exchange_init(self) -> None:
        """
        额外的交易所初始化逻辑。
        此时.api将可用。
        如果需要，必须在子方法中覆盖。
        """
        pass

    def _log_exchange_response(self, endpoint: str, response, *, add_info=None) -> None:
        """记录交易所响应"""
        if self.log_responses:
            add_info_str = "" if add_info is None else f" {add_info}: "
            logger.info(f"API {endpoint}: {add_info_str}{response}")

    def ohlcv_candle_limit(
        self, timeframe: str, candle_type: CandleType, since_ms: int | None = None
    ) -> int:
        """
        交易所ohlcv蜡烛限制
        如果交易所每个时间框架有不同的限制（例如bittrex），
        则使用ohlcv_candle_limit_per_timeframe，否则回退到ohlcv_candle_limit
        :param timeframe: 要检查的时间框架
        :param candle_type: 蜡烛类型
        :param since_ms: 起始时间戳
        :return: 蜡烛限制作为整数
        """

        ccxt_val = self.features(
            "spot" if candle_type == CandleType.SPOT else "futures", "fetchOHLCV", "limit", 500
        )
        if not isinstance(ccxt_val, float | int):
            ccxt_val = 500
        fallback_val = self._ft_has.get("ohlcv_candle_limit", ccxt_val)
        if candle_type == CandleType.FUNDING_RATE:
            fallback_val = self._ft_has.get("funding_fee_candle_limit", fallback_val)
        return int(
            self._ft_has.get("ohlcv_candle_limit_per_timeframe", {}).get(
                timeframe, str(fallback_val)
            )
        )

    def get_markets(
        self,
        base_currencies: list[str] | None = None,
        quote_currencies: list[str] | None = None,
        spot_only: bool = False,
        margin_only: bool = False,
        futures_only: bool = False,
        tradable_only: bool = True,
        active_only: bool = False,
    ) -> dict[str, Any]:
        """
        返回交易所ccxt市场，如果参数中请求，
        则按基础货币和报价货币过滤。
        """
        markets = self.markets
        if not markets:
            raise OperationalException("市场未加载。")

        if base_currencies:
            markets = {k: v for k, v in markets.items() if v["base"] in base_currencies}
        if quote_currencies:
            markets = {k: v for k, v in markets.items() if v["quote"] in quote_currencies}
        if tradable_only:
            markets = {k: v for k, v in markets.items() if self.market_is_tradable(v)}
        if spot_only:
            markets = {k: v for k, v in markets.items() if self.market_is_spot(v)}
        if margin_only:
            markets = {k: v for k, v in markets.items() if self.market_is_margin(v)}
        if futures_only:
            markets = {k: v for k, v in markets.items() if self.market_is_future(v)}
        if active_only:
            markets = {k: v for k, v in markets.items() if market_is_active(v)}
        return markets

    def get_quote_currencies(self) -> list[str]:
        """
        返回支持的报价货币列表
        """
        markets = self.markets
        return sorted(set([x["quote"] for _, x in markets.items()]))

    def get_pair_quote_currency(self, pair: str) -> str:
        """返回交易对的报价货币（基础/报价:结算）"""
        return self.markets.get(pair, {}).get("quote", "")

    def get_pair_base_currency(self, pair: str) -> str:
        """返回交易对的基础货币（基础/报价:结算）"""
        return self.markets.get(pair, {}).get("base", "")

    def market_is_future(self, market: dict[str, Any]) -> bool:
        return (
            market.get(self._ft_has["ccxt_futures_name"], False) is True
            and market.get("type", False) == "swap"
            and market.get("linear", False) is True
        )

    def market_is_spot(self, market: dict[str, Any]) -> bool:
        return market.get("spot", False) is True

    def market_is_margin(self, market: dict[str, Any]) -> bool:
        return market.get("margin", False) is True

    def market_is_tradable(self, market: dict[str, Any]) -> bool:
        """
        检查市场符号是否可由Freqtrade交易。
        确保配置的模式与之一致
        """
        return (
            market.get("quote", None) is not None
            and market.get("base", None) is not None
            and (
                self.precisionMode != TICK_SIZE
                # 精度过低会导致计算错误
                or market.get("precision", {}).get("price") is None
                or market.get("precision", {}).get("price") > 1e-11
            )
            and (
                (self.trading_mode == TradingMode.SPOT and self.market_is_spot(market))
                or (self.trading_mode == TradingMode.MARGIN and self.market_is_margin(market))
                or (self.trading_mode == TradingMode.FUTURES and self.market_is_future(market))
            )
        )

    def klines(self, pair_interval: PairWithTimeframe, copy: bool = True) -> DataFrame:
        if pair_interval in self._klines:
            return self._klines[pair_interval].copy() if copy else self._klines[pair_interval]
        else:
            return DataFrame()

    def trades(self, pair_interval: PairWithTimeframe, copy: bool = True) -> DataFrame:
        if pair_interval in self._trades:
            if copy:
                return self._trades[pair_interval].copy()
            else:
                return self._trades[pair_interval]
        else:
            return DataFrame(columns=DEFAULT_TRADES_COLUMNS)

    def get_contract_size(self, pair: str) -> float | None:
        if self.trading_mode == TradingMode.FUTURES:
            market = self.markets.get(pair, {})
            contract_size: float = 1.0
            if not market:
                return None
            if market.get("contractSize") is not None:
                # ccxt在市场中将contractSize作为字符串
                contract_size = float(market["contractSize"])
            return contract_size
        else:
            return 1

    def _trades_contracts_to_amount(self, trades: list) -> list:
        if len(trades) > 0 and "symbol" in trades[0]:
            contract_size = self.get_contract_size(trades[0]["symbol"])
            if contract_size != 1:
                for trade in trades:
                    trade["amount"] = trade["amount"] * contract_size
        return trades

    def _order_contracts_to_amount(self, order: CcxtOrder) -> CcxtOrder:
        if "symbol" in order and order["symbol"] is not None:
            contract_size = self.get_contract_size(order["symbol"])
            if contract_size != 1:
                for prop in self._ft_has.get("order_props_in_contracts", []):
                    if prop in order and order[prop] is not None:
                        order[prop] = order[prop] * contract_size
        return order

    def _amount_to_contracts(self, pair: str, amount: float) -> float:
        contract_size = self.get_contract_size(pair)
        return amount_to_contracts(amount, contract_size)

    def _contracts_to_amount(self, pair: str, num_contracts: float) -> float:
        contract_size = self.get_contract_size(pair)
        return contracts_to_amount(num_contracts, contract_size)

    def amount_to_contract_precision(self, pair: str, amount: float) -> float:
        """
        amount_to_contract_precision的辅助包装器
        """
        contract_size = self.get_contract_size(pair)

        return amount_to_contract_precision(
            amount, self.get_precision_amount(pair), self.precisionMode, contract_size
        )

    def ws_connection_reset(self):
        """
        定期调用以重置websocket连接
        """
        if self._exchange_ws:
            self._exchange_ws.reset_connections()

    async def _api_reload_markets(self, reload: bool = False) -> dict:
        try:
            await self._api_async.load_markets(reload=reload, params={})
            return self._api_async.markets  # 添加这一行返回市场数据
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于{e.__class__.__name__}在reload_markets中出错。消息: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise TemporaryError(e) from e

    def _load_async_markets(self, reload: bool = False) -> None:
        try:
            with self._loop_lock:
                markets = self.loop.run_until_complete(self._api_reload_markets(reload=reload))
            print(f'测试{markets}')
            if isinstance(markets, Exception):
                raise markets
            return None
        except asyncio.TimeoutError as e:
            logger.warning("无法加载市场。原因: %s", e)
            raise TemporaryError from e

    def reload_markets(self, force: bool = False, *, load_leverage_tiers: bool = True) -> None:
        """
        如果刷新间隔已过，则重新加载/初始化同步和异步市场

        """
        # 检查是否需要重新加载市场
        is_initial = self._last_markets_refresh == 0
        if (
            not force
            and self._last_markets_refresh > 0
            and (self._last_markets_refresh + self.markets_refresh_interval > dt_ts())
        ):
            return None
        logger.debug("执行定时市场重新加载..")
        try:
            # 初始加载时，我们重试3次以确保获得市场
            retries: int = 3 if force else 0
            # 重新加载异步市场，然后将其分配给同步api
            retrier(self._load_async_markets, retries=retries)(reload=True)
            self._markets = self._api_async.markets
            self._api.set_markets(self._api_async.markets, self._api_async.currencies)
            # 分配选项数组，因为它包含一些来自交易所的临时信息。
            self._api.options = self._api_async.options
            if self._exchange_ws:
                # 设置市场以避免在websocket api上重新加载
                self._ws_async.set_markets(self._api.markets, self._api.currencies)
                self._ws_async.options = self._api.options
            self._last_markets_refresh = dt_ts()

            if is_initial and self._ft_has["needs_trading_fees"]:
                self._trading_fees = self.fetch_trading_fees()

            if load_leverage_tiers and self.trading_mode == TradingMode.FUTURES:
                self.fill_leverage_tiers()
        except (ccxt.BaseError, TemporaryError):
            logger.exception("无法加载市场。")

    def validate_stakecurrency(self, stake_currency: str) -> None:
        """
        根据交易所可用货币检查权益货币。
        仅在启动时运行。如果市场未加载，则与交易所的连接存在问题。
        :param stake_currency: 要验证的权益货币
        :raise: 如果权益货币不可用，则引发OperationalException。
        """
        if not self._markets:
            raise OperationalException(
                "无法加载市场，因此无法启动。"
                "请调查上述错误以获取更多详细信息。"
            )
        quote_currencies = self.get_quote_currencies()
        if stake_currency not in quote_currencies:
            raise ConfigurationError(
                f"{stake_currency} 在 {self.name} 上不可作为权益货币。"
                f"可用货币为: {', '.join(quote_currencies)}"
            )

    def get_valid_pair_combination(self, curr_1: str, curr_2: str) -> Generator[str, None, None]:
        """
        通过尝试两种组合来获取curr_1和curr_2的有效交易对组合。
        """
        yielded = False
        for pair in (
            f"{curr_1}/{curr_2}",
            f"{curr_2}/{curr_1}",
            f"{curr_1}/{curr_2}:{curr_2}",
            f"{curr_2}/{curr_1}:{curr_1}",
        ):
            if pair in self.markets and self.markets[pair].get("active"):
                yielded = True
                yield pair
        if not yielded:
            raise ValueError(f"无法组合 {curr_1} 和 {curr_2} 来获得有效的交易对。")

    def validate_timeframes(self, timeframe: str | None) -> None:
        """
        检查配置中的时间框架是否是交易所支持的时间框架
        """
        if not hasattr(self._api, "timeframes") or self._api.timeframes is None:
            # 如果缺少timeframes属性（或为None），交易所可能
            # 没有fetchOHLCV方法。
            # 因此我们也显示这一点。
            raise OperationalException(
                f"ccxt库不提供交易所 {self.name} 的时间框架列表，"
                f"因此不支持此交易所。ccxt fetchOHLCV: {self.exchange_has('fetchOHLCV')}"
            )

        if timeframe and (timeframe not in self.timeframes):
            raise ConfigurationError(
                f"无效的时间框架 '{timeframe}'。此交易所支持: {self.timeframes}"
            )

        if (
            timeframe
            and self._config["runmode"] != RunMode.UTIL_EXCHANGE
            and timeframe_to_minutes(timeframe) < 1
        ):
            raise ConfigurationError("Freqtrade目前不支持小于1分钟的时间框架。")

    def validate_ordertypes(self, order_types: dict) -> None:
        """
        检查策略/配置中配置的订单类型是否受支持
        """
        if any(v == "market" for k, v in order_types.items()):
            if not self.exchange_has("createMarketOrder"):
                raise ConfigurationError(f"交易所 {self.name} 不支持市价单。")
        self.validate_stop_ordertypes(order_types)

    def validate_stop_ordertypes(self, order_types: dict) -> None:
        """
        验证止损订单类型
        """
        if order_types.get("stoploss_on_exchange") and not self._ft_has.get(
            "stoploss_on_exchange", False
        ):
            raise ConfigurationError(f"{self.name} 不支持交易所止损。")
        if self.trading_mode == TradingMode.FUTURES:
            price_mapping = self._ft_has.get("stop_price_type_value_mapping", {}).keys()
            if (
                order_types.get("stoploss_on_exchange", False) is True
                and "stoploss_price_type" in order_types
                and order_types["stoploss_price_type"] not in price_mapping
            ):
                raise ConfigurationError(
                    f"{self.name} 不支持交易所止损价格类型。"
                )

    def validate_pricing(self, pricing: dict) -> None:
        if pricing.get("use_order_book", False) and not self.exchange_has("fetchL2OrderBook"):
            raise ConfigurationError(f"{self.name} 的订单簿不可用。")
        if not pricing.get("use_order_book", False) and (
            not self.exchange_has("fetchTicker") or not self._ft_has["tickers_have_price"]
        ):
            raise ConfigurationError(f"{self.name} 的行情定价不可用。")

    def validate_order_time_in_force(self, order_time_in_force: dict) -> None:
        """
        检查策略/配置中配置的订单有效时间是否受支持
        """
        if any(
            v.upper() not in self._ft_has["order_time_in_force"]
            for k, v in order_time_in_force.items()
        ):
            raise ConfigurationError(
                f"{self.name} 尚不支持有效时间策略。"
            )

    def validate_orderflow(self, exchange: dict) -> None:
        if exchange.get("use_public_trades", False) and (
            not self.exchange_has("fetchTrades") or not self._ft_has["trades_has_history"]
        ):
            raise ConfigurationError(
                f"{self.name} 的交易数据不可用。无法使用订单流功能。"
            )

    def validate_freqai(self, config: Config) -> None:
        freqai_enabled = config.get("freqai", {}).get("enabled", False)
        if freqai_enabled and not self._ft_has["ohlcv_has_history"]:
            raise ConfigurationError(
                f"{self.name} 的历史OHLCV数据不可用。无法使用freqAI。"
            )

    def validate_required_startup_candles(self, startup_candles: int, timeframe: str) -> int:
        """
        检查所需的startup_candles是否超过ohlcv_candle_limit()。
        需要5根蜡烛的宽限期 - 因此默认允许最多494的启动期。
        """

        candle_limit = self.ohlcv_candle_limit(
            timeframe,
            self._config["candle_type_def"],
            dt_ts(date_minus_candles(timeframe, startup_candles)) if timeframe else None,
        )
        # 需要多一根蜡烛 - 以考虑仍然开放的蜡烛。
        candle_count = startup_candles + 1
        # 每个交易对允许5次调用交易所
        required_candle_call_count = int(
            (candle_count / candle_limit) + (0 if candle_count % candle_limit == 0 else 1)
        )
        if self._ft_has["ohlcv_has_history"]:
            if required_candle_call_count > 5:
                # 每个交易对仅允许5次调用，以在某种程度上限制影响
                raise ConfigurationError(
                    f"此策略需要 {startup_candles} 根蜡烛才能启动，"
                    "这超过了5倍 "
                    f"{self.name} 为 {timeframe} 提供的蜡烛数量。"
                )
        elif required_candle_call_count > 1:
            raise ConfigurationError(
                f"此策略需要 {startup_candles} 根蜡烛才能启动，这超过了 "
                f"{self.name} 为 {timeframe} 提供的蜡烛数量。"
            )
        if required_candle_call_count > 1:
            logger.warning(
                f"使用 {required_candle_call_count} 次调用来获取OHLCV。"
                f"这可能导致机器人操作变慢。请检查 "
                f"您的策略是否真的需要 {startup_candles} 根蜡烛"
            )
        return required_candle_call_count

    def validate_trading_mode_and_margin_mode(
        self,
        trading_mode: TradingMode,
        margin_mode: MarginMode | None,  # 仅当trading_mode = TradingMode.SPOT时为None
    ):
        """
        检查freqtrade是否可以使用配置的
        交易模式（保证金，期货）和保证金模式（全仓，逐仓）执行交易
        抛出OperationalException：
            如果freqtrade在此交易所不支持交易模式/保证金模式类型
        """
        if trading_mode != TradingMode.SPOT and (
            (trading_mode, margin_mode) not in self._supported_trading_mode_margin_pairs
        ):
            mm_value = margin_mode and margin_mode.value
            raise ConfigurationError(
                f"Freqtrade 在 {self.name} 上不支持 '{mm_value}' '{trading_mode}'。"
            )

    def get_option(self, param: str, default: Any | None = None) -> Any:
        """
        从_ft_has获取参数值
        """
        return self._ft_has.get(param, default)

    def exchange_has(self, endpoint: str) -> bool:
        """
        检查交易所是否实现了特定的API端点。
        ccxt 'has'属性的包装器
        :param endpoint: 端点名称（例如'fetchOHLCV'，'fetchTickers'）
        :return: bool
        """
        if endpoint in self._ft_has.get("exchange_has_overrides", {}):
            return self._ft_has["exchange_has_overrides"][endpoint]
        return endpoint in self._api_async.has and self._api_async.has[endpoint]

    def features(
        self, market_type: Literal["spot", "futures"], endpoint, attribute, default: T
    ) -> T:
        """
        返回给定市场类型的交易所功能
        https://docs.ccxt.com/#/README?id=features
        属性在嵌套字典中，包含spot和swap.linear
        例如 spot.fetchOHLCV.limit
             swap.linear.fetchOHLCV.limit
        """
        feat = (
            self._api_async.features.get("spot", {})
            if market_type == "spot"
            else self._api_async.features.get("swap", {}).get("linear", {})
        )

        return feat.get(endpoint, {}).get(attribute, default)

    def get_precision_amount(self, pair: str) -> float | None:
        """
        返回交易所的数量精度。
        :param pair: 获取精度的交易对
        :return: 数量精度或None。必须与precisionMode结合使用
        """
        return self.markets.get(pair, {}).get("precision", {}).get("amount", None)

    def get_precision_price(self, pair: str) -> float | None:
        """
        返回交易所的价格精度。
        :param pair: 获取精度的交易对
        :return: 价格精度或None。必须与precisionMode结合使用
        """
        return self.markets.get(pair, {}).get("precision", {}).get("price", None)

    def amount_to_precision(self, pair: str, amount: float) -> float:
        """
        返回交易所接受的买入或卖出数量精度

        """
        return amount_to_precision(amount, self.get_precision_amount(pair), self.precisionMode)

    def price_to_precision(self, pair: str, price: float, *, rounding_mode: int = ROUND) -> float:
        """
        返回四舍五入到交易所接受的精度的价格。
        配置中的默认price_rounding_mode是ROUND。
        对于止损计算，多头必须使用ROUND_UP，空头必须使用ROUND_DOWN。
        """
        return price_to_precision(
            price,
            self.get_precision_price(pair),
            self.precision_mode_price,
            rounding_mode=rounding_mode,
        )

    def price_get_one_pip(self, pair: str, price: float) -> float:
        """
        获取此交易对的"1 pip"值。
        在PriceFilter中用于计算1pip移动。
        """
        precision = self.markets[pair]["precision"]["price"]
        if self.precisionMode == TICK_SIZE:
            return precision
        else:
            return 1 / pow(10, precision)

    def get_min_pair_stake_amount(
        self, pair: str, price: float, stoploss: float, leverage: float = 1.0
    ) -> float | None:
        return self._get_stake_amount_limit(pair, price, stoploss, "min", leverage)

    def get_max_pair_stake_amount(self, pair: str, price: float, leverage: float = 1.0) -> float:
        max_stake_amount = self._get_stake_amount_limit(pair, price, 0.0, "max", leverage)
        if max_stake_amount is None:
            # * 不应该执行
            raise OperationalException(
                f"{self.name}.get_max_pair_stake_amount 不应该将max_stake_amount设置为None"
            )
        return max_stake_amount

    def _get_stake_amount_limit(
        self,
        pair: str,
        price: float,
        stoploss: float,
        limit: Literal["min", "max"],
        leverage: float = 1.0,
    ) -> float | None:
        isMin = limit == "min"

        try:
            market = self.markets[pair]
        except KeyError:
            raise ValueError(f"无法获取符号 {pair} 的市场信息")

        stake_limits = []
        limits = market["limits"]
        if isMin:
            # 预留配置中定义的一些百分比（默认5%）+ 止损
            margin_reserve: float = 1.0 + self._config.get(
                "amount_reserve_percent", DEFAULT_AMOUNT_RESERVE_PERCENT
            )
            stoploss_reserve = margin_reserve / (1 - abs(stoploss)) if abs(stoploss) != 1 else 1.5
            # 不应超过50%
            stoploss_reserve = max(min(stoploss_reserve, 1.5), 1)
        else:
            # is_max
            margin_reserve = 1.0
            stoploss_reserve = 1.0
            if max_from_tiers := self._get_max_notional_from_tiers(pair, leverage=leverage):
                stake_limits.append(max_from_tiers)

        if limits["cost"][limit] is not None:
            stake_limits.append(
                self._contracts_to_amount(pair, limits["cost"][limit]) * stoploss_reserve
            )

        if limits["amount"][limit] is not None:
            stake_limits.append(
                self._contracts_to_amount(pair, limits["amount"][limit]) * price * margin_reserve
            )

        if not stake_limits:
            return None if isMin else float("inf")

        # 返回的值应满足两个限制：数量（基础货币）和
        # 成本（报价，权益货币），因此这里使用max()。
        # 另请参见github上的#2575。
        return self._get_stake_amount_considering_leverage(
            max(stake_limits) if isMin else min(stake_limits), leverage or 1.0
        )

    def _get_stake_amount_considering_leverage(self, stake_amount: float, leverage: float) -> float:
        """
        获取没有杠杆的交易对的最小权益金额，并返回
        考虑杠杆时的最小权益金额
        :param stake_amount: 考虑杠杆之前的交易对权益金额
        :param leverage: 当前交易使用的杠杆金额
        """
        return stake_amount / leverage

    # 模拟运行方法

    def create_dry_run_order(
        self,
        pair: str,
        ordertype: str,
        side: BuySell,
        amount: float,
        rate: float,
        leverage: float,
        params: dict | None = None,
        stop_loss: bool = False,
    ) -> CcxtOrder:
        now = dt_now()
        order_id = f"dry_run_{side}_{pair}_{now.timestamp()}"
        # 这里的四舍五入必须考虑合约大小
        _amount = self._contracts_to_amount(
            pair, self.amount_to_precision(pair, self._amount_to_contracts(pair, amount))
        )
        dry_order: CcxtOrder = {
            "id": order_id,
            "symbol": pair,
            "price": rate,
            "average": rate,
            "amount": _amount,
            "cost": _amount * rate,
            "type": ordertype,
            "side": side,
            "filled": 0,
            "remaining": _amount,
            "datetime": now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "timestamp": dt_ts(now),
            "status": "open",
            "fee": None,
            "info": {},
        }
        if stop_loss:
            dry_order["info"] = {"stopPrice": dry_order["price"]}
            dry_order[self._ft_has["stop_price_prop"]] = dry_order["price"]
            # 解决方法以避免立即填充止损订单
            dry_order["ft_order_type"] = "stoploss"
        orderbook: OrderBook | None = None
        if self.exchange_has("fetchL2OrderBook"):
            orderbook = self.fetch_l2_order_book(pair, 20)
        if ordertype == "limit" and orderbook:
            # 允许1%的价格差异
            allowed_diff = 0.01
            if self._dry_is_price_crossed(pair, side, rate, orderbook, allowed_diff):
                logger.info(
                    f"由于价格 {rate} 跨越价差超过 {allowed_diff:.2%}，"
                    f"将订单 {pair} 转换为市价单。"
                )
                dry_order["type"] = "market"

        if dry_order["type"] == "market" and not dry_order.get("ft_order_type"):
            # 更新市价单定价
            average = self.get_dry_market_fill_price(pair, side, amount, rate, orderbook)
            dry_order.update(
                {
                    "average": average,
                    "filled": _amount,
                    "remaining": 0.0,
                    "status": "closed",
                    "cost": (_amount * average),
                }
            )
            # 市价单总是产生taker费用
            dry_order = self.add_dry_order_fee(pair, dry_order, "taker")

        dry_order = self.check_dry_limit_order_filled(
            dry_order, immediate=True, orderbook=orderbook
        )

        self._dry_run_open_orders[dry_order["id"]] = dry_order
        # 复制订单并关闭它 - 因此返回的订单是开放的，除非它是市价单
        return dry_order

    def add_dry_order_fee(
        self,
        pair: str,
        dry_order: CcxtOrder,
        taker_or_maker: MakerTaker,
    ) -> CcxtOrder:
        fee = self.get_fee(pair, taker_or_maker=taker_or_maker)
        dry_order.update(
            {
                "fee": {
                    "currency": self.get_pair_quote_currency(pair),
                    "cost": dry_order["cost"] * fee,
                    "rate": fee,
                }
            }
        )
        return dry_order

    def get_dry_market_fill_price(
        self, pair: str, side: str, amount: float, rate: float, orderbook: OrderBook | None
    ) -> float:
        """
        基于订单簿插值获取市价单成交价格
        """
        if self.exchange_has("fetchL2OrderBook"):
            if not orderbook:
                orderbook = self.fetch_l2_order_book(pair, 20)
            ob_type: OBLiteral = "asks" if side == "buy" else "bids"
            slippage = 0.05
            max_slippage_val = rate * ((1 + slippage) if side == "buy" else (1 - slippage))

            remaining_amount = amount
            filled_value = 0.0
            book_entry_price = 0.0
            for book_entry in orderbook[ob_type]:
                book_entry_price = book_entry[0]
                book_entry_coin_volume = book_entry[1]
                if remaining_amount > 0:
                    if remaining_amount < book_entry_coin_volume:
                        # 此位置的订单簿大于剩余数量
                        filled_value += remaining_amount * book_entry_price
                        break
                    else:
                        filled_value += book_entry_coin_volume * book_entry_price
                    remaining_amount -= book_entry_coin_volume
                else:
                    break
            else:
                # 如果remaining_amount没有完全消耗（没有调用break）
                filled_value += remaining_amount * book_entry_price
            forecast_avg_filled_price = max(filled_value, 0) / amount
            # 将最大滑点限制为指定值
            if side == "buy":
                forecast_avg_filled_price = min(forecast_avg_filled_price, max_slippage_val)

            else:
                forecast_avg_filled_price = max(forecast_avg_filled_price, max_slippage_val)

            return self.price_to_precision(pair, forecast_avg_filled_price)

        return rate

    def _dry_is_price_crossed(
        self,
        pair: str,
        side: str,
        limit: float,
        orderbook: OrderBook | None = None,
        offset: float = 0.0,
    ) -> bool:
        if not self.exchange_has("fetchL2OrderBook"):
            return True
        if not orderbook:
            orderbook = self.fetch_l2_order_book(pair, 1)
        try:
            if side == "buy":
                price = orderbook["asks"][0][0]
                if limit * (1 - offset) >= price:
                    return True
            else:
                price = orderbook["bids"][0][0]
                if limit * (1 + offset) <= price:
                    return True
        except IndexError:
            # 填充时忽略空订单簿 - 可以在下一次迭代中填充。
            pass
        return False

    def check_dry_limit_order_filled(
        self, order: CcxtOrder, immediate: bool = False, orderbook: OrderBook | None = None
    ) -> CcxtOrder:
        """
        检查模拟限价单成交并更新费用（如果成交）。
        """
        if (
            order["status"] != "closed"
            and order["type"] in ["limit"]
            and not order.get("ft_order_type")
        ):
            pair = order["symbol"]
            if self._dry_is_price_crossed(pair, order["side"], order["price"], orderbook):
                order.update(
                    {
                        "status": "closed",
                        "filled": order["amount"],
                        "remaining": 0,
                    }
                )

                self.add_dry_order_fee(
                    pair,
                    order,
                    "taker" if immediate else "maker",
                )

        return order

    def fetch_dry_run_order(self, order_id) -> CcxtOrder:
        """
        返回模拟订单
        仅在模拟模式下调用。
        """
        try:
            order = self._dry_run_open_orders[order_id]
            order = self.check_dry_limit_order_filled(order)
            return order
        except KeyError as e:
            from freqtrade.persistence import Order

            order = Order.order_by_id(order_id)
            if order:
                ccxt_order = order.to_ccxt_object(self._ft_has["stop_price_prop"])
                self._dry_run_open_orders[order_id] = ccxt_order
                return ccxt_order
            # 优雅地处理模拟订单的错误。
            raise InvalidOrderException(
                f"尝试获取无效的模拟订单（id: {order_id}）。消息: {e}"
            ) from e

    # 订单处理

    def _lev_prep(self, pair: str, leverage: float, side: BuySell, accept_fail: bool = False):
        if self.trading_mode != TradingMode.SPOT:
            self.set_margin_mode(pair, self.margin_mode, accept_fail)
            self._set_leverage(leverage, pair, accept_fail)

    def _get_params(
        self,
        side: BuySell,
        ordertype: str,
        leverage: float,
        reduceOnly: bool,
        time_in_force: str = "GTC",
    ) -> dict:
        params = self._params.copy()
        if time_in_force != "GTC" and ordertype != "market":
            params.update({"timeInForce": time_in_force.upper()})
        if reduceOnly:
            params.update({"reduceOnly": True})
        return params

    def _order_needs_price(self, side: BuySell, ordertype: str) -> bool:
        return (
            ordertype != "market"
            or (side == "buy" and self._api.options.get("createMarketBuyOrderRequiresPrice", False))
            or self._ft_has.get("marketOrderRequiresPrice", False)
        )

    def create_order(
        self,
        *,
        pair: str,
        ordertype: str,
        side: BuySell,
        amount: float,
        rate: float,
        leverage: float,
        reduceOnly: bool = False,
        time_in_force: str = "GTC",
    ) -> CcxtOrder:
        if self._config["dry_run"]:
            dry_order = self.create_dry_run_order(
                pair, ordertype, side, amount, self.price_to_precision(pair, rate), leverage
            )
            return dry_order

        params = self._get_params(side, ordertype, leverage, reduceOnly, time_in_force)

        try:
            # 设置交易所接受的数量和价格（费率）精度
            amount = self.amount_to_precision(pair, self._amount_to_contracts(pair, amount))
            needs_price = self._order_needs_price(side, ordertype)
            rate_for_order = self.price_to_precision(pair, rate) if needs_price else None

            if not reduceOnly:
                self._lev_prep(pair, leverage, side)

            order = self._api.create_order(
                pair,
                ordertype,
                side,
                amount,
                rate_for_order,
                params,
            )
            if order.get("status") is None:
                # 将空状态映射为开放。
                order["status"] = "open"

            if order.get("type") is None:
                order["type"] = ordertype

            self._log_exchange_response("create_order", order)
            order = self._order_contracts_to_amount(order)
            return order

        except ccxt.InsufficientFunds as e:
            raise InsufficientFundsError(
                f"资金不足，无法在市场 {pair} 上创建 {ordertype} {side} 订单。"
                f"尝试以价格 {rate} {side} 数量 {amount}。"
                f"消息: {e}"
            ) from e
        except ccxt.InvalidOrder as e:
            raise InvalidOrderException(
                f"无法在市场 {pair} 上创建 {ordertype} {side} 订单。"
                f"尝试以价格 {rate} {side} 数量 {amount}。"
                f"消息: {e}"
            ) from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于 {e.__class__.__name__} 无法下 {side} 订单。消息: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def stoploss_adjust(self, stop_loss: float, order: CcxtOrder, side: str) -> bool:
        """
        根据止损订单值（限价或价格）验证stop_loss
        如果需要调整，返回True。
        """
        if not self._ft_has.get("stoploss_on_exchange"):
            raise OperationalException(f"{self.name} 未实现止损。")
        price_param = self._ft_has["stop_price_prop"]
        return order.get(price_param, None) is None or (
            (side == "sell" and stop_loss > float(order[price_param]))
            or (side == "buy" and stop_loss < float(order[price_param]))
        )

    def _get_stop_order_type(self, user_order_type) -> tuple[str, str]:
        available_order_Types: dict[str, str] = self._ft_has["stoploss_order_types"]

        if user_order_type in available_order_Types.keys():
            ordertype = available_order_Types[user_order_type]
        else:
            # 否则只选择一个可用的
            ordertype = next(iter(available_order_Types.values()))
            user_order_type = next(iter(available_order_Types.keys()))
        return ordertype, user_order_type

    def _get_stop_limit_rate(self, stop_price: float, order_types: dict, side: str) -> float:
        # 限价阈值：由于限价应始终低于止损价
        limit_price_pct = order_types.get("stoploss_on_exchange_limit_ratio", 0.99)
        if side == "sell":
            limit_rate = stop_price * limit_price_pct
        else:
            limit_rate = stop_price * (2 - limit_price_pct)

        bad_stop_price = (stop_price < limit_rate) if side == "sell" else (stop_price > limit_rate)
        # 确保价格低于止损价
        if bad_stop_price:
            # 例如，如果止损/清算价格设置为0，就会发生这种情况
            # 如果市价单立即关闭，这是可能的。
            # InvalidOrderException将冒泡到exit_positions，在那里它将被
            # 优雅地处理。
            raise InvalidOrderException(
                "在止损限价订单中，止损价应高于限价。"
                f"止损价: {stop_price}，限价: {limit_rate}，"
                f"限价百分比: {limit_price_pct}"
            )
        return limit_rate

    def _get_stop_params(self, side: BuySell, ordertype: str, stop_price: float) -> dict:
        params = self._params.copy()
        # 验证stopPrice是否适用于您的交易所，否则配置stop_price_param
        params.update({self._ft_has["stop_price_param"]: stop_price})
        return params

    @retrier(retries=0)
    def create_stoploss(
        self,
        pair: str,
        amount: float,
        stop_price: float,
        order_types: dict,
        side: BuySell,
        leverage: float,
    ) -> CcxtOrder:
        """
        创建止损订单。
        需要将`_ft_has['stoploss_order_types']`设置为字典，将限价和市价映射到
            相应的交易所类型。

        精确的订单类型由order_types字典或交易所默认值确定。

        下面的异常不应该引发，因为我们在validate_ordertypes()中
        不允许启动机器人

        这可能适用于有限数量的其他交易所，但正确的工作
            需要单独测试。
        警告：将`stoploss_on_exchange`设置为True不会自动启用交易所止损。
            `stoploss_adjust`仍然必须实现才能使其工作。
        """
        if not self._ft_has["stoploss_on_exchange"]:
            raise OperationalException(f"{self.name} 未实现止损。")

        user_order_type = order_types.get("stoploss", "market")
        ordertype, user_order_type = self._get_stop_order_type(user_order_type)
        round_mode = ROUND_DOWN if side == "buy" else ROUND_UP
        stop_price_norm = self.price_to_precision(pair, stop_price, rounding_mode=round_mode)
        limit_rate = None
        if user_order_type == "limit":
            limit_rate = self._get_stop_limit_rate(stop_price, order_types, side)
            limit_rate = self.price_to_precision(pair, limit_rate, rounding_mode=round_mode)

        if self._config["dry_run"]:
            dry_order = self.create_dry_run_order(
                pair,
                ordertype,
                side,
                amount,
                stop_price_norm,
                stop_loss=True,
                leverage=leverage,
            )
            return dry_order

        try:
            params = self._get_stop_params(
                side=side, ordertype=ordertype, stop_price=stop_price_norm
            )
            if self.trading_mode == TradingMode.FUTURES:
                params["reduceOnly"] = True
                if "stoploss_price_type" in order_types and "stop_price_type_field" in self._ft_has:
                    price_type = self._ft_has["stop_price_type_value_mapping"][
                        order_types.get("stoploss_price_type", PriceType.LAST)
                    ]
                    params[self._ft_has["stop_price_type_field"]] = price_type

            amount = self.amount_to_precision(pair, self._amount_to_contracts(pair, amount))

            self._lev_prep(pair, leverage, side, accept_fail=True)
            order = self._api.create_order(
                symbol=pair,
                type=ordertype,
                side=side,
                amount=amount,
                price=limit_rate,
                params=params,
            )
            self._log_exchange_response("create_stoploss_order", order)
            order = self._order_contracts_to_amount(order)
            logger.info(
                f"为 {pair} 添加了止损 {user_order_type} 订单。"
                f"止损价: {stop_price}。限价: {limit_rate}"
            )
            return order
        except ccxt.InsufficientFunds as e:
            raise InsufficientFundsError(
                f"资金不足，无法在市场 {pair} 上创建 {ordertype} {side} 订单。"
                f"尝试以价格 {limit_rate} {side} 数量 {amount}，"
                f"止损价 {stop_price_norm}。消息: {e}"
            ) from e
        except (ccxt.InvalidOrder, ccxt.BadRequest, ccxt.OperationRejected) as e:
            # 错误：
            # `订单将立即触发。`
            raise InvalidOrderException(
                f"无法在市场 {pair} 上创建 {ordertype} {side} 订单。"
                f"尝试以价格 {limit_rate} {side} 数量 {amount}，"
                f"止损价 {stop_price_norm}。消息: {e}"
            ) from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于 {e.__class__.__name__} 无法下止损订单。消息: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def fetch_order_emulated(self, order_id: str, pair: str, params: dict) -> CcxtOrder:
        """
        如果交易所不支持fetch_order，但需要单独调用
        开放和已关闭订单，则模拟fetch_order。
        """
        try:
            order = self._api.fetch_open_order(order_id, pair, params=params)
            self._log_exchange_response("fetch_open_order", order)
            order = self._order_contracts_to_amount(order)
            return order
        except ccxt.OrderNotFound:
            try:
                order = self._api.fetch_closed_order(order_id, pair, params=params)
                self._log_exchange_response("fetch_closed_order", order)
                order = self._order_contracts_to_amount(order)
                return order
            except ccxt.OrderNotFound as e:
                raise RetryableOrderError(
                    f"未找到订单（交易对: {pair} id: {order_id}）。消息: {e}"
                ) from e
        except ccxt.InvalidOrder as e:
            raise InvalidOrderException(
                f"尝试获取无效订单（交易对: {pair} id: {order_id}）。消息: {e}"
            ) from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于 {e.__class__.__name__} 无法获取订单。消息: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier(retries=API_FETCH_ORDER_RETRY_COUNT)
    def fetch_order(self, order_id: str, pair: str, params: dict | None = None) -> CcxtOrder:
        if self._config["dry_run"]:
            return self.fetch_dry_run_order(order_id)
        if params is None:
            params = {}
        try:
            if not self.exchange_has("fetchOrder"):
                return self.fetch_order_emulated(order_id, pair, params)
            order = self._api.fetch_order(order_id, pair, params=params)
            self._log_exchange_response("fetch_order", order)
            order = self._order_contracts_to_amount(order)
            return order
        except ccxt.OrderNotFound as e:
            raise RetryableOrderError(
                f"未找到订单（交易对: {pair} id: {order_id}）。消息: {e}"
            ) from e
        except ccxt.InvalidOrder as e:
            raise InvalidOrderException(
                f"尝试获取无效订单（交易对: {pair} id: {order_id}）。消息: {e}"
            ) from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于 {e.__class__.__name__} 无法获取订单。消息: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def fetch_stoploss_order(
        self, order_id: str, pair: str, params: dict | None = None
    ) -> CcxtOrder:
        return self.fetch_order(order_id, pair, params)

    def fetch_order_or_stoploss_order(
        self, order_id: str, pair: str, stoploss_order: bool = False
    ) -> CcxtOrder:
        """
        简单包装器，根据stoploss_order参数调用
        fetch_order或fetch_stoploss_order
        :param order_id: 要获取订单的OrderId
        :param pair: 对应于order_id的交易对
        :param stoploss_order: 如果为true，使用fetch_stoploss_order，否则使用fetch_order。
        """
        if stoploss_order:
            return self.fetch_stoploss_order(order_id, pair)
        return self.fetch_order(order_id, pair)

    def check_order_canceled_empty(self, order: CcxtOrder) -> bool:
        """
        验证订单是否已被取消而未被部分成交
        :param order: 从fetch_order()返回的订单字典
        :return: 如果订单已被取消而未被成交，则为True，否则为False。
        """
        return order.get("status") in NON_OPEN_EXCHANGE_STATES and order.get("filled") == 0.0

    @retrier
    def cancel_order(self, order_id: str, pair: str, params: dict | None = None) -> dict[str, Any]:
        if self._config["dry_run"]:
            try:
                order = self.fetch_dry_run_order(order_id)

                order.update({"status": "canceled", "filled": 0.0, "remaining": order["amount"]})
                return order
            except InvalidOrderException:
                return {}

        if params is None:
            params = {}
        try:
            order = self._api.cancel_order(order_id, pair, params=params)
            self._log_exchange_response("cancel_order", order)
            order = self._order_contracts_to_amount(order)
            return order
        except ccxt.InvalidOrder as e:
            raise InvalidOrderException(f"无法取消订单。消息: {e}") from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于 {e.__class__.__name__} 无法取消订单。消息: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def cancel_stoploss_order(self, order_id: str, pair: str, params: dict | None = None) -> dict:
        return self.cancel_order(order_id, pair, params)

    def is_cancel_order_result_suitable(self, corder) -> TypeGuard[CcxtOrder]:
        if not isinstance(corder, dict):
            return False

        required = ("fee", "status", "amount")
        return all(corder.get(k, None) is not None for k in required)

    def cancel_order_with_result(self, order_id: str, pair: str, amount: float) -> CcxtOrder:
        """
        取消订单并返回结果。
        如果取消订单返回不可用的结果
        并且fetch_order不起作用（某些交易所不返回已取消的订单），
        则创建一个假结果
        :param order_id: 要取消的订单id
        :param pair: 对应于order_id的交易对
        :param amount: 用于假响应的数量
        :return: 如果可用，则从cancel_order返回结果，否则从fetch_order返回
        """
        try:
            corder = self.cancel_order(order_id, pair)
            if self.is_cancel_order_result_suitable(corder):
                return corder
        except InvalidOrderException:
            logger.warning(f"无法取消 {pair} 的订单 {order_id}。")
        try:
            order = self.fetch_order(order_id, pair)
        except InvalidOrderException:
            logger.warning(f"无法获取已取消的订单 {order_id}。")
            order = {
                "id": order_id,
                "status": "canceled",
                "amount": amount,
                "filled": 0.0,
                "fee": {},
                "info": {},
            }

        return order

    def cancel_stoploss_order_with_result(
        self, order_id: str, pair: str, amount: float
    ) -> CcxtOrder:
        """
        取消止损订单并返回结果。
        如果取消订单返回不可用的结果
        并且fetch_order不起作用（某些交易所不返回已取消的订单），
        则创建一个假结果
        :param order_id: 要取消的止损订单id
        :param pair: 对应于order_id的交易对
        :param amount: 用于假响应的数量
        :return: 如果可用，则从cancel_order返回结果，否则从fetch_order返回
        """
        corder = self.cancel_stoploss_order(order_id, pair)
        if self.is_cancel_order_result_suitable(corder):
            return corder
        try:
            order = self.fetch_stoploss_order(order_id, pair)
        except InvalidOrderException:
            logger.warning(f"无法获取已取消的止损订单 {order_id}。")
            order = {"id": order_id, "fee": {}, "status": "canceled", "amount": amount, "info": {}}

        return order

    @retrier
    def get_balances(self) -> CcxtBalances:
        try:
            balances = self._api.fetch_balance()
            # 从ccxt结果中删除额外信息
            balances.pop("info", None)
            balances.pop("free", None)
            balances.pop("total", None)
            balances.pop("used", None)

            self._log_exchange_response("fetch_balances", balances)
            return balances
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于 {e.__class__.__name__} 无法获取余额。消息: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def fetch_positions(self, pair: str | None = None) -> list[CcxtPosition]:
        """
        从交易所获取仓位。
        如果没有给出交易对，则返回所有仓位。
        :param pair: 查询的交易对
        """
        if self._config["dry_run"] or self.trading_mode != TradingMode.FUTURES:
            return []
        try:
            symbols = []
            if pair:
                symbols.append(pair)
            positions: list[CcxtPosition] = self._api.fetch_positions(symbols)
            self._log_exchange_response("fetch_positions", positions)
            return positions
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于 {e.__class__.__name__} 无法获取仓位。消息: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def _fetch_orders_emulate(self, pair: str, since_ms: int) -> list[CcxtOrder]:
        orders = []
        if self.exchange_has("fetchClosedOrders"):
            orders = self._api.fetch_closed_orders(pair, since=since_ms)
            if self.exchange_has("fetchOpenOrders"):
                orders_open = self._api.fetch_open_orders(pair, since=since_ms)
                orders.extend(orders_open)
        return orders

    @retrier(retries=0)
    def _fetch_orders(
        self, pair: str, since: datetime, params: dict | None = None
    ) -> list[CcxtOrder]:
        """
        获取"since"以来的交易对的所有订单
        :param pair: 查询的交易对
        :param since: 查询的起始时间
        """
        if self._config["dry_run"]:
            return []

        try:
            since_ms = int((since.timestamp() - 10) * 1000)

            if self.exchange_has("fetchOrders"):
                if not params:
                    params = {}
                try:
                    orders: list[CcxtOrder] = self._api.fetch_orders(
                        pair, since=since_ms, params=params
                    )
                except ccxt.NotSupported:
                    # 某些交易所不支持fetchOrders
                    # 尝试分别获取开放和已关闭的订单
                    orders = self._fetch_orders_emulate(pair, since_ms)
            else:
                orders = self._fetch_orders_emulate(pair, since_ms)
            self._log_exchange_response("fetch_orders", orders)
            orders = [self._order_contracts_to_amount(o) for o in orders]
            return orders
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于 {e.__class__.__name__} 无法获取仓位。消息: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def fetch_orders(
        self, pair: str, since: datetime, params: dict | None = None
    ) -> list[CcxtOrder]:
        if self._config["dry_run"]:
            return []
        if (limit := self._ft_has.get("fetch_orders_limit_minutes")) is not None:
            orders = []
            while since < dt_now():
                orders += self._fetch_orders(pair, since)
                # 带有1分钟重叠的Since
                since = since + timedelta(minutes=limit - 1)
            # 确保每个订单基于订单id是唯一的
            orders = list({order["id"]: order for order in orders}.values())
            return orders

        else:
            return self._fetch_orders(pair, since, params=params)

    @retrier
    def fetch_trading_fees(self) -> dict[str, Any]:
        """
        获取用户账户交易费用
        可以缓存，不应经常更新。
        """
        if (
            self._config["dry_run"]
            or self.trading_mode != TradingMode.FUTURES
            or not self.exchange_has("fetchTradingFees")
        ):
            return {}
        try:
            trading_fees: dict[str, Any] = self._api.fetch_trading_fees()
            self._log_exchange_response("fetch_trading_fees", trading_fees)
            return trading_fees
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于 {e.__class__.__name__} 无法获取交易费用。消息: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def fetch_bids_asks(self, symbols: list[str] | None = None, *, cached: bool = False) -> dict:
        """
        :param symbols: 要获取的符号列表
        :param cached: 允许缓存结果
        :return: fetch_bids_asks结果
        """
        if not self.exchange_has("fetchBidsAsks"):
            return {}
        if cached:
            with self._cache_lock:
                tickers = self._fetch_tickers_cache.get("fetch_bids_asks")
            if tickers:
                return tickers
        try:
            tickers = self._api.fetch_bids_asks(symbols)
            with self._cache_lock:
                self._fetch_tickers_cache["fetch_bids_asks"] = tickers
            return tickers
        except ccxt.NotSupported as e:
            raise OperationalException(
                f"交易所 {self._api.name} 不支持批量获取买价/卖价。"
                f"消息: {e}"
            ) from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于 {e.__class__.__name__} 无法加载买价/卖价。消息: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def get_tickers(
        self,
        symbols: list[str] | None = None,
        *,
        cached: bool = False,
        market_type: TradingMode | None = None,
    ) -> Tickers:
        """
        :param symbols: 要获取的符号列表
        :param cached: 允许缓存结果
        :param market_type: 要获取的市场类型 - 现货或期货。
        :return: fetch_tickers结果
        """
        tickers: Tickers
        if not self.exchange_has("fetchTickers"):
            return {}
        cache_key = f"fetch_tickers_{market_type}" if market_type else "fetch_tickers"
        if cached:
            with self._cache_lock:
                tickers = self._fetch_tickers_cache.get(cache_key)  # type: ignore
            if tickers:
                return tickers
        try:
            # 将期货重新映射到交换
            market_types = {
                TradingMode.FUTURES: "swap",
            }
            params = {"type": market_types.get(market_type, market_type)} if market_type else {}
            tickers = self._api.fetch_tickers(symbols, params)
            with self._cache_lock:
                self._fetch_tickers_cache[cache_key] = tickers
            return tickers
        except ccxt.NotSupported as e:
            raise OperationalException(
                f"交易所 {self._api.name} 不支持批量获取行情。"
                f"消息: {e}"
            ) from e
        except ccxt.BadSymbol as e:
            logger.warning(
                f"由于 {e.__class__.__name__} 无法加载行情。消息: {e}。"
                "重新加载市场。"
            )
            self.reload_markets(True)
            # 重新引发异常以重复调用。
            raise TemporaryError from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于 {e.__class__.__name__} 无法加载行情。消息: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def get_proxy_coin(self) -> str:
        """
        获取给定币种的代理币
        如果找不到代理币，则回退到权益货币
        :return: 代理币或权益货币
        """
        return self._config["stake_currency"]

    def get_conversion_rate(self, coin: str, currency: str) -> float | None:
        """
        快速缓存方式获取一种货币到另一种货币的转换率。
        然后可以用作"rate * amount"在货币之间进行转换。
        :param coin: 要转换的币种
        :param currency: 要转换到的货币
        :returns: 从币种到货币的转换率
        :raises: ExchangeErrors
        """

        if (proxy_coin := self._ft_has["proxy_coin_mapping"].get(coin, None)) is not None:
            coin = proxy_coin
        if (proxy_currency := self._ft_has["proxy_coin_mapping"].get(currency, None)) is not None:
            currency = proxy_currency
        if coin == currency:
            return 1.0
        tickers = self.get_tickers(cached=True)
        try:
            for pair in self.get_valid_pair_combination(coin, currency):
                ticker: Ticker | None = tickers.get(pair, None)
                if not ticker:
                    tickers_other: Tickers = self.get_tickers(
                        cached=True,
                        market_type=(
                            TradingMode.SPOT
                            if self.trading_mode != TradingMode.SPOT
                            else TradingMode.FUTURES
                        ),
                    )
                    ticker = tickers_other.get(pair, None)
                if ticker:
                    rate: float | None = safe_value_fallback2(ticker, ticker, "last", "ask", None)
                    if rate and pair.startswith(currency) and not pair.endswith(currency):
                        rate = 1.0 / rate
                    return rate
        except ValueError:
            return None
        return None

    @retrier
    def fetch_ticker(self, pair: str) -> Ticker:
        try:
            if pair not in self.markets or self.markets[pair].get("active", False) is False:
                raise ExchangeError(f"交易对 {pair} 不可用")
            data: Ticker = self._api.fetch_ticker(pair)
            return data
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于 {e.__class__.__name__} 无法加载行情。消息: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @staticmethod
    def get_next_limit_in_list(
        limit: int,
        limit_range: list[int] | None,
        range_required: bool = True,
        upper_limit: int | None = None,
    ):
        """
        获取列表中下一个更大的值。
        如果api仅支持有限范围，则由fetch_l2_order_book使用
        如果同时提供limit_range和upper_limit，limit_range优先。
        """
        if not limit_range:
            return min(limit, upper_limit) if upper_limit else limit

        result = min([x for x in limit_range if limit <= x] + [max(limit_range)])
        if not range_required and limit > result:
            # 范围不是必需的 - 我们可以使用None作为参数。
            return None
        return result

    @retrier
    def fetch_l2_order_book(self, pair: str, limit: int = 100) -> OrderBook:
        """
        从交易所获取L2订单簿。
        可以限制为一定数量（如果支持）。
        返回格式为
        {'asks': [price, volume], 'bids': [price, volume]} 的字典
        """
        limit1 = self.get_next_limit_in_list(
            limit,
            self._ft_has["l2_limit_range"],
            self._ft_has["l2_limit_range_required"],
            self._ft_has["l2_limit_upper"],
        )
        try:
            return self._api.fetch_l2_order_book(pair, limit1)
        except ccxt.NotSupported as e:
            raise OperationalException(
                f"交易所 {self._api.name} 不支持获取订单簿。消息: {e}"
            ) from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于 {e.__class__.__name__} 无法获取订单簿。消息: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def _get_price_side(self, side: str, is_short: bool, conf_strategy: dict) -> BidAsk:
        price_side = conf_strategy["price_side"]

        if price_side in ("same", "other"):
            price_map = {
                ("entry", "long", "same"): "bid",
                ("entry", "long", "other"): "ask",
                ("entry", "short", "same"): "ask",
                ("entry", "short", "other"): "bid",
                ("exit", "long", "same"): "ask",
                ("exit", "long", "other"): "bid",
                ("exit", "short", "same"): "bid",
                ("exit", "short", "other"): "ask",
            }
            price_side = price_map[(side, "short" if is_short else "long", price_side)]
        return price_side

    def get_rate(
        self,
        pair: str,
        refresh: bool,
        side: EntryExit,
        is_short: bool,
        order_book: OrderBook | None = None,
        ticker: Ticker | None = None,
    ) -> float:
        """
        计算买价/卖价目标
        买价率 - 在当前卖价和最后价格之间
        卖价率 - 基于订单簿使用行情买价或第一个买价
        或在任何其他情况下保持静态，因为它不会更新。
        :param pair: 获取费率的交易对
        :param refresh: 允许缓存数据
        :param side: "buy" 或 "sell"
        :return: float: 价格
        :raises PricingError 如果无法确定订单簿价格。
        """
        name = side.capitalize()
        strat_name = "entry_pricing" if side == "entry" else "exit_pricing"

        cache_rate: TTLCache = self._entry_rate_cache if side == "entry" else self._exit_rate_cache
        if not refresh:
            with self._cache_lock:
                rate = cache_rate.get(pair)
            # 检查缓存是否已失效
            if rate:
                logger.debug(f"使用 {pair} 的缓存 {side} 费率。")
                return rate

        conf_strategy = self._config.get(strat_name, {})

        price_side = self._get_price_side(side, is_short, conf_strategy)

        if conf_strategy.get("use_order_book", False):
            order_book_top = conf_strategy.get("order_book_top", 1)
            if order_book is None:
                order_book = self.fetch_l2_order_book(pair, order_book_top)
            rate = self._get_rate_from_ob(pair, side, order_book, name, price_side, order_book_top)
        else:
            logger.debug(f"使用最后 {price_side.capitalize()} / 最后价格")
            if ticker is None:
                ticker = self.fetch_ticker(pair)
            rate = self._get_rate_from_ticker(side, ticker, conf_strategy, price_side)

        if rate is None:
            raise PricingError(f"{pair} 的 {name} 费率为空。")
        with self._cache_lock:
            cache_rate[pair] = rate

        return rate

    def _get_rate_from_ticker(
        self, side: EntryExit, ticker: Ticker, conf_strategy: dict[str, Any], price_side: BidAsk
    ) -> float | None:
        """
        从行情获取费率。
        """
        ticker_rate = ticker[price_side]
        if ticker["last"] and ticker_rate:
            if side == "entry" and ticker_rate > ticker["last"]:
                balance = conf_strategy.get("price_last_balance", 0.0)
                ticker_rate = ticker_rate + balance * (ticker["last"] - ticker_rate)
            elif side == "exit" and ticker_rate < ticker["last"]:
                balance = conf_strategy.get("price_last_balance", 0.0)
                ticker_rate = ticker_rate - balance * (ticker_rate - ticker["last"])
        rate = ticker_rate
        return rate

    def _get_rate_from_ob(
        self,
        pair: str,
        side: EntryExit,
        order_book: OrderBook,
        name: str,
        price_side: BidAsk,
        order_book_top: int,
    ) -> float:
        """
        从订单簿获取费率
        :raises: PricingError 如果无法确定费率。
        """
        logger.debug("订单簿 %s", order_book)
        # top 1 = 索引 0
        try:
            obside: OBLiteral = "bids" if price_side == "bid" else "asks"
            rate = order_book[obside][order_book_top - 1][0]
        except (IndexError, KeyError) as e:
            logger.warning(
                f"{pair} - 无法从订单簿位置 {order_book_top} 确定 {name} 价格。"
                f"订单簿: {order_book}"
            )
            raise PricingError from e
        logger.debug(
            f"{pair} - 来自订单簿 {price_side.capitalize()} 侧的 {name} 价格"
            f" - 前 {order_book_top} 订单簿 {side} 费率 {rate:.8f}"
        )
        return rate

    def get_rates(self, pair: str, refresh: bool, is_short: bool) -> tuple[float, float]:
        entry_rate = None
        exit_rate = None
        if not refresh:
            with self._cache_lock:
                entry_rate = self._entry_rate_cache.get(pair)
                exit_rate = self._exit_rate_cache.get(pair)
            if entry_rate:
                logger.debug(f"使用 {pair} 的缓存买入费率。")
            if exit_rate:
                logger.debug(f"使用 {pair} 的缓存卖出费率。")

        entry_pricing = self._config.get("entry_pricing", {})
        exit_pricing = self._config.get("exit_pricing", {})
        order_book = ticker = None
        if not entry_rate and entry_pricing.get("use_order_book", False):
            order_book_top = max(
                entry_pricing.get("order_book_top", 1), exit_pricing.get("order_book_top", 1)
            )
            order_book = self.fetch_l2_order_book(pair, order_book_top)
            entry_rate = self.get_rate(pair, refresh, "entry", is_short, order_book=order_book)
        elif not entry_rate:
            ticker = self.fetch_ticker(pair)
            entry_rate = self.get_rate(pair, refresh, "entry", is_short, ticker=ticker)
        if not exit_rate:
            exit_rate = self.get_rate(
                pair, refresh, "exit", is_short, order_book=order_book, ticker=ticker
            )
        return entry_rate, exit_rate

    # 费用处理

    @retrier
    def get_trades_for_order(
        self, order_id: str, pair: str, since: datetime, params: dict | None = None
    ) -> list:
        """
        使用"fetch_my_trades"端点获取订单并按order-id过滤它们。
        传入的"since"参数来自数据库，是UTC时区，
        作为时区感知的datetime对象。
        来自python文档：
            > 原始datetime实例被假定为表示本地时间
        因此，调用"since.timestamp()"将获得UTC时间戳，在应用
        从本地时区到UTC的转换之后。
        这适用于UTC+时区，因为结果将包含几小时的交易
        而不是最后5秒的交易，但对于UTC-时区会失败，
        因为我们要求的交易带有未来的"since"参数。

        :param order_id order_id: 创建订单时给出的订单ID
        :param pair: 订单的交易对
        :param since: 订单创建时间的datetime对象。假定对象在UTC中。
        """
        if self._config["dry_run"]:
            return []
        if not self.exchange_has("fetchMyTrades"):
            return []
        try:
            # 允许5秒偏移以捕获轻微的时间偏移（在#1185中发现）
            # since需要以毫秒为单位的int
            _params = params if params else {}
            my_trades = self._api.fetch_my_trades(
                pair,
                int((since.replace(tzinfo=timezone.utc).timestamp() - 5) * 1000),
                params=_params,
            )
            matched_trades = [trade for trade in my_trades if trade["order"] == order_id]

            self._log_exchange_response("get_trades_for_order", matched_trades)

            matched_trades = self._trades_contracts_to_amount(matched_trades)

            return matched_trades
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于 {e.__class__.__name__} 无法获取交易。消息: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def get_order_id_conditional(self, order: CcxtOrder) -> str:
        return order["id"]

    @retrier
    def get_fee(
        self,
        symbol: str,
        order_type: str = "",
        side: str = "",
        amount: float = 1,
        price: float = 1,
        taker_or_maker: MakerTaker = "maker",
    ) -> float:
        """
        从交易所检索费用
        :param symbol: 交易对
        :param order_type: 订单类型（市价、限价等）
        :param side: 订单方向（买入、卖出）
        :param amount: 订单数量
        :param price: 订单价格
        :param taker_or_maker: 'maker' 或 'taker'（如果提供"type"则忽略）
        """
        if order_type and order_type == "market":
            taker_or_maker = "taker"
        try:
            if self._config["dry_run"] and self._config.get("fee", None) is not None:
                return self._config["fee"]
            # 在尝试获取费用之前验证市场是否已加载
            if self._api.markets is None or len(self._api.markets) == 0:
                self._api.load_markets(params={})

            return self._api.calculate_fee(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                takerOrMaker=taker_or_maker,
            )["rate"]
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于 {e.__class__.__name__} 无法获取费用信息。消息: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @staticmethod
    def order_has_fee(order: CcxtOrder) -> bool:
        """
        验证传入的订单字典是否具有提取费用所需的键，
        以及这些键（currency、cost）是否不为空。
        :param order: 订单或交易（单笔交易）字典
        :return: 如果费用子结构包含currency和cost，则为True，否则为false
        """
        if not isinstance(order, dict):
            return False
        return (
            "fee" in order
            and order["fee"] is not None
            and (order["fee"].keys() >= {"currency", "cost"})
            and order["fee"]["currency"] is not None
            and order["fee"]["cost"] is not None
        )

    def calculate_fee_rate(
        self, fee: dict, symbol: str, cost: float, amount: float
    ) -> float | None:
        """
        如果交易所未提供费率，则计算费率。
        :param fee: ccxt费用字典 - 必须包含cost / currency / rate
        :param symbol: 订单的符号
        :param cost: 订单的总成本
        :param amount: 订单的数量
        """
        if fee.get("rate") is not None:
            return fee.get("rate")
        fee_curr = fee.get("currency")
        if fee_curr is None:
            return None
        fee_cost = float(fee["cost"])

        # 根据订单详情计算费用
        if fee_curr == self.get_pair_base_currency(symbol):
            # 基础货币 - 除以数量
            return round(fee_cost / amount, 8)
        elif fee_curr == self.get_pair_quote_currency(symbol):
            # 报价货币 - 除以成本
            return round(fee_cost / cost, 8) if cost else None
        else:
            # 如果费用货币是不同的货币
            if not cost:
                # 如果成本是None或0.0 -> 假值，返回None
                return None
            try:
                fee_to_quote_rate = self.get_conversion_rate(
                    fee_curr, self._config["stake_currency"]
                )
                if not fee_to_quote_rate:
                    raise ValueError("未找到转换率。")
            except (ValueError, ExchangeError):
                fee_to_quote_rate = self._config["exchange"].get("unknown_fee_rate", None)
                if not fee_to_quote_rate:
                    return None
            return round((fee_cost * fee_to_quote_rate) / cost, 8)

    def extract_cost_curr_rate(
        self, fee: dict[str, Any], symbol: str, cost: float, amount: float
    ) -> tuple[float, str, float | None]:
        """
        提取成本、货币、费率的元组。
        需要先运行order_has_fee！
        :param fee: ccxt费用字典 - 必须包含cost / currency / rate
        :param symbol: 订单的符号
        :param cost: 订单的总成本
        :param amount: 订单的数量
        :return: 包含给定费用字典的成本、货币、费率的元组
        """
        return (
            float(fee["cost"]),
            fee["currency"],
            self.calculate_fee_rate(fee, symbol, cost, amount),
        )

    # 历史数据

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
        使用asyncio获取蜡烛历史并返回蜡烛列表。
        处理所有异步工作。
        对一个交易对进行异步，假设我们每次调用获得`self.ohlcv_candle_limit()`根蜡烛。
        :param pair: 要下载的交易对
        :param timeframe: 获取数据的时间框架
        :param since_ms: 获取历史记录的起始时间戳（毫秒）
        :param candle_type: ''、mark、index、premiumIndex或funding_rate
        :param is_new_pair: 由binance子类使用以允许"快速"新交易对下载
        :param until_ms: 获取历史记录的结束时间戳（毫秒）
        :return: 包含蜡烛（OHLCV）数据的Dataframe
        """
        with self._loop_lock:
            pair, _, _, data, _ = self.loop.run_until_complete(
                self._async_get_historic_ohlcv(
                    pair=pair,
                    timeframe=timeframe,
                    since_ms=since_ms,
                    until_ms=until_ms,
                    candle_type=candle_type,
                    raise_=True,
                )
            )
        logger.debug(f"从ccxt下载了 {pair} 的数据，长度为 {len(data)}。")
        return ohlcv_to_dataframe(data, timeframe, pair, fill_missing=False, drop_incomplete=True)

    async def _async_get_historic_ohlcv(
        self,
        pair: str,
        timeframe: str,
        since_ms: int,
        candle_type: CandleType,
        raise_: bool = False,
        until_ms: int | None = None,
    ) -> OHLCVResponse:
        """
        下载历史ohlcv
        :param candle_type: CandleType枚举中的任何一个（必须匹配交易模式！）
        """

        one_call = timeframe_to_msecs(timeframe) * self.ohlcv_candle_limit(
            timeframe, candle_type, since_ms
        )
        logger.debug(
            "one_call: %s毫秒 (%s)",
            one_call,
            dt_humanize_delta(dt_now() - timedelta(milliseconds=one_call)),
        )
        input_coroutines = [
            self._async_get_candle_history(pair, timeframe, candle_type, since)
            for since in range(since_ms, until_ms or dt_ts(), one_call)
        ]

        data: list = []
        # 将请求分块为100个批次，以避免淹没ccxt限流
        for input_coro in chunks(input_coroutines, 100):
            results = await asyncio.gather(*input_coro, return_exceptions=True)
            for res in results:
                if isinstance(res, BaseException):
                    logger.warning(f"异步代码引发异常: {repr(res)}")
                    if raise_:
                        raise res
                    continue
                else:
                    # 如果不是异常，则解构元组
                    p, _, c, new_data, _ = res
                    if p == pair and c == candle_type:
                        data.extend(new_data)
        # 扩展结果后再次排序数据 - 上述调用以"异步顺序"返回
        data = sorted(data, key=lambda x: x[0])
        return pair, timeframe, candle_type, data, self._ohlcv_partial_candle

    def _try_build_from_websocket(
        self, pair: str, timeframe: str, candle_type: CandleType
    ) -> Coroutine[Any, Any, OHLCVResponse] | None:
        """
        尝试构建协程以从websocket获取数据。
        """
        if self._can_use_websocket(self._exchange_ws, pair, timeframe, candle_type):
            candle_ts = dt_ts(timeframe_to_prev_date(timeframe))
            prev_candle_ts = dt_ts(date_minus_candles(timeframe, 1))
            candles = self._exchange_ws.ohlcvs(pair, timeframe)
            half_candle = int(candle_ts - (candle_ts - prev_candle_ts) * 0.5)
            last_refresh_time = int(
                self._exchange_ws.klines_last_refresh.get((pair, timeframe, candle_type), 0)
            )

            if (
                candles
                and (
                    (len(candles) > 1 and candles[-1][0] >= prev_candle_ts)
                    # 重新连接时的边缘情况，其中有1根蜡烛可用但它是当前的
                    or (len(candles) == 1 and candles[-1][0] < candle_ts)
                )
                and last_refresh_time >= half_candle
            ):
                # 可用结果，蜡烛包含前一根蜡烛。
                # 另外，我们检查最后刷新时间是否不超过半根蜡烛前。
                logger.debug(f"重用 {pair}、{timeframe}、{last_refresh_time} 的监视结果")

                return self._exchange_ws.get_ohlcv(pair, timeframe, candle_type, candle_ts)
            logger.info(
                f"无法重用 {pair}、{timeframe} 的监视，回退到REST api。"
                f"{candle_ts < last_refresh_time}、{candle_ts}、{last_refresh_time}、"
                f"{format_ms_time(candle_ts)}、{format_ms_time(last_refresh_time)} "
            )
        return None

    def _can_use_websocket(
        self, exchange_ws: ExchangeWS | None, pair: str, timeframe: str, candle_type: CandleType
    ) -> TypeGuard[ExchangeWS]:
        """
        检查我们是否可以为此交易对使用websocket。
        充当exchangeWs的类型保护
        """
        if exchange_ws and candle_type in (CandleType.SPOT, CandleType.FUTURES):
            return True
        return False

    def _build_coroutine(
        self,
        pair: str,
        timeframe: str,
        candle_type: CandleType,
        since_ms: int | None,
        cache: bool,
    ) -> Coroutine[Any, Any, OHLCVResponse]:
        not_all_data = cache and self.required_candle_call_count > 1
        if cache:
            if self._can_use_websocket(self._exchange_ws, pair, timeframe, candle_type):
                # 订阅websocket
                self._exchange_ws.schedule_ohlcv(pair, timeframe, candle_type)

        if cache and (pair, timeframe, candle_type) in self._klines:
            candle_limit = self.ohlcv_candle_limit(timeframe, candle_type)
            min_ts = dt_ts(date_minus_candles(timeframe, candle_limit - 5))

            if ws_resp := self._try_build_from_websocket(pair, timeframe, candle_type):
                # 我们有一个可用的websocket响应
                return ws_resp

            # 检查1次调用是否可以获得更新的蜡烛而不会在数据中产生空洞。
            if min_ts < self._pairs_last_refresh_time.get((pair, timeframe, candle_type), 0):
                # 可以使用缓存 - 进行一次性调用。
                not_all_data = False
            else:
                # 检测到时间跳跃，清除缓存
                logger.info(
                    f"检测到时间跳跃。清除 {pair}、{timeframe}、{candle_type} 的缓存"
                )
                del self._klines[(pair, timeframe, candle_type)]

        if not since_ms and (self._ft_has["ohlcv_require_since"] or not_all_data):
            # 一个交易对的多次调用 - 获取更多历史记录
            one_call = timeframe_to_msecs(timeframe) * self.ohlcv_candle_limit(
                timeframe, candle_type, since_ms
            )
            move_to = one_call * self.required_candle_call_count
            now = timeframe_to_next_date(timeframe)
            since_ms = dt_ts(now - timedelta(seconds=move_to // 1000))

        if since_ms:
            return self._async_get_historic_ohlcv(
                pair, timeframe, since_ms=since_ms, raise_=True, candle_type=candle_type
            )
        else:
            # 一次调用..."常规"刷新
            return self._async_get_candle_history(
                pair, timeframe, since_ms=since_ms, candle_type=candle_type
            )

    def _build_ohlcv_dl_jobs(
        self, pair_list: ListPairsWithTimeframes, since_ms: int | None, cache: bool
    ) -> tuple[list[Coroutine], list[PairWithTimeframe]]:
        """
        构建作为refresh_latest_ohlcv一部分执行的协程
        """
        input_coroutines: list[Coroutine[Any, Any, OHLCVResponse]] = []
        cached_pairs = []
        for pair, timeframe, candle_type in set(pair_list):
            if timeframe not in self.timeframes and candle_type in (
                CandleType.SPOT,
                CandleType.FUTURES,
            ):
                logger.warning(
                    f"无法下载 ({pair}, {timeframe}) 组合，因为此时间框架"
                    f"在 {self.name} 上不可用。可用的时间框架是 "
                    f"{', '.join(self.timeframes)}。"
                )
                continue

            if (
                (pair, timeframe, candle_type) not in self._klines
                or not cache
                or self._now_is_time_to_refresh(pair, timeframe, candle_type)
            ):
                input_coroutines.append(
                    self._build_coroutine(pair, timeframe, candle_type, since_ms, cache)
                )

            else:
                logger.debug(
                    f"使用 {pair}、{timeframe}、{candle_type} 的缓存蜡烛（OHLCV）数据..."
                )
                cached_pairs.append((pair, timeframe, candle_type))

        return input_coroutines, cached_pairs

    def _process_ohlcv_df(
        self,
        pair: str,
        timeframe: str,
        c_type: CandleType,
        ticks: list[list],
        cache: bool,
        drop_incomplete: bool,
    ) -> DataFrame:
        # 将最后一根蜡烛时间保留为该交易对的最后刷新时间
        if ticks and cache:
            idx = -2 if drop_incomplete and len(ticks) > 1 else -1
            self._pairs_last_refresh_time[(pair, timeframe, c_type)] = ticks[idx][0]
        # 在缓存中保留解析的数据框
        ohlcv_df = ohlcv_to_dataframe(
            ticks, timeframe, pair=pair, fill_missing=True, drop_incomplete=drop_incomplete
        )
        if cache:
            if (pair, timeframe, c_type) in self._klines:
                old = self._klines[(pair, timeframe, c_type)]
                # 重新分配，以便我们返回更新的组合df
                ohlcv_df = clean_ohlcv_dataframe(
                    concat([old, ohlcv_df], axis=0),
                    timeframe,
                    pair,
                    fill_missing=True,
                    drop_incomplete=False,
                )
                candle_limit = self.ohlcv_candle_limit(timeframe, self._config["candle_type_def"])
                # 老化旧蜡烛
                ohlcv_df = ohlcv_df.tail(candle_limit + self._startup_candle_count)
                ohlcv_df = ohlcv_df.reset_index(drop=True)
                self._klines[(pair, timeframe, c_type)] = ohlcv_df
            else:
                self._klines[(pair, timeframe, c_type)] = ohlcv_df
        return ohlcv_df

    def refresh_latest_ohlcv(
        self,
        pair_list: ListPairsWithTimeframes,
        *,
        since_ms: int | None = None,
        cache: bool = True,
        drop_incomplete: bool | None = None,
    ) -> dict[PairWithTimeframe, DataFrame]:
        """
        异步刷新内存中的OHLCV并使用结果设置`_klines`
        异步循环pair_list并异步下载所有交易对（半并行）。
        仅在dataprovider.refresh()方法中使用。
        :param pair_list: 包含要刷新的交易对、间隔的2元素元组列表
        :param since_ms: 下载的起始时间，以毫秒为单位
        :param cache: 将结果分配给_klines。对于像pairlists这样的一次性下载很有用
        :param drop_incomplete: 控制蜡烛丢弃。
            指定None默认为_ohlcv_partial_candle
        :return: [{(pair, timeframe): Dataframe}] 的字典
        """
        logger.debug("为 %d 个交易对刷新蜡烛（OHLCV）数据", len(pair_list))

        # 收集要运行的协程
        ohlcv_dl_jobs, cached_pairs = self._build_ohlcv_dl_jobs(pair_list, since_ms, cache)

        results_df = {}
        # 将请求分块为100个批次，以避免淹没ccxt限流
        for dl_jobs_batch in chunks(ohlcv_dl_jobs, 100):

            async def gather_coroutines(coro):
                return await asyncio.gather(*coro, return_exceptions=True)

            with self._loop_lock:
                results = self.loop.run_until_complete(gather_coroutines(dl_jobs_batch))

            for res in results:
                if isinstance(res, Exception):
                    logger.warning(f"异步代码引发异常: {repr(res)}")
                    continue
                # 解构元组（有5个元素）
                pair, timeframe, c_type, ticks, drop_hint = res
                drop_incomplete_ = drop_hint if drop_incomplete is None else drop_incomplete
                ohlcv_df = self._process_ohlcv_df(
                    pair, timeframe, c_type, ticks, cache, drop_incomplete_
                )

                results_df[(pair, timeframe, c_type)] = ohlcv_df

        # 返回缓存的klines
        for pair, timeframe, c_type in cached_pairs:
            results_df[(pair, timeframe, c_type)] = self.klines(
                (pair, timeframe, c_type), copy=False
            )

        return results_df

    def refresh_ohlcv_with_cache(
        self, pairs: list[PairWithTimeframe], since_ms: int
    ) -> dict[PairWithTimeframe, DataFrame]:
        """
        如有必要，刷新needed_pairs中所有交易对的ohlcv数据。
        使用每个时间框架过期的缓存数据。
        应该只用于需要"按时"过期且不再缓存的pairlists。
        """

        timeframes = {p[1] for p in pairs}
        for timeframe in timeframes:
            if (timeframe, since_ms) not in self._expiring_candle_cache:
                timeframe_in_sec = timeframe_to_seconds(timeframe)
                # 初始化缓存
                self._expiring_candle_cache[(timeframe, since_ms)] = PeriodicCache(
                    ttl=timeframe_in_sec, maxsize=1000
                )

        # 从缓存获取蜡烛
        candles = {
            c: self._expiring_candle_cache[(c[1], since_ms)].get(c, None)
            for c in pairs
            if c in self._expiring_candle_cache[(c[1], since_ms)]
        }
        pairs_to_download = [p for p in pairs if p not in candles]
        if pairs_to_download:
            candles = self.refresh_latest_ohlcv(pairs_to_download, since_ms=since_ms, cache=False)
            for c, val in candles.items():
                self._expiring_candle_cache[(c[1], since_ms)][c] = val
        return candles

    def _now_is_time_to_refresh(self, pair: str, timeframe: str, candle_type: CandleType) -> bool:
        # 时间框架（秒）
        interval_in_sec = timeframe_to_msecs(timeframe)
        plr = self._pairs_last_refresh_time.get((pair, timeframe, candle_type), 0) + interval_in_sec
        # 当前活跃蜡烛开盘日期
        now = dt_ts(timeframe_to_prev_date(timeframe))
        return plr < now

    @retrier_async
    async def _async_get_candle_history(
        self,
        pair: str,
        timeframe: str,
        candle_type: CandleType,
        since_ms: int | None = None,
    ) -> OHLCVResponse:
        """
        使用fetch_ohlcv异步获取蜡烛历史数据
        :param candle_type: ''、mark、index、premiumIndex或funding_rate
        返回元组: (pair, timeframe, ohlcv_list)
        """
        try:
            # 异步获取OHLCV
            s = "(" + dt_from_ts(since_ms).isoformat() + ") " if since_ms is not None else ""
            logger.debug(
                "获取交易对 %s，%s，间隔 %s，自 %s %s...",
                pair,
                candle_type,
                timeframe,
                since_ms,
                s,
            )
            params = deepcopy(self._ft_has.get("ohlcv_params", {}))
            candle_limit = self.ohlcv_candle_limit(
                timeframe, candle_type=candle_type, since_ms=since_ms
            )

            if candle_type and candle_type != CandleType.SPOT:
                params.update({"price": candle_type.value})
            if candle_type != CandleType.FUNDING_RATE:
                data = await self._api_async.fetch_ohlcv(
                    pair, timeframe=timeframe, since=since_ms, limit=candle_limit, params=params
                )
            else:
                # 资金费率
                data = await self._fetch_funding_rate_history(
                    pair=pair,
                    timeframe=timeframe,
                    limit=candle_limit,
                    since_ms=since_ms,
                )
            # 某些交易所按ASC顺序排序OHLCV，其他交易所按DESC排序。
            # 例如：Bittrex以ASC顺序返回OHLCV列表（最旧的在前，最新的在后）
            # 而GDAX以DESC顺序返回OHLCV列表（最新的在前，最旧的在后）
            # 仅在必要时排序以节省计算时间
            try:
                if data and data[0][0] > data[-1][0]:
                    data = sorted(data, key=lambda x: x[0])
            except IndexError:
                logger.exception("加载 %s 时出错。结果是 %s。", pair, data)
                return pair, timeframe, candle_type, [], self._ohlcv_partial_candle
            logger.debug("完成获取交易对 %s，%s 间隔 %s...", pair, candle_type, timeframe)
            return pair, timeframe, candle_type, data, self._ohlcv_partial_candle

        except ccxt.NotSupported as e:
            raise OperationalException(
                f"交易所 {self._api.name} 不支持获取历史"
                f"蜡烛（OHLCV）数据。消息: {e}"
            ) from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于 {e.__class__.__name__} 无法获取历史蜡烛（OHLCV）数据"
                f"对于 {pair}，{timeframe}，{candle_type}。"
                f"消息: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(
                f"无法获取历史蜡烛（OHLCV）数据对于 "
                f"{pair}，{timeframe}，{candle_type}。消息: {e}"
            ) from e

    async def _fetch_funding_rate_history(
        self,
        pair: str,
        timeframe: str,
        limit: int,
        since_ms: int | None = None,
    ) -> list[list]:
        """
        获取资金费率历史 - 用于子类选择性覆盖。
        """
        # 资金费率
        data = await self._api_async.fetch_funding_rate_history(pair, since=since_ms, limit=limit)
        # 将资金费率转换为蜡烛模式
        data = [[x["timestamp"], x["fundingRate"], 0, 0, 0, 0] for x in data]
        return data

    # 获取交易数据

    def needed_candle_for_trades_ms(self, timeframe: str, candle_type: CandleType) -> int:
        candle_limit = self.ohlcv_candle_limit(timeframe, candle_type)
        tf_s = timeframe_to_seconds(timeframe)
        candles_fetched = candle_limit * self.required_candle_call_count

        max_candles = self._config["orderflow"]["max_candles"]

        required_candles = min(max_candles, candles_fetched)
        move_to = (
            tf_s * candle_limit * required_candles
            if required_candles > candle_limit
            else (max_candles + 1) * tf_s
        )

        now = timeframe_to_next_date(timeframe)
        return int((now - timedelta(seconds=move_to)).timestamp() * 1000)

    def _process_trades_df(
        self,
        pair: str,
        timeframe: str,
        c_type: CandleType,
        ticks: list[list],
        cache: bool,
        first_required_candle_date: int,
    ) -> DataFrame:
        # 在缓存中保留解析的数据框
        trades_df = trades_list_to_df(ticks, True)

        if cache:
            if (pair, timeframe, c_type) in self._trades:
                old = self._trades[(pair, timeframe, c_type)]
                # 重新分配，以便我们返回更新的组合df
                combined_df = concat([old, trades_df], axis=0)
                logger.debug(f"从交易数据 {pair} 清理重复的刻度")
                trades_df = DataFrame(
                    trades_df_remove_duplicates(combined_df), columns=combined_df.columns
                )
                # 老化旧蜡烛
                trades_df = trades_df[first_required_candle_date < trades_df["timestamp"]]
                trades_df = trades_df.reset_index(drop=True)
            self._trades[(pair, timeframe, c_type)] = trades_df
        return trades_df

    async def _build_trades_dl_jobs(
        self, pairwt: PairWithTimeframe, data_handler, cache: bool
    ) -> tuple[PairWithTimeframe, DataFrame | None]:
        """
        构建协程以刷新交易（然后通过async.gather调用它们）
        """
        pair, timeframe, candle_type = pairwt
        since_ms = None
        new_ticks: list = []
        all_stored_ticks_df = DataFrame(columns=[*DEFAULT_TRADES_COLUMNS, "date"])
        first_candle_ms = self.needed_candle_for_trades_ms(timeframe, candle_type)
        # 刷新，如果
        # a. 不在_trades中
        # b. 没有使用缓存
        # c. 需要新数据
        is_in_cache = (pair, timeframe, candle_type) in self._trades
        if (
            not is_in_cache
            or not cache
            or self._now_is_time_to_refresh_trades(pair, timeframe, candle_type)
        ):
            logger.debug(f"刷新 {pair} 的交易数据")
            # 从最新的_trades获取交易并
            # 与现有交易一起存储
            try:
                until = None
                from_id = None
                if is_in_cache:
                    from_id = self._trades[(pair, timeframe, candle_type)].iloc[-1]["id"]
                    until = dt_ts()  # 现在

                else:
                    until = int(timeframe_to_prev_date(timeframe).timestamp()) * 1000
                    all_stored_ticks_df = data_handler.trades_load(
                        f"{pair}-cached", self.trading_mode
                    )

                    if not all_stored_ticks_df.empty:
                        if (
                            all_stored_ticks_df.iloc[-1]["timestamp"] > first_candle_ms
                            and all_stored_ticks_df.iloc[0]["timestamp"] <= first_candle_ms
                        ):
                            # 使用缓存并进一步填充
                            last_cached_ms = all_stored_ticks_df.iloc[-1]["timestamp"]
                            from_id = all_stored_ticks_df.iloc[-1]["id"]
                            # 仅当它比first_candle_ms更接近时才使用缓存
                            since_ms = (
                                last_cached_ms
                                if last_cached_ms > first_candle_ms
                                else first_candle_ms
                            )
                        else:
                            # 跳过缓存，它太旧了
                            all_stored_ticks_df = DataFrame(
                                columns=[*DEFAULT_TRADES_COLUMNS, "date"]
                            )

                # from_id覆盖交易所设置为id分页
                [_, new_ticks] = await self._async_get_trade_history(
                    pair,
                    since=since_ms if since_ms else first_candle_ms,
                    until=until,
                    from_id=from_id,
                )

            except Exception:
                logger.exception(f"刷新 {pair} 的交易数据失败")
                return pairwt, None

            if new_ticks:
                all_stored_ticks_list = all_stored_ticks_df[DEFAULT_TRADES_COLUMNS].values.tolist()
                all_stored_ticks_list.extend(new_ticks)
                trades_df = self._process_trades_df(
                    pair,
                    timeframe,
                    candle_type,
                    all_stored_ticks_list,
                    cache,
                    first_required_candle_date=first_candle_ms,
                )
                data_handler.trades_store(
                    f"{pair}-cached", trades_df[DEFAULT_TRADES_COLUMNS], self.trading_mode
                )
                return pairwt, trades_df
            else:
                logger.error(f"{pair} 没有新的刻度")
        return pairwt, None

    def refresh_latest_trades(
        self,
        pair_list: ListPairsWithTimeframes,
        *,
        cache: bool = True,
    ) -> dict[PairWithTimeframe, DataFrame]:
        """
        异步刷新内存中的交易并使用结果设置`_trades`
        异步循环pair_list并异步下载所有交易对（半并行）。
        仅在dataprovider.refresh()方法中使用。
        :param pair_list: 包含（pair，timeframe，candle_type）的3元素元组列表
        :param cache: 将结果分配给_trades。对于像pairlists这样的一次性下载很有用
        :return: [{(pair, timeframe): Dataframe}] 的字典
        """
        from freqtrade.data.history import get_datahandler

        data_handler = get_datahandler(
            self._config["datadir"], data_format=self._config["dataformat_trades"]
        )
        logger.debug("为 %d 个交易对刷新交易数据", len(pair_list))
        results_df = {}
        trades_dl_jobs = []
        for pair_wt in set(pair_list):
            trades_dl_jobs.append(self._build_trades_dl_jobs(pair_wt, data_handler, cache))

        async def gather_coroutines(coro):
            return await asyncio.gather(*coro, return_exceptions=True)

        for dl_job_chunk in chunks(trades_dl_jobs, 100):
            with self._loop_lock:
                results = self.loop.run_until_complete(gather_coroutines(dl_job_chunk))

            for res in results:
                if isinstance(res, Exception):
                    logger.warning(f"异步代码引发异常: {repr(res)}")
                    continue
                pairwt, trades_df = res
                if trades_df is not None:
                    results_df[pairwt] = trades_df

        return results_df

    def _now_is_time_to_refresh_trades(
        self, pair: str, timeframe: str, candle_type: CandleType
    ) -> bool:  # 时间框架（秒）
        trades = self.trades((pair, timeframe, candle_type), False)
        pair_last_refreshed = int(trades.iloc[-1]["timestamp"])
        full_candle = (
            int(timeframe_to_next_date(timeframe, dt_from_ts(pair_last_refreshed)).timestamp())
            * 1000
        )
        now = dt_ts()
        return full_candle <= now

    # 获取历史交易

    @retrier_async
    async def _async_fetch_trades(
        self, pair: str, since: int | None = None, params: dict | None = None
    ) -> tuple[list[list], Any]:
        """
        使用fetch_trades异步获取交易历史。
        处理交易所错误，对交易所进行一次调用。
        :param pair: 获取交易数据的交易对
        :param since: 作为毫秒整数时间戳的起始时间
        返回: 包含交易的字典列表，下一个迭代值（新的"since"或trade_id）
        """
        try:
            trades_limit = self._max_trades_limit
            # 异步获取交易
            if params:
                logger.debug("获取交易对 %s 的交易，参数: %s ", pair, params)
                trades = await self._api_async.fetch_trades(pair, params=params, limit=trades_limit)
            else:
                logger.debug(
                    "获取交易对 %s 的交易，自 %s %s...",
                    pair,
                    since,
                    "(" + dt_from_ts(since).isoformat() + ") " if since is not None else "",
                )
                trades = await self._api_async.fetch_trades(pair, since=since, limit=trades_limit)
            trades = self._trades_contracts_to_amount(trades)
            pagination_value = self._get_trade_pagination_next_value(trades)
            return trades_dict_to_list(trades), pagination_value
        except ccxt.NotSupported as e:
            raise OperationalException(
                f"交易所 {self._api.name} 不支持获取历史交易数据。"
                f"消息: {e}"
            ) from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于 {e.__class__.__name__} 无法加载交易历史。消息: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(f"无法获取交易数据。消息: {e}") from e

    def _valid_trade_pagination_id(self, pair: str, from_id: str) -> bool:
        """
        验证交易分页id是否有效。
        解决Kraken有时ID错误的奇怪问题。
        """
        return True

    def _get_trade_pagination_next_value(self, trades: list[dict]):
        """
        提取下一个"from_id"值的分页id
        仅适用于按id获取交易历史。
        """
        if not trades:
            return None
        if self._trades_pagination == "id":
            return trades[-1].get("id")
        else:
            return trades[-1].get("timestamp")

    async def _async_get_trade_history_id_startup(
        self, pair: str, since: int
    ) -> tuple[list[list], str]:
        """
        初始trade_history_id调用的覆盖
        """
        return await self._async_fetch_trades(pair, since=since)

    async def _async_get_trade_history_id(
        self, pair: str, *, until: int, since: int, from_id: str | None = None
    ) -> tuple[str, list[list]]:
        """
        使用fetch_trades异步获取交易历史
        当交易所使用基于id的迭代时使用此方法（检查`self._trades_pagination`）
        :param pair: 获取交易数据的交易对
        :param since: 作为毫秒整数时间戳的起始时间
        :param until: 作为毫秒整数时间戳的结束时间
        :param from_id: 从ID开始下载数据（如果id已知）。如果设置，则忽略"since"。
        返回元组: (pair, trades-list)
        """

        trades: list[list] = []
        # DEFAULT_TRADES_COLUMNS: 0 -> timestamp
        # DEFAULT_TRADES_COLUMNS: 1 -> id
        has_overlap = self._ft_has.get("trades_pagination_overlap", True)
        # 默认跳过最后一笔交易，因为它是下一次调用的键
        x = slice(None, -1) if has_overlap else slice(None)

        if not from_id or not self._valid_trade_pagination_id(pair, from_id):
            # 使用基于时间的方法获取第一个元素以获得要分页的ID
            # 根据交易所，这可能会在间隔开始时引入漂移
            # 最多一小时。
            # 例如，Binance在1小时时间间隔内返回"最后1000"根蜡烛
            # - 所以我们会错过第一笔交易。
            t, from_id = await self._async_get_trade_history_id_startup(pair, since=since)
            trades.extend(t[x])
        while True:
            try:
                t, from_id_next = await self._async_fetch_trades(
                    pair, params={self._trades_pagination_arg: from_id}
                )
                if t:
                    trades.extend(t[x])
                    if from_id == from_id_next or t[-1][0] > until:
                        logger.debug(
                            f"停止，因为from_id没有改变。达到 {t[-1][0]} > {until}"
                        )
                        # 达到定义下载期的结束 - 也添加最后一笔交易。
                        if has_overlap:
                            trades.extend(t[-1:])
                        break

                    from_id = from_id_next
                else:
                    logger.debug("停止，因为没有返回更多交易。")
                    break
            except asyncio.CancelledError:
                logger.debug("异步操作中断，中断交易DL循环。")
                break

        return (pair, trades)

    async def _async_get_trade_history_time(
        self, pair: str, until: int, since: int
    ) -> tuple[str, list[list]]:
        """
        使用fetch_trades异步获取交易历史，
        当交易所使用基于时间的迭代时（检查`self._trades_pagination`）
        :param pair: 获取交易数据的交易对
        :param since: 作为毫秒整数时间戳的起始时间
        :param until: 作为毫秒整数时间戳的结束时间
        返回元组: (pair, trades-list)
        """

        trades: list[list] = []
        # DEFAULT_TRADES_COLUMNS: 0 -> timestamp
        # DEFAULT_TRADES_COLUMNS: 1 -> id
        while True:
            try:
                t, since_next = await self._async_fetch_trades(pair, since=since)
                if t:
                    # 交易所没有更多可下载的交易，
                    # 所以我们一遍又一遍地重复获得相同的交易。
                    if since == since_next and len(t) == 1:
                        logger.debug("停止，因为没有更多交易可用。")
                        break
                    since = since_next
                    trades.extend(t)
                    # 达到定义下载期的结束
                    if until and since_next > until:
                        logger.debug(f"停止，因为达到了until。{since_next} > {until}")
                        break
                else:
                    logger.debug("停止，因为没有返回更多交易。")
                    break
            except asyncio.CancelledError:
                logger.debug("异步操作中断，中断交易DL循环。")
                break

        return (pair, trades)

    async def _async_get_trade_history(
        self,
        pair: str,
        since: int,
        until: int | None = None,
        from_id: str | None = None,
    ) -> tuple[str, list[list]]:
        """
        异步包装器处理使用基于时间或id的方法下载交易。
        """

        logger.debug(
            f"_async_get_trade_history()，交易对: {pair}，"
            f"自: {since}，直到: {until}，from_id: {from_id}"
        )

        if until is None:
            until = ccxt.Exchange.milliseconds()
            logger.debug(f"交易所毫秒: {until}")

        if self._trades_pagination == "time":
            return await self._async_get_trade_history_time(pair=pair, since=since, until=until)
        elif self._trades_pagination == "id":
            return await self._async_get_trade_history_id(
                pair=pair, since=since, until=until, from_id=from_id
            )
        else:
            raise OperationalException(
                f"交易所 {self.name} 既不使用基于时间，也不使用基于id的分页"
            )

    def get_historic_trades(
        self,
        pair: str,
        since: int,
        until: int | None = None,
        from_id: str | None = None,
    ) -> tuple[str, list]:
        """
        使用asyncio获取交易历史数据。
        处理所有异步工作并返回蜡烛列表。
        对一个交易对进行异步，假设我们每次调用获得`self.ohlcv_candle_limit()`根蜡烛。
        :param pair: 要下载的交易对
        :param since: 获取历史记录的起始时间戳（毫秒）
        :param until: 毫秒时间戳。如果未定义，默认为当前时间戳。
        :param from_id: 从ID开始下载数据（如果id已知）
        :returns 交易数据列表
        """
        if not self.exchange_has("fetchTrades"):
            raise OperationalException("此交易所不支持下载交易。")

        with self._loop_lock:
            task = asyncio.ensure_future(
                self._async_get_trade_history(pair=pair, since=since, until=until, from_id=from_id)
            )

            for sig in [signal.SIGINT, signal.SIGTERM]:
                try:
                    self.loop.add_signal_handler(sig, task.cancel)
                except NotImplementedError:
                    # 并非所有平台都实现信号（例如windows）
                    pass
            return self.loop.run_until_complete(task)

    @retrier
    def _get_funding_fees_from_exchange(self, pair: str, since: datetime | int) -> float:
        """
        返回在时间框架内为交易对交换的所有资金费用的总和
        模拟运行处理作为_calculate_funding_fees的一部分进行。
        :param pair: (例如 ADA/USDT)
        :param since: 计算资金费用的最早考虑时间，
            以unix时间或datetime形式
        """
        if not self.exchange_has("fetchFundingHistory"):
            raise OperationalException(
                f"使用 {self.name} 时fetch_funding_history()不可用"
            )

        if type(since) is datetime:
            since = dt_ts(since)

        try:
            funding_history = self._api.fetch_funding_history(symbol=pair, since=since)
            self._log_exchange_response(
                "funding_history", funding_history, add_info=f"交易对: {pair}，自: {since}"
            )
            return sum(fee["amount"] for fee in funding_history)
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于 {e.__class__.__name__} 无法获取资金费用。消息: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def get_leverage_tiers(self) -> dict[str, list[dict]]:
        try:
            return self._api.fetch_leverage_tiers()
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于 {e.__class__.__name__} 无法加载杠杆层级。消息: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier_async
    async def get_market_leverage_tiers(self, symbol: str) -> tuple[str, list[dict]]:
        """每个符号的杠杆层级"""
        try:
            tier = await self._api_async.fetch_market_leverage_tiers(symbol)
            return symbol, tier
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于 {e.__class__.__name__} 无法加载 {symbol} 的杠杆层级。"
                f"消息: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def load_leverage_tiers(self) -> dict[str, list[dict]]:
        if self.trading_mode == TradingMode.FUTURES:
            if self.exchange_has("fetchLeverageTiers"):
                # 一次获取所有杠杆层级
                return self.get_leverage_tiers()
            elif self.exchange_has("fetchMarketLeverageTiers"):
                # 必须分别获取每个市场的杠杆层级
                # * 在Okx上这很慢（~45秒），进行~90个api调用以加载所有线性交换市场
                markets = self.markets

                symbols = [
                    symbol
                    for symbol, market in markets.items()
                    if (
                        self.market_is_future(market)
                        and market["quote"] == self._config["stake_currency"]
                    )
                ]

                tiers: dict[str, list[dict]] = {}

                tiers_cached = self.load_cached_leverage_tiers(self._config["stake_currency"])
                if tiers_cached:
                    tiers = tiers_cached

                coros = [
                    self.get_market_leverage_tiers(symbol)
                    for symbol in sorted(symbols)
                    if symbol not in tiers
                ]

                # 在这里要详细，因为这会将启动延迟约1分钟。
                if coros:
                    logger.info(
                        f"为 {len(symbols)} 个市场初始化leverage_tiers。"
                        "这将需要大约一分钟。"
                    )
                else:
                    logger.info("使用缓存的leverage_tiers。")

                async def gather_results(input_coro):
                    return await asyncio.gather(*input_coro, return_exceptions=True)

                for input_coro in chunks(coros, 100):
                    with self._loop_lock:
                        results = self.loop.run_until_complete(gather_results(input_coro))

                    for res in results:
                        if isinstance(res, Exception):
                            logger.warning(f"杠杆层级异常: {repr(res)}")
                            continue
                        symbol, tier = res
                        tiers[symbol] = tier
                if len(coros) > 0:
                    self.cache_leverage_tiers(tiers, self._config["stake_currency"])
                logger.info(f"完成初始化 {len(symbols)} 个市场。")

                return tiers
        return {}

    def cache_leverage_tiers(self, tiers: dict[str, list[dict]], stake_currency: str) -> None:
        filename = self._config["datadir"] / "futures" / f"leverage_tiers_{stake_currency}.json"
        if not filename.parent.is_dir():
            filename.parent.mkdir(parents=True)
        data = {
            "updated": datetime.now(timezone.utc),
            "data": tiers,
        }
        file_dump_json(filename, data)

    def load_cached_leverage_tiers(
        self, stake_currency: str, cache_time: timedelta | None = None
    ) -> dict[str, list[dict]] | None:
        """
        从磁盘加载缓存的杠杆层级
        :param cache_time: 缓存被认为过时之前的最大年龄
        """
        if not cache_time:
            # 默认为4周
            cache_time = timedelta(weeks=4)
        filename = self._config["datadir"] / "futures" / f"leverage_tiers_{stake_currency}.json"
        if filename.is_file():
            try:
                tiers = file_load_json(filename)
                updated = tiers.get("updated")
                if updated:
                    updated_dt = parser.parse(updated)
                    if updated_dt < datetime.now(timezone.utc) - cache_time:
                        logger.info("缓存的杠杆层级已过时。将更新。")
                        return None
                return tiers.get("data")
            except Exception:
                logger.exception("加载缓存的杠杆层级时出错。正在刷新。")
        return None

    def fill_leverage_tiers(self) -> None:
        """
        将属性_leverage_tiers分配给有关每个交易对允许的杠杆的信息字典
        """
        leverage_tiers = self.load_leverage_tiers()
        for pair, tiers in leverage_tiers.items():
            pair_tiers = []
            for tier in tiers:
                pair_tiers.append(self.parse_leverage_tier(tier))
            self._leverage_tiers[pair] = pair_tiers

    def parse_leverage_tier(self, tier) -> dict:
        info = tier.get("info", {})
        return {
            "minNotional": tier["minNotional"],
            "maxNotional": tier["maxNotional"],
            "maintenanceMarginRate": tier["maintenanceMarginRate"],
            "maxLeverage": tier["maxLeverage"],
            "maintAmt": float(info["cum"]) if "cum" in info else None,
        }

    def get_max_leverage(self, pair: str, stake_amount: float | None) -> float:
        """
        返回交易对可以交易的最大杠杆
        :param pair: 正在交易的基础/报价货币对
        :stake_amount: 交易者margin_mode的总价值（以报价货币计）
        """

        if self.trading_mode == TradingMode.SPOT:
            return 1.0

        if self.trading_mode == TradingMode.FUTURES:
            # 检查和边缘情况
            if stake_amount is None:
                raise OperationalException(
                    f"{self.name}.get_max_leverage 需要参数stake_amount"
                )

            if pair not in self._leverage_tiers:
                # 也许引发异常，因为它不能在期货上交易？
                return 1.0

            pair_tiers = self._leverage_tiers[pair]

            if stake_amount == 0:
                return pair_tiers[0]["maxLeverage"]  # 最低金额的最大杠杆

            # 根据stake_amount找到适当的层级
            prior_max_lev = None
            for tier in pair_tiers:
                min_stake = tier["minNotional"] / (prior_max_lev or tier["maxLeverage"])
                max_stake = tier["maxNotional"] / tier["maxLeverage"]
                prior_max_lev = tier["maxLeverage"]
                # 通过杠杆调整名义以进行适当的比较
                if min_stake <= stake_amount <= max_stake:
                    return tier["maxLeverage"]

            #     else:  # 如果在最后一层
            if stake_amount > max_stake:
                # 如果权益 > 最大可交易金额
                raise InvalidOrderException(f"金额 {stake_amount} 对于 {pair} 太高")

            raise OperationalException(
                f"循环遍历所有层级而没有找到 {pair} 的最大杠杆。"
                "永远不应该到达这里。"
            )

        elif self.trading_mode == TradingMode.MARGIN:  # 在markets.limits中搜索最大杠杆
            market = self.markets[pair]
            if market["limits"]["leverage"]["max"] is not None:
                return market["limits"]["leverage"]["max"]
            else:
                return 1.0  # 如果找不到最大杠杆则默认
        else:
            return 1.0

    def _get_max_notional_from_tiers(self, pair: str, leverage: float) -> float | None:
        """
        从leverage_tiers获取max_notional
        :param pair: 正在交易的基础/报价货币对
        :param leverage: 要使用的杠杆
        :return: 给定杠杆的最大名义价值，如果未找到则为None
        """
        if self.trading_mode != TradingMode.FUTURES:
            return None
        if pair not in self._leverage_tiers:
            return None
        pair_tiers = self._leverage_tiers[pair]
        for tier in reversed(pair_tiers):
            if leverage <= tier["maxLeverage"]:
                return tier["maxNotional"]
        return None

    @retrier
    def _set_leverage(
        self,
        leverage: float,
        pair: str | None = None,
        accept_fail: bool = False,
    ):
        """
        在进行交易之前设置杠杆，以便不在每笔交易上都有相同的杠杆
        """
        if self._config["dry_run"] or not self.exchange_has("setLeverage"):
            # 某些交易所仅支持一种margin_mode类型
            return
        if self._ft_has.get("floor_leverage", False) is True:
            # 为binance四舍五入...
            leverage = floor(leverage)
        try:
            res = self._api.set_leverage(symbol=pair, leverage=leverage)
            self._log_exchange_response("set_leverage", res)
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.BadRequest, ccxt.OperationRejected, ccxt.InsufficientFunds) as e:
            if not accept_fail:
                raise TemporaryError(
                    f"由于 {e.__class__.__name__} 无法设置杠杆。消息: {e}"
                ) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于 {e.__class__.__name__} 无法设置杠杆。消息: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def get_interest_rate(self) -> float:
        """
        检索利率 - 保证金交易所必需。
        从回测使用时不应直接调用交易所。
        """
        return 0.0

    def funding_fee_cutoff(self, open_date: datetime) -> bool:
        """
        资金费用仅在整点收取（通常每4-8小时）。
        因此，在10:00:01开仓的交易在下一个小时之前不会收取资金费用。
        :param open_date: 交易的开仓日期
        :return: 如果日期落在整点，则为True，否则为False
        """
        return open_date.minute == 0 and open_date.second == 0

    @retrier
    def set_margin_mode(
        self,
        pair: str,
        margin_mode: MarginMode,
        accept_fail: bool = False,
        params: dict | None = None,
    ):
        """
        在交易所上为特定交易对设置保证金模式为全仓或逐仓
        :param pair: 基础/报价货币对（例如"ADA/USDT"）
        """
        if self._config["dry_run"] or not self.exchange_has("setMarginMode"):
            # 某些交易所仅支持一种margin_mode类型
            return

        if params is None:
            params = {}
        try:
            res = self._api.set_margin_mode(margin_mode.value, pair, params)
            self._log_exchange_response("set_margin_mode", res)
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.BadRequest, ccxt.OperationRejected) as e:
            if not accept_fail:
                raise TemporaryError(
                    f"由于 {e.__class__.__name__} 无法设置保证金模式。消息: {e}"
                ) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于 {e.__class__.__name__} 无法设置保证金模式。消息: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def _fetch_and_calculate_funding_fees(
        self,
        pair: str,
        amount: float,
        is_short: bool,
        open_date: datetime,
        close_date: datetime | None = None,
    ) -> float:
        """
        获取并计算期货交易期间发生的所有资金费用的总和。
        仅在模拟运行期间或交易所不提供funding_rates端点时使用。
        :param pair: 交易的报价/基础对
        :param amount: 交易的数量
        :param is_short: 交易方向
        :param open_date: 交易开始的日期和时间
        :param close_date: 交易结束的日期和时间
        """

        if self.funding_fee_cutoff(open_date):
            # 向后移动到1小时蜡烛以避免错过资金费用
            # 仅对非常接近整点的交易真正相关
            open_date = timeframe_to_prev_date("1h", open_date)
        timeframe = self._ft_has["mark_ohlcv_timeframe"]
        timeframe_ff = self._ft_has["funding_fee_timeframe"]
        mark_price_type = CandleType.from_string(self._ft_has["mark_ohlcv_price"])

        if not close_date:
            close_date = datetime.now(timezone.utc)
        since_ms = dt_ts(timeframe_to_prev_date(timeframe, open_date))

        mark_comb: PairWithTimeframe = (pair, timeframe, mark_price_type)
        funding_comb: PairWithTimeframe = (pair, timeframe_ff, CandleType.FUNDING_RATE)

        candle_histories = self.refresh_latest_ohlcv(
            [mark_comb, funding_comb],
            since_ms=since_ms,
            cache=False,
            drop_incomplete=False,
        )
        try:
            # 我们不能假设我们总是获得历史记录 - 例如在交易所停机期间
            funding_rates = candle_histories[funding_comb]
            mark_rates = candle_histories[mark_comb]
        except KeyError:
            raise ExchangeError("无法找到资金费率。") from None

        funding_mark_rates = self.combine_funding_and_mark(funding_rates, mark_rates)

        return self.calculate_funding_fees(
            funding_mark_rates,
            amount=amount,
            is_short=is_short,
            open_date=open_date,
            close_date=close_date,
        )

    @staticmethod
    def combine_funding_and_mark(
        funding_rates: DataFrame, mark_rates: DataFrame, futures_funding_rate: int | None = None
    ) -> DataFrame:
        """
        组合资金费率和标记费率数据框
        :param funding_rates: 包含资金费率的数据框（类型FUNDING_RATE）
        :param mark_rates: 包含标记费率的数据框（类型mark_ohlcv_price）
        :param futures_funding_rate: 如果funding_rates不可用，使用的假资金费率
        """
        if futures_funding_rate is None:
            return mark_rates.merge(
                funding_rates, on="date", how="inner", suffixes=["_mark", "_fund"]
            )
        else:
            if len(funding_rates) == 0:
                # 没有资金费率蜡烛 - 使用回退变量完全填充
                mark_rates["open_fund"] = futures_funding_rate
                return mark_rates.rename(
                    columns={
                        "open": "open_mark",
                        "close": "close_mark",
                        "high": "high_mark",
                        "low": "low_mark",
                        "volume": "volume_mark",
                    }
                )

            else:
                # 使用回退值填充缺失的funding_rate蜡烛
                combined = mark_rates.merge(
                    funding_rates, on="date", how="left", suffixes=["_mark", "_fund"]
                )
                combined["open_fund"] = combined["open_fund"].fillna(futures_funding_rate)
                return combined

    def calculate_funding_fees(
        self,
        df: DataFrame,
        amount: float,
        is_short: bool,
        open_date: datetime,
        close_date: datetime,
        time_in_ratio: float | None = None,
    ) -> float:
        """
        计算期货交易期间发生的所有资金费用的总和
        :param df: 包含组合资金和标记费率的数据框
                   作为`open_fund`和`open_mark`。
        :param amount: 交易的数量
        :param is_short: 交易方向
        :param open_date: 交易开始的日期和时间
        :param close_date: 交易结束的日期和时间
        :param time_in_ratio: 大多数交易所类不使用
        """
        fees: float = 0

        if not df.empty:
            df1 = df[(df["date"] >= open_date) & (df["date"] <= close_date)]
            fees = sum(df1["open_fund"] * df1["open_mark"] * amount)
        if isnan(fees):
            fees = 0.0
        # 基于实时端点，为多头否定费用，因为funding_fees期望这种方式。
        return fees if is_short else -fees

    def get_funding_fees(
        self, pair: str, amount: float, is_short: bool, open_date: datetime
    ) -> float:
        """
        获取资金费用，从交易所（实时）或基于
        资金费率/标记价格历史计算它们
        :param pair: 交易的报价/基础对
        :param is_short: 交易方向
        :param amount: 交易金额
        :param open_date: 交易的开仓日期
        :return: 自open_date以来的资金费用
        """
        if self.trading_mode == TradingMode.FUTURES:
            try:
                if self._config["dry_run"]:
                    funding_fees = self._fetch_and_calculate_funding_fees(
                        pair, amount, is_short, open_date
                    )
                else:
                    funding_fees = self._get_funding_fees_from_exchange(pair, open_date)
                return funding_fees
            except ExchangeError:
                logger.warning(f"无法更新 {pair} 的资金费用。")

        return 0.0

    def get_liquidation_price(
        self,
        pair: str,
        # 模拟运行
        open_rate: float,  # 仓位的入场价格
        is_short: bool,
        amount: float,  # 仓位大小的绝对值
        stake_amount: float,
        leverage: float,
        wallet_balance: float,
        open_trades: list | None = None,
    ) -> float | None:
        """
        在交易所上为特定交易对设置保证金模式为全仓或逐仓
        """
        if self.trading_mode == TradingMode.SPOT:
            return None
        elif self.trading_mode != TradingMode.FUTURES:
            raise OperationalException(
                f"{self.name} 不支持 {self.margin_mode} {self.trading_mode}"
            )

        liquidation_price = None
        if self._config["dry_run"] or not self.exchange_has("fetchPositions"):
            liquidation_price = self.dry_run_liquidation_price(
                pair=pair,
                open_rate=open_rate,
                is_short=is_short,
                amount=amount,
                leverage=leverage,
                stake_amount=stake_amount,
                wallet_balance=wallet_balance,
                open_trades=open_trades or [],
            )
        else:
            positions = self.fetch_positions(pair)
            if len(positions) > 0:
                pos = positions[0]
                liquidation_price = pos["liquidationPrice"]

        if liquidation_price is not None:
            buffer_amount = abs(open_rate - liquidation_price) * self.liquidation_buffer
            liquidation_price_buffer = (
                liquidation_price - buffer_amount if is_short else liquidation_price + buffer_amount
            )
            return max(liquidation_price_buffer, 0.0)
        else:
            return None

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
        重要：必须从缓存值中获取数据，因为这被回测使用！
        永续合约：
         gate: https://www.gate.io/help/futures/futures/27724/liquidation-price-bankruptcy-price
         > 清算价格 = (入场价格 ± 保证金 / 合约乘数 / 大小) /
                                [ 1 ± (维持保证金率 + Taker费率)]
            其中，"+"或"-"取决于合约是做多还是做空：
            做多为"-"，做空为"+"。

         okx: https://www.okx.com/support/hc/en-us/articles/
            360053909592-VI-Introduction-to-the-isolated-mode-of-Single-Multi-currency-Portfolio-margin

        :param pair: 计算清算价格的交易对
        :param open_rate: 仓位的入场价格
        :param is_short: 如果交易是空头则为True，否则为false
        :param amount: 仓位大小的绝对值，包括杠杆（以基础货币计）
        :param stake_amount: 权益金额 - 以结算货币计的抵押品。
        :param leverage: 此仓位使用的杠杆。
        :param wallet_balance: 用于交易的钱包中的margin_mode金额
            全仓保证金模式：crossWalletBalance
            逐仓保证金模式：isolatedWalletBalance
        :param open_trades: 同一钱包中其他开放交易的列表
        """

        market = self.markets[pair]
        taker_fee_rate = market["taker"]
        mm_ratio, _ = self.get_maintenance_ratio_and_amt(pair, stake_amount)

        if self.trading_mode == TradingMode.FUTURES and self.margin_mode == MarginMode.ISOLATED:
            if market["inverse"]:
                raise OperationalException("Freqtrade尚不支持反向合约")

            value = wallet_balance / amount

            mm_ratio_taker = mm_ratio + taker_fee_rate
            if is_short:
                return (open_rate + value) / (1 + mm_ratio_taker)
            else:
                return (open_rate - value) / (1 - mm_ratio_taker)
        else:
            raise OperationalException(
                "Freqtrade仅支持杠杆交易的逐仓期货"
            )

    def get_maintenance_ratio_and_amt(
        self,
        pair: str,
        notional_value: float,
    ) -> tuple[float, float | None]:
        """
        重要：必须从缓存值中获取数据，因为这被回测使用！
        :param pair: 市场符号
        :param notional_value: 以报价货币计的总交易金额
        :return: (维持保证金率, 维持金额)
        """

        if (
            self._config.get("runmode") in OPTIMIZE_MODES
            or self.exchange_has("fetchLeverageTiers")
            or self.exchange_has("fetchMarketLeverageTiers")
        ):
            if pair not in self._leverage_tiers:
                raise InvalidOrderException(
                    f"{self.name} 上 {pair} 的维持保证金率不可用"
                )

            pair_tiers = self._leverage_tiers[pair]

            for tier in reversed(pair_tiers):
                if notional_value >= tier["minNotional"]:
                    return (tier["maintenanceMarginRate"], tier["maintAmt"])

            raise ExchangeError("名义价值不能低于 0")
            # fetch_leverage_tiers 中任何交易对的最低 notional_floor 始终为 0，因为它
            # 描述了层级的最小金额，而最低层级总是会降到 0
        else:
            raise ExchangeError(f"无法使用 {self.name} 获取维持率")