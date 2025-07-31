# pragma pylint: disable=too-few-public-methods

"""
机器人常量
"""

from typing import Any, Literal

from freqtrade.enums import CandleType, PriceType


DOCS_LINK = "https://www.freqtrade.io/en/stable"
DEFAULT_CONFIG = "config.json"
PROCESS_THROTTLE_SECS = 5  # 秒
HYPEROPT_EPOCH = 100  # 轮次
RETRY_TIMEOUT = 30  # 秒
TIMEOUT_UNITS = ["minutes", "seconds"]
EXPORT_OPTIONS = ["none", "trades", "signals"]
DEFAULT_DB_PROD_URL = "sqlite:///tradesv3.sqlite"
DEFAULT_DB_DRYRUN_URL = "sqlite:///tradesv3.dryrun.sqlite"
UNLIMITED_STAKE_AMOUNT = "unlimited"
DEFAULT_AMOUNT_RESERVE_PERCENT = 0.05
REQUIRED_ORDERTIF = ["entry", "exit"]
REQUIRED_ORDERTYPES = ["entry", "exit", "stoploss", "stoploss_on_exchange"]
PRICING_SIDES = ["ask", "bid", "same", "other"]
ORDERTYPE_POSSIBILITIES = ["limit", "market"]
_ORDERTIF_POSSIBILITIES = ["GTC", "FOK", "IOC", "PO"]
ORDERTIF_POSSIBILITIES = _ORDERTIF_POSSIBILITIES + [t.lower() for t in _ORDERTIF_POSSIBILITIES]
STOPLOSS_PRICE_TYPES = [p for p in PriceType]
HYPEROPT_LOSS_BUILTIN = [
    "ShortTradeDurHyperOptLoss",
    "OnlyProfitHyperOptLoss",
    "SharpeHyperOptLoss",
    "SharpeHyperOptLossDaily",
    "SortinoHyperOptLoss",
    "SortinoHyperOptLossDaily",
    "CalmarHyperOptLoss",
    "MaxDrawDownHyperOptLoss",
    "MaxDrawDownRelativeHyperOptLoss",
    "MaxDrawDownPerPairHyperOptLoss",
    "ProfitDrawDownHyperOptLoss",
    "MultiMetricHyperOptLoss",
]
AVAILABLE_PAIRLISTS = [
    "StaticPairList",
    "VolumePairList",
    "PercentChangePairList",
    "ProducerPairList",
    "RemotePairList",
    "MarketCapPairList",
    "AgeFilter",
    "FullTradesFilter",
    "OffsetFilter",
    "PerformanceFilter",
    "PrecisionFilter",
    "PriceFilter",
    "RangeStabilityFilter",
    "ShuffleFilter",
    "SpreadFilter",
    "VolatilityFilter",
]
AVAILABLE_DATAHANDLERS = ["json", "jsongz", "feather", "parquet"]
BACKTEST_BREAKDOWNS = ["day", "week", "month", "year"]
BACKTEST_CACHE_AGE = ["none", "day", "week", "month"]
BACKTEST_CACHE_DEFAULT = "day"
DRY_RUN_WALLET = 1000
DATETIME_PRINT_FORMAT = "%Y-%m-%d %H:%M:%S"
MATH_CLOSE_PREC = 1e-14  # 用于浮点数比较的精度
DEFAULT_DATAFRAME_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
# 不要修改 DEFAULT_TRADES_COLUMNS 的顺序
# 这对存储的交易文件有广泛影响
DEFAULT_TRADES_COLUMNS = ["timestamp", "id", "type", "side", "price", "amount", "cost"]
DEFAULT_ORDERFLOW_COLUMNS = ["level", "bid", "ask", "delta"]
ORDERFLOW_ADDED_COLUMNS = [
    "trades",
    "orderflow",
    "imbalances",
    "stacked_imbalances_bid",
    "stacked_imbalances_ask",
    "max_delta",
    "min_delta",
    "bid",
    "ask",
    "delta",
    "total_trades",
]
TRADES_DTYPES = {
    "timestamp": "int64",
    "id": "str",
    "type": "str",
    "side": "str",
    "price": "float64",
    "amount": "float64",
    "cost": "float64",
}
TRADING_MODES = ["spot", "margin", "futures"]
MARGIN_MODES = ["cross", "isolated", ""]

LAST_BT_RESULT_FN = ".last_result.json"
FTHYPT_FILEVERSION = "fthypt_fileversion"

USERPATH_HYPEROPTS = "hyperopts"  # 超参数优化目录
USERPATH_STRATEGIES = "strategies"  # 策略目录
USERPATH_NOTEBOOKS = "notebooks"  # 笔记本目录
USERPATH_FREQAIMODELS = "freqaimodels"  # FreqAI模型目录

TELEGRAM_SETTING_OPTIONS = ["on", "off", "silent"]
WEBHOOK_FORMAT_OPTIONS = ["form", "json", "raw"]
FULL_DATAFRAME_THRESHOLD = 100
CUSTOM_TAG_MAX_LENGTH = 255
DL_DATA_TIMEFRAMES = ["1m", "5m"]

ENV_VAR_PREFIX = "FREQTRADE__"

CANCELED_EXCHANGE_STATES = ("cancelled", "canceled", "expired", "rejected")
NON_OPEN_EXCHANGE_STATES = (*CANCELED_EXCHANGE_STATES, "closed")

# 定义每个币种的输出小数位数
# 仅用于输出。
DECIMAL_PER_COIN_FALLBACK = 3  # 应该设置较低以避免列出所有可能的法币
DECIMALS_PER_COIN = {
    "BTC": 8,
    "ETH": 5,
}

DUST_PER_COIN = {"BTC": 0.0001, "ETH": 0.01}

# 源文件及其在用户目录中的目标目录
USER_DATA_FILES = {
    "sample_strategy.py": USERPATH_STRATEGIES,
    "sample_hyperopt_loss.py": USERPATH_HYPEROPTS,
    "strategy_analysis_example.ipynb": USERPATH_NOTEBOOKS,
}

SUPPORTED_FIAT = [
    "AUD",
    "BRL",
    "CAD",
    "CHF",
    "CLP",
    "CNY",
    "CZK",
    "DKK",
    "EUR",
    "GBP",
    "HKD",
    "HUF",
    "IDR",
    "ILS",
    "INR",
    "JPY",
    "KRW",
    "MXN",
    "MYR",
    "NOK",
    "NZD",
    "PHP",
    "PKR",
    "PLN",
    "RUB",
    "UAH",
    "SEK",
    "SGD",
    "THB",
    "TRY",
    "TWD",
    "ZAR",
    "USD",
    "BTC",
    "ETH",
    "XRP",
    "LTC",
    "BCH",
    "BNB",
    "",  # 允许配置中的空字段。
]

MINIMAL_CONFIG = {
    "stake_currency": "",
    "dry_run": True,
    "exchange": {
        "name": "",
        "key": "",
        "secret": "",
        "pair_whitelist": [],
        "ccxt_async_config": {},
    },
}


CANCEL_REASON = {
    "TIMEOUT": "cancelled due to timeout",  # 因超时取消
    "PARTIALLY_FILLED_KEEP_OPEN": "partially filled - keeping order open",  # 部分成交 - 保持订单开放
    "PARTIALLY_FILLED": "partially filled",  # 部分成交
    "FULLY_CANCELLED": "fully cancelled",  # 完全取消
    "ALL_CANCELLED": "cancelled (all unfilled and partially filled open orders cancelled)",  # 取消（所有未成交和部分成交的开放订单已取消）
    "CANCELLED_ON_EXCHANGE": "cancelled on exchange",  # 在交易所取消
    "FORCE_EXIT": "forcesold",  # 强制卖出
    "REPLACE": "cancelled to be replaced by new limit order",  # 取消以被新的限价单替换
    "REPLACE_FAILED": "failed to replace order, deleting Trade",  # 替换订单失败，删除交易
    "USER_CANCEL": "user requested order cancel",  # 用户请求取消订单
}

# 交易对及其时间框架的列表
PairWithTimeframe = tuple[str, str, CandleType]
ListPairsWithTimeframes = list[PairWithTimeframe]

# 交易列表的类型
TradeList = list[list]
# ticks, 交易对, 时间框架, 蜡烛类型
TickWithTimeframe = tuple[str, str, CandleType, int | None, int | None]
ListTicksWithTimeframes = list[TickWithTimeframe]

LongShort = Literal["long", "short"]  # 多空
EntryExit = Literal["entry", "exit"]  # 进场/出场
BuySell = Literal["buy", "sell"]  # 买/卖
MakerTaker = Literal["maker", "taker"]  # 挂单方/吃单方
BidAsk = Literal["bid", "ask"]  # 买价/卖价
OBLiteral = Literal["asks", "bids"]  # 卖单/买单

Config = dict[str, Any]  # 配置
# 配置的交易所部分。
ExchangeConfig = dict[str, Any]  # 交易所配置
IntOrInf = float


EntryExecuteMode = Literal["initial", "pos_adjust", "replace"]  # 进场执行模式：初始/仓位调整/替换