from typing import Any, Literal, TypedDict

from freqtrade.enums import CandleType


class FtHas(TypedDict, total=False):
    order_time_in_force: list[str]  # 订单有效期类型列表
    exchange_has_overrides: dict[str, bool]  # 交易所方法覆盖配置
    marketOrderRequiresPrice: bool  # 市价单是否需要价格

    # 交易所止损相关配置
    stoploss_on_exchange: bool  # 是否支持交易所止损
    stop_price_param: str  # 止损价格参数名
    stop_price_prop: Literal["stopPrice", "stopLossPrice"]  # 止损价格属性名
    stop_price_type_field: str  # 止损价格类型字段名
    stop_price_type_value_mapping: dict  # 止损价格类型值映射
    stoploss_order_types: dict[str, str]  # 止损订单类型映射
    stoploss_blocks_assets: bool  # 止损订单是否锁定资产

    # K线数据相关配置
    ohlcv_params: dict  # OHLCV请求参数
    ohlcv_candle_limit: int  # OHLCV蜡烛图数据最大限制
    ohlcv_has_history: bool  # 是否支持OHLCV历史数据
    ohlcv_partial_candle: bool  # 是否包含未完成的蜡烛图
    ohlcv_require_since: bool  # 是否需要since参数
    ohlcv_volume_currency: str  # 成交量货币单位
    ohlcv_candle_limit_per_timeframe: dict[str, int]  # 每个时间框架的蜡烛图限制

    # Ticker数据相关配置
    tickers_have_quoteVolume: bool  # Ticker是否包含报价成交量
    tickers_have_percentage: bool  # Ticker是否包含涨跌幅百分比
    tickers_have_bid_ask: bool  # Ticker是否包含买卖价
    tickers_have_price: bool  # Ticker是否包含价格

    # 交易历史相关配置
    trades_limit: int  # 交易历史请求限制
    trades_pagination: str  # 交易历史分页方式
    trades_pagination_arg: str  # 交易历史分页参数名
    trades_has_history: bool  # 是否支持交易历史
    trades_pagination_overlap: bool  # 分页是否有重叠

    # 订单簿相关配置
    l2_limit_range: list[int] | None  # L2订单簿深度范围
    l2_limit_range_required: bool  # 是否必须指定订单簿深度
    l2_limit_upper: int | None  # 订单簿最大深度

    # 订单查询相关配置
    fetch_orders_limit_minutes: int | None  # 查询订单的时间限制(分钟)

    # 期货相关配置
    ccxt_futures_name: str  # 通常为swap，ccxt期货名称
    mark_ohlcv_price: str  # 标记价格的K线字段
    mark_ohlcv_timeframe: str  # 标记价格的K线时间框架
    funding_fee_timeframe: str  # 资金费用的时间框架
    funding_fee_candle_limit: int  # 资金费用蜡烛图限制
    floor_leverage: bool  # 是否向下取整杠杆
    uses_leverage_tiers: bool  # 是否使用杠杆等级
    needs_trading_fees: bool  # 是否需要交易费用
    order_props_in_contracts: list[Literal["amount", "cost", "filled", "remaining"]]  # 合约单位的订单属性

    proxy_coin_mapping: dict[str, str]  # 代理货币映射

    # WebSocket相关配置
    ws_enabled: bool  # 是否支持WebSocket


class Ticker(TypedDict):
    symbol: str  # 交易对
    ask: float | None  # 卖价
    askVolume: float | None  # 卖量
    bid: float | None  # 买价
    bidVolume: float | None  # 买量
    last: float | None  # 最新成交价
    quoteVolume: float | None  # 报价货币成交量
    baseVolume: float | None  # 基础货币成交量
    percentage: float | None  # 涨跌幅百分比
    # 还有其他字段 - 只列出必要的


Tickers = dict[str, Ticker]  # 多个交易对的Ticker数据


class OrderBook(TypedDict):
    symbol: str  # 交易对
    bids: list[tuple[float, float]]  # 买单列表(价格, 数量)
    asks: list[tuple[float, float]]  # 卖单列表(价格, 数量)
    timestamp: int | None  # 时间戳
    datetime: str | None  # 日期时间字符串
    nonce: int | None  # 随机数


class CcxtBalance(TypedDict):
    free: float  # 可用余额
    used: float  # 已用余额
    total: float  # 总余额


CcxtBalances = dict[str, CcxtBalance]  # 多个货币的余额数据


class CcxtPosition(TypedDict):
    symbol: str  # 交易对
    side: str  # 方向
    contracts: float  # 合约数量
    leverage: float  # 杠杆倍数
    collateral: float | None  # 抵押品
    initialMargin: float | None  # 初始保证金
    liquidationPrice: float | None  # 强平价格


CcxtOrder = dict[str, Any]  # CCXT订单数据

# 交易对, 时间框架, 蜡烛图类型, OHLCV数据, 是否删除最后一根蜡烛
OHLCVResponse = tuple[str, str, CandleType, list, bool]