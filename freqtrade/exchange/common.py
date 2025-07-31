import asyncio
import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast, overload

from freqtrade.exceptions import DDosProtection, RetryableOrderError, TemporaryError
from freqtrade.mixins import LoggingMixin


logger = logging.getLogger(__name__)
__logging_mixin = None


def _reset_logging_mixin():
    """
    重置全局日志混合类 - 仅用于测试。
    """
    global __logging_mixin
    __logging_mixin = LoggingMixin(logger)


def _get_logging_mixin():
    # 用于缓存kucoin响应的日志混合类
    # 仅在重试器中使用
    global __logging_mixin
    if not __logging_mixin:
        __logging_mixin = LoggingMixin(logger)
    return __logging_mixin


# 默认最大重试次数。
# 函数总是被调用 RETRY_COUNT + 1 次（包括原始调用）
API_RETRY_COUNT = 4
API_FETCH_ORDER_RETRY_COUNT = 5

BAD_EXCHANGES = {
    "bitmex": "多种原因。",
    "probit": "需要定期额外调用 `signIn()`。",
    "poloniex": "不提供fetch_order端点来获取未结和已结订单。",
    "kucoinfutures": "不支持的期货交易所。",
    "poloniexfutures": "不支持的期货交易所。",
    "binancecoinm": "不支持的期货交易所。",
}

MAP_EXCHANGE_CHILDCLASS = {
    "binanceus": "binance",
    "binanceje": "binance",
    "binanceusdm": "binance",
    "okex": "okx",
    "myokx": "okx",
    "gateio": "gate",
    "huboi": "htx",
}

SUPPORTED_EXCHANGES = [
    "binance",
    "bingx",
    "bitmart",
    "bybit",
    "gate",
    "htx",
    "hyperliquid",
    "kraken",
    "okx",
]

# 主方法或替换方法（数组）是必需的
EXCHANGE_HAS_REQUIRED: dict[str, list[str]] = {
    # 必需的 / 私有
    "fetchOrder": ["fetchOpenOrder", "fetchClosedOrder"],
    "fetchL2OrderBook": ["fetchTicker"],
    "cancelOrder": [],
    "createOrder": [],
    "fetchBalance": [],
    # 公共端点
    "fetchOHLCV": [],
}

EXCHANGE_HAS_OPTIONAL = [
    # 私有
    "fetchMyTrades",  # 订单交易 - 费用检测
    "createLimitOrder",
    "createMarketOrder",  # 订单的任一OR
    # 'setLeverage',  # 保证金/期货交易
    # 'setMarginMode',  # 保证金/期货交易
    # 'fetchFundingHistory', # 期货交易
    # 公共
    "fetchOrderBook",
    "fetchL2OrderBook",
    "fetchTicker",  # 定价的OR
    "fetchTickers",  # 用于volumepairlist?
    "fetchTrades",  # 下载交易数据
    # 'fetchFundingRateHistory',  # 期货交易
    # 'fetchPositions',  # 期货交易
    # 'fetchLeverageTiers',  # 期货初始化
    # 'fetchMarketLeverageTiers',  # 期货初始化
    # 'fetchOpenOrder', 'fetchClosedOrder',  # fetchOrder的替代
    # 'fetchOpenOrders', 'fetchClosedOrders',  # 'fetchOrders',  # 重新平衡...
    # ccxt.pro
    "watchOHLCV",
]


def calculate_backoff(retrycount, max_retries):
    """
    计算退避时间
    """
    return (max_retries - retrycount) ** 2 + 1


def retrier_async(f):
    async def wrapper(*args, **kwargs):
        count = kwargs.pop("count", API_RETRY_COUNT)
        kucoin = args[0].name == "KuCoin"  # 检查交易所是否为KuCoin
        try:
            return await f(*args, **kwargs)
        except TemporaryError as ex:
            msg = f'{f.__name__}() 返回异常: "{ex}". '
            if count > 0:
                msg += f"还将重试 {count} 次。"
                count -= 1
                kwargs["count"] = count
                if isinstance(ex, DDosProtection):
                    if kucoin and "429000" in str(ex):
                        # 临时修复kucoin上的429000错误
                        # 详情见 https://github.com/freqtrade/freqtrade/issues/5700
                        _get_logging_mixin().log_once(
                            f"Kucoin 429错误，避免触发DDosProtection退避延迟。"
                            f"放弃前还剩 {count} 次尝试",
                            logmethod=logger.warning,
                        )
                        # 重置消息以避免过多日志
                        msg = ""
                    else:
                        backoff_delay = calculate_backoff(count + 1, API_RETRY_COUNT)
                        logger.info(f"应用DDosProtection退避延迟: {backoff_delay}")
                        await asyncio.sleep(backoff_delay)
                if msg:
                    logger.warning(msg)
                return await wrapper(*args, **kwargs)
            else:
                logger.warning(msg + "放弃。")
                raise ex

    return wrapper


F = TypeVar("F", bound=Callable[..., Any])


# 类型处理
@overload
def retrier(_func: F) -> F: ...


@overload
def retrier(_func: F, *, retries=API_RETRY_COUNT) -> F: ...


@overload
def retrier(*, retries=API_RETRY_COUNT) -> Callable[[F], F]: ...


def retrier(_func: F | None = None, *, retries=API_RETRY_COUNT):
    def decorator(f: F) -> F:
        @wraps(f)
        def wrapper(*args, **kwargs):
            count = kwargs.pop("count", retries)
            try:
                return f(*args, **kwargs)
            except (TemporaryError, RetryableOrderError) as ex:
                msg = f'{f.__name__}() 返回异常: "{ex}". '
                if count > 0:
                    logger.warning(msg + f"还将重试 {count} 次。")
                    count -= 1
                    kwargs.update({"count": count})
                    if isinstance(ex, DDosProtection | RetryableOrderError):
                        # 递增退避
                        backoff_delay = calculate_backoff(count + 1, retries)
                        logger.info(f"应用DDosProtection退避延迟: {backoff_delay}")
                        time.sleep(backoff_delay)
                    return wrapper(*args, **kwargs)
                else:
                    logger.warning(msg + "放弃。")
                    raise ex

        return cast(F, wrapper)

    # 支持 @retrier 和 @retrier(retries=2) 语法
    if _func is None:
        return decorator
    else:
        return decorator(_func)