"""
交易所支持工具
"""

import inspect
from datetime import datetime, timedelta, timezone
from math import ceil, floor, isnan
from typing import Any

import ccxt
from ccxt import (
    DECIMAL_PLACES,
    ROUND,
    ROUND_DOWN,
    ROUND_UP,
    SIGNIFICANT_DIGITS,
    TICK_SIZE,
    TRUNCATE,
    decimal_to_precision,
)

from freqtrade.exchange.common import (
    BAD_EXCHANGES,
    EXCHANGE_HAS_OPTIONAL,
    EXCHANGE_HAS_REQUIRED,
    MAP_EXCHANGE_CHILDCLASS,
    SUPPORTED_EXCHANGES,
)
from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.ft_types import ValidExchangesType
from freqtrade.util import FtPrecise


CcxtModuleType = Any


def is_exchange_known_ccxt(exchange_name: str, ccxt_module: CcxtModuleType | None = None) -> bool:
    """判断交易所是否为ccxt已知的交易所"""
    return exchange_name in ccxt_exchanges(ccxt_module)


def ccxt_exchanges(ccxt_module: CcxtModuleType | None = None) -> list[str]:
    """
    返回ccxt已知的所有交易所列表
    """
    return ccxt_module.exchanges if ccxt_module is not None else ccxt.exchanges


def available_exchanges(ccxt_module: CcxtModuleType | None = None) -> list[str]:
    """
    返回机器人可用的交易所，即ccxt列表中非问题交易所
    """
    exchanges = ccxt_exchanges(ccxt_module)
    return [x for x in exchanges if validate_exchange(x)[0]]


def validate_exchange(exchange: str) -> tuple[bool, str, ccxt.Exchange | None]:
    """
    返回: 能否使用, 原因, 交易所对象
        原因包括缺失的必要功能和缺失的可选功能
    """
    try:
        ex_mod = getattr(ccxt.pro, exchange.lower())()
    except AttributeError:
        ex_mod = getattr(ccxt.async_support, exchange.lower())()

    if not ex_mod or not ex_mod.has:
        return False, "", None

    result = True
    reason = ""
    missing = [
        k
        for k, v in EXCHANGE_HAS_REQUIRED.items()
        if ex_mod.has.get(k) is not True and not (all(ex_mod.has.get(x) for x in v))
    ]
    if missing:
        result = False
        reason += f"缺失: {', '.join(missing)}"

    missing_opt = [k for k in EXCHANGE_HAS_OPTIONAL if not ex_mod.has.get(k)]

    if exchange.lower() in BAD_EXCHANGES:
        result = False
        reason = BAD_EXCHANGES.get(exchange.lower(), "")

    if missing_opt:
        reason += f"{'. ' if reason else ''}缺失可选功能: {', '.join(missing_opt)}. "

    return result, reason, ex_mod


def _build_exchange_list_entry(
    exchange_name: str, exchangeClasses: dict[str, Any]
) -> ValidExchangesType:
    exchange_name = exchange_name.lower()
    valid, comment, ex_mod = validate_exchange(exchange_name)
    mapped_exchange_name = MAP_EXCHANGE_CHILDCLASS.get(exchange_name, exchange_name).lower()
    is_alias = getattr(ex_mod, "alias", False)
    result: ValidExchangesType = {
        "name": getattr(ex_mod, "name", exchange_name),
        "classname": exchange_name,
        "valid": valid,
        "supported": mapped_exchange_name in SUPPORTED_EXCHANGES and not is_alias,
        "comment": comment,
        "dex": getattr(ex_mod, "dex", False),
        "is_alias": is_alias,
        "alias_for": inspect.getmro(ex_mod.__class__)[1]().id
        if getattr(ex_mod, "alias", False)
        else None,
        "trade_modes": [{"trading_mode": "spot", "margin_mode": ""}],
    }
    if resolved := exchangeClasses.get(mapped_exchange_name):
        supported_modes = [{"trading_mode": "spot", "margin_mode": ""}] + [
            {"trading_mode": tm.value, "margin_mode": mm.value}
            for tm, mm in resolved["class"]._supported_trading_mode_margin_pairs
        ]
        result.update(
            {
                "trade_modes": supported_modes,
            }
        )

    return result


def list_available_exchanges(all_exchanges: bool) -> list[ValidExchangesType]:
    """
    :return: 交易所列表，包含交易所名称、是否有效、原因等信息的元组
    """
    exchanges = ccxt_exchanges() if all_exchanges else available_exchanges()
    from freqtrade.resolvers.exchange_resolver import ExchangeResolver

    subclassed = {e["name"].lower(): e for e in ExchangeResolver.search_all_objects({}, False)}

    exchanges_valid: list[ValidExchangesType] = [
        _build_exchange_list_entry(e, subclassed) for e in exchanges
    ]

    return exchanges_valid


def date_minus_candles(timeframe: str, candle_count: int, date: datetime | None = None) -> datetime:
    """
    从日期中减去X根蜡烛图。
    :param timeframe: 字符串格式的时间框架（例如"5m"）
    :param candle_count: 要减去的蜡烛图数量。
    :param date: 要使用的日期。默认为当前时间(utc)

    """
    if not date:
        date = datetime.now(timezone.utc)

    tf_min = timeframe_to_minutes(timeframe)
    new_date = timeframe_to_prev_date(timeframe, date) - timedelta(minutes=tf_min * candle_count)
    return new_date


def market_is_active(market: dict) -> bool:
    """
    返回市场是否活跃。
    """
    # "如果active标志没有明确设置为false，则市场是活跃的。如果该标志缺失或
    # 为true，则视为活跃。如果未定义，则很可能是活跃的，但不能100%确定)"
    # 参见 https://github.com/ccxt/ccxt/issues/4874,
    # https://github.com/ccxt/ccxt/issues/4075#issuecomment-434760520
    return market.get("active", True) is not False


def amount_to_contracts(amount: float, contract_size: float | None) -> float:
    """
    将数量转换为合约数量。
    :param amount: 要转换的数量
    :param contract_size: 合约大小 - 从exchange.get_contract_size(pair)获取
    :return: 合约数量
    """
    if contract_size and contract_size != 1:
        return float(FtPrecise(amount) / FtPrecise(contract_size))
    else:
        return amount


def contracts_to_amount(num_contracts: float, contract_size: float | None) -> float:
    """
    将合约数量转换为实际数量
    :param num_contracts: 合约数量
    :param contract_size: 合约大小 - 从exchange.get_contract_size(pair)获取
    :return: 实际数量
    """

    if contract_size and contract_size != 1:
        return float(FtPrecise(num_contracts) * FtPrecise(contract_size))
    else:
        return num_contracts


def amount_to_precision(
    amount: float, amount_precision: float | None, precisionMode: int | None
) -> float:
    """
    将买入或卖出数量调整为交易所接受的精度
    重新实现ccxt内部方法 - 确保我们可以根据定义测试结果是否正确
    :param amount: 要截断的数量
    :param amount_precision: 要使用的数量精度。
                             应从markets[pair]['precision']['amount']获取
    :param precisionMode: 要使用的精度模式。应使用precisionMode
                          为ccxt的DECIMAL_PLACES、SIGNIFICANT_DIGITS或TICK_SIZE之一
    :return: 截断后的数量
    """
    if amount_precision is not None and precisionMode is not None:
        precision = int(amount_precision) if precisionMode != TICK_SIZE else amount_precision
        # 对于非tick_size输入，precision必须是整数。
        amount = float(
            decimal_to_precision(
                amount,
                rounding_mode=TRUNCATE,
                precision=precision,
                counting_mode=precisionMode,
            )
        )

    return amount


def amount_to_contract_precision(
    amount,
    amount_precision: float | None,
    precisionMode: int | None,
    contract_size: float | None,
) -> float:
    """
    将买入或卖出数量调整为交易所接受的精度
    包括与合约的相互转换计算。
    重新实现ccxt内部方法 - 确保我们可以根据定义测试结果是否正确
    :param amount: 要截断的数量
    :param amount_precision: 要使用的数量精度。
                             应从markets[pair]['precision']['amount']获取
    :param precisionMode: 要使用的精度模式。应使用precisionMode
                          为ccxt的DECIMAL_PLACES、SIGNIFICANT_DIGITS或TICK_SIZE之一
    :param contract_size: 合约大小 - 从exchange.get_contract_size(pair)获取
    :return: 截断后的数量
    """
    if amount_precision is not None and precisionMode is not None:
        contracts = amount_to_contracts(amount, contract_size)
        amount_p = amount_to_precision(contracts, amount_precision, precisionMode)
        return contracts_to_amount(amount_p, contract_size)
    return amount


def __price_to_precision_significant_digits(
    price: float,
    price_precision: float,
    *,
    rounding_mode: int = ROUND,
) -> float:
    """
    有效数字模式下的ROUND_UP/Round_down实现。
    """
    from decimal import ROUND_DOWN as dec_ROUND_DOWN
    from decimal import ROUND_UP as dec_ROUND_UP
    from decimal import Decimal

    dec = Decimal(str(price))
    string = f"{dec:f}"
    precision = round(price_precision)

    q = precision - dec.adjusted() - 1
    sigfig = Decimal("10") ** -q
    if q < 0:
        string_to_precision = string[:precision]
        # 当精度为零时，string_to_precision为空
        below = sigfig * Decimal(string_to_precision if string_to_precision else "0")
        above = below + sigfig
        res = above if rounding_mode == ROUND_UP else below
        precise = f"{res:f}"
    else:
        precise = "{:f}".format(
            dec.quantize(
                sigfig, rounding=dec_ROUND_DOWN if rounding_mode == ROUND_DOWN else dec_ROUND_UP
            )
        )
    return float(precise)


def price_to_precision(
    price: float,
    price_precision: float | None,
    precisionMode: int | None,
    *,
    rounding_mode: int = ROUND,
) -> float:
    """
    将价格四舍五入到交易所接受的精度。
    部分重新实现ccxt内部方法decimal_to_precision()，
    该方法不支持向上取整。
    对于止损计算，多头必须使用ROUND_UP，空头必须使用ROUND_DOWN。

    TODO: 如果ccxt的decimal_to_precision()支持ROUND_UP，我们可以删除此方法并
    与amount_to_precision()保持一致。
    :param price: 要转换的价格
    :param price_precision: 要使用的价格精度。从markets[pair]['precision']['price']获取
    :param precisionMode: 要使用的精度模式。应使用precisionMode
                          为ccxt的DECIMAL_PLACES、SIGNIFICANT_DIGITS或TICK_SIZE之一
    :param rounding_mode: 要使用的取整模式。默认为ROUND
    :return: 向上取整到交易所接受精度的价格
    """
    if price_precision is not None and precisionMode is not None and not isnan(price):
        if rounding_mode not in (ROUND_UP, ROUND_DOWN):
            # 尽可能使用CCXT代码。
            return float(
                decimal_to_precision(
                    price,
                    rounding_mode=rounding_mode,
                    precision=int(price_precision)
                    if precisionMode != TICK_SIZE
                    else price_precision,
                    counting_mode=precisionMode,
                )
            )

        if precisionMode == TICK_SIZE:
            precision = FtPrecise(price_precision)
            price_str = FtPrecise(price)
            missing = price_str % precision
            if not missing == FtPrecise("0"):
                if rounding_mode == ROUND_UP:
                    res = price_str - missing + precision
                elif rounding_mode == ROUND_DOWN:
                    res = price_str - missing
                return round(float(str(res)), 14)
            return price
        elif precisionMode == DECIMAL_PLACES:
            ndigits = round(price_precision)
            ticks = price * (10**ndigits)
            if rounding_mode == ROUND_UP:
                return ceil(ticks) / (10**ndigits)
            if rounding_mode == ROUND_DOWN:
                return floor(ticks) / (10**ndigits)

            raise ValueError(f"未知的取整模式 {rounding_mode}")
        elif precisionMode == SIGNIFICANT_DIGITS:
            if rounding_mode in (ROUND_UP, ROUND_DOWN):
                return __price_to_precision_significant_digits(
                    price, price_precision, rounding_mode=rounding_mode
                )

        raise ValueError(f"未知的精度模式 {precisionMode}")
    return price