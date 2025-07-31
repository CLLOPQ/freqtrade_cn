from datetime import timedelta

from freqtrade.constants import DECIMAL_PER_COIN_FALLBACK, DECIMALS_PER_COIN


def decimals_per_coin(coin: str):
    """
    辅助方法，获取此币种的小数位数
    示例用法: f".{decimals_per_coin('USD')}f"
    :param coin: 要打印价格/值的币种
    """
    return DECIMALS_PER_COIN.get(coin, DECIMAL_PER_COIN_FALLBACK)


def strip_trailing_zeros(value: str) -> str:
    """
    从字符串中移除末尾的零
    :param value: 要移除零的数值
    :return: 移除零后的数值
    """
    return value.rstrip("0").rstrip(".")


def round_value(value: float, decimals: int, keep_trailing_zeros=False) -> str:
    """
    将数值四舍五入到指定位数的小数
    :param value: 要四舍五入的数值
    :param decimals: 四舍五入的小数位数
    :param keep_trailing_zeros: 是否保留末尾的零（"222.200" vs. "222.2"）
    :return: 四舍五入后的数值字符串
    """
    val = f"{value:.{decimals}f}"
    if not keep_trailing_zeros:
        val = strip_trailing_zeros(val)
    return val


def fmt_coin(value: float, coin: str, show_coin_name=True, keep_trailing_zeros=False) -> str:
    """
    格式化此币种的价格值
    :param value: 要打印的数值
    :param coin: 要打印价格/值的币种
    :param show_coin_name: 是否返回包含币种名称的字符串（格式："222.22 USDT" 或 "222.22"）
    :param keep_trailing_zeros: 是否保留末尾的零（"222.200" vs. "222.2"）
    :return: 格式化/四舍五入后的数值（带或不带币种名称）
    """
    val = round_value(value, decimals_per_coin(coin), keep_trailing_zeros)
    if show_coin_name:
        val = f"{val} {coin}"

    return val


def fmt_coin2(
    value: float, coin: str, decimals: int = 8, *, show_coin_name=True, keep_trailing_zeros=False
) -> str:
    """
    格式化此币种的价格值。应优先用于汇率格式化
    :param value: 要打印的数值
    :param coin: 要打印价格/值的币种
    :param decimals: 四舍五入的小数位数
    :param show_coin_name: 是否返回包含币种名称的字符串（格式："222.22 USDT" 或 "222.22"）
    :param keep_trailing_zeros: 是否保留末尾的零（"222.200" vs. "222.2"）
    :return: 格式化/四舍五入后的数值（带或不带币种名称）
    """
    val = round_value(value, decimals, keep_trailing_zeros)
    if show_coin_name:
        val = f"{val} {coin}"

    return val


def format_duration(td: timedelta) -> str:
    """
    将 timedelta 对象格式化为 "XXd HH:MM" 格式
    :param td: 要格式化的 timedelta 对象
    :return: 格式化后的时间字符串
    """
    d = td.days
    h, r = divmod(td.seconds, 3600)
    m, s = divmod(r, 60)
    return f"{d}d {h:02d}:{m:02d}"