from math import ceil

from freqtrade.exceptions import OperationalException
from freqtrade.util import FtPrecise


one = FtPrecise(1.0)
four = FtPrecise(4.0)
twenty_four = FtPrecise(24.0)


def interest(
    exchange_name: str, borrowed: FtPrecise, rate: FtPrecise, hours: FtPrecise
) -> FtPrecise:
    """
    计算保证金交易利息的公式

    :param exchange_name: 进行交易的交易所名称
    :param borrowed: 借入的货币金额
    :param rate: 利率（即日利率）
    :param hours: 货币的借入时间（小时）

    异常:
        OperationalException: 如果 freqtrade 不支持该交易所的保证金交易则抛出

    返回: 应付利息金额（货币类型与借入的一致）
    """
    exchange_name = exchange_name.lower()
    if exchange_name == "binance":
        return borrowed * rate * FtPrecise(ceil(hours)) / twenty_four
    elif exchange_name == "kraken":
        # 根据 https://kraken-fees-calculator.github.io/ 进行四舍五入
        return borrowed * rate * (one + FtPrecise(ceil(hours / four)))
    else:
        raise OperationalException(f"Freqtrade 在 {exchange_name} 上不支持杠杆交易")