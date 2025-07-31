import pytest

from freqtrade.exceptions import OperationalException
from freqtrade.leverage import interest
from freqtrade.util import FtPrecise


十分钟 = FtPrecise(1 / 6)  # 10分钟 = 1/6小时
五小时 = FtPrecise(5.0)
二十五小时 = FtPrecise(25.0)


@pytest.mark.parametrize(
    "exchange,interest_rate,hours,expected",
    [
        ("binance", 0.0005, 十分钟, 0.00125),
        ("binance", 0.00025, 十分钟, 0.000625),
        ("binance", 0.00025, 五小时, 0.003125),
        ("binance", 0.00025, 二十五小时, 0.015625),
        #  Kraken交易所
        ("kraken", 0.0005, 十分钟, 0.06),
        ("kraken", 0.00025, 十分钟, 0.03),
        ("kraken", 0.00025, 五小时, 0.045),
        ("kraken", 0.00025, 二十五小时, 0.12),
    ],
)
def test_interest(exchange, interest_rate, hours, expected):
    """测试不同交易所的杠杆利息计算"""
    借款金额 = FtPrecise(60.0)

    assert (
        pytest.approx(
            float(
                interest(
                    exchange_name=exchange,
                    borrowed=借款金额,
                    rate=FtPrecise(interest_rate),
                    hours=hours,
                )
            )
        )
        == expected
    )


def test_interest_exception():
    """测试不支持杠杆的交易所会抛出异常"""
    with pytest.raises(OperationalException, match=r"Freqtrade在.*上不支持杠杆交易"):
        interest(
            exchange_name="bitmex", 
            borrowed=FtPrecise(60.0), 
            rate=FtPrecise(0.0005), 
            hours=十分钟
        )