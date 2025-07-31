import pytest

from freqtrade.enums import CandleType


@pytest.mark.parametrize(
    "candle_type,expected",
    [
        ("", CandleType.SPOT),
        ("spot", CandleType.SPOT),
        (CandleType.SPOT, CandleType.SPOT),
        (CandleType.FUTURES, CandleType.FUTURES),
        (CandleType.INDEX, CandleType.INDEX),
        (CandleType.MARK, CandleType.MARK),
        ("futures", CandleType.FUTURES),
        ("mark", CandleType.MARK),
        ("premiumIndex", CandleType.PREMIUMINDEX),
    ],
)
def test_CandleType_from_string(candle_type, expected):
    """测试从字符串转换为CandleType枚举的功能"""
    assert CandleType.from_string(candle_type) == expected


@pytest.mark.parametrize(
    "candle_type,expected",
    [
        ("futures", CandleType.FUTURES),
        ("spot", CandleType.SPOT),
        ("margin", CandleType.SPOT),
    ],
)
def test_CandleType_get_default(candle_type, expected):
    """测试根据交易类型获取默认CandleType的功能"""
    assert CandleType.get_default(candle_type) == expected