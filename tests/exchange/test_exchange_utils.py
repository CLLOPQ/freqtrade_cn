# pragma pylint: disable=missing-docstring, protected-access, invalid-name
from datetime import datetime, timedelta, timezone
from math import isnan, nan

import pytest
from ccxt import (
    DECIMAL_PLACES,  # 小数位数
    ROUND,  # 四舍五入
    ROUND_DOWN,  # 向下取整
    ROUND_UP,  # 向上取整
    SIGNIFICANT_DIGITS,  # 有效数字
    TICK_SIZE,  # 最小变动单位
    TRUNCATE,  # 截断
)

from freqtrade.enums import RunMode  # 运行模式
from freqtrade.exceptions import OperationalException  # 操作异常
from freqtrade.exchange import (
    amount_to_contract_precision,  # 合约数量精度转换
    amount_to_precision,  # 数量精度转换
    date_minus_candles,  # 日期减去指定数量的K线
    price_to_precision,  # 价格精度转换
    timeframe_to_minutes,  # 时间框架转换为分钟
    timeframe_to_msecs,  # 时间框架转换为毫秒
    timeframe_to_next_date,  # 时间框架转换为下一个日期
    timeframe_to_prev_date,  # 时间框架转换为上一个日期
    timeframe_to_resample_freq,  # 时间框架转换为重采样频率
    timeframe_to_seconds,  # 时间框架转换为秒
)
from freqtrade.exchange.check_exchange import check_exchange  # 检查交易所
from tests.conftest import log_has_re  # 日志检查工具


def test_check_exchange(default_conf, caplog) -> None:
    # 测试Freqtrade团队官方支持的交易所
    default_conf["runmode"] = RunMode.DRY_RUN  # 模拟交易模式
    default_conf.get("exchange").update({"name": "BINANCE"})
    assert check_exchange(default_conf)
    assert log_has_re(
        r"交易所 .* 是Freqtrade开发团队官方支持的\.", caplog
    )
    caplog.clear()  # 清空日志

    # 测试Freqtrade团队官方支持的交易所（小写名称）
    default_conf.get("exchange").update({"name": "binance"})
    assert check_exchange(default_conf)
    assert log_has_re(
        r"交易所 \"binance\" 是Freqtrade开发团队官方支持的\.", caplog
    )
    caplog.clear()

    # 测试Freqtrade团队官方支持的交易所（binanceus）
    default_conf.get("exchange").update({"name": "binanceus"})
    assert check_exchange(default_conf)
    assert log_has_re(
        r"交易所 \"binanceus\" 是Freqtrade开发团队官方支持的\.",
        caplog,
    )
    caplog.clear()

    # 测试Freqtrade团队官方支持的交易所（带映射的okx）
    default_conf.get("exchange").update({"name": "okx"})
    assert check_exchange(default_conf)
    assert log_has_re(
        r"交易所 \"okx\" 是Freqtrade开发团队官方支持的\.", caplog
    )
    caplog.clear()
    
    # 测试ccxt支持但非官方支持的交易所
    default_conf.get("exchange").update({"name": "bittrade"})
    assert check_exchange(default_conf)
    assert log_has_re(
        r"交易所 .* 为ccxt库所支持，可用于机器人，"
        r"但非Freqtrade开发团队官方支持。.*",
        caplog,
    )
    caplog.clear()

    # 测试已知有严重问题的"不良"交易所
    default_conf.get("exchange").update({"name": "bitmex"})
    with pytest.raises(OperationalException, match=r"交易所 .* 无法与Freqtrade一起使用。.*"):
        check_exchange(default_conf)
    caplog.clear()

    # 测试check_for_bad=False时的"不良"交易所
    default_conf.get("exchange").update({"name": "bitmex"})
    assert check_exchange(default_conf, False)
    assert log_has_re(
        r"交易所 .* 为ccxt库所支持，可用于机器人，"
        r"但非Freqtrade开发团队官方支持。.*",
        caplog,
    )
    caplog.clear()

    # 测试无效交易所
    default_conf.get("exchange").update({"name": "unknown_exchange"})
    with pytest.raises(
        OperationalException,
        match=r'交易所 "unknown_exchange" 不为ccxt库所识别，'
        r"因此无法用于机器人.*",
    ):
        check_exchange(default_conf)

    # 测试无交易所配置（绘图模式）
    default_conf.get("exchange").update({"name": ""})
    default_conf["runmode"] = RunMode.PLOT  # 绘图模式
    assert check_exchange(default_conf)

    # 测试无交易所配置（交易所工具模式）
    default_conf.get("exchange").update({"name": ""})
    default_conf["runmode"] = RunMode.UTIL_EXCHANGE  # 交易所工具模式
    with pytest.raises(
        OperationalException, match=r"此命令需要配置交易所.*"
    ):
        check_exchange(default_conf)


def test_date_minus_candles():
    date = datetime(2019, 8, 12, 13, 25, 0, tzinfo=timezone.utc)  # UTC时间

    # 测试日期减去指定数量的K线
    assert date_minus_candles("5m", 3, date) == date - timedelta(minutes=15)  # 5分钟*3=15分钟
    assert date_minus_candles("5m", 5, date) == date - timedelta(minutes=25)  # 5分钟*5=25分钟
    assert date_minus_candles("1m", 6, date) == date - timedelta(minutes=6)  # 1分钟*6=6分钟
    assert date_minus_candles("1h", 3, date) == date - timedelta(hours=3, minutes=25)  # 1小时*3=3小时
    assert date_minus_candles("1h", 3) == timeframe_to_prev_date("1h") - timedelta(hours=3)


def test_timeframe_to_minutes():
    # 测试时间框架转换为分钟
    assert timeframe_to_minutes("5m") == 5  # 5分钟
    assert timeframe_to_minutes("10m") == 10  # 10分钟
    assert timeframe_to_minutes("1h") == 60  # 1小时=60分钟
    assert timeframe_to_minutes("1d") == 1440  # 1天=1440分钟


def test_timeframe_to_seconds():
    # 测试时间框架转换为秒
    assert timeframe_to_seconds("5m") == 300  # 5分钟=300秒
    assert timeframe_to_seconds("10m") == 600  # 10分钟=600秒
    assert timeframe_to_seconds("1h") == 3600  # 1小时=3600秒
    assert timeframe_to_seconds("1d") == 86400  # 1天=86400秒


def test_timeframe_to_msecs():
    # 测试时间框架转换为毫秒
    assert timeframe_to_msecs("5m") == 300000  # 5分钟=300000毫秒
    assert timeframe_to_msecs("10m") == 600000  # 10分钟=600000毫秒
    assert timeframe_to_msecs("1h") == 3600000  # 1小时=3600000毫秒
    assert timeframe_to_msecs("1d") == 86400000  # 1天=86400000毫秒


@pytest.mark.parametrize(
    "timeframe,expected",
    [
        ("1s", "1s"),  # 1秒
        ("15s", "15s"),  # 15秒
        ("5m", "300s"),  # 5分钟=300秒
        ("10m", "600s"),  # 10分钟=600秒
        ("1h", "3600s"),  # 1小时=3600秒
        ("1d", "86400s"),  # 1天=86400秒
        ("1w", "1W-MON"),  # 1周（从周一算起）
        ("1M", "1MS"),  # 1月
        ("1y", "1YS"),  # 1年
    ],
)
def test_timeframe_to_resample_freq(timeframe, expected):
    # 测试时间框架转换为重采样频率
    assert timeframe_to_resample_freq(timeframe) == expected


def test_timeframe_to_prev_date():
    # 2019-08-12 13:22:08（UTC时间戳）
    date = datetime.fromtimestamp(1565616128, tz=timezone.utc)

    # 测试时间框架转换为上一个日期
    tf_list = [
        # 5分钟 -> 2019-08-12 13:20:00
        ("5m", datetime(2019, 8, 12, 13, 20, 0, tzinfo=timezone.utc)),
        # 10分钟 -> 2019-08-12 13:20:00
        ("10m", datetime(2019, 8, 12, 13, 20, 0, tzinfo=timezone.utc)),
        # 1小时 -> 2019-08-12 13:00:00
        ("1h", datetime(2019, 8, 12, 13, 00, 0, tzinfo=timezone.utc)),
        # 2小时 -> 2019-08-12 12:00:00
        ("2h", datetime(2019, 8, 12, 12, 00, 0, tzinfo=timezone.utc)),
        # 4小时 -> 2019-08-12 12:00:00
        ("4h", datetime(2019, 8, 12, 12, 00, 0, tzinfo=timezone.utc)),
        # 1天 -> 2019-08-12 00:00:00
        ("1d", datetime(2019, 8, 12, 00, 00, 0, tzinfo=timezone.utc)),
    ]
    for interval, result in tf_list:
        assert timeframe_to_prev_date(interval, date) == result

    date = datetime.now(tz=timezone.utc)  # 当前UTC时间
    assert timeframe_to_prev_date("5m") < date  # 上一个5分钟K线时间早于当前时间
    
    # 测试精确匹配的时间不进行四舍五入
    time = datetime(2019, 8, 12, 13, 20, 0, tzinfo=timezone.utc)
    assert timeframe_to_prev_date("5m", time) == time  # 正好是5分钟的整数倍
    time = datetime(2019, 8, 12, 13, 0, 0, tzinfo=timezone.utc)
    assert timeframe_to_prev_date("1h", time) == time  # 正好是1小时的整数倍


def test_timeframe_to_next_date():
    # 2019-08-12 13:22:08（UTC时间戳）
    date = datetime.fromtimestamp(1565616128, tz=timezone.utc)
    
    # 测试时间框架转换为下一个日期
    tf_list = [
        # 5分钟 -> 2019-08-12 13:25:00
        ("5m", datetime(2019, 8, 12, 13, 25, 0, tzinfo=timezone.utc)),
        # 10分钟 -> 2019-08-12 13:30:00
        ("10m", datetime(2019, 8, 12, 13, 30, 0, tzinfo=timezone.utc)),
        # 1小时 -> 2019-08-12 14:00:00
        ("1h", datetime(2019, 8, 12, 14, 00, 0, tzinfo=timezone.utc)),
        # 2小时 -> 2019-08-12 14:00:00
        ("2h", datetime(2019, 8, 12, 14, 00, 0, tzinfo=timezone.utc)),
        # 4小时 -> 2019-08-12 16:00:00
        ("4h", datetime(2019, 8, 12, 16, 00, 0, tzinfo=timezone.utc)),
        # 1天 -> 2019-08-13 00:00:00
        ("1d", datetime(2019, 8, 13, 0, 0, 0, tzinfo=timezone.utc)),
    ]

    for interval, result in tf_list:
        assert timeframe_to_next_date(interval, date) == result

    date = datetime.now(tz=timezone.utc)  # 当前UTC时间
    assert timeframe_to_next_date("5m") > date  # 下一个5分钟K线时间晚于当前时间

    # 测试精确匹配的时间加一个时间单位
    date = datetime(2019, 8, 12, 13, 30, 0, tzinfo=timezone.utc)
    assert timeframe_to_next_date("5m", date) == date + timedelta(minutes=5)  # 加5分钟


@pytest.mark.parametrize(
    "amount,precision_mode,precision,expected",
    [
        (2.34559, DECIMAL_PLACES, 4, 2.3455),  # 小数位数4位
        (2.34559, DECIMAL_PLACES, 5, 2.34559),  # 小数位数5位
        (2.34559, DECIMAL_PLACES, 3, 2.345),  # 小数位数3位
        (2.9999, DECIMAL_PLACES, 3, 2.999),  # 小数位数3位
        (2.9909, DECIMAL_PLACES, 3, 2.990),  # 小数位数3位
        (2.9909, DECIMAL_PLACES, 0, 2),  # 小数位数0位
        (29991.5555, DECIMAL_PLACES, 0, 29991),  # 小数位数0位
        (29991.5555, DECIMAL_PLACES, -1, 29990),  # 小数位数-1位（十位）
        (29991.5555, DECIMAL_PLACES, -2, 29900),  # 小数位数-2位（百位）
        # 有效数字测试
        (2.34559, SIGNIFICANT_DIGITS, 4, 2.345),  # 有效数字4位
        (2.34559, SIGNIFICANT_DIGITS, 5, 2.3455),  # 有效数字5位
        (2.34559, SIGNIFICANT_DIGITS, 3, 2.34),  # 有效数字3位
        (2.9999, SIGNIFICANT_DIGITS, 3, 2.99),  # 有效数字3位
        (2.9909, SIGNIFICANT_DIGITS, 3, 2.99),  # 有效数字3位
        (0.0000077723, SIGNIFICANT_DIGITS, 5, 0.0000077723),  # 有效数字5位
        (0.0000077723, SIGNIFICANT_DIGITS, 3, 0.00000777),  # 有效数字3位
        (0.0000077723, SIGNIFICANT_DIGITS, 1, 0.000007),  # 有效数字1位
        # 最小变动单位测试
        (2.34559, TICK_SIZE, 0.0001, 2.3455),  # 最小变动单位0.0001
        (2.34559, TICK_SIZE, 0.00001, 2.34559),  # 最小变动单位0.00001
        (2.34559, TICK_SIZE, 0.001, 2.345),  # 最小变动单位0.001
        (2.9999, TICK_SIZE, 0.001, 2.999),  # 最小变动单位0.001
        (2.9909, TICK_SIZE, 0.001, 2.990),  # 最小变动单位0.001
        (2.9909, TICK_SIZE, 0.005, 2.99),  # 最小变动单位0.005
        (2.9999, TICK_SIZE, 0.005, 2.995),  # 最小变动单位0.005
    ],
)
def test_amount_to_precision(
    amount,
    precision_mode,
    precision,
    expected,
):
    """
    测试数量向下取整
    """
    # 数字计数模式
    # DECIMAL_PLACES = 2（小数位数）
    # SIGNIFICANT_DIGITS = 3（有效数字）
    # TICK_SIZE = 4（最小变动单位）

    assert amount_to_precision(amount, precision, precision_mode) == expected


@pytest.mark.parametrize(
    "price,precision_mode,precision,expected,rounding_mode",
    [
        # 小数位数，向上取整测试
        (2.34559, DECIMAL_PLACES, 4, 2.3456, ROUND_UP),
        (2.34559, DECIMAL_PLACES, 5, 2.34559, ROUND_UP),
        (2.34559, DECIMAL_PLACES, 3, 2.346, ROUND_UP),
        (2.9999, DECIMAL_PLACES, 3, 3.000, ROUND_UP),
        (2.9909, DECIMAL_PLACES, 3, 2.991, ROUND_UP),
        (2.9901, DECIMAL_PLACES, 3, 2.991, ROUND_UP),
        # 小数位数，向下取整测试
        (2.34559, DECIMAL_PLACES, 5, 2.34559, ROUND_DOWN),
        (2.34559, DECIMAL_PLACES, 4, 2.3455, ROUND_DOWN),
        (2.9901, DECIMAL_PLACES, 3, 2.990, ROUND_DOWN),
        (0.00299, DECIMAL_PLACES, 3, 0.002, ROUND_DOWN),
        # 小数位数，四舍五入测试
        (2.345600000000001, DECIMAL_PLACES, 4, 2.3456, ROUND),
        (2.345551, DECIMAL_PLACES, 4, 2.3456, ROUND),
        (2.49, DECIMAL_PLACES, 0, 2.0, ROUND),
        (2.51, DECIMAL_PLACES, 0, 3.0, ROUND),
        (5.1, DECIMAL_PLACES, -1, 10.0, ROUND),
        (4.9, DECIMAL_PLACES, -1, 0.0, ROUND),
        # 有效数字，四舍五入测试
        (0.000007222, SIGNIFICANT_DIGITS, 1, 0.000007, ROUND),
        (0.000007222, SIGNIFICANT_DIGITS, 2, 0.0000072, ROUND),
        (0.000007777, SIGNIFICANT_DIGITS, 2, 0.0000078, ROUND),
        # 最小变动单位，向上取整测试
        (2.34559, TICK_SIZE, 0.0001, 2.3456, ROUND_UP),
        (2.34559, TICK_SIZE, 0.00001, 2.34559, ROUND_UP),
        (2.34559, TICK_SIZE, 0.001, 2.346, ROUND_UP),
        (2.9999, TICK_SIZE, 0.001, 3.000, ROUND_UP),
        (2.9909, TICK_SIZE, 0.001, 2.991, ROUND_UP),
        # 最小变动单位，向下取整测试
        (2.9909, TICK_SIZE, 0.001, 2.990, ROUND_DOWN),
        (2.9909, TICK_SIZE, 0.005, 2.995, ROUND_UP),
        (2.9973, TICK_SIZE, 0.005, 3.0, ROUND_UP),
        (2.9977, TICK_SIZE, 0.005, 3.0, ROUND_UP),
        (234.43, TICK_SIZE, 0.5, 234.5, ROUND_UP),
        (234.43, TICK_SIZE, 0.5, 234.0, ROUND_DOWN),
        (234.53, TICK_SIZE, 0.5, 235.0, ROUND_UP),
        (234.53, TICK_SIZE, 0.5, 234.5, ROUND_DOWN),
        (0.891534, TICK_SIZE, 0.0001, 0.8916, ROUND_UP),
        (64968.89, TICK_SIZE, 0.01, 64968.89, ROUND_UP),
        (0.000000003483, TICK_SIZE, 1e-12, 0.000000003483, ROUND_UP),
        # 最小变动单位，四舍五入测试
        (2.49, TICK_SIZE, 1.0, 2.0, ROUND),
        (2.51, TICK_SIZE, 1.0, 3.0, ROUND),
        (2.000000051, TICK_SIZE, 0.0000001, 2.0000001, ROUND),
        (2.000000049, TICK_SIZE, 0.0000001, 2.0, ROUND),
        (2.9909, TICK_SIZE, 0.005, 2.990, ROUND),
        (2.9973, TICK_SIZE, 0.005, 2.995, ROUND),
        (2.9977, TICK_SIZE, 0.005, 3.0, ROUND),
        (234.24, TICK_SIZE, 0.5, 234.0, ROUND),
        (234.26, TICK_SIZE, 0.5, 234.5, ROUND),
        (nan, TICK_SIZE, 3, nan, ROUND),  # NaN值测试
        # 截断测试
        (2.34559, DECIMAL_PLACES, 4, 2.3455, TRUNCATE),
        (2.34559, DECIMAL_PLACES, 5, 2.34559, TRUNCATE),
        (2.34559, DECIMAL_PLACES, 3, 2.345, TRUNCATE),
        (2.9999, DECIMAL_PLACES, 3, 2.999, TRUNCATE),
        (2.9909, DECIMAL_PLACES, 3, 2.990, TRUNCATE),
        (2.9909, TICK_SIZE, 0.001, 2.990, TRUNCATE),
        (2.9909, TICK_SIZE, 0.01, 2.99, TRUNCATE),
        (2.9909, TICK_SIZE, 0.1, 2.9, TRUNCATE),
        # 有效数字截断测试
        (2.34559, SIGNIFICANT_DIGITS, 4, 2.345, TRUNCATE),
        (2.34559, SIGNIFICANT_DIGITS, 5, 2.3455, TRUNCATE),
        (2.34559, SIGNIFICANT_DIGITS, 3, 2.34, TRUNCATE),
        (2.9999, SIGNIFICANT_DIGITS, 3, 2.99, TRUNCATE),
        (2.9909, SIGNIFICANT_DIGITS, 2, 2.9, TRUNCATE),
        (0.00000777, SIGNIFICANT_DIGITS, 2, 0.0000077, TRUNCATE),
        (0.00000729, SIGNIFICANT_DIGITS, 2, 0.0000072, TRUNCATE),
        # 有效数字四舍五入测试
        (722.2, SIGNIFICANT_DIGITS, 1, 700.0, ROUND),
        (790.2, SIGNIFICANT_DIGITS, 1, 800.0, ROUND),
        (722.2, SIGNIFICANT_DIGITS, 2, 720.0, ROUND),
        # 有效数字向上取整测试
        (722.2, SIGNIFICANT_DIGITS, 1, 800.0, ROUND_UP),
        (722.2, SIGNIFICANT_DIGITS, 2, 730.0, ROUND_UP),
        (777.7, SIGNIFICANT_DIGITS, 2, 780.0, ROUND_UP),
        (777.7, SIGNIFICANT_DIGITS, 3, 778.0, ROUND_UP),
        # 有效数字向下取整测试
        (722.2, SIGNIFICANT_DIGITS, 1, 700.0, ROUND_DOWN),
        (722.2, SIGNIFICANT_DIGITS, 2, 720.0, ROUND_DOWN),
        (777.7, SIGNIFICANT_DIGITS, 2, 770.0, ROUND_DOWN),
        (777.7, SIGNIFICANT_DIGITS, 3, 777.0, ROUND_DOWN),
        # 小数值有效数字向上取整测试
        (0.000007222, SIGNIFICANT_DIGITS, 1, 0.000008, ROUND_UP),
        (0.000007222, SIGNIFICANT_DIGITS, 2, 0.0000073, ROUND_UP),
        (0.000007777, SIGNIFICANT_DIGITS, 2, 0.0000078, ROUND_UP),
        # 小数值有效数字向下取整测试
        (0.000007222, SIGNIFICANT_DIGITS, 1, 0.000007, ROUND_DOWN),
        (0.000007222, SIGNIFICANT_DIGITS, 2, 0.0000072, ROUND_DOWN),
        (0.000007777, SIGNIFICANT_DIGITS, 2, 0.0000077, ROUND_DOWN),
    ],
)
def test_price_to_precision(price, precision_mode, precision, expected, rounding_mode):
    result = price_to_precision(price, precision, precision_mode, rounding_mode=rounding_mode)
    if not isnan(expected):  # 非NaN值比较
        assert result == expected
    else:  # NaN值比较
        assert isnan(result)


@pytest.mark.parametrize(
    "amount,precision,precision_mode,contract_size,expected",
    [
        (1.17, 1.0, 4, 0.01, 1.17),  # 最小变动单位
        (1.17, 1.0, 2, 0.01, 1.17),
        (1.16, 1.0, 4, 0.01, 1.16),
        (1.16, 1.0, 2, 0.01, 1.16),
        (1.13, 1.0, 2, 0.01, 1.13),
        (10.988, 1.0, 2, 10, 10),
        (10.988, 1.0, 4, 10, 10),
    ],
)
def test_amount_to_contract_precision_standalone(
    amount, precision, precision_mode, contract_size, expected
):
    res = amount_to_contract_precision(amount, precision, precision_mode, contract_size)
    assert pytest.approx(res) == expected  # 近似比较浮点数