import numpy as np
import pandas as pd
import pytest

from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import CandleType
from freqtrade.resolvers.strategy_resolver import StrategyResolver
from freqtrade.strategy import merge_informative_pair, stoploss_from_absolute, stoploss_from_open
from tests.conftest import generate_test_data, get_patched_exchange


def test_merge_informative_pair():
    # 生成15分钟和1小时的测试数据
    data = generate_test_data("15m", 40)
    informative = generate_test_data("1h", 40)
    cols_inf = list(informative.columns)

    # 合并数据并测试结果
    result = merge_informative_pair(data, informative, "15m", "1h", ffill=True)
    assert isinstance(result, pd.DataFrame)
    assert list(informative.columns) == cols_inf  # 确保原始数据列未改变
    assert len(result) == len(data)  # 确保行数与原始数据一致
    assert "date" in result.columns
    assert result["date"].equals(data["date"])  # 确保原始日期列保持不变
    assert "date_1h" in result.columns  # 确保添加了1小时时间框架的日期列

    # 验证价格和交易量列
    assert "open" in result.columns
    assert "open_1h" in result.columns
    assert result["open"].equals(data["open"])

    assert "close" in result.columns
    assert "close_1h" in result.columns
    assert result["close"].equals(data["close"])

    assert "volume" in result.columns
    assert "volume_1h" in result.columns
    assert result["volume"].equals(data["volume"])

    # 验证前3行是NaN（没有对应的1小时数据）
    assert result.iloc[0]["date_1h"] is pd.NaT
    assert result.iloc[1]["date_1h"] is pd.NaT
    assert result.iloc[2]["date_1h"] is pd.NaT
    # 接下来4行应该包含起始日期（0:00）
    assert result.iloc[3]["date_1h"] == result.iloc[0]["date"]
    assert result.iloc[4]["date_1h"] == result.iloc[0]["date"]
    assert result.iloc[5]["date_1h"] == result.iloc[0]["date"]
    assert result.iloc[6]["date_1h"] == result.iloc[0]["date"]
    # 再接下来4行应该包含下一个小时的日期（原始数据第4行）
    assert result.iloc[7]["date_1h"] == result.iloc[4]["date"]
    assert result.iloc[8]["date_1h"] == result.iloc[4]["date"]

    # 测试不使用前向填充的情况
    informative = generate_test_data("1h", 40)
    result = merge_informative_pair(data, informative, "15m", "1h", ffill=False)
    # 前3行是NaN
    assert result.iloc[0]["date_1h"] is pd.NaT
    assert result.iloc[1]["date_1h"] is pd.NaT
    assert result.iloc[2]["date_1h"] is pd.NaT
    # 只有第一个15分钟K线包含对应的1小时数据
    assert result.iloc[3]["date_1h"] == result.iloc[0]["date"]
    assert result.iloc[4]["date_1h"] is pd.NaT
    assert result.iloc[5]["date_1h"] is pd.NaT
    assert result.iloc[6]["date_1h"] is pd.NaT
    # 下一个小时的第一个15分钟K线包含数据
    assert result.iloc[7]["date_1h"] == result.iloc[4]["date"]
    assert result.iloc[8]["date_1h"] is pd.NaT


def test_merge_informative_pair_weekly():
    # 生成覆盖大约2个月的数据 - 直到2023-01-10
    data = generate_test_data("1h", 1040, "2022-11-28")
    informative = generate_test_data("1w", 40, "2022-11-01")
    informative["day"] = informative["date"].dt.day_name()  # 添加星期几列

    # 合并数据
    result = merge_informative_pair(data, informative, "1h", "1w", ffill=True)
    assert isinstance(result, pd.DataFrame)
    
    # 2022-12-24是星期六
    candle1 = result.loc[(result["date"] == "2022-12-24T22:00:00.000Z")]
    assert candle1.iloc[0]["date"] == pd.Timestamp("2022-12-24T22:00:00.000Z")
    assert candle1.iloc[0]["date_1w"] == pd.Timestamp("2022-12-12T00:00:00.000Z")

    candle2 = result.loc[(result["date"] == "2022-12-24T23:00:00.000Z")]
    assert candle2.iloc[0]["date"] == pd.Timestamp("2022-12-24T23:00:00.000Z")
    assert candle2.iloc[0]["date_1w"] == pd.Timestamp("2022-12-12T00:00:00.000Z")

    # 2022-12-25是星期日
    candle3 = result.loc[(result["date"] == "2022-12-25T22:00:00.000Z")]
    assert candle3.iloc[0]["date"] == pd.Timestamp("2022-12-25T22:00:00.000Z")
    # 仍然使用旧的周K线
    assert candle3.iloc[0]["date_1w"] == pd.Timestamp("2022-12-12T00:00:00.000Z")

    candle4 = result.loc[(result["date"] == "2022-12-25T23:00:00.000Z")]
    assert candle4.iloc[0]["date"] == pd.Timestamp("2022-12-25T23:00:00.000Z")
    assert candle4.iloc[0]["date_1w"] == pd.Timestamp("2022-12-19T00:00:00.000Z")


def test_merge_informative_pair_monthly():
    # 生成覆盖大约2个月的数据 - 直到2023-01-10
    data = generate_test_data("1h", 1040, "2022-11-28")
    informative = generate_test_data("1M", 40, "2022-01-01")

    result = merge_informative_pair(data, informative, "1h", "1M", ffill=True)
    assert isinstance(result, pd.DataFrame)
    
    # 测试12月底的K线
    candle1 = result.loc[(result["date"] == "2022-12-31T22:00:00.000Z")]
    assert candle1.iloc[0]["date"] == pd.Timestamp("2022-12-31T22:00:00.000Z")
    assert candle1.iloc[0]["date_1M"] == pd.Timestamp("2022-11-01T00:00:00.000Z")

    # 跨月K线（12月31日23:00）
    candle2 = result.loc[(result["date"] == "2022-12-31T23:00:00.000Z")]
    assert candle2.iloc[0]["date"] == pd.Timestamp("2022-12-31T23:00:00.000Z")
    assert candle2.iloc[0]["date_1M"] == pd.Timestamp("2022-12-01T00:00:00.000Z")

    # 11月底的K线（无对应的月度数据）
    candle3 = result.loc[(result["date"] == "2022-11-30T22:00:00.000Z")]
    assert candle3.iloc[0]["date"] == pd.Timestamp("2022-11-30T22:00:00.000Z")
    assert candle3.iloc[0]["date_1M"] is pd.NaT

    # 11月底最后一小时的K线（有对应的月度数据）
    candle4 = result.loc[(result["date"] == "2022-11-30T23:00:00.000Z")]
    assert candle4.iloc[0]["date"] == pd.Timestamp("2022-11-30T23:00:00.000Z")
    assert candle4.iloc[0]["date_1M"] == pd.Timestamp("2022-11-01T00:00:00.000Z")


def test_merge_informative_pair_same():
    # 测试相同时间框架的合并
    data = generate_test_data("15m", 40)
    informative = generate_test_data("15m", 40)

    result = merge_informative_pair(data, informative, "15m", "15m", ffill=True)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(data)
    assert "date" in result.columns
    assert result["date"].equals(data["date"])
    assert "date_15m" in result.columns

    # 验证价格和交易量列
    assert "open" in result.columns
    assert "open_15m" in result.columns
    assert result["open"].equals(data["open"])

    assert "close" in result.columns
    assert "close_15m" in result.columns
    assert result["close"].equals(data["close"])

    assert "volume" in result.columns
    assert "volume_15m" in result.columns
    assert result["volume"].equals(data["volume"])

    # 日期应该一一对应
    assert result["date_15m"].equals(result["date"])


def test_merge_informative_pair_lower():
    # 测试尝试合并更快时间框架的情况（应该抛出错误）
    data = generate_test_data("1h", 40)
    informative = generate_test_data("15m", 40)

    with pytest.raises(ValueError, match=r"尝试合并更快的时间框架 .*"):
        merge_informative_pair(data, informative, "1h", "15m", ffill=True)


def test_merge_informative_pair_empty():
    # 测试合并空数据框的情况
    data = generate_test_data("1h", 40)
    informative = pd.DataFrame(columns=data.columns)

    result = merge_informative_pair(data, informative, "1h", "2h", ffill=True)
    assert result["date"].equals(data["date"])  # 确保日期列保持不变

    # 验证所有列都存在
    assert list(result.columns) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "date_2h",
        "open_2h",
        "high_2h",
        "low_2h",
        "close_2h",
        "volume_2h",
    ]
    # 合并空数据框，所有合并的列都应该是NaN
    for col in ["date_2h", "open_2h", "high_2h", "low_2h", "close_2h", "volume_2h"]:
        assert result[col].isnull().all()


def test_merge_informative_pair_suffix():
    # 测试使用自定义后缀的情况
    data = generate_test_data("15m", 20)
    informative = generate_test_data("1h", 20)

    result = merge_informative_pair(
        data, informative, "15m", "1h", append_timeframe=False, suffix="suf"
    )

    assert "date" in result.columns
    assert result["date"].equals(data["date"])
    assert "date_suf" in result.columns  # 应该使用自定义后缀

    assert "open_suf" in result.columns
    assert "open_1h" not in result.columns  # 不应该有时间框架后缀

    # 验证所有列名正确
    assert list(result.columns) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "date_suf",
        "open_suf",
        "high_suf",
        "low_suf",
        "close_suf",
        "volume_suf",
    ]


def test_merge_informative_pair_suffix_append_timeframe():
    # 测试同时指定后缀和append_timeframe（应该抛出错误）
    data = generate_test_data("15m", 20)
    informative = generate_test_data("1h", 20)

    with pytest.raises(ValueError, match=r"不能同时指定 `append_timeframe` .*"):
        merge_informative_pair(data, informative, "15m", "1h", suffix="suf")


@pytest.mark.parametrize(
    "side,profitrange",
    [
        # 多头的利润范围是[-1, inf]，空头的是[-inf, 1]
        ("long", [-0.99, 2, 30]),
        ("short", [-2.0, 0.99, 30]),
    ],
)
def test_stoploss_from_open(side, profitrange):
    # 测试不同开盘价范围
    open_price_ranges = [
        [0.01, 1.00, 30],
        [1, 100, 30],
        [100, 10000, 30],
    ]

    for open_range in open_price_ranges:
        for open_price in np.linspace(*open_range):
            for desired_stop in np.linspace(-0.50, 0.50, 30):
                if side == "long":
                    # -1不是有效的当前利润，应该返回1
                    assert stoploss_from_open(desired_stop, -1) == 1
                else:
                    # 1不是空头的有效当前利润，应该返回1
                    assert stoploss_from_open(desired_stop, 1, True) == 1

                for current_profit in np.linspace(*profitrange):
                    if side == "long":
                        current_price = open_price * (1 + current_profit)
                        expected_stop_price = open_price * (1 + desired_stop)
                        stoploss = stoploss_from_open(desired_stop, current_profit)
                        stop_price = current_price * (1 - stoploss)
                    else:
                        current_price = open_price * (1 - current_profit)
                        expected_stop_price = open_price * (1 - desired_stop)
                        stoploss = stoploss_from_open(desired_stop, current_profit, True)
                        stop_price = current_price * (1 + stoploss)

                    assert stoploss >= 0
                    # 理论上公式可能为空头产生大于1的止损值
                    # 尽管这在实际中没有意义，因为仓位会被清算
                    if side == "long":
                        assert stoploss <= 1

                    # 如果预期止损价高于当前价，则没有正确答案
                    if (side == "long" and expected_stop_price > current_price) or (
                        side == "short" and expected_stop_price < current_price
                    ):
                        assert stoploss == 0
                    else:
                        assert pytest.approx(stop_price) == expected_stop_price


@pytest.mark.parametrize(
    "side,rel_stop,curr_profit,leverage,expected",
    [
        # 多头利润范围是[-1, inf]，空头是[-inf, 1]
        ("long", 0, -1, 1, 1),
        ("long", 0, 0.1, 1, 0.09090909),
        ("long", -0.1, 0.1, 1, 0.18181818),
        ("long", 0.1, 0.2, 1, 0.08333333),
        ("long", 0.1, 0.5, 1, 0.266666666),
        ("long", 0.1, 5, 1, 0.816666666),  # 500%利润，将止损设为开盘价上方10%
        ("long", 0, 5, 10, 3.3333333),  # 500%利润，将止损设为盈亏平衡点
        ("long", 0.1, 5, 10, 3.26666666),  # 500%利润，将止损设为开盘价上方10%
        ("long", -0.1, 5, 10, 3.3999999),  # 500%利润，将止损设为开盘价下方10%
        ("short", 0, 0.1, 1, 0.1111111),
        ("short", -0.1, 0.1, 1, 0.2222222),
        ("short", 0.1, 0.2, 1, 0.125),
        ("short", 0.1, 1, 1, 1),
        ("short", -0.01, 5, 10, 10.01999999),  # 500%利润，10倍杠杆
    ],
)
def test_stoploss_from_open_leverage(side, rel_stop, curr_profit, leverage, expected):
    # 测试带杠杆的止损计算
    stoploss = stoploss_from_open(rel_stop, curr_profit, side == "short", leverage)
    assert pytest.approx(stoploss) == expected
    open_rate = 100
    if stoploss != 1:
        if side == "long":
            current_rate = open_rate * (1 + curr_profit / leverage)
            stop = current_rate * (1 - stoploss / leverage)
            assert pytest.approx(stop) == open_rate * (1 + rel_stop / leverage)
        else:
            current_rate = open_rate * (1 - curr_profit / leverage)
            stop = current_rate * (1 + stoploss / leverage)
            assert pytest.approx(stop) == open_rate * (1 - rel_stop / leverage)


def test_stoploss_from_absolute():
    # 测试从绝对价格计算止损
    assert pytest.approx(stoploss_from_absolute(90, 100)) == 1 - (90 / 100)
    assert pytest.approx(stoploss_from_absolute(90, 100)) == 0.1
    assert pytest.approx(stoploss_from_absolute(95, 100)) == 0.05
    assert pytest.approx(stoploss_from_absolute(100, 100)) == 0
    assert pytest.approx(stoploss_from_absolute(110, 100)) == 0
    assert pytest.approx(stoploss_from_absolute(100, 0)) == 1
    assert pytest.approx(stoploss_from_absolute(0, 100)) == 1
    assert pytest.approx(stoploss_from_absolute(0, 100, False, leverage=5)) == 5

    # 空头情况
    assert pytest.approx(stoploss_from_absolute(90, 100, True)) == 0
    assert pytest.approx(stoploss_from_absolute(100, 100, True)) == 0
    assert pytest.approx(stoploss_from_absolute(110, 100, True)) == -(1 - (110 / 100))
    assert pytest.approx(stoploss_from_absolute(110, 100, True)) == 0.1
    assert pytest.approx(stoploss_from_absolute(105, 100, True)) == 0.05
    assert pytest.approx(stoploss_from_absolute(105, 100, True, 5)) == 0.05 * 5
    assert pytest.approx(stoploss_from_absolute(100, 0, True)) == 1
    assert pytest.approx(stoploss_from_absolute(0, 100, True)) == 0
    assert pytest.approx(stoploss_from_absolute(100, 1, is_short=True)) == 1
    assert pytest.approx(stoploss_from_absolute(100, 1, is_short=True, leverage=5)) == 5


@pytest.mark.parametrize("trading_mode", ["futures", "spot"])
def test_informative_decorator(mocker, default_conf_usdt, trading_mode):
    # 获取交易模式对应的默认K线类型
    candle_def = CandleType.get_default(trading_mode)
    default_conf_usdt["candle_type_def"] = candle_def
    
    # 生成各种时间框架的测试数据
    test_data_5m = generate_test_data("5m", 40)
    test_data_30m = generate_test_data("30m", 40)
    test_data_1h = generate_test_data("1h", 40)
    
    # 组织测试数据
    data = {
        ("XRP/USDT", "5m", candle_def): test_data_5m,
        ("XRP/USDT", "30m", candle_def): test_data_30m,
        ("XRP/USDT", "1h", candle_def): test_data_1h,
        ("XRP/BTC", "1h", candle_def): test_data_1h,  # 来自{base}/BTC
        ("LTC/USDT", "5m", candle_def): test_data_5m,
        ("LTC/USDT", "30m", candle_def): test_data_30m,
        ("LTC/USDT", "1h", candle_def): test_data_1h,
        ("LTC/BTC", "1h", candle_def): test_data_1h,  # 来自{base}/BTC
        ("NEO/USDT", "30m", candle_def): test_data_30m,
        ("NEO/USDT", "5m", CandleType.SPOT): test_data_5m,  # 显式指定K线类型为现货
        ("NEO/USDT", "15m", candle_def): test_data_5m,  # 显式指定K线类型
        ("NEO/USDT", "1h", candle_def): test_data_1h,
        ("ETH/USDT", "1h", candle_def): test_data_1h,
        ("ETH/USDT", "30m", candle_def): test_data_30m,
        ("ETH/BTC", "1h", CandleType.SPOT): test_data_1h,  # 显式指定为现货
    }
    
    # 加载策略
    default_conf_usdt["strategy"] = "InformativeDecoratorTest"
    strategy = StrategyResolver.load_strategy(default_conf_usdt)
    exchange = get_patched_exchange(mocker, default_conf_usdt)
    strategy.dp = DataProvider({}, exchange, None)
    mocker.patch.object(
        strategy.dp, "current_whitelist", return_value=["XRP/USDT", "LTC/USDT", "NEO/USDT"]
    )

    # 验证信息对数量（与使用的装饰器数量相同）
    assert len(strategy._ft_informative) == 7
    
    # 预期的信息对列表
    informative_pairs = [
        ("XRP/USDT", "1h", candle_def),
        ("XRP/BTC", "1h", candle_def),
        ("LTC/USDT", "1h", candle_def),
        ("LTC/BTC", "1h", candle_def),
        ("XRP/USDT", "30m", candle_def),
        ("LTC/USDT", "30m", candle_def),
        ("NEO/USDT", "1h", candle_def),
        ("NEO/USDT", "30m", candle_def),
        ("NEO/USDT", "5m", candle_def),
        ("NEO/USDT", "15m", candle_def),
        ("NEO/USDT", "2h", CandleType.FUTURES),
        ("ETH/BTC", "1h", CandleType.SPOT),  # 一个K线保持为现货
        ("ETH/USDT", "30m", candle_def),
    ]
    for inf_pair in informative_pairs:
        assert inf_pair in strategy.gather_informative_pairs()

    # 模拟历史K线数据获取
    def test_historic_ohlcv(pair, timeframe, candle_type):
        return data[
            (pair, timeframe or strategy.timeframe, CandleType.from_string(candle_type))
        ].copy()

    mocker.patch(
        "freqtrade.data.dataprovider.DataProvider.historic_ohlcv", side_effect=test_historic_ohlcv
    )

    # 分析所有指标
    analyzed = strategy.advise_all_indicators(
        {p: data[(p, strategy.timeframe, candle_def)] for p in ("XRP/USDT", "LTC/USDT")}
    )
    
    # 验证所有预期列都存在
    expected_columns = [
        "rsi_1h",
        "rsi_30m",  # 堆叠的信息装饰器
        "neo_usdt_rsi_1h",  # NEO 1小时信息指标
        "rsi_NEO_USDT_neo_usdt_NEO/USDT_30m",  # 列格式化
        "rsi_from_callable",  # 自定义列格式化器
        "eth_btc_rsi_1h",  # 报价货币与基础货币不匹配
        "rsi",
        "rsi_less",  # 非信息列
        "rsi_5m",  # 手动信息数据框
    ]
    for _, dataframe in analyzed.items():
        for col in expected_columns:
            assert col in dataframe.columns