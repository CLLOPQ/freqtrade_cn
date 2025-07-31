import pandas as pd
import pytest

from freqtrade.constants import DEFAULT_trades_columns
from freqtrade.data.converterter import populate_dataframe_with_trades
from freqtrade.data.converter.orderflow import (
    orderflow_added_columns,
    stacked_imbalance,
    timeframe_to_dateoffset,
    trades_to_volumeprofile_with_total_delta_bid_ask,
)
from freqtrade.data.converterter.trade_converter import trades_list_to_df
from freqtrade.data.dataprovidervider import DataProvider
from tests.strategy.strats.strategy_test_v3 import StrategyTestV3


bin_size_scale = 0.5


def read_csv(filename):
    converter_columns: list = ["side", "type"]
    return pd.read_csv(
        filename,
        skipinitialspace=True,
        index_col=0,
        parse_dates=True,
        date_format="ISO8601",
        converters={col: str.strip for col in converter_columns},
    )


@pytest.fixture
def populate_dataframe_with_trades_dataframe(testdatadir):
    return pd.read_feather(testdatadir / "orderflow/populate_dataframe_with_trades_df.feather")


@pytest.fixture
def populate_dataframe_with_trades_trades(testdatadir):
    return pd.read_feather(testdatadir / "orderflow/populate_dataframe_with_trades_trades.feather")


@pytest.fixture
def candles(testdatadir):
    # TODO: 这个测试夹具并非必需，可以移除
    return pd.read_json(testdatadir / "orderflow/candles.json").copy()


@pytest.fixture
def public_trades_list(testdatadir):
    return read_csv(testdatadir / "orderflow/public_trades_list.csv").copy()


@pytest.fixture
def public_trades_list_simple(testdatadir):
    return read_csv(testdatadir / "orderflow/public_trades_list_simple_example.csv").copy()


def test_public_trades_columns_before_change(
    populate_dataframe_with_trades_dataframe, populate_dataframe_with_trades_trades
):
    assert populate_dataframe_with_trades_dataframe.columns.tolist() == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert populate_dataframe_with_trades_trades.columns.tolist() == [
        "timestamp",
        "id",
        "type",
        "side",
        "price",
        "amount",
        "cost",
        "date",
    ]


def test_public_trades_mock_populate_dataframe_with_trades__check_orderflow(
    populate_dataframe_with_trades_dataframe, populate_dataframe_with_trades_trades
):
    """
    测试`populate_dataframe_with_trades`函数的订单流计算。

    本测试基于提供的配置和样本数据，检查生成的数据框和订单流的特定属性。
    """
    # 创建输入数据的副本，避免修改原始数据
    dataframe = populate_dataframe_with_trades_dataframe.copy()
    trades = populate_dataframe_with_trades_trades.copy()
    # 将'date'列转换为带毫秒的datetime格式
    dataframe["date"] = pd.to_datetime(dataframe["date"], unit="ms")
    # 选择最后几行并重置索引（可选，取决于使用场景）
    dataframe = dataframe.copy().tail().reset_index(drop=True)
    # 定义订单流计算的配置
    config = {
        "timeframe": "5m",
        "orderflow": {
            "cache_size": 1000,
            "max_candles": 1500,
            "scale": 0.005,
            "imbalance_volume": 0,
            "imbalance_ratio": 3,
            "stacked_imbalance_range": 3,
        },
    }
    # 应用函数以用订单流数据填充数据框
    df, _ = populate_dataframe_with_trades(None, config, dataframe, trades)
    # 从数据框的第一行提取结果
    results = df.iloc[0]
    t = results["trades"]
    of = results["orderflow"]

    # 断言结果的基本属性
    assert 0 != len(results)
    assert 151 == len(t)

    # --- 订单流分析 ---
    # 断言订单流数据点的数量
    assert 23 == len(of)  # 断言预期的数据点数量

    assert isinstance(of, dict)

    of_values = list(of.values())

    # 断言数据框开头的特定订单流值
    assert of_values[0] == {
        "bid": 0.0,
        "ask": 1.0,
        "delta": 4.999,
        "bid_amount": 0.0,
        "ask_amount": 4.999,
        "total_volume": 4.999,
        "total_trades": 1,
    }

    # 断言数据框末尾（不包括最后一行）的特定订单流值
    assert of_values[-1] == {
        "bid": 0.0,
        "ask": 1.0,
        "delta": 0.103,
        "bid_amount": 0.0,
        "ask_amount": 0.103,
        "total_volume": 0.103,
        "total_trades": 1,
    }

    # 从数据框的最后一行提取订单流
    of = df.iloc[-1]["orderflow"]

    # 断言最后一行中订单流数据点的数量
    assert 19 == len(of)  # 断言预期的数据点数量

    of_values1 = list(of.values())
    # 断言最后一行开头的特定订单流值
    assert of_values1[0] == {
        "bid": 1.0,
        "ask": 0.0,
        "delta": -12.536,
        "bid_amount": 12.536,
        "ask_amount": 0.0,
        "total_volume": 12.536,
        "total_trades": 1,
    }

    # 断言最后一行末尾的特定订单流值
    assert pytest.approx(of_values1[-1]) == {
        "bid": 4.0,
        "ask": 3.0,
        "delta": -40.948,
        "bid_amount": 59.182,
        "ask_amount": 18.23399,
        "total_volume": 77.416,
        "total_trades": 7,
    }

    # --- 德尔塔和其他结果 ---

    # 断言第一行的德尔塔值
    assert pytest.approx(results["delta"]) == -50.519
    # 断言第一行的最小和最大德尔塔值
    assert results["min_delta"] == -79.469
    assert results["max_delta"] == 17.298

    # 断言堆叠不平衡为NaN（本测试中不适用）
    assert results["stacked_imbalances_bid"] == []
    assert results["stacked_imbalances_ask"] == []

    # 对倒数第三行重复断言
    results = df.iloc[-2]
    assert pytest.approx(results["delta"]) == -20.862
    assert pytest.approx(results["min_delta"]) == -54.559999
    assert 82.842 == results["max_delta"]
    assert results["stacked_imbalances_bid"] == [234.97]
    assert results["stacked_imbalances_ask"] == [234.94]

    # 对最后一行重复断言
    results = df.iloc[-1]
    assert pytest.approx(results["delta"]) == -49.302
    assert results["min_delta"] == -70.222
    assert pytest.approx(results["max_delta"]) == 11.213
    assert results["stacked_imbalances_bid"] == []
    assert results["stacked_imbalances_ask"] == []


def test_public_trades_trades_mock_populate_dataframe_with_trades__check_trades(
    populate_dataframe_with_trades_dataframe, populate_dataframe_with_trades_trades
):
    """
    测试`populate_dataframe_with_trades`函数对交易的处理，
    确保交易数据正确集成到生成的数据框中。
    """

    # 创建输入数据的副本，避免修改原始数据
    dataframe = populate_dataframe_with_trades_dataframe.copy()
    trades = populate_dataframe_with_trades_trades.copy()

    # --- 数据准备 ---

    # 将'date'列转换为带毫秒的datetime格式
    dataframe["date"] = pd.to_datetime(dataframe["date"], unit="ms")

    # 选择数据框的最后一行
    dataframe = dataframe.tail().reset_index(drop=True)

    # 过滤交易，只保留发生在数据框第一个日期或之后的交易
    trades = trades.loc[trades.date >= dataframe.date[0]]
    trades.reset_index(inplace=True, drop=True)  # 重置索引以清晰显示

    # 断言第一个交易ID，确保过滤工作正常
    assert trades["id"][0] == "313881442"

    # --- 配置和函数调用 ---

    # 定义订单流计算的配置（用于上下文）
    config = {
        "timeframe": "5m",
        "orderflow": {
            "cache_size": 1000,
            "max_candles": 1500,
            "scale": 0.5,
            "imbalance_volume": 0,
            "imbalance_ratio": 3,
            "stacked_imbalance_range": 3,
        },
    }

    # 用交易和订单流数据填充数据框
    df, _ = populate_dataframe_with_trades(None, config, dataframe, trades)

    # --- 数据框和交易数据验证 ---

    row = df.iloc[0]  # 提取第一行进行断言

    # 断言数据框结构
    assert list(df.columns) == [
        # ...（预期列名列表）
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
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
    # 断言德尔塔、买入和卖出值
    assert pytest.approx(row["delta"]) == -50.519
    assert row["bid"] == 219.961
    assert row["ask"] == 169.442

    # 断言交易数量
    assert len(row["trades"]) == 151

    # 断言第一笔交易的特定细节
    t = row["trades"][0]
    assert list(t.keys()) == ["timestamp", "id", "type", "side", "price", "amount", "cost", "date"]
    assert trades["id"][0] == t["id"]
    assert int(trades["timestamp"][0]) == int(t["timestamp"])
    assert t["side"] == "sell"
    assert t["id"] == "313881442"
    assert t["price"] == 234.72


def test_public_trades_put_volume_profile_into_ohlcv_candles(public_trades_list_simple, candles):
    """
    测试成交量分布图数据与OHLCV蜡烛图的整合。

    本测试验证`trades_to_volumeprofile_with_total_delta_bid_ask`
    函数是否正确计算成交量分布，以及是否将成交量分布中的德尔塔值
    正确分配到`candles`数据框中相应的蜡烛上。
    """

    # 将交易列表转换为数据框
    trades_df = trades_list_to_df(public_trades_list_simple[default_trades_columns].values.tolist())

    # 生成具有指定 bin 大小的成交量分布图
    df = trades_to_volumeprofile_with_total_delta_bid_ask(trades_df, scale=bin_size_scale)

    # 断言第二根蜡烛的总买入/德尔塔响应中的德尔塔值
    assert 0.14 == df.values.tolist()[1][2]

    # 使用`.iat`访问器的替代断言（假设赋值逻辑正确）
    assert 0.14 == df["delta"].iat[1]


def test_public_trades_binned_big_sample_list(public_trades_list):
    """
    使用不同的 bin 大小测试`trades_to_volumeprofile_with_total_delta_bid_ask`函数，
    并验证生成的数据框的结构和值。
    """

    # 定义第一个测试的 bin 大小
    bin_size_scale = 0.05

    # 将交易列表转换为数据框
    trades = trades_list_to_df(public_trades_list[default_trades_columns].values.tolist())

    # 生成具有指定 bin 大小的成交量分布图
    df = trades_to_volumeprofile_with_total_delta_bid_ask(trades, scale=bin_size_scale)

    # 断言数据框具有预期的列
    assert df.columns.tolist() == [
        "bid",
        "ask",
        "delta",
        "bid_amount",
        "ask_amount",
        "total_volume",
        "total_trades",
    ]

    # 断言数据框中的行数（此 bin 大小预期为23行）
    assert len(df) == 23

    # 断言索引值按升序排列且间距正确
    assert all(df.index[i] < df.index[i + 1] for i in range(len(df) - 1))
    assert df.index[0] + bin_size_scale == df.index[1]
    assert (trades["price"].min() - bin_size_scale) < df.index[0] < trades["price"].max()
    assert (df.index[0] + bin_size_scale) >= df.index[1]
    assert (trades["price"].max() - bin_size_scale) < df.index[-1] < trades["price"].max()

    # 断言数据框第一行和最后一行的特定值
    assert 32 == df["bid"].iloc[0]  # 买入价格
    assert 197.512 == df["bid_amount"].iloc[0]  # 总买入量
    assert 88.98 == df["ask_amount"].iloc[0]  # 总卖出量
    assert 26 == df["ask"].iloc[0]  # 卖出价格
    # 德尔塔（买入量 - 卖出量）
    assert -108.532 == pytest.approx(df["delta"].iloc[0])

    assert 3 == df["bid"].iloc[-1]  # 买入价格
    assert 50.659 == df["bid_amount"].iloc[-1]  # 总买入量
    assert 108.21 == df["ask_amount"].iloc[-1]  # 总卖出量
    assert 44 == df["ask"].iloc[-1]  # 卖出价格
    assert 57.551 == df["delta"].iloc[-1]  # 德尔塔（买入量 - 卖出量）

    # 使用更大的 bin 大小重复该过程
    bin_size_scale = 1

    # 生成具有更大 bin 大小的成交量分布图
    df = trades_to_volumeprofile_with_total_delta_bid_ask(trades, scale=bin_size_scale)

    # 断言数据框中的行数（此 bin 大小预期为2行）
    assert len(df) == 2

    # 重复类似的索引排序和间距断言
    assert all(df.index[i] < df.index[i + 1] for i in range(len(df) - 1))
    assert (trades["price"].min() - bin_size_scale) < df.index[0] < trades["price"].max()
    assert (df.index[0] + bin_size_scale) >= df.index[1]
    assert (trades["price"].max() - bin_size_scale) < df.index[-1] < trades["price"].max()

    # 断言具有更大 bin 大小的数据框最后一行的值
    assert 1667.0 == df.index[-1]
    assert 710.98 == df["bid_amount"].iat[0]
    assert 111 == df["bid"].iat[0]
    assert 52.7199999 == pytest.approx(df["delta"].iat[0])  # 德尔塔


def test_public_trades_config_max_trades(
    default_conf, populate_dataframe_with_trades_dataframe, populate_dataframe_with_trades_trades
):
    dataframe = populate_dataframe_with_trades_dataframe.copy()
    trades = populate_dataframe_with_trades_trades.copy()
    default_conf["exchange"]["use_public_trades"] = True
    orderflow_config = {
        "timeframe": "5m",
        "orderflow": {
            "cache_size": 1000,
            "max_candles": 1,
            "scale": 0.005,
            "imbalance_volume": 0,
            "imbalance_ratio": 3,
            "stacked_imbalance_range": 3,
        },
    }

    df, _ = populate_dataframe_with_trades(None, default_conf | orderflow_config, dataframe, trades)
    assert df.delta.count() == 1


def test_public_trades_testdata_sanity(
    candles,
    public_trades_list,
    public_trades_list_simple,
    populate_dataframe_with_trades_dataframe,
    populate_dataframe_with_trades_trades,
):
    assert 10999 == len(candles)
    assert 1000 == len(public_trades_list)
    assert 999 == len(populate_dataframe_with_trades_dataframe)
    assert 293532 == len(populate_dataframe_with_trades_trades)

    assert 7 == len(public_trades_list_simple)
    assert (
        5
        == public_trades_list_simple.loc[
            public_trades_list_simple["side"].str.contains("sell"), "id"
        ].count()
    )
    assert (
        2
        == public_trades_list_simple.loc[
            public_trades_list_simple["side"].str.contains("buy"), "id"
        ].count()
    )

    assert public_trades_list.columns.tolist() == [
        "timestamp",
        "id",
        "type",
        "side",
        "price",
        "amount",
        "cost",
        "date",
    ]

    assert public_trades_list.columns.tolist() == [
        "timestamp",
        "id",
        "type",
        "side",
        "price",
        "amount",
        "cost",
        "date",
    ]
    assert public_trades_list_simple.columns.tolist() == [
        "timestamp",
        "id",
        "type",
        "side",
        "price",
        "amount",
        "cost",
        "date",
    ]
    assert populate_dataframe_with_trades_dataframe.columns.tolist() == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert populate_dataframe_with_trades_trades.columns.tolist() == [
        "timestamp",
        "id",
        "type",
        "side",
        "price",
        "amount",
        "cost",
        "date",
    ]


def test_analyze_with_orderflow(
    default_conf_usdt,
    mocker,
    populate_dataframe_with_trades_dataframe,
    populate_dataframe_with_trades_trades,
):
    ohlcv_history = populate_dataframe_with_trades_dataframe
    # 不使用订单流调用
    strategy = StrategyTestV3(config=default_conf_usdt)
    strategy.dp = DataProvider(default_conf_usdt, None, None)

    mocker.patch.object(strategy.dp, "trades", return_value=populate_dataframe_with_trades_trades)
    import freqtrade.data.converter.orderflow as orderflow_module

    spy = mocker.spy(orderflow_module, "trades_to_volumeprofile_with_total_delta_bid_ask")

    pair = "ETH/BTC"
    df = strategy.advise_indicators(ohlcv_history, {"pair:": pair})
    assert len(df) == len(ohlcv_history)
    assert "open" in df.columns
    assert spy.call_count == 0

    # 预期不会运行 - 不应添加订单流列
    for col in orderflow_added_columns:
        assert col not in df.columns, f"在df.columns中发现列 {col}"

    default_conf_usdt["exchange"]["use_public_trades"] = True
    default_conf_usdt["orderflow"] = {
        "cache_size": 5,
        "max_candles": 5,
        "scale": 0.005,
        "imbalance_volume": 0,
        "imbalance_ratio": 3,
        "stacked_imbalance_range": 3,
    }

    strategy.config = default_conf_usdt
    # 第一轮 - 构建缓存
    df1 = strategy.advise_indicators(ohlcv_history, {"pair": pair})
    assert len(df1) == len(ohlcv_history)
    assert "open" in df1.columns
    assert spy.call_count == 5

    for col in orderflow_added_columns:
        assert col in df1.columns, f"在df.columns中未发现列 {col}"

        if col not in ("stacked_imbalances_bid", "stacked_imbalances_ask"):
            assert df1[col].count() == 5, f"列 {col} 有 {df1[col].count()} 个非NaN值"

    assert len(strategy._cached_grouped_trades_per_pair[pair]) == 5

    lastval_trades = df1.at[len(df1) - 1, "trades"]
    assert isinstance(lastval_trades, list)
    assert len(lastval_trades) == 122

    lastval_of = df1.at[len(df1) - 1, "orderflow"]
    assert isinstance(lastval_of, dict)

    spy.reset_mock()
    # 确保缓存工作 - 再次调用相同逻辑
    df2 = strategy.advise_indicators(ohlcv_history, {"pair": pair})
    assert len(df2) == len(ohlcv_history)
    assert "open" in df2.columns
    assert spy.call_count == 0
    for col in orderflow_added_columns:
        assert col in df2.columns, f"第二轮：在df.columns中未发现列 {col}"

        if col not in ("stacked_imbalances_bid", "stacked_imbalances_ask"):
            assert df2[col].count() == 5, (
                f"第二轮：列 {col} 有 {df2[col].count()} 个非NaN值"
            )

    lastval_trade2 = df2.at[len(df2) - 1, "trades"]
    assert isinstance(lastval_trade2, list)
    assert len(lastval_trade2) == 122

    lastval_of2 = df2.at[len(df2) - 1, "orderflow"]
    assert isinstance(lastval_of2, dict)


def test_stacked_imbalances_multiple_prices():
    """测试当存在多个价格水平时，堆叠不平衡是否正确返回这些价格水平"""
    # 测试空结果
    df_no_stacks = pd.DataFrame(
        {
            "bid_imbalance": [False, False, True, False],
            "ask_imbalance": [False, True, False, False],
        },
        index=[234.95, 234.96, 234.97, 234.98],
    )
    no_stacks = stacked_imbalance(df_no_stacks, "bid", stacked_imbalance_range=2)
    assert no_stacks == []

    # 创建具有已知不平衡的样本数据框
    df = pd.DataFrame(
        {
            "bid_imbalance": [True, True, True, False, False, True, True, False, True],
            "ask_imbalance": [False, False, True, True, True, False, False, True, True],
        },
        index=[234.95, 234.96, 234.97, 234.98, 234.99, 235.00, 235.01, 235.02, 235.03],
    )
    # 测试买入不平衡（应按升序返回价格）
    bid_prices = stacked_imbalance(df, "bid", stacked_imbalance_range=2)
    assert bid_prices == [234.95, 234.96, 235.00]

    # 测试卖出不平衡（应按降序返回价格）
    ask_prices = stacked_imbalance(df, "ask", stacked_imbalance_range=2)
    assert ask_prices == [234.97, 234.98, 235.02]

    # 使用更高的stacked_imbalance_range测试
    bid_prices_higher = stacked_imbalance(df, "bid", stacked_imbalance_range=3)
    assert bid_prices_higher == [234.95]


def test_timeframe_to_dateoffset():
    assert timeframe_to_dateoffset("1s") == pd.DateOffset(seconds=1)
    assert timeframe_to_dateoffset("1m") == pd.DateOffset(minutes=1)
    assert timeframe_to_dateoffset("5m") == pd.DateOffset(minutes=5)
    assert timeframe_to_dateoffset("1h") == pd.DateOffset(hours=1)
    assert timeframe_to_dateoffset("1d") == pd.DateOffset(days=1)
    assert timeframe_to_dateoffset("1w") == pd.DateOffset(weeks=1)
    assert timeframe_to_dateoffset("1M") == pd.DateOffset(months=1)
    assert timeframe_to_dateoffset("1y") == pd.DateOffset(years=1)
