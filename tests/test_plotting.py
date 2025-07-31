from copy import deepcopy
from unittest.mock import MagicMock

import pandas as pd
import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots

from freqtrade.commandss import start_plot_dataframe, start_plot_profit
from freqtrade.configuration import TimeRange
from freqtrade.data import history
from freqtrade.data.btanalysis import load_backtest_data
from freqtrade.data.metrics import create_cum_profit
from freqtrade.exceptions import OperationalException
from freqtrade.plot.plotting import (
    add_areas,
    add_indicators,
    add_profit,
    create_plotconfig,
    generate_candlestick_graph,
    generate_plot_filename,
    generate_profit_graph,
    init_plotscript,
    load_and_plot_trades,
    plot_profit,
    plot_trades,
    store_plot_file,
)
from freqtrade.resolvers import StrategyResolver
from tests.conftest import get_args, log_has, log_has_re, patch_exchange


def fig_generating_mock(fig, *args, **kwargs):
    """返回图表 - 用于模拟add_indicators和plot_trades"""
    return fig


def find_trace_in_fig_data(data, search_string: str):
    matches = (d for d in data if d.name == search_string)
    return next(matches)


def generate_empty_figure():
    return make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_width=[1, 1, 4],
        vertical_spacing=0.0001,
    )


def test_init_plotscript(default_conf, mocker, testdatadir):
    """测试初始化绘图脚本功能"""
    default_conf["timerange"] = "20180110-20180112"
    default_conf["trade_source"] = "file"
    default_conf["timeframe"] = "5m"
    default_conf["exportfilename"] = testdatadir / "backtest-result.json"
    supported_markets = ["TRX/BTC", "ADA/BTC"]
    ret = init_plotscript(default_conf, supported_markets)
    assert "ohlcv" in ret
    assert "trades" in ret
    assert "pairs" in ret
    assert "timerange" in ret

    default_conf["pairs"] = ["TRX/BTC", "ADA/BTC"]
    ret = init_plotscript(default_conf, supported_markets, 20)
    assert "ohlcv" in ret
    assert "TRX/BTC" in ret["ohlcv"]
    assert "ADA/BTC" in ret["ohlcv"]


def test_add_indicators(default_conf, testdatadir, caplog):
    """测试添加指标到图表的功能"""
    pair = "UNITTEST/BTC"
    timerange = TimeRange()

    data = history.load_pair_history(
        pair=pair, timeframe="1m", datadir=testdatadir, timerange=timerange
    )
    indicators1 = {"ema10": {}}
    indicators2 = {"macd": {"color": "red"}}

    strategy = StrategyResolver.load_strategy(default_conf)

    # 生成入场/出场信号和指标
    data = strategy.analyze_ticker(data, {"pair": pair})
    fig = generate_empty_figure()

    # 第1行
    fig1 = add_indicators(fig=deepcopy(fig), row=1, indicators=indicators1, data=data)
    figure = fig1.layout.figure
    ema10 = find_trace_in_fig_data(figure.data, "ema10")
    assert isinstance(ema10, go.Scatter)
    assert ema10.yaxis == "y"

    # 第3行
    fig2 = add_indicators(fig=deepcopy(fig), row=3, indicators=indicators2, data=data)
    figure = fig2.layout.figure
    macd = find_trace_in_fig_data(figure.data, "macd")
    assert isinstance(macd, go.Scatter)
    assert macd.yaxis == "y3"
    assert macd.line.color == "red"

    # 测试不存在的指标
    fig3 = add_indicators(fig=deepcopy(fig), row=3, indicators={"no_indicator": {}}, data=data)
    assert fig == fig3
    assert log_has_re(r'指标 "no_indicator" 被忽略\..*', caplog)


def test_add_areas(default_conf, testdatadir, caplog):
    """测试在图表中添加填充区域的功能"""
    pair = "UNITTEST/BTC"
    timerange = TimeRange(None, "line", 0, -1000)

    data = history.load_pair_history(
        pair=pair, timeframe="1m", datadir=testdatadir, timerange=timerange
    )
    indicators = {
        "macd": {
            "color": "red",
            "fill_color": "black",
            "fill_to": "macdhist",
            "fill_label": "MACD 填充",
        }
    }

    ind_no_label = {"macd": {"fill_color": "red", "fill_to": "macdhist"}}

    ind_plain = {"macd": {"fill_to": "macdhist"}}
    strategy = StrategyResolver.load_strategy(default_conf)

    # 生成指标
    data = strategy.analyze_ticker(data, {"pair": pair})
    fig = generate_empty_figure()

    # 测试fill_to引用不存在的指标
    fig1 = add_areas(fig, 1, data, {"ema10": {"fill_to": "no_fill_indicator"}})
    assert fig == fig1
    assert log_has_re(r'fill_to: "no_fill_indicator" 被忽略\..*', caplog)

    # 测试不存在的指标
    fig2 = add_areas(fig, 1, data, {"no_indicator": {"fill_to": "ema10"}})
    assert fig == fig2
    assert log_has_re(r'指标 "no_indicator" 被忽略\..*', caplog)

    # 测试完整配置，第3行
    fig3 = add_areas(fig, 3, data, indicators)
    figure = fig3.layout.figure
    fill_macd = find_trace_in_fig_data(figure.data, "MACD 填充")
    assert isinstance(fill_macd, go.Scatter)
    assert fill_macd.yaxis == "y3"
    assert fill_macd.fillcolor == "black"

    # 测试缺少标签，第1行
    fig4 = add_areas(fig, 1, data, ind_no_label)
    figure = fig4.layout.figure
    fill_macd = find_trace_in_fig_data(figure.data, "macd<>macdhist")
    assert isinstance(fill_macd, go.Scatter)
    assert fill_macd.yaxis == "y"
    assert fill_macd.fillcolor == "red"

    # 测试仅指定fill_to
    fig5 = add_areas(fig, 1, data, ind_plain)
    figure = fig5.layout.figure
    fill_macd = find_trace_in_fig_data(figure.data, "macd<>macdhist")
    assert isinstance(fill_macd, go.Scatter)
    assert fill_macd.yaxis == "y"


def test_plot_trades(testdatadir, caplog):
    """测试在图表上绘制交易的功能"""
    fig1 = generate_empty_figure()
    # 没有交易数据的情况
    fig = plot_trades(fig1, None)
    assert fig == fig1
    assert log_has("未找到交易。", caplog)
    
    pair = "ADA/BTC"
    filename = testdatadir / "backtest_results/backtest-result.json"
    trades = load_backtest_data(filename)
    trades = trades.loc[trades["pair"] == pair]

    fig = plot_trades(fig, trades)
    figure = fig1.layout.figure

    # 检查入场点 - 颜色、位置等
    trade_entries = find_trace_in_fig_data(figure.data, "交易入场")
    assert isinstance(trade_entries, go.Scatter)
    assert trade_entries.yaxis == "y"
    assert len(trades) == len(trade_entries.x)
    assert trade_entries.marker.color == "cyan"
    assert trade_entries.marker.symbol == "circle-open"
    assert trade_entries.text[0] == "3.99%, buy_tag, roi, 15 min"

    # 检查盈利出场点
    trade_exit = find_trace_in_fig_data(figure.data, "出场 - 盈利")
    assert isinstance(trade_exit, go.Scatter)
    assert trade_exit.yaxis == "y"
    assert len(trades.loc[trades["profit_ratio"] > 0]) == len(trade_exit.x)
    assert trade_exit.marker.color == "green"
    assert trade_exit.marker.symbol == "square-open"
    assert trade_exit.text[0] == "3.99%, buy_tag, roi, 15 min"

    # 检查亏损出场点
    trade_sell_loss = find_trace_in_fig_data(figure.data, "出场 - 亏损")
    assert isinstance(trade_sell_loss, go.Scatter)
    assert trade_sell_loss.yaxis == "y"
    assert len(trades.loc[trades["profit_ratio"] <= 0]) == len(trade_sell_loss.x)
    assert trade_sell_loss.marker.color == "red"
    assert trade_sell_loss.marker.symbol == "square-open"
    assert trade_sell_loss.text[5] == "-10.45%, stop_loss, 720 min"


def test_generate_candlestick_graph_no_signals_no_trades(default_conf, mocker, testdatadir, caplog):
    """测试在没有信号和交易的情况下生成K线图"""
    row_mock = mocker.patch(
        "freqtrade.plot.plotting.add_indicators", MagicMock(side_effect=fig_generating_mock)
    )
    trades_mock = mocker.patch(
        "freqtrade.plot.plotting.plot_trades", MagicMock(side_effect=fig_generating_mock)
    )

    pair = "UNITTEST/BTC"
    timerange = TimeRange(None, "line", 0, -1000)
    data = history.load_pair_history(
        pair=pair, timeframe="1m", datadir=testdatadir, timerange=timerange
    )
    data["enter_long"] = 0
    data["exit_long"] = 0
    data["enter_short"] = 0
    data["exit_short"] = 0

    indicators1 = []
    indicators2 = []
    fig = generate_candlestick_graph(
        pair=pair, data=data, trades=None, indicators1=indicators1, indicators2=indicators2
    )
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == pair
    figure = fig.layout.figure

    assert len(figure.data) == 2
    # K线图应该是第一个绘制的
    candles = find_trace_in_fig_data(figure.data, "价格")
    assert isinstance(candles, go.Candlestick)

    volume = find_trace_in_fig_data(figure.data, "成交量")
    assert isinstance(volume, go.Bar)

    assert row_mock.call_count == 2
    assert trades_mock.call_count == 1

    assert log_has("未找到enter_long信号。", caplog)
    assert log_has("未找到exit_long信号。", caplog)
    assert log_has("未找到enter_short信号。", caplog)
    assert log_has("未找到exit_short信号。", caplog)


def test_generate_candlestick_graph_no_trades(default_conf, mocker, testdatadir):
    """测试在没有交易但有信号的情况下生成K线图"""
    row_mock = mocker.patch(
        "freqtrade.plot.plotting.add_indicators", MagicMock(side_effect=fig_generating_mock)
    )
    trades_mock = mocker.patch(
        "freqtrade.plot.plotting.plot_trades", MagicMock(side_effect=fig_generating_mock)
    )
    pair = "UNITTEST/BTC"
    timerange = TimeRange(None, "line", 0, -1000)
    data = history.load_pair_history(
        pair=pair, timeframe="1m", datadir=testdatadir, timerange=timerange
    )

    strategy = StrategyResolver.load_strategy(default_conf)

    # 生成买入/卖出信号和指标
    data = strategy.analyze_ticker(data, {"pair": pair})

    indicators1 = []
    indicators2 = []
    fig = generate_candlestick_graph(
        pair=pair, data=data, trades=None, indicators1=indicators1, indicators2=indicators2
    )
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == pair
    figure = fig.layout.figure

    assert len(figure.data) == 8
    # K线图应该是第一个绘制的
    candles = find_trace_in_fig_data(figure.data, "价格")
    assert isinstance(candles, go.Candlestick)

    volume = find_trace_in_fig_data(figure.data, "成交量")
    assert isinstance(volume, go.Bar)

    enter_long = find_trace_in_fig_data(figure.data, "enter_long")
    assert isinstance(enter_long, go.Scatter)
    # 所有买入信号都应该被绘制
    assert int(data["enter_long"].sum()) == len(enter_long.x)

    exit_long = find_trace_in_fig_data(figure.data, "exit_long")
    assert isinstance(exit_long, go.Scatter)
    # 所有卖出信号都应该被绘制
    assert int(data["exit_long"].sum()) == len(exit_long.x)

    assert find_trace_in_fig_data(figure.data, "布林带")

    assert row_mock.call_count == 2
    assert trades_mock.call_count == 1


def test_generate_Plot_filename():
    """测试生成图表文件名的功能"""
    fn = generate_plot_filename("UNITTEST/BTC", "5m")
    assert fn == "freqtrade-plot-UNITTEST_BTC-5m.html"


def test_generate_plot_file(mocker, caplog, user_dir):
    """测试保存图表文件的功能"""
    fig = generate_empty_figure()
    plot_mock = mocker.patch("freqtrade.plot.plotting.plot", MagicMock())
    store_plot_file(
        fig, filename="freqtrade-plot-UNITTEST_BTC-5m.html", directory=user_dir / "plot"
    )

    expected_fn = str(user_dir / "plot/freqtrade-plot-UNITTEST_BTC-5m.html")
    assert plot_mock.call_count == 1
    assert plot_mock.call_args[0][0] == fig
    assert plot_mock.call_args_list[0][1]["filename"] == expected_fn
    assert log_has(f"图表已保存至 {expected_fn}", caplog)


def test_add_profit(testdatadir):
    """测试在图表上添加利润曲线的功能"""
    filename = testdatadir / "backtest_results/backtest-result.json"
    bt_data = load_backtest_data(filename)
    timerange = TimeRange.parse_timerange("20180110-20180112")

    df = history.load_pair_history(
        pair="TRX/BTC", timeframe="5m", datadir=testdatadir, timerange=timerange
    )
    fig = generate_empty_figure()

    cum_profits = create_cum_profit(
        df.set_index("date"), bt_data[bt_data["pair"] == "TRX/BTC"], "cum_profits", timeframe="5m"
    )

    fig1 = add_profit(fig, row=2, data=cum_profits, column="cum_profits", name="利润")
    figure = fig1.layout.figure
    profits = find_trace_in_fig_data(figure.data, "利润")
    assert isinstance(profits, go.Scatter)
    assert profits.yaxis == "y2"


def test_generate_profit_graph(testdatadir):
    """测试生成利润图表的功能"""
    filename = testdatadir / "backtest_results/backtest-result.json"
    trades = load_backtest_data(filename)
    timerange = TimeRange.parse_timerange("20180110-20180112")
    pairs = ["TRX/BTC", "XLM/BTC"]
    trades = trades[trades["close_date"] < pd.Timestamp("2018-01-12", tz="UTC")]

    data = history.load_data(datadir=testdatadir, pairs=pairs, timeframe="5m", timerange=timerange)

    trades = trades[trades["pair"].isin(pairs)]

    fig = generate_profit_graph(
        pairs, data, trades, timeframe="5m", stake_currency="BTC", starting_balance=0
    )
    assert isinstance(fig, go.Figure)

    assert fig.layout.title.text == "Freqtrade 利润图表"
    assert fig.layout.yaxis.title.text == "价格"
    assert fig.layout.yaxis2.title.text == "利润 BTC"
    assert fig.layout.yaxis3.title.text == "利润 BTC"

    figure = fig.layout.figure
    assert len(figure.data) == 8

    avgclose = find_trace_in_fig_data(figure.data, "平均收盘价")
    assert isinstance(avgclose, go.Scatter)

    profit = find_trace_in_fig_data(figure.data, "利润")
    assert isinstance(profit, go.Scatter)
    drawdown = find_trace_in_fig_data(figure.data, "最大回撤 73.89%")
    assert isinstance(drawdown, go.Scatter)
    parallel = find_trace_in_fig_data(figure.data, "并行交易")
    assert isinstance(parallel, go.Scatter)

    underwater = find_trace_in_fig_data(figure.data, "水下图表")
    assert isinstance(underwater, go.Scatter)

    underwater_relative = find_trace_in_fig_data(figure.data, "水下图表 (%)")
    assert isinstance(underwater_relative, go.Scatter)

    for pair in pairs:
        profit_pair = find_trace_in_fig_data(figure.data, f"{pair} 利润")
        assert isinstance(profit_pair, go.Scatter)

    # 测试没有交易的情况
    with pytest.raises(OperationalException, match=r"未找到交易.*"):
        generate_profit_graph(
            pairs,
            data,
            trades.loc[trades["pair"].isnull()],
            timeframe="5m",
            stake_currency="BTC",
            starting_balance=0,
        )


def test_start_plot_dataframe(mocker):
    """测试启动数据框绘图命令"""
    aup = mocker.patch("freqtrade.plot.plotting.load_and_plot_trades", MagicMock())
    args = [
        "plot-dataframe",
        "--config",
        "tests/testdata/testconfigs/main_test_config.json",
        "--pairs",
        "ETH/BTC",
    ]
    start_plot_dataframe(get_args(args))

    assert aup.call_count == 1
    called_config = aup.call_args_list[0][0][0]
    assert "pairs" in called_config
    assert called_config["pairs"] == ["ETH/BTC"]


def test_load_and_plot_trades(default_conf, mocker, caplog, testdatadir):
    """测试加载并绘制交易数据的功能"""
    patch_exchange(mocker)

    default_conf["trade_source"] = "file"
    default_conf["exportfilename"] = testdatadir / "backtest-result.json"
    default_conf["indicators1"] = ["sma5", "ema10"]
    default_conf["indicators2"] = ["macd"]
    default_conf["pairs"] = ["ETH/BTC", "LTC/BTC"]

    candle_mock = MagicMock()
    store_mock = MagicMock()
    mocker.patch.multiple(
        "freqtrade.plot.plotting",
        generate_candlestick_graph=candle_mock,
        store_plot_file=store_mock,
    )
    load_and_plot_trades(default_conf)

    # 每个交易对都应该调用一次这两个模拟函数
    assert candle_mock.call_count == 2
    assert store_mock.call_count == 2

    assert candle_mock.call_args_list[0][1]["indicators1"] == ["sma5", "ema10"]
    assert candle_mock.call_args_list[0][1]["indicators2"] == ["macd"]

    assert log_has("绘图过程结束。生成了 2 个图表", caplog)


def test_start_plot_profit(mocker):
    """测试启动利润绘图命令"""
    aup = mocker.patch("freqtrade.plot.plotting.plot_profit", MagicMock())
    args = [
        "plot-profit",
        "--config",
        "tests/testdata/testconfigs/main_test_config.json",
        "--pairs",
        "ETH/BTC",
    ]
    start_plot_profit(get_args(args))

    assert aup.call_count == 1
    called_config = aup.call_args_list[0][0][0]
    assert "pairs" in called_config
    assert called_config["pairs"] == ["ETH/BTC"]


def test_start_plot_profit_error(mocker):
    """测试利润绘图命令的错误处理"""
    args = ["plot-profit", "--pairs", "ETH/BTC"]
    argsp = get_args(args)
    # 确保不使用配置文件，避免因存在config.json而导致随机失败
    argsp["config"] = []
    with pytest.raises(OperationalException):
        start_plot_profit(argsp)


def test_plot_profit(default_conf, mocker, testdatadir):
    """测试绘制利润图表的功能"""
    patch_exchange(mocker)
    default_conf["trade_source"] = "file"
    default_conf["exportfilename"] = testdatadir / "backtest-result_test_nofile.json"
    default_conf["pairs"] = ["ETH/BTC", "LTC/BTC"]

    profit_mock = MagicMock()
    store_mock = MagicMock()
    mocker.patch.multiple(
        "freqtrade.plot.plotting", generate_profit_graph=profit_mock, store_plot_file=store_mock
    )
    # 测试没有找到交易文件的情况
    with pytest.raises(
        OperationalException, match=r"未找到交易，无法生成利润图表.*"
    ):
        plot_profit(default_conf)

    # 测试正常情况
    default_conf["exportfilename"] = testdatadir / "backtest_results/backtest-result.json"
    plot_profit(default_conf)

    # 利润图表生成一个综合图表
    assert profit_mock.call_count == 1
    assert store_mock.call_count == 1

    assert profit_mock.call_args_list[0][0][0] == default_conf["pairs"]
    assert store_mock.call_args_list[0][1]["auto_open"] is False

    # 测试缺少时间框架的情况
    del default_conf["timeframe"]
    with pytest.raises(OperationalException, match=r"必须设置时间框架.*--timeframe.*"):
        plot_profit(default_conf)


@pytest.mark.parametrize(
    "ind1,ind2,plot_conf,exp",
    [
        # 没有指标，使用plot_conf
        (
            [],
            [],
            {},
            {
                "main_plot": {"sma": {}, "ema3": {}, "ema5": {}},
                "subplots": {"Other": {"macd": {}, "macdsignal": {}}},
            },
        ),
        # 使用指标参数
        (
            ["sma", "ema3"],
            ["macd"],
            {},
            {"main_plot": {"sma": {}, "ema3": {}}, "subplots": {"Other": {"macd": {}}}},
        ),
        # 只有main_plot - 添加空的subplots
        ([], [], {"main_plot": {"sma": {}}}, {"main_plot": {"sma": {}}, "subplots": {}}),
        # 同时有main和subplots
        (
            [],
            [],
            {"main_plot": {"sma": {}}, "subplots": {"RSI": {"rsi": {"color": "red"}}}},
            {"main_plot": {"sma": {}}, "subplots": {"RSI": {"rsi": {"color": "red"}}}},
        ),
        # 没有main_plot，添加空的main_plot
        (
            [],
            [],
            {"subplots": {"RSI": {"rsi": {"color": "red"}}}},
            {"main_plot": {}, "subplots": {"RSI": {"rsi": {"color": "red"}}}},
        ),
        # 指标参数覆盖plot_config
        (
            ["sma", "ema3"],
            ["macd"],
            {"main_plot": {"sma": {}}, "subplots": {"RSI": {"rsi": {"color": "red"}}}},
            {"main_plot": {"sma": {}, "ema3": {}}, "subplots": {"Other": {"macd": {}}}},
        ),
        # 指标1覆盖plot_config的main_plot
        (
            ["sma", "ema3"],
            [],
            {"main_plot": {"sma": {}}, "subplots": {"RSI": {"rsi": {"color": "red"}}}},
            {"main_plot": {"sma": {}, "ema3": {}}, "subplots": {"RSI": {"rsi": {"color": "red"}}}},
        ),
        # 指标2覆盖plot_config的subplots
        (
            [],
            ["macd", "macd_signal"],
            {"main_plot": {"sma": {}}, "subplots": {"RSI": {"rsi": {"color": "red"}}}},
            {"main_plot": {"sma": {}}, "subplots": {"Other": {"macd": {}, "macd_signal": {}}}},
        ),
    ],
)
def test_create_plotconfig(ind1, ind2, plot_conf, exp):
    """测试创建绘图配置的功能"""
    res = create_plotconfig(ind1, ind2, plot_conf)
    assert "main_plot" in res
    assert "subplots" in res
    assert isinstance(res["main_plot"], dict)
    assert isinstance(res["subplots"], dict)

    assert res == exp