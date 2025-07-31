import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from freqtrade.configuration import TimeRange
from freqtrade.constants import Config
from freqtrade.data.btanalysis import (
    analyze_trade_parallelism,
    extract_trades_of_period,
    load_trades,
)
from freqtrade.data.converter import trim_dataframe
from freqtrade.data.dataprovider import DataProvider
from freqtrade.data.history import get_timerange, load_data
from freqtrade.data.metrics import (
    calculate_max_drawdown,
    calculate_underwater,
    combine_dataframes_with_mean,
    create_cum_profit,
)
from freqtrade.enums import CandleType
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_prev_date, timeframe_to_seconds
from freqtrade.misc import pair_to_filename
from freqtrade.plugins.pairlist.pairlist_helpers import expand_pairlist
from freqtrade.resolvers import ExchangeResolver, StrategyResolver
from freqtrade.strategy import IStrategy
from freqtrade.strategy.strategy_wrapper import strategy_safe_wrapper
from freqtrade.util import get_dry_run_wallet


logger = logging.getLogger(__name__)


try:
    import plotly.graph_objects as go
    from plotly.offline import plot
    from plotly.subplots import make_subplots
except ImportError:
    logger.exception("未找到plotly模块 \n 请使用 `pip3 install plotly` 安装")
    exit(1)


def init_plotscript(config, markets: list, startup_candles: int = 0):
    """
    初始化绘图所需的对象
    :return: 包含蜡烛图(OHLCV)数据、交易和交易对的字典
    """

    if "pairs" in config:
        pairs = expand_pairlist(config["pairs"], markets)
    else:
        pairs = expand_pairlist(config["exchange"]["pair_whitelist"], markets)

    # 设置要使用的时间范围
    timerange = TimeRange.parse_timerange(config.get("timerange"))

    data = load_data(
        datadir=config.get("datadir"),
        pairs=pairs,
        timeframe=config["timeframe"],
        timerange=timerange,
        startup_candles=startup_candles,
        data_format=config["dataformat_ohlcv"],
        candle_type=config.get("candle_type_def", CandleType.SPOT),
    )

    if startup_candles and data:
        min_date, max_date = get_timerange(data)
        logger.info(f"加载数据从 {min_date} 到 {max_date}")
        timerange.adjust_start_if_necessary(
            timeframe_to_seconds(config["timeframe"]), startup_candles, min_date
        )

    no_trades = False
    filename = config.get("exportfilename")
    if config.get("no_trades", False):
        no_trades = True
    elif config["trade_source"] == "file":
        if not filename.is_dir() and not filename.is_file():
            logger.warning("回测文件缺失，跳过交易数据。")
            no_trades = True
    try:
        trades = load_trades(
            config["trade_source"],
            db_url=config.get("db_url"),
            exportfilename=filename,
            no_trades=no_trades,
            strategy=config.get("strategy"),
        )
    except ValueError as e:
        raise OperationalException(e) from e
    if not trades.empty:
        trades = trim_dataframe(trades, timerange, df_date_col="open_date")

    return {
        "ohlcv": data,
        "trades": trades,
        "pairs": pairs,
        "timerange": timerange,
    }


def add_indicators(fig, row, indicators: dict[str, dict], data: pd.DataFrame) -> make_subplots:
    """
    根据配置为特定行生成用户选择的所有指标
    :param fig: 要附加的绘图对象
    :param row: 此绘图的行号
    :param indicators: 带配置选项的指标字典。字典键必须对应数据框列。
    :param data: 蜡烛图数据框
    """
    plot_kinds = {
        "scatter": go.Scatter,
        "bar": go.Bar,
    }
    for indicator, conf in indicators.items():
        logger.debug(f"指标 {indicator} 配置 {conf}")
        if indicator in data:
            kwargs = {"x": data["date"], "y": data[indicator].values, "name": indicator}

            plot_type = conf.get("type", "scatter")
            color = conf.get("color")
            if plot_type == "bar":
                kwargs.update(
                    {
                        "marker_color": color or "DarkSlateGrey",
                        "marker_line_color": color or "DarkSlateGrey",
                    }
                )
            else:
                if color:
                    kwargs.update({"line": {"color": color}})
                kwargs["mode"] = "lines"
                if plot_type != "scatter":
                    logger.warning(
                        f"指标 {indicator} 有未知的绘图类型 {plot_type}"
                        f'，假设为 "scatter"。'
                    )

            kwargs.update(conf.get("plotly", {}))
            trace = plot_kinds[plot_type](**kwargs)
            fig.add_trace(trace, row, 1)
        else:
            logger.info(
                '指标 "%s" 被忽略。原因：在您的策略中未找到此指标。',
                indicator,
            )

    return fig


def add_profit(fig, row, data: pd.DataFrame, column: str, name: str) -> make_subplots:
    """
    添加利润图表
    :param fig: 要附加的绘图对象
    :param row: 此绘图的行号
    :param data: 蜡烛图数据框
    :param column: 用于绘图的列
    :param name: 使用的名称
    :return: 添加了利润图表的fig
    """
    profit = go.Scatter(
        x=data.index,
        y=data[column],
        name=name,
    )
    fig.add_trace(profit, row, 1)

    return fig


def add_max_drawdown(
    fig, row, trades: pd.DataFrame, df_comb: pd.DataFrame, timeframe: str, starting_balance: float
) -> make_subplots:
    """
    添加表示最大回撤的散点
    """
    try:
        drawdown = calculate_max_drawdown(trades, starting_balance=starting_balance)

        drawdown = go.Scatter(
            x=[drawdown.high_date, drawdown.low_date],
            y=[
                df_comb.loc[timeframe_to_prev_date(timeframe, drawdown.high_date), "cum_profit"],
                df_comb.loc[timeframe_to_prev_date(timeframe, drawdown.low_date), "cum_profit"],
            ],
            mode="markers",
            name=f"最大回撤 {drawdown.relative_account_drawdown:.2%}",
            text=f"最大回撤 {drawdown.relative_account_drawdown:.2%}",
            marker=dict(symbol="square-open", size=9, line=dict(width=2), color="green"),
        )
        fig.add_trace(drawdown, row, 1)
    except ValueError:
        logger.warning("未找到交易 - 不绘制最大回撤。")
    return fig


def add_underwater(fig, row, trades: pd.DataFrame, starting_balance: float) -> make_subplots:
    """
    添加水下图表
    """
    try:
        underwater = calculate_underwater(
            trades, value_col="profit_abs", starting_balance=starting_balance
        )

        underwater_plot = go.Scatter(
            x=underwater["date"],
            y=underwater["drawdown"],
            name="水下图表",
            fill="tozeroy",
            fillcolor="#cc362b",
            line={"color": "#cc362b"},
        )

        underwater_plot_relative = go.Scatter(
            x=underwater["date"],
            y=(-underwater["drawdown_relative"]),
            name="水下图表 (%)",
            fill="tozeroy",
            fillcolor="green",
            line={"color": "green"},
        )

        fig.add_trace(underwater_plot, row, 1)
        fig.add_trace(underwater_plot_relative, row + 1, 1)
    except ValueError:
        logger.warning("未找到交易 - 不绘制水下图表")
    return fig


def add_parallelism(fig, row, trades: pd.DataFrame, timeframe: str) -> make_subplots:
    """
    添加显示交易并行度的图表
    """
    try:
        result = analyze_trade_parallelism(trades, timeframe)

        drawdown = go.Scatter(
            x=result.index,
            y=result["open_trades"],
            name="并行交易",
            fill="tozeroy",
            fillcolor="#242222",
            line={"color": "#242222"},
        )
        fig.add_trace(drawdown, row, 1)
    except ValueError:
        logger.warning("未找到交易 - 不绘制并行度。")
    return fig


def plot_trades(fig, trades: pd.DataFrame) -> make_subplots:
    """
    向"fig"添加交易
    """
    # 交易可能为空
    if trades is not None and len(trades) > 0:
        # 创建包含交易摘要的出场描述
        trades["desc"] = trades.apply(
            lambda row: f"{row['profit_ratio']:.2%}, "
            + (f"{row['enter_tag']}, " if row["enter_tag"] is not None else "")
            + f"{row['exit_reason']}, "
            + f"{row['trade_duration']} 分钟",
            axis=1,
        )
        入场点 = go.Scatter(
            x=trades["open_date"],
            y=trades["open_rate"],
            mode="markers",
            name="交易入场",
            text=trades["desc"],
            marker=dict(symbol="circle-open", size=11, line=dict(width=2), color="cyan"),
        )

        盈利出场 = go.Scatter(
            x=trades.loc[trades["profit_ratio"] > 0, "close_date"],
            y=trades.loc[trades["profit_ratio"] > 0, "close_rate"],
            text=trades.loc[trades["profit_ratio"] > 0, "desc"],
            mode="markers",
            name="出场 - 盈利",
            marker=dict(symbol="square-open", size=11, line=dict(width=2), color="green"),
        )
        亏损出场 = go.Scatter(
            x=trades.loc[trades["profit_ratio"] <= 0, "close_date"],
            y=trades.loc[trades["profit_ratio"] <= 0, "close_rate"],
            text=trades.loc[trades["profit_ratio"] <= 0, "desc"],
            mode="markers",
            name="出场 - 亏损",
            marker=dict(symbol="square-open", size=11, line=dict(width=2), color="red"),
        )
        fig.add_trace(入场点, 1, 1)
        fig.add_trace(盈利出场, 1, 1)
        fig.add_trace(亏损出场, 1, 1)
    else:
        logger.warning("未找到交易。")
    return fig


def create_plotconfig(
    indicators1: list[str], indicators2: list[str], plot_config: dict[str, dict]
) -> dict[str, dict]:
    """
    必要时将指标1和指标2合并到plot_config中
    :param indicators1: 包含主图表指标的列表
    :param indicators2: 包含子图表指标的列表
    :param plot_config: 包含高级绘图配置的字典的字典
    :return: plot_config - 最终包含指标1和2
    """

    if plot_config:
        if indicators1:
            plot_config["main_plot"] = {ind: {} for ind in indicators1}
        if indicators2:
            plot_config["subplots"] = {"Other": {ind: {} for ind in indicators2}}

    if not plot_config:
        # 如果没有指标且没有给出plot-config，则使用默认值。
        if not indicators1:
            indicators1 = ["sma", "ema3", "ema5"]
        if not indicators2:
            indicators2 = ["macd", "macdsignal"]

        # 如果plot_config不可用，则创建子图表配置。
        plot_config = {
            "main_plot": {ind: {} for ind in indicators1},
            "subplots": {"Other": {ind: {} for ind in indicators2}},
        }
    if "main_plot" not in plot_config:
        plot_config["main_plot"] = {}

    if "subplots" not in plot_config:
        plot_config["subplots"] = {}
    return plot_config


def plot_area(
    fig,
    row: int,
    data: pd.DataFrame,
    indicator_a: str,
    indicator_b: str,
    label: str = "",
    fill_color: str = "rgba(0,176,246,0.2)",
) -> make_subplots:
    """创建两个轨迹之间区域的图表并将其添加到fig。
    :param fig: 要附加的绘图对象
    :param row: 此绘图的行号
    :param data: 蜡烛图数据框
    :param indicator_a: 策略中填充的指标名称
    :param indicator_b: 策略中填充的指标名称
    :param label: 填充区域的标签
    :param fill_color: 用于填充区域的颜色
    :return: 添加了填充轨迹图表的fig
    """
    if indicator_a in data and indicator_b in data:
        # 使线条不可见，只绘制区域。
        line = {"color": "rgba(255,255,255,0)"}
        # TODO: 弄清楚为什么scattergl会导致问题 plotly/plotly.js#2284
        trace_a = go.Scatter(x=data.date, y=data[indicator_a], showlegend=False, line=line)
        trace_b = go.Scatter(
            x=data.date,
            y=data[indicator_b],
            name=label,
            fill="tonexty",
            fillcolor=fill_color,
            line=line,
        )
        fig.add_trace(trace_a, row, 1)
        fig.add_trace(trace_b, row, 1)
    return fig


def add_areas(fig, row: int, data: pd.DataFrame, indicators) -> make_subplots:
    """将所有区域图（在plot_config中指定）添加到fig。
    :param fig: 要附加的绘图对象
    :param row: 此绘图的行号
    :param data: 蜡烛图数据框
    :param indicators: 带指标的字典。例如：plot_config['main_plot'] 或 plot_config['subplots'][subplot_label]
    :return: 添加了填充轨迹图表的fig
    """
    for indicator, ind_conf in indicators.items():
        if "fill_to" in ind_conf:
            indicator_b = ind_conf["fill_to"]
            if indicator in data and indicator_b in data:
                label = ind_conf.get("fill_label", f"{indicator}<>{indicator_b}")
                fill_color = ind_conf.get("fill_color", "rgba(0,176,246,0.2)")
                fig = plot_area(
                    fig, row, data, indicator, indicator_b, label=label, fill_color=fill_color
                )
            elif indicator not in data:
                logger.info(
                    '指标 "%s" 被忽略。原因：在您的策略中未找到此指标。',
                    indicator,
                )
            elif indicator_b not in data:
                logger.info(
                    'fill_to: "%s" 被忽略。原因：此指标不在您的策略中。',
                    indicator_b,
                )
    return fig


def create_scatter(data, column_name, color, direction) -> go.Scatter | None:
    """创建散点图"""
    if column_name in data.columns:
        df_short = data[data[column_name] == 1]
        if len(df_short) > 0:
            shorts = go.Scatter(
                x=df_short.date,
                y=df_short.close,
                mode="markers",
                name=column_name,
                marker=dict(
                    symbol=f"triangle-{direction}-dot",
                    size=9,
                    line=dict(width=1),
                    color=color,
                ),
            )
            return shorts
        else:
            logger.warning(f"未找到 {column_name} 信号。")

    return None


def generate_candlestick_graph(
    pair: str,
    data: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    *,
    indicators1: list[str] | None = None,
    indicators2: list[str] | None = None,
    plot_config: dict[str, dict] | None = None,
) -> go.Figure:
    """
    从回测或数据库生成的数据生成图表
    成交量将始终绘制在第2行，因此第1行和第3行可供我们使用自定义指标
    :param pair: 要在图表上显示的交易对
    :param data: 包含指标和入场/出场信号的OHLCV数据框
    :param trades: 创建的所有交易
    :param indicators1: 包含主图表指标的列表
    :param indicators2: 包含子图表指标的列表
    :param plot_config: 包含高级绘图配置的字典的字典
    :return: Plotly图表
    """
    plot_config = create_plotconfig(
        indicators1 or [],
        indicators2 or [],
        plot_config or {},
    )
    rows = 2 + len(plot_config["subplots"])
    row_widths = [1 for _ in plot_config["subplots"]]
    # 定义图表
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        row_width=[*row_widths, 1, 4],
        vertical_spacing=0.0001,
    )
    fig["layout"].update(title=pair)
    fig["layout"]["yaxis1"].update(title="价格")
    fig["layout"]["yaxis2"].update(title="成交量")
    for i, name in enumerate(plot_config["subplots"]):
        fig["layout"][f"yaxis{3 + i}"].update(title=name)
    fig["layout"]["xaxis"]["rangeslider"].update(visible=False)
    fig.update_layout(modebar_add=["v1hovermode", "toggleSpikeLines"])

    # 通用信息
    蜡烛图 = go.Candlestick(
        x=data.date, open=data.open, high=data.high, low=data.low, close=data.close, name="价格"
    )
    fig.add_trace(蜡烛图, 1, 1)

    多单入场 = create_scatter(data, "enter_long", "green", "up")
    多单出场 = create_scatter(data, "exit_long", "red", "down")
    空单入场 = create_scatter(data, "enter_short", "blue", "down")
    空单出场 = create_scatter(data, "exit_short", "violet", "up")

    for scatter in [多单入场, 多单出场, 空单入场, 空单出场]:
        if scatter:
            fig.add_trace(scatter, 1, 1)

    # 添加布林带
    fig = plot_area(fig, 1, data, "bb_lowerband", "bb_upperband", label="布林带")
    # 防止bb_lower和bb_upper被绘制
    try:
        del plot_config["main_plot"]["bb_lowerband"]
        del plot_config["main_plot"]["bb_upperband"]
    except KeyError:
        pass
    # 主图表到第1行
    fig = add_indicators(fig=fig, row=1, indicators=plot_config["main_plot"], data=data)
    fig = add_areas(fig, 1, data, plot_config["main_plot"])
    fig = plot_trades(fig, trades)
    # 子图表：成交量到第2行
    成交量 = go.Bar(
        x=data["date"],
        y=data["volume"],
        name="成交量",
        marker_color="DarkSlateGrey",
        marker_line_color="DarkSlateGrey",
    )
    fig.add_trace(成交量, 2, 1)
    # 将每个子图表添加到单独的行
    for i, label in enumerate(plot_config["subplots"]):
        sub_config = plot_config["subplots"][label]
        row = 3 + i
        fig = add_indicators(fig=fig, row=row, indicators=sub_config, data=data)
        # 填充指标之间的区域 ('fill_to': 'other_indicator')
        fig = add_areas(fig, row, data, sub_config)

    return fig


def generate_profit_graph(
    pairs: str,
    data: dict[str, pd.DataFrame],
    trades: pd.DataFrame,
    timeframe: str,
    stake_currency: str,
    starting_balance: float,
) -> go.Figure:
    """生成利润图表"""
    # 合并所有交易对的收盘价，将列重命名为"pair"
    try:
        df_comb = combine_dataframes_with_mean(data, "close")
    except ValueError:
        raise OperationalException(
            "未找到数据。请确保所选时间范围和交易对有可用数据。"
        )

    # 将交易修剪到可用的OHLCV数据
    trades = extract_trades_of_period(df_comb, trades, date_index=True)
    if len(trades) == 0:
        raise OperationalException("在选定的时间范围内未找到交易。")

    # 添加合并的累积利润
    df_comb = create_cum_profit(df_comb, trades, "cum_profit", timeframe)

    # 绘制交易对的平均收盘价和总利润增长
    平均收盘价 = go.Scatter(
        x=df_comb.index,
        y=df_comb["mean"],
        name="平均收盘价",
    )

    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        row_heights=[1, 1, 1, 0.5, 0.75, 0.75],
        vertical_spacing=0.05,
        subplot_titles=[
            "平均收盘价",
            "合并利润",
            "每个交易对的利润",
            "并行度",
            "水下图表",
            "相对回撤",
        ],
    )
    fig["layout"].update(title="Freqtrade利润图表")
    fig["layout"]["yaxis1"].update(title="价格")
    fig["layout"]["yaxis2"].update(title=f"利润 {stake_currency}")
    fig["layout"]["yaxis3"].update(title=f"利润 {stake_currency}")
    fig["layout"]["yaxis4"].update(title="交易数量")
    fig["layout"]["yaxis5"].update(title="水下图表")
    fig["layout"]["yaxis6"].update(title="相对水下图表 (%)", tickformat=",.2%")
    fig["layout"]["xaxis"]["rangeslider"].update(visible=False)
    fig.update_layout(modebar_add=["v1hovermode", "toggleSpikeLines"])

    fig.add_trace(平均收盘价, 1, 1)
    fig = add_profit(fig, 2, df_comb, "cum_profit", "利润")
    fig = add_max_drawdown(fig, 2, trades, df_comb, timeframe, starting_balance)
    fig = add_parallelism(fig, 4, trades, timeframe)
    # 占用两行
    fig = add_underwater(fig, 5, trades, starting_balance)

    for pair in pairs:
        profit_col = f"cum_profit_{pair}"
        try:
            df_comb = create_cum_profit(
                df_comb, trades[trades["pair"] == pair], profit_col, timeframe
            )
            fig = add_profit(fig, 3, df_comb, profit_col, f"{pair} 利润")
        except ValueError:
            pass
    return fig


def generate_plot_filename(pair: str, timeframe: str) -> str:
    """
    为每个交易对/时间范围生成用于存储图表的文件名
    """
    pair_s = pair_to_filename(pair)
    file_name = "freqtrade-图表-" + pair_s + "-" + timeframe + ".html"

    logger.info("为 %s 生成图表文件", pair)

    return file_name


def store_plot_file(fig, filename: str, directory: Path, auto_open: bool = False) -> None:
    """
    从预填充的fig plotly对象生成图表html文件
    :param fig: 要绘制的Plotly图表
    :param filename: 存储文件的名称
    :param directory: 存储文件的目录
    :param auto_open: 自动打开保存的文件
    :return: None
    """
    directory.mkdir(parents=True, exist_ok=True)

    _filename = directory.joinpath(filename)
    plot(fig, filename=str(_filename), auto_open=auto_open)
    logger.info(f"图表已存储为 {_filename}")


def load_and_plot_trades(config: Config):
    """
    根据提供的配置
    - 初始化绘图脚本
    - 获取蜡烛图(OHLCV)数据
    - 生成填充了基于配置策略的指标和信号的数据框
    - 加载在选定期间执行的交易
    - 生成Plotly绘图对象
    - 生成图表文件
    :return: None
    """
    strategy = StrategyResolver.load_strategy(config)

    exchange = ExchangeResolver.load_exchange(config)
    IStrategy.dp = DataProvider(config, exchange)
    strategy.ft_bot_start()
    strategy_safe_wrapper(strategy.bot_loop_start)(current_time=datetime.now(timezone.utc))
    plot_elements = init_plotscript(config, list(exchange.markets), strategy.startup_candle_count)
    timerange = plot_elements["timerange"]
    trades = plot_elements["trades"]
    pair_counter = 0
    for pair, data in plot_elements["ohlcv"].items():
        pair_counter += 1
        logger.info("分析交易对 %s", pair)

        df_analyzed = strategy.analyze_ticker(data, {"pair": pair})
        df_analyzed = trim_dataframe(df_analyzed, timerange)
        if not trades.empty:
            trades_pair = trades.loc[trades["pair"] == pair]
            trades_pair = extract_trades_of_period(df_analyzed, trades_pair)
        else:
            trades_pair = trades

        fig = generate_candlestick_graph(
            pair=pair,
            data=df_analyzed,
            trades=trades_pair,
            indicators1=config.get("indicators1", []),
            indicators2=config.get("indicators2", []),
            plot_config=strategy.plot_config if hasattr(strategy, "plot_config") else {},
        )

        store_plot_file(
            fig,
            filename=generate_plot_filename(pair, config["timeframe"]),
            directory=config["user_data_dir"] / "plot",
        )

    logger.info("绘图过程结束。生成了 %s 个图表", pair_counter)


def plot_profit(config: Config) -> None:
    """
    绘制所有交易对的总利润。
    注意，利润计算并不完全符合实际情况。
    但应该成比例，因此有助于找到好的算法。
    """
    if "timeframe" not in config:
        raise OperationalException("时间范围必须在配置中或通过--timeframe设置。")

    exchange = ExchangeResolver.load_exchange(config)
    plot_elements = init_plotscript(config, list(exchange.markets))
    trades = plot_elements["trades"]
    # 过滤相关交易对的交易
    # 移除未平仓交易对 - 我们还不知道利润，因此无法计算这些交易对的利润。
    # 此外，如果只剩下一个未平仓交易对，利润生成将失败。
    trades = trades[
        (trades["pair"].isin(plot_elements["pairs"])) & (~trades["close_date"].isnull())
    ]
    if len(trades) == 0:
        raise OperationalException(
            "未找到交易，没有来自回测结果或数据库的交易无法生成利润图表。"
        )

    # 创建所有相关交易对的平均收盘价。
    # 这可能有助于评估整体市场趋势
    fig = generate_profit_graph(
        plot_elements["pairs"],
        plot_elements["ohlcv"],
        trades,
        config["timeframe"],
        config.get("stake_currency", ""),
        config.get("available_capital", get_dry_run_wallet(config)),
    )
    store_plot_file(
        fig,
        filename="freqtrade-利润图表.html",
        directory=config["user_data_dir"] / "plot",
        auto_open=config.get("plot_auto_open", False),
    )