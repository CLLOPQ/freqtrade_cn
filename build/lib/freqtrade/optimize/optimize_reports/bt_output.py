import logging
from typing import Any, Literal

from freqtrade.constants import UNLIMITED_STAKE_AMOUNT, Config
from freqtrade.ft_types import BacktestResultType
from freqtrade.optimize.optimize_reports.optimize_reports import generate_periodic_breakdown_stats
from freqtrade.util import decimals_per_coin, fmt_coin, print_rich_table


logger = logging.getLogger(__name__)


def _get_line_floatfmt(stake_currency: str) -> list[str]:
    """
    生成浮点格式（与 _generate_result_line() 一致）
    """
    return ["s", "d", ".2f", f".{decimals_per_coin(stake_currency)}f", ".2f", "d", "s", "s"]


def _get_line_header(
    first_column: str | list[str], stake_currency: str, direction: str = "Trades"
) -> list[str]:
    """
    生成表头行（与 _generate_result_line() 一致）
    """
    return [
        *([first_column] if isinstance(first_column, str) else first_column),
        direction,
        "平均收益率 %",
        f"总盈利 {stake_currency}",
        "总收益率 %",
        "平均持有时长",
        "赢  平  输  胜率",
    ]


def generate_wins_draws_losses(wins, draws, losses):
    if wins > 0 and losses == 0:
        wl_ratio = "100"
    elif wins == 0:
        wl_ratio = "0"
    else:
        wl_ratio = f"{100.0 / (wins + draws + losses) * wins:.1f}" if losses > 0 else "100"
    return f"{wins:>4}  {draws:>4}  {losses:>4}  {wl_ratio:>4}"


def text_table_bt_results(
    pair_results: list[dict[str, Any]], stake_currency: str, title: str
) -> None:
    """
    为给定的回测数据和结果数据框生成并返回文本表格
    :param pair_results: 字典列表 - 每个交易对一个条目 + 最终的 TOTAL 行
    :param stake_currency: 投资货币 - 用于正确命名表头
    :param title: 表格标题
    """

    headers = _get_line_header("交易对", stake_currency, "交易数")
    output = [
        [
            t["key"],
            t["trades"],
            t["profit_mean_pct"],
            f"{t['profit_total_abs']:.{decimals_per_coin(stake_currency)}f}",
            t["profit_total_pct"],
            t["duration_avg"],
            generate_wins_draws_losses(t["wins"], t["draws"], t["losses"]),
        ]
        for t in pair_results
    ]
    # 忽略类型，因为 floatfmt 允许元组但 mypy 不知道
    print_rich_table(output, headers, summary=title)


def text_table_tags(
    tag_type: Literal["enter_tag", "exit_tag", "mix_tag"],
    tag_results: list[dict[str, Any]],
    stake_currency: str,
) -> None:
    """
    为给定的回测数据和结果数据框生成并返回文本表格
    :param pair_results: 字典列表 - 每个标签一个条目
    :param stake_currency: 投资货币 - 用于正确命名表头
    """
    floatfmt = _get_line_floatfmt(stake_currency)
    fallback: str = ""
    is_list = False
    if tag_type == "enter_tag":
        title = "入场标签"
        headers = _get_line_header(title, stake_currency, "入场数")
    elif tag_type == "exit_tag":
        title = "出场原因"
        headers = _get_line_header(title, stake_currency, "出场数")
        fallback = "exit_reason"
    else:
        # 混合标签
        title = "混合标签"
        headers = _get_line_header(["入场标签", "出场原因"], stake_currency, "交易数")
        floatfmt.insert(0, "s")
        is_list = True

    output = [
        [
            *(
                (
                    list(t["key"])
                    if isinstance(t["key"], list | tuple)
                    else [t["key"], ""]
                    if is_list
                    else [t["key"]]
                )
                if t.get("key") is not None and len(str(t["key"])) > 0
                else [t.get(fallback, "其他")]
            ),
            t["trades"],
            t["profit_mean_pct"],
            f"{t['profit_total_abs']:.{decimals_per_coin(stake_currency)}f}",
            t["profit_total_pct"],
            t.get("duration_avg"),
            generate_wins_draws_losses(t["wins"], t["draws"], t["losses"]),
        ]
        for t in tag_results
    ]
    # 忽略类型，因为 floatfmt 允许元组但 mypy 不知道
    print_rich_table(output, headers, summary=f"{title.upper()} 统计")


def text_table_periodic_breakdown(
    days_breakdown_stats: list[dict[str, Any]], stake_currency: str, period: str
) -> None:
    """
    按日生成包含回测结果的小表格
    :param days_breakdown_stats: 按日细分指标
    :param stake_currency: 使用的投资货币
    """
    headers = [
        period.capitalize(),
        "交易数",
        f"总盈利 {stake_currency}",
        "盈利因子",
        "赢  平  输  胜率",
    ]
    output = [
        [
            d["date"],
            d.get("trades", "N/A"),
            fmt_coin(d["profit_abs"], stake_currency, False),
            round(d["profit_factor"], 2) if "profit_factor" in d else "N/A",
            generate_wins_draws_losses(d["wins"], d["draws"], d.get("losses", d.get("loses", 0))),
        ]
        for d in days_breakdown_stats
    ]
    print_rich_table(output, headers, summary=f"{period.upper()} 细分")


def text_table_strategy(strategy_results, stake_currency: str, title: str):
    """
    按策略生成汇总表格
    :param strategy_results: <策略名: DataFrame>的字典，包含所有策略的结果
    :param stake_currency: 投资货币 - 用于正确命名表头
    """
    headers = _get_line_header("策略", stake_currency, "交易数")
    # _get_line_header() 也用于每交易对汇总。每交易对的回撤通常无用
    # 因此仅在策略汇总时在此处插入该列。
    headers.append("最大回撤")

    # 将回撤字符串居中对齐（两个空格分隔）。
    if "max_drawdown_account" in strategy_results[0]:
        drawdown = [f"{t['max_drawdown_account'] * 100:.2f}" for t in strategy_results]
    else:
        # 支持早期回测结果
        drawdown = [f"{t['max_drawdown_per']:.2f}" for t in strategy_results]

    dd_pad_abs = max([len(t["max_drawdown_abs"]) for t in strategy_results])
    dd_pad_per = max([len(dd) for dd in drawdown])
    drawdown = [
        f"{t['max_drawdown_abs']:>{dd_pad_abs}} {stake_currency}  {dd:>{dd_pad_per}}%"
        for t, dd in zip(strategy_results, drawdown, strict=False)
    ]

    output = [
        [
            t["key"],
            t["trades"],
            f"{t['profit_mean_pct']:.2f}",
            f"{t['profit_total_abs']:.{decimals_per_coin(stake_currency)}f}",
            t["profit_total_pct"],
            t["duration_avg"],
            generate_wins_draws_losses(t["wins"], t["draws"], t["losses"]),
            drawdown,
        ]
        for t, drawdown in zip(strategy_results, drawdown, strict=False)
    ]
    print_rich_table(output, headers, summary=title)


def text_table_add_metrics(strat_results: dict) -> None:
    if len(strat_results["trades"]) > 0:
        best_trade = max(strat_results["trades"], key=lambda x: x["profit_ratio"])
        worst_trade = min(strat_results["trades"], key=lambda x: x["profit_ratio"])

        short_metrics = (
            [
                ("", ""),  # 空行以提高可读性
                (
                    "多仓/空仓",
                    f"{strat_results.get('trade_count_long', '总交易数')} / "
                    f"{strat_results.get('trade_count_short', 0)}",
                ),
                ("多仓总收益率 %", f"{strat_results['profit_total_long']:.2%}"),
                ("空仓总收益率 %", f"{strat_results['profit_total_short']:.2%}"),
                (
                    "多仓绝对收益",
                    fmt_coin(
                        strat_results["profit_total_long_abs"], strat_results["stake_currency"]
                    ),
                ),
                (
                    "空仓绝对收益",
                    fmt_coin(
                        strat_results["profit_total_short_abs"], strat_results["stake_currency"]
                    ),
                ),
            ]
            if strat_results.get("trade_count_short", 0) > 0
            else []
        )

        drawdown_metrics = []
        if "max_relative_drawdown" in strat_results:
            # 兼容显示旧的超参数优化结果
            drawdown_metrics.append(
                ("账户最大水下比例", f"{strat_results['max_relative_drawdown']:.2%}")
            )
        drawdown_metrics.extend(
            [
                (
                    ("绝对回撤（账户）", f"{strat_results['max_drawdown_account']:.2%}")
                    if "max_drawdown_account" in strat_results
                    else ("最大回撤", f"{strat_results['max_drawdown']:.2%}")
                ),
                (
                    "绝对回撤",
                    fmt_coin(strat_results["max_drawdown_abs"], strat_results["stake_currency"]),
                ),
                (
                    "回撤高点",
                    fmt_coin(strat_results["max_drawdown_high"], strat_results["stake_currency"]),
                ),
                (
                    "回撤低点",
                    fmt_coin(strat_results["max_drawdown_low"], strat_results["stake_currency"]),
                ),
                ("回撤开始时间", strat_results["drawdown_start"]),
                ("回撤结束时间", strat_results["drawdown_end"]),
            ]
        )

        entry_adjustment_metrics = (
            [
                ("取消的交易入场", strat_results.get("canceled_trade_entries", "N/A")),
                ("取消的入场订单", strat_results.get("canceled_entry_orders", "N/A")),
                ("替换的入场订单", strat_results.get("replaced_entry_orders", "N/A")),
            ]
            if strat_results.get("canceled_entry_orders", 0) > 0
            else []
        )

        trading_mode = (
            (
                [
                    (
                        "交易模式",
                        (
                            ""
                            if not strat_results.get("margin_mode")
                            or strat_results.get("trading_mode", "spot") == "spot"
                            else f"{strat_results['margin_mode'].capitalize()} "
                        )
                        + f"{strat_results['trading_mode'].capitalize()}",
                    )
                ]
            )
            if "trading_mode" in strat_results
            else []
        )

        # 新添加的字段如果在 strat_results 中缺失则忽略。hyperopt-show
        # 命令存储这些结果，较新版本的 freqtrade 必须能处理缺少新字段的旧结果。
        metrics = [
            ("回测开始时间", strat_results["backtest_start"]),
            ("回测结束时间", strat_results["backtest_end"]),
            *trading_mode,
            ("最大持仓数", strat_results["max_open_trades"]),
            ("", ""),  # 空行以提高可读性
            (
                "总交易数/日均交易数",
                f"{strat_results['total_trades']} / {strat_results['trades_per_day']}",
            ),
            (
                "初始资金",
                fmt_coin(strat_results["starting_balance"], strat_results["stake_currency"]),
            ),
            (
                "最终资金",
                fmt_coin(strat_results["final_balance"], strat_results["stake_currency"]),
            ),
            (
                "绝对收益",
                fmt_coin(strat_results["profit_total_abs"], strat_results["stake_currency"]),
            ),
            ("总收益率 %", f"{strat_results['profit_total']:.2%}"),
            ("年化收益率 %", f"{strat_results['cagr']:.2%}" if "cagr" in strat_results else "N/A"),
            ("索丁诺比率", f"{strat_results['sortino']:.2f}" if "sortino" in strat_results else "N/A"),
            ("夏普比率", f"{strat_results['sharpe']:.2f}" if "sharpe" in strat_results else "N/A"),
            ("卡尔玛比率", f"{strat_results['calmar']:.2f}" if "calmar" in strat_results else "N/A"),
            ("系统质量指数", f"{strat_results['sqn']:.2f}" if "sqn" in strat_results else "N/A"),
            (
                "盈利因子",
                (
                    f"{strat_results['profit_factor']:.2f}"
                    if "profit_factor" in strat_results
                    else "N/A"
                ),
            ),
            (
                "期望收益（比率）",
                (
                    f"{strat_results['expectancy']:.2f} ({strat_results['expectancy_ratio']:.2f})"
                    if "expectancy_ratio" in strat_results
                    else "N/A"
                ),
            ),
            (
                "日均收益率 %",
                f"{(strat_results['profit_total'] / strat_results['backtest_days']):.2%}",
            ),
            (
                "平均投资金额",
                fmt_coin(strat_results["avg_stake_amount"], strat_results["stake_currency"]),
            ),
            (
                "总交易金额",
                fmt_coin(strat_results["total_volume"], strat_results["stake_currency"]),
            ),
            *short_metrics,
            ("", ""),  # 空行以提高可读性
            (
                "最佳交易对",
                f"{strat_results['best_pair']['key']} "
                f"{strat_results['best_pair']['profit_total']:.2%}",
            ),
            (
                "最差交易对",
                f"{strat_results['worst_pair']['key']} "
                f"{strat_results['worst_pair']['profit_total']:.2%}",
            ),
            ("最佳交易", f"{best_trade['pair']} {best_trade['profit_ratio']:.2%}"),
            ("最差交易", f"{worst_trade['pair']} {worst_trade['profit_ratio']:.2%}"),
            (
                "最佳日收益",
                fmt_coin(strat_results["backtest_best_day_abs"], strat_results["stake_currency"]),
            ),
            (
                "最差日收益",
                fmt_coin(strat_results["backtest_worst_day_abs"], strat_results["stake_currency"]),
            ),
            (
                "盈利/平盘/亏损天数",
                f"{strat_results['winning_days']} / "
                f"{strat_results['draw_days']} / {strat_results['losing_days']}",
            ),
            (
                "盈利交易最小/最大/平均持有时长",
                f"{strat_results.get('winner_holding_min', 'N/A')} / "
                f"{strat_results.get('winner_holding_max', 'N/A')} / "
                f"{strat_results.get('winner_holding_avg', 'N/A')}",
            ),
            (
                "亏损交易最小/最大/平均持有时长",
                f"{strat_results.get('loser_holding_min', 'N/A')} / "
                f"{strat_results.get('loser_holding_max', 'N/A')} / "
                f"{strat_results.get('loser_holding_avg', 'N/A')}",
            ),
            (
                "最大连续盈利/亏损次数",
                (
                    (
                        f"{strat_results['max_consecutive_wins']} / "
                        f"{strat_results['max_consecutive_losses']}"
                    )
                    if "max_consecutive_losses" in strat_results
                    else "N/A"
                ),
            ),
            ("被拒绝的入场信号", strat_results.get("rejected_signals", "N/A")),
            (
                "入场/出场超时次数",
                f"{strat_results.get('timedout_entry_orders', 'N/A')} / "
                f"{strat_results.get('timedout_exit_orders', 'N/A')}",
            ),
            *entry_adjustment_metrics,
            ("", ""),  # 空行以提高可读性
            ("最小资金余额", fmt_coin(strat_results["csum_min"], strat_results["stake_currency"])),
            ("最大资金余额", fmt_coin(strat_results["csum_max"], strat_results["stake_currency"])),
            *drawdown_metrics,
            ("市场变化", f"{strat_results['market_change']:.2%}"),
        ]
        print_rich_table(metrics, ["指标", "数值"], summary="汇总指标", justify="left")

    else:
        start_balance = fmt_coin(strat_results["starting_balance"], strat_results["stake_currency"])
        stake_amount = (
            fmt_coin(strat_results["stake_amount"], strat_results["stake_currency"])
            if strat_results["stake_amount"] != UNLIMITED_STAKE_AMOUNT
            else "无限"
        )

        message = (
            "未产生交易。"
            f"您的初始资金为 {start_balance}，"
            f"投资金额为 {stake_amount}。"
        )
        print(message)


def _show_tag_subresults(results: dict[str, Any], stake_currency: str):
    """
    打印标签子结果（入场标签、出场原因汇总、混合标签统计）
    """
    if (enter_tags := results.get("results_per_enter_tag")) is not None:
        text_table_tags("enter_tag", enter_tags, stake_currency)

    if (exit_reasons := results.get("exit_reason_summary")) is not None:
        text_table_tags("exit_tag", exit_reasons, stake_currency)

    if (mix_tag := results.get("mix_tag_stats")) is not None:
        text_table_tags("mix_tag", mix_tag, stake_currency)


def show_backtest_result(
    strategy: str, results: dict[str, Any], stake_currency: str, backtest_breakdown: list[str]
):
    """
    打印单个策略的结果
    """
    # 打印结果
    print(f"策略 {strategy} 的结果")
    text_table_bt_results(
        results["results_per_pair"], stake_currency=stake_currency, title="回测报告"
    )
    text_table_bt_results(
        results["left_open_trades"], stake_currency=stake_currency, title="未平仓交易报告"
    )

    _show_tag_subresults(results, stake_currency)

    for period in backtest_breakdown:
        if period in results.get("periodic_breakdown", {}):
            days_breakdown_stats = results["periodic_breakdown"][period]
        else:
            days_breakdown_stats = generate_periodic_breakdown_stats(
                trade_list=results["trades"], period=period
            )
        text_table_periodic_breakdown(
            days_breakdown_stats=days_breakdown_stats, stake_currency=stake_currency, period=period
        )

    text_table_add_metrics(results)

    print()


def show_backtest_results(config: Config, backtest_stats: BacktestResultType):
    stake_currency = config["stake_currency"]

    for strategy, results in backtest_stats["strategy"].items():
        show_backtest_result(
            strategy, results, stake_currency, config.get("backtest_breakdown", [])
        )

    if len(backtest_stats["strategy"]) > 0:
        # 打印策略汇总表格
        if backtest_stats["strategy"]:
            first_strategy = next(iter(backtest_stats["strategy"].values()))
            print(
                f"回测时间 {first_strategy['backtest_start']} -> {first_strategy['backtest_end']} |"
                f" 最大持仓数：{first_strategy['max_open_trades']}"
            )
        text_table_strategy(
            backtest_stats["strategy_comparison"], stake_currency, "策略汇总"
        )


def show_sorted_pairlist(config: Config, backtest_stats: BacktestResultType):
    if config.get("backtest_show_pair_list", False):
        for strategy, results in backtest_stats["strategy"].items():
            print(f"策略 {strategy} 的交易对： \n[")
            for result in results["results_per_pair"]:
                if result["key"] != "TOTAL":
                    print(f'"{result["key"]}",  // {result["profit_mean"]:.2%}')
            print("]")