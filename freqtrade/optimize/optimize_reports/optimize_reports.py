import logging
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import numpy as np
from pandas import DataFrame, Series, concat, to_datetime

from freqtrade.constants import BACKTEST_BREAKDOWNS, DATETIME_PRINT_FORMAT
from freqtrade.data.metrics import (
    calculate_cagr,
    calculate_calmar,
    calculate_csum,
    calculate_expectancy,
    calculate_market_change,
    calculate_max_drawdown,
    calculate_sharpe,
    calculate_sortino,
    calculate_sqn,
)
from freqtrade.ft_types import (
    BacktestContentType,
    BacktestResultType,
    get_BacktestResultType_default,
)
from freqtrade.util import decimals_per_coin, fmt_coin, format_duration, get_dry_run_wallet


logger = logging.getLogger(__name__)


def generate_trade_signal_candles(
    preprocessed_df: dict[str, DataFrame], bt_results: BacktestContentType, date_col: str
) -> dict[str, DataFrame]:
    # 仅保留信号蜡烛图数据
    signal_candles_only = {}
    for pair in preprocessed_df.keys():
        # 创建信号蜡烛图DataFrame
        signal_candles_only_df = DataFrame()

        # 获取当前交易对的预处理数据
        pairdf = preprocessed_df[pair]
        # 获取回测结果中的交易对数据
        resdf = bt_results["results"]
        pairresults = resdf.loc[(resdf["pair"] == pair)]

        # 如果预处理数据不为空
        if pairdf.shape[0] > 0:
            # 遍历每个交易结果
            for t, v in pairresults.iterrows():
                # 获取交易前的所有K线
                allinds = pairdf.loc[(pairdf["date"] < v[date_col])]
                # 取交易前最后一根K线作为信号K线
                signal_inds = allinds.iloc[[-1]]
                # 合并信号K线到DataFrame
                signal_candles_only_df = concat(
                    [signal_candles_only_df.infer_objects(), signal_inds.infer_objects()]
                )

            # 存储当前交易对的信号蜡烛图
            signal_candles_only[pair] = signal_candles_only_df
    return signal_candles_only
def generate_rejected_signals(
    preprocessed_df: dict[str, DataFrame], rejected_dict: dict[str, DataFrame]
) -> dict[str, DataFrame]:
    rejected_candles_only = {}
    for pair, signals in rejected_dict.items():
        rejected_signals_only_df = DataFrame()
        pairdf = preprocessed_df[pair]

        for t in signals:
            data_df_row = pairdf.loc[(pairdf["date"] == t[0])].copy()
            data_df_row["pair"] = pair
            data_df_row["enter_tag"] = t[1]

            rejected_signals_only_df = concat(
                [rejected_signals_only_df.infer_objects(), data_df_row.infer_objects()]
            )

        rejected_candles_only[pair] = rejected_signals_only_df
    return rejected_candles_only


def _generate_result_line(
    result: DataFrame,
    min_date: datetime,
    max_date: datetime,
    starting_balance: float,
    first_column: str | list[str],
) -> dict:
    """
    生成一个结果字典，以“first_column”作为键。
    """
    profit_sum = result["profit_ratio"].sum()
    # （期末资金 - 初始资金）/ 初始资金
    profit_total = result["profit_abs"].sum() / starting_balance
    backtest_days = (max_date - min_date).days or 1
    final_balance = starting_balance + result["profit_abs"].sum()
    expectancy, expectancy_ratio = calculate_expectancy(result)
    winning_profit = result.loc[result["profit_abs"] > 0, "profit_abs"].sum()
    losing_profit = result.loc[result["profit_abs"] < 0, "profit_abs"].sum()
    profit_factor = winning_profit / abs(losing_profit) if losing_profit else 0.0

    try:
        # 计算最大回撤
        drawdown = calculate_max_drawdown(
            result, value_col="profit_abs", starting_balance=starting_balance
        )

    except ValueError:
        drawdown = None

    return {
        "key": first_column,
        "trades": len(result),
        "profit_mean": result["profit_ratio"].mean() if len(result) > 0 else 0.0,
        "profit_mean_pct": (
            round(result["profit_ratio"].mean() * 100.0, 2) if len(result) > 0 else 0.0
        ),
        "profit_sum": profit_sum,
        "profit_sum_pct": round(profit_sum * 100.0, 2),
        "profit_total_abs": result["profit_abs"].sum(),
        "profit_total": profit_total,
        "profit_total_pct": round(profit_total * 100.0, 2),
        "duration_avg": (
            str(timedelta(minutes=round(result["trade_duration"].mean())))
            if not result.empty
            else "0:00"
        ),
        # '最大回撤持续时间': str(timedelta(
        #                     minutes=round(result['trade_duration'].max()))
        #                     ) if not result.empty else '0:00',
        # '最小回撤持续时间': str(timedelta(
        #                     minutes=round(result['trade_duration'].min()))
        #                     ) if not result.empty else '0:00',
        "wins": len(result[result["profit_abs"] > 0]),
        "draws": len(result[result["profit_abs"] == 0]),
        "losses": len(result[result["profit_abs"] < 0]),
        "winrate": len(result[result["profit_abs"] > 0]) / len(result) if len(result) else 0.0,
        "cagr": calculate_cagr(backtest_days, starting_balance, final_balance),
        "expectancy": expectancy,
        "expectancy_ratio": expectancy_ratio,
        "sortino": calculate_sortino(result, min_date, max_date, starting_balance),
        "sharpe": calculate_sharpe(result, min_date, max_date, starting_balance),
        "calmar": calculate_calmar(result, min_date, max_date, starting_balance),
        "sqn": calculate_sqn(result, starting_balance),
        "profit_factor": profit_factor,
        "max_drawdown_account": drawdown.relative_account_drawdown if drawdown else 0.0,
        "max_drawdown_abs": drawdown.drawdown_abs if drawdown else 0.0,
    }
def calculate_trade_volume(trades_dict: list[dict[str, Any]]) -> float:
    # 汇总从订单成本中交易的总交易量。订单是交易列表中的嵌套字典。

    return sum(sum(order["cost"] for order in trade.get("orders", [])) for trade in trades_dict)


def generate_pair_metrics(  #
    pairlist: list[str],
    stake_currency: str,
    starting_balance: float,
    results: DataFrame,
    min_date: datetime,
    max_date: datetime,
    skip_nan: bool = False,
) -> list[dict]:
    """
    为给定的回测数据和结果数据框生成并返回一个列表
    :param pairlist: 交易对列表：使用的交易对列表
    :param stake_currency: 持仓货币：持仓货币——用于正确命名表头
    :param starting_balance: 初始余额：初始资金
    :param results: 结果：包含回测结果的数据框
    :param skip_nan: 跳过NaN：打印“未平仓”的未平仓交易
    :return: 返回：包含每个交易对指标的字典列表
    """

    tabular_data = []

    for pair in pairlist:
        result = results[results["pair"] == pair]
        if skip_nan and result["profit_abs"].isnull().all():
            continue

        tabular_data.append(
            _generate_result_line(result, min_date, max_date, starting_balance, pair)
        )

    # 按总利润率排序：
    tabular_data = sorted(tabular_data, key=lambda k: k["profit_total_abs"], reverse=True)

    # 添加总计
    tabular_data.append(
        _generate_result_line(results, min_date, max_date, starting_balance, "TOTAL")
    )

    return tabular_data
def generate_tag_metrics(
    tag_type: Literal["enter_tag", "exit_reason"] | list[Literal["enter_tag", "exit_reason"]],
    starting_balance: float,
    results: DataFrame,
    min_date: datetime,
    max_date: datetime,
    skip_nan: bool = False,
) -> list[dict]:
    """
    生成并返回给定标签交易和结果数据框的指标列表
    :param starting_balance: 起始余额
    :param results: 包含回测结果的数据框
    :param skip_nan: 跳过NaN：打印“未平仓”的未平仓交易
    :return: 按交易对包含指标的字典列表
    """

    tabular_data = []

    if all(
        tag in results.columns for tag in (tag_type if isinstance(tag_type, list) else [tag_type])
    ):
        for tags, group in results.groupby(tag_type):
            if skip_nan and group["profit_abs"].isnull().all():
                continue

            tabular_data.append(
                _generate_result_line(group, min_date, max_date, starting_balance, tags)
            )

        # Sort by total profit %:
        tabular_data = sorted(tabular_data, key=lambda k: k["profit_total_abs"], reverse=True)

        # Append Total
        tabular_data.append(
            _generate_result_line(results, min_date, max_date, starting_balance, "总计")
        )
        return tabular_data
    else:
        return []
def generate_strategy_comparison(bt_stats: dict) -> list[dict]:
    """
    为每个策略生成摘要
    :param bt_stats: 包含所有策略结果的字典，键为策略名称，值为数据框
    :return: 包含每个策略指标的字典列表
    """

    tabular_data = []
    for strategy, result in bt_stats.items():
        tabular_data.append(deepcopy(result["results_per_pair"][-1]))
        # 将"key"更新为策略名称（results_per_pair中的该键为"总计"）。
        tabular_data[-1]["key"] = strategy
        tabular_data[-1]["max_drawdown_account"] = result["max_drawdown_account"]
        tabular_data[-1]["max_drawdown_abs"] = fmt_coin(
            result["max_drawdown_abs"], result["stake_currency"], False
        )
    return tabular_data


def _get_resample_from_period(period: str) -> str:
    if period == "day":
        return "1d"
    if period == "week":
        # 周，默认周一。
        return "1W-MON"
    if period == "month":
        return "1ME"
    if period == "year":
        return "1YE"
    raise ValueError(f"不支持的周期 {period}。")
def generate_periodic_breakdown_stats(
    trade_list: list | DataFrame, period: str
) -> list[dict[str, Any]]:
    results = trade_list if not isinstance(trade_list, list) else DataFrame.from_records(trade_list)
    if len(results) == 0:
        return []
    results["close_date"] = to_datetime(results["close_date"], utc=True)
    resample_period = _get_resample_from_period(period)
    resampled = results.resample(resample_period, on="close_date")
    stats = []
    for name, day in resampled:
        profit_abs = day["profit_abs"].sum().round(10)
        wins = sum(day["profit_abs"] > 0)
        draws = sum(day["profit_abs"] == 0)
        losses = sum(day["profit_abs"] < 0)
        trades = wins + draws + losses
        winning_profit = day.loc[day["profit_abs"] > 0, "profit_abs"].sum()
        losing_profit = day.loc[day["profit_abs"] < 0, "profit_abs"].sum()
        profit_factor = winning_profit / abs(losing_profit) if losing_profit else 0.0
        stats.append(
            {
                "日期": name.strftime("%d/%m/%Y"),
                "日期时间戳": int(name.to_pydatetime().timestamp() * 1000),
                "总盈利": profit_abs,
                "盈利次数": wins,
                "平局次数": draws,
                "亏损次数": losses,
                "交易总数": trades,
                "盈亏比": round(profit_factor, 8),
            }
        )
    return stats
def generate_all_periodic_breakdown_stats(trade_list: list) -> dict[str, list]:
    result = {}
    for period in BACKTEST_BREAKDOWNS:
        result[period] = generate_periodic_breakdown_stats(trade_list, period)
    return result


def calc_streak(dataframe: DataFrame) -> tuple[int, int]:
    """
    计算连续赢和连续输的周期
    :param dataframe: 包含交易数据框的DataFrame，带有profit_ratio列
    :return: 包含连续赢和连续输的元组
    """

    df = Series(np.where(dataframe["profit_ratio"] > 0, "win", "loss")).to_frame("result")
    df["streaks"] = df["result"].ne(df["result"].shift()).cumsum().rename("streaks")
    df["counter"] = df["streaks"].groupby(df["streaks"]).cumcount() + 1
    res = df.groupby(df["result"]).max()
    #
    cons_wins = int(res.loc["win", "counter"]) if "win" in res.index else 0
    cons_losses = int(res.loc["loss", "counter"]) if "loss" in res.index else 0
    return cons_wins, cons_losses


def generate_trading_stats(results: DataFrame) -> dict[str, Any]:
    """生成整体交易统计数据"""
    if len(results) == 0:
        return {
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "winrate": 0,
            "holding_avg": timedelta(),
            "winner_holding_avg": timedelta(),
            "loser_holding_avg": timedelta(),
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
        }

    winning_trades = results.loc[results["profit_ratio"] > 0]
    winning_duration = winning_trades["trade_duration"]
    draw_trades = results.loc[results["profit_ratio"] == 0]
    losing_trades = results.loc[results["profit_ratio"] < 0]
    losing_duration = losing_trades["trade_duration"]

    holding_avg = (
        timedelta(minutes=round(results["trade_duration"].mean()))
        if not results.empty
        else timedelta()
    )
    winner_holding_min = (
        timedelta(minutes=round(winning_duration.min()))
        if not winning_duration.empty
        else timedelta()
    )
    winner_holding_max = (
        timedelta(minutes=round(winning_duration.max()))
        if not winning_duration.empty
        else timedelta()
    )
    winner_holding_avg = (
        timedelta(minutes=round(winning_duration.mean()))
        if not winning_duration.empty
        else timedelta()
    )
    loser_holding_min = (
        timedelta(minutes=round(losing_duration.min()))
        if not losing_duration.empty
        else timedelta()
    )
    loser_holding_max = (
        timedelta(minutes=round(losing_duration.max()))
        if not losing_duration.empty
        else timedelta()
    )
    loser_holding_avg = (
        timedelta(minutes=round(losing_duration.mean()))
        if not losing_duration.empty
        else timedelta()
    )
    winstreak, loss_streak = calc_streak(results)

    return {
        "wins": len(winning_trades),
        "losses": len(losing_trades),
        "draws": len(draw_trades),
        "winrate": len(winning_trades) / len(results) if len(results) else 0.0,
        "holding_avg": holding_avg,
        "holding_avg_s": holding_avg.total_seconds(),
        "winner_holding_min": format_duration(winner_holding_min),
        "winner_holding_min_s": winner_holding_min.total_seconds(),
        "winner_holding_max": format_duration(winner_holding_max),
        "winner_holding_max_s": winner_holding_max.total_seconds(),
        "winner_holding_avg": format_duration(winner_holding_avg),
        "winner_holding_avg_s": winner_holding_avg.total_seconds(),
        "loser_holding_min": format_duration(loser_holding_min),
        "loser_holding_min_s": loser_holding_min.total_seconds(),
        "loser_holding_max": format_duration(loser_holding_max),
        "loser_holding_max_s": loser_holding_max.total_seconds(),
        "loser_holding_avg": format_duration(loser_holding_avg),
        "loser_holding_avg_s": loser_holding_avg.total_seconds(),
        "max_consecutive_wins": winstreak,
        "max_consecutive_losses": loss_streak,
    }
def generate_daily_stats(results: DataFrame) -> dict[str, Any]:
    """生成每日统计数据"""
    if len(results) == 0:
        return {
            "回测最佳日": 0,
            "回测最差日": 0,
            "回测最佳日绝对收益": 0,
            "回测最差日绝对收益": 0,
            "盈利天数": 0,
            "平局天数": 0,
            "亏损天数": 0,
            "每日收益列表": [],
        }
    daily_profit_rel = results.resample("1d", on="close_date")["profit_ratio"].sum()
    daily_profit = results.resample("1d", on="close_date")["profit_abs"].sum().round(10)
    worst_rel = min(daily_profit_rel)
    best_rel = max(daily_profit_rel)
    worst = min(daily_profit)
    best = max(daily_profit)
    winning_days = sum(daily_profit > 0)
    draw_days = sum(daily_profit == 0)
    losing_days = sum(daily_profit < 0)
    daily_profit_list = [(str(idx.date()), val) for idx, val in daily_profit.items()]

    return {
        "回测最佳日": best_rel,
        "回测最差日": worst_rel,
        "回测最佳日绝对收益": best,
        "回测最差日绝对收益": worst,
        "盈利天数": winning_days,
        "平局天数": draw_days,
        "亏损天数": losing_days,
        "每日收益列表": daily_profit_list,
    }
def generate_strategy_stats(
    pairlist: list[str],
    strategy: str,
    content: BacktestContentType,
    min_date: datetime,
    max_date: datetime,
    market_change: float,
    is_hyperopt: bool = False,
) -> dict[str, Any]:
    """
    :param pairlist: 要回测的交易对列表
    :param strategy: 策略名称
    :param content: 回测结果数据，格式为：{'results': 结果数据, 'config': 配置}。
    :param min_date: 回测开始日期
    :param max_date: 回测结束日期
    :param market_change: 表示市场变化的浮点数
    :return: 包含每个策略结果和策略摘要的字典。
    """
    results: DataFrame = content["results"]
    if not isinstance(results, DataFrame):
        return {}
    config = content["config"]
    max_open_trades = min(config["max_open_trades"], len(pairlist))
    start_balance = get_dry_run_wallet(config)
    stake_currency = config["stake_currency"]

    pair_results = generate_pair_metrics(
        pairlist,
        stake_currency=stake_currency,
        starting_balance=start_balance,
        results=results,
        min_date=min_date,
        max_date=max_date,
        skip_nan=False,
    )

    enter_tag_stats = generate_tag_metrics(
        "enter_tag",
        starting_balance=start_balance,
        results=results,
        min_date=min_date,
        max_date=max_date,
        skip_nan=False,
    )
    exit_reason_stats = generate_tag_metrics(
        "exit_reason",
        starting_balance=start_balance,
        results=results,
        min_date=min_date,
        max_date=max_date,
        skip_nan=False,
    )
    mix_tag_stats = generate_tag_metrics(
        ["enter_tag", "exit_reason"],
        starting_balance=start_balance,
        results=results,
        min_date=min_date,
        max_date=max_date,
        skip_nan=False,
    )
    left_open_results = generate_pair_metrics(
        pairlist,
        stake_currency=stake_currency,
        starting_balance=start_balance,
        results=results.loc[results["exit_reason"] == "force_exit"],
        min_date=min_date,
        max_date=max_date,
        skip_nan=True,
    )

    daily_stats = generate_daily_stats(results)
    trade_stats = generate_trading_stats(results)

    periodic_breakdown = {}
    if not is_hyperopt:
        periodic_breakdown = {"periodic_breakdown": generate_all_periodic_breakdown_stats(results)}

    best_pair = (
        max(
            [pair for pair in pair_results if pair["key"] != "TOTAL"], key=lambda x: x["profit_sum"]
        )
        if len(pair_results) > 1
        else None
    )
    worst_pair = (
        min(
            [pair for pair in pair_results if pair["key"] != "TOTAL"], key=lambda x: x["profit_sum"]
        )
        if len(pair_results) > 1
        else None
    )
    winning_profit = results.loc[results["profit_abs"] > 0, "profit_abs"].sum()
    losing_profit = results.loc[results["profit_abs"] < 0, "profit_abs"].sum()
    profit_factor = winning_profit / abs(losing_profit) if losing_profit else 0.0

    expectancy, expectancy_ratio = calculate_expectancy(results)
    backtest_days = (max_date - min_date).days or 1
    trades_dict = results.to_dict(orient="records")
    strat_stats = {
        "trades": trades_dict,
        "locks": [lock.to_json() for lock in content["locks"]],
        "best_pair": best_pair,
        "worst_pair": worst_pair,
        "results_per_pair": pair_results,
        "results_per_enter_tag": enter_tag_stats,
        "exit_reason_summary": exit_reason_stats,
        "mix_tag_stats": mix_tag_stats,
        "left_open_trades": left_open_results,
        "total_trades": len(results),
        "trade_count_long": len(results.loc[~results["is_short"]]),
        "trade_count_short": len(results.loc[results["is_short"]]),
        "total_volume": calculate_trade_volume(trades_dict),
        "avg_stake_amount": results["stake_amount"].mean() if len(results) > 0 else 0,
        "profit_mean": results["profit_ratio"].mean() if len(results) > 0 else 0,
        "profit_median": results["profit_ratio"].median() if len(results) > 0 else 0,
        "profit_total": results["profit_abs"].sum() / start_balance,
        "profit_total_long": results.loc[~results["is_short"], "profit_abs"].sum() / start_balance,
        "profit_total_short": results.loc[results["is_short"], "profit_abs"].sum() / start_balance,
        "profit_total_abs": results["profit_abs"].sum(),
        "profit_total_long_abs": results.loc[~results["is_short"], "profit_abs"].sum(),
        "profit_total_short_abs": results.loc[results["is_short"], "profit_abs"].sum(),
        "cagr": calculate_cagr(backtest_days, start_balance, content["final_balance"]),
        "expectancy": expectancy,
        "expectancy_ratio": expectancy_ratio,
        "sortino": calculate_sortino(results, min_date, max_date, start_balance),
        "sharpe": calculate_sharpe(results, min_date, max_date, start_balance),
        "calmar": calculate_calmar(results, min_date, max_date, start_balance),
        "sqn": calculate_sqn(results, start_balance),
        "profit_factor": profit_factor,
        "backtest_start": min_date.strftime(DATETIME_PRINT_FORMAT),
        "backtest_start_ts": int(min_date.timestamp() * 1000),
        "backtest_end": max_date.strftime(DATETIME_PRINT_FORMAT),
        "backtest_end_ts": int(max_date.timestamp() * 1000),
        "backtest_days": backtest_days,
        "backtest_run_start_ts": content["backtest_start_time"],
        "backtest_run_end_ts": content["backtest_end_time"],
        "trades_per_day": round(len(results) / backtest_days, 2),
        "market_change": market_change,
        "pairlist": pairlist,
        "stake_amount": config["stake_amount"],
        "stake_currency": config["stake_currency"],
        "stake_currency_decimals": decimals_per_coin(config["stake_currency"]),
        "starting_balance": start_balance,
        "dry_run_wallet": start_balance,
        "final_balance": content["final_balance"],
        "rejected_signals": content["rejected_signals"],
        "timedout_entry_orders": content["timedout_entry_orders"],
        "timedout_exit_orders": content["timedout_exit_orders"],
        "canceled_trade_entries": content["canceled_trade_entries"],
        "canceled_entry_orders": content["canceled_entry_orders"],
        "replaced_entry_orders": content["replaced_entry_orders"],
        "max_open_trades": max_open_trades,
        "max_open_trades_setting": (
            config["max_open_trades"] if config["max_open_trades"] != float("inf") else -1
        ),
        "timeframe": config["timeframe"],
        "timeframe_detail": config.get("timeframe_detail", ""),
        "timerange": config.get("timerange", ""),
        "enable_protections": config.get("enable_protections", False),
        "strategy_name": strategy,
        # Parameters relevant for backtesting
        "stoploss": config["stoploss"],
        "trailing_stop": config.get("trailing_stop", False),
        "trailing_stop_positive": config.get("trailing_stop_positive"),
        "trailing_stop_positive_offset": config.get("trailing_stop_positive_offset", 0.0),
        "trailing_only_offset_is_reached": config.get("trailing_only_offset_is_reached", False),
        "use_custom_stoploss": config.get("use_custom_stoploss", False),
        "minimal_roi": config["minimal_roi"],
        "use_exit_signal": config["use_exit_signal"],
        "exit_profit_only": config["exit_profit_only"],
        "exit_profit_offset": config["exit_profit_offset"],
        "ignore_roi_if_entry_signal": config["ignore_roi_if_entry_signal"],
        "trading_mode": config["trading_mode"],
        "margin_mode": config["margin_mode"],
        **periodic_breakdown,
        **daily_stats,
        **trade_stats,
    }

    try:
        drawdown = calculate_max_drawdown(
            results, value_col="profit_abs", starting_balance=start_balance
        )
        # max_relative_drawdown = Underwater
        underwater = calculate_max_drawdown(
            results, value_col="profit_abs", starting_balance=start_balance, relative=True
        )

        strat_stats.update(
            {
                "max_drawdown_account": drawdown.relative_account_drawdown,
                "max_relative_drawdown": underwater.relative_account_drawdown,
                "max_drawdown_abs": drawdown.drawdown_abs,
                "drawdown_start": drawdown.high_date.strftime(DATETIME_PRINT_FORMAT),
                "drawdown_start_ts": drawdown.high_date.timestamp() * 1000,
                "drawdown_end": drawdown.low_date.strftime(DATETIME_PRINT_FORMAT),
                "drawdown_end_ts": drawdown.low_date.timestamp() * 1000,
                "max_drawdown_low": drawdown.low_value,
                "max_drawdown_high": drawdown.high_value,
            }
        )

        csum_min, csum_max = calculate_csum(results, start_balance)
        strat_stats.update({"csum_min": csum_min, "csum_max": csum_max})

    except ValueError:
        strat_stats.update(
            {
                "max_drawdown_account": 0.0,
                "max_relative_drawdown": 0.0,
                "max_drawdown_abs": 0.0,
                "max_drawdown_low": 0.0,
                "max_drawdown_high": 0.0,
                "drawdown_start": datetime(1970, 1, 1, tzinfo=timezone.utc),
                "drawdown_start_ts": 0,
                "drawdown_end": datetime(1970, 1, 1, tzinfo=timezone.utc),
                "drawdown_end_ts": 0,
                "csum_min": 0,
                "csum_max": 0,
            }
        )

    return strat_stats
def generate_backtest_stats(
    btdata: dict[str, DataFrame],
    all_results: dict[str, BacktestContentType],
    min_date: datetime,
    max_date: datetime,
    notes: str | None = None,
) -> BacktestResultType:
    """
    :param btdata: 回测数据
    :param all_results: 回测结果 - 字典形式：{策略名：{'results': 结果, 'config': 配置}}。
    :param min_date: 回测开始日期
    :param max_date: 回测结束日期
    :return: 包含每个策略结果和策略摘要的字典。
    """
    result: BacktestResultType = get_BacktestResultType_default()
    market_change = calculate_market_change(btdata, "收盘价", min_date=min_date)
    metadata = {}
    pairlist = list(btdata.keys())
    for strategy, content in all_results.items():
        strat_stats = generate_strategy_stats(
            pairlist, strategy, content, min_date, max_date, market_change=market_change
        )
        metadata[strategy] = {
            "运行ID": content["run_id"],
            "回测开始时间": content["backtest_start_time"],
            "周期": content["config"]["timeframe"],
            "周期详情": content["config"].get("timeframe_detail", None),
            "回测开始时间戳": int(min_date.timestamp()),
            "回测结束时间戳": int(max_date.timestamp()),
        }
        if notes:
            metadata[strategy]["备注"] = notes
        result["strategy"][strategy] = strat_stats

    strategy_results = generate_strategy_comparison(bt_stats=result["strategy"])

    result["metadata"] = metadata
    result["strategy_comparison"] = strategy_results

    return result