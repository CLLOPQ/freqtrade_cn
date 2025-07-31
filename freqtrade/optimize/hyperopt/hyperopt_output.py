import sys
from os import get_terminal_size
from typing import Any

from rich.align import Align
from rich.console import Console
from rich.table import Table
from rich.text import Text

from freqtrade.constants import Config
from freqtrade.optimize.optimize_reports import generate_wins_draws_losses
from freqtrade.util import fmt_coin


class HyperoptOutput:
    def __init__(self, streaming=False) -> None:
        self._results: list[Any] = []  # 存储结果列表
        self._streaming = streaming  # 是否流式输出
        self.__init_table()  # 初始化表格

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return Align.center(self.table)  # 居中对齐表格

    def __init_table(self) -> None:
        """初始化表格"""
        self.table = Table(
            title="超参数优化结果",
        )
        # 表头
        self.table.add_column("最佳", justify="left")
        self.table.add_column("轮次", justify="right")
        self.table.add_column("交易数", justify="right")
        self.table.add_column("胜  平  负  胜率", justify="right")
        self.table.add_column("平均利润", justify="right")
        self.table.add_column("总利润", justify="right")
        self.table.add_column("平均持仓时间", justify="right")
        self.table.add_column("目标值", justify="right")
        self.table.add_column("最大回撤(账户)", justify="right")

    def print(self, console: Console | None = None, *, print_colorized=True):
        if not console:
            console = Console(
                color_system="auto" if print_colorized else None,
                width=200 if "pytest" in sys.modules else None,
            )

        console.print(self.table)  # 打印表格

    def add_data(
        self,
        config: Config,
        results: list,
        total_epochs: int,
        highlight_best: bool,
    ) -> None:
        """格式化一行或多行数据并添加到表格中"""
        stake_currency = config["stake_currency"]  # 赌注货币
        self._results.extend(results)  # 扩展结果列表

        max_rows: int | None = None

        if self._streaming:
            try:
                ts = get_terminal_size()
                # 获取终端大小
                # 考虑表头、边框和进度条
                # 假设行不会换行
                if ts.columns < 148:
                    # 如果终端太小，无法正确显示表格
                    # 我们将显示的行数减半
                    max_rows = -(int(ts.lines / 2) - 6)
                else:
                    max_rows = -(ts.lines - 6)
            except OSError:
                # 如果无法获取终端大小，只显示最后10行
                pass

        self.__init_table()  # 重新初始化表格
        for r in self._results[max_rows:]:
            self.table.add_row(
                *[
                    # "最佳":
                    (
                        ("*" if r["is_initial_point"] or r["is_random"] else "")
                        + (" 最佳" if r["is_best"] else "")
                    ).lstrip(),
                    # "轮次":
                    f"{r['current_epoch']}/{total_epochs}",
                    # "交易数":
                    str(r["results_metrics"]["total_trades"]),
                    # "胜  平  负  胜率":
                    generate_wins_draws_losses(
                        r["results_metrics"]["wins"],
                        r["results_metrics"]["draws"],
                        r["results_metrics"]["losses"],
                    ),
                    # "平均利润":
                    f"{r['results_metrics']['profit_mean']:.2%}"
                    if r["results_metrics"]["profit_mean"] is not None
                    else "--",
                    # "总利润":
                    Text(
                        "{} {}".format(
                            fmt_coin(
                                r["results_metrics"]["profit_total_abs"],
                                stake_currency,
                                keep_trailing_zeros=True,
                            ),
                            f"({r['results_metrics']['profit_total']:,.2%})".rjust(10, " "),
                        )
                        if r["results_metrics"].get("profit_total_abs", 0) != 0.0
                        else "--",
                        style=(
                            "green"
                            if r["results_metrics"].get("profit_total_abs", 0) > 0
                            else "red"
                        )
                        if not r["is_best"]
                        else "",
                    ),
                    # "平均持仓时间":
                    str(r["results_metrics"]["holding_avg"]),
                    # "目标值":
                    f"{r['loss']:,.5f}" if r["loss"] != 100000 else "N/A",
                    # "最大回撤(账户)":
                    "{} {}".format(
                        fmt_coin(
                            r["results_metrics"]["max_drawdown_abs"],
                            stake_currency,
                            keep_trailing_zeros=True,
                        ),
                        (f"({r['results_metrics']['max_drawdown_account']:,.2%})").rjust(10, " "),
                    )
                    if r["results_metrics"]["max_drawdown_account"] != 0.0
                    else "--",
                ],
                style=" ".join(
                    [
                        "bold gold1" if r["is_best"] and highlight_best else "",
                        "italic " if r["is_initial_point"] else "",
                    ]
                ),
            )