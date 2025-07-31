目前，参数如下：

* `results`：包含结果交易的DataFrame。
    results中可用的列如下（对应回测时使用 `--export trades` 导出的文件）：  
    `pair, profit_ratio, profit_abs, open_date, open_rate, fee_open, close_date, close_rate, fee_close, amount, trade_duration, is_open, exit_reason, stake_amount, min_rate, max_rate, stop_loss_ratio, stop_loss_abs`
* `trade_count`：交易数量（与 `len(results)` 相同）
* `min_date`：所用时间范围的开始日期
* `max_date`：所用时间范围的结束日期
* `config`：使用的配置对象（注意：如果某些策略相关参数是超参数优化空间的一部分，此处可能不会更新）。
* `processed`：DataFrame字典，以交易对为键，包含用于回测的数据。
* `backtest_stats`：回测统计数据，格式与回测文件的 "strategy" 子结构相同。可用字段可在 `optimize_reports.py` 中的 `generate_strategy_stats()` 中查看。
* `starting_balance`：用于回测的初始资金。

此函数需要返回一个浮点数（`float`）。较小的数值表示更好的结果。参数和权重平衡由您决定。

!!! Note
    此函数每个epoch调用一次 - 因此请确保尽可能优化此函数，以免不必要地减慢超参数优化速度。

!!! Note "`*args` 和 `**kwargs`"
    请在接口中保留 `*args` 和 `**kwargs` 参数，以便我们将来扩展此接口。

## 覆盖预定义空间

要覆盖预定义空间（`roi_space`、`generate_roi_table`、`stoploss_space`、`trailing_space`、`max_open_trades_space`），请定义一个名为 `HyperOpt` 的嵌套类，并按如下方式定义所需空间：