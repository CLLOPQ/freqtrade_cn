## 保护机制

保护机制通过暂时停止单个交易对或所有交易对的交易，保护您的策略免受意外事件和市场条件的影响。
所有保护的结束时间都会向上取整到下一根K线，以避免在K线内突然、意外的买入。

!!! Tip "使用提示"
    并非所有保护机制都适用于所有策略，需要根据您的策略调整参数以提高性能。  

    每个保护机制可以使用不同的参数配置多次，以实现不同级别的保护（短期/长期）。

!!! Note "回测"
    回测和超参数优化支持保护机制，但必须使用 `--enable-protections` 标志显式启用。

### 可用的保护机制

* [`StoplossGuard`](#stoploss-guard) 如果在特定时间窗口内发生一定数量的止损，则停止交易。
* [`MaxDrawdown`](#maxdrawdown) 当达到最大回撤时停止交易。
* [`LowProfitPairs`](#low-profit-pairs) 锁定低利润交易对
* [`CooldownPeriod`](#cooldown-period) 卖出后不立即重新进入该交易对。

### 所有保护机制的通用设置

| 参数 | 描述 |
|------------|-------------|
| `method` | 要使用的保护机制名称。 <br> **数据类型：** 字符串，从[可用的保护机制](#available-protections)中选择
| `stop_duration_candles` | 锁定将持续多少根K线？ <br> **数据类型：** 正整数（以K线为单位）
| `stop_duration` | 保护机制应锁定多少分钟。 <br>不能与 `stop_duration_candles` 同时使用。 <br> **数据类型：** 浮点数（以分钟为单位）
| `lookback_period_candles` | 仅考虑在过去 `lookback_period_candles` 根K线内完成的交易。某些保护机制可能会忽略此设置。 <br> **数据类型：** 正整数（以K线为单位）。
| `lookback_period` | 仅考虑在 `current_time - lookback_period` 之后完成的交易。 <br>不能与 `lookback_period_candles` 同时使用。 <br>某些保护机制可能会忽略此设置。 <br> **数据类型：** 浮点数（以分钟为单位）
| `trade_limit` | 所需的最小交易数量（并非所有保护机制都使用）。 <br> **数据类型：** 正整数
| `unlock_at` | 定期解锁交易的时间（并非所有保护机制都使用）。 <br> **数据类型：** 字符串 <br>**输入格式：** "HH:MM"（24小时制）

!!! Note "持续时间"
    持续时间（`stop_duration*` 和 `lookback_period*`）可以以分钟或K线为单位定义。为了在测试不同时间框架时更灵活，以下所有示例将使用“K线”定义。

#### 止损防护（Stoploss Guard）

`StoplossGuard` 选择在 `lookback_period` 分钟内（或使用 `lookback_period_candles` 时以K线为单位）的所有交易。如果有 `trade_limit` 或更多交易触发了止损，则交易将停止 `stop_duration` 分钟（或使用 `stop_duration_candles` 时以K线为单位，或使用 `unlock_at` 时直到设定时间）。

这适用于所有交易对，除非 `only_per_pair` 设置为 true，此时将一次只查看一个交易对。

同样，此保护机制默认会查看所有交易（多单和空单）。对于期货机器人，设置 `only_per_side` 将使机器人只考虑一侧（多或空），然后仅锁定该侧，例如在一系列多单止损后允许空单继续。

`required_profit` 将确定要考虑的止损所需的相对利润（或亏损）。通常不应设置此参数，默认值为 0.0，这意味着所有亏损的止损都将触发阻止。

以下示例中，如果机器人在过去24根K线内触发了4次止损，则在最后一笔交易后，所有交易对将停止交易4根K线。