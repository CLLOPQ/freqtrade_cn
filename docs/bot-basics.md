# Freqtrade 基础知识

本页为您提供有关Freqtrade工作原理和操作的一些基本概念。

## Freqtrade 术语表

* **策略 (Strategy)**: 您的交易策略，告诉机器人该做什么。
* **交易 (Trade)**: 未平仓头寸。
* **未完成订单 (Open Order)**: 当前已在交易所下达但尚未完成的订单。
* **交易对 (Pair)**: 可交易的货币对，通常格式为基础货币/报价货币（例如现货的`XRP/USDT`，期货的`XRP/USDT:USDT`）。
* **时间周期 (Timeframe)**: 使用的K线周期（例如`"5m"`表示5分钟，`"1h"`表示1小时，...）。
* **指标 (Indicators)**: 技术指标（SMA、EMA、RSI等）。
* **限价订单 (Limit order)**: 以设定的限价或更优价格执行的订单。
* **市价订单 (Market order)**: 保证成交的订单，价格可能因订单大小而变动。
* **当前利润 (Current Profit)**: 该交易当前未实现的浮动利润。主要用于机器人内部和用户界面。
* **已实现利润 (Realized Profit)**: 已实现的利润。仅在结合[部分平仓](strategy-callbacks.md#adjust-trade-position)时相关——该文档也解释了此利润的计算逻辑。
* **总利润 (Total Profit)**: 已实现利润和未实现利润的总和。相对值（%）是相对于该交易的总投资计算的。

## 费用处理

Freqtrade的所有利润计算均包含费用。在回测/超参数优化/模拟交易模式下，使用交易所默认费用（交易所最低层级费率）。在实盘操作中，使用交易所实际收取的费用（包括BNB折扣等）。

## 交易对命名

Freqtrade遵循[ccxt命名约定](https://docs.ccxt.com/#/README?id=consistency-of-base-and-quote-currencies)来命名货币对。
在错误的市场中使用错误的命名约定通常会导致机器人无法识别交易对，通常会产生诸如“该交易对不可用”之类的错误。

### 现货交易对命名

现货交易对的命名格式为`基础货币/报价货币`（例如`ETH/USDT`）。

### 期货交易对命名

期货交易对的命名格式为`基础货币/报价货币:结算货币`（例如`ETH/USDT:USDT`）。

## 机器人执行逻辑

以模拟交易或实盘模式启动Freqtrade（使用`freqtrade trade`命令）将启动机器人并开始机器人迭代循环。
这也会运行`bot_start()`回调函数。

默认情况下，机器人循环每几秒运行一次（`internals.process_throttle_secs`），并执行以下操作：

* 从持久化存储中获取未平仓交易。
* 计算当前可交易对列表。
* 下载交易对列表的OHLCV数据，包括所有[信息性交易对](strategy-customization.md#get-data-for-non-tradeable-pairs)。  
  此步骤每个K线只执行一次，以避免不必要的网络流量。
* 调用`bot_loop_start()`策略回调函数。
* 按交易对分析策略。
  * 调用`populate_indicators()`
  * 调用`populate_entry_trend()`
  * 调用`populate_exit_trend()`
* 从交易所更新交易的未完成订单状态。
  * 为已成交订单调用`order_filled()`策略回调函数。
  * 检查未完成订单的超时情况。
    * 为未完成的入场订单调用`check_entry_timeout()`策略回调函数。
    * 为未完成的出场订单调用`check_exit_timeout()`策略回调函数。
    * 为未完成订单调用`adjust_order_price()`策略回调函数。
      * 当未实现`adjust_order_price()`时，为未完成的入场订单调用`adjust_entry_price()`策略回调函数。
      * 当未实现`adjust_order_price()`时，为未完成的出场订单调用`adjust_exit_price()`策略回调函数。
* 验证现有持仓并最终下达平仓订单。
  * 考虑止损、ROI和出场信号、`custom_exit()`及`custom_stoploss()`。
  * 根据`exit_pricing`配置设置或使用`custom_exit_price()`回调函数确定平仓价格。
  * 在下达平仓订单前，会调用`confirm_trade_exit()`策略回调函数。
* 如果启用，通过调用`adjust_trade_position()`检查未平仓交易的仓位调整，并在需要时下达额外订单。
* 检查交易槽是否仍可用（是否已达到`max_open_trades`限制）。
* 验证入场信号，尝试建立新仓位。
  * 根据`entry_pricing`配置设置或使用`custom_entry_price()`回调函数确定入场价格。
  * 在保证金和期货模式下，调用`leverage()`策略回调函数以确定所需杠杆。
  * 通过调用`custom_stake_amount()`回调函数确定持仓金额。
  * 在下达入场订单前，会调用`confirm_trade_entry()`策略回调函数。

此循环将反复进行，直到机器人停止。

## 回测/超参数优化执行逻辑

[回测](backtesting.md)或[超参数优化](hyperopt.md)仅执行上述部分逻辑，因为大多数交易操作是完全模拟的。

* 加载配置的交易对列表的历史数据。
* 调用一次`bot_start()`。
* 计算指标（每个交易对调用一次`populate_indicators()`）。
* 计算入场/出场信号（每个交易对调用一次`populate_entry_trend()`和`populate_exit_trend()`）。
* 按K线循环模拟入场和出场点。
  * 调用`bot_loop_start()`策略回调函数。
  * 检查订单超时，通过`unfilledtimeout`配置或`check_entry_timeout()`/`check_exit_timeout()`策略回调函数。
  * 为未完成订单调用`adjust_order_price()`策略回调函数。
    * 当未实现`adjust_order_price()`时，为未完成的入场订单调用`adjust_entry_price()`策略回调函数！
    * 当未实现`adjust_order_price()`时，为未完成的出场订单调用`adjust_exit_price()`策略回调函数！
  * 检查交易入场信号（`enter_long`/`enter_short`列）。
  * 确认交易入场/出场（如果策略中实现了`confirm_trade_entry()`和`confirm_trade_exit()`，则调用它们）。
  * 调用`custom_entry_price()`（如果策略中实现）以确定入场价格（价格会调整到开盘K线范围内）。
  * 在保证金和期货模式下，调用`leverage()`策略回调函数以确定所需杠杆。
  * 通过调用`custom_stake_amount()`回调函数确定持仓金额。
  * 如果启用，检查未平仓交易的仓位调整，并调用`adjust_trade_position()`以确定是否需要额外订单。
  * 为已成交的入场订单调用`order_filled()`策略回调函数。
  * 调用`custom_stoploss()`和`custom_exit()`以找到自定义出场点。
  * 对于基于出场信号、自定义出场和部分出场的情况：调用`custom_exit_price()`确定出场价格（价格会调整到收盘K线范围内）。
  * 为已成交的出场订单调用`order_filled()`策略回调函数。
* 生成回测报告输出

!!! Note
    回测和超参数优化的计算均包含交易所默认费用。可以通过指定`--fee`参数将自定义费用传递给回测/超参数优化。

!!! Warning "回调函数调用频率"
    回测最多每个K线调用一次每个回调函数（`--timeframe-detail`参数将此行为修改为每个详细K线调用一次）。实盘模式下，大多数回调函数每次迭代调用一次（通常每~5秒）——这可能导致回测不匹配。