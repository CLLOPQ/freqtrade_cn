# 交易对象 (Trade Object)

## 交易 (Trade)

Freqtrade 进入的仓位存储在 `Trade` 对象中 - 该对象会持久化到数据库。
这是 Freqtrade 的核心概念 - 你会在文档的许多部分遇到它，这些部分很可能会指向此处。

它会通过许多 [策略回调](strategy-callbacks.md) 传递给策略。传递给策略的对象不能直接修改。间接修改可能会根据回调结果发生。

## 交易 - 可用属性 (Trade - Available attributes)

每个交易都有以下属性/特性，可通过 `trade.<property>` 使用（例如 `trade.pair`）。

| 属性 (Attribute) | 数据类型 (DataType) | 描述 (Description) |
|------------|-------------|-------------|
| `pair` | string | 此交易的交易对。 |
| `is_open` | boolean | 交易当前是否为开仓状态，或已结束。 |
| `open_rate` | float | 交易的入场价格（若有交易调整，为平均入场价格）。 |
| `close_rate` | float | 平仓价格 - 仅在 is_open = False 时设置。 |
| `stake_amount` | float | 基础货币（或报价货币）的金额。 |
| `amount` | float | 当前持有的资产/基准货币数量。在初始订单成交前为 0.0。 |
| `open_date` | datetime | 交易开仓时间戳 **请改用 `open_date_utc`** |
| `open_date_utc` | datetime | 交易开仓时间戳 - UTC 时区。 |
| `close_date` | datetime | 交易平仓时间戳 **请改用 `close_date_utc`** |
| `close_date_utc` | datetime | 交易平仓时间戳 - UTC 时区。 |
| `close_profit` | float | 交易平仓时的相对利润。`0.01` 等于 1% |
| `close_profit_abs` | float | 交易平仓时的绝对利润（以基础货币计）。 |
| `realized_profit` | float | 交易仍为开仓状态时已实现的绝对利润（以基础货币计）。 |
| `leverage` | float | 此交易使用的杠杆 - 现货市场默认值为 1.0。 |
| `enter_tag` | string | 通过数据框中的 `enter_tag` 列在入场时提供的标签。 |
| `is_short` | boolean | 若为做空交易则为 True，否则为 False。 |
| `orders` | Order[] | 附加到此交易的订单对象列表（包括已成交和已取消的订单）。 |
| `date_last_filled_utc` | datetime | 最后一笔成交订单的时间。 |
| `entry_side` | "buy" / "sell" | 交易入场的订单方向。 |
| `exit_side` | "buy" / "sell" | 导致交易离场/仓位减少的订单方向。 |
| `trade_direction` | "long" / "short" | 交易方向文本 - 做多或做空。 |
| `nr_of_successful_entries` | int | 成功（已成交）的入场订单数量。 |
| `nr_of_successful_exits` | int | 成功（已成交）的离场订单数量。 |
| `has_open_orders` | boolean | 交易是否有未成交订单（不包括止损订单）。 |

## 类方法 (Class methods)

以下是类方法 - 返回通用信息，通常会对数据库执行显式查询。
可通过 `Trade.<method>` 调用 - 例如 `open_trades = Trade.get_open_trade_count()`

!!! Warning "回测/超参数优化 (Backtesting/hyperopt)"
    大多数方法在回测/超参数优化以及实盘/模拟交易模式下都能工作。
    在回测期间，仅限于在[策略回调](strategy-callbacks.md)中使用。在`populate_*()`方法中使用不受支持，会导致错误结果。

### get_trades_proxy

当策略需要现有（开仓或平仓）交易的某些信息时 - 最好使用 `Trade.get_trades_proxy()`。

用法示例：