# 已弃用功能

本页面包含被机器人开发团队声明为已弃用（DEPRECATED）且不再支持的命令行参数、配置参数和机器人功能的说明。请避免在您的配置中使用这些功能。

## 已移除功能

### `--refresh-pairs-cached` 命令行选项

`--refresh-pairs-cached` 用于在回测、超参数优化（hyperopt）和Edge中刷新回测用的K线数据。由于该选项导致了较多混淆，且会减慢回测速度（同时并非回测的必要部分），现已被单独拆分为 `freqtrade download-data` 子命令。

此命令行选项在2019.7-dev（开发分支）中被弃用，并在2019.9版本中移除。

### `--dynamic-whitelist` 命令行选项

此命令行选项于2018年被弃用，并在freqtrade 2019.6-dev（开发分支）及2019.7版本中移除。请改用[交易对列表](plugins.md#pairlists-and-pairlist-handlers)。

### `--live` 命令行选项

`--live` 用于在回测时下载最新的tick数据。但该选项仅能下载最新的500根K线，无法获取高质量的回测数据。在2019-7-dev（开发分支）及freqtrade 2019.8版本中被移除。

### `ticker_interval`（现为 `timeframe`）

对 `ticker_interval` 术语的支持在2020.6版本中被弃用，转而使用 `timeframe`，相关兼容代码在2022.3版本中移除。

### 允许多个交易对列表按顺序运行

配置中先前的 `"pairlist"` 部分已被移除，取而代之的是 `"pairlists"`——一个用于指定交易对列表序列的列表。

旧的配置参数部分（`"pairlist"`）在2019.11版本中被弃用，并在2020.4版本中移除。

### 交易量交易对列表中 bidVolume 和 askVolume 的弃用

由于只有 quoteVolume 可在不同资产间比较，其他选项（bidVolume、askVolume）在2020.4版本中被弃用，并在2020.9版本中移除。

### 使用订单簿阶梯确定退出价格

过去可通过 `order_book_min` 和 `order_book_max` 来步进订单簿并尝试寻找下一个ROI区间，以尽早挂出卖出订单。但该功能会增加风险且无实际益处，为便于维护，在2021.7版本中被移除。

### 传统超参数优化模式

使用单独超参数优化文件的方式在2021.4版本中被弃用，并在2021.9版本中移除。请切换至新的[参数化策略](hyperopt.md)以使用新的超参数优化界面。

## V2与V3策略的变化

隔离期货/做空交易功能在2022.4版本中引入。这需要对配置设置、策略接口等进行重大变更。

我们已尽力保持与现有策略的兼容性，因此如果您只想继续在现货市场使用freqtrade，无需进行任何更改。尽管未来我们可能会放弃对当前接口的支持，但会单独发布公告并提供适当的过渡期。

请参考[策略迁移](strategy_migration.md)指南，将您的策略迁移至新格式以使用新功能。

### Web钩子 - 2022.4版本的变更

#### `buy_tag` 已重命名为 `enter_tag`

此变更仅适用于您的策略及可能的web钩子。我们将保留1-2个版本的兼容层（因此 `buy_tag` 和 `enter_tag` 仍可正常工作），但之后web钩子中将不再支持 `buy_tag`。

#### 命名变更

Web钩子术语从“sell”改为“exit”，从“buy”改为“entry”，同时移除了“webhook”前缀：

* `webhookbuy`、`webhookentry` -> `entry`
* `webhookbuyfill`、`webhookentryfill` -> `entry_fill`
* `webhookbuycancel`、`webhookentrycancel` -> `entry_cancel`
* `webhooksell`、`webhookexit` -> `exit`
* `webhooksellfill`、`webhookexitfill` -> `exit_fill`
* `webhooksellcancel`、`webhookexitcancel` -> `exit_cancel`

## `populate_any_indicators` 方法的移除

2023.3版本移除了 `populate_any_indicators` 方法，转而支持将特征工程和目标函数拆分为独立方法。请阅读[迁移文档](strategy_migration.md#freqai-strategy)了解详细信息。

## 配置中 `protections` 的移除

通过 `"protections": [],` 在配置中设置保护机制的方式在2024.10版本中移除，此前已发出超过3年的弃用警告。

## HDF5数据存储

使用HDF5作为数据存储格式在2024.12版本中被弃用，并在2025.1版本中移除。建议切换至Feather数据格式。

请在更新前使用[`convert-data`子命令](data-download.md#sub-command-convert-data)将现有数据转换为支持的格式之一。

## 通过配置文件配置高级日志

通过 `--logfile systemd` 和 `--logfile journald` 分别配置syslog和journald的方式在2025.3版本中被弃用。请改用基于配置的[日志设置](advanced-setup.md#advanced-logging)。

## Edge模块的移除

Edge模块在2023.9版本中被弃用，并在2025.6版本中移除。Edge的所有功能均已删除，配置Edge将导致错误。