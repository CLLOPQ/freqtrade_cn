# 配置机器人

Freqtrade 有许多可配置的功能和可能性。
默认情况下，这些设置通过配置文件进行配置（见下文）。

## Freqtrade 配置文件

机器人在运行过程中使用一组配置参数，这些参数共同构成了机器人配置。它通常从一个文件（Freqtrade 配置文件）中读取配置。

默认情况下，机器人从当前工作目录中的 `config.json` 文件加载配置。

您可以使用 `-c/--config` 命令行选项指定机器人使用的不同配置文件。

如果您使用 [快速入门](docker_quickstart.md#docker-quick-start) 方法安装机器人，安装脚本应该已经为您创建了默认配置文件（`config.json`）。

如果未创建默认配置文件，我们建议使用 `freqtrade new-config --config user_data/config.json` 生成基本配置文件。

Freqtrade 配置文件采用 JSON 格式编写。

除了标准 JSON 语法外，您还可以在配置文件中使用单行 `// ...` 和多行 `/* ... */` 注释，以及参数列表中的尾随逗号。

如果您不熟悉 JSON 格式，不必担心——只需使用您选择的编辑器打开配置文件，对所需参数进行一些更改，保存更改，最后重新启动机器人，或者如果机器人之前已停止，则使用对配置所做的更改再次运行它。机器人在启动时会验证配置文件的语法，如果您在编辑时出错，会警告您并指出有问题的行。

### 环境变量

通过环境变量设置 Freqtrade 配置中的选项。
这优先于配置或策略中的相应值。

环境变量必须以 `FREQTRADE__` 为前缀才能加载到 freqtrade 配置中。

`__` 用作级别分隔符，因此使用的格式应对应于 `FREQTRADE__{section}__{key}`。
因此，定义为 `export FREQTRADE__STAKE_AMOUNT=200` 的环境变量将导致 `{stake_amount: 200}`。

一个更复杂的例子可能是 `export FREQTRADE__EXCHANGE__KEY=<yourExchangeKey>` 来保护您的交易所密钥。这会将值移动到配置的 `exchange.key` 部分。
使用此方案，所有配置设置也将作为环境变量可用。

请注意，环境变量将覆盖配置中的相应设置，但命令行参数将始终优先。

常见示例：

``` bash
FREQTRADE__TELEGRAM__CHAT_ID=<telegramchatid>
FREQTRADE__TELEGRAM__TOKEN=<telegramToken>
FREQTRADE__EXCHANGE__KEY=<yourExchangeKey>
FREQTRADE__EXCHANGE__SECRET=<yourExchangeSecret>
```

Json 列表被解析为 json - 因此您可以使用以下命令设置交易对列表：

``` bash
export FREQTRADE__EXCHANGE__PAIR_WHITELIST='["BTC/USDT", "ETH/USDT"]'
```

!!! 注意
    检测到的环境变量会在启动时记录 - 因此如果您无法理解为什么某个值与基于配置的预期不符，请确保它不是从环境变量加载的。

!!! 提示 "验证组合结果"
    您可以使用 [show-config 子命令](utils.md#show-config) 查看最终的组合配置。

??? 警告 "加载顺序"
    环境变量在初始配置之后加载。因此，您不能通过环境变量提供配置的路径。请使用 `--config path/to/config.json`。
    这在某种程度上也适用于 `user_dir`。虽然用户目录可以通过环境变量设置 - 但配置**不会**从该位置加载。

### 多个配置文件

机器人可以指定和使用多个配置文件，或者机器人可以从进程标准输入流读取其配置参数。

您可以在 `add_config_files` 中指定其他配置文件。此参数中指定的文件将被加载并与初始配置文件合并。文件相对于初始配置文件解析。
这类似于使用多个 `--config` 参数，但使用更简单，因为您不必为所有命令指定所有文件。

!!! 提示 "验证组合结果"
    您可以使用 [show-config 子命令](utils.md#show-config) 查看最终的组合配置。

!!! 提示 "使用多个配置文件来保密"
    您可以使用第二个包含您的秘密的配置文件。这样您就可以共享您的“主要”配置文件，同时仍将您的 API 密钥留给自己。
    第二个文件应仅指定您打算覆盖的内容。
    如果一个键同时存在于 `config.json` 和 `config-import.json` 中，那么“最后指定的配置”将获胜（在上面的示例中，`config-private.json`）。

    对于一次性命令，您也可以通过指定多个“--config”参数来使用以下语法。

    ``` bash
    freqtrade trade --config user_data/config1.json --config user_data/config-private.json <...>
    ```

    下面的示例等同于上面的示例 - 但在配置中有 2 个配置文件，便于重用。

    ``` json title="user_data/config.json"
    "add_config_files": [
        "config1.json",
        "config-private.json"
    ]
    ```

    ``` bash
    freqtrade trade --config user_data/config.json <...>
    ```

??? 注意 "配置冲突处理"
    如果相同的配置设置同时出现在 `config.json` 和 `config-import.json` 中，则父配置获胜。
    在下面的情况下，合并后 `max_open_trades` 将为 3 - 因为可重用的“导入”配置已覆盖此键。

    ``` json title="user_data/config.json"
    {
        "max_open_trades": 3,
        "stake_currency": "USDT",
        "add_config_files": [
            "config-import.json"
        ]
    }
    ```

    ``` json title="user_data/config-import.json"
    {
        "max_open_trades": 10,
        "stake_amount": "unlimited",
    }
    ```

    生成的组合配置：

    ``` json title="结果"
    {
        "max_open_trades": 3,
        "stake_currency": "USDT",
        "stake_amount": "unlimited"
    }
    ```

    如果 `add_config_files` 部分中有多个文件，则假定它们处于相同级别，最后出现的文件将覆盖较早的配置（除非父级已定义此类键）。

## 编辑器自动完成和验证

如果您使用的编辑器支持 JSON 模式，您可以通过在配置文件顶部添加以下行，使用 Freqtrade 提供的模式来获得配置文件的自动完成和验证：

``` json
{
    "$schema": "https://schema.freqtrade.io/schema.json",
}
```

??? 注意 "开发版本"
    开发模式的架构可用作 `https://schema.freqtrade.io/schema_dev.json` - 不过我们建议为获得最佳体验而坚持使用稳定版本。

## 配置参数

下表将列出所有可用的配置参数。

Freqtrade 还可以通过命令行 (CLI) 参数加载许多选项（查看命令 `--help` 输出了解详情）。

### 配置选项优先级

所有选项的优先级如下：

* CLI 参数覆盖任何其他选项
* [环境变量](#环境变量)
* 配置文件按顺序使用（最后一个文件获胜）并覆盖策略配置。
* 策略配置仅在未通过配置或命令行参数设置时使用。这些选项在下表中标记为 [策略覆盖](#parameters-in-the-strategy)。

### 参数表

必填参数标记为 **必填**，这意味着需要以其中一种可能的方式设置它们。

|  参数 | 描述 |
|------------|-------------|
| `max_open_trades` | **必填。** 您的机器人允许持有的未平仓交易数量。每个交易对只能有一个未平仓交易，因此您的交易对列表的长度是另一个可能适用的限制。如果为 -1 则忽略（即潜在的无限未平仓交易，受交易对列表限制）。[更多信息如下](#configuring-amount-per-trade)。[策略覆盖](#parameters-in-the-strategy)。<br> **数据类型：** 正整数或 -1。
| `stake_currency` | **必填。** 用于交易的加密货币。 <br> **数据类型：** 字符串
| `stake_amount` | **必填。** 您的机器人每笔交易将使用的加密货币数量。将其设置为 `"unlimited"` 以允许机器人使用所有可用余额。[更多信息如下](#configuring-amount-per-trade)。 <br> **数据类型：** 正浮点数或 `"unlimited"`。
| `tradable_balance_ratio` | 机器人允许交易的账户总余额的比例。[更多信息如下](#configuring-amount-per-trade)。 <br>*默认为 `0.99` 99%）。*<br> **数据类型：** 0.1 到 1.0 之间的正浮点数。
| `available_capital` | 机器人可用的初始资金。在同一交易所账户上运行多个机器人时很有用。[更多信息如下](#configuring-amount-per-trade)。 <br> **数据类型：** 正浮点数。
| `amend_last_stake_amount` | 必要时使用减少的最后持仓金额。[更多信息如下](#configuring-amount-per-trade)。 <br>*默认为 `false`。* <br> **数据类型：** 布尔值
| `last_stake_amount_min_ratio` | 定义必须留下并执行的最小持仓金额。仅适用于最后持仓金额被修改为减少的值时（即如果 `amend_last_stake_amount` 设置为 `true`）。[更多信息如下](#configuring-amount-per-trade)。 <br>*默认为 `0.5`。* <br> **数据类型：** 浮点数（作为比率）
| `amount_reserve_percent` | 保留一些金额作为最小交易对持仓金额。机器人在计算最小交易对持仓金额时将保留 `amount_reserve_percent` + 止损值，以避免可能的交易拒绝。 <br>*默认为 `0.05`（5%）。* <br> **数据类型：** 正浮点数（作为比率）。
| `timeframe` | 要使用的时间周期（例如 `1m`、`5m`、`15m`、`30m`、`1h` ...）。通常在配置中缺失，在策略中指定。[策略覆盖](#parameters-in-the-strategy)。 <br> **数据类型：** 字符串
| `fiat_display_currency` | 用于显示利润的法定货币。[更多信息如下](#what-values-can-be-used-for-fiat_display_currency)。 <br> **数据类型：** 字符串
| `dry_run` | **必填。** 定义机器人必须处于模拟交易模式还是生产模式。 <br>*默认为 `true`。* <br> **数据类型：** 布尔值
| `dry_run_wallet` | 定义运行在模拟交易模式下的机器人使用的模拟钱包中的初始持仓货币金额。[更多信息如下](#dry-run-wallet)<br>*默认为 `1000`。* <br> **数据类型：** 浮点数或字典
| `cancel_open_orders_on_exit` | 当发出 `/stop` RPC 命令、按下 `Ctrl+C` 或机器人意外死亡时，取消未成交订单。当设置为 `true` 时，这允许您在市场崩溃时使用 `/stop` 取消未成交和部分成交的订单。它不影响未平仓头寸。 <br>*默认为 `false`。* <br> **数据类型：** 布尔值
| `process_only_new_candles` | 仅在新蜡烛到达时启用指标处理。如果为 false，每个循环都会填充指标，这意味着同一根蜡烛会被多次处理，造成系统负载，但如果您的策略依赖于 tick 数据而不仅仅是蜡烛，这可能很有用。[策略覆盖](#parameters-in-the-strategy)。 <br>*默认为 `true`。*  <br> **数据类型：** 布尔值
| `minimal_roi` | **必填。** 设置机器人将用于退出交易的阈值（作为比率）。[更多信息如下](#understand-minimal_roi)。[策略覆盖](#parameters-in-the-strategy)。 <br> **数据类型：** 字典
| `stoploss` |  **必填。** 机器人使用的止损值（作为比率）。更多细节在 [止损文档](stoploss.md) 中。[策略覆盖](#parameters-in-the-strategy)。  <br> **数据类型：** 浮点数（作为比率）
| `trailing_stop` | 启用追踪止损（基于配置或策略文件中的 `stoploss`）。更多细节在 [止损文档](stoploss.md#trailing-stop-loss) 中。[策略覆盖](#parameters-in-the-strategy)。 <br> **数据类型：** 布尔值
| `trailing_stop_positive` | 一旦达到利润，更改止损。更多细节在 [止损文档](stoploss.md#trailing-stop-loss-different-positive-loss) 中。[策略覆盖](#parameters-in-the-strategy)。 <br> **数据类型：** 浮点数
| `trailing_stop_positive_offset` | 应用 `trailing_stop_positive` 的偏移量。应为正数的百分比值。更多细节在 [止损文档](stoploss.md#trailing-stop-loss-only-once-the-trade-has-reached-a-certain-offset) 中。[策略覆盖](#parameters-in-the-strategy)。 <br>*默认为 `0.0`（无偏移）。* <br> **数据类型：** 浮点数
| `trailing_only_offset_is_reached` | 仅当达到偏移量时才应用追踪止损。[止损文档](stoploss.md)。[策略覆盖](#parameters-in-the-strategy)。 <br>*默认为 `false`。*  <br> **数据类型：** 布尔值
| `fee` | 回测/模拟交易期间使用的费用。通常不应配置，这会使 freqtrade 回退到交易所默认费用。设置为比率（例如 0.001 = 0.1%）。每笔交易收取两次费用，一次买入时，一次卖出时。 <br> **数据类型：** 浮点数（作为比率）
| `futures_funding_rate` | 当交易所没有历史资金费率时使用的用户指定资金费率。这不覆盖真实的历史费率。建议将其设置为 0，除非您正在测试特定硬币并且您了解资金费率将如何影响 freqtrade 的利润计算。[更多信息在这里](leverage.md#unavailable-funding-rates) <br>*默认为 `None`。*<br> **数据类型：** 浮点数
| `trading_mode` | 指定您是否要进行常规交易、杠杆交易或交易价格源自匹配加密货币价格的合约。[杠杆文档](leverage.md)。 <br>*默认为 `"spot"`。*  <br> **数据类型：** 字符串
| `margin_mode` | 进行杠杆交易时，这决定了交易者拥有的抵押品是共享的还是隔离到每个交易对的 [杠杆文档](leverage.md)。 <br> **数据类型：** 字符串
| `liquidation_buffer` | 一个比率，指定在清算价格和止损之间放置多大的安全网，以防止头寸达到清算价格 [杠杆文档](leverage.md)。 <br>*默认为 `0.05`。*  <br> **数据类型：** 浮点数
| | **未成交超时**
| `unfilledtimeout.entry` | **必填。** 机器人将等待未成交的入场订单完成多长时间（以分钟或秒为单位），之后订单将被取消。[策略覆盖](#parameters-in-the-strategy)。<br> **数据类型：** 整数
| `unfilledtimeout.exit` | **必填。** 机器人将等待未成交的出场订单完成多长时间（以分钟或秒为单位），之后订单将被取消并以当前（新）价格重复，只要有信号。[策略覆盖](#parameters-in-the-strategy)。<br> **数据类型：** 整数
| `unfilledtimeout.unit` | 未成交超时设置中使用的单位。注意：如果您将 unfilledtimeout.unit 设置为 "seconds"，"internals.process_throttle_secs" 必须小于或等于超时 [策略覆盖](#parameters-in-the-strategy)。 <br> *默认为 `"minutes"`。* <br> **数据类型：** 字符串
| `unfilledtimeout.exit_timeout_count` | 出场订单可以超时多少次。一旦达到此超时次数，将触发紧急出场。0 表示禁用并允许无限订单取消。[策略覆盖](#parameters-in-the-strategy)。<br>*默认为 `0`。* <br> **数据类型：** 整数
| | **定价**
| `entry_pricing.price_side` | 选择机器人应查看的点差一侧以获取入场价格。[更多信息如下](#entry-price)。<br> *默认为 `"same"`。* <br> **数据类型：** 字符串（`ask`、`bid`、`same` 或 `other`）。
| `entry_pricing.price_last_balance` | **必填。** 插值买入价格。更多信息 [如下](#entry-price-without-orderbook-enabled)。
| `entry_pricing.use_order_book` | 启用使用 [订单簿入场](#entry-price-with-orderbook-enabled) 中的价格入场。 <br> *默认为 `true`。*<br> **数据类型：** 布尔值
| `entry_pricing.order_book_top` | 机器人将使用订单簿 "price_side" 中的前 N 个价格进入交易。即值为 2 将允许机器人在 [订单簿入场](#entry-price-with-orderbook-enabled) 中选择第 2 个条目。 <br>*默认为 `1`。*  <br> **数据类型：** 正整数
| `entry_pricing. check_depth_of_market.enabled` | 如果在订单簿中满足买入订单和卖出订单的差异，则不入场。[检查市场深度](#check-depth-of-market)。 <br>*默认为 `false`。* <br> **数据类型：** 布尔值
| `entry_pricing. check_depth_of_market.bids_to_ask_delta` | 在订单簿中找到的买入订单和卖出订单的差异比率。小于 1 的值意味着卖出订单规模更大，而大于 1 的值意味着买入订单规模更大。[检查市场深度](#check-depth-of-market) <br> *默认为 `0`。*  <br> **数据类型：** 浮点数（作为比率）
| `exit_pricing.price_side` | 选择机器人应查看的点差一侧以获取出场价格。[更多信息如下](#exit-price-side)。<br> *默认为 `"same"`。* <br> **数据类型：** 字符串（`ask`、`bid`、`same` 或 `other`）。
| `exit_pricing.price_last_balance` | 插值出场价格。更多信息 [如下](#exit-price-without-orderbook-enabled)。
| `exit_pricing.use_order_book` | 启用使用 [订单簿出场](#exit-price-with-orderbook-enabled) 平仓。 <br> *默认为 `true`。*<br> **数据类型：** 布尔值
| `exit_pricing.order_book_top` | 机器人将使用订单簿 "price_side" 中的前 N 个价格出场。即值为 2 将允许机器人在 [订单簿出场](#exit-price-with-orderbook-enabled) 中选择第 2 个要价。<br>*默认为 `1`。* <br> **数据类型：** 正整数
| `custom_price_max_distance_ratio` | 配置当前价格与自定义入场或出场价格之间的最大距离比率。 <br>*默认为 `0.02` 2%）。*<br> **数据类型：** 正浮点数
| | **订单/信号处理**
| `use_exit_signal` | 除了 `minimal_roi` 之外，使用策略产生的出场信号。 <br>将此设置为 false 将禁用 `"exit_long"` 和 `"exit_short"` 列的使用。对其他出场方法（止损、ROI、回调）没有影响。[策略覆盖](#parameters-in-the-strategy)。 <br>*默认为 `true`。* <br> **数据类型：** 布尔值
| `exit_profit_only` | 等待机器人达到 `exit_profit_offset` 后再做出场决定。[策略覆盖](#parameters-in-the-strategy)。 <br>*默认为 `false`。* <br> **数据类型：** 布尔值
| `exit_profit_offset` | 出场信号仅在该值以上有效。仅在 `exit_profit_only=True` 时有效。[策略覆盖](#parameters-in-the-strategy)。 <br>*默认为 `0.0`。* <br> **数据类型：** 浮点数（作为比率）
| `ignore_roi_if_entry_signal` | 如果入场信号仍然有效，则不出场。此设置优先于 `minimal_roi` 和 `use_exit_signal`。[策略覆盖](#parameters-in-the-strategy)。 <br>*默认为 `false`。* <br> **数据类型：** 布尔值
| `ignore_buying_expired_candle_after` | 指定买入信号不再使用的秒数。 <br> **数据类型：** 整数
| `order_types` | 根据操作（`"entry"`、`"exit"`、`"stoploss"`、`"stoploss_on_exchange"`）配置订单类型。[更多信息如下](#understand-order_types)。[策略覆盖](#parameters-in-the-strategy)。<br> **数据类型：** 字典
| `order_time_in_force` | 配置入场和出场订单的有效时间。[更多信息如下](#understand-order_time_in_force)。[策略覆盖](#parameters-in-the-strategy)。 <br> **数据类型：** 字典
| `position_adjustment_enable` | 使策略能够使用头寸调整（额外买入或卖出）。[更多信息在这里](strategy-callbacks.md#adjust-trade-position)。 <br> [策略覆盖](#parameters-in-the-strategy)。 <br>*默认为 `false`。*<br> **数据类型：** 布尔值
| `max_entry_position_adjustment` | 每个未平仓交易在第一个入场订单之上的最大额外订单数。将其设置为 `-1` 表示无限额外订单。[更多信息在这里](strategy-callbacks.md#adjust-trade-position)。 <br> [策略覆盖](#parameters-in-the-strategy)。 <br>*默认为 `-1`。*<br> **数据类型：** 正整数或 -1
| | **交易所**
| `exchange.name` | **必填。** 要使用的交易所类的名称。 <br> **数据类型：** 字符串
| `exchange.key` | 用于交易所的 API 密钥。仅在生产模式下需要。<br>**请保密，不要公开披露。** <br> **数据类型：** 字符串
| `exchange.secret` | 用于交易所的 API 密钥。仅在生产模式下需要。<br>**请保密，不要公开披露。** <br> **数据类型：** 字符串
| `exchange.password` | 用于交易所的 API 密码。仅在生产模式下以及使用密码进行 API 请求的交易所需要。<br>**请保密，不要公开披露。** <br> **数据类型：** 字符串
| `exchange.uid` | 用于交易所的 API uid。仅在生产模式下以及使用 uid 进行 API 请求的交易所需要。<br>**请保密，不要公开披露。** <br> **数据类型：** 字符串
| `exchange.pair_whitelist` | 机器人用于交易和在回测期间检查潜在交易的交易对列表。支持正则表达式交易对，如 `.*/BTC`。不被 VolumePairList 使用。[更多信息](plugins.md#pairlists-and-pairlist-handlers)。 <br> **数据类型：** 列表
| `exchange.pair_blacklist` | 机器人必须绝对避免用于交易和回测的交易对列表。[更多信息](plugins.md#pairlists-and-pairlist-handlers)。 <br> **数据类型：** 列表
| `exchange.ccxt_config` | 传递给两个 ccxt 实例（同步和异步）的附加 CCXT 参数。这通常是添加额外 ccxt 配置的正确位置。参数可能因交易所而异，并在 [ccxt 文档](https://docs.ccxt.com/#/README?id=overriding-exchange-properties-upon-instantiation) 中有详细说明。请避免在此处添加交易所密钥（改用专用字段），因为它们可能包含在日志中。 <br> **数据类型：** 字典
| `exchange.ccxt_sync_config` | 传递给常规（同步）ccxt 实例的附加 CCXT 参数。参数可能因交易所而异，并在 [ccxt 文档](https://docs.ccxt.com/#/README?id=overriding-exchange-properties-upon-instantiation) 中有详细说明 <br> **数据类型：** 字典
| `exchange.ccxt_async_config` | 传递给异步 ccxt 实例的附加 CCXT 参数。参数可能因交易所而异，并在 [ccxt 文档](https://docs.ccxt.com/#/README?id=overriding-exchange-properties-upon-instantiation) 中有详细说明 <br> **数据类型：** 字典
| `exchange.enable_ws` | 启用交易所的 WebSocket 使用。 <br>[更多信息](#consuming-exchange-websockets)。<br>*默认为 `true`。* <br> **数据类型：** 布尔值
| `exchange.markets_refresh_interval` | 重新加载市场的间隔（以分钟为单位）。 <br>*默认为 `60` 分钟。* <br> **数据类型：** 正整数
| `exchange.skip_open_order_update` | 如果交易所出现问题，在启动时跳过未成交订单更新。仅在实时条件下相关。<br>*默认为 `false`*<br> **数据类型：** 布尔值
| `exchange.unknown_fee_rate` | 计算交易费用时使用的回退值。这对于以不可交易货币收取费用的交易所可能很有用。此处提供的值将乘以“费用成本”。<br>*默认为 `None`<br> **数据类型：** 浮点数
| `exchange.log_responses` | 记录相关的交易所响应。仅用于调试模式 - 请谨慎使用。<br>*默认为 `false`*<br> **数据类型：** 布尔值
| `exchange.only_from_ccxt` | 阻止从 data.binance.vision 下载数据。保持为 false 可以大大加快下载速度，但如果该站点不可用可能会有问题。<br>*默认为 `false`*<br> **数据类型：** 布尔值
| `experimental.block_bad_exchanges` | 阻止已知不能与 freqtrade 一起使用的交易所。除非您想测试该交易所现在是否可用，否则请保持默认值。 <br>*默认为 `true`。* <br> **数据类型：** 布尔值
| | **插件**
| `pairlists` | 定义要使用的一个或多个交易对列表。[更多信息](plugins.md#pairlists-and-pairlist-handlers)。 <br>*默认为 `StaticPairList`。*  <br> **数据类型：** 字典列表
| | **Telegram**
| `telegram.enabled` | 启用 Telegram 的使用。 <br> **数据类型：** 布尔值
| `telegram.token` | 您的 Telegram 机器人令牌。仅当 `telegram.enabled` 为 `true` 时需要。 <br>**请保密，不要公开披露。** <br> **数据类型：** 字符串
| `telegram.chat_id` | 您的个人 Telegram 账户 ID。仅当 `telegram.enabled` 为 `true` 时需要。 <br>**请保密，不要公开披露。** <br> **数据类型：** 字符串
| `telegram.balance_dust_level` | 尘埃水平（以持仓货币计）- 余额低于此水平的货币将不会通过 `/balance` 显示。 <br> **数据类型：** 浮点数
| `telegram.reload` | 允许 Telegram 消息上的“重新加载”按钮。 <br>*默认为 `true`.<br> **数据类型：** 布尔值
| `telegram.notification_settings.*` | 详细的通知设置。有关详细信息，请参阅 [telegram 文档](telegram-usage.md)。<br> **数据类型：** 字典
| `telegram.allow_custom_messages` | 允许通过 dataprovider.send_msg() 函数从策略发送 Telegram 消息。 <br> **数据类型：** 布尔值
| | **Webhook**
| `webhook.enabled` | 启用 Webhook 通知的使用 <br> **数据类型：** 布尔值
| `webhook.url` | Webhook 的 URL。仅当 `webhook.enabled` 为 `true` 时需要。有关更多详细信息，请参见 [webhook 文档](webhook-config.md)。 <br> **数据类型：** 字符串
| `webhook.entry` | 入场时发送的负载。仅当 `webhook.enabled` 为 `true` 时需要。有关更多详细信息，请参见 [webhook 文档](webhook-config.md)。 <br> **数据类型：** 字符串
| `webhook.entry_cancel` | 入场订单取消时发送的负载。仅当 `webhook.enabled` 为 `true` 时需要。有关更多详细信息，请参见 [webhook 文档](webhook-config.md)。 <br> **数据类型：** 字符串
| `webhook.entry_fill` | 入场订单成交时发送的负载。仅当 `webhook.enabled` 为 `true` 时需要。有关更多详细信息，请参见 [webhook 文档](webhook-config.md)。 <br> **数据类型：** 字符串
| `webhook.exit` | 出场时发送的负载。仅当 `webhook.enabled` 为 `true` 时需要。有关更多详细信息，请参见 [webhook 文档](webhook-config.md)。 <br> **数据类型：** 字符串
| `webhook.exit_cancel` | 出场订单取消时发送的负载。仅当 `webhook.enabled` 为 `true` 时需要。有关更多详细信息，请参见 [webhook 文档](webhook-config.md)。 <br> **数据类型：** 字符串
| `webhook.exit_fill` | 出场订单成交时发送的负载。仅当 `webhook.enabled` 为 `true` 时需要。有关更多详细信息，请参见 [webhook 文档](webhook-config.md)。 <br> **数据类型：** 字符串
| `webhook.status` | 状态调用时发送的负载。仅当 `webhook.enabled` 为 `true` 时需要。有关更多详细信息，请参见 [webhook 文档](webhook-config.md)。 <br> **数据类型：** 字符串
| `webhook.allow_custom_messages` | 允许通过 dataprovider.send_msg() 函数从策略发送 Webhook 消息。 <br> **数据类型：** 布尔值
| | **Rest API / FreqUI / 生产者-消费者**
| `api_server.enabled` | 启用 API 服务器的使用。有关更多详细信息，请参见 [API 服务器文档](rest-api.md)。 <br> **数据类型：** 布尔值
| `api_server.listen_ip_address` | 绑定 IP 地址。有关更多详细信息，请参见 [API 服务器文档](rest-api.md)。 <br> **数据类型：** IPv4
| `api_server.listen_port` | 绑定端口。有关更多详细信息，请参见 [API 服务器文档](rest-api.md)。 <br>**数据类型：** 1024 到 65535 之间的整数
| `api_server.verbosity` | 日志详细程度。`info` 将打印所有 RPC 调用，而 "error" 仅显示错误。 <br>**数据类型：** 枚举，`info` 或 `error`。默认为 `info`。
| `api_server.username` | API 服务器的用户名。有关更多详细信息，请参见 [API 服务器文档](rest-api.md)。 <br>**请保密，不要公开披露。**<br> **数据类型：** 字符串
| `api_server.password` | API 服务器的密码。有关更多详细信息，请参见 [API 服务器文档](rest-api.md)。 <br>**请保密，不要公开披露。**<br> **数据类型：** 字符串
| `api_server.ws_token` | 消息 WebSocket 的 API 令牌。有关更多详细信息，请参见 [API 服务器文档](rest-api.md)。  <br>**请保密，不要公开披露。** <br> **数据类型：** 字符串
| `bot_name` | 机器人的名称。通过 API 传递给客户端 - 可以显示以区分/命名机器人。<br> *默认为 `freqtrade`*<br> **数据类型：** 字符串
| `external_message_consumer` | 启用 [生产者/消费者模式](producer-consumer.md) 了解更多详情。 <br> **数据类型：** 字典
| | **其他**
| `initial_state` | 定义初始应用程序状态。如果设置为 stopped，则必须通过 `/start` RPC 命令显式启动机器人。 <br>*默认为 `stopped`。* <br> **数据类型：** 枚举，`running`、`paused` 或 `stopped`
| `force_entry_enable` | 启用 RPC 命令强制交易入场。下面有更多信息。 <br> **数据类型：** 布尔值
| `disable_dataframe_checks` | 禁用检查从策略方法返回的 OHLCV 数据框的正确性。仅在有意更改数据框并了解自己在做什么时使用。[策略覆盖](#parameters-in-the-strategy)。<br> *默认为 `False`*。 <br> **数据类型：** 布尔值
| `internals.process_throttle_secs` | 设置进程节流，或一个机器人迭代循环的最小持续时间。以秒为单位的值。 <br>*默认为 `5` 秒。* <br> **数据类型：** 正整数
| `internals.heartbeat_interval` | 每 N 秒打印一次心跳消息。设置为 0 可禁用心跳消息。 <br>*默认为 `60` 秒。* <br> **数据类型：** 正整数或 0
| `internals.sd_notify` | 启用使用 sd_notify 协议告诉 systemd 服务管理器机器人状态的变化并发出保持活动 ping。详情请见 [这里](advanced-setup.md#configure-the-bot-running-as-a-systemd-service)。 <br> **数据类型：** 布尔值
| `strategy` | **必填** 定义要使用的策略类。建议通过 `--strategy NAME` 设置。 <br> **数据类型：** 类名
| `strategy_path` | 添加额外的策略查找路径（必须是目录）。 <br> **数据类型：** 字符串
| `recursive_strategy_search` | 设置为 `true` 以递归搜索 `user_data/strategies` 内的子目录以查找策略。 <br> **数据类型：** 布尔值
| `user_data_dir` | 包含用户数据的目录。 <br> *默认为 `./user_data/`*。 <br> **数据类型：** 字符串
| `db_url` | 声明要使用的数据库 URL。注意：如果 `dry_run` 为 `true`，默认为 `sqlite:///tradesv3.dryrun.sqlite`，生产实例默认为 `sqlite:///tradesv3.sqlite`。 <br> **数据类型：** 字符串，SQLAlchemy 连接字符串
| `logfile` | 指定日志文件名。对日志文件轮换使用滚动策略，10 个文件，每个文件限制为 1MB。 <br> **数据类型：** 字符串
| `add_config_files` | 附加配置文件。这些文件将被加载并与当前配置文件合并。文件相对于初始文件解析。<br> *默认为 `[]`*。 <br> **数据类型：** 字符串列表
| `dataformat_ohlcv` | 用于存储历史蜡烛（OHLCV）数据的数据格式。 <br> *默认为 `feather`*。 <br> **数据类型：** 字符串
| `dataformat_trades` | 用于存储历史交易数据的数据格式。 <br> *默认为 `feather`*。 <br> **数据类型：** 字符串
| `reduce_df_footprint` | 将所有数字列重铸为 float32/int32，目的是减少内存/磁盘使用（并减少回测/超参数优化和 FreqAI 中的训练/推理时间）。 <br> **数据类型：** 布尔值。 <br> 默认值：`False`。
| `log_config` | 包含 python 日志记录的日志配置的字典。[更多信息](advanced-setup.md#advanced-logging) <br> **数据类型：** 字典。 <br> 默认值：`FtRichHandler`

### 策略中的参数

以下参数可以在配置文件或策略中设置。
配置文件中设置的值始终覆盖策略中设置的值。

* `minimal_roi`
* `timeframe`
* `stoploss`
* `max_open_trades`
* `trailing_stop`
* `trailing_stop_positive`
* `trailing_stop_positive_offset`
* `trailing_only_offset_is_reached`
* `use_custom_stoploss`
* `process_only_new_candles`
* `order_types`
* `order_time_in_force`
* `unfilledtimeout`
* `disable_dataframe_checks`
* `use_exit_signal`
* `exit_profit_only`
* `exit_profit_offset`
* `ignore_roi_if_entry_signal`
* `ignore_buying_expired_candle_after`
* `position_adjustment_enable`
* `max_entry_position_adjustment`

### 配置每笔交易的金额

有几种方法可以配置机器人将使用多少持仓货币进入交易。所有方法都遵循 [可用余额配置](#tradable-balance)，如下所述。

#### 最小交易持仓

最小持仓金额将取决于交易所和交易对，通常在交易所支持页面中列出。

假设 XRP/USD 的最小可交易金额为 20 XRP（由交易所规定），价格为 0.6 美元，则买入该交易对的最小持仓金额为 `20 * 0.6 ~= 12`。
该交易所对 USD 也有一个限制 - 所有订单必须 > 10 美元 - 但在这种情况下不适用。

为保证安全执行，freqtrade 不允许以 10.1 美元的持仓金额买入，相反，它会确保在交易对下方有足够的空间设置止损（加上一个偏移量，由 `amount_reserve_percent` 定义，默认为 5%）。

如果准备金为 5%，则最小持仓金额约为 12.6 美元（`12 * (1 + 0.05)`）。如果我们再考虑 10% 的止损 - 我们最终会得到约 14 美元的值（`12.6 / (1 - 0.1)`）。

为了在止损值较大的情况下限制此计算，计算出的最小持仓限额永远不会超过实际限额的 50%。

!!! 警告
    由于交易所的限额通常是稳定的，并且不经常更新，一些交易对可能显示相当高的最低限额，这仅仅是因为自交易所上次调整限额以来价格上涨了很多。Freqtrade 会将持仓金额调整到此值，除非它比计算/期望的持仓金额高出 30% 以上 - 在这种情况下，交易将被拒绝。

#### 模拟交易钱包

在模拟交易模式下运行时，机器人将使用模拟钱包来执行交易。该钱包的初始余额由 `dry_run_wallet` 定义（默认为 1000）。
对于更复杂的场景，您还可以为 `dry_run_wallet` 分配一个字典来定义每种货币的初始余额。

```json
"dry_run_wallet": {
    "BTC": 0.01,
    "ETH": 2,
    "USDT": 1000
}
```

命令行选项（`--dry-run-wallet`）可用于覆盖配置值，但仅适用于浮点值，不适用于字典。如果您想使用字典，请调整配置文件。

!!! 注意
    非持仓货币的余额不会用于交易，但会显示为钱包余额的一部分。
    在交叉保证金交易所，钱包余额可用于计算可用于交易的抵押品。

#### 可交易余额

默认情况下，机器人假设 `完整金额 - 1%` 可供其使用，并且在使用 [动态持仓金额](#dynamic-stake-amount) 时，它会将完整余额平均分配到 `max_open_trades` 个交易桶中。
Freqtrade 将预留 1% 用于入场时的最终费用，因此默认情况下不会触及该费用。

您可以使用 `tradable_balance_ratio` 设置来配置“未触及”金额。

例如，如果您在交易所的钱包中有 10 ETH 可用，并且 `tradable_balance_ratio=0.5`（即 50%），那么机器人将使用最多 5 ETH 进行交易，并将其视为可用余额。钱包的其余部分不受交易影响。

!!! 危险
    在同一账户上运行多个机器人时，**不应**使用此设置。请改为查看 [分配可用资金给机器人](#assign-available-capital)。

!!! 警告
    `tradable_balance_ratio` 设置适用于当前余额（可用余额 + 交易中占用的余额）。因此，假设初始余额为 1000，配置 `tradable_balance_ratio=0.99` 不能保证交易所上始终有 10 个货币单位可用。例如，如果总余额减少到 500（无论是由于连续亏损还是提取余额），可用金额可能会减少到 5 个单位。

#### 分配可用资金

要在同一交易所账户上使用多个机器人时充分利用复利利润，您需要将每个机器人限制在特定的初始余额。
这可以通过将 `available_capital` 设置为所需的初始余额来实现。

假设您的账户有 10000 USDT，并且您想在该交易所上运行 2 种不同的策略。
您可以设置 `available_capital=5000` - 为每个机器人授予 5000 USDT 的初始资金。
然后，机器人会将此初始余额平均分配到 `max_open_trades` 个桶中。
有利可图的交易将导致该机器人的持仓规模增加 - 而不会影响其他机器人的持仓规模。

调整 `available_capital` 需要重新加载配置才能生效。当交易未平仓时减少可用资金不会平仓。差额在交易结束时返回钱包。结果因调整和平仓之间的价格变动而异。

!!! 警告 "与 `tradable_balance_ratio` 不兼容"
    设置此选项将取代 `tradable_balance_ratio` 的任何配置。

#### 修改最后持仓金额

假设我们的可交易余额为 1000 USDT，`stake_amount=400`，`max_open_trades=3`。
机器人将开 2 笔交易，并且无法填补最后一个交易槽，因为请求的 400 USDT 不再可用，因为 800 USDT 已经用于其他交易。

为了克服这个问题，可以将选项 `amend_last_stake_amount` 设置为 `True`，这将使机器人能够减少持仓金额到可用余额以填补最后一个交易槽。

在上面的例子中，这意味着：

* 交易1：400 USDT
* 交易2：400 USDT
* 交易3：200 USDT

!!! 注意
    此选项仅适用于 [静态持仓金额](#static-stake-amount) - 因为 [动态持仓金额](#dynamic-stake-amount) 会平均分配余额。

!!! 注意
    最小最后持仓金额可以使用 `last_stake_amount_min_ratio` 配置 - 默认为 0.5（50%）。这意味着曾经使用的最小持仓金额是 `stake_amount * 0.5`。这避免了非常低的持仓金额，接近交易对的最小可交易金额，可能会被交易所拒绝。

#### 静态持仓金额

`stake_amount` 配置静态配置您的机器人每笔交易将使用的持仓货币金额。

最小配置值为 0.0001，但是，请检查您所使用的持仓货币的交易所交易最低限额，以避免出现问题。

此设置与 `max_open_trades` 结合使用。交易中投入的最大资金为 `stake_amount * max_open_trades`。
例如，假设配置为 `max_open_trades=3` 和 `stake_amount=0.05`，机器人最多将使用（0.05 BTC x 3）= 0.15 BTC。

!!! 注意
    此设置遵循 [可用余额配置](#tradable-balance)。

#### 动态持仓金额

或者，您可以使用动态持仓金额，这将使用交易所的可用余额，并按允许的交易数量（`max_open_trades`）平均分配。

要配置此功能，请将 `stake_amount` 设置为 `"unlimited"`。我们还建议设置 `tradable_balance_ratio=0.99`（99%）- 保留最低余额用于最终费用。

在这种情况下，交易金额计算为：

```python
货币余额 / (max_open_trades - 当前未平仓交易)
```

要允许机器人使用账户中所有可用的 `stake_currency`（减去 `tradable_balance_ratio`），请设置

```json
"stake_amount" : "unlimited",
"tradable_balance_ratio": 0.99,
```

!!! 提示 "复利利润"
    此配置将允许根据机器人的性能增加/减少持仓（如果机器人亏损则减少持仓，如果机器人有盈利记录则增加持仓，因为有更多可用余额），并将导致利润复利。

!!! 注意 "使用模拟交易模式时"
    当在模拟交易、回测或超参数优化中结合使用 `"stake_amount" : "unlimited",` 时，余额将以 `dry_run_wallet` 的持仓开始模拟并演变。
    因此，将 `dry_run_wallet` 设置为合理的值（例如 BTC 为 0.05 或 0.01，USDT 为 1000 或 100）很重要，否则，它可能会模拟一次交易 100 BTC（或更多）或 0.05 USDT（或更少）- 这可能与您的实际可用余额不符，或者低于交易所对持仓货币的订单金额的最低限额。

#### 具有头寸调整的动态持仓金额

当您想使用无限持仓进行头寸调整时，您还必须实现 `custom_stake_amount` 以根据您的策略返回一个值。
典型值将在建议持仓的 25% - 50% 范围内，但在很大程度上取决于您的策略以及您希望在钱包中留下多少作为头寸调整缓冲。

例如，如果您的头寸调整假设它可以用相同的持仓金额再进行 2 次买入，那么您的缓冲应该是最初建议的无限持仓金额的 66.6667%。

或者另一个例子，如果您的头寸调整假设它可以用 3 倍于原始持仓金额进行 1 次额外买入，那么 `custom_stake_amount` 应该返回建议持仓金额的 25%，并留下 75% 用于可能的后续头寸调整。

--8<-- "includes/pricing.md"

## 更多配置细节

### 理解 minimal_roi

`minimal_roi` 配置参数是一个 JSON 对象，其中键是分钟数，值是最小 ROI（作为比率）。
见下面的例子：

```json
"minimal_roi": {
    "40": 0.0,    // 40分钟后如果利润不为负则出场
    "30": 0.01,   // 30分钟后如果至少有1%的利润则出场
    "20": 0.02,   // 20分钟后如果至少有2%的利润则出场
    "0":  0.04    // 立即出场如果至少有4%的利润
},
```

大多数策略文件已经包含了最佳的 `minimal_roi` 值。
此参数可以在策略或配置文件中设置。如果您在配置文件中使用它，它将覆盖
策略文件中的 `minimal_roi` 值。
如果在策略或配置中都没有设置，则使用默认值 1000% `{"0": 10}`，并且除非您的交易产生 1000% 的利润，否则最小 ROI 将被禁用。

!!! 注意 "特定时间后强制出场的特殊情况"
    一个特殊情况是使用 `"<N>": -1` 作为 ROI。这会迫使机器人在 N 分钟后出场，无论盈亏，因此代表限时强制出场。

### 理解 force_entry_enable

`force_entry_enable` 配置参数允许通过 Telegram 和 REST API 使用强制入场（`/forcelong`、`/forceshort`）命令。
出于安全原因，它默认是禁用的，如果启用，freqtrade 会在启动时显示警告消息。
例如，您可以向机器人发送 `/forceenter ETH/BTC`，这将导致 freqtrade 买入该交易对并持有，直到出现常规出场信号（ROI、止损、/forceexit）。

这在某些策略中可能很危险，因此请谨慎使用。

有关使用详情，请参见 [telegram 文档](telegram-usage.md)。

### 忽略过期蜡烛

当使用较大的时间周期（例如 1 小时或更长时间）并使用较低的 `max_open_trades` 值时，一旦交易槽可用，就可以处理最后一根蜡烛。处理最后一根蜡烛时，这可能导致不希望在该蜡烛上使用买入信号的情况。例如，当在您的策略中使用交叉条件时，该点可能已经过去太久，您无法在其上开始交易。

在这些情况下，您可以通过将 `ignore_buying_expired_candle_after` 设置为正数来启用忽略超过指定时间段的蜡烛的功能，该正数表示买入信号过期后的秒数。

例如，如果您的策略使用 1 小时时间周期，并且您只想在新蜡烛出现后的前 5 分钟内买入，您可以在策略中添加以下配置：

``` json
  {
    //...
    "ignore_buying_expired_candle_after": 300,
    // ...
  }
```

!!! 注意
    此设置随每个新蜡烛重置，因此它不会阻止在第 2 或第 3 根蜡烛上执行持续信号。最好为买入信号使用“触发”选择器，该选择器仅在一根蜡烛上有效。

### 理解 order_types

`order_types` 配置参数将操作（`entry`、`exit`、`stoploss`、`emergency_exit`、`force_exit`、`force_entry`）映射到订单类型（`market`、`limit` 等），以及将止损配置为在交易所上，并定义交易所止损更新间隔（以秒为单位）。

这允许使用限价单入场，使用限价单出场，并使用市价单创建止损。
它还允许设置
交易所上的止损，这意味着一旦买入订单成交，就会立即下达止损订单。

配置文件中设置的 `order_types` 会整体覆盖策略中设置的值，因此您需要在一个地方配置整个 `order_types` 字典。

如果配置了此参数，则需要存在以下 4 个值（`entry`、`exit`、`stoploss` 和 `stoploss_on_exchange`），否则机器人将无法启动。

有关（`emergency_exit`、`force_exit`、`force_entry`、`stoploss_on_exchange`、`stoploss_on_exchange_interval`、`stoploss_on_exchange_limit_ratio`）的信息，请参见止损文档 [交易所止损](stoploss.md)

策略语法：

```python
order_types = {
    "entry": "limit",
    "exit": "limit",
    "emergency_exit": "market",
    "force_entry": "market",
    "force_exit": "market",
    "stoploss": "market",
    "stoploss_on_exchange": False,
    "stoploss_on_exchange_interval": 60,
    "stoploss_on_exchange_limit_ratio": 0.99,
}
```

配置：

```json
"order_types": {
    "entry": "limit",
    "exit": "limit",
    "emergency_exit": "market",
    "force_entry": "market",
    "force_exit": "market",
    "stoploss": "market",
    "stoploss_on_exchange": false,
    "stoploss_on_exchange_interval": 60
}
```

!!! 注意 "市价单支持"
    并非所有交易所都支持“市价”订单。
    如果您的交易所不支持市价单，将显示以下消息：
    `"Exchange <yourexchange> does not support market orders."` 并且机器人将拒绝启动。

!!! 警告 "使用市价单"
    使用市价单时，请仔细阅读 [市价单定价](#market-order-pricing) 部分。

!!! 注意 "交易所止损"
    `order_types.stoploss_on_exchange_interval` 不是必需的。如果您不确定自己在做什么，请不要更改其值。有关止损如何工作的更多信息，请
    参考 [止损文档](stoploss.md)。

    如果启用了 `order_types.stoploss_on_exchange` 并且在交易所手动取消了止损，那么机器人将创建一个新的止损订单。

!!! 警告 "警告：order_types.stoploss_on_exchange 失败"
    如果由于某种原因在交易所创建止损失败，则会启动“紧急出场”。默认情况下，这将使用市价单出场。紧急出场的订单类型可以通过在 `order_types` 字典中设置 `emergency_exit` 值来更改 - 但是，不建议这样做。

### 理解 order_time_in_force

`order_time_in_force` 配置参数定义订单在交易所执行的策略。三个常用的有效时间是：

**GTC（取消前有效）：**

这大多数时候是默认的有效时间。这意味着订单将保留在交易所直到被用户取消。它可以被完全或部分成交。
如果部分成交，剩余部分将留在交易所直到取消。

**FOK（全部成交或取消）：**

这意味着如果订单没有立即且完全执行，则会被交易所取消。

**IOC（立即或取消）：**

与 FOK（上面）相同，只是它可以部分成交。剩余部分
由交易所自动取消。

**PO（仅挂单）：**

仅挂单。订单要么作为挂单（maker）下单，要么被取消。
这意味着订单必须至少在未成交状态下在订单簿上放置一段时间。

#### time_in_force 配置

`order_time_in_force` 参数包含一个带有入场和出场有效时间策略值的字典。
这可以在配置文件或策略中设置。
配置文件中设置的值会覆盖策略中设置的值。

可能的值为：`GTC`（默认）、`FOK` 或 `IOC`。

``` python
"order_time_in_force": {
    "entry": "GTC",
    "exit": "GTC"
},
```

!!! 警告
    这是正在进行的工作。目前，它仅支持 binance、gate 和 kucoin。
    除非您知道自己在做什么并且已经研究了为特定交易所使用不同值的影响，否则请不要更改默认值。

### 法定货币转换

Freqtrade 使用 Coingecko API 将硬币价值转换为相应的法定货币值，用于 Telegram 报告。
法定货币可以在配置文件中设置为 `fiat_display_currency`。

从配置中完全删除 `fiat_display_currency` 将跳过初始化 coingecko，并且不会显示任何法定货币转换。这对机器人的正确功能没有重要性。

#### fiat_display_currency 可以使用哪些值？

`fiat_display_currency` 配置参数设置用于从硬币到法定货币转换的基础货币，在机器人 Telegram 报告中。

有效值为：

```json
"AUD", "BRL", "CAD", "CHF", "CLP", "CNY", "CZK", "DKK", "EUR", "GBP", "HKD", "HUF", "IDR", "ILS", "INR", "JPY", "KRW", "MXN", "MYR", "NOK", "NZD", "PHP", "PKR", "PLN", "RUB", "SEK", "SGD", "THB", "TRY", "TWD", "ZAR", "USD"
```

除了法定货币外，还支持一系列加密货币。

有效值为：

```json
"BTC", "ETH", "XRP", "LTC", "BCH", "BNB"
```

#### Coingecko 速率限制问题

在某些 IP 范围内，coingecko 有严格的速率限制。
在这种情况下，您可能希望将您的 coingecko API 密钥添加到配置中。

``` json
{
    "fiat_display_currency": "USD",
    "coingecko": {
        "api_key": "your-api",
        "is_demo": true
    }
}
```

Freqtrade 支持 Demo 和 Pro coingecko API 密钥。

Coingecko API 密钥不是机器人正常运行所必需的。
它仅用于 Telegram 报告中从硬币到法定货币的转换，通常在没有 API 密钥的情况下也能工作。

## 消费交易所 WebSocket

Freqtrade 可以通过 ccxt.pro 消费 websocket。

Freqtrade 旨在确保数据始终可用。
如果 websocket 连接失败（或被禁用），机器人将回退到 REST API 调用。

如果您遇到怀疑是由 websocket 引起的问题，您可以通过设置 `exchange.enable_ws` 来禁用它们，默认为 true。

```jsonc
"exchange": {
    // ...
    "enable_ws": false,
    // ...
}
```

如果您需要使用代理，请参考 [代理部分](#using-a-proxy-with-freqtrade) 了解更多信息。

!!! 信息 "推出"
    我们正在缓慢实施，确保您的机器人稳定。
    目前，使用仅限于 ohlcv 数据流。
    它也仅限于少数交易所，新的交易所正在不断添加。

## 使用模拟交易模式

我们建议在模拟交易模式下启动机器人，看看您的机器人将如何表现以及您的策略的性能如何。在模拟交易模式下，机器人不会动用您的资金。它只运行实时模拟，不在交易所创建交易。

1. 编辑您的 `config.json` 配置文件。
2. 将 `dry-run` 切换为 `true` 并指定用于持久化数据库的 `db_url`。

```json
"dry_run": true,
"db_url": "sqlite:///tradesv3.dryrun.sqlite",
```

3. 删除您的交易所 API 密钥和密码（将它们更改为空值或假凭证）：

```json
"exchange": {
    "name": "binance",
    "key": "key",
    "secret": "secret",
    ...
}
```

一旦您对在模拟交易模式下运行的机器人性能感到满意，您就可以将其切换到生产模式。

!!! 注意
    模拟交易模式期间有一个模拟钱包可用，假设初始资金为 `dry_run_wallet`（默认为 1000）。

### 模拟交易的注意事项

* 可以提供也可以不提供 API 密钥。在模拟交易模式下，仅执行交易所上的只读操作（即不改变账户状态的操作）。
* 钱包（`/balance`）基于 `dry_run_wallet` 模拟。
* 订单是模拟的，不会发布到交易所。
* 市价单根据下单时的订单簿 volume 成交，最大滑点为 5%。
* 限价单在价格达到定义水平时成交 - 或根据 `unfilledtimeout` 设置超时。
* 如果限价单超过价格 1% 以上，将转换为市价单，并根据常规市价单规则立即成交（参见上面关于市价单的点）。
* 结合 `stoploss_on_exchange`，止损价格被假定为成交。
* 未平仓订单（不是存储在数据库中的交易）在机器人重启后保持打开状态，假设它们在离线时未成交。

## 切换到生产模式

在生产模式下，机器人将动用您的资金。请注意，错误的策略可能会使您损失所有资金。
当您在生产模式下运行它时，要清楚自己在做什么。

切换到生产模式时，请确保使用不同的/新的数据库，以避免模拟交易干扰您的交易所资金并最终影响您的统计数据。

### 设置您的交易所账户

您需要从交易所网站创建 API 密钥（通常您会获得 `key` 和 `secret`，有些交易所需要额外的 `password`），您需要将其插入到配置中的相应字段中，或者在 `freqtrade new-config` 命令询问时插入。
API 密钥通常仅用于实盘交易（用真钱交易，机器人运行在“生产模式”，在交易所执行真实订单），而在模拟交易（交易模拟）模式下运行的机器人不需要。当您在模拟交易模式下设置机器人时，您可以用空值填充这些字段。

### 将您的机器人切换到生产模式

**编辑您的 `config.json` 文件。**

**将 dry-run 切换为 false，不要忘记调整您的数据库 URL（如果已设置）：**

```json
"dry_run": false,
```

**插入您的交易所 API 密钥（用假 API 密钥更改它们）：**

```json
{
    "exchange": {
        "name": "binance",
        "key": "af8ddd35195e9dc500b9a6f799f6f5c93d89193b",
        "secret": "08a9dc6db3d7b53e1acebd9275677f4b0a04f1a5",
        //"password": "", // 可选，并非所有交易所都需要)
        // ...
    }
    //...
}
```

您还应该确保阅读文档的 [交易所](exchanges.md) 部分，了解特定于您的交易所的潜在配置细节。

!!! 提示 "保密您的秘密"
    为了保密您的秘密，我们建议使用第二个配置来存储您的 API 密钥。
    只需在新的配置文件（例如 `config-private.json`）中使用上面的代码片段，并将您的设置保存在此文件中。
    然后，您可以使用 `freqtrade trade --config user_data/config.json --config user_data/config-private.json <...>` 启动机器人以加载您的密钥。

    **永远不要** 与任何人分享您的私人配置文件或您的交易所密钥！

## 将代理与 Freqtrade 一起使用

要将代理与 freqtrade 一起使用，请使用设置为适当值的变量 `"HTTP_PROXY"` 和 `"HTTPS_PROXY"` 导出您的代理设置。
这将对所有内容（telegram、coingecko 等）应用代理设置 **除了** 交易所请求。

``` bash
export HTTP_PROXY="http://addr:port"
export HTTPS_PROXY="http://addr:port"
freqtrade
```

### 代理交易所请求

要将代理用于交易所连接 - 您必须将代理定义为 ccxt 配置的一部分。

``` json
{ 
  "exchange": {
    "ccxt_config": {
      "httpsProxy": "http://addr:port",
      "wsProxy": "http://addr:port",
    }
  }
}
```

有关可用代理类型的更多信息，请查阅 [ccxt 代理文档](https://docs.ccxt.com/#/README?id=proxy)。

## 下一步

现在您已经配置了 config.json，下一步是 [启动您的机器人](bot-usage.md)。