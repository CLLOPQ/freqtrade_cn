## 交易对列表和交易对列表处理器

交易对列表处理器（Pairlist Handlers）定义了机器人应该交易的交易对列表（pairlist）。它们在配置设置的`pairlists`部分进行配置。

在配置中，您可以使用静态交易对列表（由[`StaticPairList`](#static-pair-list)交易对列表处理器定义）和动态交易对列表（由[`VolumePairList`](#volume-pair-list)和[`PercentChangePairList`](#percent-change-pair-list)交易对列表处理器定义）。

此外，[`AgeFilter`](#agefilter)、[`PrecisionFilter`](#precisionfilter)、[`PriceFilter`](#pricefilter)、[`ShuffleFilter`](#shufflefilter)、[`SpreadFilter`](#spreadfilter)和[`VolatilityFilter`](#volatilityfilter)作为交易对列表过滤器，用于移除某些交易对和/或调整它们在交易对列表中的位置。

如果使用多个交易对列表处理器，它们会被链式调用，所有交易对列表处理器的组合形成机器人用于交易和回测的最终交易对列表。交易对列表处理器按照配置的顺序执行。您可以将`StaticPairList`、`VolumePairList`、`ProducerPairList`、`RemotePairList`、`MarketCapPairList`或`PercentChangePairList`定义为起始交易对列表处理器。

非活跃市场会始终从最终交易对列表中移除。明确列入黑名单的交易对（配置中`pair_blacklist`设置中的交易对）也会始终从最终交易对列表中移除。

### 交易对黑名单

交易对黑名单（通过配置中的`exchange.pair_blacklist`配置）禁止交易某些交易对。
这可以简单到排除`DOGE/BTC`——这将精确移除该交易对。

交易对黑名单还支持通配符（正则表达式风格）——因此`BNB/.*`将排除所有以BNB开头的交易对。
您也可以使用类似`.*DOWN/BTC`或`.*UP/BTC`的模式来排除杠杆代币（请检查您交易所的交易对命名规则！）

### 可用的交易对列表处理器

* [`StaticPairList`](#static-pair-list)（默认，如果未配置其他处理器）
* [`VolumePairList`](#volume-pair-list)
* [`PercentChangePairList`](#percent-change-pair-list)
* [`ProducerPairList`](#producerpairlist)
* [`RemotePairList`](#remotepairlist)
* [`MarketCapPairList`](#marketcappairlist)
* [`AgeFilter`](#agefilter)
* [`FullTradesFilter`](#fulltradesfilter)
* [`OffsetFilter`](#offsetfilter)
* [`PerformanceFilter`](#performancefilter)
* [`PrecisionFilter`](#precisionfilter)
* [`PriceFilter`](#pricefilter)
* [`ShuffleFilter`](#shufflefilter)
* [`SpreadFilter`](#spreadfilter)
* [`RangeStabilityFilter`](#rangestabilityfilter)
* [`VolatilityFilter`](#volatilityfilter)

!!! 提示 "测试交易对列表"
    交易对列表配置可能比较复杂，难以正确设置。最好使用[`test-pairlist`](utils.md#test-pairlist)工具子命令来快速测试您的配置。

#### 静态交易对列表

默认情况下，使用`StaticPairList`方法，该方法使用配置中静态定义的交易对白名单。交易对列表还支持通配符（正则表达式风格）——因此`.*/BTC`将包含所有以BTC为基础货币的交易对。

它使用`exchange.pair_whitelist`和`exchange.pair_blacklist`中的配置，在以下示例中，将交易BTC/USDT和ETH/USDT，并阻止BNB/USDT交易。

`pair_*list`参数均支持正则表达式——因此像`.*/USDT`这样的值将启用所有不在黑名单中的交易对。
`PercentChangePairList` 不支持回测模式。

#### ProducerPairList

使用 `ProducerPairList`，您可以重用来自 [生产者](producer-consumer.md) 的交易对列表，而无需在每个消费者上显式定义交易对列表。

此交易对列表需要 [消费者模式](producer-consumer.md) 才能正常工作。

该交易对列表会根据当前交易所配置检查活跃交易对，以避免尝试在无效市场上交易。

您可以使用可选参数 `number_assets` 限制交易对列表的长度。使用 `"number_assets"=0` 或省略此键将重用当前设置下所有有效的生产者交易对。
此选项默认禁用，仅当设置为 > 0 时才会生效。

`max_price` 设置会移除价格高于指定价格的交易对。如果您只想交易低价交易对，此设置非常有用。
此选项默认禁用，仅当设置为 > 0 时才会生效。

`max_value` 设置会移除最小价值变动高于指定值的交易对。
当交易所存在不平衡的限制时，此设置非常有用。例如，如果步长 = 1（因此您只能购买 1、2 或 3 个币，而不能购买 1.1 个币），且价格相当高（如 20 美元），因为自上次限制调整以来币价已大幅上涨。
上述情况导致您只能以 20 美元或 40 美元购买，而不能以 25 美元购买。
在从接收货币中扣除费用的交易所（例如币安）上，这可能导致高价值的币种/数量因金额略低于限制而无法卖出。

`low_price_ratio` 设置会移除 1 个价格单位（点）的涨幅超过 `low_price_ratio` 比率的交易对。
此选项默认禁用，仅当设置为 > 0 时才会生效。

对于 `PriceFilter`，必须至少应用其 `min_price`、`max_price` 或 `low_price_ratio` 设置中的一项。

计算示例：

SHITCOIN/BTC 的最低价格精度为 8 位小数。如果其价格为 0.00000011，那么上涨一个价格步长后为 0.00000012，比之前的价格高出约 9%。您可以通过使用 `low_price_ratio` 设置为 0.09（9%）的 PriceFilter，或相应地将 `min_price` 设置为 0.00000011 来过滤掉此交易对。

!!! Warning "低价交易对"
    具有高“1 点波动”的低价交易对很危险，因为它们通常流动性差，且可能无法设置理想的止损，这往往会导致高额损失，因为价格需要四舍五入到下一个可交易价格——因此，原本设置的 -5% 止损可能会因价格四舍五入而最终变成 -9% 的止损。

#### ShuffleFilter（随机排序过滤器）

对交易对列表中的交易对进行随机打乱。当您希望所有交易对被同等对待时，此过滤器可用于防止机器人更频繁地交易某些交易对。

默认情况下，ShuffleFilter 每根K线打乱一次交易对。
要在每次迭代时打乱，请将 `"shuffle_frequency"` 设置为 `"iteration"`，而不是默认的 `"candle"`。