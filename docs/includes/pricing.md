## 订单使用的价格

常规订单的价格可以通过参数结构控制：`entry_pricing` 用于交易入场，`exit_pricing` 用于交易出场。
价格总是在下单前获取，通过查询交易所行情（tickers）或使用订单簿（orderbook）数据。

!!! Note
    Freqtrade使用的订单簿数据是通过ccxt的`fetch_order_book()`函数从交易所获取的，通常是L2聚合订单簿数据，而行情数据是ccxt的`fetch_ticker()`/`fetch_tickers()`函数返回的结构。更多详情请参考ccxt库的[文档](https://github.com/ccxt/ccxt/wiki/Manual#market-data)。

!!! Warning "使用市价单"
    使用市价单时，请阅读[市价单定价](#market-order-pricing)部分。

### 入场价格

#### 入场价格方向

配置项 `entry_pricing.price_side` 定义机器人买入时查看订单簿的方向。

以下是订单簿的示例：