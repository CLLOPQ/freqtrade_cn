freqtrade new-strategy --strategy AwesomeStrategy
通过让机器人知道需要多少历史数据，回测交易可以在回测和超参数优化期间从指定的时间范围开始。

!!! Warning "使用多次调用获取OHLCV数据"
    如果您收到类似 `WARNING - Using 3 calls to get OHLCV. This can result in slower operations for the bot. Please check if you really need 1500 candles for your strategy` 的警告，您应该考虑是否真的需要这么多历史数据来生成信号。
    这会导致Freqtrade对同一交易对进行多次调用，显然比单次网络请求要慢。
    因此，Freqtrade刷新K线数据的时间会更长，应尽可能避免这种情况。
    为避免给交易所造成过载或使Freqtrade过慢，调用次数上限为5次。

!!! Warning
    `startup_candle_count` 应小于 `ohlcv_candle_limit * 5`（大多数交易所为500 * 5 = 2500），因为在模拟交易/实盘交易期间，只有这么多K线数据可用。

#### 示例

让我们尝试使用上述带有EMA100的示例策略，回测1个月（2019年1月）的5分钟K线数据。
目前，其中包含`pair`，可通过`metadata['pair']`访问，返回格式为`XRP/BTC`的交易对（对于期货市场则为`XRP/BTC:BTC`）。

metadata字典不应被修改，且不会在策略中的多个函数间持久化信息。

相反，请查看[存储信息](strategy-advanced.md#storing-information-persistent)部分。

--8<-- "includes/strategy-imports.md"

## 策略文件加载

默认情况下，freqtrade会尝试从`userdir`（默认`user_data/strategies`）中的所有`.py`文件加载策略。

假设您的策略名为`AwesomeStrategy`，存储在文件`user_data/strategies/AwesomeStrategy.py`中，则可以通过以下命令以模拟（或实盘，取决于您的配置）模式启动freqtrade：
# 所有信息样本上的指标必须在此之前计算完成
    dataframe = pd.merge(dataframe, informative, left_on='date', right_on=f'date_merge_{inf_tf}', how='left')
    # 使用FFill使1天的数据值在全天的每一行中都可用
    # 如果没有此步骤，比较每天只能进行一次
    dataframe = dataframe.ffill()
free_eth = self.wallets.get_free('ETH')  # 可用ETH余额
    used_eth = self.wallets.get_used('ETH')  # 已用ETH余额（挂单中）
    total_eth = self.wallets.get_total('ETH')  # 总ETH余额


### 钱包的可用方法

- `get_free(asset)` - 当前可用于交易的可用余额
- `get_used(asset)` - 当前被占用的余额（挂单中）
- `get_total(asset)` - 总可用余额 - 上述两项之和

***

## 额外数据（交易记录）

策略中可以通过查询数据库获取交易历史记录。

在文件顶部，导入所需对象：