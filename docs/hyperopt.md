source .venv/bin/activate
pip install -r requirements-hyperopt.txt
buy_ema_short = IntParameter(3, 50, default=5)
    buy_ema_long = IntParameter(15, 200, default=50)


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """生成策略使用的所有指标"""
        
        # 计算所有ema_short值
        for val in self.buy_ema_short.range:
            dataframe[f'ema_short_{val}'] = ta.EMA(dataframe, timeperiod=val)
        
        # 计算所有ema_long值
        for val in self.buy_ema_long.range:
            dataframe[f'ema_long_{val}'] = ta.EMA(dataframe, timeperiod=val)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(qtpylib.crossed_above(
                dataframe[f'ema_short_{self.buy_ema_short.value}'], dataframe[f'ema_long_{self.buy_ema_long.value}']
            ))

        # 检查成交量不为0
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(qtpylib.crossed_above(
                dataframe[f'ema_long_{self.buy_ema_long.value}'], dataframe[f'ema_short_{self.buy_ema_short.value}']
            ))

        # 检查成交量不为0
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'] = 1
        return dataframe
读取命令（`hyperopt-list`、`hyperopt-show`）可以使用 `--hyperopt-filename <filename>` 来读取和显示旧的超参数优化结果。
    你可以使用 `ls -l user_data/hyperopt_results/` 命令查看文件名列表。

### 使用不同的历史数据源执行超参数优化

如果你想使用磁盘上已有的备用历史数据集来优化参数，可以使用 `--datadir PATH` 选项。默认情况下，超参数优化使用 `user_data/data` 目录中的数据。

### 使用较小的测试集运行超参数优化

使用 `--timerange` 参数可以更改要使用的测试集范围。
例如，要使用一个月的数据，可以在超参数优化命令中传递 `--timerange 20210101-20210201`（从2021年1月到2021年2月）。

完整命令：
默认情况下，超参数优化（hyperopt）模拟Freqtrade实盘运行/模拟运行的行为，即每个交易对只允许一个未平仓订单。所有交易对的未平仓订单总数也受`max_open_trades`设置限制。在超参数优化/回测期间，这可能导致潜在订单被已有的未平仓订单隐藏（或掩盖）。

`--eps`/`--enable-position-stacking`参数允许模拟多次买入同一交易对。将`--max-open-trades`设置为非常高的数值将禁用未平仓订单数量限制。

!!! Note
    模拟/实盘运行**不会**使用仓位叠加——因此，在不启用此功能的情况下验证策略也是有意义的，因为这更接近实际情况。

你也可以在配置文件中通过显式设置`"position_stacking"=true`来启用仓位叠加。

## 内存不足错误

由于超参数优化消耗大量内存（每个并行回测进程需要将完整数据一次性加载到内存中），你可能会遇到“内存不足”错误。为解决这些问题，你有多种选择：

* 减少交易对数量。
* 减少使用的时间范围（`--timerange <timerange>`）。
* 避免使用`--timeframe-detail`（这会将大量额外数据加载到内存中）。
* 减少并行进程数量（`-j <n>`）。
* 增加机器内存。
* 如果你使用了很多带有`.range`功能的参数，请使用`--analyze-per-epoch`。


## 目标函数此前已在该点评估过

如果你看到`The objective has been evaluated at this point before.`——这表明你的参数空间已耗尽，或接近耗尽。基本上，参数空间中的所有点都已被尝试（或已达到局部最小值）——超参数优化不再能找到多维空间中尚未尝试的点。Freqtrade尝试通过在这种情况下使用新的随机点来解决“局部最小值”问题。

示例：