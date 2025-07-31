这将告诉Freqtrade输出一个序列化字典（pickled dictionary），包含策略、交易对以及对应产生入场和出场信号的K线数据DataFrame。根据策略产生的入场次数，此文件可能会变得相当大，因此请定期检查`user_data/backtest_results`文件夹并删除旧的导出文件。

在运行下一次回测之前，请确保删除旧的回测结果，或使用`--cache none`选项运行回测，以确保不使用缓存结果。

如果一切顺利，你现在应该在`user_data/backtest_results`文件夹中看到`backtest-result-{timestamp}_signals.pkl`和`backtest-result-{timestamp}_exited.pkl`文件。

要分析入场/出场标签，我们需要使用`freqtrade backtesting-analysis`命令，并提供`--analysis-groups`选项及空格分隔的参数：