usage: freqtrade hyperopt [-h] [-v] [--no-color] [--logfile FILE] [-V]
                          [-c PATH] [-d PATH] [--userdir PATH] [-s NAME]
                          [--strategy-path PATH] [--recursive-strategy-search]
                          [--freqaimodel NAME] [--freqaimodel-path PATH]
                          [-i TIMEFRAME] [--timerange TIMERANGE]
                          [--data-format-ohlcv {json,jsongz,feather,parquet}]
                          [--max-open-trades INT]
                          [--stake-amount STAKE_AMOUNT] [--fee FLOAT]
                          [-p PAIRS [PAIRS ...]] [--hyperopt-path PATH]
                          [--eps] [--enable-protections]
                          [--dry-run-wallet DRY_RUN_WALLET]
                          [--timeframe-detail TIMEFRAME_DETAIL] [-e INT]
                          [--spaces {all,buy,sell,roi,stoploss,trailing,protection,trades,default} [{all,buy,sell,roi,stoploss,trailing,protection,trades,default} ...]]
                          [--print-all] [--print-json] [-j JOBS]
                          [--random-state INT] [--min-trades INT]
                          [--hyperopt-loss NAME] [--disable-param-export]
                          [--ignore-missing-spaces] [--analyze-per-epoch]
                          [--early-stop INT]

options:
  -h, --help            显示此帮助消息并退出
  -i TIMEFRAME, --timeframe TIMEFRAME
                        指定时间框架（`1m`、`5m`、`30m`、`1h`、`1d`）。
  --timerange TIMERANGE
                        指定要使用的数据时间范围。
  --data-format-ohlcv {json,jsongz,feather,parquet}
                        下载的K线（OHLCV）数据的存储格式。（默认：`feather`）。
  --max-open-trades INT
                        覆盖`max_open_trades`配置设置的值。
  --stake-amount STAKE_AMOUNT
                        覆盖`stake_amount`配置设置的值。
  --fee FLOAT           指定费率。将应用两次（在交易入场和出场时）。
  -p PAIRS [PAIRS ...], --pairs PAIRS [PAIRS ...]
                        将命令限制为这些交易对。交易对之间用空格分隔。
  --hyperopt-path PATH  为超参数优化损失函数指定额外的查找路径。
  --eps, --enable-position-stacking
                        允许多次购买同一交易对（仓位堆叠）。
  --enable-protections, --enableprotections
                        为回测启用保护机制。会显著减慢回测速度，但会包含已配置的保护机制
  --dry-run-wallet DRY_RUN_WALLET, --starting-balance DRY_RUN_WALLET
                        初始资金，用于回测/超参数优化和模拟交易。
  --timeframe-detail TIMEFRAME_DETAIL
                        为回测指定详细时间框架（`1m`、`5m`、`30m`、`1h`、`1d`）。
  -e INT, --epochs INT  指定迭代次数（默认：100）。
  --spaces {all,buy,sell,roi,stoploss,trailing,protection,trades,default} [{all,buy,sell,roi,stoploss,trailing,protection,trades,default} ...]
                        指定要超参数优化的参数。用空格分隔的列表。
  --print-all           打印所有结果，而不仅仅是最佳结果。
  --print-json          以JSON格式打印输出。
  -j JOBS, --job-workers JOBS
                        超参数优化的并发运行作业数（超参数优化工作进程）。如果为-1（默认），则使用所有CPU；为-2，则使用除一个外的所有CPU，依此类推。如果为1，则完全不使用并行计算代码。
  --random-state INT    将随机状态设置为某个正整数，以获得可重现的超参数优化结果。
  --min-trades INT      为超参数优化路径中的评估设置最小期望交易数量（默认：1）。
  --hyperopt-loss NAME, --hyperoptloss NAME
                        指定超参数优化损失函数类（IHyperOptLoss）的类名。不同的函数会产生完全不同的结果，因为优化目标不同。内置的超参数优化损失函数包括：
                        ShortTradeDurHyperOptLoss、OnlyProfitHyperOptLoss、
                        SharpeHyperOptLoss、SharpeHyperOptLossDaily、
                        SortinoHyperOptLoss、SortinoHyperOptLossDaily、
                        CalmarHyperOptLoss、MaxDrawDownHyperOptLoss、
                        MaxDrawDownRelativeHyperOptLoss、
                        MaxDrawDownPerPairHyperOptLoss、
                        ProfitDrawDownHyperOptLoss、MultiMetricHyperOptLoss
  --disable-param-export
                        禁用超参数优化参数的自动导出。
  --ignore-missing-spaces, --ignore-unparameterized-spaces
                        对任何请求的不含参数的超参数优化空间抑制错误。
  --analyze-per-epoch   每个迭代周期运行一次populate_indicators。
  --early-stop INT      如果在（默认：0）个迭代周期后没有改进，则提前停止超参数优化。

Common arguments:
  -v, --verbose         详细模式（-vv 表示更多，-vvv 表示所有消息）。
  --no-color            禁用超参数优化结果的彩色显示。如果将输出重定向到文件，这可能很有用。
  --logfile FILE, --log-file FILE
                        记录到指定的文件。特殊值为：'syslog'、'journald'。有关更多详细信息，请参阅文档。
  -V, --version         显示程序版本号并退出
  -c PATH, --config PATH
                        指定配置文件（默认：`userdir/config.json` 或 `config.json`，以存在的为准）。可以使用多个 --config 选项。可以设置为 `-` 以从标准输入读取配置。
  -d PATH, --datadir PATH, --data-dir PATH
                        包含历史回测数据的交易所基础目录路径。要查看期货数据，请额外使用交易模式。
  --userdir PATH, --user-data-dir PATH
                        用户数据目录的路径。

Strategy arguments:
  -s NAME, --strategy NAME
                        指定机器人将使用的策略类名。
  --strategy-path PATH  指定额外的策略查找路径。
  --recursive-strategy-search
                        在策略文件夹中递归搜索策略。
  --freqaimodel NAME    指定自定义的freqaimodel。
  --freqaimodel-path PATH
                        为freqaimodel指定额外的查找路径。