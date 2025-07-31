usage: freqtrade plot-dataframe [-h] [-v] [--no-color] [--logfile FILE] [-V]
                                [-c PATH] [-d PATH] [--userdir PATH] [-s NAME]
                                [--strategy-path PATH]
                                [--recursive-strategy-search]
                                [--freqaimodel NAME] [--freqaimodel-path PATH]
                                [-p PAIRS [PAIRS ...]]
                                [--indicators1 INDICATORS1 [INDICATORS1 ...]]
                                [--indicators2 INDICATORS2 [INDICATORS2 ...]]
                                [--plot-limit INT] [--db-url PATH]
                                [--trade-source {DB,file}]
                                [--export {none,trades,signals}]
                                [--export-filename PATH]
                                [--timerange TIMERANGE] [-i TIMEFRAME]
                                [--no-trades]

选项:
  -h, --help            显示此帮助消息并退出
  -p PAIRS [PAIRS ...], --pairs PAIRS [PAIRS ...]
                        将命令限制为这些交易对。交易对之间用空格分隔。
  --indicators1 INDICATORS1 [INDICATORS1 ...]
                        设置策略中要显示在图表第一行的指标。空格分隔的列表。示例：
                        `ema3 ema5`。默认值：`['sma', 'ema3', 'ema5']`。
  --indicators2 INDICATORS2 [INDICATORS2 ...]
                        设置策略中要显示在图表第三行的指标。空格分隔的列表。示例：
                        `fastd fastk`。默认值：`['macd', 'macdsignal']`。
  --plot-limit INT      指定绘图的K线限制。注意：过高的值会导致文件过大。默认值：750。
  --db-url PATH         覆盖交易数据库URL，这在自定义部署中很有用（默认值：实盘模式为
                        `sqlite:///tradesv3.sqlite`，回测模式为
                        `sqlite:///tradesv3.dryrun.sqlite`）。
  --trade-source {DB,file}
                        指定交易数据的来源（可以是DB或文件（回测文件））默认值：file
  --export {none,trades,signals}
                        导出回测结果（默认值：trades）。
  --export-filename PATH, --backtest-filename PATH
                        使用此文件名保存回测结果。需要同时设置`--export`。示例：
                        `--export-filename=user_data/backtest_results/backtest_today.json`
  --timerange TIMERANGE
                        指定要使用的数据时间范围。
  -i TIMEFRAME, --timeframe TIMEFRAME
                        指定时间周期（`1m`、`5m`、`30m`、`1h`、`1d`）。
  --no-trades           跳过使用回测文件和数据库中的交易数据。

通用参数:
  -v, --verbose         详细模式（-vv获取更多信息，-vvv获取所有消息）。
  --no-color            禁用超参数优化结果的彩色显示。如果将输出重定向到文件，这可能很有用。
  --logfile FILE, --log-file FILE
                        记录日志到指定文件。特殊值：'syslog'、'journald'。详见文档获取更多信息。
  -V, --version         显示程序版本号并退出
  -c PATH, --config PATH
                        指定配置文件（默认值：`userdir/config.json` 或 `config.json`，以存在的为准）。
                        可使用多个--config选项。可设置为`-`以从标准输入读取配置。
  -d PATH, --datadir PATH, --data-dir PATH
                        包含交易所历史回测数据的基础目录路径。要查看期货数据，需额外使用trading-mode。
  --userdir PATH, --user-data-dir PATH
                        用户数据目录路径。

策略参数:
  -s NAME, --strategy NAME
                        指定机器人将使用的策略类名。
  --strategy-path PATH  指定额外的策略查找路径。
  --recursive-strategy-search
                        在策略文件夹中递归搜索策略。
  --freqaimodel NAME    指定自定义的freqaimodels。
  --freqaimodel-path PATH
                        指定freqaimodels的额外查找路径。