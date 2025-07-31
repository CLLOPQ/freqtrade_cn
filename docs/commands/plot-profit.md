usage: freqtrade plot-profit [-h] [-v] [--no-color] [--logfile FILE] [-V]
                             [-c PATH] [-d PATH] [--userdir PATH] [-s NAME]
                             [--strategy-path PATH]
                             [--recursive-strategy-search]
                             [--freqaimodel NAME] [--freqaimodel-path PATH]
                             [-p PAIRS [PAIRS ...]] [--timerange TIMERANGE]
                             [--export {none,trades,signals}]
                             [--export-filename PATH] [--db-url PATH]
                             [--trade-source {DB,file}] [-i TIMEFRAME]
                             [--auto-open]

选项:
  -h, --help            显示此帮助消息并退出
  -p PAIRS [PAIRS ...], --pairs PAIRS [PAIRS ...]
                        将命令限制为这些交易对。交易对之间用空格分隔。
  --timerange TIMERANGE
                        指定要使用的数据时间范围。
  --export {none,trades,signals}
                        导出回测结果（默认：trades）。
  --export-filename PATH, --backtest-filename PATH
                        使用此文件名保存回测结果。需要同时设置`--export`。示例：`--export-filename=user_data/backtest_results/backtest_today.json`
  --db-url PATH         覆盖交易数据库URL，这在自定义部署中很有用（默认：实盘模式为`sqlite:///tradesv3.sqlite`，模拟盘模式为`sqlite:///tradesv3.dryrun.sqlite`）。
  --trade-source {DB,file}
                        指定交易数据的来源（可以是DB或文件（回测文件））默认：file
  -i TIMEFRAME, --timeframe TIMEFRAME
                        指定时间周期（`1m`、`5m`、`30m`、`1h`、`1d`）。
  --auto-open           自动打开生成的图表。

通用参数:
  -v, --verbose         详细模式（-vv获取更多信息，-vvv获取所有消息）。
  --no-color            禁用超参数优化结果的彩色显示。如果将输出重定向到文件，这可能很有用。
  --logfile FILE, --log-file FILE
                        记录日志到指定文件。特殊值包括：'syslog'、'journald'。有关更多详细信息，请参阅文档。
  -V, --version         显示程序版本号并退出
  -c PATH, --config PATH
                        指定配置文件（默认：`userdir/config.json`或`config.json`，以存在者为准）。可以使用多个--config选项。可设置为`-`以从标准输入读取配置。
  -d PATH, --datadir PATH, --data-dir PATH
                        包含交易所历史回测数据的基础目录路径。要查看期货数据，需额外使用交易模式。
  --userdir PATH, --user-data-dir PATH
                        用户数据目录的路径。

策略参数:
  -s NAME, --strategy NAME
                        指定机器人将使用的策略类名。
  --strategy-path PATH  指定额外的策略查找路径。
  --recursive-strategy-search
                        在策略文件夹中递归搜索策略。
  --freqaimodel NAME    指定自定义的freqaimodels。
  --freqaimodel-path PATH
                        为freqaimodels指定额外的查找路径。