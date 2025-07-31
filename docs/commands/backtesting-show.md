用法: freqtrade backtesting-show [-h] [-v] [--no-color] [--logfile FILE] [-V]
                                  [-c PATH] [-d PATH] [--userdir PATH]
                                  [--export-filename PATH] [--show-pair-list]
                                  [--breakdown {day,week,month,year} [{day,week,month,year} ...]]

选项:
  -h, --help            显示此帮助信息并退出
  --export-filename PATH, --backtest-filename PATH
                        使用此文件名保存回测结果。需要同时设置`--export`。
                        示例: `--export-filename=user_data/backtest_results/backtest_today.json`
  --show-pair-list      显示按利润排序的回测交易对列表。
  --breakdown {day,week,month,year} [{day,week,month,year} ...]
                        按[日、周、月、年]显示回测细目。

通用参数:
  -v, --verbose         详细模式(-vv 更多, -vvv 获取所有消息)。
  --no-color            禁用输出结果的颜色显示。如果将输出重定向到文件可能有用。
  --logfile FILE, --log-file FILE
                        日志输出到指定文件。特殊值: 'syslog', 'journald'。
                        更多详情请参见文档。
  -V, --version         显示程序版本号并退出
  -c PATH, --config PATH
                        指定配置文件(默认: `userdir/config.json` 或 `config.json`，以存在者为准)。
                        可以使用多个--config选项。可以设置为`-`从标准输入读取配置。
  -d PATH, --datadir PATH, --data-dir PATH
                        包含历史回测数据的交易所基础目录路径。要查看期货数据，
                        需额外使用trading-mode。
  --userdir PATH, --user-data-dir PATH
                        用户数据目录路径。