usage: freqtrade hyperopt-show [-h] [-v] [--no-color] [--logfile FILE] [-V]
                               [-c PATH] [-d PATH] [--userdir PATH] [--best]
                               [--profitable] [-n INT] [--print-json]
                               [--hyperopt-filename FILENAME] [--no-header]
                               [--disable-param-export]
                               [--breakdown {day,week,month,year} [{day,week,month,year} ...]]

选项:
  -h, --help            显示此帮助消息并退出
  --best                仅选择最佳轮次。
  --profitable          仅选择盈利的轮次。
  -n INT, --index INT   指定要打印详细信息的轮次索引。
  --print-json          以JSON格式打印输出。
  --hyperopt-filename FILENAME
                        超参数优化结果文件名。示例：`--hyperopt-
                        filename=hyperopt_results_2020-09-27_16-20-48.pickle`
  --no-header           不打印轮次详情标题。
  --disable-param-export
                        禁用超参数优化参数的自动导出。
  --breakdown {day,week,month,year} [{day,week,month,year} ...]
                        按[日、周、月、年]显示回测细分数据。

通用参数:
  -v, --verbose         详细模式（-vv获取更多信息，-vvv获取所有消息）。
  --no-color            禁用超参数优化结果的彩色显示。如果将输出重定向到文件，此选项可能有用。
  --logfile FILE, --log-file FILE
                        记录日志到指定文件。特殊值有：'syslog'、'journald'。详见文档获取更多详情。
  -V, --version         显示程序版本号并退出
  -c PATH, --config PATH
                        指定配置文件（默认：`userdir/config.json` 或 `config.json`，存在哪个使用哪个）。可使用多个--config选项。可设置为`-`从标准输入读取配置。
  -d PATH, --datadir PATH, --data-dir PATH
                        包含交易所历史回测数据的基础目录路径。要查看期货数据，需额外使用trading-mode参数。
  --userdir PATH, --user-data-dir PATH
                        用户数据目录路径。