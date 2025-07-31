usage: freqtrade list-timeframes [-h] [-v] [--no-color] [--logfile FILE] [-V]
                                 [-c PATH] [-d PATH] [--userdir PATH]
                                 [--exchange EXCHANGE] [-1]

选项:
  -h, --help            显示此帮助消息并退出
  --exchange EXCHANGE   交易所名称。仅在未提供配置文件时有效。
  -1, --one-column      单列打印输出。

通用参数:
  -v, --verbose         详细模式（-vv 表示更多，-vvv 获取所有消息）。
  --no-color            禁用超参数优化结果的颜色显示。如果将输出重定向到文件，这可能很有用。
  --logfile FILE, --log-file FILE
                        记录日志到指定文件。特殊值有：'syslog'、'journald'。详见文档了解更多信息。
  -V, --version         显示程序版本号并退出
  -c PATH, --config PATH
                        指定配置文件（默认：`userdir/config.json` 或 `config.json`，取先存在的那个）。可使用多个 --config 选项。可设置为 `-` 从标准输入读取配置。
  -d PATH, --datadir PATH, --data-dir PATH
                        包含交易所历史回测数据的基础目录路径。若要查看期货数据，需额外使用交易模式参数。
  --userdir PATH, --user-data-dir PATH
                        用户数据目录路径。