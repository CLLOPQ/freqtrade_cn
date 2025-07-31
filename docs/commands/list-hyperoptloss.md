usage: freqtrade list-hyperoptloss [-h] [-v] [--no-color] [--logfile FILE]
                                   [-V] [-c PATH] [-d PATH] [--userdir PATH]
                                   [--hyperopt-path PATH] [-1]

选项:
  -h, --help            显示此帮助消息并退出
  --hyperopt-path PATH  指定超参数优化（Hyperopt）损失函数的额外查找路径。
  -1, --one-column      单列打印输出。

通用参数:
  -v, --verbose         详细模式（-vv获取更多信息，-vvv获取所有消息）。
  --no-color            禁用超参数优化结果的彩色显示。如果将输出重定向到文件，此选项可能有用。
  --logfile FILE, --log-file FILE
                        记录日志到指定文件。特殊值包括：'syslog'、'journald'。更多详情请参阅文档。
  -V, --version         显示程序版本号并退出
  -c PATH, --config PATH
                        指定配置文件（默认：`userdir/config.json` 或 `config.json`，存在哪个使用哪个）。可使用多个--config选项。可设置为 `-` 从标准输入读取配置。
  -d PATH, --datadir PATH, --data-dir PATH
                        包含交易所历史回测数据的基础目录路径。要查看期货数据，请额外使用trading-mode参数。
  --userdir PATH, --user-data-dir PATH
                        用户数据目录路径。