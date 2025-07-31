usage: freqtrade list-strategies [-h] [-v] [--no-color] [--logfile FILE] [-V]
                                 [-c PATH] [-d PATH] [--userdir PATH]
                                 [--strategy-path PATH] [-1]
                                 [--recursive-strategy-search]

选项:
  -h, --help            显示此帮助消息并退出
  --strategy-path PATH  指定额外的策略查找路径。
  -1, --one-column      单列打印输出。
  --recursive-strategy-search
                        在策略文件夹中递归搜索策略。

通用参数:
  -v, --verbose         详细模式（-vv获取更多信息，-vvv获取所有消息）。
  --no-color            禁用超参数优化结果的彩色显示。如果将输出重定向到文件，此选项可能有用。
  --logfile FILE, --log-file FILE
                        记录日志到指定文件。特殊值包括：'syslog'、'journald'。有关更多详细信息，请参阅文档。
  -V, --version         显示程序版本号并退出
  -c PATH, --config PATH
                        指定配置文件（默认值：`userdir/config.json` 或 `config.json`，存在哪个使用哪个）。可以使用多个 --config 选项。可设置为 `-` 以从标准输入读取配置。
  -d PATH, --datadir PATH, --data-dir PATH
                        包含交易所历史回测数据的基础目录路径。要查看期货数据，请额外使用 trading-mode 参数。
  --userdir PATH, --user-data-dir PATH
                        用户数据目录路径。