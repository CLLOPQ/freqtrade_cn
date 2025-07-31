usage: freqtrade strategy-updater [-h] [-v] [--no-color] [--logfile FILE] [-V]
                                  [-c PATH] [-d PATH] [--userdir PATH]
                                  [--strategy-list STRATEGY_LIST [STRATEGY_LIST ...]]
                                  [--strategy-path PATH]
                                  [--recursive-strategy-search]

选项:
  -h, --help            显示此帮助消息并退出
  --strategy-list STRATEGY_LIST [STRATEGY_LIST ...]
                        提供以空格分隔的策略列表用于回测。请注意，时间框架需要在配置文件中或通过命令行设置。当与`--export trades`一起使用时，策略名称会被注入到文件名中（例如`backtest-data.json`会变为`backtest-data-SampleStrategy.json`
  --strategy-path PATH  指定额外的策略查找路径。
  --recursive-strategy-search
                        在策略文件夹中递归搜索策略。

通用参数:
  -v, --verbose         详细模式（-vv获取更多信息，-vvv获取所有消息）。
  --no-color            禁用超参数优化结果的彩色显示。如果将输出重定向到文件，此选项可能有用。
  --logfile FILE, --log-file FILE
                        记录日志到指定文件。特殊值包括：'syslog'、'journald'。详见文档获取更多详细信息。
  -V, --version         显示程序版本号并退出
  -c PATH, --config PATH
                        指定配置文件（默认值：`userdir/config.json` 或 `config.json`，取先存在的那个）。可使用多个--config选项。设置为`-`可从标准输入读取配置。
  -d PATH, --datadir PATH, --data-dir PATH
                        包含交易所历史回测数据的基础目录路径。若要查看期货数据，需额外使用交易模式。
  --userdir PATH, --user-data-dir PATH
                        用户数据目录路径。