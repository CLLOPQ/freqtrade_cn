usage: freqtrade convert-trade-data [-h] [-v] [--no-color] [--logfile FILE]
                                    [-V] [-c PATH] [-d PATH] [--userdir PATH]
                                    [-p PAIRS [PAIRS ...]] --format-from
                                    {json,jsongz,feather,parquet,kraken_csv}
                                    --format-to {json,jsongz,feather,parquet}
                                    [--erase] [--exchange EXCHANGE]

选项:
  -h, --help            显示此帮助消息并退出
  -p PAIRS [PAIRS ...], --pairs PAIRS [PAIRS ...]
                        将命令限制为这些交易对。交易对之间用空格分隔。
  --format-from {json,jsongz,feather,parquet,kraken_csv}
                        数据转换的源格式。
  --format-to {json,jsongz,feather,parquet}
                        数据转换的目标格式。
  --erase               清除所选交易所/交易对/时间框架的所有现有数据。
  --exchange EXCHANGE   交易所名称。仅在未提供配置文件时有效。

通用参数:
  -v, --verbose         详细模式（-vv 获取更多信息，-vvv 获取所有消息）。
  --no-color            禁用超参数优化结果的彩色显示。如果将输出重定向到文件，这可能很有用。
  --logfile FILE, --log-file FILE
                        记录日志到指定文件。特殊值包括：'syslog'、'journald'。详见文档获取更多信息。
  -V, --version         显示程序版本号并退出
  -c PATH, --config PATH
                        指定配置文件（默认：`userdir/config.json` 或 `config.json`，取先存在的那个）。可使用多个 --config 选项。可设置为 `-` 以从标准输入读取配置。
  -d PATH, --datadir PATH, --data-dir PATH
                        包含交易所历史回测数据的基础目录路径。若要查看期货数据，需额外使用交易模式参数。
  --userdir PATH, --user-data-dir PATH
                        用户数据目录路径。