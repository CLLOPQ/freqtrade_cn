usage: freqtrade list-data [-h] [-v] [--no-color] [--logfile FILE] [-V]
                           [-c PATH] [-d PATH] [--userdir PATH]
                           [--exchange EXCHANGE]
                           [--data-format-ohlcv {json,jsongz,feather,parquet}]
                           [--data-format-trades {json,jsongz,feather,parquet}]
                           [--trades] [-p PAIRS [PAIRS ...]]
                           [--trading-mode {spot,margin,futures}]
                           [--show-timerange]

选项:
  -h, --help            显示此帮助消息并退出
  --exchange EXCHANGE   交易所名称。仅在未提供配置文件时有效。
  --data-format-ohlcv {json,jsongz,feather,parquet}
                        下载的K线（OHLCV）数据的存储格式。（默认：`feather`）。
  --data-format-trades {json,jsongz,feather,parquet}
                        下载的交易数据的存储格式。（默认：`feather`）。
  --trades              处理交易数据而非OHLCV数据。
  -p PAIRS [PAIRS ...], --pairs PAIRS [PAIRS ...]
                        将命令限制为这些交易对。交易对之间用空格分隔。
  --trading-mode {spot,margin,futures}, --tradingmode {spot,margin,futures}
                        选择交易模式
  --show-timerange      显示可用数据的时间范围。（计算可能需要一段时间）。

通用参数:
  -v, --verbose         详细模式（-vv获取更多信息，-vvv获取所有消息）。
  --no-color            禁用超参数优化结果的彩色显示。如果将输出重定向到文件，此选项可能有用。
  --logfile FILE, --log-file FILE
                        将日志记录到指定文件。特殊值为：'syslog'、'journald'。有关更多详细信息，请参阅文档。
  -V, --version         显示程序版本号并退出
  -c PATH, --config PATH
                        指定配置文件（默认：`userdir/config.json` 或 `config.json`，存在哪个使用哪个）。可以使用多个 --config 选项。可设置为 `-` 以从标准输入读取配置。
  -d PATH, --datadir PATH, --data-dir PATH
                        包含交易所历史回测数据的基础目录路径。要查看期货数据，需额外使用交易模式参数。
  --userdir PATH, --user-data-dir PATH
                        用户数据目录路径。