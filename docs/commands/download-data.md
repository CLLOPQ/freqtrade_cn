usage: freqtrade download-data [-h] [-v] [--no-color] [--logfile FILE] [-V]
                               [-c PATH] [-d PATH] [--userdir PATH]
                               [-p PAIRS [PAIRS ...]] [--pairs-file FILE]
                               [--days INT] [--new-pairs-days INT]
                               [--include-inactive-pairs]
                               [--timerange TIMERANGE] [--dl-trades]
                               [--convert] [--exchange EXCHANGE]
                               [-t TIMEFRAMES [TIMEFRAMES ...]] [--erase]
                               [--data-format-ohlcv {json,jsongz,feather,parquet}]
                               [--data-format-trades {json,jsongz,feather,parquet}]
                               [--trading-mode {spot,margin,futures}]
                               [--prepend]

选项:
  -h, --help            显示此帮助消息并退出
  -p PAIRS [PAIRS ...], --pairs PAIRS [PAIRS ...]
                        将命令限制为这些交易对。交易对之间用空格分隔。
  --pairs-file FILE     包含交易对列表的文件。优先级高于--pairs选项或配置中设置的交易对。
  --days INT            下载指定天数的数据。
  --new-pairs-days INT  为新交易对下载指定天数的数据。默认值：`None`。
  --include-inactive-pairs
                        同时下载非活跃交易对的数据。
  --timerange TIMERANGE
                        指定要使用的数据时间范围。
  --dl-trades           下载交易数据而非OHLCV数据。
  --convert             将下载的交易数据转换为OHLCV数据。仅在与`--dl-trades`结合使用时有效。对于没有历史OHLCV数据的交易所（例如Kraken），此转换将自动进行。如果未提供此选项，请使用`trades-to-ohlcv`命令将交易数据转换为OHLCV数据。
  --exchange EXCHANGE   交易所名称。仅在未提供配置文件时有效。
  -t TIMEFRAMES [TIMEFRAMES ...], --timeframes TIMEFRAMES [TIMEFRAMES ...]
                        指定要下载的时间周期。空格分隔的列表。默认值：`1m 5m`。
  --erase               清除所选交易所/交易对/时间周期的所有现有数据。
  --data-format-ohlcv {json,jsongz,feather,parquet}
                        下载的K线（OHLCV）数据的存储格式。（默认值：`feather`）。
  --data-format-trades {json,jsongz,feather,parquet}
                        下载的交易数据的存储格式。（默认值：`feather`）。
  --trading-mode {spot,margin,futures}, --tradingmode {spot,margin,futures}
                        选择交易模式
  --prepend             允许数据前置。（数据追加已禁用）

通用参数:
  -v, --verbose         详细模式（-vv获取更多信息，-vvv获取所有消息）。
  --no-color            禁用超参数优化结果的颜色显示。如果将输出重定向到文件，此选项可能有用。
  --logfile FILE, --log-file FILE
                        记录日志到指定文件。特殊值有：'syslog'、'journald'。有关更多详细信息，请参阅文档。
  -V, --version         显示程序版本号并退出
  -c PATH, --config PATH
                        指定配置文件（默认值：`userdir/config.json` 或 `config.json`，取两者中存在的那个）。可以使用多个--config选项。可设置为`-`以从标准输入读取配置。
  -d PATH, --datadir PATH, --data-dir PATH
                        包含交易所历史回测数据的基础目录路径。要查看期货数据，请额外使用交易模式选项。
  --userdir PATH, --user-data-dir PATH
                        用户数据目录路径。