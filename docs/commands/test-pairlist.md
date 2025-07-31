usage: freqtrade test-pairlist [-h] [--userdir PATH] [-v] [-c PATH]
                               [--quote QUOTE_CURRENCY [QUOTE_CURRENCY ...]]
                               [-1] [--print-json] [--exchange EXCHANGE]

options:
  -h, --help            显示此帮助消息并退出
  --userdir PATH, --user-data-dir PATH
                        用户数据目录的路径。
  -v, --verbose         详细模式（-vv获取更多信息，-vvv获取所有消息）。
  -c PATH, --config PATH
                        指定配置文件（默认：
                        `userdir/config.json` 或 `config.json`，存在哪个使用哪个）。可使用多个 --config 选项。可设置为 `-` 从标准输入读取配置。
  --quote QUOTE_CURRENCY [QUOTE_CURRENCY ...]
                        指定报价货币（可多个）。以空格分隔的列表。
  -1, --one-column      单列打印输出。
  --print-json          以JSON格式打印交易对或市场符号列表。
  --exchange EXCHANGE   交易所名称。仅在未提供配置时有效。