用法: freqtrade convert-db [-h] [--db-url PATH] [--db-url-from PATH]

选项:
  -h, --help          显示此帮助消息并退出
  --db-url PATH       覆盖交易数据库URL，这在自定义部署中很有用（默认值：实盘运行模式为`sqlite:///tradesv3.sqlite`，模拟运行为`sqlite:///tradesv3.dryrun.sqlite`）。
  --db-url-from PATH  迁移数据库时使用的源数据库URL。