usage: freqtrade backtesting-analysis [-h] [-v] [--no-color] [--logfile FILE]
                                      [-V] [-c PATH] [-d PATH]
                                      [--userdir PATH]
                                      [--export-filename PATH]
                                      [--analysis-groups {0,1,2,3,4,5} [{0,1,2,3,4,5} ...]]
                                      [--enter-reason-list ENTER_REASON_LIST [ENTER_REASON_LIST ...]]
                                      [--exit-reason-list EXIT_REASON_LIST [EXIT_REASON_LIST ...]]
                                      [--indicator-list INDICATOR_LIST [INDICATOR_LIST ...]]
                                      [--entry-only] [--exit-only]
                                      [--timerange TIMERANGE]
                                      [--rejected-signals] [--analysis-to-csv]
                                      [--analysis-csv-path ANALYSIS_CSV_PATH]

选项:
  -h, --help            显示此帮助信息并退出
  --export-filename PATH, --backtest-filename PATH
                        回测结果文件的文件名（用于加载分析数据）。
                        示例: `--export-filename=user_data/backtest_results/backtest_today.json`
  --analysis-groups {0,1,2,3,4,5} [{0,1,2,3,4,5} ...]
                        分组输出 - 0: 按入场标签的简单盈亏分析, 1: 按入场标签分析,
                        2: 按入场标签和出场标签分析, 3: 按交易对和入场标签分析,
                        4: 按交易对、入场和出场标签分析(结果可能较大), 5: 按出场标签分析
  --enter-reason-list ENTER_REASON_LIST [ENTER_REASON_LIST ...]
                        要分析的入场原因（标签）空格分隔列表。默认: 所有。
                        例如: 'entry_tag_a entry_tag_b'
  --exit-reason-list EXIT_REASON_LIST [EXIT_REASON_LIST ...]
                        要分析的出场原因（标签）空格分隔列表。默认: 所有。
                        例如: 'exit_tag_a roi stop_loss trailing_stop_loss'
  --indicator-list INDICATOR_LIST [INDICATOR_LIST ...]
                        要分析的指标空格分隔列表。例如:
                        'close rsi bb_lowerband profit_abs'
  --entry-only          仅分析入场原因（标签）。
  --exit-only           仅分析出场原因（标签）。
  --timerange TIMERANGE
                        指定要分析的数据时间范围。
  --rejected-signals    分析被拒绝的交易信号
  --analysis-to-csv     将选定的分析表格保存为单独的CSV文件
  --analysis-csv-path ANALYSIS_CSV_PATH
                        启用--analysis-to-csv时，指定保存分析CSV文件的路径。
                        默认: user_data/backtesting_results/

通用参数:
  -v, --verbose         详细模式(-vv 更详细, -vvv 显示所有消息)。
  --no-color            禁用输出内容的颜色显示。当输出重定向到文件时可能有用。
  --logfile FILE, --log-file FILE
                        日志输出到指定文件。特殊值: 'syslog', 'journald'。
                        更多详情请参见文档。
  -V, --version         显示程序版本号并退出
  -c PATH, --config PATH
                        指定配置文件(默认: `userdir/config.json` 或 `config.json`，以存在者为准)。
                        可以使用多个--config选项。可以设置为`-`从标准输入读取配置。
  -d PATH, --datadir PATH, --data-dir PATH
                        包含历史回测数据的交易所基础目录路径。要查看期货数据，
                        需额外使用trading-mode参数。
  --userdir PATH, --user-data-dir PATH
                        用户数据目录路径。