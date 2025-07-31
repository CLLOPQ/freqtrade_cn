usage: freqtrade backtesting [-h] [-v] [--no-color] [--logfile FILE] [-V]
                             [-c PATH] [-d PATH] [--userdir PATH] [-s NAME]
                             [--strategy-path PATH]
                             [--recursive-strategy-search]
                             [--freqaimodel NAME] [--freqaimodel-path PATH]
                             [-i TIMEFRAME] [--timerange TIMERANGE]
                             [--data-format-ohlcv {json,jsongz,feather,parquet}]
                             [--max-open-trades INT]
                             [--stake-amount STAKE_AMOUNT] [--fee FLOAT]
                             [-p PAIRS [PAIRS ...]] [--eps]
                             [--enable-protections]
                             [--dry-run-wallet DRY_RUN_WALLET]
                             [--timeframe-detail TIMEFRAME_DETAIL]
                             [--strategy-list STRATEGY_LIST [STRATEGY_LIST ...]]
                             [--export {none,trades,signals}]
                             [--export-filename PATH]
                             [--breakdown {day,week,month,year} [{day,week,month,year} ...]]
                             [--cache {none,day,week,month}]
                             [--freqai-backtest-live-models] [--notes TEXT]

选项:
  -h, --help            显示此帮助信息并退出
  -i TIMEFRAME, --timeframe TIMEFRAME
                        指定时间框架（`1m`、`5m`、`30m`、`1h`、`1d`）。
  --timerange TIMERANGE
                        指定要使用的时间范围。
  --data-format-ohlcv {json,jsongz,feather,parquet}
                        下载的K线（OHLCV）数据的存储格式。（默认：`feather`）。
  --max-open-trades INT
                        覆盖`max_open_trades`配置设置的值。
  --stake-amount STAKE_AMOUNT
                        覆盖`stake_amount`配置设置的值。
  --fee FLOAT           指定手续费比例。将被应用两次（在交易入场和出场时）。
  -p PAIRS [PAIRS ...], --pairs PAIRS [PAIRS ...]
                        将命令限制为这些交易对。交易对之间用空格分隔。
  --eps, --enable-position-stacking
                        允许多次购买同一交易对（仓位堆叠）。
  --enable-protections, --enableprotections
                        为回测启用保护机制。会显著减慢回测速度，但会包含已配置的保护措施。
  --dry-run-wallet DRY_RUN_WALLET, --starting-balance DRY_RUN_WALLET
                        起始资金，用于回测/超参数优化和模拟交易。
  --timeframe-detail TIMEFRAME_DETAIL
                        为回测指定详细时间框架（`1m`、`5m`、`30m`、`1h`、`1d`）。
  --strategy-list STRATEGY_LIST [STRATEGY_LIST ...]
                        提供以空格分隔的策略列表进行回测。请注意，时间框架需要在配置文件中设置
                        或通过命令行设置。当与`--export trades`一起使用时，策略名称会被注入
                        文件名中（因此`backtest-data.json`变为`backtest-data-SampleStrategy.json`）。
  --export {none,trades,signals}
                        导出回测结果（默认：trades）。
  --export-filename PATH, --backtest-filename PATH
                        使用此文件名保存回测结果。需要同时设置`--export`。
                        示例: `--export-filename=user_data/backtest_results/backtest_today.json`
  --breakdown {day,week,month,year} [{day,week,month,year} ...]
                        按[日、周、月、年]显示回测细目。
  --cache {none,day,week,month}
                        加载不超过指定时间的缓存回测结果（默认：day）。
  --freqai-backtest-live-models
                        使用实时模型运行回测。
  --notes TEXT          为回测结果添加备注。

通用参数:
  -v, --verbose         详细模式（-vv 更多，-vvv 获取所有消息）。
  --no-color            禁用超参数优化结果的颜色显示。如果将输出重定向到文件可能有用。
  --logfile FILE, --log-file FILE
                        日志输出到指定文件。特殊值：'syslog'，'journald'。
                        更多详情请参见文档。
  -V, --version         显示程序版本号并退出。
  -c PATH, --config PATH
                        指定配置文件（默认：`userdir/config.json` 或 `config.json`，以存在者为准）。
                        可以使用多个--config选项。可以设置为`-`从标准输入读取配置。
  -d PATH, --datadir PATH, --data-dir PATH
                        包含历史回测数据的交易所基础目录路径。要查看期货数据，
                        需额外使用交易模式（trading-mode）。
  --userdir PATH, --user-data-dir PATH
                        用户数据目录路径。

策略参数:
  -s NAME, --strategy NAME
                        指定机器人将要使用的策略类名。
  --strategy-path PATH  指定额外的策略查找路径。
  --recursive-strategy-search
                        在策略文件夹中递归搜索策略。
  --freqaimodel NAME    指定自定义的FreqAI模型。
  --freqaimodel-path PATH
                        为FreqAI模型指定额外的查找路径。