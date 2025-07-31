usage: freqtrade lookahead-analysis [-h] [-v] [--no-color] [--logfile FILE]
                                    [-V] [-c PATH] [-d PATH] [--userdir PATH]
                                    [-s NAME] [--strategy-path PATH]
                                    [--recursive-strategy-search]
                                    [--freqaimodel NAME]
                                    [--freqaimodel-path PATH] [-i TIMEFRAME]
                                    [--timerange TIMERANGE]
                                    [--data-format-ohlcv {json,jsongz,feather,parquet}]
                                    [--max-open-trades INT]
                                    [--stake-amount STAKE_AMOUNT]
                                    [--fee FLOAT] [-p PAIRS [PAIRS ...]]
                                    [--enable-protections]
                                    [--dry-run-wallet DRY_RUN_WALLET]
                                    [--timeframe-detail TIMEFRAME_DETAIL]
                                    [--strategy-list STRATEGY_LIST [STRATEGY_LIST ...]]
                                    [--export {none,trades,signals}]
                                    [--export-filename PATH]
                                    [--freqai-backtest-live-models]
                                    [--minimum-trade-amount INT]
                                    [--targeted-trade-amount INT]
                                    [--lookahead-analysis-exportfilename LOOKAHEAD_ANALYSIS_EXPORTFILENAME]

选项:
  -h, --help            显示此帮助消息并退出
  -i TIMEFRAME, --timeframe TIMEFRAME
                        指定时间周期（`1m`、`5m`、`30m`、`1h`、`1d`）。
  --timerange TIMERANGE
                        指定要使用的数据时间范围。
  --data-format-ohlcv {json,jsongz,feather,parquet}
                        下载的K线（OHLCV）数据的存储格式。（默认：`feather`）。
  --max-open-trades INT
                        覆盖配置设置中的`max_open_trades`值。
  --stake-amount STAKE_AMOUNT
                        覆盖配置设置中的`stake_amount`值。
  --fee FLOAT           指定费率。将应用两次（在交易入场和出场时）。
  -p PAIRS [PAIRS ...], --pairs PAIRS [PAIRS ...]
                        将命令限制为这些交易对。交易对之间用空格分隔。
  --enable-protections, --enableprotections
                        为回测启用保护机制。会显著减慢回测速度，但会包含已配置的保护机制
  --dry-run-wallet DRY_RUN_WALLET, --starting-balance DRY_RUN_WALLET
                        初始余额，用于回测/超参数优化和模拟交易。
  --timeframe-detail TIMEFRAME_DETAIL
                        指定回测的详细时间周期（`1m`、`5m`、`30m`、`1h`、`1d`）。
  --strategy-list STRATEGY_LIST [STRATEGY_LIST ...]
                        提供以空格分隔的策略列表进行回测。请注意，时间周期需要在配置文件中或通过命令行设置。当与`--export trades`一起使用时，策略名称会被注入到文件名中（例如`backtest-data.json`会变为`backtest-data-SampleStrategy.json`）
  --export {none,trades,signals}
                        导出回测结果（默认：trades）。
  --export-filename PATH, --backtest-filename PATH
                        使用此文件名保存回测结果。需要同时设置`--export`。示例：`--export-filename=user_data/backtest_results/backtest_today.json`
  --freqai-backtest-live-models
                        使用已准备好的模型运行回测。
  --minimum-trade-amount INT
                        前瞻分析的最小交易数量
  --targeted-trade-amount INT
                        前瞻分析的目标交易数量
  --lookahead-analysis-exportfilename LOOKAHEAD_ANALYSIS_EXPORTFILENAME
                        使用此CSV文件名存储前瞻分析结果

通用参数:
  -v, --verbose         详细模式（-vv获取更多信息，-vvv获取所有消息）。
  --no-color            禁用超参数优化结果的彩色显示。如果将输出重定向到文件，这可能很有用。
  --logfile FILE, --log-file FILE
                        记录日志到指定文件。特殊值有：'syslog'、'journald'。详见文档了解更多详情。
  -V, --version         显示程序版本号并退出
  -c PATH, --config PATH
                        指定配置文件（默认：`userdir/config.json` 或 `config.json`，以存在者为准）。可使用多个--config选项。可设置为`-`从标准输入读取配置。
  -d PATH, --datadir PATH, --data-dir PATH
                        包含交易所历史回测数据的基础目录路径。要查看期货数据，需额外使用交易模式。
  --userdir PATH, --user-data-dir PATH
                        用户数据目录路径。

策略参数:
  -s NAME, --strategy NAME
                        指定机器人将使用的策略类名。
  --strategy-path PATH  指定额外的策略查找路径。
  --recursive-strategy-search
                        在策略文件夹中递归搜索策略。
  --freqaimodel NAME    指定自定义的freqaimodel。
  --freqaimodel-path PATH
                        指定freqaimodel的额外查找路径。