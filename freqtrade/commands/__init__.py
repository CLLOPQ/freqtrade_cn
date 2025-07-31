# flake8: noqa: F401
"""
命令模块。
包含所有启动命令、子命令和CLI界面创建。

注意：在这些子文件中要小心文件级别的导入。
    由于它们在启动时被解析，不应加载任何包含可选模块的内容。
"""

from freqtrade.commands.analyze_commands import start_analysis_entries_exits
from freqtrade.commands.arguments import Arguments
from freqtrade.commands.build_config_commands import start_new_config, start_show_config
from freqtrade.commands.data_commands import (
    start_convert_data,
    start_convert_trades,
    start_download_data,
    start_list_data,
    start_list_trades_data,
)
from freqtrade.commands.db_commands import start_convert_db
from freqtrade.commands.deploy_commands import (
    start_create_userdir,
    start_install_ui,
    start_new_strategy,
)
from freqtrade.commands.hyperopt_commands import start_hyperopt_list, start_hyperopt_show
from freqtrade.commands.list_commands import (
    start_list_exchanges,
    start_list_freqAI_models,
    start_list_hyperopt_loss_functions,
    start_list_markets,
    start_list_strategies,
    start_list_timeframes,
    start_show_trades,
)
from freqtrade.commands.optimize_commands import (
    start_backtesting,
    start_backtesting_show,
    start_edge,
    start_hyperopt,
    start_lookahead_analysis,
    start_recursive_analysis,
)
from freqtrade.commands.pairlist_commands import start_test_pairlist
from freqtrade.commands.plot_commands import start_plot_dataframe, start_plot_profit
from freqtrade.commands.strategy_utils_commands import start_strategy_update
from freqtrade.commands.trade_commands import start_trading
from freqtrade.commands.webserver_commands import start_webserver