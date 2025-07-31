# pragma pylint: disable=missing-docstring, protected-access, invalid-name

import re
import shutil
from pathlib import Path

from freqtrade.commands.strategy_utils_commands import start_strategy_update
from freqtrade.strategy.strategyupdater import StrategyUpdater
from tests.conftest import get_args


def test_strategy_updater_start(user_dir, capsys) -> None:
    """测试策略更新器的启动和实际更新功能"""
    # 无模拟的有效测试
    teststrats = Path(__file__).parent / "strategy/strats"
    tmpdirp = Path(user_dir) / "strategies"
    tmpdirp.mkdir(parents=True, exist_ok=True)
    # 复制测试策略文件到临时目录
    shutil.copy(teststrats / "strategy_test_v2.py", tmpdirp)
    old_code = (teststrats / "strategy_test_v2.py").read_text()

    # 准备命令行参数
    args = ["strategy-updater", "--userdir", str(user_dir), "--strategy-list", "StrategyTestV2"]
    pargs = get_args(args)
    pargs["config"] = None

    # 执行策略更新
    start_strategy_update(pargs)

    # 验证备份目录是否创建
    assert Path(user_dir / "strategies_orig_updater").exists()
    # 验证备份文件是否存在
    assert Path(user_dir / "strategies_orig_updater" / "strategy_test_v2.py").exists()
    # 验证更新后的文件是否存在
    new_file = tmpdirp / "strategy_test_v2.py"
    assert new_file.exists()
    new_code = new_file.read_text()
    # 验证接口版本已更新
    assert "INTERFACE_VERSION = 3" in new_code
    assert "INTERFACE_VERSION = 2" in old_code
    # 验证输出日志
    captured = capsys.readouterr()
    assert "Conversion of strategy_test_v2.py started." in captured.out
    assert re.search(r"Conversion of strategy_test_v2\.py took .* seconds", captured.out)


def test_strategy_updater_methods(default_conf, caplog) -> None:
    """测试策略更新器对方法名称的更新"""
    instance_strategy_updater = StrategyUpdater()
    modified_code1 = instance_strategy_updater.update_code(
        """
class testClass(IStrategy):
    def populate_buy_trend():
        pass
    def populate_sell_trend():
        pass
    def check_buy_timeout():
        pass
    def check_sell_timeout():
        pass
    def custom_sell():
        pass
"""
    )

    # 验证方法名称已从buy/sell更新为entry/exit
    assert "populate_entry_trend" in modified_code1
    assert "populate_exit_trend" in modified_code1
    assert "check_entry_timeout" in modified_code1
    assert "check_exit_timeout" in modified_code1
    assert "custom_exit" in modified_code1
    # 验证接口版本已更新
    assert "INTERFACE_VERSION = 3" in modified_code1


def test_strategy_updater_params(default_conf, caplog) -> None:
    """测试策略更新器对参数名称的更新"""
    instance_strategy_updater = StrategyUpdater()

    modified_code2 = instance_strategy_updater.update_code(
        """
ticker_interval = '15m'
buy_some_parameter = IntParameter(space='buy')
sell_some_parameter = IntParameter(space='sell')
"""
    )

    # 验证ticker_interval已更新为timeframe
    assert "timeframe" in modified_code2
    # 验证超参数空间未被修改（buy/sell空间仍然有效）
    assert "space='buy'" in modified_code2
    assert "space='sell'" in modified_code2


def test_strategy_updater_constants(default_conf, caplog) -> None:
    """测试策略更新器对常量的更新"""
    instance_strategy_updater = StrategyUpdater()
    modified_code3 = instance_strategy_updater.update_code(
        """
use_sell_signal = True
sell_profit_only = True
sell_profit_offset = True
ignore_roi_if_buy_signal = True
forcebuy_enable = True
"""
    )

    # 验证常量名称已从sell/buy更新为exit/entry
    assert "use_exit_signal" in modified_code3
    assert "exit_profit_only" in modified_code3
    assert "exit_profit_offset" in modified_code3
    assert "ignore_roi_if_entry_signal" in modified_code3
    assert "force_entry_enable" in modified_code3


def test_strategy_updater_df_columns(default_conf, caplog) -> None:
    """测试策略更新器对DataFrame列名的更新"""
    instance_strategy_updater = StrategyUpdater()
    modified_code = instance_strategy_updater.update_code(
        """
dataframe.loc[reduce(lambda x, y: x & y, conditions), ["buy", "buy_tag"]] = (1, "buy_signal_1")
dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1
"""
    )

    # 验证DataFrame列名已从buy/sell更新为enter_long/exit_long
    assert "enter_long" in modified_code
    assert "exit_long" in modified_code
    assert "enter_tag" in modified_code


def test_strategy_updater_method_params(default_conf, caplog) -> None:
    """测试策略更新器对方法参数的更新"""
    instance_strategy_updater = StrategyUpdater()
    modified_code = instance_strategy_updater.update_code(
        """
def confirm_trade_exit(sell_reason: str):
    nr_orders = trade.nr_of_successful_buys
    pass
    """
    )
    # 验证参数名已从sell_reason更新为exit_reason
    assert "exit_reason" in modified_code
    # 验证属性名已从nr_of_successful_buys更新为nr_of_successful_entries
    assert "nr_orders = trade.nr_of_successful_entries" in modified_code


def test_strategy_updater_dicts(default_conf, caplog) -> None:
    """测试策略更新器对字典键的更新"""
    instance_strategy_updater = StrategyUpdater()
    modified_code = instance_strategy_updater.update_code(
        """
order_time_in_force = {
    'buy': 'gtc',
    'sell': 'ioc'
}
order_types = {
    'buy': 'limit',
    'sell': 'market',
    'stoploss': 'market',
    'stoploss_on_exchange': False
}
unfilledtimeout = {
    'buy': 1,
    'sell': 2
}
"""
    )

    # 验证字典键已从buy/sell更新为entry/exit
    assert "'entry': 'gtc'" in modified_code
    assert "'exit': 'ioc'" in modified_code
    assert "'entry': 'limit'" in modified_code
    assert "'exit': 'market'" in modified_code
    assert "'entry': 1" in modified_code
    assert "'exit': 2" in modified_code


def test_strategy_updater_comparisons(default_conf, caplog) -> None:
    """测试策略更新器对比较表达式的更新"""
    instance_strategy_updater = StrategyUpdater()
    modified_code = instance_strategy_updater.update_code(
        """
def confirm_trade_exit(sell_reason):
    if (sell_reason == 'stop_loss'):
        pass
"""
    )
    # 验证变量名已从sell_reason更新为exit_reason
    assert "exit_reason" in modified_code
    assert "exit_reason == 'stop_loss'" in modified_code


def test_strategy_updater_strings(default_conf, caplog) -> None:
    """测试策略更新器对字符串常量的更新"""
    instance_strategy_updater = StrategyUpdater()

    modified_code = instance_strategy_updater.update_code(
        """
sell_reason == 'sell_signal'
sell_reason == 'force_sell'
sell_reason == 'emergency_sell'
"""
    )

    # 验证字符串和变量名已更新
    assert "exit_signal" in modified_code
    assert "exit_reason" in modified_code
    assert "force_exit" in modified_code
    assert "emergency_exit" in modified_code


def test_strategy_updater_comments(default_conf, caplog) -> None:
    """测试策略更新器对注释的处理（不应修改注释）"""
    instance_strategy_updater = StrategyUpdater()
    modified_code = instance_strategy_updater.update_code(
        """
# This is the 1st comment
import talib.abstract as ta
# This is the 2nd comment
import freqtrade.vendor.qtpylib.indicators as qtpylib


class someStrategy(IStrategy):
    INTERFACE_VERSION = 2
    # This is the 3rd comment
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "0": 0.50
    }

    # This is the 4th comment
    stoploss = -0.1
"""
    )

    # 验证注释保持不变
    assert "This is the 1st comment" in modified_code
    assert "This is the 2nd comment" in modified_code
    assert "This is the 3rd comment" in modified_code
    # 验证接口版本已更新
    assert "INTERFACE_VERSION = 3" in modified_code
    # 以下内容目前尚未实现更新：
    # Webhook术语、Telegram通知设置、策略/配置设置