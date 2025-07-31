# pragma pylint: disable=missing-docstring

import re
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

from freqtrade.commands import Arguments
from freqtrade.enums import State
from freqtrade.exceptions import ConfigurationError, FreqtradeException, OperationalException
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.main import main
from freqtrade.worker import Worker
from tests.conftest import (
    log_has,
    log_has_re,
    patch_exchange,
    patched_configuration_load_config_file,
)



def test_parse_args_None(caplog) -> None:
    """测试不带任何参数时的情况"""
    with pytest.raises(SystemExit):
        main([])
    assert log_has_re(r"使用Freqtrade需要一个子命令.*", caplog)


def test_parse_args_version(capsys) -> None:
    """测试版本信息输出"""
    with pytest.raises(SystemExit):
        main(["-V"])
    captured = capsys.readouterr()
    # 验证版本信息中包含CCXT版本
    assert re.search(r"CCXT Version:\s.*", captured.out, re.MULTILINE)
    # 验证版本信息中包含Freqtrade版本
    assert re.search(r"Freqtrade Version:\s+freqtrade\s.*", captured.out, re.MULTILINE)


def test_parse_args_backtesting(mocker) -> None:
    """
    测试main()能否启动回测，并确保可以传递特定参数
    进一步的参数解析在test_arguments.py中测试
    """
    mocker.patch.object(Path, "is_file", MagicMock(side_effect=[False, True]))
    backtesting_mock = mocker.patch("freqtrade.commands.start_backtesting")
    backtesting_mock.__name__ = PropertyMock("start_backtesting")
    # 回测结束时会调用sys.exit(0)
    with pytest.raises(SystemExit):
        main(["backtesting"])
    # 验证回测函数被调用
    assert backtesting_mock.call_count == 1
    call_args = backtesting_mock.call_args[0][0]
    # 验证参数是否正确传递
    assert call_args["config"] == ["config.json"]
    assert call_args["verbosity"] is None
    assert call_args["command"] == "backtesting"
    assert call_args["func"] is not None
    assert callable(call_args["func"])
    assert call_args["timeframe"] is None


def test_main_start_hyperopt(mocker) -> None:
    """测试启动超参数优化功能"""
    mocker.patch.object(Path, "is_file", MagicMock(side_effect=[False, True]))
    hyperopt_mock = mocker.patch("freqtrade.commands.start_hyperopt", MagicMock())
    hyperopt_mock.__name__ = PropertyMock("start_hyperopt")
    # 超参数优化结束时会调用sys.exit(0)
    with pytest.raises(SystemExit):
        main(["hyperopt"])
    # 验证超参数优化函数被调用
    assert hyperopt_mock.call_count == 1
    call_args = hyperopt_mock.call_args[0][0]
    # 验证参数是否正确传递
    assert call_args["config"] == ["config.json"]
    assert call_args["verbosity"] is None
    assert call_args["command"] == "hyperopt"
    assert call_args["func"] is not None
    assert callable(call_args["func"])


def test_main_fatal_exception(mocker, default_conf, caplog) -> None:
    """测试发生致命异常时的处理"""
    patch_exchange(mocker)
    mocker.patch("freqtrade.freqtradebot.FreqtradeBot.cleanup", MagicMock())
    mocker.patch("freqtrade.worker.Worker._worker", MagicMock(side_effect=Exception))
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch("freqtrade.freqtradebot.RPCManager", MagicMock())
    mocker.patch("freqtrade.freqtradebot.init_db", MagicMock())

    args = ["trade", "-c", "tests/testdata/testconfigs/main_test_config.json"]

    # 测试主程序和异常处理
    with pytest.raises(SystemExit):
        main(args)
    # 验证日志输出
    assert log_has("使用配置: tests/testdata/testconfigs/main_test_config.json ...", caplog)
    assert log_has("致命异常!", caplog)


def test_main_keyboard_interrupt(mocker, default_conf, caplog) -> None:
    """测试键盘中断(CTRL+C)的处理"""
    patch_exchange(mocker)
    mocker.patch("freqtrade.freqtradebot.FreqtradeBot.cleanup", MagicMock())
    mocker.patch("freqtrade.worker.Worker._worker", MagicMock(side_effect=KeyboardInterrupt))
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch("freqtrade.freqtradebot.RPCManager", MagicMock())
    mocker.patch("freqtrade.wallets.Wallets.update", MagicMock())
    mocker.patch("freqtrade.freqtradebot.init_db", MagicMock())

    args = ["trade", "-c", "tests/testdata/testconfigs/main_test_config.json"]

    # 测试主程序和键盘中断处理
    with pytest.raises(SystemExit):
        main(args)
    # 验证日志输出
    assert log_has("使用配置: tests/testdata/testconfigs/main_test_config.json ...", caplog)
    assert log_has("收到SIGINT信号，正在终止 ...", caplog)


def test_main_operational_exception(mocker, default_conf, caplog) -> None:
    """测试操作异常的处理"""
    patch_exchange(mocker)
    mocker.patch("freqtrade.freqtradebot.FreqtradeBot.cleanup", MagicMock())
    mocker.patch(
        "freqtrade.worker.Worker._worker", MagicMock(side_effect=FreqtradeException("出错了!"))
    )
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch("freqtrade.wallets.Wallets.update", MagicMock())
    mocker.patch("freqtrade.freqtradebot.RPCManager", MagicMock())
    mocker.patch("freqtrade.freqtradebot.init_db", MagicMock())

    args = ["trade", "-c", "tests/testdata/testconfigs/main_test_config.json"]

    # 测试主程序和异常处理
    with pytest.raises(SystemExit):
        main(args)
    # 验证日志输出
    assert log_has("使用配置: tests/testdata/testconfigs/main_test_config.json ...", caplog)
    assert log_has("出错了!", caplog)


def test_main_operational_exception1(mocker, default_conf, caplog) -> None:
    """测试另一种操作异常的处理情况"""
    patch_exchange(mocker)
    mocker.patch(
        "freqtrade.exchange.list_available_exchanges",
        MagicMock(side_effect=ValueError("出错了!")),
    )
    patched_configuration_load_config_file(mocker, default_conf)

    args = ["list-exchanges"]

    # 测试主程序和异常处理
    with pytest.raises(SystemExit):
        main(args)

    assert log_has("致命异常!", caplog)
    assert not log_has_re(r"SIGINT.*", caplog)
    
    # 测试键盘中断情况
    mocker.patch(
        "freqtrade.exchange.list_available_exchanges",
        MagicMock(side_effect=KeyboardInterrupt),
    )
    with pytest.raises(SystemExit):
        main(args)

    assert log_has_re(r"SIGINT.*", caplog)


def test_main_ConfigurationError(mocker, default_conf, caplog) -> None:
    """测试配置错误的处理"""
    patch_exchange(mocker)
    mocker.patch(
        "freqtrade.exchange.list_available_exchanges",
        MagicMock(side_effect=ConfigurationError("出错了!")),
    )
    patched_configuration_load_config_file(mocker, default_conf)

    args = ["list-exchanges"]

    # 测试主程序和配置错误处理
    with pytest.raises(SystemExit):
        main(args)
    assert log_has_re("配置错误: 出错了!", caplog)


def test_main_reload_config(mocker, default_conf, caplog) -> None:
    """测试重新加载配置的功能"""
    patch_exchange(mocker)
    mocker.patch("freqtrade.freqtradebot.FreqtradeBot.cleanup", MagicMock())
    # 模拟运行、重新加载、再运行的工作流程
    worker_mock = MagicMock(
        side_effect=[
            State.RUNNING,
            State.RELOAD_CONFIG,
            State.RUNNING,
            OperationalException("出错了!"),
        ]
    )
    mocker.patch("freqtrade.worker.Worker._worker", worker_mock)
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch("freqtrade.wallets.Wallets.update", MagicMock())
    reconfigure_mock = mocker.patch("freqtrade.worker.Worker._reconfigure", MagicMock())

    mocker.patch("freqtrade.freqtradebot.RPCManager", MagicMock())
    mocker.patch("freqtrade.freqtradebot.init_db", MagicMock())

    args = Arguments(
        ["trade", "-c", "tests/testdata/testconfigs/main_test_config.json"]
    ).get_parsed_arg()
    worker = Worker(args=args, config=default_conf)
    with pytest.raises(SystemExit):
        main(["trade", "-c", "tests/testdata/testconfigs/main_test_config.json"])

    # 验证日志和调用次数
    assert log_has("使用配置: tests/testdata/testconfigs/main_test_config.json ...", caplog)
    assert worker_mock.call_count == 4
    assert reconfigure_mock.call_count == 1
    assert isinstance(worker.freqtrade, FreqtradeBot)


def test_reconfigure(mocker, default_conf) -> None:
    """测试重新配置功能"""
    patch_exchange(mocker)
    mocker.patch("freqtrade.freqtradebot.FreqtradeBot.cleanup", MagicMock())
    mocker.patch(
        "freqtrade.worker.Worker._worker", MagicMock(side_effect=OperationalException("出错了!"))
    )
    mocker.patch("freqtrade.wallets.Wallets.update", MagicMock())
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch("freqtrade.freqtradebot.RPCManager", MagicMock())
    mocker.patch("freqtrade.freqtradebot.init_db", MagicMock())

    args = Arguments(
        ["trade", "-c", "tests/testdata/testconfigs/main_test_config.json"]
    ).get_parsed_arg()
    worker = Worker(args=args, config=default_conf)
    freqtrade = worker.freqtrade

    # 更新配置模拟以返回修改后的数据
    conf = deepcopy(default_conf)
    conf["stake_amount"] += 1
    patched_configuration_load_config_file(mocker, conf)

    worker._config = conf
    # 重新配置应该返回一个新实例
    worker._reconfigure()
    freqtrade2 = worker.freqtrade

    # 验证我们得到了一个带有新配置的新实例
    assert freqtrade is not freqtrade2
    assert freqtrade.config["stake_amount"] + 1 == freqtrade2.config["stake_amount"]