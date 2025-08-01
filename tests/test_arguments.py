# pragma pylint: disable=missing-docstring, C0103
import argparse
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from freqtrade.commands import Arguments, arguments
from freqtrade.commands.cli_options import (
    AVAILABLE_CLI_OPTIONS,
    check_int_nonzero,
    check_int_positive,
)
from tests.conftest import CURRENT_TEST_STRATEGY


def test_available_cli_options():
    """
    测试AVAILABLE_CLI_OPTIONS中的键是否是所有ARGS_*中值的并集 - CLI所必需的
    每个ARGS_*列表中的成员也应存在于AVAILABLE_CLI_OPTIONS中
    """
    args_union = {
        arg
        for variable, value in vars(arguments).items()
        if variable.startswith("ARGS_")
        for arg in value
    }
    expected_options = set(AVAILABLE_CLI_OPTIONS)
    only_in_command_args = expected_options.difference(args_union)
    only_in_all_args = args_union.difference(expected_options)
    if only_in_all_args or only_in_command_args:
        pytest.fail(
            "命令行参数相关变量未保持同步:\n"
            f" * 仅存在于某些ARGS_*列表但不在AVAILABLE_CLI_OPTIONS中的参数: {only_in_all_args}\n"
            " * 仅存在于AVAILABLE_CLI_OPTIONS但不在某些ARGS_*列表中的参数: "
            f"{only_in_command_args}"
        )


def test_arguments_match_available_cli_options(monkeypatch):
    """测试AVAILABLE_CLI_OPTIONS中的所有条目都在参数解析中使用"""
    parsed_options = set()
    actual_build_args = Arguments._build_args

    def build_args_monitor(self, optionlist, parser):
        parsed_options.update(optionlist)
        return actual_build_args(self, optionlist=optionlist, parser=parser)

    monkeypatch.setattr(Arguments, "_build_args", build_args_monitor)
    # 这将构建一个解析器，以便我们可以检查使用的参数
    Arguments([]).get_parsed_arg()
    assert parsed_options == set(AVAILABLE_CLI_OPTIONS)


# 解析通用命令行参数，所有工具都使用
def test_parse_args_none() -> None:
    arguments = Arguments(["trade"])
    assert isinstance(arguments, Arguments)
    x = arguments.get_parsed_arg()
    assert isinstance(x, dict)
    assert isinstance(arguments.parser, argparse.ArgumentParser)


def test_parse_args_defaults(mocker) -> None:
    mocker.patch.object(Path, "is_file", MagicMock(side_effect=[False, True]))
    args = Arguments(["trade"]).get_parsed_arg()
    assert args["config"] == ["config.json"]
    assert args["strategy_path"] is None
    assert args["datadir"] is None
    assert args["verbosity"] is None


def test_parse_args_default_userdatadir(mocker) -> None:
    mocker.patch.object(Path, "is_file", MagicMock(return_value=True))
    args = Arguments(["trade"]).get_parsed_arg()
    # 如果user_data可用，配置默认为user_data
    assert args["config"] == [str(Path("user_data/config.json"))]
    assert args["strategy_path"] is None
    assert args["datadir"] is None
    assert args["verbosity"] is None


def test_parse_args_userdatadir(mocker) -> None:
    mocker.patch.object(Path, "is_file", MagicMock(return_value=True))
    args = Arguments(["trade", "--user-data-dir", "user_data", "-v"]).get_parsed_arg()
    # 如果user_data可用，配置默认为user_data
    assert args["config"] == [str(Path("user_data/config.json"))]
    assert args["strategy_path"] is None
    assert args["datadir"] is None
    assert args["verbosity"] == 1


def test_parse_args_config() -> None:
    args = Arguments(["trade", "-c", "/dev/null"]).get_parsed_arg()
    assert args["config"] == ["/dev/null"]

    args = Arguments(["trade", "--config", "/dev/null"]).get_parsed_arg()
    assert args["config"] == ["/dev/null"]

    args = Arguments(
        ["trade", "--config", "/dev/null", "--config", "/dev/zero"],
    ).get_parsed_arg()
    assert args["config"] == ["/dev/null", "/dev/zero"]


def test_parse_args_db_url() -> None:
    args = Arguments(["trade", "--db-url", "sqlite:///test.sqlite"]).get_parsed_arg()
    assert args["db_url"] == "sqlite:///test.sqlite"


def test_parse_args_verbose() -> None:
    args = Arguments(["trade", "-v"]).get_parsed_arg()
    assert args["verbosity"] == 1

    args = Arguments(["trade", "--verbose"]).get_parsed_arg()
    assert args["verbosity"] == 1


def test_common_scripts_options() -> None:
    args = Arguments(["download-data", "-p", "ETH/BTC", "XRP/BTC"]).get_parsed_arg()

    assert args["pairs"] == ["ETH/BTC", "XRP/BTC"]
    assert "func" in args


def test_parse_args_version() -> None:
    args = Arguments(["--version"]).get_parsed_arg()
    assert args["version_main"] is True


def test_parse_args_invalid() -> None:
    with pytest.raises(SystemExit, match=r"2"):
        Arguments(["-c"]).get_parsed_arg()


def test_parse_args_strategy() -> None:
    args = Arguments(["trade", "--strategy", "SomeStrategy"]).get_parsed_arg()
    assert args["strategy"] == "SomeStrategy"


def test_parse_args_strategy_invalid() -> None:
    with pytest.raises(SystemExit, match=r"2"):
        Arguments(["--strategy"]).get_parsed_arg()


def test_parse_args_strategy_path() -> None:
    args = Arguments(["trade", "--strategy-path", "/some/path"]).get_parsed_arg()
    assert args["strategy_path"] == "/some/path"


def test_parse_args_strategy_path_invalid() -> None:
    with pytest.raises(SystemExit, match=r"2"):
        Arguments(["--strategy-path"]).get_parsed_arg()


def test_parse_args_backtesting_invalid() -> None:
    with pytest.raises(SystemExit, match=r"2"):
        Arguments(["backtesting --timeframe"]).get_parsed_arg()

    with pytest.raises(SystemExit, match=r"2"):
        Arguments(["backtesting --timeframe", "abc"]).get_parsed_arg()


def test_parse_args_backtesting_custom() -> None:
    args = [
        "backtesting",
        "-c",
        "test_conf.json",
        "--timeframe",
        "1m",
        "--strategy-list",
        CURRENT_TEST_STRATEGY,
        "SampleStrategy",
    ]
    call_args = Arguments(args).get_parsed_arg()
    assert call_args["config"] == ["test_conf.json"]
    assert call_args["verbosity"] is None
    assert call_args["command"] == "backtesting"
    assert call_args["func"] is not None
    assert call_args["timeframe"] == "1m"
    assert isinstance(call_args["strategy_list"], list)
    assert len(call_args["strategy_list"]) == 2


def test_parse_args_hyperopt_custom() -> None:
    args = ["hyperopt", "-c", "test_conf.json", "--epochs", "20", "--spaces", "buy"]
    call_args = Arguments(args).get_parsed_arg()
    assert call_args["config"] == ["test_conf.json"]
    assert call_args["epochs"] == 20
    assert call_args["verbosity"] is None
    assert call_args["command"] == "hyperopt"
    assert call_args["spaces"] == ["buy"]
    assert call_args["func"] is not None
    assert callable(call_args["func"])


def test_download_data_options() -> None:
    args = [
        "download-data",
        "--datadir",
        "datadir/directory",
        "--pairs-file",
        "file_with_pairs",
        "--days",
        "30",
        "--exchange",
        "binance",
    ]
    pargs = Arguments(args).get_parsed_arg()

    assert pargs["pairs_file"] == "file_with_pairs"
    assert pargs["datadir"] == "datadir/directory"
    assert pargs["days"] == 30
    assert pargs["exchange"] == "binance"


def test_plot_dataframe_options() -> None:
    args = [
        "plot-dataframe",
        "-c",
        "tests/testdata/testconfigs/main_test_config.json",
        "--indicators1",
        "sma10",
        "sma100",
        "--indicators2",
        "macd",
        "fastd",
        "fastk",
        "--plot-limit",
        "30",
        "-p",
        "UNITTEST/BTC",
    ]
    pargs = Arguments(args).get_parsed_arg()

    assert pargs["indicators1"] == ["sma10", "sma100"]
    assert pargs["indicators2"] == ["macd", "fastd", "fastk"]
    assert pargs["plot_limit"] == 30
    assert pargs["pairs"] == ["UNITTEST/BTC"]


@pytest.mark.parametrize("auto_open_arg", [True, False])
def test_plot_profit_options(auto_open_arg: bool) -> None:
    args = [
        "plot-profit",
        "-p",
        "UNITTEST/BTC",
        "--trade-source",
        "DB",
        "--db-url",
        "sqlite:///whatever.sqlite",
    ]
    if auto_open_arg:
        args.append("--auto-open")
    pargs = Arguments(args).get_parsed_arg()

    assert pargs["trade_source"] == "DB"
    assert pargs["pairs"] == ["UNITTEST/BTC"]
    assert pargs["db_url"] == "sqlite:///whatever.sqlite"
    assert pargs["plot_auto_open"] == auto_open_arg


def test_config_notallowed(mocker) -> None:
    mocker.patch.object(Path, "is_file", MagicMock(return_value=False))
    args = [
        "create-userdir",
    ]
    pargs = Arguments(args).get_parsed_arg()

    assert "config" not in pargs

    # 当文件存在时：
    mocker.patch.object(Path, "is_file", MagicMock(return_value=True))
    args = [
        "create-userdir",
    ]
    pargs = Arguments(args).get_parsed_arg()
    # 即使配置文件存在，create-userdir也不会添加config参数，因为它在不允许的列表中
    assert "config" not in pargs


def test_config_notrequired(mocker) -> None:
    mocker.patch.object(Path, "is_file", MagicMock(return_value=False))
    args = [
        "download-data",
    ]
    pargs = Arguments(args).get_parsed_arg()

    assert pargs["config"] is None

    # 当文件存在时：
    mocker.patch.object(Path, "is_file", MagicMock(side_effect=[False, True]))
    args = [
        "download-data",
    ]
    pargs = Arguments(args).get_parsed_arg()
    # 如果配置文件存在，则会添加config参数
    assert pargs["config"] == ["config.json"]


def test_check_int_positive() -> None:
    assert check_int_positive("3") == 3
    assert check_int_positive("1") == 1
    assert check_int_positive("100") == 100

    with pytest.raises(argparse.ArgumentTypeError):
        check_int_positive("-2")

    with pytest.raises(argparse.ArgumentTypeError):
        check_int_positive("0")

    with pytest.raises(argparse.ArgumentTypeError):
        check_int_positive(0)

    with pytest.raises(argparse.ArgumentTypeError):
        check_int_positive("3.5")

    with pytest.raises(argparse.ArgumentTypeError):
        check_int_positive("DeadBeef")


def test_check_int_nonzero() -> None:
    assert check_int_nonzero("3") == 3
    assert check_int_nonzero("1") == 1
    assert check_int_nonzero("100") == 100

    assert check_int_nonzero("-2") == -2

    with pytest.raises(argparse.ArgumentTypeError):
        check_int_nonzero("0")

    with pytest.raises(argparse.ArgumentTypeError):
        check_int_nonzero(0)

    with pytest.raises(argparse.ArgumentTypeError):
        check_int_nonzero("3.5")

    with pytest.raises(argparse.ArgumentTypeError):
        check_int_nonzero("DeadBeef")