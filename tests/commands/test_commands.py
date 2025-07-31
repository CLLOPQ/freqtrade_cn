import json
import re
import shutil
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock
from zipfile import ZipFile

import pytest

from freqtrade.commands import (
    start_backtesting_show,
    start_convert_data,
    start_convert_db,
    start_convert_trades,
    start_create_userdir,
    start_download_data,
    start_edge,
    start_hyperopt_list,
    start_hyperopt_show,
    start_install_ui,
    start_list_data,
    start_list_exchanges,
    start_list_freqAI_models,
    start_list_hyperopt_loss_functions,
    start_list_markets,
    start_list_strategies,
    start_list_timeframes,
    start_new_strategy,
    start_show_config,
    start_show_trades,
    start_strategy_update,
    start_test_pairlist,
    start_trading,
    start_webserver,
)
from freqtrade.commands.deploy_ui import (
    clean_ui_subdir,
    download_and_install_ui,
    get_ui_download_url,
    read_ui_version,
)
from freqtrade.configuration import setup_utils_configuration
from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.persistence.models import init_db
from freqtrade.persistence.pairlock_middleware import PairLocks
from freqtrade.util import dt_utc
from tests.conftest import (
    CURRENT_TEST_STRATEGY,
    EXMS,
    create_mock_trades,
    get_args,
    log_has,
    log_has_re,
    patch_exchange,
    patched_configuration_load_config_file,
)
from tests.conftest_hyperopt import hyperopt_test_result
from tests.conftest_trades import MOCK_TRADE_COUNT


def test_setup_utils_configuration():
    """测试工具配置的设置"""
    args = [
        "list-exchanges",
        "--config",
        "tests/testdata/testconfigs/main_test_config.json",
    ]

    config = setup_utils_configuration(get_args(args), RunMode.OTHER)
    assert "exchange" in config
    assert config["dry_run"] is True

    args = [
        "list-exchanges",
        "--config",
        "tests/testdata/testconfigs/testconfig.json",
    ]

    config = setup_utils_configuration(get_args(args), RunMode.OTHER, set_dry=False)
    assert "exchange" in config
    assert config["dry_run"] is False


def test_start_trading_fail(mocker, caplog):
    """测试交易启动失败的情况"""
    mocker.patch("freqtrade.worker.Worker.run", MagicMock(side_effect=OperationalException))

    mocker.patch("freqtrade.worker.Worker.__init__", MagicMock(return_value=None))

    exitmock = mocker.patch("freqtrade.worker.Worker.exit", MagicMock())
    args = ["trade", "-c", "tests/testdata/testconfigs/main_test_test_config.json"]
    with pytest.raises(OperationalException):
        start_trading(get_args(args))
    assert exitmockmock.call_count == 1

    exitmock.reset_mock()
    caplog.clear()
    mocker.patch("freqtrade.worker.Worker.__init__", MagicMock(side_effect=OperationalException))
    with pytest.raises(OperationalException):
        start_trading(get_args(args))
    assert exitmock.call_count == 0


def test_start_webserver(mocker, caplog):
    """测试启动Web服务器"""
    api_server_mock = mocker.patch(
        "freqtrade.rpc.api_server.ApiServer",
    )

    args = ["webserver", "-c", "tests/testdata/testconfigs/main_test_test_config.json"]
    start_webserver(get_args(args))
    assert api_server_mock.call_count == 1


def test_list_exchanges(capsys):
    """测试列出交易市场列表"""
    args = [
        "list-exchanges",
    ]

    start_list_exchanges(get_args(args))
    captured = capsys.readouterr()
    assert re.search(r".*Freqtrade支持的交易市场.*", captured.out)
    assert re.search(r".*binance.*", captured.out)
    assert re.search(r".*bybit.*", captured.out)

    # 测试--one-column参数
    args = [
        "list-exchanges",
        "--one-column",
    ]

    start_list_exchanges(get_args(args))
    captured = capsys.readouterr()
    assert re.search(r"^binance$", captured.out, re.MULTILINE)
    assert re.search(r"^bybit$", captured.out, re.MULTILINE)

    # 测试--all参数
    args = [
        "list-exchanges",
        "--all",
    ]

    start_list_exchanges(get_args(args))
    captured = capsys.readouterr()
    assert re.search(r"ccxt库支持的所有交易市场.*", captured.out)
    assert re.search(r".*binance.*", captured.out)
    assert re.search(r".*bingx.*", captured.out)
    assert re.search(r".*bitmex.*", captured.out)

    # 测试--one-column --all组合参数
    args = [
        "list-exchanges",
        "--one-column",
        "--all",
    ]

    start_list_exchanges(get_args(args))
    captured = capsys.readouterr()
    assert re.search(r"^binance$", captured.out, re.MULTILINE)
    assert re.search(r"^bingx$", captured.out, re.MULTILINE)
    assert re.search(r"^bitmex$", captured.out, re.MULTILINE)


def test_list_timeframes(mocker, capsys):
    """测试列出时间周期列表"""
    api_mock = MagicMock()
    api_mock.timeframes = {
        "1m": "oneMin",
        "5m": "fiveMin",
        "30m": "thirtyMin",
        "1h": "hour",
        "1d": "day",
    }
    patch_exchange(mocker, api_mock=api_mock, exchange="bybit")
    args = [
        "list-timeframes",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    with pytest.raises(
        OperationalException, match=r"此命令需要配置交易市场.*"
    ):
        start_list_timeframes(pargs)

    # 测试带配置文件的情况
    args = [
        "list-timeframes",
        "--config",
        "tests/testdata/testconfigs/main_test_config.json",
    ]
    start_list_timeframes(get_args(args))
    captured = capsys.readouterr()
    assert re.match(
        "Bybit交易市场可用的时间周期: 1m, 5m, 30m, 1h, 1d", captured.out
    )

    # 测试--exchange参数
    args = [
        "list-timeframes",
        "--exchange",
        "bybit",
    ]
    start_list_timeframes(get_args(args))
    captured = capsys.readouterr()
    assert re.match(
        "Bybit交易市场可用的时间周期: 1m, 5m, 30m, 1h, 1d", captured.out
    )

    api_mock.timeframes = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "6h": "6h",
        "12h": "12h",
        "1d": "1d",
        "3d": "3d",
    }
    patch_exchange(mocker, api_mock=api_mock, exchange="binance")
    # 测试币安交易市场
    args = [
        "list-timeframes",
        "--exchange",
        "binance",
    ]
    start_list_timeframes(get_args(args))
    captured = capsys.readouterr()
    assert re.match(
        "Binance交易市场可用的时间周期: 1m, 5m, 15m, 30m, 1h, 6h, 12h, 1d, 3d",
        captured.out,
    )

    # 测试--one-column参数
    args = [
        "list-timeframes",
        "--config",
        "tests/testdata/testconfigs/main_test_test_config.json",
        "--one-column",
    ]
    start_list_timeframes(get_args(args))
    captured = capsys.readouterr()
    assert re.search(r"^1m$", captured.out, re.MULTILINE)
    assert re.search(r"^5m$", captured.out, re.MULTILINE)
    assert re.search(r"^1h$", captured.out, re.MULTILINE)
    assert re.search(r"^1d$", captured.out, re.MULTILINE)

    # 测试--exchange binance --one-column组合参数
    args = [
        "list-timeframes",
        "--exchange",
        "binance",
        "--one-column",
    ]
    start_list_timeframes(get_args(args))
    captured = capsys.readouterr()
    assert re.search(r"^1m$", captured.out, re.MULTILINE)
    assert re.search(r"^5m$", captured.out, re.MULTILINE)
    assert re.search(r"^1h$", captured.out, re.MULTILINE)
    assert re.search(r"^1d$", captured.out, re.MULTILINE)


def test_list_markets(mocker, markets_static, capsys):
    """测试列出交易市场"""
    api_mock = MagicMock()
    patch_exchange(mocker, api_mock=api_mock, exchange="binance", mock_markets=markets_static)

    # 测试没有--config参数的情况
    args = [
        "list-markets",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    with pytest.raises(
        OperationalException, match=r"此命令需要配置交易市场.*"
    ):
        start_list_markets(pargs, False)

    # 测试带配置文件的情况
    args = [
        "list-markets",
        "--config",
        "tests/testdata/testconfigs/main_test_config.json",
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert (
        "Binance交易市场有12个活跃交易对: "
        "ADA/USDT:USDT, BLK/BTC, ETH/BTC, ETH/USDT, ETH/USDT:USDT, LTC/BTC, "
        "LTC/ETH, LTC/USD, NEO/BTC, TKN/BTC, XLTCUSDT, XRP/BTC.\n" in captured.out
    )

    patch_exchange(mocker, api_mock=api_mock, exchange="binance", mock_markets=markets_static)
    # 测试--exchange参数
    args = ["list-markets", "--exchange", "binance"]
    pargs = get_args(args)
    pargs["config"] = None
    start_list_markets(pargs, False)
    captured = capsys.readouterr()
    assert re.search(r".*Binance交易市场有12个活跃交易对.*", captured.out)

    patch_exchange(mocker, api_mock=api_mock, exchange="binance", mock_markets=markets_static)
    # 测试--all参数: 所有交易对
    args = [
        "list-markets",
        "--all",
        "--config",
        "tests/testdata/testconfigs/main_test_config.json",
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert (
        "Binance交易市场有14个交易对: "
        "ADA/USDT:USDT, BLK/BTC, BTT/BTC, ETH/BTC, ETH/USDT, ETH/USDT:USDT, "
        "LTC/BTC, LTC/ETH, LTC/USD, LTC/USDT, NEO/BTC, TKN/BTC, XLTCUSDT, XRP/BTC.\n"
        in captured.out
    )

    # 测试list-pairs子命令: 活跃交易对
    args = [
        "list-pairs",
        "--config",
        "tests/testdata/testconfigs/main_test_config.json",
        "--print-list",
    ]
    start_list_markets(get_args(args), True)
    captured = capsys.readouterr()
    assert (
        "Binance交易市场有9个活跃交易对: "
        "BLK/BTC, ETH/BTC, ETH/USDT, LTC/BTC, LTC/ETH, LTC/USD, NEO/BTC, TKN/BTC, XRP/BTC.\n"
        in captured.out
    )

    # 测试list-pairs子命令带--all参数: 所有交易对
    args = [
        "list-pairs",
        "--all",
        "--config",
        "tests/testdata/testconfigs/main_test_config.json",
        "--print-list",
    ]
    start_list_markets(get_args(args), True)
    captured = capsys.readouterr()
    assert (
        "Binance交易市场有11个交易对: "
        "BLK/BTC, BTT/BTC, ETH/BTC, ETH/USDT, LTC/BTC, LTC/ETH, LTC/USD, LTC/USDT, NEO/BTC, "
        "TKN/BTC, XRP/BTC.\n" in captured.out
    )

    # 活跃市场, 基础货币=ETH, LTC
    args = [
        "list-markets",
        "--config",
        "tests/testdata/testconfigs/main_test_config.json",
        "--base",
        "ETH",
        "LTC",
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert (
        "Binance交易市场有7个以ETH, LTC为基础货币的活跃交易对: "
        "ETH/BTC, ETH/USDT, ETH/USDT:USDT, LTC/BTC, LTC/ETH, LTC/USD, XLTCUSDT.\n" in captured.out
    )

    # 活跃市场, 基础货币=LTC
    args = [
        "list-markets",
        "--config",
        "tests/testdata/testconfigs/main_test_config.json",
        "--base",
        "LTC",
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert (
        "Binance交易市场有4个以LTC为基础货币的活跃交易对: "
        "LTC/BTC, LTC/ETH, LTC/USD, XLTCUSDT.\n" in captured.out
    )

    # 活跃市场, 计价货币=USDT, USD
    args = [
        "list-markets",
        "--config",
        "tests/testdata/testconfigs/main_test_config.json",
        "--quote",
        "USDT",
        "USD",
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert (
        "Binance交易市场有5个以USDT, USD为计价货币的活跃交易对: "
        "ADA/USDT:USDT, ETH/USDT, ETH/USDT:USDT, LTC/USD, XLTCUSDT.\n" in captured.out
    )

    # 活跃市场, 计价货币=USDT
    args = [
        "list-markets",
        "--config",
        "tests/testdata/testconfigs/main_test_config.json",
        "--quote",
        "USDT",
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert (
        "Binance交易市场有4个以USDT为计价货币的活跃交易对: "
        "ADA/USDT:USDT, ETH/USDT, ETH/USDT:USDT, XLTCUSDT.\n" in captured.out
    )

    # 活跃市场, 基础货币=LTC, 计价货币=USDT
    args = [
        "list-markets",
        "--config",
        "tests/testdata/testconfigs/main_test_config.json",
        "--base",
        "LTC",
        "--quote",
        "USDT",
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert (
        "Binance交易市场有1个以LTC为基础货币且以USDT为计价货币的活跃交易对: "
        "XLTCUSDT.\n" in captured.out
    )

    # 活跃交易对, 基础货币=LTC, 计价货币=USD
    args = [
        "list-pairs",
        "--config",
        "tests/testdata/testconfigs/main_test_config.json",
        "--base",
        "LTC",
        "--quote",
        "USD",
        "--print-list",
    ]
    start_list_markets(get_args(args), True)
    captured = capsys.readouterr()
    assert (
        "Binance交易市场有1个以LTC为基础货币且以USD为计价货币的活跃交易对: "
        "LTC/USD.\n" in captured.out
    )

    # 活跃市场, 基础货币=LTC, 计价货币=USDT, NONEXISTENT
    args = [
        "list-markets",
        "--config",
        "tests/testdata/testconfigs/main_test_config.json",
        "--base",
        "LTC",
        "--quote",
        "USDT",
        "NONEXISTENT",
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert (
        "Binance交易市场有1个以LTC为基础货币且以USDT, NONEXISTENT为计价货币的活跃交易对: "
        "XLTCUSDT.\n" in captured.out
    )

    # 活跃市场, 基础货币=LTC, 计价货币=NONEXISTENT
    args = [
        "list-markets",
        "--config",
        "tests/testdata/testconfigs/main_test_config.json",
        "--base",
        "LTC",
        "--quote",
        "NONEXISTENT",
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert (
        "Binance交易市场有0个以LTC为基础货币且以NONEXISTENT为计价货币的活跃交易对.\n" in captured.out
    )

    # 测试表格输出
    args = [
        "list-markets",
        "--config",
        "tests/testdata/testconfigs/main_test_config.json",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert "Binance交易市场有12个活跃交易对" in captured.out

    # 测试表格输出, 未找到交易对
    args = [
        "list-markets",
        "--config",
        "tests/testdata/testconfigs/main_test_config.json",
        "--base",
        "LTC",
        "--quote",
        "NONEXISTENT",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert (
        "Binance交易市场有0个以LTC为基础货币且以NONEXISTENT为计价货币的活跃交易对.\n" in captured.out
    )

    # 测试--print-json参数
    args = [
        "list-markets",
        "--config",
        "tests/testdata/testconfigs/main_test_config.json",
        "--print-json",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert (
        '["ADA/USDT:USDT","BLK/BTC","ETH/BTC","ETH/USDT","ETH/USDT:USDT",'
        '"LTC/BTC","LTC/ETH","LTC/USD","NEO/BTC","TKN/BTC","XLTCUSDT","XRP/BTC"]' in captured.out
    )

    # 测试--print-csv参数
    args = [
        "list-markets",
        "--config",
        "tests/testdata/testconfigs/main_test_config.json",
        "--print-csv",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert "Id,Symbol,Base,Quote,Active,Spot,Margin,Future,Leverage" in captured.out
    assert "blkbtc,BLK/BTC,BLK,BTC,True,Spot" in captured.out
    assert "USD-LTC,LTC/USD,LTC,USD,True,Spot" in captured.out

    # 测试--one-column参数
    args = [
        "list-markets",
        "--config",
        "tests/testdata/testconfigs/main_test_config.json",
        "--one-column",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert re.search(r"^BLK/BTC$", captured.out, re.MULTILINE)
    assert re.search(r"^LTC/USD$", captured.out, re.MULTILINE)

    mocker.patch(f"{EXMS}.markets", PropertyMock(side_effect=ValueError))
    # 测试--one-column参数错误情况
    args = [
        "list-markets",
        "--config",
        "tests/testdata/testconfigs/main_test_config.json",
        "--one-column",
    ]
    with pytest.raises(OperationalException, match=r"无法获取交易对.*"):
        start_list_markets(get_args(args), False)


def test_create_datadir_failed(caplog):
    """测试创建数据目录失败的情况"""
    args = [
        "create-userdir",
    ]
    with pytest.raises(SystemExit):
        start_create_userdir(get_args(args))
    assert log_has("`create-userdir`需要设置--userdir参数。", caplog)


def test_create_datadir(mocker):
    """测试创建数据目录"""
    cud = mocker.patch(
        "freqtrade.configuration.directory_operations.create_userdata_dir", MagicMock()
    )
    csf = mocker.patch(
        "freqtrade.configuration.directory_operations.copy_sample_files", MagicMock()
    )
    args = ["create-userdir", "--userdir", "/temp/freqtrade/test"]
    start_create_userdir(get_args(args))

    assert cud.call_count == 1
    assert csf.call_count == 1


def test_start_new_strategy(caplog, user_dir):
    """测试创建新策略"""
    strategy_dir = user_dir / "strategies"
    strategy_dir.mkdir(parents=True, exist_ok=True)

    assert strategy_dir.is_dir()
    args = ["new-strategy", "--strategy", "CoolNewStrategy"]
    start_new_strategy(get_args(args))
    assert strategy_dir.exists()
    assert (strategy_dir / "CoolNewStrategy.py").exists()

    assert log_has_re("正在将策略写入.*", caplog)

    with pytest.raises(
        OperationalException, match=r".*已存在。请选择其他策略名称。"
    ):
        start_new_strategy(get_args(args))

    args = ["new-strategy", "--strategy", "CoolNewStrategy", "--strategy-path", str(user_dir)]
    start_new_strategy(get_args(args))
    assert (user_dir / "CoolNewStrategy.py").exists()

    # 测试不存在的策略路径
    args = [
        "new-strategy",
        "--strategy",
        "CoolNewStrategy",
        "--strategy-path",
        str(user_dir / "nonexistent"),
    ]
    start_new_strategy(get_args(args))
    assert (user_dir / "CoolNewStrategy.py").exists()

    assert log_has_re("正在创建策略目录.*", caplog)
    assert (user_dir / "nonexistent").is_dir()
    assert (user_dir / "nonexistent" / "CoolNewStrategy.py").exists()

    shutil.rmtree(str(user_dir))


def test_start_new_strategy_no_arg():
    """测试没有参数的新策略创建"""
    args = [
        "new-strategy",
    ]
    with pytest.raises(OperationalException, match="`new-strategy`需要设置--strategy参数。"):
        start_new_strategy(get_args(args))


def test_start_install_ui(mocker):
    """测试安装UI"""
    clean_mock = mocker.patch("freqtrade.commands.deploy_ui.clean_ui_subdir")
    get_url_mock = mocker.patch(
        "freqtrade.commands.deploy_ui.get_ui_download_url",
        return_value=("https://example.com/whatever", "0.0.1"),
    )
    download_mock = mocker.patch("freqtrade.commands.deploy_ui.download_and_install_ui")
    mocker.patch("freqtrade.commands.deploy_ui.read_ui_version", return_value=None)
    args = [
        "install-ui",
    ]
    start_install_ui(get_args(args))
    assert clean_mock.call_count == 1
    assert get_url_mock.call_count == 1
    assert download_mock.call_count == 1

    clean_mock.reset_mock()
    get_url_mock.reset_mock()
    download_mock.reset_mock()

    args = [
        "install-ui",
        "--erase",
    ]
    start_install_ui(get_args(args))
    assert clean_mock.call_count == 1
    assert get_url_mock.call_count == 1
    assert download_mock.call_count == 0


def test_clean_ui_subdir(mocker, tmp_path, caplog):
    """测试清理UI子目录"""
    mocker.patch("freqtrade.commands.deploy_ui.Path.is_dir", side_effect=[True, True])
    mocker.patch("freqtrade.commands.deploy_ui.Path.is_file", side_effect=[False, True])
    rd_mock = mocker.patch("freqtrade.commands.deploy_ui.Path.rmdir")
    ul_mock = mocker.patch("freqtrade.commands.deploy_ui.Path.unlink")

    mocker.patch(
        "freqtrade.commands.deploy_ui.Path.glob",
        return_value=[Path("test1"), Path("test2"), Path(".gitkeep")],
    )
    folder = tmp_path / "uitests"
    clean_ui_subdir(folder)
    assert log_has("正在删除UI目录内容。", caplog)
    assert rd_mock.call_count == 1
    assert ul_mock.call_count == 1


def test_download_and_install_ui(mocker, tmp_path):
    """测试下载并安装UI"""
    # 创建zip文件
    requests_mock = MagicMock()
    file_like_object = BytesIO()
    with ZipFile(file_like_object, mode="w") as zipfile:
        for file in ("test1.txt", "hello/", "test2.txt"):
            zipfile.writestr(file, file)
    file_like_object.seek(0)
    requests_mock.content = file_like_object.read()

    mocker.patch("freqtrade.commands.deploy_ui.requests.get", return_value=requests_mock)

    mocker.patch("freqtrade.commands.deploy_ui.Path.is_dir", side_effect=[True, False])
    wb_mock = mocker.patch("freqtrade.commands.deploy_ui.Path.write_bytes")

    folder = tmp_path / "uitests_dl"
    folder.mkdir(exist_ok=True)

    assert read_ui_version(folder) is None

    download_and_install_ui(folder, "http://whatever.xxx/download/file.zip", "22")

    assert wb_mock.call_count == 2

    assert read_ui_version(folder) == "22"


def test_get_ui_download_url(mocker):
    """测试获取UI下载URL"""
    response = MagicMock()
    responses = [
        [
            {
                # 预发布版本会被忽略
                "assets_url": "http://whatever.json",
                "name": "0.0.2",
                "created_at": "2024-02-01T00:00:00Z",
                "prerelease": True,
            },
            {
                "assets_url": "http://whatever.json",
                "name": "0.0.1",
                "created_at": "2024-01-01T00:00:00Z",
                "prerelease": False,
            },
        ],
        [{"browser_download_url": "http://download.zip"}],
    ]
    response.json = MagicMock(side_effect=responses)
    get_mock = mocker.patch("freqtrade.commands.deploy_ui.requests.get", return_value=response)
    x, last_version = get_ui_download_url(None, False)
    assert get_mock.call_count == 2
    assert last_version == "0.0.1"
    assert x == "http://download.zip"

    response.json = MagicMock(side_effect=responses)
    get_mock.reset_mock()
    x, last_version = get_ui_download_url(None, True)
    assert get_mock.call_count == 2
    assert last_version == "0.0.2"
    assert x == "http://download.zip"


def test_get_ui_download_url_direct(mocker):
    """测试直接获取UI下载URL"""
    response = MagicMock()
    response.json = MagicMock(
        return_value=[
            {
                "assets_url": "http://whatever.json",
                "name": "0.0.2",
                "created_at": "2024-02-01T00:00:00Z",
                "prerelease": False,
                "assets": [{"browser_download_url": "http://download22.zip"}],
            },
            {
                "assets_url": "http://whatever.json",
                "name": "0.0.1",
                "created_at": "2024-01-01T00:00:00Z",
                "prerelease": False,
                "assets": [{"browser_download_url": "http://download1.zip"}],
            },
        ]
    )
    get_mock = mocker.patch("freqtrade.commands.deploy_ui.requests.get", return_value=response)
    x, last_version = get_ui_download_url(None, False)
    assert get_mock.call_count == 1
    assert last_version == "0.0.2"
    assert x == "http://download22.zip"
    get_mock.reset_mock()
    response.json.reset_mock()

    x, last_version = get_ui_download_url("0.0.1", False)
    assert last_version == "0.0.1"
    assert x == "http://download1.zip"

    with pytest.raises(ValueError, match="未找到UI版本。"):
        x, last_version = get_ui_download_url("0.0.3", False)


def test_download_data_keyboardInterrupt(mocker, markets):
    """测试下载数据时的键盘中断"""
    dl_mock = mocker.patch(
        "freqtrade.data.history.download_data_main",
        MagicMock(side_effect=KeyboardInterrupt),
    )
    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets))
    args = [
        "download-data",
        "--exchange",
        "binance",
        "--pairs",
        "ETH/BTC",
        "XRP/BTC",
    ]
    with pytest.raises(SystemExit):
        pargs = get_args(args)
        pargs["config"] = None

        start_download_data(pargs)

    assert dl_mock.call_count == 1


@pytest.mark.parametrize("time", ["00:00", "00:03", "00:30", "23:56"])
@pytest.mark.parametrize(
    "tzoffset",
    ["00:00", "+01:00", "-01:00", "+05:00", "-05:00"],
)
def test_download_data_timerange(mocker, markets, time_machine, time, tzoffset):
    """测试下载数据的时间范围"""
    time_machine.move_to(f"2024-11-01 {time}:00 {tzoffset}")
    dl_mock = mocker.patch(
        "freqtrade.data.history.history_utils.refresh_backtest_ohlcv_data",
        MagicMock(return_value=["ETH/BTC", "XRP/BTC"]),
    )
    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets))
    args = [
        "download-data",
        "--exchange",
        "binance",
        "--pairs",
        "ETH/BTC",
        "XRP/BTC",
        "--days",
        "20",
        "--timerange",
        "20200101-",
    ]
    with pytest.raises(OperationalException, match=r"--days和--timerange参数不能同时使用.*"):
        pargs = get_args(args)
        pargs["config"] = None
        start_download_data(pargs)
    assert dl_mock.call_count == 0

    args = [
        "download-data",
        "--exchange",
        "binance",
        "--pairs",
        "ETH/BTC",
        "XRP/BTC",
        "--days",
        "20",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_download_data(pargs)
    assert dl_mock.call_count == 1
    # 20天前
    days_ago = datetime.now() - timedelta(days=20)
    days_ago = dt_utc(days_ago.year, days_ago.month, days_ago.day)
    assert dl_mock.call_args_list[0][1]["timerange"].startts == days_ago.timestamp()

    dl_mock.reset_mock()
    args = [
        "download-data",
        "--exchange",
        "binance",
        "--pairs",
        "ETH/BTC",
        "XRP/BTC",
        "--timerange",
        "20200101-",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_download_data(pargs)
    assert dl_mock.call_count == 1

    assert dl_mock.call_args_list[0][1]["timerange"].startts == int(dt_utc(2020, 1, 1).timestamp())


def test_download_data_no_exchange(mocker):
    """测试没有交易市场的情况下载数据"""
    mocker.patch(
        "freqtrade.data.history.history_utils.refresh_backtest_ohlcv_data",
        MagicMock(return_value=["ETH/BTC", "XRP/BTC"]),
    )
    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.get_markets", return_value={})
    args = [
        "download-data",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    with pytest.raises(
        OperationalException, match=r"此命令需要配置交易市场.*"
    ):
        start_download_data(pargs)


def test_download_data_no_pairs(mocker):
    """测试没有交易对的情况下载数据"""
    mocker.patch(
        "freqtrade.data.history.history_utils.refresh_backtest_ohlcv_data",
        MagicMock(return_value=["ETH/BTC", "XRP/BTC"]),
    )
    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value={}))
    args = [
        "download-data",
        "--exchange",
        "binance",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    with pytest.raises(
        OperationalException, match=r"下载数据需要交易对列表。.*"
    ):
        start_download_data(pargs)


def test_download_data_all_pairs(mocker, markets):
    """测试下载所有交易对数据"""
    dl_mock = mocker.patch(
        "freqtrade.data.history.history_utils.refresh_backtest_ohlcv_data",
        MagicMock(return_value=["ETH/BTC", "XRP/BTC"]),
    )
    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets))
    args = ["download-data", "--exchange", "binance", "--pairs", ".*/USDT"]
    pargs = get_args(args)
    pargs["config"] = None
    start_download_data(pargs)
    expected = set(["BTC/USDT", "ETH/USDT", "XRP/USDT", "NEO/USDT", "TKN/USDT"])
    assert set(dl_mock.call_args_list[0][1]["pairs"]) == expected
    assert dl_mock.call_count == 1

    dl_mock.reset_mock()
    args = [
        "download-data",
        "--exchange",
        "binance",
        "--pairs",
        ".*/USDT",
        "--include-inactive-pairs",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_download_data(pargs)
    expected = set(["BTC/USDT", "ETH/USDT", "LTC/USDT", "XRP/USDT", "NEO/USDT", "TKN/USDT"])
    assert set(dl_mock.call_args_list[0][1]["pairs"]) == expected


def test_download_data_trades(mocker):
    """测试下载交易数据"""
    dl_mock = mocker.patch(
        "freqtrade.data.history.history_utils.refresh_backtest_trades_data",
        MagicMock(return_value=[]),
    )
    convert_mock = mocker.patch(
        "freqtrade.data.history.history_utils.convert_trades_to_ohlcv", MagicMock(return_value=[])
    )
    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.get_markets", return_value={"ETH/BTC": {}, "XRP/BTC": {}})
    args = [
        "download-data",
        "--exchange",
        "kraken",
        "--pairs",
        "ETH/BTC",
        "XRP/BTC",
        "--days",
        "20",
        "--dl-trades",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_download_data(pargs)
    assert dl_mock.call_args[1]["timerange"].starttype == "date"
    assert dl_mock.call_count == 1
    assert convert_mock.call_count == 1
    args = [
        "download-data",
        "--exchange",
        "kraken",
        "--pairs",
        "ETH/BTC",
        "XRP/BTC",
        "--days",
        "20",
        "--trading-mode",
        "futures",
        "--dl-trades",
    ]


def test_download_data_data_invalid(mocker):
    """测试下载无效数据"""
    patch_exchange(mocker, exchange="kraken")
    mocker.patch(f"{EXMS}.get_markets", return_value={"ETH/BTC": {}, "XRP/BTC": {}})
    args = [
        "download-data",
        "--exchange",
        "kraken",
        "--pairs",
        "ETH/BTC",
        "XRP/BTC",
        "--days",
        "20",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    with pytest.raises(OperationalException, match=r"历史K线数据在.*不可用"):
        start_download_data(pargs)


def test_start_convert_trades(mocker):
    """测试转换交易数据"""
    convert_mock = mocker.patch(
        "freqtrade.data.converter.convert_trades_to_ohlcv", MagicMock(return_value=[])
    )
    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.get_markets")
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value={}))
    args = [
        "trades-to-ohlcv",
        "--exchange",
        "kraken",
        "--pairs",
        "ETH/BTC",
        "XRP/BTC",
    ]
    start_convert_trades(get_args(args))
    assert convert_mock.call_count == 1


def test_start_list_strategies(capsys):
    """测试列出策略"""
    args = [
        "list-strategies",
        "--strategy-path",
        str(Path(__file__).parent.parent / "strategy" / "strats"),
        "-1",
    ]
    pargs = get_args(args)
    # pargs['config'] = None
    start_list_strategies(pargs)
    captured = capsys.readouterr()
    assert "StrategyTestV2" in captured.out
    assert "strategy_test_v2.py" not in captured.out
    assert CURRENT_TEST_STRATEGY in captured.out

    # 测试常规输出
    args = [
        "list-strategies",
        "--strategy-path",
        str(Path(__file__).parent.parent / "strategy" / "strats"),
        "--no-color",
    ]
    pargs = get_args(args)
    # pargs['config'] = None
    start_list_strategies(pargs)
    captured = capsys.readouterr()
    assert "StrategyTestV2" in captured.out
    assert "strategy_test_v2.py" in captured.out
    assert CURRENT_TEST_STRATEGY in captured.out

    # 测试彩色输出
    args = [
        "list-strategies",
        "--strategy-path",
        str(Path(__file__).parent.parent / "strategy" / "strats"),
    ]
    pargs = get_args(args)
    # pargs['config'] = None
    start_list_strategies(pargs)
    captured = capsys.readouterr()
    assert "StrategyTestV2" in captured.out
    assert "strategy_test_v2.py" in captured.out
    assert CURRENT_TEST_STRATEGY in captured.out
    assert "加载失败" in captured.out
    # 递归搜索
    assert "TestStrategyNoImplements" not in captured.out

    # 测试递归搜索
    args = [
        "list-strategies",
        "--strategy-path",
        str(Path(__file__).parent.parent / "strategy" / "strats"),
        "--no-color",
        "--recursive-strategy-search",
    ]
    pargs = get_args(args)
    # pargs['config'] = None
    start_list_strategies(pargs)
    captured = capsys.readouterr()
    assert "StrategyTestV2" in captured.out
    assert "strategy_test_v2.py" in captured.out
    assert "StrategyTestV2" in captured.out
    assert "TestStrategyNoImplements" in captured.out
    assert str(Path("broken_strats/broken_futures_strategies.py")) in captured.out


def test_start_list_hyperopt_loss_functions(capsys):
    """测试列出超参数优化损失函数"""
    args = ["list-hyperoptloss", "-1"]
    pargs = get_args(args)
    pargs["config"] = None
    start_list_hyperopt_loss_functions(pargs)
    captured = capsys.readouterr()
    assert "CalmarHyperOptLoss" in captured.out
    assert "MaxDrawDownHyperOptLoss" in captured.out
    assert "SortinoHyperOptLossDaily" in captured.out
    assert "<builtin>/hyperopt_loss_sortino_daily.py" not in captured.out

    args = ["list-hyperoptloss"]
    pargs = get_args(args)
    pargs["config"] = None
    start_list_hyperopt_loss_functions(pargs)
    captured = capsys.readouterr()
    assert "CalmarHyperOptLoss" in captured.out
    assert "MaxDrawDownHyperOptLoss" in captured.out
    assert "SortinoHyperOptLossDaily" in captured.out
    assert "<builtin>/hyperopt_loss_sortino_daily.py" in captured.out


def test_start_list_freqAI_models(capsys):
    """测试列出FreqAI模型"""
    args = ["list-freqaimodels", "-1"]
    pargs = get_args(args)
    pargs["config"] = None
    start_list_freqAI_models(pargs)
    captured = capsys.readouterr()
    assert "LightGBMClassifier" in captured.out
    assert "LightGBMRegressor" in captured.out
    assert "XGBoostRegressor" in captured.out
    assert "<builtin>/LightGBMRegressor.py" not in captured.out

    args = [
        "list-freqaimodels",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_list_freqAI_models(pargs)
    captured = capsys.readouterr()
    assert "LightGBMClassifier" in captured.out
    assert "LightGBMRegressor" in captured.out
    assert "XGBoostRegressor" in captured.out
    assert "<builtin>/LightGBMRegressor.py" in captured.out


def test_start_test_pairlist(mocker, caplog, tickers, default_conf, capsys):
    """测试交易对列表"""
    patch_exchange(mocker, mock_markets=True)
    mocker.patch.multiple(
        EXMS,
        exchange_has=MagicMock(return_value=True),
        get_tickers=tickers,
    )

    default_conf["pairlists"] = [
        {
            "method": "VolumePairList",
            "number_assets": 5,
            "sort_key": "quoteVolume",
        },
        {"method": "PrecisionFilter"},
        {"method": "PriceFilter", "low_price_ratio": 0.02},
    ]

    patched_configuration_load_config_file(mocker, default_conf)
    args = ["test-pairlist", "-c", "tests/testdata/testconfigs/main_test_config.json"]

    start_test_pairlist(get_args(args))

    assert log_has_re(r"^使用解析后的交易对列表 VolumePairList.*", caplog)
    assert log_has_re(r"^使用解析后的交易对列表 PrecisionFilter.*", caplog)
    assert log_has_re(r"^使用解析后的交易对列表 PriceFilter.*", caplog)
    captured = capsys.readouterr()
    assert re.match(r".*的交易对", captured.out)
    assert re.match("['ETH/BTC', 'TKN/BTC', 'BLK/BTC', 'LTC/BTC', 'XRP/BTC']", captured.out)

    args = [
        "test-pairlist",
        "-c",
        "tests/testdata/testconfigs/main_test_config.json",
        "--one-column",
    ]
    start_test_pairlist(get_args(args))
    captured = capsys.readouterr()
    assert re.match(r"ETH/BTC\nTKN/BTC\nBLK/BTC\nLTC/BTC\nXRP/BTC\n", captured.out)

    args = [
        "test-pairlist",
        "-c",
        "tests/testdata/testconfigs/main_test_config.json",
        "--print-json",
    ]
    start_test_pairlist(get_args(args))
    captured = capsys.readouterr()
    try:
        json_pairs = json.loads(captured.out)
        assert "ETH/BTC" in json_pairs
        assert "TKN/BTC" in json_pairs
        assert "BLK/BTC" in json_pairs
        assert "LTC/BTC" in json_pairs
        assert "XRP/BTC" in json_pairs
    except json.decoder.JSONDecodeError:
        pytest.fail(f"预期格式正确的JSON，但解析失败: {captured.out}")


def test_hyperopt_list(mocker, capsys, caplog, tmp_path):
    """测试超参数优化列表"""
    saved_hyperopt_results = hyperopt_test_result()
    csv_file = tmp_path / "test.csv"
    mocker.patch(
        "freqtrade.optimize.hyperopt_tools.HyperoptTools._test_hyperopt_results_exist",
        return_value=True,
    )

    def fake_iterator(*args, **kwargs):
        yield from [saved_hyperopt_results]

    mocker.patch(
        "freqtrade.optimize.hyperopt_tools.HyperoptTools._read_results", side_effect=fake_iterator
    )

    args = [
        "hyperopt-list",
        "--no-details",
        "--no-color",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all(
        x in captured.out
        for x in [
            " 1/12",
            " 2/12",
            " 3/12",
            " 4/12",
            " 5/12",
            " 6/12",
            " 7/12",
            " 8/12",
            " 9/12",
            " 10/12",
            " 11/12",
            " 12/12",
        ]
    )
    args = [
        "hyperopt-list",
        "--best",
        "--no-details",
        "--no-color",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all(x in captured.out for x in [" 1/12", " 5/12", " 10/12"])
    assert all(
        x not in captured.out
        for x in [" 2/12", " 3/12", " 4/12", " 6/12", " 7/12", " 8/12", " 9/12", " 11/12", " 12/12"]
    )
    args = [
        "hyperopt-list",
        "--profitable",
        "--no-details",
        "--no-color",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all(x in captured.out for x in [" 2/12", " 10/12"])
    assert all(
        x not in captured.out
        for x in [
            " 1/12",
            " 3/12",
            " 4/12",
            " 5/12",
            " 6/12",
            " 7/12",
            " 8/12",
            " 9/12",
            " 11/12",
            " 12/12",
        ]
    )
    args = [
        "hyperopt-list",
        "--profitable",
        "--no-color",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all(
        x in captured.out
        for x in [
            " 2/12",
            " 10/12",
            "最佳结果:",
            "买入超参数",
            "卖出超参数",
            "ROI表",
            "止损",
        ]
    )
    assert all(
        x not in captured.out
        for x in [
            " 1/12",
            " 3/12",
            " 4/12",
            " 5/12",
            " 6/12",
            " 7/12",
            " 8/12",
            " 9/12",
            " 11/12",
            " 12/12",
        ]
    )
    args = [
        "hyperopt-list",
        "--no-details",
        "--no-color",
        "--min-trades",
        "20",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all(x in captured.out for x in [" 3/12", " 6/12", " 7/12", " 9/12", " 11/12"])
    assert all(
        x not in captured.out
        for x in [" 1/12", " 2/12", " 4/12", " 5/12", " 8/12", " 10/12", " 12/12"]
    )
    args = [
        "hyperopt-list",
        "--profitable",
        "--no-details",
        "--no-color",
        "--max-trades",
        "20",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all(x in captured.out for x in [" 2/12", " 10/12"])
    assert all(
        x not in captured.out
        for x in [
            " 1/12",
            " 3/12",
            " 4/12",
            " 5/12",
            " 6/12",
            " 7/12",
            " 8/12",
            " 9/12",
            " 11/12",
            " 12/12",
        ]
    )
    args = [
        "hyperopt-list",
        "--profitable",
        "--no-details",
        "--no-color",
        "--min-avg-profit",
        "0.11",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all(x in captured.out for x in [" 2/12"])
    assert all(
        x not in captured.out
        for x in [
            " 1/12",
            " 3/12",
            " 4/12",
            " 5/12",
            " 6/12",
            " 7/12",
            " 8/12",
            " 9/12",
            " 10/12",
            " 11/12",
            " 12/12",
        ]
    )
    args = [
        "hyperopt-list",
        "--no-details",
        "--no-color",
        "--max-avg-profit",
        "0.10",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all(
        x in captured.out
        for x in [" 1/12", " 3/12", " 5/12", " 6/12", " 7/12", " 8/12", " 9/12", " 11/12"]
    )
    assert all(x not in captured.out for x in [" 2/12", " 4/12", " 10/12", " 12/12"])
    args = [
        "hyperopt-list",
        "--no-details",
        "--no-color",
        "--min-total-profit",
        "0.4",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all(x in captured.out for x in [" 10/12"])
    assert all(
        x not in captured.out
        for x in [
            " 1/12",
            " 2/12",
            " 3/12",
            " 4/12",
            " 5/12",
            " 6/12",
            " 7/12",
            " 8/12",
            " 9/12",
            " 11/12",
            " 12/12",
        ]
    )
    args = [
        "hyperopt-list",
        "--no-details",
        "--no-color",
        "--max-total-profit",
        "0.4",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all(
        x in captured.out
        for x in [" 1/12", " 2/12", " 3/12", " 5/12", " 6/12", " 7/12", " 8/12", " 9/12", " 11/12"]
    )
    assert all(x not in captured.out for x in [" 4/12", " 10/12", " 12/12"])
    args = [
        "hyperopt-list",
        "--no-details",
        "--no-color",
        "--min-objective",
        "0.1",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all(x in captured.out for x in [" 10/12"])
    assert all(
        x not in captured.out
        for x in [
            " 1/12",
            " 2/12",
            " 3/12",
            " 4/12",
            " 5/12",
            " 6/12",
            " 7/12",
            " 8/12",
            " 9/12",
            " 11/12",
            " 12/12",
        ]
    )
    args = [
        "hyperopt-list",
        "--no-details",
        "--max-objective",
        "0.1",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all(
        x in captured.out
        for x in [" 1/12", " 2/12", " 3/12", " 5/12", " 6/12", " 7/12", " 8/12", " 9/12", " 11/12"]
    )
    assert all(x not in captured.out for x in [" 4/12", " 10/12", " 12/12"])
    args = [
        "hyperopt-list",
        "--profitable",
        "--no-details",
        "--no-color",
        "--min-avg-time",
        "2000",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all(x in captured.out for x in [" 10/12"])
    assert all(
        x not in captured.out
        for x in [
            " 1/12",
            " 2/12",
            " 3/12",
            " 4/12",
            " 5/12",
            " 6/12",
            " 7/12",
            " 8/12",
            " 9/12",
            " 11/12",
            " 12/12",
        ]
    )
    args = [
        "hyperopt-list",
        "--no-details",
        "--no-color",
        "--max-avg-time",
        "1500",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all(x in captured.out for x in [" 2/12", " 6/12"])
    assert all(
        x not in captured.out
        for x in [
            " 1/12",
            " 3/12",
            " 4/12",
            " 5/12",
            " 7/12",
            " 8/12",
            " 9/12",
            " 10/12",
            " 11/12",
            " 12/12",
        ]
    )
    args = [
        "hyperopt-list",
        "--no-details",
        "--no-color",
        "--export-csv",
        str(csv_file),
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    log_has("CSV文件已创建: test_file.csv", caplog)
    assert csv_file.is_file()
    line = csv_file.read_text()
    assert (
        'Best,1,2,-1.25%,-1.2222,-0.00125625,BTC,-2.51,"3,930.0 m",-0.00125625,23.00%,0.43662'
        in line
        or "Best,1,2,-1.25%,-1.2222,-0.00125625,BTC,-2.51,2 days 17:30:00,2,0,-0.00125625,23.00%,"
        "0.43662"
        in line
    )
    csv_file.unlink()


def test_hyperopt_show(mocker, capsys):
    """测试显示超参数优化结果"""
    saved_hyperopt_results = hyperopt_test_result()
    mocker.patch(
        "freqtrade.optimize.hyperopt_tools.HyperoptTools._test_hyperopt_results_exist",
        return_value=True,
    )

    def fake_iterator(*args, **kwargs):
        yield from [saved_hyperopt_results]

    mocker.patch(
        "freqtrade.optimize.hyperopt_tools.HyperoptTools._read_results", side_effect=fake_iterator
    )
    mocker.patch("freqtrade.optimize.optimize_reports.show_backtest_result")

    args = [
        "hyperopt-show",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_hyperopt_show(pargs)
    captured = capsys.readouterr()
    assert " 12/12" in captured.out

    args = ["hyperopt-show", "--best"]
    pargs = get_args(args)
    pargs["config"] = None
    start_hyperopt_show(pargs)
    captured = capsys.readouterr()
    assert " 10/12" in captured.out

    args = ["hyperopt-show", "-n", "1"]
    pargs = get_args(args)
    pargs["config"] = None
    start_hyperopt_show(pargs)
    captured = capsys.readouterr()
    assert " 1/12" in captured.out

    args = ["hyperopt-show", "--best", "-n", "2"]
    pargs = get_args(args)
    pargs["config"] = None
    start_hyperopt_show(pargs)
    captured = capsys.readouterr()
    assert " 5/12" in captured.out

    args = ["hyperopt-show", "--best", "-n", "-1"]
    pargs = get_args(args)
    pargs["config"] = None
    start_hyperopt_show(pargs)
    captured = capsys.readouterr()
    assert " 10/12" in captured.out

    args = ["hyperopt-show", "--best", "-n", "-4"]
    pargs = get_args(args)
    pargs["config"] = None
    with pytest.raises(
        OperationalException, match="要显示的周期索引应大于-4。"
    ):
        start_hyperopt_show(pargs)

    args = ["hyperopt-show", "--best", "-n", "4"]
    pargs = get_args(args)
    pargs["config"] = None
    with pytest.raises(
        OperationalException, match="要显示的周期索引应小于4。"
    ):
        start_hyperopt_show(pargs)


def test_convert_data(mocker, testdatadir):
    """测试转换数据格式"""
    ohlcv_mock = mocker.patch("freqtrade.data.converter.convert_ohlcv_format")
    trades_mock = mocker.patch("freqtrade.data.converter.convert_trades_format")
    args = [
        "convert-data",
        "--format-from",
        "json",
        "--format-to",
        "jsongz",
        "--datadir",
        str(testdatadir),
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_convert_data(pargs, True)
    assert trades_mock.call_count == 0
    assert ohlcv_mock.call_count == 1
    assert ohlcv_mock.call_args[1]["convert_from"] == "json"
    assert ohlcv_mock.call_args[1]["convert_to"] == "jsongz"
    assert ohlcv_mock.call_args[1]["erase"] is False


def test_convert_data_trades(mocker, testdatadir):
    """测试转换交易数据格式"""
    ohlcv_mock = mocker.patch("freqtrade.data.converter.convert_ohlcv_format")
    trades_mock = mocker.patch("freqtrade.data.converter.convert_trades_format")
    args = [
        "convert-trade-data",
        "--format-from",
        "jsongz",
        "--format-to",
        "json",
        "--datadir",
        str(testdatadir),
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_convert_data(pargs, False)
    assert ohlcv_mock.call_count == 0
    assert trades_mock.call_count == 1
    assert trades_mock.call_args[1]["convert_from"] == "jsongz"
    assert trades_mock.call_args[1]["convert_to"] == "json"
    assert trades_mock.call_args[1]["erase"] is False


def test_start_list_data(testdatadir, capsys):
    """测试列出数据"""
    args = [
        "list-data",
        "--datadir",
        str(testdatadir),
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_list_data(pargs)
    captured = capsys.readouterr()
    assert "找到16个交易对/时间周期组合。" in captured.out
    assert re.search(r".*交易对.*时间周期.*类型.*\n", captured.out)
    assert re.search(r"\n.* UNITTEST/BTC .* 1m, 5m, 8m, 30m .* spot |\n", captured.out)

    args = [
        "list-data",
        "--data-format-ohlcv",
        "feather",
        "--pairs",
        "XRP/ETH",
        "--datadir",
        str(testdatadir),
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_list_data(pargs)
    captured = capsys.readouterr()
    assert "找到2个交易对/时间周期组合。" in captured.out
    assert re.search(r".*交易对.*时间周期.*类型.*\n", captured.out)
    assert "UNITTEST/BTC" not in captured.out
    assert re.search(r"\n.* XRP/ETH .* 1m, 5m .* spot |\n", captured.out)

    args = [
        "list-data",
        "--trading-mode",
        "futures",
        "--datadir",
        str(testdatadir),
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_list_data(pargs)
    captured = capsys.readouterr()

    assert "找到6个交易对/时间周期组合。" in captured.out
    assert re.search(r".*交易对.*时间周期.*类型.*\n", captured.out)
    assert re.search(r"\n.* XRP/USDT:USDT .* 5m, 1h .* futures |\n", captured.out)
    assert re.search(r"\n.* XRP/USDT:USDT .* 1h, 8h .* mark |\n", captured.out)

    args = [
        "list-data",
        "--pairs",
        "XRP/ETH",
        "--datadir",
        str(testdatadir),
        "--show-timerange",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_list_data(pargs)
    captured = capsys.readouterr()
    assert "找到2个交易对/时间周期组合。" in captured.out
    assert re.search(r".*交易对.*时间周期.*类型.*起始 .* 结束 .* K线数量 .*\n", captured.out)
    assert "UNITTEST/BTC" not in captured.out
    assert re.search(
        r"\n.* XRP/USDT .* 1m .* spot .* 2019-10-11 00:00:00 .* 2019-10-13 11:19:00 .* 2469 |\n",
        captured.out,
    )


def test_start_list_trades_data(testdatadir, capsys):
    """测试列出交易数据"""
    args = [
        "list-data",
        "--datadir",
        str(testdatadir),
        "--trades",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_list_data(pargs)
    captured = capsys.readouterr()
    assert "找到1个交易对的交易数据。" in captured.out
    assert re.search(r".*交易对.*类型.*\n", captured.out)
    assert re.search(r"\n.* XRP/ETH .* spot |\n", captured.out)

    args = [
        "list-data",
        "--datadir",
        str(testdatadir),
        "--trades",
        "--show-timerange",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_list_data(pargs)
    captured = capsys.readouterr()
    assert "找到1个交易对的交易数据。" in captured.out
    assert re.search(r".*交易对.*类型.*起始.*结束.*交易数量.*\n", captured.out)
    assert re.search(
        r"\n.* XRP/ETH .* spot .* 2019-10-11 00:00:01 .* 2019-10-13 11:19:28 .* 12477 .*|\n",
        captured.out,
    )

    args = [
        "list-data",
        "--datadir",
        str(testdatadir),
        "--trades",
        "--trading-mode",
        "futures",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_list_data(pargs)
    captured = capsys.readouterr()
    assert "找到0个交易对的交易数据。" in captured.out


@pytest.mark.usefixtures("init_persistence")
def test_show_trades(mocker, fee, capsys, caplog):
    """测试显示交易记录"""
    mocker.patch("freqtrade.persistence.init_db")
    create_mock_trades(fee, False)
    args = ["show-trades", "--db-url", "sqlite:///"]
    pargs = get_args(args)
    pargs["config"] = None
    start_show_trades(pargs)
    assert log_has(f"正在显示 {MOCK_TRADE_COUNT} 条交易记录: ", caplog)
    captured = capsys.readouterr()
    assert "Trade(id=1" in captured.out
    assert "Trade(id=2" in captured.out
    assert "Trade(id=3" in captured.out
    args = ["show-trades", "--db-url", "sqlite:///", "--print-json", "--trade-ids", "1", "2"]
    pargs = get_args(args)
    pargs["config"] = None
    start_show_trades(pargs)

    captured = capsys.readouterr()
    assert log_has("正在显示 2 条交易记录: ", caplog)
    assert '"trade_id": 1' in captured.out
    assert '"trade_id": 2' in captured.out
    assert '"trade_id": 3' not in captured.out
    args = [
        "show-trades",
    ]
    pargs = get_args(args)
    pargs["config"] = None

    with pytest.raises(OperationalException, match=r"此命令需要--db-url参数。"):
        start_show_trades(pargs)


def test_backtesting_show(mocker, testdatadir, capsys):
    """测试显示回测结果"""
    sbr = mocker.patch("freqtrade.optimize.optimize_reports.show_backtest_results")
    args = [
        "backtesting-show",
        "--export-filename",
        f"{testdatadir / 'backtest_results/backtest-result.json'}",
        "--show-pair-list",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_backtesting_show(pargs)
    assert sbr.call_count == 1
    out, _err = capsys.readouterr()
    assert "策略的交易对" in out


def test_start_convert_db(fee, tmp_path):
    """测试转换数据库"""
    db_src_file = tmp_path / "db.sqlite"
    db_from = f"sqlite:///{db_src_file}"
    db_target_file = tmp_path / "db_target.sqlite"
    db_to = f"sqlite:///{db_target_file}"
    args = [
        "convert-db",
        "--db-url-from",
        db_from,
        "--db-url",
        db_to,
    ]

    assert not db_src_file.is_file()
    init_db(db_from)

    create_mock_trades(fee)

    PairLocks.timeframe = "5m"
    PairLocks.lock_pair("XRP/USDT", datetime.now(), "随机原因 125", side="long")
    assert db_src_file.is_file()
    assert not db_target_file.is_file()

    pargs = get_args(args)
    pargs["config"] = None
    start_convert_db(pargs)

    assert db_target_file.is_file()


def test_start_strategy_updater(mocker, tmp_path):
    """测试策略更新器"""
    sc_mock = mocker.patch("freqtrade.commands.strategy_utils_commands.start_conversion")
    teststrats = Path(__file__).parent.parent / "strategy/strats"
    args = [
        "strategy-updater",
        "--userdir",
        str(tmp_path),
        "--strategy-path",
        str(teststrats),
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_strategy_update(pargs)
    # 测试目录中策略的数量
    assert sc_mock.call_count == 12

    sc_mock.reset_mock()
    args = [
        "strategy-updater",
        "--userdir",
        str(tmp_path),
        "--strategy-path",
        str(teststrats),
        "--strategy-list",
        "StrategyTestV3",
        "StrategyTestV2",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    start_strategy_update(pargs)
    # 测试目录中策略的数量
    assert sc_mock.call_count == 2


def test_start_show_config(capsys, caplog):
    """测试显示配置"""
    args = [
        "show-config",
        "--config",
        "tests/testdata/testconfigs/main_test_config.json",
    ]
    pargs = get_args(args)
    start_show_config(pargs)

    captured = capsys.readouterr()
    assert "您的组合配置是：" in captured.out
    assert '"max_open_trades":' in captured.out
    assert '"secret": "已隐藏"' in captured.out

    args = [
        "show-config",
        "--config",
        "tests/testdata/testconfigs/main_test_config.json",
        "--show-sensitive",
    ]
    pargs = get_args(args)
    start_show_config(pargs)

    captured = capsys.readouterr()
    assert "您的组合配置是：" in captured.out
    assert '"max_open_trades":' in captured.out
    assert '"secret": "已隐藏"' not in captured.out
    assert log_has_re(r"即将输出的内容中将显示敏感信息.*", caplog)


def test_start_edge():
    """测试Edge模块（已弃用）"""
    args = [
        "edge",
        "--config",
        "tests/testdata/testconfigs/main_test_config.json",
    ]

    pargs = get_args(args)
    with pytest.raises(OperationalException, match="Edge模块已在2023.9版本中弃用"):
        start_edge(pargs)
