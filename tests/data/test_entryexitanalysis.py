import logging
from unittest.mock import MagicMock, PropertyMock

import pandas as pd
import pytest

from freqtrade.commands.analyze_commands import start_analysis_entries_exits
from freqtrade.commands.optimize_commands import start_backtesting
from freqtrade.enums import ExitType
from freqtrade.exceptions import OperationalException
from freqtrade.optimize.backtesting import Backtesting
from tests.conftest import get_args, patch_exchange, patched_configuration_load_config_file


@pytest.fixture(autouse=True)
def entryexitanalysis_cleanup() -> None:
    yield None

    Backtesting.cleanup()


def test_backtest_analysis_on_entry_and_rejected_signals_nomock(
    default_conf, mocker, caplog, testdatadir, user_dir, capsys
):
    caplog.set_level(logging.INFO)
    (user_dir / "backtest_results").mkdir(parents=True, exist_ok=True)

    default_conf.update(
        {
            "use_exit_signal": True,
            "exit_profit_only": False,
            "exit_profit_offset": 0.0,
            "ignore_roi_if_entry_signal": False,
        }
    )
    patch_exchange(mocker)
    result1 = pd.DataFrame(
        {
            "pair": ["ETH/BTC", "LTC/BTC", "ETH/BTC", "LTC/BTC"],
            "profit_ratio": [0.025, 0.05, -0.1, -0.05],
            "profit_abs": [0.5, 2.0, -4.0, -2.0],
            "open_date": pd.to_datetime(
                [
                    "2018-01-29 18:40:00",
                    "2018-01-30 03:30:00",
                    "2018-01-30 08:10:00",
                    "2018-01-31 13:30:00",
                ],
                utc=True,
            ),
            "close_date": pd.to_datetime(
                [
                    "2018-01-29 20:45:00",
                    "2018-01-30 05:35:00",
                    "2018-01-30 09:10:00",
                    "2018-01-31 15:00:00",
                ],
                utc=True,
            ),
            "trade_duration": [235, 40, 60, 90],
            "is_open": [False, False, False, False],
            "stake_amount": [0.01, 0.01, 0.01, 0.01],
            "open_rate": [0.104445, 0.10302485, 0.10302485, 0.10302485],
            "close_rate": [0.104969, 0.103541, 0.102041, 0.102541],
            "is_short": [False, False, False, False],
            "enter_tag": [
                "enter_tag_long_a",
                "enter_tag_long_b",
                "enter_tag_long_a",
                "enter_tag_long_b",
            ],
            "exit_reason": [
                ExitType.ROI.value,
                ExitType.EXIT_SIGNAL.value,
                ExitType.STOP_LOSS.value,
                ExitType.TRAILING_STOP_LOSS.value,
            ],
        }
    )

    backtestmock = MagicMock(
        side_effect=[
            {
                "results": result1,
                "config": default_conf,
                "locks": [],
                "rejected_signals": 20,
                "timedout_entry_orders": 0,
                "timedout_exit_orders": 0,
                "canceled_trade_entries": 0,
                "canceled_entry_orders": 0,
                "replaced_entry_orders": 0,
                "final_balance": 1000,
            }
        ]
    )
    mocker.patch(
        "freqtrade.plugins.pairlistmanager.PairListManager.whitelist",
        PropertyMock(return_value=["ETH/BTC", "LTC/BTC", "DASH/BTC"]),
    )
    mocker.patch("freqtrade.optimize.backtesting.Backtesting.backtest", backtestmock)

    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        "backtesting",
        "--config",
        "config.json",
        "--datadir",
        str(testdatadir),
        "--user-data-dir",
        str(user_dir),
        "--timeframe",
        "5m",
        "--timerange",
        "1515560100-1517287800",
        "--export",
        "signals",
        "--cache",
        "none",
    ]
    args = get_args(args)
    start_backtesting(args)

    captured = capsys.readouterr()
    assert "回测报告" in captured.out
    assert "出场原因统计" in captured.out
    assert "未平仓交易报告" in captured.out

    base_args = [
        "backtesting-analysis",
        "--config",
        "config.json",
        "--datadir",
        str(testdatadir),
        "--user-data-dir",
        str(user_dir),
    ]

    # 测试分组0和指标列表
    args = get_args(
        [*base_args, "--analysis-groups", "0", "--indicator-list", "close", "rsi", "profit_abs"]
    )
    start_analysis_entries_exits(args)
    captured = capsys.readouterr()
    assert "LTC/BTC" in captured.out
    assert "ETH/BTC" in captured.out
    assert "enter_tag_long_a" in captured.out
    assert "enter_tag_long_b" in captured.out
    assert "exit_signal" in captured.out
    assert "roi" in captured.out
    assert "stop_loss" in captured.out
    assert "trailing_stop_loss" in captured.out
    assert "0.5" in captured.out
    assert "-4" in captured.out
    assert "-2" in captured.out
    assert "-3.5" in captured.out
    assert "50" in captured.out
    assert "0" in captured.out
    assert "0.016" in captured.out
    assert "34.049" in captured.out
    assert "0.104" in captured.out
    assert "52.829" in captured.out
    # 验证指标列表
    assert "close (进场)" in captured.out
    assert "0.016" in captured.out
    assert "rsi (进场)" in captured.out
    assert "54.320" in captured.out
    assert "close (出场)" in captured.out
    assert "rsi (出场)" in captured.out
    assert "52.829" in captured.out
    assert "profit_abs" in captured.out

    # 测试分组1
    args = get_args([*base_args, "--analysis-groups", "1"])
    start_analysis_entries_exits(args)
    captured = capsys.readouterr()
    assert "enter_tag_long_a" in captured.out
    assert "enter_tag_long_b" in captured.out
    assert "总利润百分比" in captured.out
    assert "-3.5" in captured.out
    assert "-1.75" in captured.out
    assert "-7.5" in captured.out
    assert "-3.75" in captured.out
    assert "0" in captured.out

    # 测试分组2
    args = get_args([*base_args, "--analysis-groups", "2"])
    start_analysis_entries_exits(args)
    captured = capsys.readouterr()
    assert "enter_tag_long_a" in captured.out
    assert "enter_tag_long_b" in captured.out
    assert "exit_signal" in captured.out
    assert "roi" in captured.out
    assert "stop_loss" in captured.out
    assert "trailing_stop_loss" in captured.out
    assert "总利润百分比" in captured.out
    assert "-10" in captured.out
    assert "-5" in captured.out
    assert "2.5" in captured.out

    # 测试分组3
    args = get_args([*base_args, "--analysis-groups", "3"])
    start_analysis_entries_exits(args)
    captured = capsys.readouterr()
    assert "LTC/BTC" in captured.out
    assert "ETH/BTC" in captured.out
    assert "enter_tag_long_a" in captured.out
    assert "enter_tag_long_b" in captured.out
    assert "总利润百分比" in captured.out
    assert "-7.5" in captured.out
    assert "-3.75" in captured.out
    assert "-1.75" in captured.out
    assert "0" in captured.out
    assert "2" in captured.out

    # 测试分组4
    args = get_args([*base_args, "--analysis-groups", "4"])
    start_analysis_entries_exits(args)
    captured = capsys.readouterr()
    assert "LTC/BTC" in captured.out
    assert "ETH/BTC" in captured.out
    assert "enter_tag_long_a" in captured.out
    assert "enter_tag_long_b" in captured.out
    assert "exit_signal" in captured.out
    assert "roi" in captured.out
    assert "stop_loss" in captured.out
    assert "trailing_stop_loss" in captured.out
    assert "总利润百分比" in captured.out
    assert "-10" in captured.out
    assert "-5" in captured.out
    assert "-4" in captured.out
    assert "0.5" in captured.out
    assert "1" in captured.out
    assert "2.5" in captured.out

    # 测试分组5
    args = get_args([*base_args, "--analysis-groups", "5"])
    start_analysis_entries_exits(args)
    captured = capsys.readouterr()
    assert "exit_signal" in captured.out
    assert "roi" in captured.out
    assert "stop_loss" in captured.out
    assert "trailing_stop_loss" in captured.out

    # 测试日期过滤
    args = get_args(
        [*base_args, "--analysis-groups", "0", "1", "2", "--timerange", "20180129-20180130"]
    )
    start_analysis_entries_exits(args)
    captured = capsys.readouterr()
    assert "enter_tag_long_a" in captured.out
    assert "enter_tag_long_b" not in captured.out

    # 由于回测模拟，没有生成被拒绝的信号
    args = get_args([*base_args, "--rejected-signals"])
    start_analysis_entries_exits(args)
    captured = capsys.readouterr()
    assert "没有被拒绝的信号" in captured.out


def test_backtest_analysis_with_invalid_config(
    default_conf, mocker, caplog, testdatadir, user_dir, capsys
):
    caplog.set_level(logging.INFO)
    (user_dir / "backtest_results").mkdir(parents=True, exist_ok=True)

    default_conf.update(
        {
            "use_exit_signal": True,
            "exit_profit_only": False,
            "exit_profit_offset": 0.0,
            "ignore_roi_if_entry_signal": False,
        }
    )
    patch_exchange(mocker)
    result1 = pd.DataFrame(
        {
            "pair": ["ETH/BTC", "LTC/BTC", "ETH/BTC", "LTC/BTC"],
            "profit_ratio": [0.025, 0.05, -0.1, -0.05],
            "profit_abs": [0.5, 2.0, -4.0, -2.0],
            "open_date": pd.to_datetime(
                [
                    "2018-01-29 18:40:00",
                    "2018-01-30 03:30:00",
                    "2018-01-30 08:10:00",
                    "2018-01-31 13:30:00",
                ],
                utc=True,
            ),
            "close_date": pd.to_datetime(
                [
                    "2018-01-29 20:45:00",
                    "2018-01-30 05:35:00",
                    "2018-01-30 09:10:00",
                    "2018-01-31 15:00:00",
                ],
                utc=True,
            ),
            "trade_duration": [235, 40, 60, 90],
            "is_open": [False, False, False, False],
            "stake_amount": [0.01, 0.01, 0.01, 0.01],
            "open_rate": [0.104445, 0.10302485, 0.10302485, 0.10302485],
            "close_rate": [0.104969, 0.103541, 0.102041, 0.102541],
            "is_short": [False, False, False, False],
            "enter_tag": [
                "enter_tag_long_a",
                "enter_tag_long_b",
                "enter_tag_long_a",
                "enter_tag_long_b",
            ],
            "exit_reason": [
                ExitType.ROI.value,
                ExitType.EXIT_SIGNAL.value,
                ExitType.STOP_LOSS.value,
                ExitType.TRAILING_STOP_LOSS.value,
            ],
        }
    )

    backtestmock = MagicMock(
        side_effect=[
            {
                "results": result1,
                "config": default_conf,
                "locks": [],
                "rejected_signals": 20,
                "timedout_entry_orders": 0,
                "timedout_exit_orders": 0,
                "canceled_trade_entries": 0,
                "canceled_entry_orders": 0,
                "replaced_entry_orders": 0,
                "final_balance": 1000,
            }
        ]
    )
    mocker.patch(
        "freqtrade.plugins.pairlistmanager.PairListManager.whitelist",
        PropertyMock(return_value=["ETH/BTC", "LTC/BTC", "DASH/BTC"]),
    )
    mocker.patch("freqtrade.optimize.backtesting.Backtesting.backtest", backtestmock)

    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        "backtesting",
        "--config",
        "config.json",
        "--datadir",
        str(testdatadir),
        "--user-data-dir",
        str(user_dir),
        "--timeframe",
        "5m",
        "--timerange",
        "1515560100-1517287800",
        "--export",
        "signals",
        "--cache",
        "none",
    ]
    args = get_args(args)
    start_backtesting(args)

    captured = capsys.readouterr()
    assert "回测报告" in captured.out
    assert "出场原因统计" in captured.out
    assert "未平仓交易报告" in captured.out

    base_args = [
        "backtesting-analysis",
        "--config",
        "config.json",
        "--datadir",
        str(testdatadir),
        "--user-data-dir",
        str(user_dir),
    ]

    # 测试同时使用进场和出场仅有的参数
    args = get_args(
        [
            *base_args,
            "--analysis-groups",
            "0",
            "--indicator-list",
            "close",
            "rsi",
            "profit_abs",
            "--entry-only",
            "--exit-only",
        ]
    )
    with pytest.raises(
        OperationalException,
        match=r"不能同时使用--entry-only和--exit-only。请选择其中一个。",
    ):
        start_analysis_entries_exits(args)


def test_backtest_analysis_on_entry_and_rejected_signals_only_entry_signals(
    default_conf, mocker, caplog, testdatadir, user_dir, capsys
):
    caplog.set_level(logging.INFO)
    (user_dir / "backtest_results").mkdir(parents=True, exist_ok=True)

    default_conf.update(
        {
            "use_exit_signal": True,
            "exit_profit_only": False,
            "exit_profit_offset": 0.0,
            "ignore_roi_if_entry_signal": False,
        }
    )
    patch_exchange(mocker)