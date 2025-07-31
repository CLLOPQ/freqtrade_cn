# pragma pylint: disable=missing-docstring,W0212,C0103
from datetime import datetime, timedelta
from functools import partial, wraps
from pathlib import Path
from unittest.mock import ANY, MagicMock, PropertyMock

import pandas as pd
import pytest
from filelock import Timeout

from freqtrade.commands.optimize_commands import setup_optimize_configuration, start_hyperopt
from freqtrade.data.history import load_data
from freqtrade.enums import ExitType, RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.optimize.hyperopt import Hyperopt
from freqtrade.optimize.hyperopt.hyperopt_auto import HyperOptAuto
from freqtrade.optimize.hyperopt_tools import HyperoptTools
from freqtrade.optimize.optimize_reports import generate_strategy_stats
from freqtrade.optimize.space import SKDecimal, ft_IntDistribution
from freqtrade.strategy import IntParameter
from freqtrade.util import dt_utc
from tests.conftest import (
    CURRENT_TEST_STRATEGY,
    EXMS,
    get_args,
    get_markets,
    log_has,
    log_has_re,
    patch_exchange,
    patched_configuration_load_config_file,
)


def generate_result_metrics():
    """生成测试用的结果指标数据"""
    return {
        "trade_count": 1,          # 交易数量
        "total_trades": 1,         # 总交易数
        "avg_profit": 0.1,         # 平均利润
        "total_profit": 0.001,     # 总利润
        "profit": 0.01,            # 利润
        "duration": 20.0,          # 持续时间
        "wins": 1,                 # 盈利次数
        "draws": 0,                # 平局次数
        "losses": 0,               # 亏损次数
        "profit_mean": 0.01,       # 平均利润率
        "profit_total_abs": 0.001, # 总绝对利润
        "profit_total": 0.01,      # 总利润率
        "holding_avg": timedelta(minutes=20),  # 平均持有时间
        "max_drawdown_account": 0.001,        # 最大账户回撤
        "max_drawdown_abs": 0.001,            # 最大绝对回撤
        "loss": 0.001,             # 损失
        "is_initial_point": 0.001, # 是否初始点
        "is_random": False,        # 是否随机
        "is_best": 1,              # 是否最佳
    }


def test_setup_hyperopt_configuration_without_arguments(mocker, default_conf, caplog) -> None:
    """测试不带参数的超参数优化配置设置"""
    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        "hyperopt",
        "--config",
        "config.json",
        "--strategy",
        "HyperoptableStrategy",
    ]

    config = setup_optimize_configuration(get_args(args), RunMode.HYPEROPT)
    assert "max_open_trades" in config          # 最大同时持仓数
    assert "stake_currency" in config           # 交易货币
    assert "stake_amount" in config             # 每次交易金额
    assert "exchange" in config                 # 交易所配置
    assert "pair_whitelist" in config["exchange"]  # 交易对白名单
    assert "datadir" in config                  # 数据目录
    assert log_has("使用数据目录: {} ...".format(config["datadir"]), caplog)
    assert "timeframe" in config                # 时间周期

    assert "position_stacking" not in config    # 持仓堆叠配置
    assert not log_has("检测到参数 --enable-position-stacking ...", caplog)

    assert "timerange" not in config            # 时间范围
    assert "runmode" in config                  # 运行模式
    assert config["runmode"] == RunMode.HYPEROPT  # 确认是超参数优化模式


def test_setup_hyperopt_configuration_with_arguments(mocker, default_conf, caplog) -> None:
    """测试带参数的超参数优化配置设置"""
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch("freqtrade.configuration.configuration.create_datadir", lambda c, x: x)

    args = [
        "hyperopt",
        "--config",
        "config.json",
        "--strategy",
        "HyperoptableStrategy",
        "--datadir",
        "/foo/bar",
        "--timeframe",
        "1m",
        "--timerange",
        ":100",
        "--enable-position-stacking",
        "--epochs",
        "1000",
        "--spaces",
        "default",
        "--print-all",
    ]

    config = setup_optimize_configuration(get_args(args), RunMode.HYPEROPT)
    assert "max_open_trades" in config
    assert "stake_currency" in config
    assert "stake_amount" in config
    assert "exchange" in config
    assert "pair_whitelist" in config["exchange"]
    assert "datadir" in config
    assert config["runmode"] == RunMode.HYPEROPT

    assert log_has("使用数据目录: {} ...".format(config["datadir"]), caplog)
    assert "timeframe" in config
    assert log_has("检测到参数 -i/--timeframe ... 使用时间周期: 1m ...", caplog)

    assert "position_stacking" in config
    assert log_has("检测到参数 --enable-position-stacking ...", caplog)

    assert "timerange" in config
    assert log_has("检测到参数 --timerange: {} ...".format(config["timerange"]), caplog)

    assert "epochs" in config
    assert log_has(
        "检测到参数 --epochs ... 超参数优化将运行 1000 轮 ...", caplog
    )

    assert "spaces" in config
    assert log_has("检测到参数 -s/--spaces: {}".format(config["spaces"]), caplog)
    assert "print_all" in config
    assert log_has("检测到参数 --print-all ...", caplog)


def test_setup_hyperopt_configuration_stake_amount(mocker, default_conf) -> None:
    """测试超参数优化配置中的交易金额设置"""
    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        "hyperopt",
        "--config",
        "config.json",
        "--strategy",
        "HyperoptableStrategy",
        "--stake-amount",
        "1",
        "--starting-balance",
        "2",
    ]
    conf = setup_optimize_configuration(get_args(args), RunMode.HYPEROPT)
    assert isinstance(conf, dict)

    args = [
        "hyperopt",
        "--config",
        "config.json",
        "--strategy",
        CURRENT_TEST_STRATEGY,
        "--stake-amount",
        "1",
        "--starting-balance",
        "0.5",
    ]
    with pytest.raises(OperationalException, match=r"初始资金 .* 小于 .*"):
        setup_optimize_configuration(get_args(args), RunMode.HYPEROPT)


def test_setup_hyperopt_early_stop_setup(mocker, default_conf, caplog) -> None:
    """测试超参数优化的早停设置"""
    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        "hyperopt",
        "--config",
        "config.json",
        "--strategy",
        "HyperoptableStrategy",
        "--early-stop",
        "1",
    ]
    conf = setup_optimize_configuration(get_args(args), RunMode.HYPEROPT)
    assert isinstance(conf, dict)
    assert conf["early_stop"] == 20
    msg = (
        r"检测到参数 --early-stop ... "
        r"如果经过 (20|25) 轮仍无改进，将提前停止超参数优化 ..."
    )
    msg_adjust = r"早停轮数 .* 小于20，将被替换为20。"
    assert log_has_re(msg_adjust, caplog)
    assert log_has_re(msg, caplog)

    caplog.clear()

    args = [
        "hyperopt",
        "--config",
        "config.json",
        "--strategy",
        CURRENT_TEST_STRATEGY,
        "--early-stop",
        "25",
    ]
    conf1 = setup_optimize_configuration(get_args(args), RunMode.HYPEROPT)
    assert isinstance(conf1, dict)

    assert conf1["early_stop"] == 25
    assert not log_has_re(msg_adjust, caplog)
    assert log_has_re(msg, caplog)


def test_start_not_installed(mocker, default_conf, import_fails) -> None:
    """测试未安装必要依赖时的超参数优化启动"""
    start_mock = MagicMock()
    patched_configuration_load_config_file(mocker, default_conf)

    mocker.patch("freqtrade.optimize.hyperopt.Hyperopt.start", start_mock)
    patch_exchange(mocker)

    args = [
        "hyperopt",
        "--config",
        "config.json",
        "--strategy",
        "HyperoptableStrategy",
        "--epochs",
        "5",
        "--hyperopt-loss",
        "SharpeHyperOptLossDaily",
    ]
    pargs = get_args(args)

    with pytest.raises(OperationalException, match=r"请确保已安装超参数优化所需的依赖"):
        start_hyperopt(pargs)


def test_start_no_hyperopt_allowed(mocker, hyperopt_conf, caplog) -> None:
    """测试不允许使用独立Hyperopt文件时的启动"""
    start_mock = MagicMock()
    patched_configuration_load_config_file(mocker, hyperopt_conf)
    mocker.patch("freqtrade.optimize.hyperopt.Hyperopt.start", start_mock)
    patch_exchange(mocker)

    args = [
        "hyperopt",
        "--config",
        "config.json",
        "--hyperopt",
        "HyperoptTestSepFile",
        "--hyperopt-loss",
        "SharpeHyperOptLossDaily",
        "--epochs",
        "5",
    ]
    pargs = get_args(args)
    with pytest.raises(OperationalException, match=r"使用独立的Hyperopt文件已经被.*"):
        start_hyperopt(pargs)


def test_start_no_data(mocker, hyperopt_conf, tmp_path) -> None:
    """测试没有数据时的超参数优化启动"""
    hyperopt_conf["user_data_dir"] = tmp_path
    patched_configuration_load_config_file(mocker, hyperopt_conf)
    mocker.patch("freqtrade.data.history.load_pair_history", MagicMock(return_value=pd.DataFrame))
    mocker.patch(
        "freqtrade.optimize.hyperopt.hyperopt_optimizer.get_timerange",
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13))),
    )

    patch_exchange(mocker)
    args = [
        "hyperopt",
        "--config",
        "config.json",
        "--strategy",
        "HyperoptableStrategy",
        "--hyperopt-loss",
        "SharpeHyperOptLossDaily",
        "--epochs",
        "5",
    ]
    pargs = get_args(args)
    with pytest.raises(OperationalException, match="未找到数据。终止。"):
        start_hyperopt(pargs)

    # 清理失败的超参数优化留下的锁文件
    try:
        Path(Hyperopt.get_lock_filename(hyperopt_conf)).unlink()
    except Exception:
        pass


def test_start_filelock(mocker, hyperopt_conf, caplog) -> None:
    """测试文件锁机制"""
    hyperopt_mock = MagicMock(side_effect=Timeout(Hyperopt.get_lock_filename(hyperopt_conf)))
    patched_configuration_load_config_file(mocker, hyperopt_conf)
    mocker.patch("freqtrade.optimize.hyperopt.Hyperopt.__init__", hyperopt_mock)
    patch_exchange(mocker)

    args = [
        "hyperopt",
        "--config",
        "config.json",
        "--strategy",
        "HyperoptableStrategy",
        "--hyperopt-loss",
        "SharpeHyperOptLossDaily",
        "--epochs",
        "5",
    ]
    pargs = get_args(args)
    start_hyperopt(pargs)
    assert log_has("检测到另一个正在运行的freqtrade超参数优化实例。", caplog)


def test_log_results_if_loss_improves(hyperopt, capsys) -> None:
    """测试当损失改善时的结果日志"""
    hyperopt.current_best_loss = 2
    hyperopt.total_epochs = 2

    hyperopt.print_results(
        {
            "loss": 1,
            "results_metrics": generate_result_metrics(),
            "total_profit": 0,
            "current_epoch": 2,  # 从1开始计数（人类友好方式）
            "is_initial_point": False,
            "is_random": False,
            "is_best": True,
        }
    )
    hyperopt._hyper_out.print()
    out, _err = capsys.readouterr()
    assert all(
        x in out for x in ["最佳", "2/2", "1", "0.10%", "0.00100000 BTC    (1.00%)", "0:20:00"]
    )


def test_no_log_if_loss_does_not_improve(hyperopt, caplog) -> None:
    """测试当损失没有改善时不记录日志"""
    hyperopt.current_best_loss = 2
    hyperopt.print_results(
        {
            "is_best": False,
            "loss": 3,
            "current_epoch": 1,
        }
    )
    assert caplog.record_tuples == []


def test_roi_table_generation(hyperopt) -> None:
    """测试ROI表生成"""
    params = {
        "roi_t1": 5,
        "roi_t2": 10,
        "roi_t3": 15,
        "roi_p1": 1,
        "roi_p2": 2,
        "roi_p3": 3,
    }

    assert hyperopt.hyperopter.custom_hyperopt.generate_roi_table(params) == {
        0: 6,
        15: 3,
        25: 1,
        30: 0,
    }


def test_params_no_optimize_details(hyperopt) -> None:
    """测试未优化参数的详情"""
    hyperopt.hyperopter.config["spaces"] = ["buy"]
    res = hyperopt.hyperopter._get_no_optimize_details()
    assert isinstance(res, dict)
    assert "trailing" in res                  # 追踪止损
    assert res["trailing"]["trailing_stop"] is False
    assert "roi" in res                       # 收益目标
    assert res["roi"]["0"] == 0.04
    assert "stoploss" in res                  # 止损
    assert res["stoploss"]["stoploss"] == -0.1
    assert "max_open_trades" in res           # 最大同时持仓数
    assert res["max_open_trades"]["max_open_trades"] == 1


def test_start_calls_optimizer(mocker, hyperopt_conf, capsys) -> None:
    """测试启动时是否调用优化器"""
    dumper = mocker.patch("freqtrade.optimize.hyperopt.hyperopt_optimizer.dump")
    dumper2 = mocker.patch("freqtrade.optimize.hyperopt.Hyperopt._save_result")
    mocker.patch(
        "freqtrade.optimize.hyperopt.hyperopt_optimizer.calculate_market_change", return_value=1.5
    )
    mocker.patch("freqtrade.optimize.hyperopt.hyperopt.file_dump_json")

    mocker.patch(
        "freqtrade.optimize.backtesting.Backtesting.load_bt_data",
        MagicMock(return_value=(MagicMock(), None)),
    )
    mocker.patch(
        "freqtrade.optimize.hyperopt.hyperopt_optimizer.get_timerange",
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13))),
    )
    # 减少初始点数量以确保scikit-learn生成新值
    mocker.patch("freqtrade.optimize.hyperopt.hyperopt.INITIAL_POINTS", 2)

    parallel = mocker.patch(
        "freqtrade.optimize.hyperopt.Hyperopt.run_optimizer_parallel",
        MagicMock(
            return_value=[
                {
                    "loss": 1,
                    "results_explanation": "测试结果",
                    "params": {"buy": {}, "sell": {}, "roi": {}, "stoploss": 0.0},
                    "results_metrics": generate_result_metrics(),
                }
            ]
        ),
    )
    patch_exchange(mocker)
    # 从策略加载时间周期进行联合测试
    del hyperopt_conf["timeframe"]

    hyperopt = Hyperopt(hyperopt_conf)
    hyperopt.hyperopter.backtesting.strategy.advise_all_indicators = MagicMock()
    hyperopt.hyperopter.custom_hyperopt.generate_roi_table = MagicMock(return_value={})

    hyperopt.start()

    parallel.assert_called_once()

    out, _err = capsys.readouterr()
    assert "最佳结果:\n\n*    1/1: 测试结果 目标值: 1.00000\n" in out
    # 应调用历史K线数据处理
    assert dumper.call_count == 1
    assert dumper2.call_count == 1
    assert hasattr(hyperopt.hyperopter.backtesting.strategy, "advise_exit")  # 退出建议方法
    assert hasattr(hyperopt.hyperopter.backtesting.strategy, "advise_entry")  # 进入建议方法
    assert (
        hyperopt.hyperopter.backtesting.strategy.max_open_trades == hyperopt_conf["max_open_trades"]
    )
    assert hasattr(hyperopt.hyperopter.backtesting, "_position_stacking")  # 持仓堆叠属性


def test_hyperopt_format_results(hyperopt):
    """测试超参数优化结果格式化"""
    bt_result = {
        "results": pd.DataFrame(
            {
                "pair": ["UNITTEST/BTC", "UNITTEST/BTC", "UNITTEST/BTC", "UNITTEST/BTC"],
                "profit_ratio": [0.003312, 0.010801, 0.013803, 0.002780],
                "profit_abs": [0.000003, 0.000011, 0.000014, 0.000003],
                "open_date": [
                    dt_utc(2017, 11, 14, 19, 32, 00),
                    dt_utc(2017, 11, 14, 21, 36, 00),
                    dt_utc(2017, 11, 14, 22, 12, 00),
                    dt_utc(2017, 11, 14, 22, 44, 00),
                ],
                "close_date": [
                    dt_utc(2017, 11, 14, 21, 35, 00),
                    dt_utc(2017, 11, 14, 22, 10, 00),
                    dt_utc(2017, 11, 14, 22, 43, 00),
                    dt_utc(2017, 11, 14, 22, 58, 00),
                ],
                "open_rate": [0.002543, 0.003003, 0.003089, 0.003214],
                "close_rate": [0.002546, 0.003014, 0.003103, 0.003217],
                "trade_duration": [123, 34, 31, 14],
                "is_open": [False, False, False, True],
                "is_short": [False, False, False, False],
                "stake_amount": [0.01, 0.01, 0.01, 0.01],
                "exit_reason": [
                    ExitType.ROI.value,
                    ExitType.STOP_LOSS.value,
                    ExitType.ROI.value,
                    ExitType.FORCE_EXIT.value,
                ],
            }
        ),
        "config": hyperopt.config,
        "locks": [],
        "final_balance": 0.02,
        "rejected_signals": 2,
        "timedout_entry_orders": 0,
        "timedout_exit_orders": 0,
        "canceled_trade_entries": 0,
        "canceled_entry_orders": 0,
        "replaced_entry_orders": 0,
        "backtest_start_time": 1619718665,
        "backtest_end_time": 1619718665,
    }
    results_metrics = generate_strategy_stats(
        ["XRP/BTC"],
        "",
        bt_result,
        dt_utc(2017, 11, 14, 19, 32, 00),
        dt_utc(2017, 12, 14, 19, 32, 00),
        market_change=0,
    )

    results_explanation = HyperoptTools.format_results_explanation_string(results_metrics, "BTC")
    total_profit = results_metrics["profit_total_abs"]

    results = {
        "loss": 0.0,
        "params_dict": None,
        "params_details": None,
        "results_metrics": results_metrics,
        "results_explanation": results_explanation,
        "total_profit": total_profit,
        "current_epoch": 1,
        "is_initial_point": True,
    }

    result = HyperoptTools._format_explanation_string(results, 1)
    assert " 0.71%" in result
    assert "总利润  0.00003100 BTC" in result
    assert "0:50:00 分钟" in result


def test_populate_indicators(hyperopt, testdatadir) -> None:
    """测试指标填充"""
    data = load_data(testdatadir, "1m", ["UNITTEST/BTC"], fill_up_missing=True)
    dataframes = hyperopt.hyperopter.backtesting.strategy.advise_all_indicators(data)
    dataframe = dataframes["UNITTEST/BTC"]

    # 检查是否生成了一些指标，不需要测试所有指标
    assert "adx" in dataframe    # ADX指标
    assert "macd" in dataframe   # MACD指标
    assert "rsi" in dataframe    # RSI指标


def test_generate_optimizer(mocker, hyperopt_conf) -> None:
    """测试优化器生成"""
    hyperopt_conf.update(
        {
            "spaces": "all",
            "hyperopt_min_trades": 1,
        }
    )

    backtest_result = {
        "results": pd.DataFrame(
            {
                "pair": ["UNITTEST/BTC", "UNITTEST/BTC", "UNITTEST/BTC", "UNITTEST/BTC"],
                "profit_ratio": [0.003312, 0.010801, 0.013803, 0.002780],
                "profit_abs": [0.000003, 0.000011, 0.000014, 0.000003],
                "open_date": [
                    dt_utc(2017, 11, 14, 19, 32, 00),
                    dt_utc(2017, 11, 14, 21, 36, 00),
                    dt_utc(2017, 11, 14, 22, 12, 00),
                    dt_utc(2017, 11, 14, 22, 44, 00),
                ],
                "close_date": [
                    dt_utc(2017, 11, 14, 21, 35, 00),
                    dt_utc(2017, 11, 14, 22, 10, 00),
                    dt_utc(2017, 11, 14, 22, 43, 00),
                    dt_utc(2017, 11, 14, 22, 58, 00),
                ],
                "open_rate": [0.002543, 0.003003, 0.003089, 0.003214],
                "close_rate": [0.002546, 0.003014, 0.003103, 0.003217],
                "trade_duration": [123, 34, 31, 14],
                "is_open": [False, False, False, True],
                "is_short": [False, False, False, False],
                "stake_amount": [0.01, 0.01, 0.01, 0.01],
                "exit_reason": [
                    ExitType.ROI.value,
                    ExitType.STOP_LOSS.value,
                    ExitType.ROI.value,
                    ExitType.FORCE_EXIT.value,
                ],
            }
        ),
        "config": hyperopt_conf,
        "locks": [],
        "rejected_signals": 20,
        "timedout_entry_orders": 0,
        "timedout_exit_orders": 0,
        "canceled_trade_entries": 0,
        "canceled_entry_orders": 0,
        "replaced_entry_orders": 0,
        "final_balance": 1000,
    }

    mocker.patch(
        "freqtrade.optimize.hyperopt.hyperopt_optimizer.Backtesting.backtest",
        return_value=backtest_result,
    )
    mocker.patch(
        "freqtrade.optimize.hyperopt.hyperopt_optimizer.get_timerange",
        return_value=(dt_utc(2017, 12, 10), dt_utc(2017, 12, 13)),
    )
    patch_exchange(mocker)
    mocker.patch.object(Path, "open")
    mocker.patch("freqtrade.configuration.config_validation.validate_config_schema")
    mocker.patch(
        "freqtrade.optimize.hyperopt.hyperopt_optimizer.load", return_value={"XRP/BTC": None}
    )

    optimizer_param = {
        "buy_plusdi": 0.02,
        "buy_rsi": 35,
        "sell_minusdi": 0.02,
        "sell_rsi": 75,
        "protection_cooldown_lookback": 20,
        "protection_enabled": True,
        "roi_t1": 60.0,
        "roi_t2": 30.0,
        "roi_t3": 20.0,
        "roi_p1": 0.01,
        "roi_p2": 0.01,
        "roi_p3": 0.1,
        "stoploss": -0.4,
        "trailing_stop": True,
        "trailing_stop_positive": 0.02,
        "trailing_stop_positive_offset_p1": 0.05,
        "trailing_only_offset_is_reached": False,
        "max_open_trades": 3,
    }
    response_expected = {
        "loss": 1.9147239021396234,
        "results_explanation": (
            "     4 笔交易。4/0/0 盈利/平局/亏损。 "
            "平均利润   0.77%。中位数利润   0.71%。总利润  "
            "0.00003100 BTC (   0.00%)。 "
            "平均持续时间 0:50:00 分钟。"
        ),
        "params_details": {
            "buy": {
                "buy_plusdi": 0.02,
                "buy_rsi": 35,
            },
            "roi": {"0": 0.12, "20.0": 0.02, "50.0": 0.01, "110.0": 0},
            "protection": {
                "protection_cooldown_lookback": 20,
                "protection_enabled": True,
            },
            "sell": {
                "sell_minusdi": 0.02,
                "sell_rsi": 75,
            },
            "stoploss": {"stoploss": -0.4},
            "trailing": {
                "trailing_only_offset_is_reached": False,
                "trailing_stop": True,
                "trailing_stop_positive": 0.02,
                "trailing_stop_positive_offset": 0.07,
            },
            "max_open_trades": {"max_open_trades": 3},
        },
        "params_dict": optimizer_param,
        "params_not_optimized": {"buy": {}, "protection": {}, "sell": {}},
        "results_metrics": ANY,
        "total_profit": 3.1e-08,
    }

    hyperopt = Hyperopt(hyperopt_conf)
    hyperopt.hyperopter.min_date = dt_utc(2017, 12, 10)
    hyperopt.hyperopter.max_date = dt_utc(2017, 12, 13)
    hyperopt.hyperopter.init_spaces()
    generate_optimizer_value = hyperopt.hyperopter.generate_optimizer(optimizer_param)
    assert generate_optimizer_value == response_expected


def test_clean_hyperopt(mocker, hyperopt_conf, caplog):
    """测试清理超参数优化文件"""
    patch_exchange(mocker)

    mocker.patch(
        "freqtrade.strategy.hyper.HyperStrategyMixin.load_params_from_file",
        MagicMock(return_value={}),
    )
    mocker.patch("freqtrade.optimize.hyperopt.hyperopt.Path.is_file", MagicMock(return_value=True))
    unlinkmock = mocker.patch("freqtrade.optimize.hyperopt.hyperopt.Path.unlink", MagicMock())
    h = Hyperopt(hyperopt_conf)

    assert unlinkmock.call_count == 2
    assert log_has(f"删除 `{h.data_pickle_file}`。", caplog)


def test_print_json_spaces_all(mocker, hyperopt_conf, capsys) -> None:
    """测试打印所有空间的JSON结果"""
    dumper = mocker.patch("freqtrade.optimize.hyperopt.hyperopt_optimizer.dump")
    dumper2 = mocker.patch("freqtrade.optimize.hyperopt.Hyperopt._save_result")
    mocker.patch("freqtrade.optimize.hyperopt.hyperopt.file_dump_json")
    mocker.patch(
        "freqtrade.optimize.hyperopt.hyperopt_optimizer.calculate_market_change", return_value=1.5
    )

    mocker.patch(
        "freqtrade.optimize.backtesting.Backtesting.load_bt_data",
        MagicMock(return_value=(MagicMock(), None)),
    )
    mocker.patch(
        "freqtrade.optimize.hyperopt.hyperopt_optimizer.get_timerange",
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13))),
    )

    parallel = mocker.patch(
        "freqtrade.optimize.hyperopt.Hyperopt.run_optimizer_parallel",
        MagicMock(
            return_value=[
                {
                    "loss": 1,
                    "results_explanation": "测试结果",
                    "params": {},
                    "params_details": {
                        "buy": {"mfi-value": None},
                        "sell": {"sell-mfi-value": None},
                        "roi": {},
                        "stoploss": {"stoploss": None},
                        "trailing": {"trailing_stop": None},
                        "max_open_trades": {"max_open_trades": None},
                    },
                    "results_metrics": generate_result_metrics(),
                }
            ]
        ),
    )
    patch_exchange(mocker)

    hyperopt_conf.update(
        {
            "spaces": "all",
            "hyperopt_jobs": 1,
            "print_json": True,
        }
    )

    hyperopt = Hyperopt(hyperopt_conf)
    hyperopt.hyperopter.backtesting.strategy.advise_all_indicators = MagicMock()
    hyperopt.hyperopter.custom_hyperopt.generate_roi_table = MagicMock(return_value={})

    hyperopt.start()

    parallel.assert_called_once()

    out, _err = capsys.readouterr()
    result_str = (
        '{"params":{"mfi-value":null,"sell-mfi-value":null},"minimal_roi"'
        ':{},"stoploss":null,"trailing_stop":null,"max_open_trades":null}'
    )
    assert result_str in out
    # 应调用历史K线数据处理
    assert dumper.call_count == 1
    assert dumper2.call_count == 1


def test_print_json_spaces_default(mocker, hyperopt_conf, capsys) -> None:
    """测试打印默认空间的JSON结果"""
    dumper = mocker.patch("freqtrade.optimize.hyperopt.hyperopt_optimizer.dump")
    dumper2 = mocker.patch("freqtrade.optimize.hyperopt.Hyperopt._save_result")
    mocker.patch("freqtrade.optimize.hyperopt.hyperopt.file_dump_json")
    mocker.patch(
        "freqtrade.optimize.hyperopt.hyperopt_optimizer.calculate_market_change", return_value=1.5
    )
    mocker.patch(
        "freqtrade.optimize.backtesting.Backtesting.load_bt_data",
        MagicMock(return_value=(MagicMock(), None)),
    )
    mocker.patch(
        "freqtrade.optimize.hyperopt.hyperopt_optimizer.get_timerange",
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13))),
    )

    parallel = mocker.patch(
        "freqtrade.optimize.hyperopt.Hyperopt.run_optimizer_parallel",
        MagicMock(
            return_value=[
                {
                    "loss": 1,
                    "results_explanation": "测试结果",
                    "params": {},
                    "params_details": {
                        "buy": {"mfi-value": None},
                        "sell": {"sell-mfi-value": None},
                        "roi": {},
                        "stoploss": {"stoploss": None},
                    },
                    "results_metrics": generate_result_metrics(),
                }
            ]
        ),
    )
    patch_exchange(mocker)

    hyperopt_conf.update({"print_json": True})

    hyperopt = Hyperopt(hyperopt_conf)
    hyperopt.hyperopter.backtesting.strategy.advise_all_indicators = MagicMock()
    hyperopt.hyperopter.custom_hyperopt.generate_roi_table = MagicMock(return_value={})

    hyperopt.start()

    parallel.assert_called_once()

    out, _err = capsys.readouterr()
    assert (
        '{"params":{"mfi-value":null,"sell-mfi-value":null},"minimal_roi":{},"stoploss":null}'
        in out
    )
    # 应调用历史K线数据处理
    assert dumper.call_count == 1
    assert dumper2.call_count == 1


def test_print_json_spaces_roi_stoploss(mocker, hyperopt_conf, capsys) -> None:
    """测试打印ROI和止损空间的JSON结果"""
    dumper = mocker.patch("freqtrade.optimize.hyperopt.hyperopt_optimizer.dump")
    dumper2 = mocker.patch("freqtrade.optimize.hyperopt.Hyperopt._save_result")
    mocker.patch(
        "freqtrade.optimize.hyperopt.hyperopt_optimizer.calculate_market_change", return_value=1.5
    )
    mocker.patch("freqtrade.optimize.hyperopt.hyperopt.file_dump_json")
    mocker.patch(
        "freqtrade.optimize.backtesting.Backtesting.load_bt_data",
        MagicMock(return_value=(MagicMock(), None)),
    )
    mocker.patch(
        "freqtrade.optimize.hyperopt.hyperopt_optimizer.get_timerange",
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13))),
    )

    parallel = mocker.patch(
        "freqtrade.optimize.hyperopt.Hyperopt.run_optimizer_parallel",
        MagicMock(
            return_value=[
                {
                    "loss": 1,
                    "results_explanation": "测试结果",
                    "params": {},
                    "params_details": {"roi": {}, "stoploss": {"stoploss": None}},
                    "results_metrics": generate_result_metrics(),
                }
            ]
        ),
    )
    patch_exchange(mocker)

    hyperopt_conf.update(
        {
            "spaces": "roi stoploss",
            "hyperopt_jobs": 1,
            "print_json": True,
        }
    )

    hyperopt = Hyperopt(hyperopt_conf)
    hyperopt.hyperopter.backtesting.strategy.advise_all_indicators = MagicMock()
    hyperopt.hyperopter.custom_hyperopt.generate_roi_table = MagicMock(return_value={})

    hyperopt.start()

    parallel.assert_called_once()

    out, _err = capsys.readouterr()
    assert '{"minimal_roi":{},"stoploss":null}' in out

    assert dumper.call_count == 1
    assert dumper2.call_count == 1


def test_simplified_interface_roi_stoploss(mocker, hyperopt_conf, capsys) -> None:
    """测试ROI和止损空间的简化接口"""
    dumper = mocker.patch("freqtrade.optimize.hyperopt.hyperopt_optimizer.dump")
    dumper2 = mocker.patch("freqtrade.optimize.hyperopt.Hyperopt._save_result")
    mocker.patch(
        "freqtrade.optimize.hyperopt.hyperopt_optimizer.calculate_market_change", return_value=1.5
    )
    mocker.patch("freqtrade.optimize.hyperopt.hyperopt.file_dump_json")
    mocker.patch(
        "freqtrade.optimize.backtesting.Backtesting.load_bt_data",
        MagicMock(return_value=(MagicMock(), None)),
    )
    mocker.patch(
        "freqtrade.optimize.hyperopt.hyperopt_optimizer.get_timerange",
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13))),
    )

    parallel = mocker.patch(
        "freqtrade.optimize.hyperopt.Hyperopt.run_optimizer_parallel",
        MagicMock(
            return_value=[
                {
                    "loss": 1,
                    "results_explanation": "测试结果",
                    "params": {"stoploss": 0.0},
                    "results_metrics": generate_result_metrics(),
                }
            ]
        ),
    )
    patch_exchange(mocker)

    hyperopt_conf.update({"spaces": "roi stoploss"})

    hyperopt = Hyperopt(hyperopt_conf)
    hyperopt.hyperopter.backtesting.strategy.advise_all_indicators = MagicMock()
    hyperopt.hyperopter.custom_hyperopt.generate_roi_table = MagicMock(return_value={})

    hyperopt.start()

    parallel.assert_called_once()

    out, _err = capsys.readouterr()
    assert "最佳结果:\n\n*    1/1: 测试结果 目标值: 1.00000\n" in out
    assert dumper.call_count == 1
    assert dumper2.call_count == 1

    assert hasattr(hyperopt.hyperopter.backtesting.strategy, "advise_exit")
    assert hasattr(hyperopt.hyperopter.backtesting.strategy, "advise_entry")
    assert (
        hyperopt.hyperopter.backtesting.strategy.max_open_trades == hyperopt_conf["max_open_trades"]
    )
    assert hasattr(hyperopt.hyperopter.backtesting, "_position_stacking")


def test_simplified_interface_all_failed(mocker, hyperopt_conf, caplog) -> None:
    """测试所有空间简化接口失败情况"""
    mocker.patch("freqtrade.optimize.hyperopt.hyperopt_optimizer.dump", MagicMock())
    mocker.patch("freqtrade.optimize.hyperopt.hyperopt.file_dump_json")
    mocker.patch(
        "freqtrade.optimize.backtesting.Backtesting.load_bt_data",
        MagicMock(return_value=(MagicMock(), None)),
    )
    mocker.patch(
        "freqtrade.optimize.hyperopt.hyperopt_optimizer.get_timerange",
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13))),
    )

    patch_exchange(mocker)

    hyperopt_conf.update(
        {
            "spaces": "all",
        }
    )

    mocker.patch(
        "freqtrade.optimize.hyperopt.hyperopt_auto.HyperOptAuto._generate_indicator_space",
        return_value=[],
    )

    hyperopt = Hyperopt(hyperopt_conf)
    hyperopt.hyperopter.backtesting.strategy.advise_all_indicators = MagicMock()
    hyperopt.hyperopter.custom_hyperopt.generate_roi_table = MagicMock(return_value={})

    with pytest.raises(OperationalException, match=r"'protection'空间包含在*"):
        hyperopt.hyperopter.init_spaces()

    hyperopt.config["hyperopt_ignore_missing_space"] = True
    caplog.clear()
    hyperopt.hyperopter.init_spaces()
    assert log_has_re(r"'protection'空间包含在*", caplog)
    assert hyperopt.hyperopter.protection_space == []


def test_simplified_interface_buy(mocker, hyperopt_conf, capsys) -> None:
    """测试买入空间的简化接口"""
    dumper = mocker.patch("freqtrade.optimize.hyperopt.hyperopt_optimizer.dump")
    dumper2 = mocker.patch("freqtrade.optimize.hyperopt.Hyperopt._save_result")
    mocker.patch(
        "freqtrade.optimize.hyperopt.hyperopt_optimizer.calculate_market_change", return_value=1.5
    )
    mocker.patch("freqtrade.optimize.hyperopt.hyperopt.file_dump_json")
    mocker.patch(
        "freqtrade.optimize.backtesting.Backtesting.load_bt_data",
        MagicMock(return_value=(MagicMock(), None)),
    )
    mocker.patch(
        "freqtrade.optimize.hyperopt.hyperopt_optimizer.get_timerange",
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13))),
    )

    parallel = mocker.patch(
        "freqtrade.optimize.hyperopt.Hyperopt.run_optimizer_parallel",
        MagicMock(
            return_value=[
                {
                    "loss": 1,
                    "results_explanation": "测试结果",
                    "params": {},
                    "results_metrics": generate_result_metrics(),
                }
            ]
        ),
    )
    patch_exchange(mocker)

    hyperopt_conf.update({"spaces": "buy"})

    hyperopt = Hyperopt(hyperopt_conf)
    hyperopt.hyperopter.backtesting.strategy.advise_all_indicators = MagicMock()
    hyperopt.hyperopter.custom_hyperopt.generate_roi_table = MagicMock(return_value={})

    hyperopt.start()

    parallel.assert_called_once()

    out, _err = capsys.readouterr()
    assert "最佳结果:\n\n*    1/1: 测试结果 目标值: 1.00000\n" in out
    assert dumper.called
    assert dumper.call_count == 1
    assert dumper2.call_count == 1
    assert hasattr(hyperopt.hyperopter.backtesting.strategy, "advise_exit")
    assert hasattr(hyperopt.hyperopter.backtesting.strategy, "advise_entry")
    assert (
        hyperopt.hyperopter.backtesting.strategy.max_open_trades == hyperopt_conf["max_open_trades"]
    )
    assert hasattr(hyperopt.hyperopter.backtesting, "_position_stacking")


def test_simplified_interface_sell(mocker, hyperopt_conf, capsys) -> None:
    """测试卖出空间的简化接口"""
    dumper = mocker.patch("freqtrade.optimize.hyperopt.hyperopt_optimizer.dump")
    dumper2 = mocker.patch("freqtrade.optimize.hyperopt.Hyperopt._save_result")
    mocker.patch(
        "freqtrade.optimize.hyperopt.hyperopt_optimizer.calculate_market_change", return_value=1.5
    )
    mocker.patch("freqtrade.optimize.hyperopt.hyperopt.file_dump_json")
    mocker.patch(
        "freqtrade.optimize.backtesting.Backtesting.load_bt_data",
        MagicMock(return_value=(MagicMock(), None)),
    )
    mocker.patch(
        "freqtrade.optimize.hyperopt.hyperopt_optimizer.get_timerange",
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13))),
    )

    parallel = mocker.patch(
        "freqtrade.optimize.hyperopt.Hyperopt.run_optimizer_parallel",
        MagicMock(
            return_value=[
                {
                    "loss": 1,
                    "results_explanation": "测试结果",
                    "params": {},
                    "results_metrics": generate_result_metrics(),
                }
            ]
        ),
    )
    patch_exchange(mocker)

    hyperopt_conf.update(
        {
            "spaces": "sell",
        }
    )

    hyperopt = Hyperopt(hyperopt_conf)
    hyperopt.hyperopter.backtesting.strategy.advise_all_indicators = MagicMock()
    hyperopt.hyperopter.custom_hyperopt.generate_roi_table = MagicMock(return_value={})

    hyperopt.start()

    parallel.assert_called_once()

    out, _err = capsys.readouterr()
    assert "最佳结果:\n\n*    1/1: 测试结果 目标值: 1.00000\n" in out
    assert dumper.called
    assert dumper.call_count == 1
    assert dumper2.call_count == 1
    assert hasattr(hyperopt.hyperopter.backtesting.strategy, "advise_exit")
    assert hasattr(hyperopt.hyperopter.backtesting.strategy, "advise_entry")
    assert (
        hyperopt.hyperopter.backtesting.strategy.max_open_trades == hyperopt_conf["max_open_trades"]
    )
    assert hasattr(hyperopt.hyperopter.backtesting, "_position_stacking")


@pytest.mark.parametrize(
    "space",
    [
        ("buy"),
        ("sell"),
        ("protection"),
    ],
)
def test_simplified_interface_failed(mocker, hyperopt_conf, space) -> None:
    """测试简化接口失败情况"""
    mocker.patch("freqtrade.optimize.hyperopt.hyperopt_optimizer.dump", MagicMock())
    mocker.patch("freqtrade.optimize.hyperopt.hyperopt.file_dump_json")
    mocker.patch(
        "freqtrade.optimize.backtesting.Backtesting.load_bt_data",
        MagicMock(return_value=(MagicMock(), None)),
    )
    mocker.patch(
        "freqtrade.optimize.hyperopt.hyperopt_optimizer.get_timerange",
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13))),
    )
    mocker.patch(
        "freqtrade.optimize.hyperopt.hyperopt_auto.HyperOptAuto._generate_indicator_space",
        return_value=[],
    )

    patch_exchange(mocker)

    hyperopt_conf.update({"spaces": space})

    hyperopt = Hyperopt(hyperopt_conf)
    hyperopt.hyperopter.backtesting.strategy.advise_all_indicators = MagicMock()
    hyperopt.hyperopter.custom_hyperopt.generate_roi_table = MagicMock(return_value={})

    with pytest.raises(OperationalException, match=f"'{space}'空间包含在*"):
        hyperopt.start()


def test_in_strategy_auto_hyperopt(mocker, hyperopt_conf, tmp_path, fee) -> None:
    """测试策略内自动超参数优化"""
    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.get_fee", fee)
    # 减少初始点数量以确保scikit-learn生成新值
    mocker.patch("freqtrade.optimize.hyperopt.hyperopt.INITIAL_POINTS", 2)
    (tmp_path / "hyperopt_results").mkdir(parents=True)
    # 无需额外hyperopt
    hyperopt_conf.update(
        {
            "strategy": "HyperoptableStrategy",
            "user_data_dir": tmp_path,
            "hyperopt_random_state": 42,
            "spaces": ["all"],
        }
    )
    hyperopt = Hyperopt(hyperopt_conf)
    opt = hyperopt.hyperopter
    opt.backtesting.exchange.get_max_leverage = MagicMock(return_value=1.0)
    assert isinstance(opt.custom_hyperopt, HyperOptAuto)
    assert isinstance(opt.backtesting.strategy.buy_rsi, IntParameter)
    assert opt.backtesting.strategy.bot_started is True
    assert opt.backtesting.strategy.bot_loop_started is False

    assert opt.backtesting.strategy.buy_rsi.in_space is True
    assert opt.backtesting.strategy.buy_rsi.value == 35
    assert opt.backtesting.strategy.sell_rsi.value == 74
    assert opt.backtesting.strategy.protection_cooldown_lookback.value == 30
    assert opt.backtesting.strategy.max_open_trades == 1
    buy_rsi_range = opt.backtesting.strategy.buy_rsi.range
    assert isinstance(buy_rsi_range, range)
    # 范围从0到50（包含）
    assert len(list(buy_rsi_range)) == 51

    hyperopt.start()
    # 所有值都应已更改
    assert opt.backtesting.strategy.protection_cooldown_lookback.value != 30
    assert opt.backtesting.strategy.buy_rsi.value != 35
    assert opt.backtesting.strategy.sell_rsi.value != 74
    assert opt.backtesting.strategy.max_open_trades != 1

    opt.custom_hyperopt.generate_estimator = lambda *args, **kwargs: "ET1"
    with pytest.raises(OperationalException, match="Optuna采样器ET1不支持。"):
        opt.get_optimizer(42)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_in_strategy_auto_hyperopt_with_parallel(mocker, hyperopt_conf, tmp_path, fee) -> None:
    """测试带并行的策略内自动超参数优化"""
    mocker.patch(f"{EXMS}.validate_config", MagicMock())
    mocker.patch(f"{EXMS}.get_fee", fee)
    mocker.patch(f"{EXMS}.reload_markets")
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=get_markets()))
    (tmp_path / "hyperopt_results").mkdir(parents=True)
    # 减少初始点数量以确保scikit-learn生成新值
    mocker.patch("freqtrade.optimize.hyperopt.hyperopt.INITIAL_POINTS", 2)
    # 无需额外hyperopt
    hyperopt_conf.update(
        {
            "strategy": "HyperoptableStrategy",
            "user_data_dir": tmp_path,
            "hyperopt_random_state": 42,
            "spaces": ["all"],
            # 强制并行
            "epochs": 2,
            "hyperopt_jobs": 2,
            "fee": fee.return_value,
        }
    )
    hyperopt = Hyperopt(hyperopt_conf)
    opt = hyperopt.hyperopter
    opt.backtesting.exchange.get_max_leverage = lambda *x, **xx: 1.0
    opt.backtesting.exchange.get_min_pair_stake_amount = lambda *x, **xx: 0.00001
    opt.backtesting.exchange.get_max_pair_stake_amount = lambda *x, **xx: 100.0
    opt.backtesting.exchange._markets = get_markets()

    assert isinstance(opt.custom_hyperopt, HyperOptAuto)
    assert isinstance(opt.backtesting.strategy.buy_rsi, IntParameter)
    assert opt.backtesting.strategy.bot_started is True
    assert opt.backtesting.strategy.bot_loop_started is False

    assert opt.backtesting.strategy.buy_rsi.in_space is True
    assert opt.backtesting.strategy.buy_rsi.value == 35
    assert opt.backtesting.strategy.sell_rsi.value == 74
    assert opt.backtesting.strategy.protection_cooldown_lookback.value == 30
    buy_rsi_range = opt.backtesting.strategy.buy_rsi.range
    assert isinstance(buy_rsi_range, range)
    # 范围从0到50（包含）
    assert len(list(buy_rsi_range)) == 51

    hyperopt.start()


def test_in_strategy_auto_hyperopt_per_epoch(mocker, hyperopt_conf, tmp_path, fee) -> None:
    """测试每轮都分析的策略内自动超参数优化"""
    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.get_fee", fee)
    (tmp_path / "hyperopt_results").mkdir(parents=True)

    hyperopt_conf.update(
        {
            "strategy": "HyperoptableStrategy",
            "user_data_dir": tmp_path,
            "hyperopt_random_state": 42,
            "spaces": ["all"],
            "epochs": 3,
            "analyze_per_epoch": True,
        }
    )
    go = mocker.patch(
        "freqtrade.optimize.hyperopt.hyperopt_optimizer.HyperOptimizer.generate_optimizer",
        return_value={
            "loss": 0.05,
            "results_explanation": "测试结果",
            "params": {},
            "results_metrics": generate_result_metrics(),
        },
    )
    hyperopt = Hyperopt(hyperopt_conf)
    opt = hyperopt.hyperopter
    opt.backtesting.exchange.get_max_leverage = MagicMock(return_value=1.0)
    assert isinstance(opt.custom_hyperopt, HyperOptAuto)
    assert isinstance(opt.backtesting.strategy.buy_rsi, IntParameter)
    assert opt.backtesting.strategy.bot_loop_started is False
    assert opt.backtesting.strategy.bot_started is True

    assert opt.backtesting.strategy.buy_rsi.in_space is True
    assert opt.backtesting.strategy.buy_rsi.value == 35
    assert opt.backtesting.strategy.sell_rsi.value == 74
    assert opt.backtesting.strategy.protection_cooldown_lookback.value == 30
    buy_rsi_range = opt.backtesting.strategy.buy_rsi.range
    assert isinstance(buy_rsi_range, range)
    # 范围从0到50（包含）
    assert len(list(buy_rsi_range)) == 51

    hyperopt.start()
    # 回测应被调用3次（每轮一次）
    assert go.call_count == 3


def test_SKDecimal():
    """测试SKDecimal类"""
    space = SKDecimal(1, 2, decimals=2)
    assert space._contains(1.5)
    assert not space._contains(2.5)
    assert space.low == 1
    assert space.high == 2

    assert space._contains(1.51)
    assert space._contains(1.01)
    # 2位小数时超出范围
    assert not space._contains(1.511)
    assert not space._contains(1.111222)

    with pytest.raises(ValueError):
        SKDecimal(1, 2, step=5, decimals=0.2)

    with pytest.raises(ValueError):
        SKDecimal(1, 2, step=None, decimals=None)

    s = SKDecimal(1, 2, step=0.1, decimals=None)
    assert s.step == 0.1
    assert s._contains(1.1)
    assert not s._contains(1.11)


def test_stake_amount_unlimited_max_open_trades(mocker, hyperopt_conf, tmp_path, fee) -> None:
    """测试无限交易金额时的最大同时持仓数"""
    # 本测试确保当交易金额为无限时，回测忽略无限的最大同时持仓数
    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.get_fee", fee)
    (tmp_path / "hyperopt_results").mkdir(parents=True)
    hyperopt_conf.update(
        {
            "strategy": "HyperoptableStrategy",
            "user_data_dir": tmp_path,
            "hyperopt_random_state": 42,
            "spaces": ["trades"],
            "stake_amount": "unlimited",
        }
    )
    hyperopt = Hyperopt(hyperopt_conf)

    assert isinstance(hyperopt.hyperopter.custom_hyperopt, HyperOptAuto)

    assert hyperopt.hyperopter.backtesting.strategy.max_open_trades == 1

    hyperopt.start()

    assert hyperopt.hyperopter.backtesting.strategy.max_open_trades == 3


def test_max_open_trades_dump(mocker, hyperopt_conf, tmp_path, fee, capsys) -> None:
    """测试最大同时持仓数在输出JSON参数中不被保存为inf"""
    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.get_fee", fee)
    (tmp_path / "hyperopt_results").mkdir(parents=True)
    hyperopt_conf.update(
        {
            "strategy": "HyperoptableStrategy",
            "user_data_dir": tmp_path,
            "hyperopt_random_state": 42,
            "spaces": ["trades"],
        }
    )
    hyperopt = Hyperopt(hyperopt_conf)

    def optuna_mock(hyperopt, *args, **kwargs):
        a = hyperopt.get_optuna_asked_points(*args, **kwargs)
        a[0]._cached_frozen_trial.params["max_open_trades"] = -1
        return a, [True]

    mocker.patch(
        "freqtrade.optimize.hyperopt.Hyperopt.get_asked_points",
        side_effect=partial(optuna_mock, hyperopt),
    )

    assert isinstance(hyperopt.hyperopter.custom_hyperopt, HyperOptAuto)

    hyperopt.start()

    out, _err = capsys.readouterr()

    assert "max_open_trades = -1" in out
    assert "max_open_trades = inf" not in out

    ##############

    hyperopt_conf.update({"print_json": True})

    hyperopt = Hyperopt(hyperopt_conf)
    mocker.patch(
        "freqtrade.optimize.hyperopt.Hyperopt.get_asked_points",
        side_effect=partial(optuna_mock, hyperopt),
    )

    assert isinstance(hyperopt.hyperopter.custom_hyperopt, HyperOptAuto)

    hyperopt.start()

    out, _err = capsys.readouterr()

    assert '"max_open_trades":-1' in out


def test_max_open_trades_consistency(mocker, hyperopt_conf, tmp_path, fee) -> None:
    """测试最大同时持仓数在所有需要它的函数中的一致性"""
    # 本测试确保在超参数优化改变最大同时持仓数后，所有需要它的函数使用相同的值
    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.get_fee", return_value=0)

    (tmp_path / "hyperopt_results").mkdir(parents=True)
    hyperopt_conf.update(
        {
            "strategy": "HyperoptableStrategy",
            "user_data_dir": tmp_path,
            "hyperopt_random_state": 42,
            "spaces": ["trades"],
            "stake_amount": "unlimited",
            "dry_run_wallet": 8,
            "available_capital": 8,
            "dry_run": True,
            "epochs": 1,
        }
    )
    hyperopt = Hyperopt(hyperopt_conf)

    assert isinstance(hyperopt.hyperopter.custom_hyperopt, HyperOptAuto)

    hyperopt.hyperopter.custom_hyperopt.max_open_trades_space = lambda: [
        ft_IntDistribution(1, 10, "max_open_trades")
    ]

    first_time_evaluated = False

    def stake_amount_interceptor(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal first_time_evaluated

            stake_amount = func(*args, **kwargs)
            if first_time_evaluated is False:
                assert stake_amount == 2
                first_time_evaluated = True
            return stake_amount

        return wrapper

    hyperopt.hyperopter.backtesting.wallets._calculate_unlimited_stake_amount = (
        stake_amount_interceptor(
            hyperopt.hyperopter.backtesting.wallets._calculate_unlimited_stake_amount
        )
    )

    hyperopt.start()

    assert hyperopt.hyperopter.backtesting.strategy.max_open_trades == 4
    assert hyperopt.config["max_open_trades"] == 4
