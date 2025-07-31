import logging
import re
from pathlib import Path

import numpy as np
import pytest
import rapidjson

from freqtrade.constants import FTHYPT_FILEVERSION
from freqtrade.exceptions import OperationalException
from freqtrade.optimize.hyperopt_tools import HyperoptTools, hyperopt_serializer
from tests.conftest import CURRENT_TEST_STRATEGY, log_has, log_has_re


# 用于重复对象补丁的函数
def create_results() -> list[dict]:
    return [{"loss": 1, "result": "foo", "params": {}, "is_best": True}]


def test_save_results_saves_epochs(hyperopt, tmp_path, caplog) -> None:
    hyperopt.results_file = tmp_path / "ut_results.fthypt"

    hyperopt_epochs = HyperoptTools.load_filtered_results(hyperopt.results_file, {})
    assert log_has_re("超参数优化文件 .* 未找到。", caplog)
    assert hyperopt_epochs == ([], 0)

    # 测试写入临时目录并再次读取
    epochs = create_results()

    caplog.set_level(logging.DEBUG)

    for epoch in epochs:
        hyperopt._save_result(epoch)
    assert log_has(f"1 个周期已保存至 '{hyperopt.results_file}'。", caplog)

    hyperopt._save_result(epochs[0])
    assert log_has(f"2 个周期已保存至 '{hyperopt.results_file}'。", caplog)

    hyperopt_epochs = HyperoptTools.load_filtered_results(hyperopt.results_file, {})
    assert len(hyperopt_epochs) == 2
    assert hyperopt_epochs[1] == 2
    assert len(hyperopt_epochs[0]) == 2

    result_gen = HyperoptTools._read_results(hyperopt.results_file, 1)
    epoch = next(result_gen)
    assert len(epoch) == 1
    assert epoch[0] == epochs[0]
    epoch = next(result_gen)
    assert len(epoch) == 1
    epoch = next(result_gen)
    assert len(epoch) == 0
    with pytest.raises(StopIteration):
        next(result_gen)


def test_load_previous_results2(mocker, testdatadir, caplog) -> None:
    results_file = testdatadir / "hyperopt_results_SampleStrategy.pickle"
    with pytest.raises(
        OperationalException, match=r"不再支持旧版超参数优化结果。*"
    ):
        HyperoptTools.load_filtered_results(results_file, {})


@pytest.mark.parametrize(
    "spaces, expected_results",
    [
        (
            ["buy"],
            {
                "buy": True,
                "sell": False,
                "roi": False,
                "stoploss": False,
                "trailing": False,
                "protection": False,
                "trades": False,
            },
        ),
        (
            ["sell"],
            {
                "buy": False,
                "sell": True,
                "roi": False,
                "stoploss": False,
                "trailing": False,
                "protection": False,
                "trades": False,
            },
        ),
        (
            ["roi"],
            {
                "buy": False,
                "sell": False,
                "roi": True,
                "stoploss": False,
                "trailing": False,
                "protection": False,
                "trades": False,
            },
        ),
        (
            ["stoploss"],
            {
                "buy": False,
                "sell": False,
                "roi": False,
                "stoploss": True,
                "trailing": False,
                "protection": False,
                "trades": False,
            },
        ),
        (
            ["trailing"],
            {
                "buy": False,
                "sell": False,
                "roi": False,
                "stoploss": False,
                "trailing": True,
                "protection": False,
                "trades": False,
            },
        ),
        (
            ["buy", "sell", "roi", "stoploss"],
            {
                "buy": True,
                "sell": True,
                "roi": True,
                "stoploss": True,
                "trailing": False,
                "protection": False,
                "trades": False,
            },
        ),
        (
            ["buy", "sell", "roi", "stoploss", "trailing"],
            {
                "buy": True,
                "sell": True,
                "roi": True,
                "stoploss": True,
                "trailing": True,
                "protection": False,
                "trades": False,
            },
        ),
        (
            ["buy", "roi"],
            {
                "buy": True,
                "sell": False,
                "roi": True,
                "stoploss": False,
                "trailing": False,
                "protection": False,
                "trades": False,
            },
        ),
        (
            ["all"],
            {
                "buy": True,
                "sell": True,
                "roi": True,
                "stoploss": True,
                "trailing": True,
                "protection": True,
                "trades": True,
            },
        ),
        (
            ["default"],
            {
                "buy": True,
                "sell": True,
                "roi": True,
                "stoploss": True,
                "trailing": False,
                "protection": False,
                "trades": False,
            },
        ),
        (
            ["default", "trailing"],
            {
                "buy": True,
                "sell": True,
                "roi": True,
                "stoploss": True,
                "trailing": True,
                "protection": False,
                "trades": False,
            },
        ),
        (
            ["all", "buy"],
            {
                "buy": True,
                "sell": True,
                "roi": True,
                "stoploss": True,
                "trailing": True,
                "protection": True,
                "trades": True,
            },
        ),
        (
            ["default", "buy"],
            {
                "buy": True,
                "sell": True,
                "roi": True,
                "stoploss": True,
                "trailing": False,
                "protection": False,
                "trades": False,
            },
        ),
        (
            ["all"],
            {
                "buy": True,
                "sell": True,
                "roi": True,
                "stoploss": True,
                "trailing": True,
                "protection": True,
                "trades": True,
            },
        ),
        (
            ["protection"],
            {
                "buy": False,
                "sell": False,
                "roi": False,
                "stoploss": False,
                "trailing": False,
                "protection": True,
                "trades": False,
            },
        ),
        (
            ["trades"],
            {
                "buy": False,
                "sell": False,
                "roi": False,
                "stoploss": False,
                "trailing": False,
                "protection": False,
                "trades": True,
            },
        ),
        (
            ["default", "trades"],
            {
                "buy": True,
                "sell": True,
                "roi": True,
                "stoploss": True,
                "trailing": False,
                "protection": False,
                "trades": True,
            },
        ),
    ],
)
def test_has_space(hyperopt_conf, spaces, expected_results):
    for s in ["buy", "sell", "roi", "stoploss", "trailing", "protection", "trades"]:
        hyperopt_conf.update({"spaces": spaces})
        assert HyperoptTools.has_space(hyperopt_conf, s) == expected_results[s]


def test_show_epoch_details(capsys):
    test_result = {
        "params_details": {
            "trailing": {
                "trailing_stop": True,
                "trailing_stop_positive": 0.02,
                "trailing_stop_positive_offset": 0.04,
                "trailing_only_offset_is_reached": True,
            },
            "roi": {0: 0.18, 90: 0.14, 225: 0.05, 430: 0},
        },
        "results_explanation": "测试结果",
        "is_initial_point": False,
        "total_profit": 0,
        "current_epoch": 2,  # 从1开始（人性化显示）
        "is_best": True,
    }

    HyperoptTools.show_epoch_details(test_result, 5, False, no_header=True)
    captured = capsys.readouterr()
    assert "# 追踪止损：" in captured.out
    # re.match(r"交易对 .*", captured.out)
    assert re.search(r"^\s+trailing_stop = True$", captured.out, re.MULTILINE)
    assert re.search(r"^\s+trailing_stop_positive = 0.02$", captured.out, re.MULTILINE)
    assert re.search(r"^\s+trailing_stop_positive_offset = 0.04$", captured.out, re.MULTILINE)
    assert re.search(r"^\s+trailing_only_offset_is_reached = True$", captured.out, re.MULTILINE)

    assert "# ROI表格：" in captured.out
    assert re.search(r"^\s+minimal_roi = \{$", captured.out, re.MULTILINE)
    assert re.search(r"^\s+\"90\"\:\s0.14,\s*$", captured.out, re.MULTILINE)


def test__pprint_dict():
    params = {"buy_std": 1.2, "buy_rsi": 31, "buy_enable": True, "buy_what": "asdf"}
    non_params = {"buy_notoptimied": 55}

    x = HyperoptTools._pprint_dict(params, non_params)
    assert (
        x
        == """{
    "buy_std": 1.2,
    "buy_rsi": 31,
    "buy_enable": True,
    "buy_what": "asdf",
    "buy_notoptimied": 55,  # 从策略加载的值
}"""
    )


def test_get_strategy_filename(default_conf, tmp_path):
    default_conf["user_data_dir"] = tmp_path
    x = HyperoptTools.get_strategy_filename(default_conf, "StrategyTestV3")
    assert isinstance(x, Path)
    assert x == Path(__file__).parents[1] / "strategy/strats/strategy_test_v3.py"

    x = HyperoptTools.get_strategy_filename(default_conf, "NonExistingStrategy")
    assert x is None


def test_export_params(tmp_path):
    filename = tmp_path / f"{CURRENT_TEST_STRATEGY}.json"
    assert not filename.is_file()
    params = {
        "params_details": {
            "buy": {"buy_rsi": 30},
            "sell": {"sell_rsi": 70},
            "roi": {"0": 0.528, "346": 0.08499, "507": 0.049, "1595": 0},
            "max_open_trades": {"max_open_trades": 5},
        },
        "params_not_optimized": {
            "stoploss": -0.05,
            "trailing": {
                "trailing_stop": False,
                "trailing_stop_positive": 0.05,
                "trailing_stop_positive_offset": 0.1,
                "trailing_only_offset_is_reached": True,
            },
        },
    }
    HyperoptTools.export_params(params, CURRENT_TEST_STRATEGY, filename)

    assert filename.is_file()

    with filename.open("r") as f:
        content = rapidjson.load(f)
    assert content["strategy_name"] == CURRENT_TEST_STRATEGY
    assert "params" in content
    assert "buy" in content["params"]
    assert "sell" in content["params"]
    assert "roi" in content["params"]
    assert "stoploss" in content["params"]
    assert "trailing" in content["params"]
    assert "max_open_trades" in content["params"]


def test_try_export_params(default_conf, tmp_path, caplog, mocker):
    default_conf["disableparamexport"] = False
    default_conf["user_data_dir"] = tmp_path
    export_mock = mocker.patch("freqtrade.optimize.hyperopt_tools.HyperoptTools.export_params")

    filename = tmp_path / f"{CURRENT_TEST_STRATEGY}.json"
    assert not filename.is_file()
    params = {
        "params_details": {
            "buy": {"buy_rsi": 30},
            "sell": {"sell_rsi": 70},
            "roi": {"0": 0.528, "346": 0.08499, "507": 0.049, "1595": 0},
        },
        "params_not_optimized": {
            "stoploss": -0.05,
            "trailing": {
                "trailing_stop": False,
                "trailing_stop_positive": 0.05,
                "trailing_stop_positive_offset": 0.1,
                "trailing_only_offset_is_reached": True,
            },
        },
        FTHYPT_FILEVERSION: 2,
    }
    HyperoptTools.try_export_params(default_conf, "StrategyTestVXXX", params)

    assert log_has("未找到策略，不导出参数文件。", caplog)
    assert export_mock.call_count == 0
    caplog.clear()

    HyperoptTools.try_export_params(default_conf, CURRENT_TEST_STRATEGY, params)

    assert export_mock.call_count == 1
    assert export_mock.call_args_list[0][0][1] == CURRENT_TEST_STRATEGY
    assert export_mock.call_args_list[0][0][2].name == "strategy_test_v3.json"


def test_params_print(capsys):
    params = {
        "buy": {"buy_rsi": 30},
        "sell": {"sell_rsi": 70},
    }
    non_optimized = {
        "buy": {"buy_adx": 44},
        "sell": {"sell_adx": 65},
        "stoploss": {
            "stoploss": -0.05,
        },
        "roi": {
            "0": 0.05,
            "20": 0.01,
        },
        "trailing": {
            "trailing_stop": False,
            "trailing_stop_positive": 0.05,
            "trailing_stop_positive_offset": 0.1,
            "trailing_only_offset_is_reached": True,
        },
        "max_open_trades": {"max_open_trades": 5},
    }
    HyperoptTools._params_pretty_print(params, "buy", "无标题", non_optimized)

    captured = capsys.readouterr()
    assert re.search("# 无标题", captured.out)
    assert re.search('"buy_rsi": 30,\n', captured.out)
    assert re.search('"buy_adx": 44,  # 从.*加载的值\n', captured.out)
    assert not re.search("sell", captured.out)

    HyperoptTools._params_pretty_print(params, "sell", "卖出标题", non_optimized)
    captured = capsys.readouterr()
    assert re.search("# 卖出标题", captured.out)
    assert re.search('"sell_rsi": 70,\n', captured.out)
    assert re.search('"sell_adx": 65,  # 从.*加载的值\n', captured.out)

    HyperoptTools._params_pretty_print(params, "roi", "ROI表格：", non_optimized)
    captured = capsys.readouterr()
    assert re.search("# ROI表格：  # 从.*加载的值\n", captured.out)
    assert re.search("minimal_roi = {\n", captured.out)
    assert re.search('"20": 0.01\n', captured.out)

    HyperoptTools._params_pretty_print(params, "trailing", "追踪止损：", non_optimized)
    captured = capsys.readouterr()
    assert re.search("# 追踪止损：", captured.out)
    assert re.search("trailing_stop = False  # 从.*加载的值\n", captured.out)
    assert re.search("trailing_stop_positive = 0.05  # 从.*加载的值\n", captured.out)
    assert re.search("trailing_stop_positive_offset = 0.1  # 从.*加载的值\n", captured.out)
    assert re.search("trailing_only_offset_is_reached = True  # 从.*加载的值\n", captured.out)

    HyperoptTools._params_pretty_print(params, "max_open_trades", "最大开仓数：", non_optimized)
    captured = capsys.readouterr()

    assert re.search("# 最大开仓数：", captured.out)
    assert re.search("max_open_trades = 5  # 从.*加载的值\n", captured.out)


def test_hyperopt_serializer():
    assert isinstance(hyperopt_serializer(np.int_(5)), int)
    assert isinstance(hyperopt_serializer(np.bool_(True)), bool)
    assert isinstance(hyperopt_serializer(np.bool_(False)), bool)