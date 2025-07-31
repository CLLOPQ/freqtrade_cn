# pragma pylint: disable=missing-docstring, W0212, line-too-long, C0103, unused-argument
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

from freqtrade.commands.optimize_commands import start_recursive_analysis
from freqtrade.data.history import get_timerange
from freqtrade.exceptions import OperationalException
from freqtrade.optimize.analysis.recursive import RecursiveAnalysis
from freqtrade.optimize.analysis.recursive_helpers import RecursiveAnalysisSubFunctions
from tests.conftest import EXMS, get_args, log_has_re, patch_exchange


@pytest.fixture
def recursive_conf(default_conf_usdt, tmp_path):
    default_conf_usdt["user_data_dir"] = tmp_path
    default_conf_usdt["timerange"] = "20220101-20220501"

    default_conf_usdt["strategy_path"] = str(Path(__file__).parent.parent / "strategy/strats")
    default_conf_usdt["strategy"] = "strategy_test_v3_recursive_issue"
    default_conf_usdt["pairs"] = ["UNITTEST/USDT"]
    default_conf_usdt["startup_candle"] = [100]
    return default_conf_usdt


def test_start_recursive_analysis(mocker):
    single_mock = MagicMock()
    text_table_mock = MagicMock()
    mocker.patch.multiple(
        "freqtrade.optimize.analysis.recursive_helpers.RecursiveAnalysisSubFunctions",
        initialize_single_recursive_analysis=single_mock,
        text_table_recursive_analysis_instances=text_table_mock,
    )
    args = [
        "recursive-analysis",
        "--strategy",
        "strategy_test_v3_recursive_issue",
        "--strategy-path",
        str(Path(__file__).parent.parent / "strategy/strats"),
        "--pairs",
        "UNITTEST/BTC",
        "--timerange",
        "20220101-20220201",
    ]
    pargs = get_args(args)
    pargs["config"] = None

    start_recursive_analysis(pargs)
    assert single_mock.call_count == 1
    assert text_table_mock.call_count == 1

    single_mock.reset_mock()

    # 缺少时间范围
    args = [
        "recursive-analysis",
        "--strategy",
        "strategy_test_v3_with_recursive_bias",
        "--strategy-path",
        str(Path(__file__).parent.parent / "strategy/strats"),
        "--pairs",
        "UNITTEST/BTC",
    ]
    pargs = get_args(args)
    pargs["config"] = None
    with pytest.raises(OperationalException, match=r"请设置时间范围\..*"):
        start_recursive_analysis(pargs)


def test_recursive_helper_no_strategy_defined(recursive_conf):
    conf = deepcopy(recursive_conf)
    conf["pairs"] = ["UNITTEST/USDT"]
    del conf["strategy"]
    with pytest.raises(OperationalException, match=r"未指定策略"):
        RecursiveAnalysisSubFunctions.start(conf)


def test_recursive_helper_start(recursive_conf, mocker) -> None:
    single_mock = MagicMock()
    text_table_mock = MagicMock()
    mocker.patch.multiple(
        "freqtrade.optimize.analysis.recursive_helpers.RecursiveAnalysisSubFunctions",
        initialize_single_recursive_analysis=single_mock,
        text_table_recursive_analysis_instances=text_table_mock,
    )
    RecursiveAnalysisSubFunctions.start(recursive_conf)
    assert single_mock.call_count == 1
    assert text_table_mock.call_count == 1

    single_mock.reset_mock()
    text_table_mock.reset_mock()


def test_recursive_helper_text_table_recursive_analysis_instances(recursive_conf):
    dict_diff = dict()
    dict_diff["rsi"] = {}
    dict_diff["rsi"][100] = "0.078%"

    strategy_obj = {
        "name": "strategy_test_v3_recursive_issue",
        "location": Path(recursive_conf["strategy_path"], f"{recursive_conf['strategy']}.py"),
    }

    instance = RecursiveAnalysis(recursive_conf, strategy_obj)
    instance.dict_recursive = dict_diff
    data = RecursiveAnalysisSubFunctions.text_table_recursive_analysis_instances([instance])

    # 检查信号太少的测试的行内容
    assert data[0][0] == "rsi"
    assert data[0][1] == "0.078%"
    assert len(data[0]) == 2

    # 现在检查没有问题的情况
    dict_diff = dict()
    instance = RecursiveAnalysis(recursive_conf, strategy_obj)
    instance.dict_recursive = dict_diff
    data = RecursiveAnalysisSubFunctions.text_table_recursive_analysis_instances([instance])
    assert len(data) == 0


def test_initialize_single_recursive_analysis(recursive_conf, mocker, caplog):
    mocker.patch("freqtrade.data.history.get_timerange", get_timerange)
    patch_exchange(mocker)
    mocker.patch(
        "freqtrade.plugins.pairlistmanager.PairListManager.whitelist",
        PropertyMock(return_value=["UNITTEST/BTC"]),
    )
    recursive_conf["pairs"] = ["UNITTEST/BTC"]

    recursive_conf["timeframe"] = "5m"
    recursive_conf["timerange"] = "20180119-20180122"
    start_mock = mocker.patch("freqtrade.optimize.analysis.recursive.RecursiveAnalysis.start")
    strategy_obj = {
        "name": "strategy_test_v3_recursive_issue",
        "location": Path(recursive_conf["strategy_path"], f"{recursive_conf['strategy']}.py"),
    }

    instance = RecursiveAnalysisSubFunctions.initialize_single_recursive_analysis(
        recursive_conf, strategy_obj
    )
    assert log_has_re(r"开始对.*进行递归测试\.", caplog)
    assert start_mock.call_count == 1

    assert instance.strategy_obj["name"] == "strategy_test_v3_recursive_issue"


@pytest
@pytest.mark.parametrize("scenario", ["no_bias", "bias1", "bias2"])
def test_recursive_biased_strategy(recursive_conf, mocker, caplog, scenario) -> None:
    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.get_fee", return_value=0.0)
    mocker.patch("freqtrade.data.history.get_timerange", get_timerange)
    mocker.patch(
        "freqtrade.plugins.pairlistmanager.PairListManager.whitelist",
        PropertyMock(return_value=["UNITTEST/BTC"]),
    )
    recursive_conf["pairs"] = ["UNITTEST/BTC"]

    recursive_conf["timeframe"] = "5m"
    recursive_conf["timerange"] = "20180119-20180122"
    recursive_conf["startup_candle"] = [100]

    # 修补场景参数以方便选择
    mocker.patch(
        "freqtrade.strategy.hyper.HyperStrategyMixin.load_params_from_file",
        return_value={"params": {"buy": {"scenario": scenario}}},
    )

    strategy_obj = {"name": "strategy_test_v3_recursive_issue"}
    instance = RecursiveAnalysis(recursive_conf, strategy_obj)
    instance.start()
    # 断言初始化正确
    assert log_has_re(f"策略参数: scenario = {scenario}", caplog)

    if scenario == "bias2":
        assert log_has_re("=> 在指标rsi中发现前瞻偏差", caplog)
    diff_pct = abs(float(instance.dict_recursive["rsi"][100].replace("%", "")))
    # 检查无偏差策略
    if scenario == "no_bias":
        assert diff_pct < 0.01
    # 检查有偏差策略
    elif scenario in ("bias1", "bias2"):
        assert diff_pct >= 0.01
