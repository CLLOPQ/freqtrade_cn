import logging
import time
from datetime import timedelta
from unittest.mock import MagicMock, PropertyMock

import pytest
import time_machine

from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import State
from freqtrade.worker import Worker
from tests.conftest import EXMS, get_patched_worker, log_has, log_has_re


def test_worker_state(mocker, default_conf, markets) -> None:
    """测试工作器状态初始化"""
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets))
    worker = get_patched_worker(mocker, default_conf)
    assert worker.freqtrade.state is State.RUNNING

    default_conf.pop("initial_state")
    worker = Worker(args=None, config=default_conf)
    assert worker.freqtrade.state is State.STOPPED


def test_worker_running(mocker, default_conf, caplog) -> None:
    """测试工作器运行状态"""
    mock_throttle = MagicMock()
    mocker.patch("freqtrade.worker.Worker._throttle", mock_throttle)
    mocker.patch("freqtrade.persistence.Trade.stoplossloss_reinitialization", MagicMock())

    worker = get_patched_worker(mocker, default_conf)

    state = worker._worker(old_state=None)
    assert state is State.RUNNING
    assert log_has("Changing state to: RUNNING", caplog)
    assert mock_throttle.call_count == 1
    # 检查策略已加载载并接收到数据提供器对象
    assert worker.freqtrade.strategy
    assert worker.freqtrade.strategy.dp
    assert isinstance(worker.freqtrade.strategy.dp, DataProvider)


def test_worker_paused(mocker, default_conf, caplog) -> None:
    """测试工作器暂停状态"""
    mock_throttle = MagicMock()
    mocker.patch("freqtrade.worker.Worker._throttle", mock_throttle)
    mocker.patch("freqtrade.persistence.Trade.stoploss_reinitialization", MagicMock())

    worker = get_patched_worker(mocker, default_conf)

    worker.freqtrade.state = State.PAUSED
    state = worker._worker(old_state=State.RUNNING)

    assert state is State.PAUSED
    assert log_has("Changing state from RUNNING to: PAUSED", caplog)
    assert mock_throttle.call_count == 1
    # 检查策略已加载载并接收到数据提供器对象
    assert worker.freqtrade.strategy
    assert worker.freqtrade.strategy.dp
    assert isinstance(worker.freqtrade.strategy.dp, DataProvider)


def test_worker_stopped(mocker, default_conf, caplog) -> None:
    """测试工作器停止状态"""
    mock_throttle = MagicMock()
    mocker.patch("freqtrade.worker.Worker._throttle", mock_throttle)

    worker = get_patched_worker(mocker, default_conf)
    worker.freqtrade.state = State.STOPPED
    state = worker._worker(old_state=State.RUNNING)
    assert state is State.STOPPED
    assert log_has("Changing state from RUNNING to: STOPPED", caplog)
    assert mock_throttle.call_count == 1


@pytest.mark.parametrize(
    "old_state,target_state,startup_call,log_fragment",
    [
        (State.STOPPED, State.PAUSED, True, "Changing state from STOPPED to: PAUSED"),
        (State.RUNNING, State.PAUSED, False, "Changing state from RUNNING to: PAUSED"),
        (State.PAUSED, State.RUNNING, False, "Changing state from PAUSED to: RUNNING"),
        (State.PAUSED, State.STOPPED, False, "Changing state from PAUSED to: STOPPED"),
        (State.RELOAD_CONFIG, State.RUNNING, True, "Changing state from RELOAD_CONFIG to: RUNNING"),
        (
            State.RELOAD_CONFIG,
            State.STOPPED,
            False,
            "Changing state from RELOAD_CONFIG to: STOPPED",
        ),
    ],
)
def test_worker_lifecycle(
    mocker,
    default_conf,
    caplog,
    old_state,
    target_state,
    startup_call,
    log_fragment,
):
    """测试工作器生命周期状态转换"""
    mock_throttle = mocker.MagicMock()
    mocker.patch("freqtrade.worker.Worker._throttle", mock_throttle)
    mocker.patch("freqtrade.persistence.Trade.stoplossloss_reinitialization")
    startup = mocker.patch("freqtrade.freqtradebot.FreqtradeBot.startup")

    worker = get_patched_worker(mocker, default_conf)
    worker.freqtrade.state = target_state

    new_state = worker._worker(old_state=old_state)

    assert new_state is target_state
    assert log_has(log_fragment, caplog)
    assert mock_throttle.call_count == 1
    assert startup.call_count == (1 if startup_call else 0)

    # 对于任何何何应初始化策略的状态
    if target_state in (State.RUNNING, State.PAUSED):
        assert worker.freqtrade.strategy
        assert isinstance(worker.freqtrade.strategy.dp, DataProvider)
    else:
        assert new_state is State.STOPPED


def test_throttle(mocker, default_conf, caplog) -> None:
    """测试节流控制功能"""
    def throttled_func():
        return 42

    caplog.set_level(logging.DEBUG)
    worker = get_patched_worker(mocker, default_conf)

    start = time.time()
    result = worker._throttle(throttled_func, throttle_secs=0.1)
    end = time.time()

    assert result == 42
    assert 0.3 > end - start > 0.1
    assert log_has_re(r"Throttling with 'throttled_func\(\)': sleep for \d\.\d{2} s.*", caplog)

    result = worker._throttle(throttled_func, throttle_secs=-1)
    assert result == 42


def test_throttle_sleep_time(mocker, default_conf, caplog) -> None:
    """测试节流睡眠时间计算"""
    caplog.set_level(logging.DEBUG)
    worker = get_patched_worker(mocker, default_conf)
    sleep_mock = mocker.patch("freqtrade.worker.Worker._sleep")
    with time_machine.travel("2022-09-01 05:00:00 +00:00") as t:

        def throttled_func(x=1):
            t.shift(timedelta(seconds=x))
            return 42

        assert worker._throttle(throttled_func, throttle_secs=5) == 42
        # 这会将时钟向前移动1秒
        assert sleep_mock.call_count == 1
        assert 3.8 < sleep_mock.call_args[0][0] < 4.1

        sleep_mock.reset_mock()
        # 这会将时钟向前移动1秒
        assert worker._throttle(throttled_func, throttle_secs=10) == 42
        assert sleep_mock.call_count == 1
        assert 8.8 < sleep_mock.call_args[0][0] < 9.1

        sleep_mock.reset_mock()
        # 这会将时钟向前移动5秒，因此我们只需节流5秒
        assert worker._throttle(throttled_func, throttle_secs=10, x=5) == 42
        assert sleep_mock.call_count == 1
        assert 4.8 < sleep_mock.call_args[0][0] < 5.1

        t.move_to("2022-09-01 05:01:00 +00:00")
        sleep_mock.reset_mock()
        # 节流超过5分钟（1个时间框架）
        assert worker._throttle(throttled_func, throttle_secs=400, x=5) == 42
        assert sleep_mock.call_count == 1
        assert 394.8 < sleep_mock.call_args[0][0] < 395.1

        t.move_to("2022-09-01 05:01:00 +00:00")

        sleep_mock.reset_mock()
        # 节流超过5分钟（1个时间框架）
        assert (
            worker._throttle(
                throttled_func, throttle_secs=400, timeframe="5m", timeframe_offset=0.4, x=5
            )
            == 42
        )
        assert sleep_mock.call_count == 1
        # 300（5分钟）- 60（1分钟 - 见上面设置的时间）- 5（throttled_func的执行时间）= 235
        assert 235.2 < sleep_mock.call_args[0][0] < 235.6

        t.move_to("2022-09-01 05:04:51 +00:00")
        sleep_mock.reset_mock()
        # 偏移5秒，因此我们处于"K线"和"K线偏移"之间的最佳点
        # 这不应该进行节流迭代，以避免延迟获取K线
        assert (
            worker._throttle(
                throttled_func, throttle_secs=10, timeframe="5m", timeframe_offset=5, x=1.2
            )
            == 42
        )
        assert sleep_mock.call_count == 1
        # 由于较高的时间框架偏移，时间略大于节流秒数
        assert 11.1 < sleep_mock.call_args[0][0] < 13.2


def test_throttle_with_assets(mocker, default_conf) -> None:
    """测试带参数的节流功能"""
    def throttled_func(nb_assets=-1):
        return nb_assets

    worker = get_patched_worker(mocker, default_conf)

    result = worker._throttle(throttled_func, throttle_secs=0.1, nb_assets=666)
    assert result == 666

    result = worker._throttle(throttled_func, throttle_secs=0.1)
    assert result == -1


def test_worker_heartbeat_running(default_conf, mocker, caplog):
    """测试工作器运行状态下的心跳功能"""
    message = r"Bot heartbeat\. PID=.*state='RUNNING'"

    mock_throttle = MagicMock()
    mocker.patch("freqtrade.worker.Worker._throttle", mock_throttle)
    worker = get_patched_worker(mocker, default_conf)

    worker.freqtrade.state = State.RUNNING
    worker._worker(old_state=State.STOPPED)
    assert log_has_re(message, caplog)

    caplog.clear()
    # 间隔时间未到时不显示消息
    worker._worker(old_state=State.RUNNING)
    assert not log_has_re(message, caplog)

    caplog.clear()
    # 设置时钟 - 70秒
    worker._heartbeat_msg -= 70
    worker._worker(old_state=State.RUNNING)
    assert log_has_re(message, caplog)


def test_worker_heartbeat_stopped(default_conf, mocker, caplog):
    """测试工作器停止状态下的心跳功能"""
    message = r"Bot heartbeat\. PID=.*state='STOPPED'"

    mock_throttle = MagicMock()
    mocker.patch("freqtrade.worker.Worker._throttle", mock_throttle)
    worker = get_patched_worker(mocker, default_conf)

    worker.freqtrade.state = State.STOPPED
    worker._worker(old_state=State.RUNNING)
    assert log_has_re(message, caplog)

    caplog.clear()
    # 间隔时间未到时不显示消息
    worker._worker(old_state=State.STOPPED)
    assert not log_has_re(message, caplog)

    caplog.clear()
    # 设置时钟 - 70秒
    worker._heartbeat_msg -= 70
    worker._worker(old_state=State.STOPPED)
    assert log_has_re(message, caplog)
