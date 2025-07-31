import pytest

from freqtrade.exceptions import StrategyError
from freqtrade.persistence import Trade
from freqtrade.strategy.strategy_wrapper import strategy_safe_wrapper
from freqtrade.util.datetime_helpers import dt_now
from tests.conftest import create_mock_trades, log_has_re

from .strats.strategy_test_v3 import StrategyTestV3


@pytest.mark.parametrize(
    "error",
    [
        ValueError,
        KeyError,
        Exception,
    ],
)
def test_strategy_safe_wrapper_error(caplog, error):
    """测试策略安全包装器在发生错误时的行为"""
    def failing_method():
        raise error("这是一个错误。")

    # 测试错误是否被正确捕获并转换为StrategyError
    with pytest.raises(StrategyError, match=r"这是一个错误。"):
        strategy_safe_wrapper(failing_method, message="DeadBeef")()

    # 验证错误日志是否被正确记录
    assert log_has_re(r"DeadBeef.*", caplog)
    
    # 测试设置设置默认返回值
    ret = strategy_safe_wrapper(failing_method, message="DeadBeef", default_retval=True)()
    assert isinstance(ret, bool)
    assert ret

    # 测试错误抑制功能
    caplog.clear()
    ret = strategy_safe_wrapper(failing_method, message="DeadBeef", supress_error=True)()
    assert log_has_re(r"DeadBeef.*", caplog)


@pytest.mark.parametrize(
    "value", [1, 22, 55, True, False, {"a": 1, "b": "112"}, [1, 2, 3, 4], (4, 2, 3, 6)]
)
def test_strategy_safe_wrapper(value):
    """测试策略安全包装器在正常工作时的行为"""
    def working_method(argumentpassedin):
        return argumentpassedin

    # 调用包装有安全包装器的方法
    ret = strategy_safe_wrapper(working_method, message="DeadBeef")(value)

    # 验证返回值类型和内容是否正确
    assert isinstance(ret, type(value))
    assert ret == value


@pytest.mark.usefixtures("init_persistence")
def test_strategy_safe_wrapper_trade_copy(fee, mocker):
    """测试    测试策略安全包装器是否对Trade对象进行深拷贝，
    确保原始对象不会被修改
    """
    # 创建模拟交易
    create_mock_trades(fee)
    import freqtrade.strategy.strategy_wrapper as swm

    # 监视深拷贝函数
    deepcopy_mock = mocker.spy(swm, "deepcopy")

    # 获取一个打开的交易
    trade_ = Trade.get_open_trades()[0]
    strat = StrategyTestV3(config={})

    # 定义一个会修改交易对象的方法
    def working_method(trade):
        # 验证证交易有订单
        assert len(trade.orders) > 0
        assert trade.orders
        # 清空订单列表
        trade.orders = []
        assert len(trade.orders) == 0
        # 验证这是一个拷贝，不是原始对象
        assert id(trade_) != id(trade)
        return trade

    strat.working_method = working_method

    # 调用包装器方法
    ret = strategy_safe_wrapper(strat.working_method, message="DeadBeef")(trade=trade_)
    
    # 验证证返回值是Trade实例且是一个新对象
    assert isinstance(ret, Trade)
    assert id(trade_) != id(ret)
    # 验证证原始交易的订单没有被修改
    assert len(trade_.orders) > 0
    # 验证返回的交易订单已被清空
    assert len(ret.orders) == 0
    # 验证深拷贝被调用了一次
    assert deepcopy_mock.call_count == 1
    deepcopy_mock.reset_mock()

    # 调用未被覆盖的方法 - 不应深拷贝交易
    ret = strategy_safe_wrapper(strat.custom_entry_price, message="DeadBeef")(
        pair="ETH/USDT",
        trade=trade_,
        current_time=dt_now(),
        proposed_rate=0.5,
        entry_tag="",
        side="long",
    )

    # 验证深拷贝没有被调用
    assert deepcopy_mock.call_count == 0