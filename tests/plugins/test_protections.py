import random
from datetime import datetime, timedelta, timezone

import pytest

from freqtrade.enums import ExitType
from freqtrade.exceptions import OperationalException
from freqtrade.persistence import PairLocks, Trade
from freqtrade.persistence.trade_model import Order
from freqtrade.plugins.protectionmanager import ProtectionManager
from tests.conftest import get_patched_freqtradebot, log_has_re


AVAILABLE_PROTECTIONS = ["CooldownPeriod", "LowProfitPairs", "MaxDrawdown", "StoplossGuard"]


def generate_mock_trade(
    pair: str,
    fee: float,
    is_open: bool,
    exit_reason: str = ExitType.EXIT_SIGNAL,
    min_ago_open: int | None = None,
    min_ago_close: int | None = None,
    profit_rate: float = 0.9,
    is_short: bool = False,
):
    open_rate = random.random()

    trade = Trade(
        pair=pair,
        stake_amount=0.01,
        fee_open=fee,
        fee_close=fee,
        open_date=datetime.now(timezone.utc) - timedelta(minutes=min_ago_open or 200),
        close_date=datetime.now(timezone.utc) - timedelta(minutes=min_ago_close or 30),
        open_rate=open_rate,
        is_open=is_open,
        amount=0.01 / open_rate,
        exchange="binance",
        is_short=is_short,
        leverage=1,
    )

    trade.orders.append(
        Order(
            ft_order_side=trade.entry_side,
            order_id=f"{pair}-{trade.entry_side}-{trade.open_date}",
            ft_is_open=False,
            ft_pair=pair,
            ft_amount=trade.amount,
            ft_price=trade.open_rate,
            amount=trade.amount,
            filled=trade.amount,
            remaining=0,
            price=open_rate,
            average=open_rate,
            status="closed",
            order_type="market",
            side=trade.entry_side,
        )
    )
    if not is_open:
        close_price = open_rate * (2 - profit_rate if is_short else profit_rate)
        trade.orders.append(
            Order(
                ft_order_side=trade.exit_side,
                order_id=f"{pair}-{trade.exit_side}-{trade.close_date}",
                ft_is_open=False,
                ft_pair=pair,
                ft_amount=trade.amount,
                ft_price=trade.open_rate,
                amount=trade.amount,
                filled=trade.amount,
                remaining=0,
                price=close_price,
                average=close_price,
                status="closed",
                order_type="market",
                side=trade.exit_side,
            )
        )

    trade.recalc_open_trade_value()
    if not is_open:
        trade.close(close_price)
        trade.exit_reason = exit_reason

    Trade.session.add(trade)
    Trade.commit()
    return trade


def test_protectionmanager(mocker, default_conf):
    default_conf["_strategy_protections"] = [
        {"method": protection} for protection in AVAILABLE_PROTECTIONS
    ]
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    for handler in freqtrade.protections._protection_handlers:
        assert handler.name in AVAILABLE_PROTECTIONS
        if not handler.has_global_stop:
            assert handler.global_stop(datetime.now(timezone.utc), "*") is None
        if not handler.has_local_stop:
            assert handler.stop_per_pair("XRP/BTC", datetime.now.now(timezone.utc), "*") is None


@pytest.mark.parametrize(
    "protconf,expected",
    [
        ([], None),
        ([{"method": "StoplossGuard", "lookback_period": 2000, "stop_duration_candles": 10}], None),
        ([{"method": "StoplossGuard", "lookback_period_candles": 20, "stop_duration": 10}], None),
        (
            [
                {
                    "method": "StoplossGuard",
                    "lookback_period_candles": 20,
                    "lookback_period": 2000,
                    "stop_duration": 10,
                }
            ],
            r"保护措施必须指定`lookback_period`（分钟）或`lookback_period_candles`（K线数量），但不能同时指定。",
        ),
        (
            [
                {
                    "method": "StoplossGuard",
                    "lookback_period": 20,
                    "stop_duration": 10,
                    "stop_duration_candles": 10,
                }
            ],
            r"保护措施措施必须指定`stop_duration`（分钟）或`stop_duration_candles`（K线数量），但不能同时指定。",
        ),
        (
            [
                {
                    "method": "StoplossGuard",
                    "lookback_period": 20,
                    "stop_duration": 10,
                    "unlock_at": "20:02",
                }
            ],
            r"保护措施措施必须指定`unlock_at`、`stop_duration`或`stop_duration_candles`中的一个，但不能同时指定多个。",
        ),
        (
            [{"method": "StoplossGuard", "lookback_period_candles": 20, "unlock_at": "20:02"}],
            None,
        ),
        (
            [{"method": "StoplossGuard", "lookback_period_candles": 20, "unlock_at": "55:102"}],
            "unlock_at的日期格式无效：55:102。",
        ),
    ],
)
def test_validate_protections(protconf, expected):
    if expected:
        with pytest.raises(OperationalException, match=expected):
            ProtectionManager.validate_protections(protconf)
    else:
        ProtectionManager.validate_protections(protconf)


@pytest.mark.parametrize(
    "timeframe,expected_lookback,expected_stop,protconf",
    [
        (
            "1m",
            20,
            10,
            [{"method": "StoplossGuard", "lookback_period_candles": 20, "stop_duration": 10}],
        ),
        (
            "5m",
            100,
            15,
            [{"method": "StoplossGuard", "lookback_period_candles": 20, "stop_duration": 15}],
        ),
        (
            "1h",
            1200,
            40,
            [{"method": "StoplossGuard", "lookback_period_candles": 20, "stop_duration": 40}],
        ),
        (
            "1d",
            1440,
            5,
            [{"method": "StoplossGuard", "lookback_period_candles": 1, "stop_duration": 5}],
        ),
        (
            "1m",
            20,
            5,
            [{"method": "StoplossGuard", "lookback_period": 20, "stop_duration_candles": 5}],
        ),
        (
            "5m",
            15,
            25,
            [{"method": "StoplossGuard", "lookback_period": 15, "stop_duration_candles": 5}],
        ),
        (
            "1h",
            50,
            600,
            [{"method": "StoplossGuard", "lookback_period": 50, "stop_duration_candles": 10}],
        ),
        (
            "1h",
            60,
            540,
            [{"method": "StoplossGuard", "lookback_period_candles": 1, "stop_duration_candles": 9}],
        ),
        (
            "1m",
            20,
            "01:00",
            [{"method": "StoplossGuard", "lookback_period_candles": 20, "unlock_at": "01:00"}],
        ),
        (
            "5m",
            100,
            "02:00",
            [{"method": "StoplossGuard", "lookback_period_candles": 20, "unlock_at": "02:00"}],
        ),
        (
            "1h",
            1200,
            "03:00",
            [{"method": "StoplossGuard", "lookback_period_candles": 20, "unlock_at": "03:00"}],
        ),
        (
            "1d",
            1440,
            "04:00",
            [{"method": "StoplossGuard", "lookback_period_candles": 1, "unlock_at": "04:00"}],
        ),
    ],
)
def test_protections_init(default_conf, timeframe, expected_lookback, expected_stop, protconf):
    """
    测试不同配置下保护措施的初始化，包括unlock_at。
    """
    default_conf["timeframe"] = timeframe
    man = ProtectionManager(default_conf, protconf)
    assert len(man._protection_handlers) == len(protconf)
    assert man._protection_handlers[0]._lookback_period == expected_lookback
    if isinstance(expected_stop, int):
        assert man._protection_handlers[0]._stop_duration == expected_stop
    else:
        assert man._protection_handlers[0]._unlock_at == expected_stop


@pytest.mark.parametrize("is_short", [False, True])
@pytest.mark.usefixtures("init_persistence")
def test_stoploss_guard(mocker, default_conf, fee, caplog, is_short):
    # 对多空双方都有效
    default_conf["_strategy_protections"] = [
        {"method": "StoplossGuard", "lookback_period": 60, "stop_duration": 40, "trade_limit": 3}
    ]
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    message = r"由于.*停止交易"
    assert not freqtrade.protections.global_stop()
    assert not log_has_re(message, caplog)
    caplog.clear()

    generate_mock_trade(
        "XRP/BTC",
        fee.return_value,
        False,
        exit_reason=ExitType.STOP_LOSS.value,
        min_ago_open=200,
        min_ago_close=30,
        is_short=is_short,
    )

    assert not freqtrade.protections.global_stop()
    assert not log_has_re(message, caplog)
    caplog.clear()
    # 这笔交易不计入，因为关闭时间太久了
    generate_mock_trade(
        "BCH/BTC",
        fee.return_value,
        False,
        exit_reason=ExitType.STOP_LOSS.value,
        min_ago_open=250,
        min_ago_close=100,
        is_short=is_short,
    )

    generate_mock_trade(
        "ETH/BTC",
        fee.return_value,
        False,
        exit_reason=ExitType.STOP_LOSS.value,
        min_ago_open=240,
        min_ago_close=30,
        is_short=is_short,
    )
    # 3笔已关闭交易 - 但第2笔关闭时间太久了
    assert not freqtrade.protections.global_stop()
    assert not log_has_re(message, caplog)
    caplog.clear()

    generate_mock_trade(
        "LTC/BTC",
        fee.return_value,
        False,
        exit_reason=ExitType.STOP_LOSS.value,
        min_ago_open=180,
        min_ago_close=30,
        is_short=is_short,
    )

    assert freqtrade.protections.global_stop()
    assert log_has_re(message, caplog)
    assert PairLocks.is_global_lock()

    # 测试锁定期后5分钟 - 应该应该尝试重新锁定交易对，但结束时间应该是之前的结束时间
    end_time = PairLocks.get_pair_longest_lock("*").lock_end_time + timedelta(minutes=5)
    freqtrade.protections.global_stop(end_time)
    assert not PairLocks.is_global_lock(end_time)


@pytest.mark.parametrize("only_per_pair", [False, True])
@pytest.mark.parametrize("only_per_side", [False, True])
@pytest.mark.usefixtures("init_persistence")
def test_stoploss_guard_perpair(mocker, default_conf, fee, caplog, only_per_pair, only_per_side):
    default_conf["_strategy_protections"] = [
        {
            "method": "StoplossGuard",
            "lookback_period": 60,
            "trade_limit": 2,
            "stop_duration": 60,
            "only_per_pair": only_per_pair,
            "only_per_side": only_per_side,
        }
    ]
    check_side = "long" if only_per_side else "*"
    is_short = False
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    message = r"由于.*停止交易"
    pair = "XRP/BTC"
    assert not freqtrade.protections.stop_per_pair(pair)
    assert not freqtrade.protections.global_stop()
    assert not log_has_re(message, caplog)
    caplog.clear()

    generate_mock_trade(
        pair,
        fee.return_value,
        False,
        exit_reason=ExitType.STOP_LOSS.value,
        min_ago_open=200,
        min_ago_close=30,
        profit_rate=0.9,
        is_short=is_short,
    )

    assert not freqtrade.protections.stop_per_pair(pair)
    assert not freqtrade.protections.global_stop()
    assert not log_has_re(message, caplog)
    caplog.clear()
    # 这笔交易不计入，因为关闭时间太久了
    generate_mock_trade(
        pair,
        fee.return_value,
        False,
        exit_reason=ExitType.STOP_LOSS.value,
        min_ago_open=250,
        min_ago_close=100,
        profit_rate=0.9,
        is_short=is_short,
    )
    # 这笔交易不计入单交易对停止，因为是错误的交易对
    generate_mock_trade(
        "ETH/BTC",
        fee.return_value,
        False,
        exit_reason=ExitType.STOP_LOSS.value,
        min_ago_open=240,
        min_ago_close=30,
        profit_rate=0.9,
        is_short=is_short,
    )
    # 3笔已关闭交易 - 但第2笔关闭时间太久了
    assert not freqtrade.protections.stop_per_pair(pair)
    assert freqtrade.protections.global_stop() != only_per_pair
    if not only_per_pair:
        assert log_has_re(message, caplog)
    else:
        assert not log_has_re(message, caplog)

    caplog.clear()

    # 这笔交易可能不计入，因为方向错误
    generate_mock_trade(
        pair,
        fee.return_value,
        False,
        exit_reason=ExitType.STOP_LOSS.value,
        min_ago_open=150,
        min_ago_close=25,
        profit_rate=0.9,
        is_short=not is_short,
    )
    freqtrade.protections.stop_per_pair(pair)
    assert freqtrade.protections.global_stop() != only_per_pair
    assert PairLocks.is_pair_locked(pair, side=check_side) != (only_per_side and only_per_pair)
    assert PairLocks.is_global_lock(side=check_side) != only_per_pair
    if only_per_side:
        assert not PairLocks.is_pair_locked(pair, side="*")
        assert not PairLocks.is_global_lock(side="*")

    caplog.clear()

    # 第2笔计入的交易，正确的交易对
    generate_mock_trade(
        pair,
        fee.return_value,
        False,
        exit_reason=ExitType.STOP_LOSS.value,
        min_ago_open=180,
        min_ago_close=31,
        profit_rate=0.9,
        is_short=is_short,
    )

    freqtrade.protections.stop_per_pair(pair)
    assert freqtrade.protections.global_stop() != only_per_pair
    assert PairLocks.is_pair_locked(pair, side=check_side)
    assert PairLocks.is_global_lock(side=check_side) != only_per_pair
    if only_per_side:
        assert not PairLocks.is_pair_locked(pair, side="*")
        assert not PairLocks.is_global_lock(side="*")


@pytest.mark.usefixtures("init_persistence")
def test_CooldownPeriod(mocker, default_conf, fee, caplog):
    default_conf["_strategy_protections"] = [
        {
            "method": "CooldownPeriod",
            "stop_duration": 60,
        }
    ]
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    message = r"由于.*停止交易"
    assert not freqtrade.protections.global_stop()
    assert not freqtrade.protections.stop_per_pair("XRP/BTC")

    assert not log_has_re(message, caplog)
    caplog.clear()

    generate_mock_trade(
        "XRP/BTC",
        fee.return_value,
        False,
        exit_reason=ExitType.STOP_LOSS.value,
        min_ago_open=200,
        min_ago_close=30,
    )

    assert not freqtrade.protections.global_stop()
    assert freqtrade.protections.stop_per_pair("XRP/BTC")
    assert PairLocks.is_pair_locked("XRP/BTC")
    assert not PairLocks.is_global_lock()

    generate_mock_trade(
        "ETH/BTC",
        fee.return_value,
        False,
        exit_reason=ExitType.ROI.value,
        min_ago_open=205,
        min_ago_close=35,
    )

    assert not freqtrade.protections.global_stop()
    assert not PairLocks.is_pair_locked("ETH/BTC")
    assert freqtrade.protections.stop_per_pair("ETH/BTC")
    assert PairLocks.is_pair_locked("ETH/BTC")
    assert not PairLocks.is_global_lock()


@pytest.mark.usefixtures("init_persistence")
def test_CooldownPeriod_unlock_at(mocker, default_conf, fee, caplog, time_machine):
    default_conf["_strategy_protections"] = [
        {
            "method": "CooldownPeriod",
            "unlock_at": "05:00",
        }
    ]
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    message = r"由于.*停止交易"
    assert not freqtrade.protections.global_stop()
    assert not freqtrade.protections.stop_per_pair("XRP/BTC")

    assert not log_has_re(message, caplog)
    caplog.clear()

    start_dt = datetime(2024, 5, 2, 0, 30, 0, tzinfo=timezone.utc)
    time_machine.move_to(start_dt, tick=False)

    generate_mock_trade(
        "XRP/BTC",
        fee.return_value,
        False,
        exit_reason=ExitType.STOP_LOSS.value,
        min_ago_open=20,
        min_ago_close=10,
    )

    assert not freqtrade.protections.global_stop()
    assert freqtrade.protections.stop_per_pair("XRP/BTC")
    assert PairLocks.is_pair_locked("XRP/BTC")
    assert not PairLocks.is_global_lock()

    # 移动时间到"4:30"
    time_machine.move_to(start_dt + timedelta(hours=4), tick=False)
    assert PairLocks.is_pair_locked("XRP/BTC")
    assert not PairLocks.is_global_lock()

    # 移动时间到"5:00之后"
    time_machine.move_to(start_dt + timedelta(hours=5), tick=False)
    assert not PairLocks.is_pair_locked("XRP/BTC")
    assert not PairLocks.is_global_lock()

    # 强制滚动到第二天
    start_dt = datetime(2024, 5, 2, 22, 00, 0, tzinfo=timezone.utc)
    time_machine.move_to(start_dt, tick=False)
    generate_mock_trade(
        "ETH/BTC",
        fee.return_value,
        False,
        exit_reason=ExitType.ROI.value,
        min_ago_open=20,
        min_ago_close=10,
    )

    assert not freqtrade.protections.global_stop()
    assert not PairLocks.is_pair_locked("ETH/BTC")
    assert freqtrade.protections.stop_per_pair("ETH/BTC")
    assert PairLocks.is_pair_locked("ETH/BTC")
    assert not PairLocks.is_global_lock()
    # 移动到23:00
    time_machine.move_to(start_dt + timedelta(hours=1), tick=False)
    assert PairLocks.is_pair_locked("ETH/BTC")
    assert not PairLocks.is_global_lock()

    # 移动到04:59（应该仍然锁定）
    time_machine.move_to(start_dt + timedelta(hours=6, minutes=59), tick=False)
    assert PairLocks.is_pair_locked("ETH/BTC")
    assert not PairLocks.is_global_lock()

    # 移动到05:01（应该仍然锁定 - 直到05:05的K线结束才解锁）
    time_machine.move_to(start_dt + timedelta(hours=7, minutes=1), tick=False)

    assert PairLocks.is_pair_locked("ETH/BTC")
    assert not PairLocks.is_global_lock()

    # 移动到05:01（已解锁）
    time_machine.move_to(start_dt + timedelta(hours=7, minutes=5), tick=False)

    assert not PairLocks.is_pair_locked("ETH/BTC")
    assert not PairLocks.is_global_lock()


@pytest.mark.parametrize("only_per_side", [False, True])
@pytest.mark.usefixtures("init_persistence")
def test_LowProfitPairs(mocker, default_conf, fee, caplog, only_per_side):
    default_conf["_strategy_protections"] = [
        {
            "method": "LowProfitPairs",
            "lookback_period": 400,
            "stop_duration": 60,
            "trade_limit": 2,
            "required_profit": 0.0,
            "only_per_side": only_per_side,
        }
    ]
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    message = r"由于.*停止交易"
    assert not freqtrade.protections.global_stop()
    assert not freqtrade.protections.stop_per_pair("XRP/BTC")

    assert not log_has_re(message, caplog)
    caplog.clear()

    generate_mock_trade(
        "XRP/BTC",
        fee.return_value,
        False,
        exit_reason=ExitType.STOP_LOSS.value,
        min_ago_open=800,
        min_ago_close=450,
        profit_rate=0.9,
    )

    Trade.commit()
    # 1笔交易不锁定
    assert not freqtrade.protections.global_stop()
    assert not freqtrade.protections.stop_per_pair("XRP/BTC")
    assert not PairLocks.is_pair_locked("XRP/BTC")
    assert not PairLocks.is_global_lock()

    generate_mock_trade(
        "XRP/BTC",
        fee.return_value,
        False,
        exit_reason=ExitType.STOP_LOSS.value,
        min_ago_open=200,
        min_ago_close=120,
        profit_rate=0.9,
    )

    Trade.commit()
    # 1笔交易不锁定（第一笔交易在回溯期外）
    assert not freqtrade.protections.global_stop()
    assert not freqtrade.protections.stop_per_pair("XRP/BTC")
    assert not PairLocks.is_pair_locked("XRP/BTC")
    assert not PairLocks.is_global_lock()

    # 添加盈利交易
    generate_mock_trade(
        "XRP/BTC",
        fee.return_value,
        False,
        exit_reason=ExitType.ROI.value,
        min_ago_open=20,
        min_ago_close=10,
        profit_rate=1.15,
        is_short=True,
    )
    Trade.commit()
    assert freqtrade.protections.stop_per_pair("XRP/BTC") != only_per_side
    assert not PairLocks.is_pair_locked("XRP/BTC", side="*")
    assert PairLocks.is_pair_locked("XRP/BTC", side="long") == only_per_side

    generate_mock_trade(
        "XRP/BTC",
        fee.return_value,
        False,
        exit_reason=ExitType.STOP_LOSS.value,
        min_ago_open=110,
        min_ago_close=21,
        profit_rate=0.8,
    )
    Trade.commit()

    # 由于第2笔交易锁定
    assert freqtrade.protections.global_stop() != only_per_side
    assert freqtrade.protections.stop_per_pair("XRP/BTC") != only_per_side
    assert PairLocks.is_pair_locked("XRP/BTC", side="long")
    assert PairLocks.is_pair_locked("XRP/BTC", side="*") != only_per_side
    assert not PairLocks.is_global_lock()
    Trade.commit()


@pytest.mark.usefixtures("init_persistence")
def test_MaxDrawdown(mocker, default_conf, fee, caplog):
    default_conf["_strategy_protections"] = [
        {
            "method": "MaxDrawdown",
            "lookback_period": 1000,
            "stop_duration": 60,
            "trade_limit": 3,
            "max_allowed_drawdown": 0.15,
        }
    ]
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    message = r"由于最大回撤.*停止交易"

    assert not freqtrade.protections.global_stop()
    assert not freqtrade.protections.stop_per_pair("XRP/BTC")
    caplog.clear()

    generate_mock_trade(
        "XRP/BTC",
        fee.return_value,
        False,
        exit_reason=ExitType.STOP_LOSS.value,
        min_ago_open=1000,
        min_ago_close=900,
        profit_rate=1.1,
    )
    generate_mock_trade(
        "ETH/BTC",
        fee.return_value,
        False,
        exit_reason=ExitType.STOP_LOSS.value,
        min_ago_open=1000,
        min_ago_close=900,
        profit_rate=1.1,
    )
    generate_mock_trade(
        "NEO/BTC",
        fee.return_value,
        False,
        exit_reason=ExitType.STOP_LOSS.value,
        min_ago_open=1000,
        min_ago_close=900,
        profit_rate=1.1,
    )
    Trade.commit()
    # 还没有亏损交易...所以最大回撤会抛出异常
    assert not freqtrade.protections.global_stop()
    assert not freqtrade.protections.stop_per_pair("XRP/BTC")

    generate_mock_trade(
        "XRP/BTC",
        fee.return_value,
        False,
        exit_reason=ExitType.STOP_LOSS.value,
        min_ago_open=500,
        min_ago_close=400,
        profit_rate=0.9,
    )
    # 一笔交易不锁定
    assert not freqtrade.protections.global_stop()
    assert not freqtrade.protections.stop_per_pair("XRP/BTC")
    assert not PairLocks.is_pair_locked("XRP/BTC")
    assert not PairLocks.is_global_lock()

    generate_mock_trade(
        "XRP/BTC",
        fee.return_value,
        False,
        exit_reason=ExitType.STOP_LOSS.value,
        min_ago_open=1200,
        min_ago_close=1100,
        profit_rate=0.5,
    )
    Trade.commit()

    # 一笔交易不锁定（第2笔交易在回溯期外）
    assert not freqtrade.protections.global_stop()
    assert not freqtrade.protections.stop_per_pair("XRP/BTC")
    assert not PairLocks.is_pair_locked("XRP/BTC")
    assert not PairLocks.is_global_lock()
    assert not log_has_re(message, caplog)

    # 盈利交易...（不应锁定，不改变回撤！）
    generate_mock_trade(
        "XRP/BTC",
        fee.return_value,
        False,
        exit_reason=ExitType.ROI.value,
        min_ago_open=320,
        min_ago_close=410,
        profit_rate=1.5,
    )
    Trade.commit()
    assert not freqtrade.protections.global_stop()
    assert not PairLocks.is_global_lock()

    caplog.clear()

    # 添加额外的亏损交易，导致亏损>15%
    generate_mock_trade(
        "XRP/BTC",
        fee.return_value,
        False,
        exit_reason=ExitType.ROI.value,
        min_ago_open=20,
        min_ago_close=10,
        profit_rate=0.8,
    )
    Trade.commit()
    assert not freqtrade.protections.stop_per_pair("XRP/BTC")
    # 不支持局部锁定
    assert not PairLocks.is_pair_locked("XRP/BTC")
    assert freqtrade.protections.global_stop()
    assert PairLocks.is_global_lock()
    assert log_has_re(message, caplog)


@pytest.mark.parametrize(
    "protectionconf,desc_expected,exception_expected",
    [
        (
            {
                "method": "StoplossGuard",
                "lookback_period": 60,
                "trade_limit": 2,
                "stop_duration": 60,
            },
            "[{'StoplossGuard': 'StoplossGuard - 频繁止损保护，60分钟内有2次止损且利润<0.00%。'}]",
            None,
        ),
        (
            {"method": "CooldownPeriod", "stop_duration": 60},
            "[{'CooldownPeriod': 'CooldownPeriod - 冷却期60分钟。'}]",
            None,
        ),
        (
            {"method": "LowProfitPairs", "lookback_period": 60, "stop_duration": 60},
            "[{'LowProfitPairs': 'LowProfitPairs - 低利润保护，锁定60分钟内利润<0.0的交易对。'}]",
            None,
        ),
        (
            {"method": "MaxDrawdown", "lookback_period": 60, "stop_duration": 60},
            "[{'MaxDrawdown': 'MaxDrawdown - 最大回撤保护，60分钟内回撤>0.0时停止交易。'}]",
            None,
        ),
        (
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 12,
                "trade_limit": 2,
                "required_profit": -0.05,
                "stop_duration": 60,
            },
            "[{'StoplossGuard': 'StoplossGuard - 频繁止损保护，12根K线内有2次止损且利润<-5.00%。'}]",
            None,
        ),
        (
            {"method": "CooldownPeriod", "stop_duration_candles": 5},
            "[{'CooldownPeriod': 'CooldownPeriod - 冷却期5根K线。'}]",
            None,
        ),
        (
            {"method": "LowProfitPairs", "lookback_period_candles": 11, "stop_duration": 60},
            "[{'LowProfitPairs': 'LowProfitPairs - 低利润保护，锁定11根K线内利润<0.0的交易对。'}]",
            None,
        ),
        (
            {"method": "MaxDrawdown", "lookback_period_candles": 20, "stop_duration": 60},
            "[{'MaxDrawdown': 'MaxDrawdown - 最大回撤保护，20根K线内回撤>0.0时停止交易。'}]",
            None,
        ),
        (
            {
                "method": "CooldownPeriod",
                "unlock_at": "01:00",
            },
            "[{'CooldownPeriod': 'CooldownPeriod - 冷却期至01:00。'}]",
            None,
        ),
        (
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 12,
                "trade_limit": 2,
                "required_profit": -0.05,
                "unlock_at": "01:00",
            },
            "[{'StoplossGuard': 'StoplossGuard - 频繁止损保护，12根K线内有2次止损且利润<-5.00%。'}]",
            None,
        ),
        (
            {"method": "LowProfitPairs", "lookback_period_candles": 11, "unlock_at": "03:00"},
            "[{'LowProfitPairs': 'LowProfitPairs - 低利润保护，锁定11根K线内利润<0.0的交易对。'}]",
            None,
        ),
        (
            {"method": "MaxDrawdown", "lookback_period_candles": 20, "unlock_at": "04:00"},
            "[{'MaxDrawdown': 'MaxDrawdown - 最大回撤保护，20根K线内回撤>0.0时停止交易。'}]",
            None,
        ),
    ],
)
def test_protection_manager_desc(
    mocker, default_conf, protectionconf, desc_expected, exception_expected
):
    default_conf["_strategy_protections"] = [protectionconf]
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    short_desc = str(freqtrade.protections.short_desc())
    assert short_desc == desc_expected