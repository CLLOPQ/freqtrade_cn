# pragma pylint: disable=missing-docstring, C0103
# pragma pylint: disable=protected-access, unused-argument, invalid-name
# pragma pylint: disable=too-many-lines, too-many-arguments

import asyncio
import logging
import re
import threading
from datetime import timedelta
from functools import reduce
from random import choice, randint
from string import ascii_uppercase
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest
import time_machine
from pandas import DataFrame
from sqlalchemy import select
from telegram import Chat, Message, ReplyKeyboardMarkup, Update, User
from telegram.error import BadRequest, NetworkError, TelegramError

from freqtrade import __version__
from freqtrade.constants import CANCEL_REASON
from freqtrade.enums import (
    ExitType,
    MarketDirection,
    RPCMessageType,
    RunMode,
    SignalDirection,
    State,
)
from freqtrade.exceptions import OperationalException
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.loggers import setup_logging
from freqtrade.persistence import PairLocks, Trade
from freqtrade.persistence.models import Order
from freqtrade.rpc import RPC
from freqtrade.rpc.rpc import RPCException
from freqtrade.rpc.telegram import Telegram, authorized_only
from freqtrade.util.datetime_helpers import dt_now
from tests.conftest import (
    CURRENT_TEST_STRATEGY,
    EXMS,
    create_mock_trades,
    create_mock_trades_usdt,
    get_patched_freqtradebot,
    log_has,
    log_has_re,
    patch_exchange,
    patch_get_signal,
    patch_whitelist,
)


@pytest.fixture(autouse=True)
def mock_exchange_loop(mocker):
    mocker.patch("freqtrade.exchange.exchange.Exchange._init_async_loop")


@pytest.fixture
def default_conf(default_conf) -> dict:
    # 默认启用 Telegram
    default_conf["telegram"]["enabled"] = True
    return default_conf


@pytest.fixture
def update():
    message = Message(
        0,
        dt_now(),
        Chat(1235, 0),
        from_user=User(5432, "test", is_bot=False),
    )
    _update = Update(0, message=message)

    return _update


def patch_eventloop_threading(telegrambot):
    is_init = False

    def thread_fuck():
        nonlocal is_init
        telegrambot._loop = asyncio.new_event_loop()
        is_init = True
        telegrambot._loop.run_forever()

    x = threading.Thread(target=thread_fuck, daemon=True)
    x.start()
    while not is_init:
        pass


class DummyCls(Telegram):
    """
    用于测试 Telegram @authorized_only 装饰器的虚拟类
    """

    def __init__(self, rpc: RPC, config) -> None:
        super().__init__(rpc, config)
        self.state = {"called": False}

    def _init(self):
        pass

    @authorized_only
    async def dummy_handler(self, *args, **kwargs) -> None:
        """
        只改变对象状态的虚假方法
        """
        self.state["called"] = True

    @authorized_only
    async def dummy_exception(self, *args, **kwargs) -> None:
        """
        抛出异常的虚假方法
        """
        raise Exception("test")


def get_telegram_testobject(mocker, default_conf, mock=True, ftbot=None):
    msg_mock = AsyncMock()
    if mock:
        mocker.patch.multiple(
            "freqtrade.rpc.telegram.Telegram",
            _init=MagicMock(),
            _send_msg=msg_mock,
            _start_thread=MagicMock(),
        )
    if not ftbot:
        ftbot = get_patched_freqtradebot(mocker, default_conf)
    rpc = RPC(ftbot)
    telegram = Telegram(rpc, default_conf)
    telegram._loop = MagicMock()
    patch_eventloop_threading(telegram)

    return telegram, ftbot, msg_mock


def test_telegram__init__(default_conf, mocker) -> None:
    mocker.patch("freqtrade.rpc.telegram.Telegram._init", MagicMock())

    telegram, _, _ = get_telegram_testobject(mocker, default_conf)
    assert telegram._config == default_conf


def test_telegram_init(default_conf, mocker, caplog) -> None:
    app_mock = MagicMock()
    mocker.patch("freqtrade.rpc.telegram.Telegram._start_thread", MagicMock())
    mocker.patch("freqtrade.rpc.telegram.Telegram._init_telegram_app", return_value=app_mock)
    mocker.patch("freqtrade.rpc.telegram.Telegram._startup_telegram", AsyncMock())

    telegram, _, _ = get_telegram_testobject(mocker, default_conf, mock=False)
    telegram._init()
    assert app_mock.call_count == 0

    # 注册的处理程序数量
    assert app_mock.add_handler.call_count > 0

    message_str = (
        "rpc.telegram 正在监听以下命令: [['status'], ['profit'], "
        "['balance'], ['start'], ['stop'], "
        "['forceexit', 'forcesell', 'fx'], ['forcebuy', 'forcelong'], ['forceshort'], "
        "['reload_trade'], ['trades'], ['delete'], ['cancel_open_order', 'coo'], "
        "['performance'], ['buys', 'entries'], ['exits', 'sells'], ['mix_tags'], "
        "['stats'], ['daily'], ['weekly'], ['monthly'], "
        "['count'], ['locks'], ['delete_locks', 'unlock'], "
        "['reload_conf', 'reload_config'], ['show_conf', 'show_config'], "
        "['pause', 'stopbuy', 'stopentry'], ['whitelist'], ['blacklist'], "
        "['bl_delete', 'blacklist_delete'], "
        "['logs'], ['health'], ['help'], ['version'], ['marketdir'], "
        "['order'], ['list_custom_data'], ['tg_info']]"
    )

    assert log_has(message_str, caplog)


async def test_telegram_startup(default_conf, mocker) -> None:
    app_mock = MagicMock()
    app_mock.initialize = AsyncMock()
    app_mock.start = AsyncMock()
    app_mock.updater.start_polling = AsyncMock()
    app_mock.updater.running = False
    sleep_mock = mocker.patch("freqtrade.rpc.telegram.asyncio.sleep", AsyncMock())

    telegram, _, _ = get_telegram_testobject(mocker, default_conf)
    telegram._app = app_mock
    await telegram._startup_telegram()
    assert app_mock.initialize.call_count == 1
    assert app_mock.start.call_count == 1
    assert app_mock.updater.start_polling.call_count == 1
    assert sleep_mock.call_count == 1


async def test_telegram_cleanup(
    default_conf,
    mocker,
) -> None:
    app_mock = MagicMock()
    app_mock.stop = AsyncMock()
    app_mock.initialize = AsyncMock()

    updater_mock = MagicMock()
    updater_mock.stop = AsyncMock()
    app_mock.updater = updater_mock

    telegram, _, _ = get_telegram_testobject(mocker, default_conf)
    telegram._app = app_mock
    telegram._loop = asyncio.get_running_loop()
    telegram._thread = MagicMock()
    telegram.cleanup()
    await asyncio.sleep(0.1)
    assert app_mock.stop.call_count == 1
    assert telegram._thread.join.call_count == 1


async def test_authorized_only(default_conf, mocker, caplog, update) -> None:
    patch_exchange(mocker)
    caplog.set_level(logging.DEBUG)
    default_conf["telegram"]["enabled"] = False
    bot = FreqtradeBot(default_conf)
    rpc = RPC(bot)
    dummy = DummyCls(rpc, default_conf)

    patch_get_signal(bot)
    await dummy.dummy_handler(update=update, context=MagicMock())
    assert dummy.state["called"] is True
    assert log_has("执行处理程序: dummy_handler 聊天ID: 1235", caplog)
    assert not log_has("拒绝未授权消息来自: 1235", caplog)
    assert not log_has("Telegram 模块内发生异常", caplog)


async def test_authorized_only_unauthorized(default_conf, mocker, caplog) -> None:
    patch_exchange(mocker)
    caplog.set_level(logging.DEBUG)
    message = Message(
        randint(1, 100),
        dt_now(),
        Chat(0xDEADBEEF, 0),
        from_user=User(5432, "test", is_bot=False),
    )
    update = Update(randint(1, 100), message=message)

    default_conf["telegram"]["enabled"] = False
    bot = FreqtradeBot(default_conf)
    rpc = RPC(bot)
    dummy = DummyCls(rpc, default_conf)

    patch_get_signal(bot)
    await dummy.dummy_handler(update=update, context=MagicMock())
    assert dummy.state["called"] is False
    assert not log_has("执行处理程序: dummy_handler 聊天ID: 3735928559", caplog)
    assert log_has("拒绝未授权消息来自: 3735928559", caplog)
    assert not log_has("Telegram 模块内发生异常", caplog)


async def test_authorized_users(default_conf, mocker, caplog, update) -> None:
    patch_exchange(mocker)
    caplog.set_level(logging.DEBUG)
    default_conf["telegram"]["enabled"] = False
    default_conf["telegram"]["authorized_users"] = ["5432"]
    bot = FreqtradeBot(default_conf)
    rpc = RPC(bot)
    dummy = DummyCls(rpc, default_conf)

    await dummy.dummy_handler(update=update, context=MagicMock())
    assert dummy.state["called"] is True
    assert log_has("执行处理程序: dummy_handler 聊天ID: 1235", caplog)
    caplog.clear()
    # 测试空白情况
    default_conf["telegram"]["authorized_users"] = []
    dummy1 = DummyCls(rpc, default_conf)
    await dummy1.dummy_handler(update=update, context=MagicMock())
    assert dummy1.state["called"] is False
    assert log_has_re(r"未授权用户尝试 .*5432", caplog)
    caplog.clear()
    # 测试错误用户
    default_conf["telegram"]["authorized_users"] = ["1234"]
    dummy1 = DummyCls(rpc, default_conf)
    await dummy1.dummy_handler(update=update, context=MagicMock())
    assert dummy1.state["called"] is False
    assert log_has_re(r"未授权用户尝试 .*5432", caplog)
    caplog.clear()

    # 再次测试反向情况
    default_conf["telegram"]["authorized_users"] = ["5432"]
    dummy1 = DummyCls(rpc, default_conf)
    await dummy1.dummy_handler(update=update, context=MagicMock())
    assert dummy1.state["called"] is True
    assert not log_has_re(r"未授权用户尝试 .*5432", caplog)


async def test_authorized_only_exception(default_conf, mocker, caplog, update) -> None:
    patch_exchange(mocker)

    default_conf["telegram"]["enabled"] = False

    bot = FreqtradeBot(default_conf)
    rpc = RPC(bot)
    dummy = DummyCls(rpc, default_conf)
    patch_get_signal(bot)

    await dummy.dummy_exception(update=update, context=MagicMock())
    assert dummy.state["called"] is False
    assert not log_has("执行处理程序: dummy_handler 聊天ID: 0", caplog)
    assert not log_has("拒绝未授权消息来自: 0", caplog)
    assert log_has("Telegram 模块内发生异常", caplog)


async def test_telegram_status(default_conf, update, mocker) -> None:
    default_conf["telegram"]["enabled"] = False

    status_table = MagicMock()
    mocker.patch("freqtrade.rpc.telegram.Telegram._status_table", status_table)

    mocker.patch.multiple(
        "freqtrade.rpc.rpc.RPC",
        _rpc_trade_status=MagicMock(
            return_value=[
                {
                    "trade_id": 1,
                    "pair": "ETH/BTC",
                    "base_currency": "ETH",
                    "quote_currency": "BTC",
                    "open_date": dt_now(),
                    "close_date": None,
                    "open_rate": 1.099e-05,
                    "close_rate": None,
                    "current_rate": 1.098e-05,
                    "amount": 90.99181074,
                    "stake_amount": 90.99181074,
                    "max_stake_amount": 90.99181074,
                    "buy_tag": None,
                    "enter_tag": None,
                    "close_profit_ratio": None,
                    "profit": -0.0059,
                    "profit_ratio": -0.0059,
                    "profit_abs": -0.225,
                    "realized_profit": 0.0,
                    "total_profit_abs": -0.225,
                    "initial_stop_loss_abs": 1.098e-05,
                    "stop_loss_abs": 1.099e-05,
                    "exit_order_status": None,
                    "initial_stop_loss_ratio": -0.0005,
                    "stoploss_current_dist": 1e-08,
                    "stoploss_current_dist_ratio": -0.0002,
                    "stop_loss_ratio": -0.0001,
                    "open_order": "(限价买入 剩余=0.00000000)",
                    "is_open": True,
                    "is_short": False,
                    "filled_entry_orders": [],
                    "orders": [],
                }
            ]
        ),
    )

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    await telegram._status(update=update, context=MagicMock())
    assert msg_mock.call_count == 1

    context = MagicMock()
    # /status table
    context.args = ["table"]
    await telegram._status(update=update, context=context)
    assert status_table.call_count == 1


@pytest.mark.usefixtures("init_persistence")
async def test_telegram_status_multi_entry(default_conf, update, mocker, fee) -> None:
    default_conf["telegram"]["enabled"] = False
    default_conf["position_adjustment_enable"] = True
    mocker.patch.multiple(
        EXMS,
        fetch_order=MagicMock(return_value=None),
        get_rate=MagicMock(return_value=0.22),
    )

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    create_mock_trades(fee)
    trades = Trade.get_open_trades()
    trade = trades[3]
    # 某些交易所的平均价可能为空
    trade.orders[0].average = 0
    trade.orders.append(
        Order(
            order_id="5412vbb",
            ft_order_side="buy",
            ft_pair=trade.pair,
            ft_is_open=False,
            ft_amount=trade.amount,
            ft_price=trade.open_rate,
            status="closed",
            symbol=trade.pair,
            order_type="market",
            side="buy",
            price=trade.open_rate * 0.95,
            average=0,
            filled=trade.amount,
            remaining=0,
            cost=trade.amount,
            order_date=trade.open_date,
            order_filled_date=trade.open_date,
        )
    )
    trade.recalc_trade_from_orders()
    Trade.commit()

    await telegram._status(update=update, context=MagicMock())
    assert msg_mock.call_count == 4
    msg = msg_mock.call_args_list[3][0][0]
    assert re.search(r"入场次数.*2", msg)
    assert re.search(r"出场次数.*1", msg)
    assert re.search(r"平仓日期:", msg) is None
    assert re.search(r"平仓利润:", msg) is None


@pytest.mark.usefixtures("init_persistence")
async def test_telegram_status_closed_trade(default_conf, update, mocker, fee) -> None:
    default_conf["position_adjustment_enable"] = True
    mocker.patch.multiple(
        EXMS,
        fetch_order=MagicMock(return_value=None),
        get_rate=MagicMock(return_value=0.22),
    )

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    create_mock_trades(fee)
    trade = Trade.get_trades([Trade.is_open.is_(False)]).first()
    context = MagicMock()
    context.args = [str(trade.id)]
    await telegram._status(update=update, context=context)
    assert msg_mock.call_count == 1
    msg = msg_mock.call_args_list[0][0][0]
    assert re.search(r"平仓日期:", msg)
    assert re.search(r"平仓利润:", msg)


async def test_order_handle(default_conf, update, ticker, fee, mocker) -> None:
    default_conf["max_open_trades"] = 3
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
        _dry_is_price_crossed=MagicMock(return_value=True),
    )
    status_table = MagicMock()
    mocker.patch.multiple(
        "freqtrade.rpc.telegram.Telegram",
        _status_table=status_table,
    )

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    patch_get_signal(freqtradebot)

    freqtradebot.state = State.RUNNING
    msg_mock.reset_mock()

    # 创建一些测试数据
    freqtradebot.enter_positions()

    mocker.patch("freqtrade.rpc.telegram.MAX_MESSAGE_LENGTH", 500)

    msg_mock.reset_mock()
    context = MagicMock()
    context.args = ["2"]
    await telegram._order(update=update, context=context)

    assert msg_mock.call_count == 1

    msg1 = msg_mock.call_args_list[0][0][0]

    assert "交易 #*`2` 的订单列表" in msg1

    msg_mock.reset_mock()
    mocker.patch("freqtrade.rpc.telegram.MAX_MESSAGE_LENGTH", 50)
    context = MagicMock()
    context.args = ["2"]
    await telegram._order(update=update, context=context)

    assert msg_mock.call_count == 2

    msg1 = msg_mock.call_args_list[0][0][0]
    msg2 = msg_mock.call_args_list[1][0][0]

    assert "交易 #*`2` 的订单列表" in msg1
    assert "*交易 #*`2` 的订单列表 - 续" in msg2


@pytest.mark.usefixtures("init_persistence")
async def test_telegram_order_multi_entry(default_conf, update, mocker, fee) -> None:
    default_conf["telegram"]["enabled"] = False
    default_conf["position_adjustment_enable"] = True
    mocker.patch.multiple(
        EXMS,
        fetch_order=MagicMock(return_value=None),
        get_rate=MagicMock(return_value=0.22),
    )

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    create_mock_trades(fee)
    trades = Trade.get_open_trades()
    trade = trades[3]
    # 某些交易所的平均价可能为空
    trade.orders[0].average = 0
    trade.orders.append(
        Order(
            order_id="5412vbb",
            ft_order_side="buy",
            ft_pair=trade.pair,
            ft_is_open=False,
            ft_amount=trade.amount,
            ft_price=trade.open_rate,
            status="closed",
            symbol=trade.pair,
            order_type="market",
            side="buy",
            price=trade.open_rate * 0.95,
            average=0,
            filled=trade.amount,
            remaining=0,
            cost=trade.amount,
            order_date=trade.open_date,
            order_filled_date=trade.open_date,
        )
    )
    trade.recalc_trade_from_orders()
    Trade.commit()

    await telegram._order(update=update, context=MagicMock())
    assert msg_mock.call_count == 4
    msg = msg_mock.call_args_list[3][0][0]
    assert re.search(r"从第一个入场价格", msg)
    assert re.search(r"订单已成交", msg)


async def test_status_handle(default_conf, update, ticker, fee, mocker) -> None:
    default_conf["max_open_trades"] = 3
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
        _dry_is_price_crossed=MagicMock(return_value=True),
    )
    status_table = MagicMock()
    mocker.patch.multiple(
        "freqtrade.rpc.telegram.Telegram",
        _status_table=status_table,
    )

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    patch_get_signal(freqtradebot)

    freqtradebot.state = State.STOPPED
    # 停止时状态也启用
    await telegram._status(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert "没有活跃交易" in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    freqtradebot.state = State.RUNNING
    await telegram._status(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert "没有活跃交易" in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    # 创建一些测试数据
    freqtradebot.enter_positions()
    # 当我们有开放交易的已完成订单时触发状态
    await telegram._status(update=update, context=MagicMock())

    # close_rate 不应包含在消息中，因为交易未关闭
    # 并且没有行应该为空
    lines = msg_mock.call_args_list[0][0][0].split("\n")
    assert "" not in lines[:-1]
    assert "平仓价格" not in "".join(lines)
    assert "平仓利润" not in "".join(lines)

    assert msg_mock.call_count == 3
    assert "ETH/BTC" in msg_mock.call_args_list[0][0][0]
    assert "LTC/BTC" in msg_mock.call_args_list[1][0][0]

    msg_mock.reset_mock()
    context = MagicMock()
    context.args = ["2", "3"]

    await telegram._status(update=update, context=context)

    lines = msg_mock.call_args_list[0][0][0].split("\n")
    assert "" not in lines[:-1]
    assert "平仓价格" not in "".join(lines)
    assert "平仓利润" not in "".join(lines)

    assert msg_mock.call_count == 2
    assert "LTC/BTC" in msg_mock.call_args_list[0][0][0]

    mocker.patch("freqtrade.rpc.telegram.MAX_MESSAGE_LENGTH", 500)

    msg_mock.reset_mock()
    context = MagicMock()
    context.args = ["2"]
    await telegram._status(update=update, context=context)

    assert msg_mock.call_count == 1

    msg1 = msg_mock.call_args_list[0][0][0]

    assert "平仓价格" not in msg1
    assert "交易ID:* `2`" in msg1


async def test_status_table_handle(default_conf, update, ticker, fee, mocker) -> None:
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )

    default_conf["stake_amount"] = 15.0

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    patch_get_signal(freqtradebot)

    freqtradebot.state = State.STOPPED
    # 停止时状态表也启用
    await telegram._status_table(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert "没有活跃交易" in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    freqtradebot.state = State.RUNNING
    await telegram._status_table(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert "没有活跃交易" in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    # 创建一些测试数据
    freqtradebot.enter_positions()

    await telegram._status_table(update=update, context=MagicMock())

    text = re.sub("</?pre>", "", msg_mock.call_args_list[-1][0][0])
    line = text.split("\n")
    fields = re.sub("[ ]+", " ", line[2].strip()).split(" ")

    assert int(fields[0]) == 1
    # assert 'L' in fields[1]
    assert "ETH/BTC" in fields[1]
    assert msg_mock.call_count == 1


async def test_daily_handle(default_conf_usdt, update, ticker, fee, mocker, time_machine) -> None:
    mocker.patch("freqtrade.rpc.rpc.CryptoToFiatConverter._find_price", return_value=1.1)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )

    telegram, _freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)

    # 将日期移动到一天内
    time_machine.move_to("2022-06-11 08:00:00+00:00")
    # 创建一些测试数据
    create_mock_trades_usdt(fee)

    # 尝试有效数据
    # /daily 2
    context = MagicMock()
    context.args = ["2"]
    await telegram._daily(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "过去 2 天的每日利润</b>:" in msg_mock.call_args_list[0][0][0]
    assert "日期 " in msg_mock.call_args_list[0][0][0]
    assert str(dt_now().date()) in msg_mock.call_args_list[0][0][0]
    assert "  6.83 USDT" in msg_mock.call_args_list[0][0][0]
    assert "  7.51 USD" in msg_mock.call_args_list[0][0][0]
    assert "(2)" in msg_mock.call_args_list[0][0][0]
    assert "(2)  6.83 USDT  7.51 USD  0.64%" in msg_mock.call_args_list[0][0][0]
    assert "(0)" in msg_mock.call_args_list[0][0][0]

    # 重置 msg_mock
    msg_mock.reset_mock()
    context.args = []
    await telegram._daily(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "过去 7 天的每日利润</b>:" in msg_mock.call_args_list[0][0][0]
    assert str(dt_now().date()) in msg_mock.call_args_list[0][0][0]
    assert str((dt_now() - timedelta(days=5)).date()) in msg_mock.call_args_list[0][0][0]
    assert "  6.83 USDT" in msg_mock.call_args_list[0][0][0]
    assert "  7.51 USD" in msg_mock.call_args_list[0][0][0]
    assert "(2)" in msg_mock.call_args_list[0][0][0]
    assert "(1)" in msg_mock.call_args_list[0][0][0]
    assert "(0)" in msg_mock.call_args_list[0][0][0]

    # 重置 msg_mock
    msg_mock.reset_mock()

    # /daily 1
    context = MagicMock()
    context.args = ["1"]
    await telegram._daily(update=update, context=context)
    assert "  6.83 USDT" in msg_mock.call_args_list[0][0][0]
    assert "  7.51 USD" in msg_mock.call_args_list[0][0][0]
    assert "(2)" in msg_mock.call_args_list[0][0][0]


async def test_daily_wrong_input(default_conf, update, ticker, mocker) -> None:
    mocker.patch.multiple(EXMS, fetch_ticker=ticker)

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)

    # 尝试无效数据
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    # /daily -2
    context = MagicMock()
    context.args = ["-2"]
    await telegram._daily(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "必须是大于 0 的整数" in msg_mock.call_args_list[0][0][0]

    # 尝试无效数据
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    # /daily today
    context = MagicMock()
    context.args = ["today"]
    await telegram._daily(update=update, context=context)
    assert "过去 7 天的每日利润</b>:" in msg_mock.call_args_list[0][0][0]


async def test_weekly_handle(default_conf_usdt, update, ticker, fee, mocker, time_machine) -> None:
    default_conf_usdt["max_open_trades"] = 1
    mocker.patch("freqtrade.rpc.rpc.CryptoToFiatConverter._find_price", return_value=1.1)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)
    # 移动到周六 - 所以所有交易都在那一周内
    time_machine.move_to("2022-06-11")
    create_mock_trades_usdt(fee)

    # 尝试有效数据
    # /weekly 2
    context = MagicMock()
    context.args = ["2"]
    await telegram._weekly(update=update, context=context)
    assert msg_mock.call_count == 1
    assert (
        "过去 2 周的每周利润（从星期一开始）</b>:"
        in msg_mock.call_args_list[0][0][0]
    )
    assert "星期一 " in msg_mock.call_args_list[0][0][0]
    today = dt_now().date()
    first_iso_day_of_current_week = today - timedelta(days=today.weekday())
    assert str(first_iso_day_of_current_week) in msg_mock.call_args_list[0][0][0]
    assert "  2.74 USDT" in msg_mock.call_args_list[0][0][0]
    assert "  3.01 USD" in msg_mock.call_args_list[0][0][0]
    assert "(3)" in msg_mock.call_args_list[0][0][0]
    assert "(0)" in msg_mock.call_args_list[0][0][0]

    # 重置 msg_mock
    msg_mock.reset_mock()
    context.args = []
    await telegram._weekly(update=update, context=context)
    assert msg_mock.call_count == 1
    assert (
        "过去 8 周的每周利润（从星期一开始）</b>:"
        in msg_mock.call_args_list[0][0][0]
    )
    assert "每周" in msg_mock.call_args_list[0][0][0]
    assert "  2.74 USDT" in msg_mock.call_args_list[0][0][0]
    assert "  3.01 USD" in msg_mock.call_args_list[0][0][0]
    assert "(3)" in msg_mock.call_args_list[0][0][0]
    assert "(0)" in msg_mock.call_args_list[0][0][0]

    # 尝试无效数据
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    # /weekly -3
    context = MagicMock()
    context.args = ["-3"]
    await telegram._weekly(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "必须是大于 0 的整数" in msg_mock.call_args_list[0][0][0]

    # 尝试无效数据
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    # /weekly this week
    context = MagicMock()
    context.args = ["this week"]
    await telegram._weekly(update=update, context=context)
    assert (
        "过去 8 周的每周利润（从星期一开始）</b>:"
        in msg_mock.call_args_list[0][0][0]
    )


async def test_monthly_handle(default_conf_usdt, update, ticker, fee, mocker, time_machine) -> None:
    default_conf_usdt["max_open_trades"] = 1
    mocker.patch("freqtrade.rpc.rpc.CryptoToFiatConverter._find_price", return_value=1.1)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)
    # 移动到月内的某一天，以便所有模拟交易都落在本周内。
    time_machine.move_to("2022-06-11")
    create_mock_trades_usdt(fee)

    # 尝试有效数据
    # /monthly 2
    context = MagicMock()
    context.args = ["2"]
    await telegram._monthly(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "过去 2 个月的每月利润</b>:" in msg_mock.call_args_list[0][0][0]
    assert "月份 " in msg_mock.call_args_list[0][0][0]
    today = dt_now().date()
    current_month = f"{today.year}-{today.month:02} "
    assert current_month in msg_mock.call_args_list[0][0][0]
    assert "  2.74 USDT" in msg_mock.call_args_list[0][0][0]
    assert "  3.01 USD" in msg_mock.call_args_list[0][0][0]
    assert "(3)" in msg_mock.call_args_list[0][0][0]
    assert "(0)" in msg_mock.call_args_list[0][0][0]

    # 重置 msg_mock
    msg_mock.reset_mock()
    context.args = []
    await telegram._monthly(update=update, context=context)
    assert msg_mock.call_count == 1
    # 默认为 6 个月
    assert "过去 6 个月的每月利润</b>:" in msg_mock.call_args_list[0][0][0]
    assert "月份 " in msg_mock.call_args_list[0][0][0]
    assert current_month in msg_mock.call_args_list[0][0][0]
    assert "  2.74 USDT" in msg_mock.call_args_list[0][0][0]
    assert "  3.01 USD" in msg_mock.call_args_list[0][0][0]
    assert "(3)" in msg_mock.call_args_list[0][0][0]
    assert "(0)" in msg_mock.call_args_list[0][0][0]

    # 重置 msg_mock
    msg_mock.reset_mock()

    # /monthly 12
    context = MagicMock()
    context.args = ["12"]
    await telegram._monthly(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "过去 12 个月的每月利润</b>:" in msg_mock.call_args_list[0][0][0]
    assert "  2.74 USDT" in msg_mock.call_args_list[0][0][0]
    assert "  3.01 USD" in msg_mock.call_args_list[0][0][0]
    assert "(3)" in msg_mock.call_args_list[0][0][0]

    # 一位数月份应该包含零，例如：2021 年 9 月 = "2021-09"
    # 由于我们加载了过去 12 个月，任何月份都应该出现
    assert "-09" in msg_mock.call_args_list[0][0][0]

    # 尝试无效数据
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    # /monthly -3
    context = MagicMock()
    context.args = ["-3"]
    await telegram._monthly(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "必须是大于 0 的整数" in msg_mock.call_args_list[0][0][0]

    # 尝试无效数据
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    # /monthly february
    context = MagicMock()
    context.args = ["february"]
    await telegram._monthly(update=update, context=context)
    assert "过去 6 个月的每月利润</b>:" in msg_mock.call_args_list[0][0][0]


async def test_telegram_profit_handle(
    default_conf_usdt, update, ticker_usdt, ticker_sell_up, fee, limit_sell_order_usdt, mocker
) -> None:
    mocker.patch("freqtrade.rpc.rpc.CryptoToFiatConverter._find_price", return_value=1.1)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        get_fee=fee,
    )

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)
    patch_get_signal(freqtradebot)

    await telegram._profit(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert "还没有交易。" in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    # 创建一些测试数据
    freqtradebot.enter_positions()
    trade = Trade.session.scalars(select(Trade)).first()

    context = MagicMock()
    # 使用无效的第二个参数进行测试（应该静默通过）
    context.args = ["aaa"]
    await telegram._profit(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "没有已平仓交易" in msg_mock.call_args_list[-1][0][0]
    assert "*投资回报率:* 所有交易" in msg_mock.call_args_list[-1][0][0]
    mocker.patch("freqtrade.wallets.Wallets.get_starting_balance", return_value=1000)
    assert (
        "∙ `0.298 USDT (0.50%) (0.03 \N{GREEK CAPITAL LETTER SIGMA}%)`"
        in msg_mock.call_args_list[-1][0][0]
    )
    msg_mock.reset_mock()

    # 用上涨的市场更新行情
    mocker.patch(f"{EXMS}.fetch_ticker", ticker_sell_up)
    # 模拟交易的已完成限价卖出订单
    trade = Trade.session.scalars(select(Trade)).first()
    oobj = Order.parse_from_ccxt_object(
        limit_sell_order_usdt, limit_sell_order_usdt["symbol"], "sell"
    )
    trade.orders.append(oobj)
    trade.update_trade(oobj)

    trade.close_date = dt_now()
    trade.is_open = False
    Trade.commit()

    context.args = [3]
    await telegram._profit(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "*投资回报率:* 已平仓交易" in msg_mock.call_args_list[-1][0][0]
    assert (
        "∙ `5.685 USDT (9.45%) (0.57 \N{GREEK CAPITAL LETTER SIGMA}%)`"
        in msg_mock.call_args_list[-1][0][0]
    )
    assert "∙ `6.253 USD`" in msg_mock.call_args_list[-1][0][0]
    assert "*投资回报率:* 所有交易" in msg_mock.call_args_list[-1][0][0]
    assert (
        "∙ `5.685 USDT (9.45%) (0.57 \N{GREEK CAPITAL LETTER SIGMA}%)`"
        in msg_mock.call_args_list[-1][0][0]
    )
    assert "∙ `6.253 USD`" in msg_mock.call_args_list[-1][0][0]

    assert "*最佳表现:* `ETH/USDT: 5.685 USDT (9.47%)`" in msg_mock.call_args_list[-1][0][0]
    assert "*最大回撤:*" in msg_mock.call_args_list[-1][0][0]
    assert "*利润因子:*" in msg_mock.call_args_list[-1][0][0]
    assert "*胜率:*" in msg_mock.call_args_list[-1][0][0]
    assert "*期望值（比率）:*" in msg_mock.call_args_list[-1][0][0]
    assert "*交易量:* `126 USDT`" in msg_mock.call_args_list[-1][0][0]


@pytest.mark.parametrize("is_short", [True, False])
async def test_telegram_stats(default_conf, update, ticker, fee, mocker, is_short) -> None:
    mocker.patch("freqtrade.rpc.rpc.CryptoToFiatConverter._find_price", return_value=15000.0)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)

    await telegram._stats(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert "还没有交易。" in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    # 创建一些测试数据
    create_mock_trades(fee, is_short=is_short)

    await telegram._stats(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert "退出原因" in msg_mock.call_args_list[-1][0][0]
    assert "投资回报率" in msg_mock.call_args_list[-1][0][0]
    assert "平均持续时间" in msg_mock.call_args_list[-1][0][0]
    # 持续时间不仅仅是 N/A
    assert "0:19:00" in msg_mock.call_args_list[-1][0][0]
    assert "N/A" in msg_mock.call_args_list[-1][0][0]
    msg_mock.reset_mock()


async def test_telegram_balance_handle(default_conf, update, mocker, rpc_balance, tickers) -> None:
    default_conf["dry_run"] = False
    mocker.patch(f"{EXMS}.get_balances", return_value=rpc_balance)
    mocker.patch(f"{EXMS}.get_tickers", tickers)
    mocker.patch(f"{EXMS}.get_valid_pair_combination", side_effect=lambda a, b: [f"{a}/{b}"])

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)

    await telegram._balance(update=update, context=MagicMock())
    context = MagicMock()
    context.args = ["full"]
    await telegram._balance(update=update, context=context)
    result = msg_mock.call_args_list[0][0][0]
    result_full = msg_mock.call_args_list[1][0][0]
    assert msg_mock.call_count == 2
    assert "*BTC:*" in result
    assert "*ETH:*" not in result
    assert "*USDT:*" not in result
    assert "*EUR:*" not in result
    assert "*LTC:*" not in result

    assert "*LTC:*" in result_full
    assert "*XRP:*" not in result
    assert "余额:" in result
    assert "估计 BTC:" in result
    assert "BTC: 11" in result
    assert "BTC: 12" in result_full
    assert "*3 其他货币 (< 0.0001 BTC):*" in result
    assert "BTC: 0.00000309" in result
    assert "*估计价值*:" in result_full
    assert "*估计价值（仅机器人管理的资产）*:" in result


async def test_telegram_balance_handle_futures(
    default_conf, update, rpc_balance, mocker, tickers
) -> None:
    default_conf.update(
        {
            "dry_run": False,
            "trading_mode": "futures",
            "margin_mode": "isolated",
        }
    )
    mock_pos = [
        {
            "symbol": "ETH/USDT:USDT",
            "timestamp": None,
            "datetime": None,
            "initialMargin": 0.0,
            "initialMarginPercentage": None,
            "maintenanceMargin": 0.0,
            "maintenanceMarginPercentage": 0.005,
            "entryPrice": 0.0,
            "notional": 10.0,
            "leverage": 5.0,
            "unrealizedPnl": 0.0,
            "contracts": 1.0,
            "contractSize": 1,
            "marginRatio": None,
            "liquidationPrice": 0.0,
            "markPrice": 2896.41,
            "collateral": 20,
            "marginType": "isolated",
            "side": "short",
            "percentage": None,
        },
        {
            "symbol": "XRP/USDT:USDT",
            "timestamp": None,
            "datetime": None,
            "initialMargin": 0.0,
            "initialMarginPercentage": None,
            "maintenanceMargin": 0.0,
            "maintenanceMarginPercentage": 0.005,
            "entryPrice": 0.0,
            "notional": 10.0,
            "leverage": None,
            "unrealizedPnl": 0.0,
            "contracts": 1.0,
            "contractSize": 1,
            "marginRatio": None,
            "liquidationPrice": 0.0,
            "markPrice": 2896.41,
            "collateral": 20,
            "marginType": "isolated",
            "side": "short",
            "percentage": None,
        },
    ]
    mocker.patch(f"{EXMS}.get_balances", return_value=rpc_balance)
    mocker.patch(f"{EXMS}.fetch_positions", return_value=mock_pos)
    mocker.patch(f"{EXMS}.get_tickers", tickers)
    mocker.patch(f"{EXMS}.get_valid_pair_combination", side_effect=lambda a, b: [f"{a}/{b}"])

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)

    await telegram._balance(update=update, context=MagicMock())
    result = msg_mock.call_args_list[0][0][0]
    assert msg_mock.call_count == 1

    assert "ETH/USDT:USDT" in result
    assert "`空头: 10" in result
    assert "XRP/USDT:USDT" in result


async def test_balance_handle_empty_response(default_conf, update, mocker) -> None:
    default_conf["dry_run"] = False
    mocker.patch(f"{EXMS}.get_balances", return_value={})

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)

    freqtradebot.config["dry_run"] = False
    await telegram._balance(update=update, context=MagicMock())
    result = msg_mock.call_args_list[0][0][0]
    assert msg_mock.call_count == 1
    assert "启动资金: `0 BTC" in result


async def test_balance_handle_empty_response_dry(default_conf, update, mocker) -> None:
    mocker.patch(f"{EXMS}.get_balances", return_value={})

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)

    await telegram._balance(update=update, context=MagicMock())
    result = msg_mock.call_args_list[0][0][0]
    assert msg_mock.call_count == 1
    assert "*警告:* 模拟模式下的模拟余额。" in result
    assert "启动资金: `990 BTC`" in result


async def test_balance_handle_too_large_response(default_conf, update, mocker) -> None:
    balances = []
    for i in range(100):
        curr = choice(ascii_uppercase) + choice(ascii_uppercase) + choice(ascii_uppercase)
        balances.append(
            {
                "currency": curr,
                "free": 1.0,
                "used": 0.5,
                "balance": i,
                "bot_owned": 0.5,
                "est_stake": 1,
                "est_stake_bot": 1,
                "stake": "BTC",
                "is_position": False,
                "leverage": 1.0,
                "position": 0.0,
                "side": "long",
                "is_bot_managed": True,
            }
        )
    mocker.patch(
        "freqtrade.rpc.rpc.RPC._rpc_balance",
        return_value={
            "currencies": balances,
            "total": 100.0,
            "total_bot": 100.0,
            "symbol": 100.0,
            "value": 1000.0,
            "value_bot": 1000.0,
            "starting_capital": 1000,
            "starting_capital_fiat": 1000,
        },
    )

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)

    await telegram._balance(update=update, context=MagicMock())
    assert msg_mock.call_count > 1
    # 测试是否在 4000 左右换行 -
    # 每个单独的货币输出大约 120 个字符长，所以我们需要
    # 一个偏移量以避免随机测试失败
    assert len(msg_mock.call_args_list[0][0][0]) < 4096
    assert len(msg_mock.call_args_list[0][0][0]) > (4096 - 120)


async def test_start_handle(default_conf, update, mocker) -> None:
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    freqtradebot.state = State.STOPPED
    assert freqtradebot.state == State.STOPPED
    await telegram._start(update=update, context=MagicMock())
    assert freqtradebot.state == State.RUNNING
    assert msg_mock.call_count == 1


async def test_start_handle_already_running(default_conf, update, mocker) -> None:
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    freqtradebot.state = State.RUNNING
    assert freqtradebot.state == State.RUNNING
    await telegram._start(update=update, context=MagicMock())
    assert freqtradebot.state == State.RUNNING
    assert msg_mock.call_count == 1
    assert "已经在运行" in msg_mock.call_args_list[0][0][0]


async def test_stop_handle(default_conf, update, mocker) -> None:
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    freqtradebot.state = State.RUNNING
    assert freqtradebot.state == State.RUNNING
    await telegram._stop(update=update, context=MagicMock())
    assert freqtradebot.state == State.STOPPED
    assert msg_mock.call_count == 1
    assert "正在停止交易员" in msg_mock.call_args_list[0][0][0]


async def test_stop_handle_already_stopped(default_conf, update, mocker) -> None:
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    freqtradebot.state = State.STOPPED
    assert freqtradebot.state == State.STOPPED
    await telegram._stop(update=update, context=MagicMock())
    assert freqtradebot.state == State.STOPPED
    assert msg_mock.call_count == 1
    assert "已经停止" in msg_mock.call_args_list[0][0][0]


async def test_pause_handle(default_conf, update, mocker) -> None:
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    assert freqtradebot.state == State.RUNNING
    await telegram._pause(update=update, context=MagicMock())
    assert freqtradebot.state == State.PAUSED
    assert msg_mock.call_count == 1
    assert (
        "已暂停，从现在开始不再有新的入场。运行 /start 以启用入场。"
        in msg_mock.call_args_list[0][0][0]
    )


async def test_reload_config_handle(default_conf, update, mocker) -> None:
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    freqtradebot.state = State.RUNNING
    assert freqtradebot.state == State.RUNNING
    await telegram._reload_config(update=update, context=MagicMock())
    assert freqtradebot.state == State.RELOAD_CONFIG
    assert msg_mock.call_count == 1
    assert "正在重新加载配置" in msg_mock.call_args_list[0][0][0]


async def test_telegram_forceexit_handle(
    default_conf, update, ticker, fee, ticker_sell_up, mocker
) -> None:
    mocker.patch("freqtrade.rpc.rpc.CryptoToFiatConverter._find_price", return_value=15000.0)
    msg_mock = mocker.patch("freqtrade.rpc.telegram.Telegram.send_msg", MagicMock())
    mocker.patch("freqtrade.rpc.telegram.Telegram._init", MagicMock())
    patch_exchange(mocker)
    patch_whitelist(mocker, default_conf)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
        _dry_is_price_crossed=MagicMock(return_value=True),
    )

    freqtradebot = FreqtradeBot(default_conf)
    rpc = RPC(freqtradebot)
    telegram = Telegram(rpc, default_conf)
    patch_get_signal(freqtradebot)

    # 创建一些测试数据
    freqtradebot.enter_positions()

    trade = Trade.session.scalars(select(Trade)).first()
    assert trade

    # 提高价格并卖出
    mocker.patch(f"{EXMS}.fetch_ticker", ticker_sell_up)

    # /forceexit 1
    context = MagicMock()
    context.args = ["1"]
    await telegram._force_exit(update=update, context=context)

    assert msg_mock.call_count == 4
    last_msg = msg_mock.call_args_list[-2][0][0]
    assert {
        "type": RPCMessageType.EXIT,
        "trade_id": 1,
        "exchange": "Binance",
        "pair": "ETH/BTC",
        "gain": "profit",
        "leverage": 1.0,
        "limit": 1.173e-05,
        "order_rate": 1.173e-05,
        "amount": 91.07468123,
        "order_type": "limit",
        "open_rate": 1.098e-05,
        "current_rate": 1.173e-05,
        "direction": "多头",
        "profit_amount": 6.314e-05,
        "profit_ratio": 0.0629778,
        "stake_currency": "BTC",
        "quote_currency": "BTC",
        "base_currency": "ETH",
        "fiat_currency": "USD",
        "buy_tag": ANY,
        "enter_tag": ANY,
        "exit_reason": ExitType.FORCE_EXIT.value,
        "open_date": ANY,
        "close_date": ANY,
        "close_rate": ANY,
        "stake_amount": 0.0009999999999054,
        "sub_trade": False,
        "cumulative_profit": 0.0,
        "is_final_exit": False,
        "final_profit_ratio": None,
    } == last_msg


async def test_telegram_force_exit_down_handle(
    default_conf, update, ticker, fee, ticker_sell_down, mocker
) -> None:
    mocker.patch(
        "freqtrade.rpc.fiat_convert.CryptoToFiatConverter._find_price", return_value=15000.0
    )
    msg_mock = mocker.patch("freqtrade.rpc.telegram.Telegram.send_msg", MagicMock())
    mocker.patch("freqtrade.rpc.telegram.Telegram._init", MagicMock())
    patch_exchange(mocker)
    patch_whitelist(mocker, default_conf)

    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
        _dry_is_price_crossed=MagicMock(return_value=True),
    )

    freqtradebot = FreqtradeBot(default_conf)
    rpc = RPC(freqtradebot)
    telegram = Telegram(rpc, default_conf)
    patch_get_signal(freqtradebot)

    # 创建一些测试数据
    freqtradebot.enter_positions()

    # 降低价格并卖出
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_sell_down)

    trade = Trade.session.scalars(select(Trade)).first()
    assert trade

    # /forceexit 1
    context = MagicMock()
    context.args = ["1"]
    await telegram._force_exit(update=update, context=context)

    assert msg_mock.call_count == 4

    last_msg = msg_mock.call_args_list[-2][0][0]
    assert {
        "type": RPCMessageType.EXIT,
        "trade_id": 1,
        "exchange": "Binance",
        "pair": "ETH/BTC",
        "gain": "loss",
        "leverage": 1.0,
        "limit": 1.043e-05,
        "order_rate": 1.043e-05,
        "amount": 91.07468123,
        "order_type": "limit",
        "open_rate": 1.098e-05,
        "current_rate": 1.043e-05,
        "direction": "多头",
        "profit_amount": -5.497e-05,
        "profit_ratio": -0.05482878,
        "stake_currency": "BTC",
        "quote_currency": "BTC",
        "base_currency": "ETH",
        "fiat_currency": "USD",
        "buy_tag": ANY,
        "enter_tag": ANY,
        "exit_reason": ExitType.FORCE_EXIT.value,
        "open_date": ANY,
        "close_date": ANY,
        "close_rate": ANY,
        "stake_amount": 0.0009999999999054,
        "sub_trade": False,
        "cumulative_profit": 0.0,
        "is_final_exit": False,
        "final_profit_ratio": None,
    } == last_msg


async def test_forceexit_all_handle(default_conf, update, ticker, fee, mocker) -> None:
    patch_exchange(mocker)
    mocker.patch(
        "freqtrade.rpc.fiat_convert.CryptoToFiatConverter._find_price", return_value=15000.0
    )
    msg_mock = mocker.patch("freqtrade.rpc.telegram.Telegram.send_msg", MagicMock())
    mocker.patch("freqtrade.rpc.telegram.Telegram._init", MagicMock())
    patch_whitelist(mocker, default_conf)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
        _dry_is_price_crossed=MagicMock(return_value=True),
    )
    default_conf["max_open_trades"] = 4
    freqtradebot = FreqtradeBot(default_conf)
    rpc = RPC(freqtradebot)
    telegram = Telegram(rpc, default_conf)
    patch_get_signal(freqtradebot)

    # 创建一些测试数据
    freqtradebot.enter_positions()
    msg_mock.reset_mock()

    # /forceexit all
    context = MagicMock()
    context.args = ["all"]
    await telegram._force_exit(update=update, context=context)

    # 每个交易调用 2 次
    assert msg_mock.call_count == 8
    msg = msg_mock.call_args_list[0][0][0]
    assert {
        "type": RPCMessageType.EXIT,
        "trade_id": 1,
        "exchange": "Binance",
        "pair": "ETH/BTC",
        "gain": "loss",
        "leverage": 1.0,
        "order_rate": 1.099e-05,
        "limit": 1.099e-05,
        "amount": 91.07468123,
        "order_type": "limit",
        "open_rate": 1.098e-05,
        "current_rate": 1.099e-05,
        "direction": "多头",
        "profit_amount": -4.09e-06,
        "profit_ratio": -0.00408133,
        "stake_currency": "BTC",
        "quote_currency": "BTC",
        "base_currency": "ETH",
        "fiat_currency": "USD",
        "buy_tag": ANY,
        "enter_tag": ANY,
        "exit_reason": ExitType.FORCE_EXIT.value,
        "open_date": ANY,
        "close_date": ANY,
        "close_rate": ANY,
        "stake_amount": 0.0009999999999054,
        "sub_trade": False,
        "cumulative_profit": 0.0,
        "is_final_exit": False,
        "final_profit_ratio": None,
    } == msg


async def test_forceexit_handle_invalid(default_conf, update, mocker) -> None:
    mocker.patch(
        "freqtrade.rpc.fiat_convert.CryptoToFiatConverter._find_price", return_value=15000.0
    )

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)

    # 交易员未运行
    freqtradebot.state = State.STOPPED
    # /forceexit 1
    context = MagicMock()
    context.args = ["1"]
    await telegram._force_exit(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "未运行" in msg_mock.call_args_list[0][0][0]

    # 无效参数
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING
    # /forceexit 123456
    context = MagicMock()
    context.args = ["123456"]
    await telegram._force_exit(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "无效参数" in msg_mock.call_args_list[0][0][0]


async def test_force_exit_no_pair(default_conf, update, ticker, fee, mocker) -> None:
    default_conf["max_open_trades"] = 4
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
        _dry_is_price_crossed=MagicMock(return_value=True),
    )
    femock = mocker.patch("freqtrade.rpc.rpc.RPC._rpc_force_exit")
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    patch_get_signal(freqtradebot)

    # /forceexit
    context = MagicMock()
    context.args = []
    await telegram._force_exit(update=update, context=context)
    # 没有交易对
    assert msg_mock.call_args_list[0][1]["msg"] == "没有找到开放交易。"

    # 创建一些测试数据
    freqtradebot.enter_positions()
    msg_mock.reset_mock()

    # /forceexit
    await telegram._force_exit(update=update, context=context)
    keyboard = msg_mock.call_args_list[0][1]["keyboard"]
    # 4 个交易对 + 取消
    assert reduce(lambda acc, x: acc + len(x), keyboard, 0) == 5
    assert keyboard[-1][0].text == "取消"

    assert keyboard[1][0].callback_data == "force_exit__2 "
    update = MagicMock()
    update.callback_query = AsyncMock()
    update.callback_query.data = keyboard[1][0].callback_data
    await telegram._force_exit_inline(update, None)
    assert update.callback_query.answer.call_count == 1
    assert update.callback_query.edit_message_text.call_count == 1
    assert femock.call_count == 1
    assert femock.call_args_list[0][0][0] == "2"

    # 重试退出 - 但改为取消
    update.callback_query.reset_mock()
    await telegram._force_exit(update=update, context=context)
    # 使用取消按钮
    update.callback_query.data = keyboard[-1][0].callback_data
    await telegram._force_exit_inline(update, None)
    query = update.callback_query
    assert query.answer.call_count == 1
    assert query.edit_message_text.call_count == 1
    assert query.edit_message_text.call_args_list[-1][1]["text"] == "强制退出已取消。"


async def test_force_enter_handle(default_conf, update, mocker) -> None:
    mocker.patch("freqtrade.rpc.rpc.CryptoToFiatConverter._find_price", return_value=15000.0)

    fbuy_mock = MagicMock(return_value=None)
    mocker.patch("freqtrade.rpc.rpc.RPC._rpc_force_entry", fbuy_mock)

    telegram, freqtradebot, _ = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)

    # /forcelong ETH/BTC
    context = MagicMock()
    context.args = ["ETH/BTC"]
    await telegram._force_enter(update=update, context=context, order_side=SignalDirection.LONG)

    assert fbuy_mock.call_count == 1
    assert fbuy_mock.call_args_list[0][0][0] == "ETH/BTC"
    assert fbuy_mock.call_args_list[0][0][1] is None
    assert fbuy_mock.call_args_list[0][1]["order_side"] == SignalDirection.LONG

    # 重置并重试指定价格
    fbuy_mock = MagicMock(return_value=None)
    mocker.patch("freqtrade.rpc.rpc.RPC._rpc_force_entry", fbuy_mock)
    # /forcelong ETH/BTC 0.055
    context = MagicMock()
    context.args = ["ETH/BTC", "0.055"]
    await telegram._force_enter(update=update, context=context, order_side=SignalDirection.LONG)

    assert fbuy_mock.call_count == 1
    assert fbuy_mock.call_args_list[0][0][0] == "ETH/BTC"
    assert isinstance(fbuy_mock.call_args_list[0][0][1], float)
    assert fbuy_mock.call_args_list[0][0][1] == 0.055


async def test_force_enter_handle_exception(default_conf, update, mocker) -> None:
    mocker.patch("freqtrade.rpc.rpc.CryptoToFiatConverter._find_price", return_value=15000.0)

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)

    await telegram._force_enter(update=update, context=MagicMock(), order_side=SignalDirection.LONG)

    assert msg_mock.call_count == 1
    assert msg_mock.call_args_list[0][0][0] == "强制入场未启用。"


async def test_force_enter_no_pair(default_conf, update, mocker) -> None:
    mocker.patch("freqtrade.rpc.rpc.CryptoToFiatConverter._find_price", return_value=15000.0)

    fbuy_mock = MagicMock(return_value=None)
    mocker.patch("freqtrade.rpc.rpc.RPC._rpc_force_entry", fbuy_mock)

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    patch_get_signal(freqtradebot)

    context = MagicMock()
    context.args = []
    await telegram._force_enter(update=update, context=context, order_side=SignalDirection.LONG)

    assert fbuy_mock.call_count == 0
    assert msg_mock.call_count == 1
    assert msg_mock.call_args_list[0][1]["msg"] == "哪个交易对？"
    keyboard = msg_mock.call_args_list[0][1]["keyboard"]
    # 一个额外的按钮 - 取消
    assert reduce(lambda acc, x: acc + len(x), keyboard, 0) == 5
    update = MagicMock()
    update.callback_query = AsyncMock()
    update.callback_query.data = "force_enter__XRP/USDT_||_long"
    await telegram._force_enter_inline(update, None)
    assert fbuy_mock.call_count == 1

    fbuy_mock.reset_mock()
    update.callback_query = AsyncMock()
    update.callback_query.data = "force_enter__cancel"
    await telegram._force_enter_inline(update, None)
    assert fbuy_mock.call_count == 0
    query = update.callback_query
    assert query.edit_message_text.call_count == 1
    assert query.edit_message_text.call_args_list[-1][1]["text"] == "强制入场已取消。"


async def test_telegram_performance_handle(default_conf_usdt, update, ticker, fee, mocker) -> None:
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )
    telegram, _freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)

    # 创建一些测试数据
    create_mock_trades_usdt(fee)

    await telegram._performance(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert "表现" in msg_mock.call_args_list[0][0][0]
    assert "<code>XRP/USDT\t2.842 USDT (9.47%) (1)</code>" in msg_mock.call_args_list[0][0][0]


async def test_telegram_entry_tag_performance_handle(
    default_conf_usdt, update, ticker, fee, mocker
) -> None:
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)
    patch_get_signal(freqtradebot)

    create_mock_trades_usdt(fee)

    context = MagicMock()
    await telegram._enter_tag_performance(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "入场标签表现" in msg_mock.call_args_list[0][0][0]
    assert "`TEST1\t3.987 USDT (1.99%) (1)`" in msg_mock.call_args_list[0][0][0]

    context.args = ["XRP/USDT"]
    await telegram._enter_tag_performance(update=update, context=context)
    assert msg_mock.call_count == 2

    msg_mock.reset_mock()
    mocker.patch(
        "freqtrade.rpc.rpc.RPC._rpc_enter_tag_performance", side_effect=RPCException("错误")
    )
    await telegram._enter_tag_performance(update=update, context=MagicMock())

    assert msg_mock.call_count == 1
    assert "错误" in msg_mock.call_args_list[0][0][0]


async def test_telegram_exit_reason_performance_handle(
    default_conf_usdt, update, ticker, fee, mocker
) -> None:
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)
    patch_get_signal(freqtradebot)

    create_mock_trades_usdt(fee)

    context = MagicMock()
    await telegram._exit_reason_performance(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "退出原因表现" in msg_mock.call_args_list[0][0][0]
    assert "`roi\t2.842 USDT (9.47%) (1)`" in msg_mock.call_args_list[0][0][0]
    context.args = ["XRP/USDT"]

    await telegram._exit_reason_performance(update=update, context=context)
    assert msg_mock.call_count == 2

    msg_mock.reset_mock()
    mocker.patch(
        "freqtrade.rpc.rpc.RPC._rpc_exit_reason_performance", side_effect=RPCException("错误")
    )
    await telegram._exit_reason_performance(update=update, context=MagicMock())

    assert msg_mock.call_count == 1
    assert "错误" in msg_mock.call_args_list[0][0][0]


async def test_telegram_mix_tag_performance_handle(
    default_conf_usdt, update, ticker, fee, mocker
) -> None:
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)
    patch_get_signal(freqtradebot)

    # 创建一些测试数据
    create_mock_trades_usdt(fee)

    context = MagicMock()
    await telegram._mix_tag_performance(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "混合标签表现" in msg_mock.call_args_list[0][0][0]
    assert "`TEST3 roi\t2.842 USDT (10.00%) (1)`" in msg_mock.call_args_list[0][0][0]

    context.args = ["XRP/USDT"]
    await telegram._mix_tag_performance(update=update, context=context)
    assert msg_mock.call_count == 2

    msg_mock.reset_mock()
    mocker.patch(
        "freqtrade.rpc.rpc.RPC._rpc_mix_tag_performance", side_effect=RPCException("错误")
    )
    await telegram._mix_tag_performance(update=update, context=MagicMock())

    assert msg_mock.call_count == 1
    assert "错误" in msg_mock.call_args_list[0][0][0]


async def test_count_handle(default_conf, update, ticker, fee, mocker) -> None:
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)

    freqtradebot.state = State.STOPPED
    await telegram._count(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert "未运行" in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()
    freqtradebot.state = State.RUNNING

    # 创建一些测试数据
    freqtradebot.enter_positions()
    msg_mock.reset_mock()
    await telegram._count(update=update, context=MagicMock())

    msg = (
        "<pre>  当前    最大    总投资\n---------  -----  -------------\n"
        "        1      {}          {}</pre>"
    ).format(default_conf["max_open_trades"], default_conf["stake_amount"])
    assert msg in msg_mock.call_args_list[0][0][0]


async def test_telegram_lock_handle(default_conf, update, ticker, fee, mocker) -> None:
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)
    patch_get_signal(freqtradebot)
    await telegram._locks(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert "没有活动锁。" in msg_mock.call_args_list[0][0][0]

    msg_mock.reset_mock()

    PairLocks.lock_pair("ETH/BTC", dt_now() + timedelta(minutes=4), "随机原因")
    PairLocks.lock_pair("XRP/BTC", dt_now() + timedelta(minutes=20), "deadbeef")

    await telegram._locks(update=update, context=MagicMock())

    assert "交易对" in msg_mock.call_args_list[0][0][0]
    assert "直到" in msg_mock.call_args_list[0][0][0]
    assert "原因\n" in msg_mock.call_args_list[0][0][0]
    assert "ETH/BTC" in msg_mock.call_args_list[0][0][0]
    assert "XRP/BTC" in msg_mock.call_args_list[0][0][0]
    assert "deadbeef" in msg_mock.call_args_list[0][0][0]
    assert "随机原因" in msg_mock.call_args_list[0][0][0]

    context = MagicMock()
    context.args = ["XRP/BTC"]
    msg_mock.reset_mock()
    await telegram._delete_locks(update=update, context=context)

    assert "ETH/BTC" in msg_mock.call_args_list[0][0][0]
    assert "随机原因" in msg_mock.call_args_list[0][0][0]
    assert "XRP/BTC" not in msg_mock.call_args_list[0][0][0]
    assert "deadbeef" not in msg_mock.call_args_list[0][0][0]


async def test_whitelist_static(default_conf, update, mocker) -> None:
    telegram, _freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    await telegram._whitelist(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert (
        "使用白名单 `['StaticPairList']` 包含 4 个交易对\n"
        "`ETH/BTC, LTC/BTC, XRP/BTC, NEO/BTC`" in msg_mock.call_args_list[0][0][0]
    )

    context = MagicMock()
    context.args = ["sorted"]
    msg_mock.reset_mock()
    await telegram._whitelist(update=update, context=context)
    assert (
        "使用白名单 `['StaticPairList']` 包含 4 个交易对\n"
        "`ETH/BTC, LTC/BTC, NEO/BTC, XRP/BTC`" in msg_mock.call_args_list[0][0][0]
    )

    context = MagicMock()
    context.args = ["baseonly"]
    msg_mock.reset_mock()
    await telegram._whitelist(update=update, context=context)
    assert (
        "使用白名单 `['StaticPairList']` 包含 4 个交易对\n"
        "`ETH, LTC, XRP, NEO`" in msg_mock.call_args_list[0][0][0]
    )

    context = MagicMock()
    context.args = ["baseonly", "sorted"]
    msg_mock.reset_mock()
    await telegram._whitelist(update=update, context=context)
    assert (
        "使用白名单 `['StaticPairList']` 包含 4 个交易对\n"
        "`ETH, LTC, NEO, XRP`" in msg_mock.call_args_list[0][0][0]
    )


async def test_whitelist_dynamic(default_conf, update, mocker) -> None:
    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    default_conf["pairlists"] = [{"method": "VolumePairList", "number_assets": 4}]
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    await telegram._whitelist(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert (
        "使用白名单 `['VolumePairList']` 包含 4 个交易对\n"
        "`ETH/BTC, LTC/BTC, XRP/BTC, NEO/BTC`" in msg_mock.call_args_list[0][0][0]
    )

    context = MagicMock()
    context.args = ["sorted"]
    msg_mock.reset_mock()
    await telegram._whitelist(update=update, context=context)
    assert (
        "使用白名单 `['VolumePairList']` 包含 4 个交易对\n"
        "`ETH/BTC, LTC/BTC, NEO/BTC, XRP/BTC`" in msg_mock.call_args_list[0][0][0]
    )

    context = MagicMock()
    context.args = ["baseonly"]
    msg_mock.reset_mock()
    await telegram._whitelist(update=update, context=context)
    assert (
        "使用白名单 `['VolumePairList']` 包含 4 个交易对\n"
        "`ETH, LTC, XRP, NEO`" in msg_mock.call_args_list[0][0][0]
    )

    context = MagicMock()
    context.args = ["baseonly", "sorted"]
    msg_mock.reset_mock()
    await telegram._whitelist(update=update, context=context)
    assert (
        "使用白名单 `['VolumePairList']` 包含 4 个交易对\n"
        "`ETH, LTC, NEO, XRP`" in msg_mock.call_args_list[0][0][0]
    )


async def test_blacklist_static(default_conf, update, mocker) -> None:
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    await telegram._blacklist(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert "黑名单包含 2 个交易对\n`DOGE/BTC, HOT/BTC`" in msg_mock.call_args_list[0][0][0]

    msg_mock.reset_mock()

    # /blacklist ETH/BTC
    context = MagicMock()
    context.args = ["ETH/BTC"]
    await telegram._blacklist(update=update, context=context)
    assert msg_mock.call_count == 1
    assert (
        "黑名单包含 3 个交易对\n`DOGE/BTC, HOT/BTC, ETH/BTC`"
        in msg_mock.call_args_list[0][0][0]
    )
    assert freqtradebot.pairlists.blacklist == ["DOGE/BTC", "HOT/BTC", "ETH/BTC"]

    msg_mock.reset_mock()
    context = MagicMock()
    context.args = ["XRP/.*"]
    await telegram._blacklist(update=update, context=context)
    assert msg_mock.call_count == 1

    assert (
        "黑名单包含 4 个交易对\n`DOGE/BTC, HOT/BTC, ETH/BTC, XRP/.*`"
        in msg_mock.call_args_list[0][0][0]
    )
    assert freqtradebot.pairlists.blacklist == ["DOGE/BTC", "HOT/BTC", "ETH/BTC", "XRP/.*"]

    msg_mock.reset_mock()
    context.args = ["DOGE/BTC"]
    await telegram._blacklist_delete(update=update, context=context)
    assert msg_mock.call_count == 1
    assert (
        "黑名单包含 3 个交易对\n`HOT/BTC, ETH/BTC, XRP/.*`" in msg_mock.call_args_list[0][0][0]
    )


async def test_telegram_logs(default_conf, update, mocker) -> None:
    mocker.patch.multiple(
        "freqtrade.rpc.telegram.Telegram",
        _init=MagicMock(),
    )
    setup_logging(default_conf)

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    context = MagicMock()
    context.args = []
    await telegram._logs(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "freqtrade\\.rpc\\.telegram" in msg_mock.call_args_list[0][0][0]

    msg_mock.reset_mock()
    context.args = ["1"]
    await telegram._logs(update=update, context=context)
    assert msg_mock.call_count == 1

    msg_mock.reset_mock()
    # 测试更改后的 MaxMessageLength
    mocker.patch("freqtrade.rpc.telegram.MAX_MESSAGE_LENGTH", 200)
    context = MagicMock()
    context.args = []
    await telegram._logs(update=update, context=context)
    # 至少调用 2 次。确切的次数会随着与设置消息无关的更改而变化
    # 因此我们不明确测试这个
    assert msg_mock.call_count >= 2


@pytest.mark.parametrize(
    "is_short,regex_pattern",
    [(True, r"现在[ ]*XRP\/BTC \(#3\)  -1.00% \("), (False, r"现在[ ]*XRP\/BTC \(#3\)  1.00% \(")],
)
async def test_telegram_trades(mocker, update, default_conf, fee, is_short, regex_pattern):
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    context = MagicMock()
    context.args = []

    await telegram._trades(update=update, context=context)
    assert "<b>0 个最近交易</b>:" in msg_mock.call_args_list[0][0][0]
    assert "<pre>" not in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    context.args = ["hello"]
    await telegram._trades(update=update, context=context)
    assert "<b>0 个最近交易</b>:" in msg_mock.call_args_list[0][0][0]
    assert "<pre>" not in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    create_mock_trades(fee, is_short=is_short)

    context = MagicMock()
    context.args = [5]
    await telegram._trades(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "2 个最近交易</b>:" in msg_mock.call_args_list[0][0][0]
    assert "利润 (" in msg_mock.call_args_list[0][0][0]
    assert "平仓日期" in msg_mock.call_args_list[0][0][0]
    assert "<pre>" in msg_mock.call_args_list[0][0][0]
    assert bool(re.search(regex_pattern, msg_mock.call_args_list[0][0][0]))


@pytest.mark.parametrize("is_short", [True, False])
async def test_telegram_delete_trade(mocker, update, default_conf, fee, is_short):
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    context = MagicMock()
    context.args = []

    await telegram._delete_trade(update=update, context=context)
    assert "交易ID未设置。" in msg_mock.call_args_list[0][0][0]

    msg_mock.reset_mock()
    create_mock_trades(fee, is_short=is_short)

    context = MagicMock()
    context.args = [1]
    await telegram._delete_trade(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "已删除交易 1。" in msg_mock.call_args_list[0][0][0]
    assert "请确保妥善处理此资产" in msg_mock.call_args_list[0][0][0]


@pytest.mark.parametrize("is_short", [True, False])
async def test_telegram_reload_trade_from_exchange(mocker, update, default_conf, fee, is_short):
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    context = MagicMock()
    context.args = []

    await telegram._reload_trade_from_exchange(update=update, context=context)
    assert "交易ID未设置。" in msg_mock.call_args_list[0][0][0]

    msg_mock.reset_mock()
    create_mock_trades(fee, is_short=is_short)

    context.args = [5]

    await telegram._reload_trade_from_exchange(update=update, context=context)
    assert "状态: `从交易所订单重新加载`" in msg_mock.call_args_list[0][0][0]


@pytest.mark.parametrize("is_short", [True, False])
async def test_telegram_delete_open_order(mocker, update, default_conf, fee, is_short, ticker):
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
    )
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    context = MagicMock()
    context.args = []

    await telegram._cancel_open_order(update=update, context=context)
    assert "交易ID未设置。" in msg_mock.call_args_list[0][0][0]

    msg_mock.reset_mock()
    create_mock_trades(fee, is_short=is_short)

    context = MagicMock()
    context.args = [5]
    await telegram._cancel_open_order(update=update, context=context)
    assert "交易ID没有未完成订单" in msg_mock.call_args_list[0][0][0]

    msg_mock.reset_mock()

    trade = Trade.get_trades([Trade.id == 6]).first()
    mocker.patch(f"{EXMS}.fetch_order", return_value=trade.orders[-1].to_ccxt_object())
    context = MagicMock()
    context.args = [6]
    await telegram._cancel_open_order(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "未完成订单已取消。" in msg_mock.call_args_list[0][0][0]


async def test_help_handle(default_conf, update, mocker) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    await telegram._help(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert "*/help:* `此帮助消息`" in msg_mock.call_args_list[0][0][0]


async def test_version_handle(default_conf, update, mocker) -> None:
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    await telegram._version(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert f"*版本:* `{__version__}`" in msg_mock.call_args_list[0][0][0]

    msg_mock.reset_mock()
    freqtradebot.strategy.version = lambda: "1.1.1"

    await telegram._version(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert f"*版本:* `{__version__}`" in msg_mock.call_args_list[0][0][0]
    assert "*策略版本: * `1.1.1`" in msg_mock.call_args_list[0][0][0]


async def test_show_config_handle(default_conf, update, mocker) -> None:
    default_conf["runmode"] = RunMode.DRY_RUN

    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    await telegram._show_config(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert "*模式:* `{}`".format("模拟运行") in msg_mock.call_args_list[0][0][0]
    assert "*交易所:* `binance`" in msg_mock.call_args_list[0][0][0]
    assert f"*策略:* `{CURRENT_TEST_STRATEGY}`" in msg_mock.call_args_list[0][0][0]
    assert "*止损:* `-0.1`" in msg_mock.call_args_list[0][0][0]

    msg_mock.reset_mock()
    freqtradebot.config["trailing_stop"] = True
    await telegram._show_config(update=update, context=MagicMock())
    assert msg_mock.call_count == 1
    assert "*模式:* `{}`".format("模拟运行") in msg_mock.call_args_list[0][0][0]
    assert "*交易所:* `binance`" in msg_mock.call_args_list[0][0][0]
    assert f"*策略:* `{CURRENT_TEST_STRATEGY}`" in msg_mock.call_args_list[0][0][0]
    assert "*初始止损:* `-0.1`" in msg_mock.call_args_list[0][0][0]


@pytest.mark.parametrize(
    "message_type,enter,enter_signal,leverage",
    [
        (RPCMessageType.ENTRY, "多头", "long_signal_01", None),
        (RPCMessageType.ENTRY, "多头", "long_signal_01", 1.0),
        (RPCMessageType.ENTRY, "多头", "long_signal_01", 5.0),
        (RPCMessageType.ENTRY, "空头", "short_signal_01", 2.0),
    ],
)
def test_send_msg_enter_notification(
    default_conf, mocker, caplog, message_type, enter, enter_signal, leverage
) -> None:
    default_conf["telegram"]["notification_settings"]["show_candle"] = "ohlc"
    df = DataFrame(
        {
            "open": [1.1],
            "high": [2.2],
            "low": [1.0],
            "close": [1.5],
        }
    )
    mocker.patch(
        "freqtrade.data.dataprovider.DataProvider.get_analyzed_dataframe", return_value=(df, 1)
    )

    msg = {
        "type": message_type,
        "trade_id": 1,
        "enter_tag": enter_signal,
        "exchange": "Binance",
        "pair": "ETH/BTC",
        "leverage": leverage,
        "open_rate": 1.099e-05,
        "order_type": "limit",
        "direction": enter,
        "stake_amount": 0.01465333,
        "stake_amount_fiat": 0.0,
        "stake_currency": "BTC",
        "quote_currency": "BTC",
        "base_currency": "ETH",
        "fiat_currency": "USD",
        "sub_trade": False,
        "current_rate": 1.099e-05,
        "amount": 1333.3333333333335,
        "analyzed_candle": {"open": 1.1, "high": 2.2, "low": 1.0, "close": 1.5},
        "open_date": dt_now() + timedelta(hours=-1),
    }
    telegram, freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf)

    telegram.send_msg(msg)
    leverage_text = f" ({leverage:.3g}x)" if leverage and leverage != 1.0 else ""

    assert msg_mock.call_args[0][0] == (
        f"\N{LARGE BLUE CIRCLE} *Binance (模拟):* 新交易 (#1)\n"
        f"*交易对:* `ETH/BTC`\n"
        "*K线 OHLC*: `1.1, 2.2, 1.0, 1.5`\n"
        f"*入场标签:* `{enter_signal}`\n"
        "*数量:* `1333.33333333`\n"
        f"*方向:* `{enter}"
        f"{leverage_text}`\n"
        "*开盘价:* `0.00001099 BTC`\n"
        "*当前价格:* `0.00001099 BTC`\n"
        "*总计:* `0.01465333 BTC / 180.895 USD`"
    )

    freqtradebot.config["telegram"]["notification_settings"] = {"entry": "off"}
    caplog.clear()
    msg_mock.reset_mock()
    telegram.send_msg(msg)
    assert msg_mock.call_count == 0
    assert log_has("通知 'entry' 未发送。", caplog)

    freqtradebot.config["telegram"]["notification_settings"] = {"entry": "silent"}
    caplog.clear()
    msg_mock.reset_mock()

    telegram.send_msg(msg)
    assert msg_mock.call_count == 1
    assert msg_mock.call_args_list[0][1]["disable_notification"] is True


@pytest.mark.parametrize(
    "message_type,enter_signal",
    [
        (RPCMessageType.ENTRY_CANCEL, "long_signal_01"),
        (RPCMessageType.ENTRY_CANCEL, "short_signal_01"),
    ],
)
def test_send_msg_enter_cancel_notification(
    default_conf, mocker, message_type, enter_signal
) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    telegram.send_msg(
        {
            "type": message_type,
            "enter_tag": enter_signal,
            "trade_id": 1,
            "exchange": "Binance",
            "pair": "ETH/BTC",
            "reason": CANCEL_REASON["TIMEOUT"],
        }
    )
    assert (
        msg_mock.call_args[0][0] == "\N{WARNING SIGN} *Binance (模拟):* "
        "取消 ETH/BTC (#1) 的入场订单。"
        "原因: 由于超时而取消。"
    )


def test_send_msg_protection_notification(default_conf, mocker, time_machine) -> None:
    default_conf["telegram"]["notification_settings"]["protection_trigger"] = "on"

    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    time_machine.move_to("2021-09-01 05:00:00 +00:00")
    lock = PairLocks.lock_pair("ETH/BTC", dt_now() + timedelta(minutes=6), "随机原因")
    msg = {
        "type": RPCMessageType.PROTECTION_TRIGGER,
    }
    msg.update(lock.to_json())
    telegram.send_msg(msg)
    assert (
        msg_mock.call_args[0][0] == "*保护* 由于随机原因而触发。"
        "`ETH/BTC` 将被锁定直到 `2021-09-01 05:10:00`。"
    )

    msg_mock.reset_mock()
    # 测试全局保护

    msg = {
        "type": RPCMessageType.PROTECTION_TRIGGER_GLOBAL,
    }
    lock = PairLocks.lock_pair("*", dt_now() + timedelta(minutes=100), "随机原因")
    msg.update(lock.to_json())
    telegram.send_msg(msg)
    assert (
        msg_mock.call_args[0][0] == "*保护* 由于随机原因而触发。"
        "*所有交易对* 将被锁定直到 `2021-09-01 06:45:00`。"
    )


@pytest.mark.parametrize(
    "message_type,entered,enter_signal,leverage",
    [
        (RPCMessageType.ENTRY_FILL, "多头", "long_signal_01", 1.0),
        (RPCMessageType.ENTRY_FILL, "多头", "long_signal_02", 2.0),
        (RPCMessageType.ENTRY_FILL, "空头", "short_signal_01", 2.0),
    ],
)
def test_send_msg_entry_fill_notification(
    default_conf, mocker, message_type, entered, enter_signal, leverage
) -> None:
    default_conf["telegram"]["notification_settings"]["entry_fill"] = "on"
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    telegram.send_msg(
        {
            "type": message_type,
            "trade_id": 1,
            "enter_tag": enter_signal,
            "exchange": "Binance",
            "pair": "ETH/BTC",
            "leverage": leverage,
            "stake_amount": 0.01465333,
            "direction": entered,
            "sub_trade": False,
            "stake_currency": "BTC",
            "quote_currency": "BTC",
            "base_currency": "ETH",
            "fiat_currency": "USD",
            "open_rate": 1.099e-05,
            "amount": 1333.3333333333335,
            "open_date": dt_now() - timedelta(hours=1),
        }
    )
    leverage_text = f" ({leverage:.3g}x)" if leverage != 1.0 else ""
    assert msg_mock.call_args[0][0] == (
        f"\N{CHECK MARK} *Binance (模拟):* 新交易已成交 (#1)\n"
        f"*交易对:* `ETH/BTC`\n"
        f"*入场标签:* `{enter_signal}`\n"
        "*数量:* `1333.33333333`\n"
        f"*方向:* `{entered}"
        f"{leverage_text}`\n"
        "*开盘价:* `0.00001099 BTC`\n"
        "*总计:* `0.01465333 BTC / 180.895 USD`"
    )

    msg_mock.reset_mock()
    telegram.send_msg(
        {
            "type": message_type,
            "trade_id": 1,
            "enter_tag": enter_signal,
            "exchange": "Binance",
            "pair": "ETH/BTC",
            "leverage": leverage,
            "stake_amount": 0.01465333,
            "sub_trade": True,
            "direction": entered,
            "stake_currency": "BTC",
            "quote_currency": "BTC",
            "base_currency": "ETH",
            "fiat_currency": "USD",
            "open_rate": 1.099e-05,
            "amount": 1333.3333333333335,
            "open_date": dt_now() - timedelta(hours=1),
        }
    )

    assert msg_mock.call_args[0][0] == (
        f"\N{CHECK MARK} *Binance (模拟):* 仓位增加已成交 (#1)\n"
        f"*交易对:* `ETH/BTC`\n"
        f"*入场标签:* `{enter_signal}`\n"
        "*数量:* `1333.33333333`\n"
        f"*方向:* `{entered}"
        f"{leverage_text}`\n"
        "*开盘价:* `0.00001099 BTC`\n"
        "*新总计:* `0.01465333 BTC / 180.895 USD`"
    )


def test_send_msg_exit_notification(default_conf, mocker) -> None:
    with time_machine.travel("2022-09-01 05:00:00 +00:00", tick=False):
        telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

        old_convamount = telegram._rpc._fiat_converter.convert_amount
        telegram._rpc._fiat_converter.convert_amount = lambda a, b, c: -24.812
        telegram.send_msg(
            {
                "type": RPCMessageType.EXIT,
                "trade_id": 1,
                "exchange": "Binance",
                "pair": "KEY/ETH",
                "leverage": 1.0,
                "direction": "多头",
                "gain": "loss",
                "order_rate": 3.201e-04,
                "amount": 1333.3333333333335,
                "order_type": "market",
                "open_rate": 7.5e-04,
                "current_rate": 3.201e-04,
                "profit_amount": -0.05746268,
                "profit_ratio": -0.57405275,
                "stake_currency": "ETH",
                "quote_currency": "ETH",
                "base_currency": "KEY",
                "fiat_currency": "USD",
                "enter_tag": "buy_signal1",
                "exit_reason": ExitType.STOP_LOSS.value,
                "open_date": dt_now() - timedelta(hours=1),
                "close_date": dt_now(),
            }
        )
        assert msg_mock.call_args[0][0] == (
            "\N{WARNING SIGN} *Binance (模拟):* 退出 KEY/ETH (#1)\n"
            "*未实现利润:* `-57.41% (亏损: -0.05746 ETH / -24.812 USD)`\n"
            "*入场标签:* `buy_signal1`\n"
            "*退出原因:* `stop_loss`\n"
            "*方向:* `多头`\n"
            "*数量:* `1333.33333333`\n"
            "*开盘价:* `0.00075 ETH`\n"
            "*当前价格:* `0.0003201 ETH`\n"
            "*退出价格:* `0.0003201 ETH`\n"
            "*持续时间:* `1:00:00 (60.0 分钟)`"
        )

        msg_mock.reset_mock()
        telegram.send_msg(
            {
                "type": RPCMessageType.EXIT,
                "trade_id": 1,
                "exchange": "Binance",
                "pair": "KEY/ETH",
                "direction": "多头",
                "gain": "loss",
                "order_rate": 3.201e-04,
                "amount": 1333.3333333333335,
                "order_type": "market",
                "open_rate": 7.5e-04,
                "current_rate": 3.201e-04,
                "cumulative_profit": -0.15746268,
                "profit_amount": -0.05746268,
                "profit_ratio": -0.57405275,
                "stake_currency": "ETH",
                "quote_currency": "ETH",
                "base_currency": "KEY",
                "fiat_currency": "USD",
                "enter_tag": "buy_signal1",
                "exit_reason": ExitType.STOP_LOSS.value,
                "open_date": dt_now() - timedelta(days=1, hours=2, minutes=30),
                "close_date": dt_now(),
                "stake_amount": 0.01,
                "sub_trade": True,
            }
        )
        assert msg_mock.call_args[0][0] == (
            "\N{WARNING SIGN} *Binance (模拟):* 部分退出 KEY/ETH (#1)\n"
            "*未实现子利润:* `-57.41% (亏损: -0.05746 ETH / -24.812 USD)`\n"
            "*累计利润:* `-0.15746 ETH / -24.812 USD`\n"
            "*入场标签:* `buy_signal1`\n"
            "*退出原因:* `stop_loss`\n"
            "*方向:* `多头`\n"
            "*数量:* `1333.33333333`\n"
            "*开盘价:* `0.00075 ETH`\n"
            "*当前价格:* `0.0003201 ETH`\n"
            "*退出价格:* `0.0003201 ETH`\n"
            "*剩余:* `0.01 ETH / -24.812 USD`"
        )

        msg_mock.reset_mock()
        telegram.send_msg(
            {
                "type": RPCMessageType.EXIT,
                "trade_id": 1,
                "exchange": "Binance",
                "pair": "KEY/ETH",
                "direction": "多头",
                "gain": "loss",
                "order_rate": 3.201e-04,
                "amount": 1333.3333333333335,
                "order_type": "market",
                "open_rate": 7.5e-04,
                "current_rate": 3.201e-04,
                "profit_amount": -0.05746268,
                "profit_ratio": -0.57405275,
                "stake_currency": "ETH",
                "quote_currency": "ETH",
                "base_currency": "KEY",
                "fiat_currency": None,
                "enter_tag": "buy_signal1",
                "exit_reason": ExitType.STOP_LOSS.value,
                "open_date": dt_now() - timedelta(days=1, hours=2, minutes=30),
                "close_date": dt_now(),
            }
        )
        assert msg_mock.call_args[0][0] == (
            "\N{WARNING SIGN} *Binance (模拟):* 退出 KEY/ETH (#1)\n"
            "*未实现利润:* `-57.41% (亏损: -0.05746 ETH)`\n"
            "*入场标签:* `buy_signal1`\n"
            "*退出原因:* `stop_loss`\n"
            "*方向:* `多头`\n"
            "*数量:* `1333.33333333`\n"
            "*开盘价:* `0.00075 ETH`\n"
            "*当前价格:* `0.0003201 ETH`\n"
            "*退出价格:* `0.0003201 ETH`\n"
            "*持续时间:* `1 天, 2:30:00 (1590.0 分钟)`"
        )
        # 重置单例函数以避免随机中断
        telegram._rpc._fiat_converter.convert_amount = old_convamount


async def test_send_msg_exit_cancel_notification(default_conf, mocker) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    old_convamount = telegram._rpc._fiat_converter.convert_amount
    telegram._rpc._fiat_converter.convert_amount = lambda a, b, c: -24.812
    telegram.send_msg(
        {
            "type": RPCMessageType.EXIT_CANCEL,
            "trade_id": 1,
            "exchange": "Binance",
            "pair": "KEY/ETH",
            "reason": "在交易所取消",
        }
    )
    assert msg_mock.call_args[0][0] == (
        "\N{WARNING SIGN} *Binance (模拟):* 取消 KEY/ETH (#1) 的退出订单。"
        " 原因: 在交易所取消。"
    )

    msg_mock.reset_mock()
    # 测试实盘模式（无模拟附录）
    telegram._config["dry_run"] = False
    telegram.send_msg(
        {
            "type": RPCMessageType.EXIT_CANCEL,
            "trade_id": 1,
            "exchange": "Binance",
            "pair": "KEY/ETH",
            "reason": "timeout",
        }
    )
    assert msg_mock.call_args[0][0] == (
        "\N{WARNING SIGN} *Binance:* 取消 KEY/ETH (#1) 的退出订单。原因: timeout。"
    )
    # 重置单例函数以避免随机中断
    telegram._rpc._fiat_converter.convert_amount = old_convamount


@pytest.mark.parametrize(
    "direction,enter_signal,leverage",
    [
        ("多头", "long_signal_01", None),
        ("多头", "long_signal_01", 1.0),
        ("多头", "long_signal_01", 5.0),
        ("空头", "short_signal_01", 2.0),
    ],
)
def test_send_msg_exit_fill_notification(
    default_conf, mocker, direction, enter_signal, leverage
) -> None:
    default_conf["telegram"]["notification_settings"]["exit_fill"] = "on"
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    with time_machine.travel("2022-09-01 05:00:00 +00:00", tick=False):
        telegram.send_msg(
            {
                "type": RPCMessageType.EXIT_FILL,
                "trade_id": 1,
                "exchange": "Binance",
                "pair": "KEY/ETH",
                "leverage": leverage,
                "direction": direction,
                "gain": "loss",
                "limit": 3.201e-04,
                "amount": 1333.3333333333335,
                "order_type": "market",
                "open_rate": 7.5e-04,
                "close_rate": 3.201e-04,
                "profit_amount": -0.05746268,
                "profit_ratio": -0.57405275,
                "stake_currency": "ETH",
                "quote_currency": "ETH",
                "base_currency": "KEY",
                "fiat_currency": None,
                "enter_tag": enter_signal,
                "exit_reason": ExitType.STOP_LOSS.value,
                "open_date": dt_now() - timedelta(days=1, hours=2, minutes=30),
                "close_date": dt_now(),
            }
        )

        leverage_text = f" ({leverage:.3g}x)`\n" if leverage and leverage != 1.0 else "`\n"
        assert msg_mock.call_args[0][0] == (
            "\N{WARNING SIGN} *Binance (模拟):* 已退出 KEY/ETH (#1)\n"
            "*利润:* `-57.41% (亏损: -0.05746 ETH)`\n"
            f"*入场标签:* `{enter_signal}`\n"
            "*退出原因:* `stop_loss`\n"
            f"*方向:* `{direction}"
            f"{leverage_text}"
            "*数量:* `1333.33333333`\n"
            "*开盘价:* `0.00075 ETH`\n"
            "*退出价格:* `0.0003201 ETH`\n"
            "*持续时间:* `1 天, 2:30:00 (1590.0 分钟)`"
        )


def test_send_msg_status_notification(default_conf, mocker) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    telegram.send_msg({"type": RPCMessageType.STATUS, "status": "running"})
    assert msg_mock.call_args[0][0] == "*状态:* `running`"


async def test_warning_notification(default_conf, mocker) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    telegram.send_msg({"type": RPCMessageType.WARNING, "status": "message"})
    assert msg_mock.call_args[0][0] == "\N{WARNING SIGN} *警告:* `message`"


def test_startup_notification(default_conf, mocker) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    telegram.send_msg({"type": RPCMessageType.STARTUP, "status": "*自定义:* `Hello World`"})
    assert msg_mock.call_args[0][0] == "*自定义:* `Hello World`"


def test_send_msg_strategy_msg_notification(default_conf, mocker) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    telegram.send_msg({"type": RPCMessageType.STRATEGY_MSG, "msg": "hello world, 测试消息"})
    assert msg_mock.call_args[0][0] == "hello world, 测试消息"


def test_send_msg_unknown_type(default_conf, mocker) -> None:
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)
    telegram.send_msg(
        {
            "type": None,
        }
    )
    assert msg_mock.call_count == 0


@pytest.mark.parametrize(
    "message_type,enter,enter_signal,leverage",
    [
        (RPCMessageType.ENTRY, "多头", "long_signal_01", None),
        (RPCMessageType.ENTRY, "多头", "long_signal_01", 2.0),
        (RPCMessageType.ENTRY, "空头", "short_signal_01", 2.0),
    ],
)
def test_send_msg_buy_notification_no_fiat(
    default_conf, mocker, message_type, enter, enter_signal, leverage
) -> None:
    del default_conf["fiat_display_currency"]
    default_conf["dry_run"] = False
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    telegram.send_msg(
        {
            "type": message_type,
            "enter_tag": enter_signal,
            "trade_id": 1,
            "exchange": "Binance",
            "pair": "ETH/BTC",
            "leverage": leverage,
            "open_rate": 1.099e-05,
            "order_type": "limit",
            "direction": enter,
            "sub_trade": False,
            "stake_amount": 0.01465333,
            "stake_amount_fiat": 0.0,
            "stake_currency": "BTC",
            "quote_currency": "BTC",
            "base_currency": "ETH",
            "fiat_currency": None,
            "current_rate": 1.099e-05,
            "amount": 1333.3333333333335,
            "open_date": dt_now() - timedelta(hours=1),
        }
    )

    leverage_text = f" ({leverage:.3g}x)" if leverage and leverage != 1.0 else ""
    assert msg_mock.call_args[0][0] == (
        f"\N{LARGE BLUE CIRCLE} *Binance:* 新交易 (#1)\n"
        "*交易对:* `ETH/BTC`\n"
        f"*入场标签:* `{enter_signal}`\n"
        "*数量:* `1333.33333333`\n"
        f"*方向:* `{enter}"
        f"{leverage_text}`\n"
        "*开盘价:* `0.00001099 BTC`\n"
        "*当前价格:* `0.00001099 BTC`\n"
        "*总计:* `0.01465333 BTC`"
    )


@pytest.mark.parametrize(
    "direction,enter_signal,leverage",
    [
        ("多头", "long_signal_01", None),
        ("多头", "long_signal_01", 1.0),
        ("多头", "long_signal_01", 5.0),
        ("空头", "short_signal_01", 2.0),
    ],
)
@pytest.mark.parametrize("fiat", ["", None])
def test_send_msg_exit_notification_no_fiat(
    default_conf, mocker, direction, enter_signal, leverage, time_machine, fiat
) -> None:
    if fiat is None:
        del default_conf["fiat_display_currency"]
    else:
        default_conf["fiat_display_currency"] = fiat
    time_machine.move_to("2022-05-02 00:00:00 +00:00", tick=False)
    telegram, _, msg_mock = get_telegram_testobject(mocker, default_conf)

    telegram.send_msg(
        {
            "type": RPCMessageType.EXIT,
            "trade_id": 1,
            "exchange": "Binance",
            "pair": "KEY/ETH",
            "gain": "loss",
            "leverage": leverage,
            "direction": direction,
            "sub_trade": False,
            "order_rate": 3.201e-04,
            "amount": 1333.3333333333335,
            "order_type": "limit",
            "open_rate": 7.5e-04,
            "current_rate": 3.201e-04,
            "profit_amount": -0.05746268,
            "profit_ratio": -0.57405275,
            "stake_currency": "ETH",
            "quote_currency": "ETH",
            "base_currency": "KEY",
            "fiat_currency": "USD",
            "enter_tag": enter_signal,
            "exit_reason": ExitType.STOP_LOSS.value,
            "open_date": dt_now() - timedelta(hours=2, minutes=35, seconds=3),
            "close_date": dt_now(),
        }
    )

    leverage_text = f" ({leverage:.3g}x)" if leverage and leverage != 1.0 else ""
    assert msg_mock.call_args[0][0] == (
        "\N{WARNING SIGN} *Binance (模拟):* 退出 KEY/ETH (#1)\n"
        "*未实现利润:* `-57.41% (亏损: -0.05746 ETH)`\n"
        f"*入场标签:* `{enter_signal}`\n"
        "*退出原因:* `stop_loss`\n"
        f"*方向:* `{direction}"
        f"{leverage_text}`\n"
        "*数量:* `1333.33333333`\n"
        "*开盘价:* `0.00075 ETH`\n"
        "*当前价格:* `0.0003201 ETH`\n"
        "*退出价格:* `0.0003201 ETH`\n"
        "*持续时间:* `2:35:03 (155.1 分钟)`"
    )


@pytest.mark.parametrize(
    "msg,expected",
    [
        ({"profit_ratio": 0.201, "exit_reason": "roi"}, "\N{ROCKET}"),
        ({"profit_ratio": 0.051, "exit_reason": "roi"}, "\N{ROCKET}"),
        ({"profit_ratio": 0.0256, "exit_reason": "roi"}, "\N{EIGHT SPOKED ASTERISK}"),
        ({"profit_ratio": 0.01, "exit_reason": "roi"}, "\N{EIGHT SPOKED ASTERISK}"),
        ({"profit_ratio": 0.0, "exit_reason": "roi"}, "\N{EIGHT SPOKED ASTERISK}"),
        ({"profit_ratio": -0.05, "exit_reason": "stop_loss"}, "\N{WARNING SIGN}"),
        ({"profit_ratio": -0.02, "exit_reason": "sell_signal"}, "\N{CROSS MARK}"),
    ],
)
def test__exit_emoji(default_conf, mocker, msg, expected):
    del default_conf["fiat_display_currency"]

    telegram, _, _ = get_telegram_testobject(mocker, default_conf)

    assert telegram._get_exit_emoji(msg) == expected


async def test_telegram__send_msg(default_conf, mocker, caplog) -> None:
    mocker.patch("freqtrade.rpc.telegram.Telegram._init", MagicMock())
    bot = MagicMock()
    bot.send_message = AsyncMock()
    bot.edit_message_text = AsyncMock()
    telegram, _, _ = get_telegram_testobject(mocker, default_conf, mock=False)
    telegram._app = MagicMock()
    telegram._app.bot = bot

    await telegram._send_msg("测试")
    assert len(bot.method_calls) == 1

    # 测试更新
    query = MagicMock()
    query.edit_message_text = AsyncMock()
    await telegram._send_msg("测试", callback_path="DeadBeef", query=query, reload_able=True)
    assert query.edit_message_text.call_count == 1
    assert "更新时间: " in query.edit_message_text.call_args_list[0][1]["text"]

    query.edit_message_text = AsyncMock(side_effect=BadRequest("未修改"))
    await telegram._send_msg("测试", callback_path="DeadBeef", query=query)
    assert query.edit_message_text.call_count == 1
    assert not log_has_re(r"TelegramError: .*", caplog)

    query.edit_message_text = AsyncMock(side_effect=BadRequest(""))
    await telegram._send_msg("测试2", callback_path="DeadBeef", query=query)
    assert query.edit_message_text.call_count == 1
    assert log_has_re(r"TelegramError: .*", caplog)

    query.edit_message_text = AsyncMock(side_effect=TelegramError("DeadBEEF"))
    await telegram._send_msg("测试3", callback_path="DeadBeef", query=query)

    assert log_has_re(r"TelegramError: DeadBEEF! 放弃尝试.*", caplog)


async def test__send_msg_network_error(default_conf, mocker, caplog) -> None:
    mocker.patch("freqtrade.rpc.telegram.Telegram._init", MagicMock())
    bot = MagicMock()
    bot.send_message = MagicMock(side_effect=NetworkError("网络错误"))
    telegram, _, _ = get_telegram_testobject(mocker, default_conf, mock=False)
    telegram._app = MagicMock()
    telegram._app.bot = bot

    telegram._config["telegram"]["enabled"] = True
    await telegram._send_msg("测试")

    # 机器人应该尝试发送两次
    assert len(bot.method_calls) == 2
    assert log_has("Telegram 网络错误: 网络错误! 再试一次。", caplog)


@pytest.mark.filterwarnings("ignore:.*ChatPermissions")
async def test__send_msg_keyboard(default_conf, mocker, caplog) -> None:
    mocker.patch("freqtrade.rpc.telegram.Telegram._init", MagicMock())
    bot = MagicMock()
    bot.send_message = AsyncMock()
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc = RPC(freqtradebot)

    invalid_keys_list = [["/not_valid", "/profit"], ["/daily"], ["/alsoinvalid"]]
    default_keys_list = [
        ["/daily", "/profit", "/balance"],
        ["/status", "/status table", "/performance"],
        ["/count", "/start", "/stop", "/help"],
    ]
    default_keyboard = ReplyKeyboardMarkup(default_keys_list)

    custom_keys_list = [
        ["/daily", "/stats", "/balance", "/profit", "/profit 5"],
        ["/count", "/start", "/reload_config", "/help"],
    ]
    custom_keyboard = ReplyKeyboardMarkup(custom_keys_list)

    def init_telegram(freqtradebot):
        telegram = Telegram(rpc, default_conf)
        telegram._app = MagicMock()
        telegram._app.bot = bot
        return telegram

    # 配置中没有键盘 -> 默认键盘
    freqtradebot.config["telegram"]["enabled"] = True
    telegram = init_telegram(freqtradebot)
    await telegram._send_msg("测试")
    used_keyboard = bot.send_message.call_args[1]["reply_markup"]
    assert used_keyboard == default_keyboard

    # 配置中的无效键盘 -> 默认键盘
    freqtradebot.config["telegram"]["enabled"] = True
    freqtradebot.config["telegram"]["keyboard"] = invalid_keys_list
    err_msg = (
        re.escape(
            "config.telegram.keyboard: 自定义 Telegram 键盘的无效命令: "
            "['/not_valid', '/alsoinvalid']"
            "\n有效命令是: "
        )
        + r"*"
    )
    with pytest.raises(OperationalException, match=err_msg):
        telegram = init_telegram(freqtradebot)

    # 配置中的有效键盘 -> 自定义键盘
    freqtradebot.config["telegram"]["enabled"] = True
    freqtradebot.config["telegram"]["keyboard"] = custom_keys_list
    telegram = init_telegram(freqtradebot)
    await telegram._send_msg("测试")
    used_keyboard = bot.send_message.call_args[1]["reply_markup"]
    assert used_keyboard == custom_keyboard
    assert log_has(
        "使用 config.json 中的自定义键盘: "
        "[['/daily', '/stats', '/balance', '/profit', '/profit 5'], ['/count', "
        "'/start', '/reload_config', '/help']]",
        caplog,
    )


async def test_change_market_direction(default_conf, mocker, update) -> None:
    telegram, _, _msg_mock = get_telegram_testobject(mocker, default_conf)
    assert telegram._rpc._freqtrade.strategy.market_direction == MarketDirection.NONE
    context = MagicMock()
    context.args = ["long"]
    await telegram._changemarketdir(update, context)
    assert telegram._rpc._freqtrade.strategy.market_direction == MarketDirection.LONG
    context = MagicMock()
    context.args = ["invalid"]
    await telegram._changemarketdir(update, context)
    assert telegram._rpc._freqtrade.strategy.market_direction == MarketDirection.LONG


async def test_telegram_list_custom_data(default_conf_usdt, update, ticker, fee, mocker) -> None:
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )
    telegram, _freqtradebot, msg_mock = get_telegram_testobject(mocker, default_conf_usdt)

    # 创建一些测试数据
    create_mock_trades_usdt(fee)
    # 没有交易ID
    context = MagicMock()
    await telegram._list_custom_data(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "交易ID未设置。" in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    #
    context.args = ["1"]
    await telegram._list_custom_data(update=update, context=context)
    assert msg_mock.call_count == 1
    assert "未找到交易ID 1 的自定义数据。" in msg_mock.call_args_list[0][0][0]
    msg_mock.reset_mock()

    # 添加一些自定义数据
    trade1 = Trade.get_trades_proxy()[0]
    trade1.set_custom_data("test_int", 1)
    trade1.set_custom_data("test_dict", {"test": "dict"})
    Trade.commit()
    context.args = [f"{trade1.id}"]
    await telegram._list_custom_data(update=update, context=context)
    assert msg_mock.call_count == 3
    assert "找到自定义数据条目: " in msg_mock.call_args_list[0][0][0]
    assert (
        "*键:* `test_int`\n*类型:* `int`\n*值:* `1`\n*创建日期:*"
    ) in msg_mock.call_args_list[1][0][0]
    assert (
        "*键:* `test_dict`\n*类型:* `dict`\n*值:* `{'test': 'dict'}`\n*创建日期:* `"
    ) in msg_mock.call_args_list[2][0][0]

    msg_mock.reset_mock()


def test_noficiation_settings(default_conf_usdt, mocker):
    (telegram, _, _) = get_telegram_testobject(mocker, default_conf_usdt)
    telegram._config["telegram"].update(
        {
            "notification_settings": {
                "status": "silent",
                "warning": "on",
                "startup": "off",
                "entry": "silent",
                "entry_fill": "on",
                "entry_cancel": "silent",
                "exit": {
                    "roi": "silent",
                    "emergency_exit": "on",
                    "force_exit": "on",
                    "exit_signal": "silent",
                    "trailing_stop_loss": "on",
                    "stop_loss": "on",
                    "stoploss_on_exchange": "on",
                    "custom_exit": "silent",
                    "partial_exit": "off",
                },
                "exit_fill": {
                    "roi": "silent",
                    "partial_exit": "off",
                    "*": "silent",  # 默认为静默
                },
                "exit_cancel": "on",
                "protection_trigger": "off",
                "protection_trigger_global": "on",
                "strategy_msg": "off",
                "show_candle": "off",
            }
        }
    )

    loudness = telegram._message_loudness

    assert loudness({"type": RPCMessageType.ENTRY, "exit_reason": ""}) == "silent"
    assert loudness({"type": RPCMessageType.ENTRY_FILL, "exit_reason": ""}) == "on"
    assert loudness({"type": RPCMessageType.EXIT, "exit_reason": ""}) == "on"
    # 由于 "*" 定义而默认为静默
    assert loudness({"type": RPCMessageType.EXIT_FILL, "exit_reason": ""}) == "silent"
    assert loudness({"type": RPCMessageType.PROTECTION_TRIGGER, "exit_reason": ""}) == "off"
    assert loudness({"type": RPCMessageType.EXIT, "exit_reason": "roi"}) == "silent"
    assert loudness({"type": RPCMessageType.EXIT, "exit_reason": "partial_exit"}) == "off"
    # 未给定的键默认为开启
    assert loudness({"type": RPCMessageType.EXIT, "exit_reason": "cust_exit112"}) == "on"

    assert loudness({"type": RPCMessageType.EXIT_FILL, "exit_reason": "roi"}) == "silent"
    assert loudness({"type": RPCMessageType.EXIT_FILL, "exit_reason": "partial_exit"}) == "off"
    # 由于 "*" 定义而默认为静默
    assert loudness({"type": RPCMessageType.EXIT_FILL, "exit_reason": "cust_exit112"}) == "silent"

    # 简化的退出设置
    telegram._config["telegram"].update(
        {
            "notification_settings": {
                "status": "silent",
                "warning": "on",
                "startup": "off",
                "entry": "silent",
                "entry_fill": "on",
                "entry_cancel": "silent",
                "exit": "off",
                "exit_cancel": "on",
                "exit_fill": "on",
                "protection_trigger": "off",
                "protection_trigger_global": "on",
                "strategy_msg": "off",
                "show_candle": "off",
            }
        }
    )

    assert loudness({"type": RPCMessageType.EXIT_FILL, "exit_reason": "roi"}) == "on"
    # 所有常规退出都是关闭的
    assert loudness({"type": RPCMessageType.EXIT, "exit_reason": "roi"}) == "off"
    assert loudness({"type": RPCMessageType.EXIT, "exit_reason": "partial_exit"}) == "off"
    assert loudness({"type": RPCMessageType.EXIT, "exit_reason": "cust_exit112"}) == "off"


async def test__tg_info(default_conf_usdt, mocker, update):
    (telegram, _, _) = get_telegram_testobject(mocker, default_conf_usdt)
    context = AsyncMock()

    await telegram._tg_info(update, context)

    assert context.bot.send_message.call_count == 1
    content = context.bot.send_message.call_args[1]["text"]
    assert "Freqtrade 机器人信息:\n" in content
    assert '"chat_id": "1235"' in content