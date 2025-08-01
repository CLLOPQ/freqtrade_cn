# pragma pylint: disable=missing-docstring
from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from sqlalchemy import select

from freqtrade.constants import UNLIMITED_STAKE_AMOUNT
from freqtrade.exceptions import DependencyException
from freqtrade.persistence import Trade
from tests.conftest import (
    EXMS,
    create_mock_trades,
    create_mock_trades_usdt,
    get_patched_freqtradebot,
    patch_wallet,
)


def test_sync_wallet_at_boot(mocker, default_conf):
    """测试启动时同步钱包功能"""
    default_conf["dry_run"] = False
    # 模拟交易所返回的余额数据
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(
            return_value={
                "BNT": {"free": 1.0, "used": 2.0, "total": 3.0},
                "GAS": {"free": 0.260739, "used": 0.0, "total": 0.260739},
                "USDT": {"free": 20, "used": 20, "total": 40},
            }
        ),
    )

    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    # 验证初始钱包数据
    assert len(freqtrade.wallets._wallets) == 3
    assert freqtrade.wallets._wallets["BNT"].free == 1.0
    assert freqtrade.wallets._wallets["BNT"].used == 2.0
    assert freqtrade.wallets._wallets["BNT"].total == 3.0
    assert freqtrade.wallets._wallets["GAS"].free == 0.260739
    assert freqtrade.wallets._wallets["GAS"].used == 0.0
    assert freqtrade.wallets._wallets["GAS"].total == 0.260739
    assert freqtrade.wallets.get_free("BNT") == 1.0
    assert "USDT" in freqtrade.wallets._wallets
    assert freqtrade.wallets._last_wallet_refresh is not None

    # 模拟更新后的余额数据
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(
            return_value={
                "BNT": {"free": 1.2, "used": 1.9, "total": 3.5},
                "GAS": {"free": 0.270739, "used": 0.1, "total": 0.260439},
            }
        ),
    )

    # 执行钱包更新
    freqtrade.wallets.update()

    # 验证更新后的钱包数据
    # USDT在第二次结果中缺失，因此不应再存在
    assert len(freqtrade.wallets._wallets) == 2
    assert freqtrade.wallets._wallets["BNT"].free == 1.2
    assert freqtrade.wallets._wallets["BNT"].used == 1.9
    assert freqtrade.wallets._wallets["BNT"].total == 3.5
    assert freqtrade.wallets._wallets["GAS"].free == 0.270739
    assert freqtrade.wallets._wallets["GAS"].used == 0.1
    assert freqtrade.wallets._wallets["GAS"].total == 0.260439
    assert freqtrade.wallets.get_free("GAS") == 0.270739
    assert freqtrade.wallets.get_used("GAS") == 0.1
    assert freqtrade.wallets.get_total("GAS") == 0.260439
    assert freqtrade.wallets.get_owned("GAS/USDT", "GAS") == 0.260439

    # 测试条件更新
    update_mock = mocker.patch("freqtrade.wallets.Wallets._update_live")
    freqtrade.wallets.update(False)
    assert update_mock.call_count == 0
    freqtrade.wallets.update()
    assert update_mock.call_count == 1

    # 测试不存在的货币
    assert freqtrade.wallets.get_free("NOCURRENCY") == 0
    assert freqtrade.wallets.get_used("NOCURRENCY") == 0
    assert freqtrade.wallets.get_total("NOCURRENCY") == 0
    assert freqtrade.wallets.get_owned("NOCURRENCY/USDT", "NOCURRENCY") == 0


def test_sync_wallet_missing_data(mocker, default_conf):
    """测试同步包含缺失数据的钱包"""
    default_conf["dry_run"] = False
    # 模拟包含不完整数据的余额（GAS缺少used字段）
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(
            return_value={
                "BNT": {"free": 1.0, "used": 2.0, "total": 3.0},
                "GAS": {"free": 0.260739, "total": 0.260739},
            }
        ),
    )

    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    # 验证钱包数据已正确处理缺失字段
    assert len(freqtrade.wallets._wallets) == 2
    assert freqtrade.wallets._wallets["BNT"].free == 1.0
    assert freqtrade.wallets._wallets["BNT"].used == 2.0
    assert freqtrade.wallets._wallets["BNT"].total == 3.0
    assert freqtrade.wallets._wallets["GAS"].free == 0.260739
    assert freqtrade.wallets._wallets["GAS"].used == 0.0  # 缺失时默认为0
    assert freqtrade.wallets._wallets["GAS"].total == 0.260739
    assert freqtrade.wallets.get_free("GAS") == 0.260739


def test_get_trade_stake_amount_no_stake_amount(default_conf, mocker) -> None:
    """测试在没有足够保证金时获取交易 stake 金额"""
    # 模拟可用余额不足
    patch_wallet(mocker, free=default_conf["stake_amount"] * 0.5)
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    # 验证在余额不足时抛出异常
    with pytest.raises(DependencyException, match=r".*stake amount.*"):
        freqtrade.wallets.get_trade_stake_amount("ETH/BTC", 1)


@pytest.mark.parametrize(
    "balance_ratio,capital,result1,result2",
    [
        (1, None, 50, 66.66666),
        (0.99, None, 49.5, 66.0),
        (0.50, None, 25, 33.3333),
        # 当指定capital时忽略balance_ratio
        (1, 100, 50, 0.0),
        (0.99, 200, 50, 66.66666),
        (0.99, 150, 50, 50),
        (0.50, 50, 25, 0.0),
        (0.50, 10, 5, 0.0),
    ],
)
def test_get_trade_stake_amount_unlimited_amount(
    default_conf,
    ticker,
    balance_ratio,
    capital,
    result1,
    result2,
    limit_buy_order_open,
    fee,
    mocker,
) -> None:
    """测试在无限 stake 模式下获取交易金额"""
    # 模拟交易所方法
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        create_order=MagicMock(return_value=limit_buy_order_open),
        get_fee=fee,
    )

    # 配置无限stake模式
    conf = deepcopy(default_conf)
    conf["stake_amount"] = UNLIMITED_STAKE_AMOUNT
    conf["dry_run_wallet"] = 100
    conf["tradable_balance_ratio"] = balance_ratio
    if capital is not None:
        conf["available_capital"] = capital

    freqtrade = get_patched_freqtradebot(mocker, conf)

    # 无未平仓交易时，订单金额应为 '余额 / 最大开仓数'
    result = freqtrade.wallets.get_trade_stake_amount("ETH/USDT", 2)
    assert result == result1

    # 创建一个交易，订单金额应为 '余额 / (最大开仓数 - 已开仓数)'
    freqtrade.execute_entry("ETH/USDT", result)
    result = freqtrade.wallets.get_trade_stake_amount("LTC/USDT", 2)
    assert result == result1

    # 创建2个交易，订单金额应为0（达到最大开仓数）
    freqtrade.execute_entry("LTC/BTC", result)
    result = freqtrade.wallets.get_trade_stake_amount("XRP/USDT", 2)
    assert result == 0

    # 调整钱包金额和最大开仓数
    freqtrade.config["dry_run_wallet"] = 200
    freqtrade.wallets._start_cap["BTC"] = 200
    result = freqtrade.wallets.get_trade_stake_amount("XRP/USDT", 3)
    assert round(result, 4) == round(result2, 4)

    # 最大开仓数为0时，不应交易
    result = freqtrade.wallets.get_trade_stake_amount("NEO/USDT", 0)
    assert result == 0


@pytest.mark.parametrize(
    "stake_amount,min_stake,stake_available,max_stake,trade_amount,expected",
    [
        (22, 11, 50, 10000, None, 22),
        (100, 11, 500, 10000, None, 100),
        (1000, 11, 500, 10000, None, 500),  # 超过可用余额
        (700, 11, 1000, 400, None, 400),  # 超过最大限额，低于可用余额
        (20, 15, 10, 10000, None, 0),  # 最小限额 > 可用余额
        (9, 11, 100, 10000, None, 11),  # 低于最小限额
        (1, 15, 10, 10000, None, 0),  # 低于最小限额且最小限额 > 可用余额
        (20, 50, 100, 10000, None, 0),  # 低于最小限额且stake * 1.3 > 最小限额
        (1000, None, 1000, 10000, None, 1000),  # 无法确定最小限额
        # 重新买入 - 导致过高的stake金额，应调整
        (2000, 15, 2000, 3000, 1500, 1500),
    ],
)
def test_validate_stake_amount(
    mocker,
    default_conf,
    stake_amount,
    min_stake,
    stake_available,
    max_stake,
    trade_amount,
    expected,
):
    """测试验证stake金额的功能"""
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    # 模拟可用stake金额
    mocker.patch(
        "freqtrade.wallets.Wallets.get_available_stake_amount", return_value=stake_available
    )
    
    # 验证stake金额验证结果
    res = freqtrade.wallets.validate_stake_amount(
        "XRP/USDT", stake_amount, min_stake, max_stake, trade_amount
    )
    assert res == expected


@pytest.mark.parametrize(
    "available_capital,closed_profit,open_stakes,free,expected",
    [
        (None, 10, 100, 910, 1000),
        (None, 0, 0, 2500, 2500),
        (None, 500, 0, 2500, 2000),
        (None, 500, 0, 2500, 2000),
        (None, -70, 0, 1930, 2000),
        # 当设置available_capital时，仅该值有效
        (100, 0, 0, 0, 100),
        (1000, 0, 2, 5, 1000),
        (1235, 2250, 2, 5, 1235),
        (1235, -2250, 2, 5, 1235),
    ],
)
def test_get_starting_balance(
    mocker, default_conf, available_capital, closed_profit, open_stakes, free, expected
):
    """测试获取初始余额的功能"""
    if available_capital:
        default_conf["available_capital"] = available_capital
    
    # 模拟必要的方法返回值
    mocker.patch(
        "freqtrade.persistence.models.Trade.get_total_closed_profit", return_value=closed_profit
    )
    mocker.patch(
        "freqtrade.persistence.models.Trade.total_open_trades_stakes", return_value=open_stakes
    )
    mocker.patch("freqtrade.wallets.Wallets.get_free", return_value=free)

    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    # 验证初始余额计算结果
    assert freqtrade.wallets.get_starting_balance() == expected * (1 if available_capital else 0.99)


def test_sync_wallet_futures_live(mocker, default_conf):
    """测试实盘期货模式下同步钱包"""
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    
    # 模拟期货持仓数据
    mock_result = [
        {
            "symbol": "ETH/USDT:USDT",
            "timestamp": None,
            "datetime": None,
            "initialMargin": 0.0,
            "initialMarginPercentage": None,
            "maintenanceMargin": 0.0,
            "maintenanceMarginPercentage": 0.005,
            "entryPrice": 0.0,
            "notional": 100.0,
            "leverage": 5.0,
            "unrealizedPnl": 0.0,
            "contracts": 100.0,
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
            "symbol": "ADA/USDT:USDT",
            "timestamp": None,
            "datetime": None,
            "initialMargin": 0.0,
            "initialMarginPercentage": None,
            "maintenanceMargin": 0.0,
            "maintenanceMarginPercentage": 0.005,
            "entryPrice": 0.0,
            "notional": 100.0,
            "leverage": 5.0,
            "unrealizedPnl": 0.0,
            "contracts": 100.0,
            "contractSize": 1,
            "marginRatio": None,
            "liquidationPrice": 0.0,
            "markPrice": 0.91,
            "collateral": 20,
            "marginType": "isolated",
            "side": "short",
            "percentage": None,
        },
        {
            # 已平仓
            "symbol": "SOL/BUSD:BUSD",
            "timestamp": None,
            "datetime": None,
            "initialMargin": 0.0,
            "initialMarginPercentage": None,
            "maintenanceMargin": 0.0,
            "maintenanceMarginPercentage": 0.005,
            "entryPrice": 0.0,
            "notional": 0.0,
            "leverage": 5.0,
            "unrealizedPnl": 0.0,
            "contracts": 0.0,
            "contractSize": 1,
            "marginRatio": None,
            "liquidationPrice": 0.0,
            "markPrice": 15.41,
            "collateral": 0.0,
            "marginType": "isolated",
            "side": "short",
            "percentage": None,
        },
    ]
    
    # 模拟交易所方法
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(
            return_value={
                "USDT": {"free": 900, "used": 100, "total": 1000},
            }
        ),
        fetch_positions=MagicMock(return_value=mock_result),
    )

    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    # 验证初始同步结果
    assert len(freqtrade.wallets._wallets) == 1
    assert len(freqtrade.wallets._positions) == 2  # 只包含未平仓的两个持仓

    assert "USDT" in freqtrade.wallets._wallets
    assert "ETH/USDT:USDT" in freqtrade.wallets._positions
    assert freqtrade.wallets._last_wallet_refresh is not None
    assert freqtrade.wallets.get_owned("ETH/USDT:USDT", "ETH") == 1000
    assert freqtrade.wallets.get_owned("SOL/USDT:USDT", "SOL") == 0

    # 移除ETH/USDT:USDT持仓并验证更新结果
    del mock_result[0]
    freqtrade.wallets.update()
    assert len(freqtrade.wallets._positions) == 1
    assert "ETH/USDT:USDT" not in freqtrade.wallets._positions


def test_sync_wallet_dry(mocker, default_conf_usdt, fee):
    """测试回测模式下同步钱包"""
    default_conf_usdt["dry_run"] = True
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    
    # 验证初始状态
    assert len(freqtrade.wallets._wallets) == 1
    assert len(freqtrade.wallets._positions) == 0
    assert freqtrade.wallets.get_total("USDT") == 1000

    # 创建模拟交易
    create_mock_trades_usdt(fee, is_short=None)

    # 更新钱包
    freqtrade.wallets.update()

    # 验证更新后的钱包状态
    assert len(freqtrade.wallets._wallets) == 5
    assert len(freqtrade.wallets._positions) == 0
    bal = freqtrade.wallets.get_all_balances()
    
    # NEO交易尚未成交
    assert bal["NEO"].total == 0
    assert bal["XRP"].total == 10
    assert bal["LTC"].total == 2
    usdt_bal = bal["USDT"]
    assert usdt_bal.free == 922.74
    assert usdt_bal.total == 942.74
    assert usdt_bal.used == 20.0
    # 可用余额 + 已用余额 应等于总余额
    assert usdt_bal.total == usdt_bal.free + usdt_bal.used

    # 验证初始余额计算
    assert (
        freqtrade.wallets.get_starting_balance()
        == default_conf_usdt["dry_run_wallet"] * default_conf_usdt["tradable_balance_ratio"]
    )
    
    # 验证LTC余额计算
    total = freqtrade.wallets.get_total("LTC")
    free = freqtrade.wallets.get_free("LTC")
    used = freqtrade.wallets.get_used("LTC")
    assert used != 0
    assert free + used == total


def test_sync_wallet_futures_dry(mocker, default_conf, fee):
    """测试回测期货模式下同步钱包"""
    default_conf["dry_run"] = True
    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    
    # 验证初始状态
    assert len(freqtrade.wallets._wallets) == 1
    assert len(freqtrade.wallets._positions) == 0

    # 创建模拟交易
    create_mock_trades(fee, is_short=None)

    # 更新钱包
    freqtrade.wallets.update()

    # 验证更新后的状态
    assert len(freqtrade.wallets._wallets) == 1
    assert len(freqtrade.wallets._positions) == 4
    positions = freqtrade.wallets.get_all_positions()
    assert positions["ETH/BTC"].side == "short"
    assert positions["ETC/BTC"].side == "long"
    assert positions["XRP/BTC"].side == "long"
    assert positions["LTC/BTC"].side == "short"

    # 验证初始余额计算
    assert (
        freqtrade.wallets.get_starting_balance()
        == default_conf["dry_run_wallet"] * default_conf["tradable_balance_ratio"]
    )
    
    # 验证BTC余额计算
    total = freqtrade.wallets.get_total("BTC")
    free = freqtrade.wallets.get_free("BTC")
    used = freqtrade.wallets.get_used("BTC")
    assert free + used == total


def test_check_exit_amount(mocker, default_conf, fee):
    """测试检查出场金额的功能"""
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    update_mock = mocker.patch("freqtrade.wallets.Wallets.update")
    total_mock = mocker.patch("freqtrade.wallets.Wallets.get_total", return_value=50.0)

    # 创建模拟交易
    create_mock_trades(fee, is_short=None)
    trade = Trade.session.scalars(select(Trade)).first()
    assert trade.amount == 50.0

    # 余额充足的情况
    assert freqtrade.wallets.check_exit_amount(trade) is True
    assert update_mock.call_count == 0
    assert total_mock.call_count == 1

    # 余额不足的情况
    update_mock.reset_mock()
    # 模拟返回的金额低于交易金额，应触发钱包更新并返回False
    total_mock = mocker.patch("freqtrade.wallets.Wallets.get_total", return_value=40)
    assert freqtrade.wallets.check_exit_amount(trade) is False
    assert update_mock.call_count == 1
    assert total_mock.call_count == 2


def test_check_exit_amount_futures(mocker, default_conf, fee):
    """测试期货模式下检查出场金额的功能"""
    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    total_mock = mocker.patch("freqtrade.wallets.Wallets.get_total", return_value=50)

    # 创建模拟交易
    create_mock_trades(fee, is_short=None)
    trade = Trade.session.scalars(select(Trade)).first()
    trade.trading_mode = "futures"
    assert trade.amount == 50

    # 金额匹配的情况
    assert freqtrade.wallets.check_exit_amount(trade) is True
    assert total_mock.call_count == 0

    # 金额不匹配的情况
    update_mock = mocker.patch("freqtrade.wallets.Wallets.update")
    trade.amount = 150
    # 金额不匹配应触发钱包更新并返回False
    assert freqtrade.wallets.check_exit_amount(trade) is False
    assert total_mock.call_count == 0
    assert update_mock.call_count == 1


@pytest.mark.parametrize(
    "config,wallets",
    [
        (
            {"stake_currency": "USDT", "dry_run_wallet": 1000.0},
            {"USDT": {"currency": "USDT", "free": 1000.0, "used": 0.0, "total": 1000.0}},
        ),
        (
            {"stake_currency": "USDT", "dry_run_wallet": {"USDT": 1000.0, "BTC": 0.1, "ETH": 2.0}},
            {
                "USDT": {"currency": "USDT", "free": 1000.0, "used": 0.0, "total": 1000.0},
                "BTC": {"currency": "BTC", "free": 0.1, "used": 0.0, "total": 0.1},
                "ETH": {"currency": "ETH", "free": 2.0, "used": 0.0, "total": 2.0},
            },
        ),
        (
            {
                "stake_currency": "USDT",
                "margin_mode": "cross",
                "dry_run_wallet": {"USDC": 1000.0, "BTC": 0.1, "ETH": 2.0},
            },
            {
                # 跨币种模式下，USDT钱包应创建并包含其他币种转换后的可用余额
                "USDT": {"currency": "USDT", "free": 4200.0, "used": 0.0, "total": 0.0},
                "USDC": {"currency": "USDC", "free": 1000.0, "used": 0.0, "total": 1000.0},
                "BTC": {"currency": "BTC", "free": 0.1, "used": 0.0, "total": 0.1},
                "ETH": {"currency": "ETH", "free": 2.0, "used": 0.0, "total": 2.0},
            },
        ),
        (
            {
                "stake_currency": "USDT",
                "margin_mode": "cross",
                "dry_run_wallet": {"USDT": 500, "USDC": 1000.0, "BTC": 0.1, "ETH": 2.0},
            },
            {
                # 跨币种模式下，USDT钱包应包含原有余额加上其他币种转换后的可用余额
                "USDT": {"currency": "USDT", "free": 4700.0, "used": 0.0, "total": 500.0},
                "USDC": {"currency": "USDC", "free": 1000.0, "used": 0.0, "total": 1000.0},
                "BTC": {"currency": "BTC", "free": 0.1, "used": 0.0, "total": 0.1},
                "ETH": {"currency": "ETH", "free": 2.0, "used": 0.0, "total": 2.0},
            },
        ),
        (
            # 同上，但非跨币种模式
            {
                "stake_currency": "USDT",
                "dry_run_wallet": {"USDT": 500, "USDC": 1000.0, "BTC": 0.1, "ETH": 2.0},
            },
            {
                # 非跨币种模式下，USDT钱包不包含其他币种转换的可用余额
                "USDT": {"currency": "USDT", "free": 500.0, "used": 0.0, "total": 500.0},
                "USDC": {"currency": "USDC", "free": 1000.0, "used": 0.0, "total": 1000.0},
                "BTC": {"currency": "BTC", "free": 0.1, "used": 0.0, "total": 0.1},
                "ETH": {"currency": "ETH", "free": 2.0, "used": 0.0, "total": 2.0},
            },
        ),
        (
            # 同上，但期货跨币种模式
            {
                "stake_currency": "USDT",
                "margin_mode": "cross",
                "trading_mode": "futures",
                "dry_run_wallet": {"USDT": 500, "USDC": 1000.0, "BTC": 0.1, "ETH": 2.0},
            },
            {
                # 期货跨币种模式下，USDT钱包应包含原有余额加上其他币种转换后的可用余额
                "USDT": {"currency": "USDT", "free": 4700.0, "used": 0.0, "total": 500.0},
                "USDC": {"currency": "USDC", "free": 1000.0, "used": 0.0, "total": 1000.0},
                "BTC": {"currency": "BTC", "free": 0.1, "used": 0.0, "total": 0.1},
                "ETH": {"currency": "ETH", "free": 2.0, "used": 0.0, "total": 2.0},
            },
        ),
    ],
)
def test_dry_run_wallet_initialization(mocker, default_conf_usdt, config, wallets):
    """测试回测模式下钱包初始化"""
    default_conf_usdt.update(config)
    # 模拟交易对价格
    mocker.patch(
        f"{EXMS}.get_tickers",
        return_value={
            "USDC/USDT": {"last": 1.0},
            "BTC/USDT": {"last": 20_000.0},
            "ETH/USDT": {"last": 1100.0},
        },
    )
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    stake_currency = config["stake_currency"]
    
    # 验证每个钱包是否符合预期值
    for currency, expected_wallet in wallets.items():
        wallet = freqtrade.wallets._wallets[currency]
        assert wallet.currency == expected_wallet["currency"]
        assert wallet.free == expected_wallet["free"]
        assert wallet.used == expected_wallet["used"]
        assert wallet.total == expected_wallet["total"]

    # 验证没有创建额外的钱包
    assert len(freqtrade.wallets._wallets) == len(wallets)

    # 创建交易并验证新币种已添加到钱包
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.0)
    mocker.patch(f"{EXMS}.get_rate", return_value=2.22)
    mocker.patch(
        f"{EXMS}.fetch_ticker",
        return_value={
            "bid": 0.20,
            "ask": 0.22,
            "last": 0.22,
        },
    )
    
    # 没有持仓时，抵押品应与可用余额相同
    assert freqtrade.wallets.get_collateral() == freqtrade.wallets.get_free(stake_currency)
    freqtrade.execute_entry("NEO/USDT", 100.0)

    # 更新钱包并验证NEO已包含在内
    freqtrade.wallets.update()
    if default_conf_usdt["trading_mode"] != "futures":
        # 现货模式
        assert "NEO" in freqtrade.wallets._wallets

        # 验证NEO余额 (100 USDT / 0.22)
        assert freqtrade.wallets._wallets["NEO"].total == 45.04504504
        assert freqtrade.wallets._wallets["NEO"].used == 0.0
        assert freqtrade.wallets._wallets["NEO"].free == 45.04504504
        assert freqtrade.wallets.get_collateral() == freqtrade.wallets.get_free(stake_currency)
        
        # 验证USDT钱包已减去交易金额
        assert (
            pytest.approx(freqtrade.wallets._wallets[stake_currency].total)
            == wallets[stake_currency]["total"] - 100.0
        )
        # 验证钱包总数增加了1（新增NEO）
        assert len(freqtrade.wallets._wallets) == len(wallets) + 1
    else:
        # 期货模式
        assert "NEO" not in freqtrade.wallets._wallets
        assert freqtrade.wallets._positions["NEO/USDT"].position == 45.04504504
        assert pytest.approx(freqtrade.wallets._positions["NEO/USDT"].collateral) == 100

        # 验证USDT钱包的可用余额已减去交易金额
        assert (
            pytest.approx(freqtrade.wallets.get_collateral())
            == freqtrade.wallets.get_free(stake_currency) + 100
        )
        assert (
            pytest.approx(freqtrade.wallets._wallets[stake_currency].free)
            == wallets[stake_currency]["free"] - 100.0
        )