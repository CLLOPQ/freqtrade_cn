# pragma pylint: disable=missing-docstring, C0103
from datetime import datetime, timedelta, timezone
from types import FunctionType

import pytest
from sqlalchemy import select

from freqtrade.constants import CUSTOM_TAG_MAX_LENGTH, DATETIME_PRINT_FORMAT
from freqtrade.enums import TradingMode
from freqtrade.exceptions import DependencyException
from freqtrade.exchange.exchange_utils import TICK_SIZE
from freqtrade.persistence import LocalTrade, Order, Trade, init_db
from freqtrade.util import dt_now
from tests.conftest import (
    create_mock_trades,
    create_mock_trades_usdt,
    create_mock_trades_with_leverage,
    log_has,
    log_has_re,
)


spot, margin, futures = TradingMode.SPOT, TradingMode.MARGIN, TradingMode.FUTURES


@pytest.mark.parametrize("is_short", [False, True])
@pytest.mark.usefixtures("init_persistence")
def test_enter_exit_side(fee, is_short):
    """测试入场和出场方向"""
    entry_side, exit_side = ("sell", "buy") if is_short else ("buy", "sell")
    trade = Trade(
        id=2,
        pair="ADA/USDT",
        stake_amount=0.001,
        open_rate=0.01,
        amount=5,
        is_open=True,
        open_date=dt_now(),
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange="binance",
        is_short=is_short,
        leverage=2.0,
        trading_mode=margin,
    )
    assert trade.entry_side == entry_side
    assert trade.exit_side == exit_side
    assert trade.trade_direction == "short" if is_short else "long"


@pytest.mark.usefixtures("init_persistence")
def test_set_stop_loss_liquidation(fee):
    """测试设置止损和清算价格"""
    trade = Trade(
        id=2,
        pair="ADA/USDT",
        stake_amount=60.0,
        open_rate=2.0,
        amount=30.0,
        is_open=True,
        open_date=dt_now(),
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange="binance",
        is_short=False,
        leverage=2.0,
        trading_mode=margin,
    )
    trade.set_liquidation_price(0.09)
    assert trade.liquidation_price == 0.09
    assert trade.stop_loss is None
    assert trade.initial_stop_loss is None

    trade.adjust_stop_loss(2.0, 0.2, True)
    assert trade.liquidation_price == 0.09
    assert trade.stop_loss == 1.8
    assert trade.initial_stop_loss == 1.8

    trade.set_liquidation_price(0.08)
    assert trade.liquidation_price == 0.08
    assert trade.stop_loss == 1.8
    assert trade.initial_stop_loss == 1.8

    trade.set_liquidation_price(0.11)
    trade.adjust_stop_loss(2.0, 0.2)
    assert trade.liquidation_price == 0.11
    # 止损不会因清算价格而改变
    assert trade.stop_loss == 1.8
    assert trade.stop_loss_pct == -0.2
    assert trade.initial_stop_loss == 1.8

    # 较低的止损不会移动止损
    trade.adjust_stop_loss(1.8, 0.2)
    assert trade.liquidation_price == 0.11
    assert trade.stop_loss == 1.8
    assert trade.stop_loss_pct == -0.2
    assert trade.initial_stop_loss == 1.8

    # 使用"allow_refresh"的较低止损会移动止损
    trade.adjust_stop_loss(1.8, 0.22, allow_refresh=True)
    assert trade.liquidation_price == 0.11
    assert trade.stop_loss == 1.602
    assert trade.stop_loss_pct == -0.22
    assert trade.initial_stop_loss == 1.8

    # 较高的止损会移动止损
    trade.adjust_stop_loss(2.1, 0.1)
    assert trade.liquidation_price == 0.11
    assert pytest.approx(trade.stop_loss) == 1.994999
    assert trade.stop_loss_pct == -0.1
    assert trade.initial_stop_loss == 1.8
    assert trade.stoploss_or_liquidation == trade.stop_loss

    trade.stop_loss = None
    trade.liquidation_price = None
    trade.initial_stop_loss = None
    trade.initial_stop_loss_pct = None

    trade.adjust_stop_loss(2.0, 0.1, True)
    assert trade.liquidation_price is None
    assert trade.stop_loss == 1.9
    assert trade.initial_stop_loss == 1.9
    assert trade.stoploss_or_liquidation == 1.9

    trade.is_short = True
    trade.recalc_open_trade_value()
    trade.stop_loss = None
    trade.initial_stop_loss = None
    trade.initial_stop_loss_pct = None

    trade.set_liquidation_price(3.09)
    assert trade.liquidation_price == 3.09
    assert trade.stop_loss is None
    assert trade.initial_stop_loss is None

    trade.adjust_stop_loss(2.0, 0.2)
    assert trade.liquidation_price == 3.09
    assert trade.stop_loss == 2.2
    assert trade.initial_stop_loss == 2.2
    assert trade.stoploss_or_liquidation == 2.2

    trade.set_liquidation_price(3.1)
    assert trade.liquidation_price == 3.1
    assert trade.stop_loss == 2.2
    assert trade.initial_stop_loss == 2.2
    assert trade.stoploss_or_liquidation == 2.2

    trade.set_liquidation_price(3.8)
    assert trade.liquidation_price == 3.8
    # 止损不会因清算价格而改变
    assert trade.stop_loss == 2.2
    assert trade.stop_loss_pct == -0.2
    assert trade.initial_stop_loss == 2.2

    # 止损不会将止损移得更高
    trade.adjust_stop_loss(2.0, 0.3)
    assert trade.liquidation_price == 3.8
    assert trade.stop_loss == 2.2
    assert trade.stop_loss_pct == -0.2
    assert trade.initial_stop_loss == 2.2

    # 使用"allow_refresh"止损会将止损移得更高
    trade.adjust_stop_loss(2.0, 0.3, allow_refresh=True)
    assert trade.liquidation_price == 3.8
    assert trade.stop_loss == 2.3
    assert trade.stop_loss_pct == -0.3
    assert trade.initial_stop_loss == 2.2

    # 止损确实会降低
    trade.set_liquidation_price(1.5)
    trade.adjust_stop_loss(1.8, 0.1)
    assert trade.liquidation_price == 1.5
    assert pytest.approx(trade.stop_loss) == 1.89
    assert trade.stop_loss_pct == -0.1
    assert trade.initial_stop_loss == 2.2
    assert trade.stoploss_or_liquidation == 1.5


@pytest.mark.parametrize(
    "exchange,is_short,lev,minutes,rate,interest,trading_mode",
    [
        ("binance", False, 3, 10, 0.0005, round(0.0008333333333333334, 8), margin),
        ("binance", True, 3, 10, 0.0005, 0.000625, margin),
        ("binance", False, 3, 295, 0.0005, round(0.004166666666666667, 8), margin),
        ("binance", True, 3, 295, 0.0005, round(0.0031249999999999997, 8), margin),
        ("binance", False, 3, 295, 0.00025, round(0.0020833333333333333, 8), margin),
        ("binance", True, 3, 295, 0.00025, round(0.0015624999999999999, 8), margin),
        ("binance", False, 5, 295, 0.0005, 0.005, margin),
        ("binance", True, 5, 295, 0.0005, round(0.0031249999999999997, 8), margin),
        ("binance", False, 1, 295, 0.0005, 0.0, spot),
        ("binance", True, 1, 295, 0.0005, 0.003125, margin),
        ("binance", False, 3, 10, 0.0005, 0.0, futures),
        ("binance", True, 3, 295, 0.0005, 0.0, futures),
        ("binance", False, 5, 295, 0.0005, 0.0, futures),
        ("binance", True, 5, 295, 0.0005, 0.0, futures),
        ("binance", False, 1, 295, 0.0005, 0.0, futures),
        ("binance", True, 1, 295, 0.0005, 0.0, futures),
        ("kraken", False, 3, 10, 0.0005, 0.040, margin),
        ("kraken", True, 3, 10, 0.0005, 0.030, margin),
        ("kraken", False, 3, 295, 0.0005, 0.06, margin),
        ("kraken", True, 3, 295, 0.0005, 0.045, margin),
        ("kraken", False, 3, 295, 0.00025, 0.03, margin),
        ("kraken", True, 3, 295, 0.00025, 0.0225, margin),
        ("kraken", False, 5, 295, 0.0005, round(0.07200000000000001, 8), margin),
        ("kraken", True, 5, 295, 0.0005, 0.045, margin),
        ("kraken", False, 1, 295, 0.0005, 0.0, spot),
        ("kraken", True, 1, 295, 0.0005, 0.045, margin),
    ],
)
@pytest.mark.usefixtures("init_persistence")
def test_interest(fee, exchange, is_short, lev, minutes, rate, interest, trading_mode):
    """
    测试利息计算
    在Binance/Kraken上进行10分钟、5小时的限价交易，3倍、5倍杠杆
    费用: 0.25% 报价
    利率: 每4小时0.05%
    开盘价: 2.00 报价
    收盘价: 2.20 报价
    数量: = 30.0 加密货币
    投入金额
        3x, -3x: 20.0  报价
        5x, -5x: 12.0  报价
    借入
      10分钟
         3x: 40 报价
        -3x: 30 加密货币
         5x: 48 报价
        -5x: 30 加密货币
         1x: 0
        -1x: 30 加密货币
    小时数: 1/6 (10分钟)
    时间周期:
        10分钟
            kraken: (1 + 1) 4小时周期 = 2个4小时周期
            binance: 1/24个24小时周期
        4.95小时
            kraken: ceil(1 + 4.95/4) 4小时周期 = 3个4小时周期
            binance: ceil(4.95)/24个24小时周期 = 5/24个24小时周期
    利息: 借入 * 利率 * 时间周期
      10分钟
        binance     3x: 40 * 0.0005 * 1/24 = 0.0008333333333333334 报价
        kraken      3x: 40 * 0.0005 * 2    = 0.040 报价
        binace     -3x: 30 * 0.0005 * 1/24 = 0.000625 加密货币
        kraken     -3x: 30 * 0.0005 * 2    = 0.030 加密货币
      5小时
        binance     3x: 40 * 0.0005 * 5/24 = 0.004166666666666667 报价
        kraken      3x: 40 * 0.0005 * 3    = 0.06 报价
        binace     -3x: 30 * 0.0005 * 5/24 = 0.0031249999999999997 加密货币
        kraken     -3x: 30 * 0.0005 * 3    = 0.045 加密货币
      0.00025 利息
        binance     3x: 40 * 0.00025 * 5/24 = 0.0020833333333333333 报价
        kraken      3x: 40 * 0.00025 * 3    = 0.03 报价
        binace     -3x: 30 * 0.00025 * 5/24 = 0.0015624999999999999 加密货币
        kraken     -3x: 30 * 0.00025 * 3    = 0.0225 加密货币
      5倍杠杆, 0.0005 利息, 5小时
        binance     5x: 48 * 0.0005 * 5/24 = 0.005 报价
        kraken      5x: 48 * 0.0005 * 3    = 0.07200000000000001 报价
        binace     -5x: 30 * 0.0005 * 5/24 = 0.0031249999999999997 加密货币
        kraken     -5x: 30 * 0.0005 * 3    = 0.045 加密货币
      1倍杠杆, 0.0005 利息, 5小时
        binance,kraken 1x: 0.0 报价
        binace        -1x: 30 * 0.0005 * 5/24 = 0.003125 加密货币
        kraken        -1x: 30 * 0.0005 * 3    = 0.045 加密货币
    """

    trade = Trade(
        pair="ADA/USDT",
        stake_amount=20.0,
        amount=30.0,
        open_rate=2.0,
        open_date=datetime.now(timezone.utc) - timedelta(minutes=minutes),
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange=exchange,
        leverage=lev,
        interest_rate=rate,
        is_short=is_short,
        trading_mode=trading_mode,
    )

    assert round(float(trade.calculate_interest()), 8) == interest


@pytest.mark.parametrize(
    "is_short,lev,borrowed,trading_mode",
    [
        (False, 1.0, 0.0, spot),
        (True, 1.0, 30.0, margin),
        (False, 3.0, 40.0, margin),
        (True, 3.0, 30.0, margin),
    ],
)
@pytest.mark.usefixtures("init_persistence")
def test_borrowed(fee, is_short, lev, borrowed, trading_mode):
    """
    测试借入金额计算
    在Binance/Kraken上进行10分钟限价交易，1倍、3倍杠杆
    费用: 0.25% 报价
    利率: 每4小时0.05%
    开盘价: 2.00 报价
    收盘价: 2.20 报价
    数量: = 30.0 加密货币
    投入金额
        1x,-1x: 60.0  报价
        3x,-3x: 20.0  报价
    借入
         1x:  0 报价
         3x: 40 报价
        -1x: 30 加密货币
        -3x: 30 加密货币
    小时数: 1/6 (10分钟)
    时间周期:
        kraken: (1 + 1) 4小时周期 = 2个4小时周期
        binance: 1/24个24小时周期
    利息: 借入 * 利率 * 时间周期
        1x            :  /
        binance     3x: 40 * 0.0005 * 1/24 = 0.0008333333333333334 报价
        kraken      3x: 40 * 0.0005 * 2 = 0.040 报价
        binace -1x,-3x: 30 * 0.0005 * 1/24 = 0.000625 加密货币
        kraken -1x,-3x: 30 * 0.0005 * 2 = 0.030 加密货币
    开仓价值: (数量 * 开盘价) ± (数量 * 开盘价 * 费用)
         1x, 3x: 30 * 2 + 30 * 2 * 0.0025 = 60.15 报价
        -1x,-3x: 30 * 2 - 30 * 2 * 0.0025 = 59.850 报价
    平仓数量:
        1x, 3x         : 数量
        -1x, -3x       : 数量 + 利息
        binance -1x,-3x: 30 + 0.000625 = 30.000625 加密货币
        kraken  -1x,-3x: 30 + 0.03 = 30.03 加密货币
    平仓价值:
         1x, 3x: (平仓数量 * 收盘价) - (平仓数量 * 收盘价 * 费用) - 利息
        -1x,-3x: (平仓数量 * 收盘价) + (平仓数量 * 收盘价 * 费用)
        binance,kraken 1x: (30.00 * 2.20) - (30.00 * 2.20 * 0.0025)         = 65.835
        binance        3x: (30.00 * 2.20) - (30.00 * 2.20 * 0.0025) - 0.00083333 = 65.83416667
        kraken         3x: (30.00 * 2.20) - (30.00 * 2.20 * 0.0025) - 0.040 = 65.795
        binance   -1x,-3x: (30.000625 * 2.20) + (30.000625 * 2.20 * 0.0025) = 66.16637843750001
        kraken    -1x,-3x: (30.03 * 2.20) + (30.03 * 2.20 * 0.0025)         = 66.231165
    总利润:
        1x, 3x : 平仓价值 - 开仓价值
        -1x,-3x: 开仓价值  - 平仓价值
        binance,kraken 1x: 65.835 - 60.15             = 5.685
        binance        3x: 65.83416667 - 60.15        = 5.684166670000003
        kraken         3x: 65.795 - 60.15             = 5.645
        binance   -1x,-3x: 59.850 - 66.16637843750001 = -6.316378437500013
        kraken    -1x,-3x: 59.850 - 66.231165          = -6.381165
    总利润率:
        1x, 3x : ((平仓价值/开仓价值) - 1) * 杠杆
        -1x,-3x: (1 - (平仓价值/开仓价值)) * 杠杆
        binance  1x: ((65.835 / 60.15) - 1)  * 1 = 0.0945137157107232
        binance  3x: ((65.83416667 / 60.15) - 1)  * 3 = 0.2834995845386534
        kraken   1x: ((65.835 / 60.15) - 1)  * 1 = 0.0945137157107232
        kraken   3x: ((65.795 / 60.15) - 1)  * 3 = 0.2815461346633419
        binance -1x: (1-(66.1663784375 / 59.85)) * 1 = -0.1055368159983292
        binance -3x: (1-(66.1663784375 / 59.85)) * 3 = -0.3166104479949876
        kraken  -1x: (1-(66.2311650 / 59.85)) * 1    = -0.106619298245614
        kraken  -3x: (1-(66.2311650 / 59.85)) * 3    = -0.319857894736842
    """

    trade = Trade(
        id=2,
        pair="ADA/USDT",
        stake_amount=60.0,
        open_rate=2.0,
        amount=30.0,
        is_open=True,
        open_date=dt_now(),
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange="binance",
        is_short=is_short,
        leverage=lev,
        trading_mode=trading_mode,
    )
    assert trade.borrowed == borrowed


@pytest.mark.parametrize(
    "is_short,open_rate,close_rate,lev,profit,trading_mode",
    [
        (False, 2.0, 2.2, 1.0, 0.09451372, spot),
        (True, 2.2, 2.0, 3.0, 0.25894253, margin),
    ],
)
@pytest.mark.usefixtures("init_persistence")
def test_update_limit_order(
    fee,
    caplog,
    limit_buy_order_usdt,
    limit_sell_order_usdt,
    time_machine,
    is_short,
    open_rate,
    close_rate,
    lev,
    profit,
    trading_mode,
):
    """
    测试更新限价订单
    
    在Binance/Kraken上进行10分钟限价交易，1倍、3倍杠杆
    费用: 0.25% 报价
    利率: 每4小时0.05%
    开盘价: 2.00 报价
    收盘价: 2.20 报价
    数量: = 30.0 加密货币
    投入金额
        1x,-1x: 60.0  报价
        3x,-3x: 20.0  报价
    借入
         1x:  0 报价
         3x: 40 报价
        -1x: 30 加密货币
        -3x: 30 加密货币
    小时数: 1/6 (10分钟)
    时间周期:
        kraken: (1 + 1) 4小时周期 = 2个4小时周期
        binance: 1/24个24小时周期
    利息: 借入 * 利率 * 时间周期
        1x            :  /
        binance     3x: 40 * 0.0005 * 1/24 = 0.0008333333333333334 报价
        kraken      3x: 40 * 0.0005 * 2 = 0.040 报价
        binace -1x,-3x: 30 * 0.0005 * 1/24 = 0.000625 加密货币
        kraken -1x,-3x: 30 * 0.0005 * 2 = 0.030 加密货币
    开仓价值: (数量 * 开盘价) ± (数量 * 开盘价 * 费用)
         1x, 3x: 30 * 2 + 30 * 2 * 0.0025 = 60.15 报价
        -1x,-3x: 30 * 2 - 30 * 2 * 0.0025 = 59.850 报价
    平仓数量:
        1x, 3x         : 数量
        -1x, -3x       : 数量 + 利息
        binance -1x,-3x: 30 + 0.000625 = 30.000625 加密货币
        kraken  -1x,-3x: 30 + 0.03 = 30.03 加密货币
    平仓价值:
         1x, 3x: (平仓数量 * 收盘价) - (平仓数量 * 收盘价 * 费用) - 利息
        -1x,-3x: (平仓数量 * 收盘价) + (平仓数量 * 收盘价 * 费用)
        binance,kraken 1x: (30.00 * 2.20) - (30.00 * 2.20 * 0.0025)         = 65.835
        binance        3x: (30.00 * 2.20) - (30.00 * 2.20 * 0.0025) - 0.00083333 = 65.83416667
        kraken         3x: (30.00 * 2.20) - (30.00 * 2.20 * 0.0025) - 0.040 = 65.795
        binance   -1x,-3x: (30.000625 * 2.20) + (30.000625 * 2.20 * 0.0025) = 66.16637843750001
        kraken    -1x,-3x: (30.03 * 2.20) + (30.03 * 2.20 * 0.0025)         = 66.231165
    总利润:
        1x, 3x : 平仓价值 - 开仓价值
        -1x,-3x: 开仓价值  - 平仓价值
        binance,kraken 1x: 65.835 - 60.15             = 5.685
        binance        3x: 65.83416667 - 60.15        = 5.684166670000003
        kraken         3x: 65.795 - 60.15             = 5.645
        binance   -1x,-3x: 59.850 - 66.16637843750001 = -6.316378437500013
        kraken    -1x,-3x: 59.850 - 66.231165          = -6.381165
    总利润率:
        1x, 3x : ((平仓价值/开仓价值) - 1) * 杠杆
        -1x,-3x: (1 - (平仓价值/开仓价值)) * 杠杆
        binance  1x: ((65.835 / 60.15) - 1)  * 1 = 0.0945137157107232
        binance  3x: ((65.83416667 / 60.15) - 1)  * 3 = 0.2834995845386534
        kraken   1x: ((65.835 / 60.15) - 1)  * 1 = 0.0945137157107232
        kraken   3x: ((65.795 / 60.15) - 1)  * 3 = 0.2815461346633419
        binance -1x: (1-(66.1663784375 / 59.85)) * 1 = -0.1055368159983292
        binance -3x: (1-(66.1663784375 / 59.85)) * 3 = -0.3166104479949876
        kraken  -1x: (1-(66.2311650 / 59.85)) * 1    = -0.106619298245614
        kraken  -3x: (1-(66.2311650 / 59.85)) * 3    = -0.319857894736842
    开盘价: 2.2, 收盘价: 2.0, -3x, binance, 做空
        开仓价值: 30 * 2.2 - 30 * 2.2 * 0.0025 = 65.835 报价
        平仓数量: 30 + 0.000625 = 30.000625 加密货币
        平仓价值: (30.000625 * 2.0) + (30.000625 * 2.0 * 0.0025) = 60.151253125
        总利润: 65.835 - 60.151253125 = 5.683746874999997
        总利润率: (1-(60.151253125/65.835)) * 3 = 0.2589996297562085

    """
    time_machine.move_to("2022-03-31 20:45:00 +00:00")

    enter_order = limit_sell_order_usdt if is_short else limit_buy_order_usdt
    exit_order = limit_buy_order_usdt if is_short else limit_sell_order_usdt
    entry_side, exit_side = ("sell", "buy") if is_short else ("buy", "sell")

    trade = Trade(
        id=2,
        pair="ADA/USDT",
        stake_amount=60.0,
        open_rate=open_rate,
        amount=30.0,
        is_open=True,
        open_date=dt_now(),
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange="binance",
        is_short=is_short,
        interest_rate=0.0005,
        leverage=lev,
        trading_mode=trading_mode,
    )
    assert not trade.has_open_orders
    assert trade.close_profit is None
    assert trade.close_date is None

    oobj = Order.parse_from_ccxt_object(enter_order, "ADA/USDT", entry_side)
    trade.orders.append(oobj)
    trade.update_trade(oobj)
    assert not trade.has_open_orders
    assert trade.open_rate == open_rate
    assert trade.close_profit is None
    assert trade.close_date is None
    assert log_has_re(
        f"LIMIT_{entry_side.upper()} 已为交易"
        r"\(id=2, pair=ADA/USDT, amount=30.00000000, "
        f"is_short={is_short}, leverage={lev}, open_rate={open_rate}0000000, "
        r"open_since=.*\)完成.",
        caplog,
    )

    caplog.clear()
    time_machine.move_to("2022-03-31 21:45:05 +00:00")
    oobj = Order.parse_from_ccxt_object(exit_order, "ADA/USDT", exit_side)
    trade.orders.append(oobj)
    trade.update_trade(oobj)

    assert not trade.has_open_orders
    assert trade.close_rate == close_rate
    assert pytest.approx(trade.close_profit) == profit
    assert trade.close_date is not None
    assert log_has_re(
        f"LIMIT_{exit_side.upper()} 已为交易"
        r"\(id=2, pair=ADA/USDT, amount=30.00000000, "
        f"is_short={is_short}, leverage={lev}, open_rate={open_rate}0000000, "
        r"open_since=.*\)完成.",
        caplog,
    )
    caplog.clear()


@pytest.mark.usefixtures("init_persistence")
def test_update_market_order(market_buy_order_usdt, market_sell_order_usdt, fee, caplog):
    """测试更新市价订单"""
    trade = Trade(
        id=1,
        pair="ADA/USDT",
        stake_amount=60.0,
        open_rate=2.0,
        amount=30.0,
        is_open=True,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_date=dt_now(),
        exchange="binance",
        trading_mode=margin,
        leverage=1.0,
    )

    oobj = Order.parse_from_ccxt_object(market_buy_order_usdt, "ADA/USDT", "buy")
    trade.orders.append(oobj)
    trade.update_trade(oobj)
    assert not trade.has_open_orders
    assert trade.open_rate == 2.0
    assert trade.close_profit is None
    assert trade.close_date is None
    assert log_has_re(
        r"MARKET_BUY 已为交易\(id=1, "
        r"pair=ADA/USDT, amount=30.00000000, is_short=False, leverage=1.0, "
        r"open_rate=2.00000000, open_since=.*\)完成.",
        caplog,
    )

    caplog.clear()
    trade.is_open = True
    oobj = Order.parse_from_ccxt_object(market_sell_order_usdt, "ADA/USDT", "sell")
    trade.orders.append(oobj)
    trade.update_trade(oobj)
    assert not trade.has_open_orders
    assert trade.close_rate == 2.2
    assert pytest.approx(trade.close_profit) == 0.094513715710723
    assert trade.close_date is not None
    assert log_has_re(
        r"MARKET_SELL 已为交易\(id=1, "
        r"pair=ADA/USDT, amount=30.00000000, is_short=False, leverage=1.0, "
        r"open_rate=2.00000000, open_since=.*\)完成.",
        caplog,
    )


@pytest.mark.parametrize(
    "exchange,is_short,lev,open_value,close_value,profit,profit_ratio,trading_mode,funding_fees",
    [
        ("binance", False, 1, 60.15, 65.835, 5.685, 0.09451371, spot, 0.0),
        ("binance", True, 1, 65.835, 60.151253125, 5.68374687, 0.08633321, margin, 0.0),
        ("binance", False, 3, 60.15, 65.83416667, 5.68416667, 0.28349958, margin, 0.0),
        ("binance", True, 3, 65.835, 60.151253125, 5.68374687, 0.25899963, margin, 0.0),
        ("kraken", False, 1, 60.15, 65.835, 5.685, 0.09451371, spot, 0.0),
        ("kraken", True, 1, 65.835, 60.21015, 5.62485, 0.0854386, margin, 0.0),
        ("kraken", False, 3, 60.15, 65.795, 5.645, 0.28154613, margin, 0.0),
        ("kraken", True, 3, 65.835, 60.21015, 5.62485, 0.25631579, margin, 0.0),
        ("binance", False, 1, 60.15, 65.835, 5.685, 0.09451371, futures, 0.0),
        ("binance", False, 1, 60.15, 66.835, 6.685, 0.11113881, futures, 1.0),
        ("binance", True, 1, 65.835, 60.15, 5.685, 0.08635224, futures, 0.0),
        ("binance", True, 1, 65.835, 61.15, 4.685, 0.07116276, futures, -1.0),
        ("binance", True, 3, 65.835, 59.15, 6.685, 0.3046252, futures, 1.0),
        ("binance", False, 3, 60.15, 64.835, 4.685, 0.23366583, futures, -1.0),
    ],
)
@pytest.mark.usefixtures("init_persistence")
def test_calc_open_close_trade_price(
    limit_order,
    fee,
    exchange,
    is_short,
    lev,
    open_value,
    close_value,
    profit,
    profit_ratio,
    trading_mode,
    funding_fees,
):
    """测试计算开仓和平仓交易价格"""
    trade: Trade = Trade(
        pair="ADA/USDT",
        stake_amount=60.0,
        open_rate=2.0,
        amount=30.0,
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=10),
        interest_rate=0.0005,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange=exchange,
        is_short=is_short,
        leverage=lev,
        trading_mode=trading_mode,
    )
    entry_order = limit_order[trade.entry_side]
    exit_order = limit_order[trade.exit_side]

    oobj = Order.parse_from_ccxt_object(entry_order, "ADA/USDT", trade.entry_side)
    oobj._trade_live = trade
    oobj.update_from_ccxt_object(entry_order)
    trade.update_trade(oobj)

    trade.funding_fee_running = funding_fees

    oobj = Order.parse_from_ccxt_object(exit_order, "ADA/USDT", trade.exit_side)
    oobj._trade_live = trade
    oobj.update_from_ccxt_object(exit_order)
    trade.update_trade(oobj)

    assert trade.is_open is False
    # 资金费用从funding_fee_running转移到funding_fees
    assert trade.funding_fees == funding_fees
    assert trade.orders[-1].funding_fee == funding_fees

    assert pytest.approx(trade._calc_open_trade_value(trade.amount, trade.open_rate)) == open_value
    assert pytest.approx(trade.calc_close_trade_value(trade.close_rate)) == close_value
    assert pytest.approx(trade.close_profit_abs) == profit
    assert pytest.approx(trade.close_profit) == profit_ratio


@pytest.mark.usefixtures("init_persistence")
def test_trade_close(fee, time_machine):
    """测试交易关闭"""
    time_machine.move_to("2022-09-01 05:00:00 +00:00", tick=False)

    trade = Trade(
        pair="ADA/USDT",
        stake_amount=60.0,
        open_rate=2.0,
        amount=30.0,
        is_open=True,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_date=dt_now() - timedelta(minutes=10),
        interest_rate=0.0005,
        exchange="binance",
        trading_mode=margin,
        leverage=1.0,
    )
    trade.orders.append(
        Order(
            ft_order_side=trade.entry_side,
            order_id=f"{trade.pair}-{trade.entry_side}-{trade.open_date}",
            ft_is_open=False,
            ft_pair=trade.pair,
            amount=trade.amount,
            filled=trade.amount,
            remaining=0,
            price=trade.open_rate,
            average=trade.open_rate,
            status="closed",
            order_type="limit",
            side=trade.entry_side,
            order_filled_date=trade.open_date,
        )
    )
    trade.orders.append(
        Order(
            ft_order_side=trade.exit_side,
            order_id=f"{trade.pair}-{trade.exit_side}-{trade.open_date}",
            ft_is_open=False,
            ft_pair=trade.pair,
            amount=trade.amount,
            filled=trade.amount,
            remaining=0,
            price=2.2,
            average=2.2,
            status="closed",
            order_type="limit",
            side=trade.exit_side,
            order_filled_date=dt_now(),
        )
    )
    assert trade.close_profit is None
    assert trade.close_date is None
    assert trade.is_open is True
    trade.close(2.2)
    assert trade.is_open is False
    assert pytest.approx(trade.close_profit) == 0.094513715
    assert trade.close_date is not None
    assert trade.close_date_utc == dt_now()

    new_date = dt_now() + timedelta(minutes=5)
    assert trade.close_date_utc != new_date
    # 如果交易已经关闭，close不应该更新close_date
    assert trade.is_open is False
    trade.close_date = new_date
    trade.close(2.2)
    assert trade.close_date_utc == new_date


@pytest.mark.usefixtures("init_persistence")
def test_calc_close_trade_price_exception(limit_buy_order_usdt, fee):
    """测试计算平仓交易价格异常"""
    trade = Trade(
        pair="ADA/USDT",
        stake_amount=60.0,
        open_rate=2.0,
        amount=30.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange="binance",
        trading_mode=margin,
        leverage=1.0,
    )

    oobj = Order.parse_from_ccxt_object(limit_buy_order_usdt, "ADA/USDT", "buy")
    trade.update_trade(oobj)
    assert trade.calc_close_trade_value(trade.close_rate) == 0.0


@pytest.mark.usefixtures("init_persistence")
def test_update_open_order(limit_buy_order_usdt):
    """测试更新开放订单"""
    trade = Trade(
        pair="ADA/USDT",
        stake_amount=60.0,
        open_rate=2.0,
        amount=30.0,
        fee_open=0.1,
        fee_close=0.1,
        exchange="binance",
        trading_mode=margin,
    )

    assert not trade.has_open_orders
    assert trade.close_profit is None
    assert trade.close_date is None

    limit_buy_order_usdt["status"] = "open"
    oobj = Order.parse_from_ccxt_object(limit_buy_order_usdt, "ADA/USDT", "buy")
    trade.update_trade(oobj)

    assert not trade.has_open_orders
    assert trade.close_profit is None
    assert trade.close_date is None


@pytest.mark.usefixtures("init_persistence")
def test_update_invalid_order(limit_buy_order_usdt):
    """测试更新无效订单"""
    trade = Trade(
        pair="ADA/USDT",
        stake_amount=60.0,
        amount=30.0,
        open_rate=2.0,
        fee_open=0.1,
        fee_close=0.1,
        exchange="binance",
        trading_mode=margin,
    )
    limit_buy_order_usdt["type"] = "invalid"
    oobj = Order.parse_from_ccxt_object(limit_buy_order_usdt, "ADA/USDT", "meep")
    with pytest.raises(ValueError, match=r"未知订单类型"):
        trade.update_trade(oobj)


@pytest.mark.parametrize("exchange", ["binance", "kraken"])
@pytest.mark.parametrize("trading_mode", [spot, margin, futures])
@pytest.mark.parametrize("lev", [1, 3])
@pytest.mark.parametrize(
    "is_short,fee_rate,result",
    [
        (False, 0.003, 60.18),
        (False, 0.0025, 60.15),
        (False, 0.003, 60.18),
        (False, 0.0025, 60.15),
        (True, 0.003, 59.82),
        (True, 0.0025, 59.85),
        (True, 0.003, 59.82),
        (True, 0.0025, 59.85),
    ],
)
@pytest.mark.usefixtures("init_persistence")
def test_calc_open_trade_value(
    limit_buy_order_usdt, exchange, lev, is_short, fee_rate, result, trading_mode
):
    """
    测试计算开仓交易价值
    在Binance/Kraken上进行10分钟限价交易，1倍、3倍杠杆
    费用: 0.25 %, 0.3% 报价
    开盘价: 2.00 报价
    数量: = 30.0 加密货币
    投入金额
        1x, -1x: 60.0  报价
        3x, -3x: 20.0  报价
    开仓价值: (数量 * 开盘价) ± (数量 * 开盘价 * 费用)
    0.25% 费用
         1x, 3x: 30 * 2 + 30 * 2 * 0.0025 = 60.15 报价
        -1x,-3x: 30 * 2 - 30 * 2 * 0.0025 = 59.85 报价
    0.3% 费用
         1x, 3x: 30 * 2 + 30 * 2 * 0.003  = 60.18 报价
        -1x,-3x: 30 * 2 - 30 * 2 * 0.003  = 59.82 报价
    """
    trade = Trade(
        pair="ADA/USDT",
        stake_amount=60.0,
        amount=30.0,
        open_rate=2.0,
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=10),
        fee_open=fee_rate,
        fee_close=fee_rate,
        exchange=exchange,
        leverage=lev,
        is_short=is_short,
        trading_mode=trading_mode,
    )
    oobj = Order.parse_from_ccxt_object(
        limit_buy_order_usdt, "ADA/USDT", "sell" if is_short else "buy"
    )
    trade.update_trade(oobj)  # 买入 @ 2.0

    # 获取带有标准费率的开盘价
    assert trade._calc_open_trade_value(trade.amount, trade.open_rate) == result


@pytest.mark.parametrize(
    "exchange,is_short,lev,open_rate,close_rate,fee_rate,result,trading_mode,funding_fees",
    [
        ("binance", False, 1, 2.0, 2.5, 0.0025, 74.8125, spot, 0),
        ("binance", False, 1, 2.0, 2.5, 0.003, 74.775, spot, 0),
        ("binance", False, 1, 2.0, 2.2, 0.005, 65.67, margin, 0),
        ("binance", False, 3, 2.0, 2.5, 0.0025, 74.81166667, margin, 0),
        ("binance", False, 3, 2.0, 2.5, 0.003, 74.77416667, margin, 0),
        ("binance", True, 3, 2.2, 2.5, 0.0025, 75.18906641, margin, 0),
        ("binance", True, 3, 2.2, 2.5, 0.003, 75.22656719, margin, 0),
        ("binance", True, 1, 2.2, 2.5, 0.0025, 75.18906641, margin, 0),
        ("binance", True, 1, 2.2, 2.5, 0.003, 75.22656719, margin, 0),
        # Kraken
        ("kraken", False, 3, 2.0, 2.5, 0.0025, 74.7725, margin, 0),
        ("kraken", False, 3, 2.0, 2.5, 0.003, 74.735, margin, 0),
        ("kraken", True, 3, 2.2, 2.5, 0.0025, 75.2626875, margin, 0),
        ("kraken", True, 3, 2.2, 2.5, 0.003, 75.300225, margin, 0),
        ("kraken", True, 1, 2.2, 2.5, 0.0025, 75.2626875, margin, 0),
        ("kraken", True, 1, 2.2, 2.5, 0.003, 75.300225, margin, 0),
        ("binance", False, 1, 2.0, 2.5, 0.0025, 75.8125, futures, 1),
        ("binance", False, 3, 2.0, 2.5, 0.0025, 73.8125, futures, -1),
        ("binance", True, 3, 2.0, 2.5, 0.0025, 74.1875, futures, 1),
        ("binance", True, 1, 2.0, 2.5, 0.0025, 76.1875, futures, -1),
    ],
)
@pytest.mark.usefixtures("init_persistence")
def test_calc_close_trade_price(
    open_rate, exchange, is_short, lev, close_rate, fee_rate, result, trading_mode, funding_fees
):
    """测试计算平仓交易价格"""
    trade = Trade(
        pair="ADA/USDT",
        stake_amount=60.0,
        amount=30.0,
        open_rate=open_rate,
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=10),
        fee_open=fee_rate,
        fee_close=fee_rate,
        exchange=exchange,
        interest_rate=0.0005,
        is_short=is_short,
        leverage=lev,
        trading_mode=trading_mode,
        funding_fees=funding_fees,
    )
    assert round(trade.calc_close_trade_value(rate=close_rate), 8) == result


@pytest.mark.parametrize(
    "exchange,is_short,lev,close_rate,fee_close,profit,profit_ratio,trading_mode,funding_fees",
    [
        ("binance", False, 1, 2.1, 0.0025, 2.6925, 0.044763092, spot, 0),
        ("binance", False, 3, 2.1, 0.0025, 8.075, 0.134247714, margin, 0),
        ("binance", True, 1, 2.1, 0.0025, -3.3088157, -0.055285142, margin, 0),
        ("binance", True, 3, 2.1, 0.0025, -9.92644709, -0.16585542, margin, 0),
        ("binance", False, 1, 1.9, 0.0025, -3.2925, -0.054738154, margin, 0),
        ("binance", False, 3, 1.9, 0.0025, -9.88, -0.164256026, margin, 0),
        ("binance", True, 1, 1.9, 0.0025, 2.70630953, 0.0452182043, margin, 0),
        ("binance", True, 3, 1.9, 0.0025, 8.11892859, 0.135654613, margin, 0),
        ("binance", False, 1, 2.2, 0.0025, 5.685, 0.09451371, margin, 0),
        ("binance", False, 3, 2.2, 0.0025, 17.0525, 0.28349958, margin, 0),
        ("binance", True, 1, 2.2, 0.0025, -6.3163784, -0.10553681, margin, 0),
        ("binance", True, 3, 2.2, 0.0025, -18.94913, -0.31661044, margin, 0),
        # Kraken
        ("kraken", False, 1, 2.1, 0.0025, 2.6925, 0.044763092, spot, 0),
        ("kraken", False, 3, 2.1, 0.0025, 7.9575, 0.132294264, margin, 0),
        ("kraken", True, 1, 2.1, 0.0025, -3.3706575, -0.056318421, margin, 0),
        ("kraken", True, 3, 2.1, 0.0025, -10.1119725, -0.168955263, margin, 0),
        ("kraken", False, 1, 1.9, 0.0025, -3.2925, -0.054738154, margin, 0),
        ("kraken", False, 3, 1.9, 0.0025, -9.9975, -0.166209476, margin, 0),
        ("kraken", True, 1, 1.9, 0.0025, 2.6503575, 0.044283333, margin, 0),
        ("kraken", True, 3, 1.9, 0.0025, 7.9510725, 0.132850000, margin, 0),
        ("kraken", False, 1, 2.2, 0.0025, 5.685, 0.09451371, margin, 0),
        ("kraken", False, 3, 2.2, 0.0025, 16.935, 0.28154613, margin, 0),
        ("kraken", True, 1, 2.2, 0.0025, -6.381165, -0.1066192, margin, 0),
        ("kraken", True, 3, 2.2, 0.0025, -19.143495, -0.3198578, margin, 0),
        ("binance", False, 1, 2.1, 0.003, 2.66100000, 0.044239401, spot, 0),
        ("binance", False, 1, 1.9, 0.003, -3.3209999, -0.055211970, spot, 0),
        ("binance", False, 1, 2.2, 0.003, 5.6520000, 0.093965087, spot, 0),
        # 期货, 资金费用=1
        ("binance", False, 1, 2.1, 0.0025, 3.6925, 0.06138819, futures, 1),
        ("binance", False, 3, 2.1, 0.0025, 9.0775, 0.15091438, futures, 1),
        ("binance", True, 1, 2.1, 0.0025, -2.3074999, -0.03855472, futures, 1),
        ("binance", True, 3, 2.1, 0.0025, -8.9225, -0.14908104, futures, 1),
        ("binance", False, 1, 1.9, 0.0025, -2.2925, -0.03811305, futures, 1),
        ("binance", False, 3, 1.9, 0.0025, -8.8775, -0.14758936, futures, 1),
        ("binance", True, 1, 1.9, 0.0025, 3.7075, 0.06194653, futures, 1),
        ("binance", True, 3, 1.9, 0.0025, 9.1225, 0.15242272, futures, 1),
        ("binance", False, 1, 2.2, 0.0025, 6.685, 0.11113881, futures, 1),
        ("binance", False, 3, 2.2, 0.0025, 18.055, 0.30016625, futures, 1),
        ("binance", True, 1, 2.2, 0.0025, -5.315, -0.08880534, futures, 1),
        ("binance", True, 3, 2.2, 0.0025, -17.945, -0.29983292, futures, 1),
        # 期货, 资金费用=-1
        ("binance", False, 1, 2.1, 0.0025, 1.6925, 0.02813798, futures, -1),
        ("binance", False, 3, 2.1, 0.0025, 7.0775, 0.11766417, futures, -1),
        ("binance", True, 1, 2.1, 0.0025, -4.307499, -0.07197159, futures, -1),
        ("binance", True, 3, 2.1, 0.0025, -10.92249, -0.18249791, futures, -1),
        ("binance", False, 1, 1.9, 0.0025, -4.292499, -0.07136325, futures, -1),
        ("binance", False, 3, 1.9, 0.0025, -10.87749, -0.18083957, futures, -1),
        ("binance", True, 1, 1.9, 0.0025, 1.7075, 0.02852965, futures, -1),
        ("binance", True, 3, 1.9, 0.0025, 7.1225, 0.11900585, futures, -1),
        ("binance", False, 1, 2.2, 0.0025, 4.684999, 0.07788861, futures, -1),
        ("binance", False, 3, 2.2, 0.0025, 16.055, 0.26691604, futures, -1),
        ("binance", True, 1, 2.2, 0.0025, -7.315, -0.12222222, futures, -1),
        ("binance", True, 3, 2.2, 0.0025, -19.945, -0.33324979, futures, -1),
        # 期货, 资金费用=0
        ("binance", False, 1, 2.1, 0.0025, 2.6925, 0.04476309, futures, 0),
        ("binance", False, 3, 2.1, 0.0025, 8.0775, 0.13428928, futures, 0),
        ("binance", True, 1, 2.1, 0.0025, -3.3074999, -0.05526316, futures, 0),
        ("binance", True, 3, 2.1, 0.0025, -9.9224997, -0.16578947, futures, 0),
        ("binance", False, 1, 1.9, 0.0025, -3.2925, -0.05473815, futures, 0),
        ("binance", False, 3, 1.9, 0.0025, -9.8775, -0.16421446, futures, 0),
        ("binance", True, 1, 1.9, 0.0025, 2.7075, 0.0452381, futures, 0),
        ("binance", True, 3, 1.9, 0.0025, 8.1225, 0.13571429, futures, 0),
    ],
)
@pytest.mark.usefixtures("init_persistence")
def test_calc_profit(
    exchange, is_short, lev, close_rate, fee_close, profit, profit_ratio, trading_mode, funding_fees
):
    """
    测试计算利润
    在Binance/Kraken上进行10分钟限价交易，1倍、3倍杠杆
    参数:
        费用:
            0.25% 报价
            0.30% 报价
        利率: 每4小时0.05%
        开盘价: 2.0 报价
        收盘价:
            1.9 报价
            2.1 报价
            2.2 报价
        数量: = 30.0 加密货币
        投入金额
            1x,-1x: 60.0  报价
            3x,-3x: 20.0  报价
        小时数: 1/6 (10分钟)
        资金费用: 1
    借入
         1x:  0 报价
         3x: 40 报价
        -1x: 30 加密货币
        -3x: 30 加密货币
    时间周期:
        kraken: (1 + 1) 4小时周期 = 2个4小时周期
        binance: 1/24个24小时周期
    利息: 借入 * 利率 * 时间周期
        1x            :  /
        binance     3x: 40 * 0.0005 * 1/24 = 0.0008333333333333334 报价
        kraken      3x: 40 * 0.0005 * 2    = 0.040 报价
        binace -1x,-3x: 30 * 0.0005 * 1/24 = 0.000625 加密货币
        kraken -1x,-3x: 30 * 0.0005 * 2    = 0.030 加密货币
    开仓价值: (数量 * 开盘价) ± (数量 * 开盘价 * 费用)
      0.0025 费用
         1x, 3x: 30 * 2 + 30 * 2 * 0.0025 = 60.15 报价
        -1x,-3x: 30 * 2 - 30 * 2 * 0.0025 = 59.85 报价
      0.003 费用: 在本测试中仅应用于收盘价
    平仓数量:
        1x, 3x                         = 数量
        -1x, -3x                       = 数量 + 利息
        binance -1x,-3x: 30 + 0.000625 = 30.000625 加密货币
        kraken  -1x,-3x: 30 + 0.03     = 30.03 加密货币
    平仓价值:
        公式:
            1x, 3x: (平仓数量 * 收盘价) - (平仓数量 * 收盘价 * 费用) - 利息
            -1x,-3x: (平仓数量 * 收盘价) + (平仓数量 * 收盘价 * 费用)
        2.1 报价
            bin,krak  1x: (30.00 * 2.1) - (30.00 * 2.1 * 0.0025)                = 62.8425
            bin       3x: (30.00 * 2.1) - (30.00 * 2.1 * 0.0025) - 0.0008333333 = 62.8416666667
            krak      3x: (30.00 * 2.1) - (30.00 * 2.1 * 0.0025) - 0.040        = 62.8025
            bin  -1x,-3x: (30.000625 * 2.1) + (30.000625 * 2.1 * 0.0025)        = 63.15881578125
            krak -1x,-3x: (30.03 * 2.1) + (30.03 * 2.1 * 0.0025)                = 63.2206575
        1.9 报价
            bin,krak  1x: (30.00 * 1.9) - (30.00 * 1.9 * 0.0025)                = 56.8575
            bin       3x: (30.00 * 1.9) - (30.00 * 1.9 * 0.0025) - 0.0008333333 = 56.85666667
            krak      3x: (30.00 * 1.9) - (30.00 * 1.9 * 0.0025) - 0.040        = 56.8175
            bin  -1x,-3x: (30.000625 * 1.9) + (30.000625 * 1.9 * 0.0025)        = 57.14369046875
            krak -1x,-3x: (30.03 * 1.9) + (30.03 * 1.9 * 0.0025)                = 57.1996425
        2.2 报价
            bin,krak  1x: (30.00 * 2.20) - (30.00 * 2.20 * 0.0025)              = 65.835
            bin       3x: (30.00 * 2.20) - (30.00 * 2.20 * 0.0025) - 0.00083333 = 65.83416667
            krak      3x: (30.00 * 2.20) - (30.00 * 2.20 * 0.0025) - 0.040      = 65.795
            bin  -1x,-3x: (30.000625 * 2.20) + (30.000625 * 2.20 * 0.0025)      = 66.1663784375
            krak -1x,-3x: (30.03 * 2.20) + (30.03 * 2.20 * 0.0025)              = 66.231165
    总利润:
        公式:
            1x, 3x : 平仓价值 - 开仓价值
            -1x,-3x: 开仓价值 - 平仓价值
        2.1 报价
            binance,kraken 1x: 62.8425     - 60.15          = 2.6925
            binance        3x: 62.84166667 - 60.15          = 2.69166667
            kraken         3x: 62.8025     - 60.15          = 2.6525
            binance   -1x,-3x: 59.850      - 63.15881578125 = -3.308815781249997
            kraken    -1x,-3x: 59.850      - 63.2206575     = -3.3706575
        1.9 报价
            binance,kraken 1x: 56.8575     - 60.15          = -3.2925
            binance        3x: 56.85666667 - 60.15          = -3.29333333
            kraken         3x: 56.8175     - 60.15          = -3.3325
            binance   -1x,-3x: 59.850      - 57.14369046875 = 2.7063095312499996
            kraken    -1x,-3x: 59.850      - 57.1996425     = 2.6503575
        2.2 报价
            binance,kraken 1x: 65.835      - 60.15          = 5.685
            binance        3x: 65.83416667 - 60.15          = 5.68416667
            kraken         3x: 65.795      - 60.15          = 5.645
            binance   -1x,-3x: 59.850      - 66.1663784375  = -6.316378437499999
            kraken    -1x,-3x: 59.850      - 66.231165      = -6.381165
    总利润率:
        公式:
            1x, 3x : ((平仓价值/开仓价值) - 1) * 杠杆
            -1x,-3x: (1 - (平仓价值/开仓价值)) * 杠杆
        2.1 报价
            binance,kraken 1x: (62.8425 / 60.15) - 1             = 0.04476309226932673
            binance        3x: ((62.84166667 / 60.15) - 1)*3     = 0.13424771421446402
            kraken         3x: ((62.8025 / 60.15) - 1)*3         = 0.13229426433915248
            binance       -1x: 1 - (63.15881578125 / 59.850)     = -0.05528514254385963
            binance       -3x: (1 - (63.15881578125 / 59.850))*3 = -0.1658554276315789
            kraken        -1x: 1 - (63.2206575 / 59.850)         = -0.05631842105263152
            kraken        -3x: (1 - (63.2206575 / 59.850))*3     = -0.16895526315789455
        1.9 报价
            binance,kraken 1x: (56.8575 / 60.15) - 1             = -0.05473815461346632
            binance        3x: ((56.85666667 / 60.15) - 1)*3     = -0.16425602643391513
            kraken         3x: ((56.8175 / 60.15) - 1)*3         = -0.16620947630922667
            binance       -1x: 1 - (57.14369046875 / 59.850)     = 0.045218204365079395
            binance       -3x: (1 - (57.14369046875 / 59.850))*3 = 0.13565461309523819
            kraken        -1x: 1 - (57.1996425 / 59.850)         = 0.04428333333333334
            kraken        -3x: (1 - (57.1996425 / 59.850))*3     = 0.13285000000000002
        2.2 报价
            binance,kraken 1x: (65.835 / 60.15) - 1             = 0.0945137157107232
            binance        3x: ((65.83416667 / 60.15) - 1)*3     = 0.2834995845386534
            kraken         3x: ((65.795 / 60.15) - 1)*3         = 0.2815461346633419
            binance       -1x: 1 - (66.1663784375 / 59.850)     = -0.1055368159983292
            binance       -3x: (1 - (66.1663784375 / 59.850))*3 = -0.3166104479949876
            kraken        -1x: 1 - (66.231165 / 59.850)         = -0.106619298245614
            kraken        -3x: (1 - (66.231165 / 59.850))*3     = -0.319857894736842
    费用: 0.003, 1x
        平仓价值:
            2.1 报价: (30.00 * 2.1) - (30.00 * 2.1 * 0.003) = 62.811
            1.9 报价: (30.00 * 1.9) - (30.00 * 1.9 * 0.003) = 56.829
            2.2 报价: (30.00 * 2.2) - (30.00 * 2.2 * 0.003) = 65.802
        总利润
            费用: 0.003, 1x
                2.1 报价: 62.811 - 60.15 = 2.6610000000000014
                1.9 报价: 56.829 - 60.15 = -3.320999999999998
                2.2 报价: 65.802 - 60.15 = 5.652000000000008
        总利润率
            费用: 0.003, 1x
                2.1 报价: (62.811 / 60.15) - 1 = 0.04423940149625927
                1.9 报价: (56.829 / 60.15) - 1 = -0.05521197007481293
                2.2 报价: (65.802 / 60.15) - 1 = 0.09396508728179565
    期货 (实时):
        资金费用: 1
            平仓价值:
                公式:
                    1x,3x: (数量 * 收盘价) - (数量 * 收盘价 * 费用) + 资金费用
                    -1x,-3x: (数量 * 收盘价) + (数量 * 收盘价 * 费用) - 资金费用
                2.1 报价
                    1x,3x: (30.00 * 2.1) - (30.00 * 2.1 * 0.0025) + 1   = 63.8425
                    -1x,-3x: (30.00 * 2.1) + (30.00 * 2.1 * 0.0025) - 1   = 62.1575
                1.9 报价
                    1x,3x: (30.00 * 1.9) - (30.00 * 1.9 * 0.0025) + 1   = 57.8575
                    -1x,-3x: (30.00 * 1.9) + (30.00 * 1.9 * 0.0025) - 1   = 56.1425
                2.2 报价:
                    1x,3x: (30.00 * 2.20) - (30.00 * 2.20 * 0.0025) + 1 = 66.835
                    -1x,-3x: (30.00 * 2.20) + (30.00 * 2.20 * 0.0025) - 1 = 65.165
            总利润:
                2.1 报价
                    1x,3x:   63.8425     - 60.15          = 3.6925
                    -1x,-3x: 59.850      - 62.1575        = -2.3074999999999974
                1.9 报价
                    1x,3x:   57.8575     - 60.15          = -2.2925
                    -1x,-3x: 59.850      - 56.1425        = 3.707500000000003
                2.2 报价:
                    1x,3x:   66.835      - 60.15          = 6.685
                    -1x,-3x: 59.850      - 65.165         = -5.315000000000005
            总利润率:
                2.1 报价
                    1x: (63.8425 / 60.15) - 1             = 0.06138819617622615
                    3x: ((63.8425 / 60.15) - 1)*3         = 0.18416458852867845
                    -1x: 1 - (62.1575 / 59.850)           = -0.038554720133667564
                    -3x: (1 - (62.1575 / 59.850))*3       = -0.11566416040100269
                1.9 报价
                    1x: (57.8575 / 60.15) - 1             = -0.0381130507065669
                    3x: ((57.8575 / 60.15) - 1)*3         = -0.1143391521197007
                    -1x: 1 - (56.1425 / 59.850)           = 0.06194653299916464
                    -3x: (1 - (56.1425 / 59.850))*3       = 0.18583959899749392
                2.2 报价
                    1x: (66.835 / 60.15) - 1             = 0.11113881961762262
                    3x: ((66.835 / 60.15) - 1)*3         = 0.33341645885286786
                    -1x: 1 - (65.165 / 59.850)           = -0.08880534670008355
                    -3x: (1 - (65.165 / 59.850))*3       = -0.26641604010025066
        资金费用: -1
            平仓价值:
                公式:
                    (数量 * 收盘价) - (数量 * 收盘价 * 费用) + 资金费用
                    (数量 * 收盘价) - (数量 * 收盘价 * 费用) - 资金费用
                2.1 报价
                    1x,3x:  (30.00 * 2.1) - (30.00 * 2.1 * 0.0025) + (-1)   = 61.8425
                    -1x,-3x: (30.00 * 2.1) + (30.00 * 2.1 * 0.0025) - (-1)   = 64.1575
                1.9 报价
                    1x,3x:  (30.00 * 1.9) - (30.00 * 1.9 * 0.0025) + (-1)   = 55.8575
                    -1x,-3x: (30.00 * 1.9) + (30.00 * 1.9 * 0.0025) - (-1)   = 58.1425
                2.2 报价:
                    1x,3x:  (30.00 * 2.20) - (30.00 * 2.20 * 0.0025) + (-1) = 64.835
                    -1x,-3x: (30.00 * 2.20) + (30.00 * 2.20 * 0.0025) - (-1) = 67.165
            总利润:
                2.1 报价
                    1x,3x:   61.8425     - 60.15          = 1.6925000000000026
                    -1x,-3x: 59.850      - 64.1575        = -4.307499999999997
                1.9 报价
                    1x,3x:   55.8575     - 60.15          = -4.292499999999997
                    -1x,-3x: 59.850      - 58.1425        = 1.7075000000000031
                2.2 报价:
                    1x,3x:   64.835      - 60.15          = 4.684999999999995
                    -1x,-3x: 59.850      - 67.165         = -7.315000000000005
            总利润率:
                2.1 报价
                    1x: (61.8425 / 60.15) - 1             = 0.028137988362427313
                    3x: ((61.8425 / 60.15) - 1)*3         = 0.08441396508728194
                    -1x: 1 - (64.1575 / 59.850)           = -0.07197159565580624
                    -3x: (1 - (64.1575 / 59.850))*3       = -0.21591478696741873
                1.9 报价
                    1x: (55.8575 / 60.15) - 1             = -0.07136325852036574
                    3x: ((55.8575 / 60.15) - 1)*3         = -0.2140897755610972
                    -1x: 1 - (58.1425 / 59.850)           = 0.02852965747702596
                    -3x: (1 - (58.1425 / 59.850))*3       = 0.08558897243107788
                2.2 报价
                    1x: (64.835 / 60.15) - 1              = 0.07788861180382378
                    3x: ((64.835 / 60.15) - 1)*3          = 0.23366583541147135
                    -1x: 1 - (67.165 / 59.850)            = -0.12222222222222223
                    -3x: (1 - (67.165 / 59.850))*3        = -0.3666666666666667
    """
    trade = Trade(
        pair="ADA/USDT",
        stake_amount=60.0,
        amount=30.0 * lev,
        open_rate=2.0,
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=10),
        interest_rate=0.0005,
        exchange=exchange,
        is_short=is_short,
        leverage=lev,
        fee_open=0.0025,
        fee_close=fee_close,
        max_stake_amount=60.0,
        trading_mode=trading_mode,
        funding_fees=funding_fees,
    )

    profit_res = trade.calculate_profit(close_rate)
    assert pytest.approx(profit_res.profit_abs) == round(profit, 8)
    assert pytest.approx(profit_res.profit_ratio) == round(profit_ratio, 8)
    val = trade.open_trade_value * (profit_res.profit_ratio) / lev
    assert pytest.approx(val) == profit_res.profit_abs

    assert pytest.approx(profit_res.total_profit) == round(profit, 8)
    assert pytest.approx(profit_res.total_profit_ratio) == round(profit_ratio, 8)

    assert pytest.approx(trade.calc_profit(rate=close_rate)) == round(profit, 8)
    assert pytest.approx(trade.calc_profit_ratio(rate=close_rate)) == round(profit_ratio, 8)

    profit_res2 = trade.calculate_profit(close_rate, trade.amount, trade.open_rate)
    assert pytest.approx(profit_res2.profit_abs) == round(profit, 8)
    assert pytest.approx(profit_res2.profit_ratio) == round(profit_ratio, 8)

    assert pytest.approx(profit_res2.total_profit) == round(profit, 8)
    assert pytest.approx(profit_res2.total_profit_ratio) == round(profit_ratio, 8)

    assert pytest.approx(trade.calc_profit(close_rate, trade.amount, trade.open_rate)) == round(
        profit, 8
    )
    assert pytest.approx(
        trade.calc_profit_ratio(close_rate, trade.amount, trade.open_rate)
    ) == round(profit_ratio, 8)


def test_adjust_stop_loss(fee):
    """测试调整止损"""
    trade = Trade(
        pair="ADA/USDT",
        stake_amount=30.0,
        amount=30,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange="binance",
        open_rate=1,
        max_rate=1,
    )

    trade.adjust_stop_loss(trade.open_rate, 0.05, True)
    assert trade.stop_loss == 0.95
    assert trade.stop_loss_pct == -0.05
    assert trade.initial_stop_loss == 0.95
    assert trade.initial_stop_loss_pct == -0.05

    # 使用较低价格获取利润百分比
    trade.adjust_stop_loss(0.96, 0.05)
    assert trade.stop_loss == 0.95
    assert trade.stop_loss_pct == -0.05
    assert trade.initial_stop_loss == 0.95
    assert trade.initial_stop_loss_pct == -0.05

    # 使用自定义价格（高于开盘价）获取利润百分比
    trade.adjust_stop_loss(1.3, -0.1)
    assert pytest.approx(trade.stop_loss) == 1.17
    assert trade.stop_loss_pct == -0.1
    assert trade.initial_stop_loss == 0.95
    assert trade.initial_stop_loss_pct == -0.05

    # 当前价格再次降低...不应该改变
    trade.adjust_stop_loss(1.2, 0.1)
    assert pytest.approx(trade.stop_loss) == 1.17
    assert trade.initial_stop_loss == 0.95
    assert trade.initial_stop_loss_pct == -0.05

    # 当前价格更高...应该提高止损
    trade.adjust_stop_loss(1.4, 0.1)
    assert pytest.approx(trade.stop_loss) == 1.26
    assert trade.initial_stop_loss == 0.95
    assert trade.initial_stop_loss_pct == -0.05

    # Initial为true但已设置止损 - 所以不执行任何操作
    trade.adjust_stop_loss(1.7, 0.1, True)
    assert pytest.approx(trade.stop_loss) == 1.26
    assert trade.initial_stop_loss == 0.95
    assert trade.initial_stop_loss_pct == -0.05
    assert trade.stop_loss_pct == -0.1


def test_adjust_stop_loss_short(fee):
    """测试做空时调整止损"""
    trade = Trade(
        pair="ADA/USDT",
        stake_amount=0.001,
        amount=5,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange="binance",
        open_rate=1,
        max_rate=1,
        is_short=True,
    )
    trade.adjust_stop_loss(trade.open_rate, 0.05, True)
    assert trade.stop_loss == 1.05
    assert trade.stop_loss_pct == -0.05
    assert trade.initial_stop_loss == 1.05
    assert trade.initial_stop_loss_pct == -0.05
    # 使用较低价格获取利润百分比
    trade.adjust_stop_loss(1.04, 0.05)
    assert trade.stop_loss == 1.05
    assert trade.stop_loss_pct == -0.05
    assert trade.initial_stop_loss == 1.05
    assert trade.initial_stop_loss_pct == -0.05
    # 使用自定义价格（高于开盘价）获取利润百分比
    trade.adjust_stop_loss(0.7, 0.1)
    # 如果价格下降到0.7，跟踪止损为0.1，
    # 在0.7上方0.1的新止损将高出0.7*0.1
    assert round(trade.stop_loss, 8) == 0.77
    assert trade.stop_loss_pct == -0.1
    assert trade.initial_stop_loss == 1.05
    assert trade.initial_stop_loss_pct == -0.05
    # 当前价格再次降低...不应该改变
    trade.adjust_stop_loss(0.8, -0.1)
    assert round(trade.stop_loss, 8) == 0.77
    assert trade.initial_stop_loss == 1.05
    assert trade.initial_stop_loss_pct == -0.05
    # 当前价格更高...应该提高止损
    trade.adjust_stop_loss(0.6, -0.1)
    assert round(trade.stop_loss, 8) == 0.66
    assert trade.initial_stop_loss == 1.05
    assert trade.initial_stop_loss_pct == -0.05
    # Initial为true但已设置止损 - 所以不执行任何操作
    trade.adjust_stop_loss(0.3, -0.1, True)
    assert round(trade.stop_loss, 8) == 0.66
    assert trade.initial_stop_loss == 1.05
    assert trade.initial_stop_loss_pct == -0.05
    assert trade.stop_loss_pct == -0.1
    # 清算价格低于止损 - 所以清算会先触发。
    trade.set_liquidation_price(0.63)
    trade.adjust_stop_loss(0.59, -0.1)
    assert trade.stop_loss == 0.649
    assert trade.liquidation_price == 0.63


def test_adjust_min_max_rates(fee):
    """测试调整最小最大价格"""
    trade = Trade(
        pair="ADA/USDT",
        stake_amount=30.0,
        amount=30.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange="binance",
        open_rate=1,
    )

    trade.adjust_min_max_rates(trade.open_rate, trade.open_rate)
    assert trade.max_rate == 1
    assert trade.min_rate == 1

    # 检查最小值调整，最大值保持不变
    trade.adjust_min_max_rates(0.96, 0.96)
    assert trade.max_rate == 1
    assert trade.min_rate == 0.96

    # 检查最大值调整，最小值保持不变
    trade.adjust_min_max_rates(1.05, 1.05)
    assert trade.max_rate == 1.05
    assert trade.min_rate == 0.96

    # 当前价格"在中间" - 不调整
    trade.adjust_min_max_rates(1.03, 1.03)
    assert trade.max_rate == 1.05
    assert trade.min_rate == 0.96

    # 当前价格"在中间" - 不调整
    trade.adjust_min_max_rates(1.10, 0.91)
    assert trade.max_rate == 1.10
    assert trade.min_rate == 0.91


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("use_db", [True, False])
@pytest.mark.parametrize("is_short", [True, False])
def test_get_open(fee, is_short, use_db):
    """测试获取开放交易"""
    Trade.use_db = use_db
    Trade.reset_trades()

    create_mock_trades(fee, is_short, use_db)
    assert len(Trade.get_open_trades()) == 4
    assert Trade.get_open_trade_count() == 4

    Trade.use_db = True


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("use_db", [True, False])
def test_get_open_lev(fee, use_db):
    """测试获取带杠杆的开放交易"""
    Trade.use_db = use_db
    Trade.reset_trades()

    create_mock_trades_with_leverage(fee, use_db)
    assert len(Trade.get_open_trades()) == 5
    assert Trade.get_open_trade_count() == 5

    Trade.use_db = True


@pytest.mark.parametrize("is_short", [True, False])
@pytest.mark.parametrize("use_db", [True, False])
@pytest.mark.usefixtures("init_persistence")
def test_get_open_orders(fee, is_short, use_db):
    """测试获取开放订单"""
    Trade.use_db = use_db
    Trade.reset_trades()

    create_mock_trades_usdt(fee, is_short, use_db)
    # Trade.commit()
    trade = Trade.get_trades_proxy(pair="XRP/USDT")[0]
    # assert trade.id == 3
    assert len(trade.orders) == 2
    assert len(trade.open_orders) == 0
    assert not trade.has_open_orders

    Trade.use_db = True


@pytest.mark.usefixtures("init_persistence")
def test_to_json(fee):
    """测试转换为JSON"""
    # 模拟dry_run条目
    trade = Trade(
        pair="ADA/USDT",
        stake_amount=0.001,
        amount=123.0,
        amount_requested=123.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_date=dt_now() - timedelta(hours=2),
        open_rate=0.123,
        exchange="binance",
        enter_tag=None,
        precision_mode=1,
        precision_mode_price=1,
        amount_precision=8.0,
        price_precision=7.0,
        contract_size=1,
    )
    result = trade.to_json()
    assert isinstance(result, dict)

    assert result == {
        "trade_id": None,
        "pair": "ADA/USDT",
        "base_currency": "ADA",
        "quote_currency": "USDT",
        "is_open": None,
        "open_date": trade.open_date.strftime(DATETIME_PRINT_FORMAT),
        "open_timestamp": int(trade.open_date.timestamp() * 1000),
        "open_fill_date": None,
        "open_fill_timestamp": None,
        "close_date": None,
        "close_timestamp": None,
        "open_rate": 0.123,
        "open_rate_requested": None,
        "open_trade_value": 15.1668225,
        "fee_close": 0.0025,
        "fee_close_cost": None,
        "fee_close_currency": None,
        "fee_open": 0.0025,
        "fee_open_cost": None,
        "fee_open_currency": None,
        "close_rate": None,
        "close_rate_requested": None,
        "amount": 123.0,
        "amount_requested": 123.0,
        "stake_amount": 0.001,
        "max_stake_amount": None,
        "trade_duration": None,
        "trade_duration_s": None,
        "realized_profit": 0.0,
        "realized_profit_ratio": None,
        "close_profit": None,
        "close_profit_pct": None,
        "close_profit_abs": None,
        "profit_ratio": None,
        "profit_pct": None,
        "profit_abs": None,
        "exit_reason": None,
        "exit_order_status": None,
        "stop_loss_abs": None,
        "stop_loss_ratio": None,
        "stop_loss_pct": None,
        "stoploss_last_update": None,
        "stoploss_last_update_timestamp": None,
        "initial_stop_loss_abs": None,
        "initial_stop_loss_pct": None,
        "initial_stop_loss_ratio": None,
        "min_rate": None,
        "max_rate": None,
        "strategy": None,
        "enter_tag": None,
        "timeframe": None,
        "exchange": "binance",
        "leverage": None,
        "interest_rate": None,
        "liquidation_price": None,
        "is_short": None,
        "trading_mode": None,
        "funding_fees": None,
        "amount_precision": 8.0,
        "price_precision": 7.0,
        "precision_mode": 1,
        "precision_mode_price": 1,
        "contract_size": 1,
        "orders": [],
        "has_open_orders": False,
    }

    # 模拟dry_run条目
    trade = Trade(
        pair="XRP/BTC",
        stake_amount=0.001,
        amount=100.0,
        amount_requested=101.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_date=dt_now() - timedelta(hours=2),
        close_date=dt_now() - timedelta(hours=1),
        open_rate=0.123,
        close_rate=0.125,
        enter_tag="buys_signal_001",
        exchange="binance",
        precision_mode=2,
        precision_mode_price=1,
        amount_precision=7.0,
        price_precision=8.0,
        contract_size=1,
    )
    result = trade.to_json()
    assert isinstance(result, dict)

    assert result == {
        "trade_id": None,
        "pair": "XRP/BTC",
        "base_currency": "XRP",
        "quote_currency": "BTC",
        "open_date": trade.open_date.strftime(DATETIME_PRINT_FORMAT),
        "open_timestamp": int(trade.open_date.timestamp() * 1000),
        "open_fill_date": None,
        "open_fill_timestamp": None,
        "close_date": trade.close_date.strftime(DATETIME_PRINT_FORMAT),
        "close_timestamp": int(trade.close_date.timestamp() * 1000),
        "open_rate": 0.123,
        "close_rate": 0.125,
        "amount": 100.0,
        "amount_requested": 101.0,
        "stake_amount": 0.001,
        "max_stake_amount": None,
        "trade_duration": 60,
        "trade_duration_s": 3600,
        "stop_loss_abs": None,
        "stop_loss_pct": None,
        "stop_loss_ratio": None,
        "stoploss_last_update": None,
        "stoploss_last_update_timestamp": None,
        "initial_stop_loss_abs": None,
        "initial_stop_loss_pct": None,
        "initial_stop_loss_ratio": None,
        "realized_profit": 0.0,
        "realized_profit_ratio": None,
        "close_profit": None,
        "close_profit_pct": None,
        "close_profit_abs": None,
        "profit_ratio": None,
        "profit_pct": None,
        "profit_abs": None,
        "close_rate_requested": None,
        "fee_close": 0.0025,
        "fee_close_cost": None,
        "fee_close_currency": None,
        "fee_open": 0.0025,
        "fee_open_cost": None,
        "fee_open_currency": None,
        "is_open": None,
        "max_rate": None,
        "min_rate": None,
        "open_rate_requested": None,
        "open_trade_value": 12.33075,
        "exit_reason": None,
        "exit_order_status": None,
        "strategy": None,
        "enter_tag": "buys_signal_001",
        "timeframe": None,
        "exchange": "binance",
        "leverage": None,
        "interest_rate": None,
        "liquidation_price": None,
        "is_short": None,
        "trading_mode": None,
        "funding_fees": None,
        "amount_precision": 7.0,
        "price_precision": 8.0,
        "precision_mode": 2,
        "precision_mode_price": 1,
        "contract_size": 1,
        "orders": [],
        "has_open_orders": False,
    }


def test_stoploss_reinitialization(default_conf, fee):
    """测试止损重新初始化"""
    init_db(default_conf["db_url"])
    trade = Trade(
        pair="ADA/USDT",
        stake_amount=30.0,
        fee_open=fee.return_value,
        open_date=dt_now() - timedelta(hours=2),
        amount=30.0,
        fee_close=fee.return_value,
        exchange="binance",
        open_rate=1,
        max_rate=1,
    )

    trade.adjust_stop_loss(trade.open_rate, 0.05, True)
    assert trade.stop_loss == 0.95
    assert trade.stop_loss_pct == -0.05
    assert trade.initial_stop_loss == 0.95
    assert trade.initial_stop_loss_pct == -0.05
    Trade.session.add(trade)
    Trade.commit()

    # 降低止损
    Trade.stoploss_reinitialization(0.06)

    trades = Trade.get_open_trades()
    assert len(trades) == 1
    trade_adj = trades[0]
    assert trade_adj.stop_loss == 0.94
    assert trade_adj.stop_loss_pct == -0.06
    assert trade_adj.initial_stop_loss == 0.94
    assert trade_adj.initial_stop_loss_pct == -0.06

    # 提高止损
    Trade.stoploss_reinitialization(0.04)

    trades = Trade.get_open_trades()
    assert len(trades) == 1
    trade_adj = trades[0]
    assert trade_adj.stop_loss == 0.96
    assert trade_adj.stop_loss_pct == -0.04
    assert trade_adj.initial_stop_loss == 0.96
    assert trade_adj.initial_stop_loss_pct == -0.04

    # 跟踪止损（将止损向上移动一点）
    trade.adjust_stop_loss(1.02, 0.04)
    assert trade_adj.stop_loss == 0.9792
    assert trade_adj.initial_stop_loss == 0.96

    Trade.stoploss_reinitialization(0.04)

    trades = Trade.get_open_trades()
    assert len(trades) == 1
    trade_adj = trades[0]
    # 在这种情况下止损不应该改变。
    assert trade_adj.stop_loss == 0.9792
    assert trade_adj.stop_loss_pct == -0.04
    assert trade_adj.initial_stop_loss == 0.96
    assert trade_adj.initial_stop_loss_pct == -0.04


def test_stoploss_reinitialization_leverage(default_conf, fee):
    """测试带杠杆的止损重新初始化"""
    init_db(default_conf["db_url"])
    trade = Trade(
        pair="ADA/USDT",
        stake_amount=30.0,
        fee_open=fee.return_value,
        open_date=dt_now() - timedelta(hours=2),
        amount=30.0,
        fee_close=fee.return_value,
        exchange="binance",
        open_rate=1,
        max_rate=1,
        leverage=5.0,
    )

    trade.adjust_stop_loss(trade.open_rate, 0.1, True)
    assert trade.stop_loss == 0.98
    assert trade.stop_loss_pct == -0.1
    assert trade.initial_stop_loss == 0.98
    assert trade.initial_stop_loss_pct == -0.1
    Trade.session.add(trade)
    Trade.commit()

    # 降低止损
    Trade.stoploss_reinitialization(0.15)

    trades = Trade.get_open_trades()
    assert len(trades) == 1
    trade_adj = trades[0]
    assert trade_adj.stop_loss == 0.97
    assert trade_adj.stop_loss_pct == -0.15
    assert trade_adj.initial_stop_loss == 0.97
    assert trade_adj.initial_stop_loss_pct == -0.15

    # 提高止损
    Trade.stoploss_reinitialization(0.05)

    trades = Trade.get_open_trades()
    assert len(trades) == 1
    trade_adj = trades[0]
    assert trade_adj.stop_loss == 0.99
    assert trade_adj.stop_loss_pct == -0.05
    assert trade_adj.initial_stop_loss == 0.99
    assert trade_adj.initial_stop_loss_pct == -0.05

    # 跟踪止损（将止损向上移动一点）
    trade.adjust_stop_loss(1.02, 0.05)
    assert trade_adj.stop_loss == 1.0098
    assert trade_adj.initial_stop_loss == 0.99

    Trade.stoploss_reinitialization(0.05)

    trades = Trade.get_open_trades()
    assert len(trades) == 1
    trade_adj = trades[0]
    # 在这种情况下止损不应该改变。
    assert trade_adj.stop_loss == 1.0098
    assert trade_adj.stop_loss_pct == -0.05
    assert trade_adj.initial_stop_loss == 0.99
    assert trade_adj.initial_stop_loss_pct == -0.05


def test_stoploss_reinitialization_short(default_conf, fee):
    """测试做空时的止损重新初始化"""
    init_db(default_conf["db_url"])
    trade = Trade(
        pair="ADA/USDT",
        stake_amount=0.001,
        fee_open=fee.return_value,
        open_date=dt_now() - timedelta(hours=2),
        amount=10,
        fee_close=fee.return_value,
        exchange="binance",
        open_rate=1,
        max_rate=1,
        is_short=True,
        leverage=5.0,
    )
    trade.adjust_stop_loss(trade.open_rate, -0.1, True)
    assert trade.stop_loss == 1.02
    assert trade.stop_loss_pct == -0.1
    assert trade.initial_stop_loss == 1.02
    assert trade.initial_stop_loss_pct == -0.1
    Trade.session.add(trade)
    Trade.commit()
    # 降低止损
    Trade.stoploss_reinitialization(-0.15)
    trades = Trade.get_open_trades()
    assert len(trades) == 1
    trade_adj = trades[0]
    assert trade_adj.stop_loss == 1.03
    assert trade_adj.stop_loss_pct == -0.15
    assert trade_adj.initial_stop_loss == 1.03
    assert trade_adj.initial_stop_loss_pct == -0.15
    # 提高止损
    Trade.stoploss_reinitialization(-0.05)
    trades = Trade.get_open_trades()
    assert len(trades) == 1
    trade_adj = trades[0]
    assert trade_adj.stop_loss == 1.01
    assert trade_adj.stop_loss_pct == -0.05
    assert trade_adj.initial_stop_loss == 1.01
    assert trade_adj.initial_stop_loss_pct == -0.05
    # 跟踪止损
    trade.adjust_stop_loss(0.98, -0.05)
    assert trade_adj.stop_loss == 0.9898
    assert trade_adj.initial_stop_loss == 1.01
    Trade.stoploss_reinitialization(-0.05)
    trades = Trade.get_open_trades()
    assert len(trades) == 1
    trade_adj = trades[0]
    # 在这种情况下止损不应该改变。
    assert trade_adj.stop_loss == 0.9898
    assert trade_adj.stop_loss_pct == -0.05
    assert trade_adj.initial_stop_loss == 1.01
    assert trade_adj.initial_stop_loss_pct == -0.05
    # 止损不能高于清算价格
    trade_adj.set_liquidation_price(0.985)
    trade.adjust_stop_loss(0.9799, -0.05)
    assert trade_adj.stop_loss == 0.989699
    assert trade_adj.liquidation_price == 0.985


def test_update_fee(fee):
    """测试更新费用"""
    trade = Trade(
        pair="ADA/USDT",
        stake_amount=30.0,
        fee_open=fee.return_value,
        open_date=dt_now() - timedelta(hours=2),
        amount=30.0,
        fee_close=fee.return_value,
        exchange="binance",
        open_rate=1,
        max_rate=1,
    )
    fee_cost = 0.15
    fee_currency = "BTC"
    fee_rate = 0.0075
    assert trade.fee_open_currency is None
    assert not trade.fee_updated("buy")
    assert not trade.fee_updated("sell")

    trade.update_fee(fee_cost, fee_currency, fee_rate, "buy")
    assert trade.fee_updated("buy")
    assert not trade.fee_updated("sell")
    assert trade.fee_open_currency == fee_currency
    assert trade.fee_open_cost == fee_cost
    assert trade.fee_open == fee_rate
    # 设置买入费率应该"猜测"卖出费率
    assert trade.fee_close == fee_rate
    assert trade.fee_close_currency is None
    assert trade.fee_close_cost is None

    fee_rate = 0.0076
    trade.update_fee(fee_cost, fee_currency, fee_rate, "sell")
    assert trade.fee_updated("buy")
    assert trade.fee_updated("sell")
    assert trade.fee_close == 0.0076
    assert trade.fee_close_cost == fee_cost
    assert trade.fee_close == fee_rate


def test_fee_updated(fee):
    """测试费用是否已更新"""
    trade = Trade(
        pair="ADA/USDT",
        stake_amount=30.0,
        fee_open=fee.return_value,
        open_date=dt_now() - timedelta(hours=2),
        amount=30.0,
        fee_close=fee.return_value,
        exchange="binance",
        open_rate=1,
        max_rate=1,
    )

    assert trade.fee_open_currency is None
    assert not trade.fee_updated("buy")
    assert not trade.fee_updated("sell")
    assert not trade.fee_updated("asdf")

    trade.update_fee(0.15, "BTC", 0.0075, "buy")
    assert trade.fee_updated("buy")
    assert not trade.fee_updated("sell")
    assert trade.fee_open_currency is not None
    assert trade.fee_close_currency is None

    trade.update_fee(0.15, "ABC", 0.0075, "sell")
    assert trade.fee_updated("buy")
    assert trade.fee_updated("sell")
    assert not trade.fee_updated("asfd")


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("is_short", [True, False])
@pytest.mark.parametrize("use_db", [True, False])
def test_total_open_trades_stakes(fee, is_short, use_db):
    """测试计算开放交易的总投入"""
    Trade.use_db = use_db
    Trade.reset_trades()
    res = Trade.total_open_trades_stakes()
    assert res == 0
    create_mock_trades(fee, is_short, use_db)
    res = Trade.total_open_trades_stakes()
    assert res == 0.004

    Trade.use_db = True


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize(
    "is_short,result",
    [
        (True, -0.006739127),
        (False, 0.000739127),
        (None, -0.005429127),
    ],
)
@pytest.mark.parametrize("use_db", [True, False])
def test_get_total_closed_profit(fee, use_db, is_short, result):
    """测试获取总关闭利润"""
    Trade.use_db = use_db
    Trade.reset_trades()
    res = Trade.get_total_closed_profit()
    assert res == 0
    create_mock_trades(fee, is_short, use_db)
    res = Trade.get_total_closed_profit()
    assert pytest.approx(res) == result

    Trade.use_db = True


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("is_short", [True, False])
@pytest.mark.parametrize("use_db", [True, False])
def test_get_trades_proxy(fee, use_db, is_short):
    """测试获取交易代理"""
    Trade.use_db = use_db
    Trade.reset_trades()
    create_mock_trades(fee, is_short, use_db)
    trades = Trade.get_trades_proxy()
    assert len(trades) == 6

    assert isinstance(trades[0], Trade)

    trades = Trade.get_trades_proxy(is_open=True)
    assert len(trades) == 4
    assert trades[0].is_open
    trades = Trade.get_trades_proxy(is_open=False)

    assert len(trades) == 2
    assert not trades[0].is_open

    opendate = datetime.now(tz=timezone.utc) - timedelta(minutes=15)

    assert len(Trade.get_trades_proxy(open_date=opendate)) == 3

    Trade.use_db = True


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("is_short", [True, False])
def test_get_trades__query(fee, is_short):
    """测试获取交易查询"""
    query = Trade.get_trades_query([])
    # 没有订单时不应该发出join。
    query1 = Trade.get_trades_query([], include_orders=False)

    # 空的"with-options -> 默认 - 选择"
    assert query._with_options == ()
    assert query1._with_options != ()

    create_mock_trades(fee, is_short)
    query = Trade.get_trades_query([])
    query1 = Trade.get_trades_query([], include_orders=False)

    assert query._with_options == ()
    assert query1._with_options != ()


def test_get_trades_backtest():
    """测试回测中获取交易"""
    Trade.use_db = False
    with pytest.raises(NotImplementedError, match=r"`Trade.get_trades\(\)` not .*"):
        Trade.get_trades([])
    Trade.use_db = True


@pytest.mark.usefixtures("init_persistence")
# @pytest.mark.parametrize('is_short', [True, False])
def test_get_overall_performance(fee):
    """测试获取整体表现"""
    create_mock_trades(fee, False)
    res = Trade.get_overall_performance()

    assert len(res) == 2
    assert "pair" in res[0]
    assert "profit" in res[0]
    assert "count" in res[0]


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize(
    "is_short,pair,profit",
    [
        (True, "XRP/BTC", -0.00018780487),
        (False, "ETC/BTC", 0.00003860975),
        (None, "XRP/BTC", 0.000025203252),
    ],
)
def test_get_best_pair(fee, is_short, pair, profit):
    """测试获取最佳交易对"""
    res = Trade.get_best_pair()
    assert res is None

    create_mock_trades(fee, is_short)
    res = Trade.get_best_pair()
    assert len(res) == 4
    assert res[0] == pair
    assert pytest.approx(res[1]) == profit


@pytest.mark.usefixtures("init_persistence")
def test_get_best_pair_lev(fee):
    """测试获取带杠杆的最佳交易对"""
    res = Trade.get_best_pair()
    assert res is None

    create_mock_trades_with_leverage(fee)
    res = Trade.get_best_pair()
    assert len(res) == 4
    assert res[0] == "ETC/BTC"
    assert pytest.approx(res[1]) == 0.00003860975


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("is_short", [True, False])
def test_get_canceled_exit_order_count(fee, is_short):
    """测试获取取消的退出订单数量"""
    create_mock_trades(fee, is_short=is_short)
    trade = Trade.get_trades([Trade.pair == "ETC/BTC"]).first()
    # 没有取消的订单。
    assert trade.get_canceled_exit_order_count() == 0
    # 属性返回相同的结果
    assert trade.canceled_exit_order_count == 0

    trade.orders[-1].status = "canceled"
    assert trade.get_canceled_exit_order_count() == 1
    assert trade.canceled_exit_order_count == 1


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("is_short", [True, False])
def test_fully_canceled_entry_order_count(fee, is_short):
    """测试完全取消的入场订单数量"""
    create_mock_trades(fee, is_short=is_short)
    trade = Trade.get_trades([Trade.pair == "ETC/BTC"]).first()
    # 没有取消的订单。
    assert trade.fully_canceled_entry_order_count == 0

    trade.orders[0].status = "canceled"
    trade.orders[0].filled = 0
    assert trade.fully_canceled_entry_order_count == 1


@pytest.mark.usefixtures("init_persistence")
def test_update_order_from_ccxt(caplog, time_machine):
    """测试从CCXT更新订单"""
    start = datetime(2023, 1, 1, 4, tzinfo=timezone.utc)
    time_machine.move_to(start, tick=False)

    # 最基本的订单返回（只有订单ID）
    o = Order.parse_from_ccxt_object({"id": "1234"}, "ADA/USDT", "buy", 20.01, 1234.6)
    assert isinstance(o, Order)
    assert o.ft_pair == "ADA/USDT"
    assert o.ft_order_side == "buy"
    assert o.order_id == "1234"
    assert o.ft_price == 1234.6
    assert o.ft_amount == 20.01
    assert o.ft_is_open
    ccxt_order = {
        "id": "1234",
        "side": "buy",
        "symbol": "ADA/USDT",
        "type": "limit",
        "price": 1234.5,
        "amount": 20.0,
        "filled": 9,
        "remaining": 11,
        "status": "open",
        "timestamp": 1599394315123,
    }
    o = Order.parse_from_ccxt_object(ccxt_order, "ADA/USDT", "buy", 20.01, 1234.6)
    assert isinstance(o, Order)
    assert o.ft_pair == "ADA/USDT"
    assert o.ft_order_side == "buy"
    assert o.order_id == "1234"
    assert o.order_type == "limit"
    assert o.price == 1234.5
    assert o.ft_price == 1234.6
    assert o.ft_amount == 20.01
    assert o.filled == 9
    assert o.remaining == 11
    assert o.order_date is not None
    assert o.ft_is_open
    assert o.order_filled_date is None

    # 订单未成交，"filled"未设置
    # https://github.com/freqtrade/freqtrade/issues/5404
    ccxt_order.update({"filled": None, "remaining": 20.0, "status": "canceled"})
    o.update_from_ccxt_object(ccxt_order)

    # 订单已关闭
    ccxt_order.update({"filled": 20.0, "remaining": 0.0, "status": "closed"})
    o.update_from_ccxt_object(ccxt_order)

    assert o.filled == 20.0
    assert o.remaining == 0.0
    assert not o.ft_is_open
    assert o.order_filled_date == start
    # 移动时间
    time_machine.move_to(start + timedelta(hours=1), tick=False)

    ccxt_order.update({"id": "somethingelse"})
    with pytest.raises(DependencyException, match=r"订单ID不匹配"):
        o.update_from_ccxt_object(ccxt_order)

    message = "aaaa 不是有效的响应对象。"
    assert not log_has(message, caplog)
    Order.update_orders([o], "aaaa")
    assert log_has(message, caplog)

    # 调用常规更新 - 不应该失败。
    Order.update_orders([o], {"id": "1234"})
    assert o.order_filled_date == start

    # 再次填充订单 - 不应该更新填充日期
    ccxt_order.update({"id": "1234"})
    Order.update_orders([o], ccxt_order)
    assert o.order_filled_date == start


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("is_short", [True, False])
def test_select_order(fee, is_short):
    """测试选择订单"""
    create_mock_trades(fee, is_short)

    trades = Trade.get_trades().all()

    # 开放买入订单，没有卖出订单
    order = trades[0].select_order(trades[0].entry_side, True)
    assert order is not None
    order = trades[0].select_order(trades[0].entry_side, False)
    assert order is None
    order = trades[0].select_order(trades[0].exit_side, None)
    assert order is None

    # 关闭的买入订单和开放的卖出订单
    order = trades[1].select_order(trades[1].entry_side, True)
    assert order is None
    order = trades[1].select_order(trades[1].entry_side, False)
    assert order is not None
    order = trades[1].select_order(trades[1].entry_side, None)
    assert order is not None
    order = trades[1].select_order(trades[1].exit_side, True)
    assert order is None
    order = trades[1].select_order(trades[1].exit_side, False)
    assert order is not None

    # 有开放的买入订单
    order = trades[3].select_order(trades[3].entry_side, True)
    assert order is not None
    order = trades[3].select_order(trades[3].entry_side, False)
    assert order is None

    # 开放卖出订单
    order = trades[4].select_order(trades[4].entry_side, True)
    assert order is None
    order = trades[4].select_order(trades[4].entry_side, False)
    assert order is not None

    trades[4].orders[1].ft_order_side = trades[4].exit_side
    order = trades[4].select_order(trades[4].exit_side, True)
    assert order is not None

    trades[4].orders[1].ft_order_side = "stoploss"
    order = trades[4].select_order("stoploss", None)
    assert order is not None
    assert order.ft_order_side == "stoploss"


def test_Trade_object_idem():
    """测试Trade对象的一致性"""
    assert issubclass(Trade, LocalTrade)

    trade = vars(Trade)
    localtrade = vars(LocalTrade)

    excludes = (
        "delete",
        "session",
        "commit",
        "rollback",
        "query",
        "open_date",
        "get_best_pair",
        "get_overall_performance",
        "get_total_closed_profit",
        "total_open_trades_stakes",
        "get_closed_trades_without_assigned_fees",
        "get_open_trades_without_assigned_fees",
        "get_trades",
        "get_trades_query",
        "get_exit_reason_performance",
        "get_enter_tag_performance",
        "get_mix_tag_performance",
        "get_trading_volume",
        "validate_string_len",
        "custom_data",
    )
    EXCLUDES2 = (
        "bt_trades",
        "bt_trades_open",
        "bt_trades_open_pp",
        "bt_open_open_trade_count",
        "bt_total_profit",
        "from_json",
    )

    # 父类（LocalTrade）应该具有相同的属性
    for item in trade:
        # 排除私有属性和open_date（因为它没有分配默认值）
        if not item.startswith("_") and item not in excludes:
            assert item in localtrade

    # 如果只添加了列而没有相应的父字段，则失败
    for item in localtrade:
        if (
            not item.startswith("__")
            and item not in EXCLUDES2
            and type(getattr(LocalTrade, item)) not in (property, FunctionType)
        ):
            assert item in trade


@pytest.mark.usefixtures("init_persistence")
def test_trade_truncates_string_fields():
    """测试交易截断字符串字段"""
    trade = Trade(
        pair="ADA/USDT",
        stake_amount=20.0,
        amount=30.0,
        open_rate=2.0,
        open_date=datetime.now(timezone.utc) - timedelta(minutes=20),
        fee_open=0.001,
        fee_close=0.001,
        exchange="binance",
        leverage=1.0,
        trading_mode="futures",
        enter_tag="a" * CUSTOM_TAG_MAX_LENGTH * 2,
        exit_reason="b" * CUSTOM_TAG_MAX_LENGTH * 2,
    )
    Trade.session.add(trade)
    Trade.commit()

    trade1 = Trade.session.scalars(select(Trade)).first()

    assert trade1.enter_tag == "a" * CUSTOM_TAG_MAX_LENGTH
    assert trade1.exit_reason == "b" * CUSTOM_TAG_MAX_LENGTH


def test_recalc_trade_from_orders(fee):
    """测试从订单重新计算交易"""
    o1_amount = 100
    o1_rate = 1
    o1_cost = o1_amount * o1_rate
    o1_fee_cost = o1_cost * fee.return_value
    o1_trade_val = o1_cost + o1_fee_cost

    trade = Trade(
        pair="ADA/USDT",
        stake_amount=o1_cost,
        open_date=dt_now() - timedelta(hours=2),
        amount=o1_amount,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange="binance",
        open_rate=o1_rate,
        max_rate=o1_rate,
        leverage=1,
    )

    assert fee.return_value == 0.0025
    assert trade._calc_open_trade_value(trade.amount, trade.open_rate) == o1_trade_val
    assert trade.amount == o1_amount
    assert trade.stake_amount == o1_cost
    assert trade.open_rate == o1_rate
    assert trade.open_trade_value == o1_trade_val

    # 不带订单调用不应该抛出异常并且不改变任何东西
    trade.recalc_trade_from_orders()
    assert trade.amount == o1_amount
    assert trade.stake_amount == o1_cost
    assert trade.open_rate == o1_rate
    assert trade.open_trade_value == o1_trade_val

    trade.update_fee(o1_fee_cost, "BNB", fee.return_value, "buy")

    assert len(trade.orders) == 0

    # 检查1个订单
    order1 = Order(
        ft_order_side="buy",
        ft_pair=trade.pair,
        ft_is_open=False,
        status="closed",
        symbol=trade.pair,
        order_type="market",
        side="buy",
        price=o1_rate,
        average=o1_rate,
        filled=o1_amount,
        remaining=0,
        cost=o1_amount,
        order_date=trade.open_date,
        order_filled_date=trade.open_date,
    )
    trade.orders.append(order1)
    trade.recalc_trade_from_orders()

    # 使用单个初始订单调用recalc不应该改变任何东西
    assert trade.amount == o1_amount
    assert trade.stake_amount == o1_amount
    assert trade.open_rate == o1_rate
    assert trade.fee_open_cost == o1_fee_cost
    assert trade.open_trade_value == o1_trade_val

    # 一个额外的调整/DCA订单
    o2_amount = 125
    o2_rate = 0.9
    o2_cost = o2_amount * o2_rate
    o2_fee_cost = o2_cost * fee.return_value
    o2_trade_val = o2_cost + o2_fee_cost

    order2 = Order(
        ft_order_side="buy",
        ft_pair=trade.pair,
        ft_is_open=False,
        status="closed",
        symbol=trade.pair,
        order_type="market",
        side="buy",
        price=o2_rate,
        average=o2_rate,
        filled=o2_amount,
        remaining=0,
        cost=o2_cost,
        order_date=dt_now() - timedelta(hours=1),
        order_filled_date=dt_now() - timedelta(hours=1),
    )
    trade.orders.append(order2)
    trade.recalc_trade_from_orders()

    # 验证交易现在具有新的平均开盘价和总价值
    avg_price = (o1_cost + o2_cost) / (o1_amount + o2_amount)
    assert trade.amount == o1_amount + o2_amount
    assert trade.stake_amount == o1_amount + o2_cost
    assert trade.open_rate == avg_price
    assert trade.fee_open_cost == o1_fee_cost + o2_fee_cost
    assert trade.open_trade_value == o1_trade_val + o2_trade_val

    # 让我们尝试多个额外的订单
    o3_amount = 150
    o3_rate = 0.85
    o3_cost = o3_amount * o3_rate
    o3_fee_cost = o3_cost * fee.return_value
    o3_trade_val = o3_cost + o3_fee_cost

    order3 = Order(
        ft_order_side="buy",
        ft_pair=trade.pair,
        ft_is_open=False,
        status="closed",
        symbol=trade.pair,
        order_type="market",
        side="buy",
        price=o3_rate,
        average=o3_rate,
        filled=o3_amount,
        remaining=0,
        cost=o3_cost,
        order_date=dt_now() - timedelta(hours=1),
        order_filled_date=dt_now() - timedelta(hours=1),
    )
    trade.orders.append(order3)
    trade.recalc_trade_from_orders()

    # 验证总和仍然正确且开盘价是平均的
    avg_price = (o1_cost + o2_cost + o3_cost) / (o1_amount + o2_amount + o3_amount)
    assert trade.amount == o1_amount + o2_amount + o3_amount
    assert trade.stake_amount == o1_cost + o2_cost + o3_cost
    assert trade.open_rate == avg_price
    assert pytest.approx(trade.fee_open_cost) == o1_fee_cost + o2_fee_cost + o3_fee_cost
    assert pytest.approx(trade.open_trade_value) == o1_trade_val + o2_trade_val + o3_trade_val

    # 只是为了确保忽略完整的卖出订单，让我们再计算一次。

    sell1 = Order(
        ft_order_side="sell",
        ft_pair=trade.pair,
        ft_is_open=False,
        status="closed",
        symbol=trade.pair,
        order_type="market",
        side="sell",
        price=avg_price + 0.95,
        average=avg_price + 0.95,
        filled=o1_amount + o2_amount + o3_amount,
        remaining=0,
        cost=o1_cost + o2_cost + o3_cost,
        order_date=trade.open_date,
        order_filled_date=trade.open_date,
    )
    trade.orders.append(sell1)
    trade.recalc_trade_from_orders()

    assert trade.amount == o1_amount + o2_amount + o3_amount
    assert trade.stake_amount == o1_cost + o2_cost + o3_cost
    assert trade.open_rate == avg_price
    assert pytest.approx(trade.fee_open_cost) == o1_fee_cost + o2_fee_cost + o3_fee_cost
    assert pytest.approx(trade.open_trade_value) == o1_trade_val + o2_trade_val + o3_trade_val


@pytest.mark.usefixtures("init_persistence")
def test_recalc_trade_from_orders_kucoin():
    """测试从订单重新计算交易（KuCoin案例）"""
    # 取自 https://github.com/freqtrade/freqtrade/issues/9346
    o1_amount = 11511963.8634448908
    o2_amount = 11750101.7743937783
    o3_amount = 23262065.6378386617  # 退出金额 - 几乎没有平衡

    res = o1_amount + o2_amount - o3_amount
    assert res > 0.0
    assert res < 0.1
    o1_rate = 0.000029901
    o2_rate = 0.000029295
    o3_rate = 0.000029822

    o1_cost = o1_amount * o1_rate

    trade = Trade(
        pair="FLOKI/USDT",
        stake_amount=o1_cost,
        open_date=dt_now() - timedelta(hours=2),
        amount=o1_amount,
        fee_open=0.001,
        fee_close=0.001,
        exchange="binance",
        open_rate=o1_rate,
        max_rate=o1_rate,
        leverage=1,
    )
    # 检查1个订单
    order1 = Order(
        ft_order_side="buy",
        ft_pair=trade.pair,
        ft_is_open=False,
        status="closed",
        symbol=trade.pair,
        order_type="market",
        side="buy",
        price=o1_rate,
        average=o1_rate,
        filled=o1_amount,
        remaining=0,
        cost=o1_cost,
        order_date=trade.open_date,
        order_filled_date=trade.open_date,
    )
    trade.orders.append(order1)
    order2 = Order(
        ft_order_side="buy",
        ft_pair=trade.pair,
        ft_is_open=False,
        status="closed",
        symbol=trade.pair,
        order_type="market",
        side="buy",
        price=o2_rate,
        average=o2_rate,
        filled=o2_amount,
        remaining=0,
        cost=o2_amount * o2_rate,
        order_date=trade.open_date,
        order_filled_date=trade.open_date,
    )
    trade.orders.append(order2)
    trade.recalc_trade_from_orders()
    assert trade.amount == o1_amount + o2_amount
    profit = trade.calculate_profit(o3_rate)
    assert profit.profit_abs == pytest.approx(3.90069871)
    assert profit.profit_ratio == pytest.approx(0.00566035)

    order3 = Order(
        ft_order_side="sell",
        ft_pair=trade.pair,
        ft_is_open=False,
        status="closed",
        symbol=trade.pair,
        order_type="market",
        side="sell",
        price=o3_rate,
        average=o3_rate,
        filled=o3_amount,
        remaining=0,
        cost=o2_amount * o2_rate,
        order_date=trade.open_date,
        order_filled_date=trade.open_date,
    )

    trade.orders.append(order3)
    trade.update_trade(order3)
    assert trade.is_open is False
    # 交易正确关闭 - 但留下了最小金额。
    assert trade.amount == 8e-09
    assert pytest.approx(trade.close_profit_abs) == 3.90069871
    assert pytest.approx(trade.close_profit) == 0.00566035


@pytest.mark.parametrize("is_short", [True, False])
def test_recalc_trade_from_orders_ignores_bad_orders(fee, is_short):
    """测试从订单重新计算交易时忽略坏订单"""
    o1_amount = 100
    o1_rate = 1
    o1_cost = o1_amount * o1_rate
    o1_fee_cost = o1_cost * fee.return_value
    o1_trade_val = o1_cost - o1_fee_cost if is_short else o1_cost + o1_fee_cost
    entry_side = "sell" if is_short else "buy"
    exit_side = "buy" if is_short else "sell"

    trade = Trade(
        pair="ADA/USDT",
        stake_amount=o1_cost,
        open_date=dt_now() - timedelta(hours=2),
        amount=o1_amount,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange="binance",
        open_rate=o1_rate,
        max_rate=o1_rate,
        is_short=is_short,
        leverage=1.0,
    )
    trade.update_fee(o1_fee_cost, "BNB", fee.return_value, entry_side)
    # 检查1个订单
    order1 = Order(
        ft_order_side=entry_side,
        ft_pair=trade.pair,
        ft_is_open=False,
        status="closed",
        symbol=trade.pair,
        order_type="market",
        side=entry_side,
        price=o1_rate,
        average=o1_rate,
        filled=o1_amount,
        remaining=0,
        cost=o1_amount,
        order_date=trade.open_date,
        order_filled_date=trade.open_date,
    )
    trade.orders.append(order1)
    trade.recalc_trade_from_orders()

    # 使用单个初始订单调用recalc不应该改变任何东西
    assert trade.amount == o1_amount
    assert trade.stake_amount == o1_amount
    assert trade.open_rate == o1_rate
    assert trade.fee_open_cost == o1_fee_cost
    assert trade.open_trade_value == o1_trade_val
    assert trade.nr_of_successful_entries == 1

    order2 = Order(
        ft_order_side=entry_side,
        ft_pair=trade.pair,
        ft_is_open=True,
        status="open",
        symbol=trade.pair,
        order_type="market",
        side=entry_side,
        price=o1_rate,
        average=o1_rate,
        filled=o1_amount,
        remaining=0,
        cost=o1_cost,
        order_date=dt_now() - timedelta(hours=1),
        order_filled_date=dt_now() - timedelta(hours=1),
    )
    trade.orders.append(order2)
    trade.recalc_trade_from_orders()

    # 验证交易价值没有改变
    assert trade.amount == o1_amount
    assert trade.stake_amount == o1_amount
    assert trade.open_rate == o1_rate
    assert trade.fee_open_cost == o1_fee_cost
    assert trade.open_trade_value == o1_trade_val
    assert trade.nr_of_successful_entries == 1

    # 让我们尝试一些其他订单
    order3 = Order(
        ft_order_side=entry_side,
        ft_pair=trade.pair,
        ft_is_open=False,
        status="cancelled",
        symbol=trade.pair,
        order_type="market",
        side=entry_side,
        price=1,
        average=2,
        filled=0,
        remaining=4,
        cost=5,
        order_date=dt_now() - timedelta(hours=1),
        order_filled_date=dt_now() - timedelta(hours=1),
    )
    trade.orders.append(order3)
    trade.recalc_trade_from_orders()

    # 验证订单值仍然忽略订单2和3
    assert trade.amount == o1_amount
    assert trade.stake_amount == o1_amount
    assert trade.open_rate == o1_rate
    assert trade.fee_open_cost == o1_fee_cost
    assert trade.open_trade_value == o1_trade_val
    assert trade.nr_of_successful_entries == 1

    order4 = Order(
        ft_order_side=entry_side,
        ft_pair=trade.pair,
        ft_is_open=False,
        status="closed",
        symbol=trade.pair,
        order_type="market",
        side=entry_side,
        price=o1_rate,
        average=o1_rate,
        filled=o1_amount,
        remaining=0,
        cost=o1_cost,
        order_date=dt_now() - timedelta(hours=1),
        order_filled_date=dt_now() - timedelta(hours=1),
    )
    trade.orders.append(order4)
    trade.recalc_trade_from_orders()

    # 验证交易价值已更改
    assert trade.amount == 2 * o1_amount
    assert trade.stake_amount == 2 * o1_amount
    assert trade.open_rate == o1_rate
    assert trade.fee_open_cost == trade.nr_of_successful_entries * o1_fee_cost
    assert trade.open_trade_value == 2 * o1_trade_val
    assert trade.nr_of_successful_entries == 2

    # 减少仓位 - 这将再次减少数量。
    sell1 = Order(
        ft_order_side=exit_side,
        ft_pair=trade.pair,
        ft_is_open=False,
        status="closed",
        symbol=trade.pair,
        order_type="market",
        side=exit_side,
        price=4,
        average=3,
        filled=o1_amount,
        remaining=1,
        cost=5,
        order_date=trade.open_date,
        order_filled_date=trade.open_date,
    )
    trade.orders.append(sell1)
    trade.recalc_trade_from_orders()

    assert trade.amount == o1_amount
    assert trade.stake_amount == o1_amount
    assert trade.open_rate == o1_rate
    assert trade.fee_open_cost == trade.nr_of_successful_entries * o1_fee_cost
    assert trade.open_trade_value == o1_trade_val
    assert trade.nr_of_successful_entries == 2

    # 检查1个订单
    order_noavg = Order(
        ft_order_side=entry_side,
        ft_pair=trade.pair,
        ft_is_open=False,
        status="closed",
        symbol=trade.pair,
        order_type="market",
        side=entry_side,
        price=o1_rate,
        average=None,
        filled=o1_amount,
        remaining=0,
        cost=o1_amount,
        order_date=trade.open_date,
        order_filled_date=trade.open_date,
    )
    trade.orders.append(order_noavg)
    trade.recalc_trade_from_orders()

    # 使用单个初始订单调用recalc不应该改变任何东西
    assert trade.amount == 2 * o1_amount
    assert trade.stake_amount == 2 * o1_amount
    assert trade.open_rate == o1_rate
    assert trade.fee_open_cost == trade.nr_of_successful_entries * o1_fee_cost
    assert trade.open_trade_value == 2 * o1_trade_val
    assert trade.nr_of_successful_entries == 3


@pytest.mark.usefixtures("init_persistence")
def test_select_filled_orders(fee):
    """测试选择已填充的订单"""
    create_mock_trades(fee)

    trades = Trade.get_trades().all()

    # 关闭的买入订单，没有卖出订单
    orders = trades[0].select_filled_orders("buy")
    assert isinstance(orders, list)
    assert len(orders) == 0

    orders = trades[0].select_filled_orders("sell")
    assert orders is not None
    assert len(orders) == 0

    # 关闭的买入订单和关闭的卖出订单
    orders = trades[1].select_filled_orders("buy")
    assert isinstance(orders, list)
    assert len(orders) == 1
    order = orders[0]
    assert order.amount > 0
    assert order.filled > 0
    assert order.side == "buy"
    assert order.ft_order_side == "buy"
    assert order.status == "closed"

    orders = trades[1].select_filled_orders("sell")
    assert isinstance(orders, list)
    assert len(orders) == 1

    # 有开放的买入订单
    orders = trades[3].select_filled_orders("buy")
    assert isinstance(orders, list)
    assert len(orders) == 0
    orders = trades[3].select_filled_orders("sell")
    assert isinstance(orders, list)
    assert len(orders) == 0

    # 开放卖出订单
    orders = trades[4].select_filled_orders("buy")
    assert isinstance(orders, list)
    assert len(orders) == 1
    orders = trades[4].select_filled_orders("sell")
    assert isinstance(orders, list)
    assert len(orders) == 0


@pytest.mark.usefixtures("init_persistence")
def test_select_filled_orders_usdt(fee):
    """测试选择已填充的USDT订单"""
    create_mock_trades_usdt(fee)

    trades = Trade.get_trades().all()

    # 关闭的买入订单，没有卖出订单
    orders = trades[0].select_filled_orders("buy")
    assert isinstance(orders, list)
    assert len(orders) == 1
    assert orders[0].amount == 2.0
    assert orders[0].filled == 2.0
    assert orders[0].side == "buy"
    assert orders[0].price == 10.0
    assert orders[0].stake_amount == 20
    assert orders[0].stake_amount_filled == 20

    orders = trades[3].select_filled_orders("buy")
    assert isinstance(orders, list)
    assert len(orders) == 0
    orders = trades[3].select_filled_or_open_orders()
    assert isinstance(orders, list)
    assert len(orders) == 1
    assert orders[0].price == 2.0
    assert orders[0].amount == 10
    assert orders[0].filled == 0
    assert orders[0].stake_amount == 20
    assert orders[0].stake_amount_filled == 0


@pytest.mark.usefixtures("init_persistence")
def test_order_to_ccxt(limit_buy_order_open, limit_sell_order_usdt_open):
    """测试订单转换为CCXT格式"""
    order = Order.parse_from_ccxt_object(limit_buy_order_open, "mocked", "buy")
    order.ft_trade_id = 1
    order.session.add(order)
    Order.session.commit()

    order_resp = Order.order_by_id(limit_buy_order_open["id"])
    assert order_resp

    raw_order = order_resp.to_ccxt_object()
    del raw_order["fee"]
    del raw_order["datetime"]
    del raw_order["info"]
    assert raw_order.get("stopPrice") is None
    raw_order.pop("stopPrice", None)
    del limit_buy_order_open["datetime"]
    assert raw_order == limit_buy_order_open

    order1 = Order.parse_from_ccxt_object(limit_sell_order_usdt_open, "mocked", "sell")
    order1.ft_order_side = "stoploss"
    order1.stop_price = order1.price * 0.9
    order1.ft_trade_id = 1
    order1.session.add(order1)
    Order.session.commit()

    order_resp1 = Order.order_by_id(limit_sell_order_usdt_open["id"])
    raw_order1 = order_resp1.to_ccxt_object()

    assert raw_order1.get("stopPrice") is not None


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize(
    "data",
    [
        # 元组1 - 方向、数量、价格
        # 元组2 - 数量、开盘价、投入金额、累计利润、已实现利润、相对利润
        {
            "orders": [
                (("buy", 100, 10), (100.0, 10.0, 1000.0, 0.0, None, None)),
                (("buy", 100, 15), (200.0, 12.5, 2500.0, 0.0, None, None)),
                (("sell", 50, 12), (150.0, 12.5, 1875.0, -25.0, -25.0, -0.01)),
                (("sell", 100, 20), (50.0, 12.5, 625.0, 725.0, 750.0, 0.29)),
                (("sell", 50, 5), (50.0, 12.5, 625.0, 350.0, -375.0, 0.14)),
            ],
            "end_profit": 350.0,
            "end_profit_ratio": 0.14,
            "fee": 0.0,
        },
        {
            "orders": [
                (("buy", 100, 10), (100.0, 10.0, 1000.0, 0.0, None, None)),
                (("buy", 100, 15), (200.0, 12.5, 2500.0, 0.0, None, None)),
                (("sell", 50, 12), (150.0, 12.5, 1875.0, -28.0625, -28.0625, -0.011197)),
                (("sell", 100, 20), (50.0, 12.5, 625.0, 713.8125, 741.875, 0.2848129)),
                (("sell", 50, 5), (50.0, 12.5, 625.0, 336.625, -377.1875, 0.1343142)),
            ],
            "end_profit": 336.625,
            "end_profit_ratio": 0.1343142,
            "fee": 0.0025,
        },
        {
            "orders": [
                (("buy", 100, 3), (100.0, 3.0, 300.0, 0.0, None, None)),
                (("buy", 100, 7), (200.0, 5.0, 1000.0, 0.0, None, None)),
                (("sell", 100, 11), (100.0, 5.0, 500.0, 596.0, 596.0, 0.5945137)),
                (("buy", 150, 15), (250.0, 11.0, 2750.0, 596.0, 596.0, 0.5945137)),
                (("sell", 100, 19), (150.0, 11.0, 1650.0, 1388.5, 792.5, 0.4261653)),
                (("sell", 150, 23), (150.0, 11.0, 1650.0, 3175.75, 1787.25, 0.9747170)),
            ],
            "end_profit": 3175.75,
            "end_profit_ratio": 0.9747170,
            "fee": 0.0025,
        },
        {
            # 上面的测试，没有费用
            "orders": [
                (("buy", 100, 3), (100.0, 3.0, 300.0, 0.0, None, None)),
                (("buy", 100, 7), (200.0, 5.0, 1000.0, 0.0, None, None)),
                (("sell", 100, 11), (100.0, 5.0, 500.0, 600.0, 600.0, 0.6)),
                (("buy", 150, 15), (250.0, 11.0, 2750.0, 600.0, 600.0, 0.6)),
                (("sell", 100, 19), (150.0, 11.0, 1650.0, 1400.0, 800.0, 0.43076923)),
                (("sell", 150, 23), (150.0, 11.0, 1650.0, 3200.0, 1800.0, 0.98461538)),
            ],
            "end_profit": 3200.0,
            "end_profit_ratio": 0.98461538,
            "fee": 0.0,
        },
        {
            "orders": [
                (("buy", 100, 8), (100.0, 8.0, 800.0, 0.0, None, None)),
                (("buy", 100, 9), (200.0, 8.5, 1700.0, 0.0, None, None)),
                (("sell", 100, 10), (100.0, 8.5, 850.0, 150.0, 150.0, 0.08823529)),
                (("buy", 150, 11), (250.0, 10, 2500.0, 150.0, 150.0, 0.08823529)),
                (("sell", 100, 12), (150.0, 10.0, 1500.0, 350.0, 200.0, 0.1044776)),
                (("sell", 150, 14), (150.0, 10.0, 1500.0, 950.0, 600.0, 0.283582)),
            ],
            "end_profit": 950.0,
            "end_profit_ratio": 0.283582,
            "fee": 0.0,
        },
    ],
)
def test_recalc_trade_from_orders_dca(data) -> None:
    """测试从订单重新计算DCA交易"""
    pair = "ETH/USDT"
    trade = Trade(
        id=2,
        pair=pair,
        stake_amount=1000,
        open_rate=data["orders"][0][0][2],
        amount=data["orders"][0][0][1],
        is_open=True,
        open_date=dt_now(),
        fee_open=data["fee"],
        fee_close=data["fee"],
        exchange="binance",
        is_short=False,
        leverage=1.0,
        trading_mode=TradingMode.SPOT,
        price_precision=0.001,
        precision_mode_price=TICK_SIZE,
    )
    Trade.session.add(trade)

    for idx, (order, result) in enumerate(data["orders"]):
        amount = order[1]
        price = order[2]

        order_obj = Order(
            ft_order_side=order[0],
            ft_pair=trade.pair,
            order_id=f"order_{order[0]}_{idx}",
            ft_is_open=False,
            ft_amount=amount,
            ft_price=price,
            status="closed",
            symbol=trade.pair,
            order_type="market",
            side=order[0],
            price=price,
            average=price,
            filled=amount,
            remaining=0,
            cost=amount * price,
            order_date=dt_now() - timedelta(hours=10 + idx),
            order_filled_date=dt_now() - timedelta(hours=10 + idx),
        )
        trade.orders.append(order_obj)
        trade.recalc_trade_from_orders()
        Trade.commit()

        orders1 = Order.session.scalars(select(Order)).all()
        assert orders1
        assert len(orders1) == idx + 1

        trade = Trade.session.scalars(select(Trade)).first()
        assert trade
        assert len(trade.orders) == idx + 1
        if idx < len(data) - 1:
            assert trade.is_open is True
        assert not trade.has_open_orders
        assert trade.amount == result[0]
        assert trade.open_rate == result[1]
        assert trade.stake_amount == result[2]
        assert pytest.approx(trade.realized_profit) == result[3]
        assert pytest.approx(trade.close_profit_abs) == result[4]
        assert pytest.approx(trade.close_profit) == result[5]

    trade.close(price)
    assert pytest.approx(trade.close_profit_abs) == data["end_profit"]
    assert pytest.approx(trade.close_profit) == data["end_profit_ratio"]
    assert not trade.is_open
    trade = Trade.session.scalars(select(Trade)).first()
    assert trade
    assert not trade.has_open_orders