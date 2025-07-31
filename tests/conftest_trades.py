from datetime import datetime, timedelta, timezone

from freqtrade.persistence.models import Order, Trade


MOCK_TRADE_COUNT = 6


def entry_side(is_short: bool):
    """根据是否为做空交易返回对应的入场方向"""
    return "sell" if is_short else "buy"


def exit_side(is_short: bool):
    """根据是否为做空交易返回对应的出场方向"""
    return "buy" if is_short else "sell"


def direc(is_short: bool):
    """根据是否为做空交易返回交易方向字符串"""
    return "short" if is_short else "long"


def mock_order_1(is_short: bool):
    """创建模拟入场订单1"""
    return {
        "id": f"1234_{direc(is_short)}",
        "symbol": "ETH/BTC",
        "status": "open",
        "side": entry_side(is_short),
        "type": "limit",
        "price": 0.123,
        "average": 0.123,
        "amount": 123.0,
        "filled": 50.0,
        "cost": 15.129,
        "remaining": 123.0 - 50.0,
    }


def mock_trade_1(fee, is_short: bool):
    """创建模拟交易1（进行中）"""
    trade = Trade(
        pair="ETH/BTC",
        stake_amount=0.001,
        amount=50.0,
        amount_requested=123.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        is_open=True,
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=17),
        open_rate=0.123,
        exchange="binance",
        strategy="StrategyTestV3",
        timeframe=5,
        is_short=is_short,
    )
    # 添加入场订单
    o = Order.parse_from_ccxt_object(mock_order_1(is_short), "ETH/BTC", entry_side(is_short))
    trade.orders.append(o)
    return trade


def mock_order_2(is_short: bool):
    """创建模拟入场订单2"""
    return {
        "id": f"1235_{direc(is_short)}",
        "symbol": "ETC/BTC",
        "status": "closed",
        "side": entry_side(is_short),
        "type": "limit",
        "price": 0.123,
        "amount": 123.0,
        "filled": 123.0,
        "cost": 15.129,
        "remaining": 0.0,
    }


def mock_order_2_sell(is_short: bool):
    """创建模拟出场订单2"""
    return {
        "id": f"12366_{direc(is_short)}",
        "symbol": "ETC/BTC",
        "status": "closed",
        "side": exit_side(is_short),
        "type": "limit",
        "price": 0.128,
        "amount": 123.0,
        "filled": 123.0,
        "cost": 15.129,
        "remaining": 0.0,
    }


def mock_trade_2(fee, is_short: bool):
    """
    创建模拟交易2（已平仓）
    """
    trade = Trade(
        pair="ETC/BTC",
        stake_amount=0.001,
        amount=123.0,
        amount_requested=123.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=0.123,
        close_rate=0.128,
        close_profit=-0.005 if is_short else 0.005,
        close_profit_abs=-0.005584127 if is_short else 0.000584127,
        exchange="binance",
        is_open=False,
        strategy="StrategyTestV3",
        timeframe=5,
        enter_tag="TEST1",
        exit_reason="sell_signal",
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=20),
        close_date=datetime.now(tz=timezone.utc) - timedelta(minutes=2),
        is_short=is_short,
    )
    # 添加入场订单
    o = Order.parse_from_ccxt_object(mock_order_2(is_short), "ETC/BTC", entry_side(is_short))
    trade.orders.append(o)
    # 添加出场订单
    o = Order.parse_from_ccxt_object(mock_order_2_sell(is_short), "ETC/BTC", exit_side(is_short))
    trade.orders.append(o)
    return trade


def mock_order_3(is_short: bool):
    """创建模拟入场订单3"""
    return {
        "id": f"41231a12a_{direc(is_short)}",
        "symbol": "XRP/BTC",
        "status": "closed",
        "side": entry_side(is_short),
        "type": "limit",
        "price": 0.05,
        "amount": 123.0,
        "filled": 123.0,
        "cost": 15.129,
        "remaining": 0.0,
    }


def mock_order_3_sell(is_short: bool):
    """创建模拟出场订单3"""
    return {
        "id": f"41231a666a_{direc(is_short)}",
        "symbol": "XRP/BTC",
        "status": "closed",
        "side": exit_side(is_short),
        "type": "stop_loss_limit",
        "price": 0.06,
        "average": 0.06,
        "amount": 123.0,
        "filled": 123.0,
        "cost": 15.129,
        "remaining": 0.0,
    }


def mock_trade_3(fee, is_short: bool):
    """
    创建模拟交易3（已平仓）
    """
    trade = Trade(
        pair="XRP/BTC",
        stake_amount=0.001,
        amount=123.0,
        amount_requested=123.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=0.05,
        close_rate=0.06,
        close_profit=-0.01 if is_short else 0.01,
        close_profit_abs=-0.001155 if is_short else 0.000155,
        exchange="binance",
        is_open=False,
        strategy="StrategyTestV3",
        timeframe=5,
        exit_reason="roi",
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=20),
        close_date=datetime.now(tz=timezone.utc),
        is_short=is_short,
    )
    # 添加入场订单
    o = Order.parse_from_ccxt_object(mock_order_3(is_short), "XRP/BTC", entry_side(is_short))
    trade.orders.append(o)
    # 添加出场订单
    o = Order.parse_from_ccxt_object(mock_order_3_sell(is_short), "XRP/BTC", exit_side(is_short))
    trade.orders.append(o)
    return trade


def mock_order_4(is_short: bool):
    """创建模拟入场订单4"""
    return {
        "id": f"prod_buy_{direc(is_short)}_12345",
        "symbol": "ETC/BTC",
        "status": "open",
        "side": entry_side(is_short),
        "type": "limit",
        "price": 0.123,
        "amount": 123.0,
        "filled": 0.0,
        "cost": 15.129,
        "remaining": 123.0,
    }


def mock_trade_4(fee, is_short: bool):
    """
    创建模拟交易4（进行中）
    """
    trade = Trade(
        pair="ETC/BTC",
        stake_amount=0.001,
        amount=0.0,
        amount_requested=124.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=14),
        is_open=True,
        open_rate=0.123,
        exchange="binance",
        strategy="StrategyTestV3",
        timeframe=5,
        is_short=is_short,
        stop_loss_pct=0.10,
    )
    # 添加入场订单
    o = Order.parse_from_ccxt_object(mock_order_4(is_short), "ETC/BTC", entry_side(is_short))
    trade.orders.append(o)
    return trade


def mock_order_5(is_short: bool):
    """创建模拟入场订单5"""
    return {
        "id": f"prod_buy_{direc(is_short)}_3455",
        "symbol": "XRP/BTC",
        "status": "closed",
        "side": entry_side(is_short),
        "type": "limit",
        "price": 0.123,
        "amount": 123.0,
        "filled": 123.0,
        "cost": 15.129,
        "remaining": 0.0,
    }


def mock_order_5_stoploss(is_short: bool):
    """创建模拟止损订单5"""
    return {
        "id": f"prod_stoploss_{direc(is_short)}_3455",
        "symbol": "XRP/BTC",
        "status": "open",
        "side": exit_side(is_short),
        "type": "stop_loss_limit",
        "price": 0.123,
        "amount": 123.0,
        "filled": 0.0,
        "cost": 0.0,
        "remaining": 123.0,
    }


def mock_trade_5(fee, is_short: bool):
    """
    创建模拟交易5（带有止损订单的进行中交易）
    """
    trade = Trade(
        pair="XRP/BTC",
        stake_amount=0.001,
        amount=123.0,
        amount_requested=124.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=12),
        is_open=True,
        open_rate=0.123,
        exchange="binance",
        strategy="SampleStrategy",
        enter_tag="TEST1",
        timeframe=5,
        is_short=is_short,
        stop_loss_pct=0.10,
    )
    # 添加入场订单
    o = Order.parse_from_ccxt_object(mock_order_5(is_short), "XRP/BTC", entry_side(is_short))
    trade.orders.append(o)
    # 添加止损订单
    o = Order.parse_from_ccxt_object(mock_order_5_stoploss(is_short), "XRP/BTC", "stoploss")
    trade.orders.append(o)
    return trade


def mock_order_6(is_short: bool):
    """创建模拟入场订单6"""
    return {
        "id": f"prod_buy_{direc(is_short)}_6",
        "symbol": "LTC/BTC",
        "status": "closed",
        "side": entry_side(is_short),
        "type": "limit",
        "price": 0.15,
        "amount": 2.0,
        "filled": 2.0,
        "cost": 0.3,
        "remaining": 0.0,
    }


def mock_order_6_sell(is_short: bool):
    """创建模拟出场订单6"""
    return {
        "id": f"prod_sell_{direc(is_short)}_6",
        "symbol": "LTC/BTC",
        "status": "open",
        "side": exit_side(is_short),
        "type": "limit",
        "price": 0.15 if is_short else 0.20,
        "amount": 2.0,
        "filled": 0.0,
        "cost": 0.0,
        "remaining": 2.0,
    }


def mock_trade_6(fee, is_short: bool):
    """
    创建模拟交易6（带有未平仓出场订单的进行中交易）
    """
    trade = Trade(
        pair="LTC/BTC",
        stake_amount=0.001,
        amount=2.0,
        amount_requested=2.0,
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=5),
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        is_open=True,
        open_rate=0.15,
        exchange="binance",
        strategy="SampleStrategy",
        enter_tag="TEST2",
        timeframe=5,
        is_short=is_short,
    )
    # 添加入场订单
    o = Order.parse_from_ccxt_object(mock_order_6(is_short), "LTC/BTC", entry_side(is_short))
    trade.orders.append(o)
    # 添加出场订单
    o = Order.parse_from_ccxt_object(mock_order_6_sell(is_short), "LTC/BTC", exit_side(is_short))
    trade.orders.append(o)
    return trade


def short_order():
    """创建模拟做空入场订单"""
    return {
        "id": "1236",
        "symbol": "ETC/BTC",
        "status": "closed",
        "side": "sell",
        "type": "limit",
        "price": 0.123,
        "amount": 123.0,
        "filled": 123.0,
        "cost": 15.129,
        "remaining": 0.0,
    }


def exit_short_order():
    """创建模拟做空出场订单"""
    return {
        "id": "12367",
        "symbol": "ETC/BTC",
        "status": "closed",
        "side": "buy",
        "type": "limit",
        "price": 0.128,
        "amount": 123.0,
        "filled": 123.0,
        "cost": 15.744,
        "remaining": 0.0,
    }


def short_trade(fee):
    """
    创建模拟做空交易（币安10分钟做空限价交易）

    做空交易详情：
    手续费：0.25% 基础货币
    利率：0.05% 每天
    开仓价：0.123 基础货币
    平仓价：0.128 基础货币
    数量：123.0 加密货币
    保证金金额：15.129 基础货币
    借入：123.0 加密货币
    时间段：10分钟（向上取整为1/24天）
    利息：借入 * 利率 * 时间段
                = 123.0 * 0.0005 * 1/24 = 0.0025625 加密货币
    开仓价值：(数量 * 开仓价) - (数量 * 开仓价 * 手续费)
        = (123 * 0.123) - (123 * 0.123 * 0.0025)
        = 15.091177499999999
    平仓数量：数量 + 利息 = 123 + 0.0025625 = 123.0025625
    平仓价值：(平仓数量 * 平仓价) + (平仓数量 * 平仓价 * 手续费)
        = (123.0025625 * 0.128) + (123.0025625 * 0.128 * 0.0025)
        = 15.78368882
    总利润 = 开仓价值 - 平仓价值
        = 15.091177499999999 - 15.78368882
        = -0.6925113200000013
    总利润率 = 总利润 / 保证金金额
        = -0.6925113200000013 / 15.129
        = -0.04577376693766946
    """
    trade = Trade(
        pair="ETC/BTC",
        stake_amount=15.129,
        amount=123.0,
        amount_requested=123.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=0.123,
        # close_rate=0.128,
        # close_profit=-0.04577376693766946,
        # close_profit_abs=-0.6925113200000013,
        exchange="binance",
        is_open=True,
        strategy="DefaultStrategy",
        timeframe=5,
        exit_reason="sell_signal",
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=20),
        # close_date=datetime.now(tz=timezone.utc) - timedelta(minutes=2),
        is_short=True,
    )
    # 添加做空入场订单
    o = Order.parse_from_ccxt_object(short_order(), "ETC/BTC", "sell")
    trade.orders.append(o)
    # 添加做空出场订单
    o = Order.parse_from_ccxt_object(exit_short_order(), "ETC/BTC", "sell")
    trade.orders.append(o)
    return trade


def leverage_order():
    """创建模拟杠杆入场订单"""
    return {
        "id": "1237",
        "symbol": "DOGE/BTC",
        "status": "closed",
        "side": "buy",
        "type": "limit",
        "price": 0.123,
        "amount": 123.0,
        "filled": 123.0,
        "remaining": 0.0,
        "cost": 15.129,
        "leverage": 5.0,
    }


def leverage_order_sell():
    """创建模拟杠杆出场订单"""
    return {
        "id": "12368",
        "symbol": "DOGE/BTC",
        "status": "closed",
        "side": "sell",
        "type": "limit",
        "price": 0.128,
        "amount": 123.0,
        "filled": 123.0,
        "remaining": 0.0,
        "cost": 15.744,
        "leverage": 5.0,
    }


def leverage_trade(fee):
    """
    创建模拟杠杆交易（Kraken 5小时杠杆交易）

    杠杆交易详情：
    手续费：0.25% 基础货币
    利率：0.05% 每天
    开仓价：0.123 基础货币
    平仓价：0.128 基础货币
    数量：615 加密货币
    保证金金额：15.129 基础货币
    借入：60.516 基础货币
    杠杆：5倍
    时长：5小时
    利息：借入 * 利率 * 向上取整(1 + 小时/4)
                = 60.516 * 0.0005 * 向上取整(1 + 5/4) = 0.090774 基础货币
    开仓价值：(数量 * 开仓价) + (数量 * 开仓价 * 手续费)
        = (615.0 * 0.123) + (615.0 * 0.123 * 0.0025)
        = 75.83411249999999

    平仓价值：(平仓数量 * 平仓价) - (平仓数量 * 平仓价 * 手续费) - 利息
        = (615.0 * 0.128) - (615.0 * 0.128 * 0.0025) - 0.090774
        = 78.432426
    总利润 = 平仓价值 - 开仓价值
        = 78.432426 - 75.83411249999999
        = 2.5983135000000175
    总利润率 = ((平仓价值/开仓价值)-1) * 杠杆
        = ((78.432426/75.83411249999999)-1) * 5
        = 0.1713156134055116
    """
    trade = Trade(
        pair="DOGE/BTC",
        stake_amount=15.129,
        amount=615.0,
        leverage=5.0,
        amount_requested=615.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=0.123,
        close_rate=0.128,
        close_profit=0.1713156134055116,
        close_profit_abs=2.5983135000000175,
        exchange="kraken",
        is_open=False,
        strategy="DefaultStrategy",
        timeframe=5,
        exit_reason="sell_signal",
        open_date=datetime.now(tz=timezone.utc) - timedelta(minutes=300),
        close_date=datetime.now(tz=timezone.utc),
        interest_rate=0.0005,
    )
    # 添加杠杆入场订单
    o = Order.parse_from_ccxt_object(leverage_order(), "DOGE/BTC", "sell")
    trade.orders.append(o)
    # 添加杠杆出场订单
    o = Order.parse_from_ccxt_object(leverage_order_sell(), "DOGE/BTC", "sell")
    trade.orders.append(o)
    return trade