# pragma pylint: disable=missing-docstring, W0212, line-too-long, C0103, unused-argument

from copy import deepcopy
from unittest.mock import MagicMock

import pandas as pd
import pytest

from freqtrade.configuration import TimeRange
from freqtrade.data import history
from freqtrade.data.history import get_timerange
from freqtrade.enums import ExitType
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.util.datetime_helpers import dt_utc
from tests.conftest import EXMS, patch_exchange


def test_backtest_position_adjustment(default_conf, fee, mocker, testdatadir) -> None:
    default_conf["use_exit_signal"] = False
    default_conf["max_open_trades"] = 10
    mocker.patch(f"{EXMS}.get_fee", fee)
    mocker.patch(
        "freqtrade.optimize.backtesting.amount_to_contract_precision",
        lambda x, *args, **kwargs: round(x, 8),
    )
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    patch_exchange(mocker)
    default_conf.update(
        {"stake_amount": 100.0, "dry_run_wallet": 1000.0, "strategy": "StrategyTestV3"}
    )
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    pair = "UNITTEST/BTC"
    timerange = TimeRange("date", None, 1517227800, 0)
    data = history.load_data(
        datadir=testdatadir, timeframe="5m", pairs=["UNITTEST/BTC"], timerange=timerange
    )
    backtesting.strategy.position_adjustment_enable = True
    processed = backtesting.strategy.advise_all_indicators(data)
    min_date, max_date = get_timerange(processed)
    result = backtesting.backtest(
        processed=deepcopy(processed),
        start_date=min_date,
        end_date=max_date,
    )
    results = result["results"]
    assert not results.empty
    assert len(results) == 2

    expected = pd.DataFrame(
        {
            "pair": [pair, pair],
            "stake_amount": [500.0, 100.0],
            "max_stake_amount": [500.0, 100],
            "amount": [4806.87657523, 970.63960782],
            "open_date": pd.to_datetime(
                [dt_utc(2018, 1, 29, 18, 40, 0), dt_utc(2018, 1, 30, 3, 30, 0)], utc=True
            ),
            "close_date": pd.to_datetime(
                [dt_utc(2018, 1, 29, 22, 00, 0), dt_utc(2018, 1, 30, 4, 10, 0)], utc=True
            ),
            "open_rate": [0.10401764891917063, 0.10302485],
            "close_rate": [0.10453904064307624, 0.10354126528822055],
            "fee_open": [0.0025, 0.0025],
            "fee_close": [0.0025, 0.0025],
            "trade_duration": [200, 40],
            "profit_ratio": [0.0, 0.0],
            "profit_abs": [0.0, 0.0],
            "exit_reason": [ExitType.ROI.value, ExitType.ROI.value],
            "initial_stop_loss_abs": [0.0940005, 0.092722365],
            "initial_stop_loss_ratio": [-0.1, -0.1],
            "stop_loss_abs": [0.0940005, 0.092722365],
            "stop_loss_ratio": [-0.1, -0.1],
            "min_rate": [0.10370188, 0.10300000000000001],
            "max_rate": [0.10481985, 0.10388887000000001],
            "is_open": [False, False],
            "enter_tag": ["", ""],
            "leverage": [1.0, 1.0],
            "is_short": [False, False],
            "open_timestamp": [1517251200000, 1517283000000],
            "close_timestamp": [1517263200000, 1517285400000],
            "funding_fees": [0.0, 0.0],
        }
    )
    results_no = results.drop(columns=["orders"])
    pd.testing.assert_frame_equal(results_no, expected, check_exact=True)

    data_pair = processed[pair]
    assert len(results.iloc[0]["orders"]) == 6
    assert len(results.iloc[1]["orders"]) == 2

    for _, t in results.iterrows():
        ln = data_pair.loc[data_pair["date"] == t["open_date"]]
        # 检查开仓价格是否与开盘价一致
        assert ln is not None
        # 检查平仓价格是否与收盘价一致，或者在最高价和最低价之间
        ln = data_pair.loc[data_pair["date"] == t["close_date"]]
        assert round(ln.iloc[0]["open"], 6) == round(t["close_rate"], 6) or round(
            ln.iloc[0]["low"], 6
        ) < round(t["close_rate"], 6) < round(ln.iloc[0]["high"], 6)


@pytest.mark.parametrize("leverage", [1, 2])
def test_backtest_position_adjustment_detailed(default_conf, fee, mocker, leverage) -> None:
    default_conf["use_exit_signal"] = False
    mocker.patch(f"{EXMS}.get_fee", fee)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=10)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    mocker.patch(f"{EXMS}.get_max_leverage", return_value=10)
    mocker.patch(f"{EXMS}.get_maintenance_ratio_and_amt", return_value=(0.1, 0.1))
    mocker.patch("freqtrade.optimize.backtesting.Backtesting._run_funding_fees")

    patch_exchange(mocker)
    default_conf.update(
        {
            "stake_amount": 100.0,
            "dry_run_wallet": 1000.0,
            "strategy": "StrategyTestV3",
            "trading_mode": "futures",
            "margin_mode": "isolated",
        }
    )
    default_conf["pairlists"] = [{"method": "StaticPairList", "allow_inactive": True}]
    backtesting = Backtesting(default_conf)
    backtesting._can_short = True
    backtesting._set_strategy(backtesting.strategylist[0])
    pair = "XRP/USDT:USDT"
    row_enter = [
        pd.Timestamp(year=2020, month=1, day=1, hour=4, minute=0),
        2.1,  # 开盘价
        2.2,  # 最高价
        1.9,  # 最低价
        2.1,  # 收盘价
        1,  # 做多入场信号
        0,  # 做多出场信号
        0,  # 做空入场信号
        0,  # 做空出场信号
        "",  # 入场标签
        "",  # 出场标签
    ]
    # 出场行 - 略有不同的值
    row_exit = [
        pd.Timestamp(year=2020, month=1, day=1, hour=5, minute=0),
        2.2,  # 开盘价
        2.3,  # 最高价
        2.0,  # 最低价
        2.2,  # 收盘价
        1,  # 做多入场信号
        0,  # 做多出场信号
        0,  # 做空入场信号
        0,  # 做空出场信号
        "",  # 入场标签
        "",  # 出场标签
    ]
    backtesting.strategy.leverage = MagicMock(return_value=leverage)
    trade = backtesting._enter_trade(pair, row=row_enter, direction="long")
    current_time = row_enter[0].to_pydatetime()
    assert trade
    assert pytest.approx(trade.stake_amount) == 100.0
    assert pytest.approx(trade.amount) == 47.61904762 * leverage
    assert len(trade.orders) == 1
    backtesting.strategy.adjust_trade_position = MagicMock(return_value=None)
    assert pytest.approx(trade.liquidation_price) == (0.10278333 if leverage == 1 else 1.2122249)

    trade = backtesting._check_adjust_trade_for_candle(trade, row_enter, current_time)
    assert trade
    assert pytest.approx(trade.stake_amount) == 100.0
    assert pytest.approx(trade.amount) == 47.61904762 * leverage
    assert len(trade.orders) == 1
    # 增加100仓位
    backtesting.strategy.adjust_trade_position = MagicMock(return_value=(100, "PartIncrease"))

    trade = backtesting._check_adjust_trade_for_candle(trade, row_enter, current_time)

    强平价格 = 0.1038916 if leverage == 1 else 1.2127791
    assert trade
    assert pytest.approx(trade.stake_amount) == 200.0
    assert pytest.approx(trade.amount) == 95.23809524 * leverage
    assert len(trade.orders) == 2
    assert trade.orders[-1].ft_order_tag == "PartIncrease"
    assert pytest.approx(trade.liquidation_price) == 强平价格

    # 减少的仓位超过现有仓位 - 交易无变化
    backtesting.strategy.adjust_trade_position = MagicMock(return_value=-500)
    current_time = row_exit[0].to_pydatetime()

    trade = backtesting._check_adjust_trade_for_candle(trade, row_exit, current_time)

    assert trade
    assert pytest.approx(trade.stake_amount) == 200.0
    assert pytest.approx(trade.amount) == 95.23809524 * leverage
    assert len(trade.orders) == 2
    assert trade.nr_of_successful_entries == 2
    assert pytest.approx(trade.liquidation_price) == 强平价格

    # 减少50仓位
    backtesting.strategy.adjust_trade_position = MagicMock(return_value=(-100, "partDecrease"))
    trade = backtesting._check_adjust_trade_for_candle(trade, row_exit, current_time)

    assert trade
    assert pytest.approx(trade.stake_amount) == 100.0
    assert pytest.approx(trade.amount) == 47.61904762 * leverage
    assert len(trade.orders) == 3
    assert trade.orders[-1].ft_order_tag == "partDecrease"
    assert trade.nr_of_successful_entries == 2
    assert trade.nr_of_successful_exits == 1
    assert pytest.approx(trade.liquidation_price) == 强平价格

    # 调整至低于最小值
    backtesting.strategy.adjust_trade_position = MagicMock(return_value=-99)
    trade = backtesting._check_adjust_trade_for_candle(trade, row_exit, current_time)

    assert trade
    assert pytest.approx(trade.stake_amount) == 100.0
    assert pytest.approx(trade.amount) == 47.61904762 * leverage
    assert len(trade.orders) == 3
    assert trade.nr_of_successful_entries == 2
    assert trade.nr_of_successful_exits == 1
    assert pytest.approx(trade.liquidation_price) == 强平价格

    # 调整以平仓
    backtesting.strategy.adjust_trade_position = MagicMock(return_value=-trade.stake_amount)
    trade = backtesting._check_adjust_trade_for_candle(trade, row_exit, current_time)
    assert trade.is_open is False