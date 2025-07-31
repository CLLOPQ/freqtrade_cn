from unittest.mock import MagicMock

import pytest

from freqtrade.enums.marginmode import  MarginMode
from freqtrade.leverage.liquidation_price import update_liquidation_prices


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("margin_mode", [MarginMode.CROSS, MarginMode.ISOLATED])
def test_update_liquidation_prices(mocker, margin_mode, dry_run):
    # 高度模拟的测试 - 仅测试函数的逻辑
    # 在逐仓模式下更新交易的清算价格
    # 在全仓模式下更新所有交易的清算价格
    exchange = MagicMock()
    exchange.margin_mode = margin_mode
    wallets = MagicMock()
    trade_mock = MagicMock()

    mocker.patch("freqtrade.persistence.Trade.get_open_trades", return_value=[trade_mock])

    update_liquidation_prices(
        trade=trade_mock,
        exchange=exchange,
        wallets=wallets,
        stake_currency="USDT",
        dry_run=dry_run,
    )

    assert trade_mock.set_liquidation_price.call_count == 1

    assert wallets.get_collateral.call_count == (
        0 if margin_mode == MarginMode.ISOLATED or not dry_run else 1
    )

    # 测试多个交易的情况
    trade_mock.reset_mock()
    trade_mock_2 = MagicMock()

    mocker.patch(
        "freqtrade.persistence.Trade.get_open_trades", return_value=[trade_mock, trade_mock_2]
    )

    update_liquidation_prices(
        trade=trade_mock,
        exchange=exchange,
        wallets=wallets,
        stake_currency="USDT",
        dry_run=dry_run,
    )
    # 仅在全仓模式下更新第二个交易
    assert trade_mock_2.set_liquidation_price.call_count == (
        1 if margin_mode == MarginMode.CROSS else 0
    )
    assert trade_mock.set_liquidation_price.call_count == 1

    assert wallets.call_count == 0 if not dry_run else 1