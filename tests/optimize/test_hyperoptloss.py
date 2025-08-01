from datetime import datetime
from unittest.mock import MagicMock

import pytest

from freqtrade.exceptions import OperationalException
from freqtrade.optimize.hyperopt_loss.hyperopt_loss_short_trade_dur import ShortTradeDurHyperOptLoss
from freqtrade.resolvers.hyperopt_resolver import HyperOptLossResolver


def test_hyperoptlossresolver_noname(default_conf):
    with pytest.raises(
        OperationalException,
        match="未设置超参数优化损失函数。请使用 `--hyperopt-loss` 指定要使用的超参数优化损失类。",
    ):
        HyperOptLossResolver.load_hyperoptloss(default_conf)


def test_hyperoptlossresolver(mocker, default_conf) -> None:
    hl = ShortTradeDurHyperOptLoss
    mocker.patch(
        "freqtrade.resolvers.hyperopt_resolver.HyperOptLossResolver.load_object",
        MagicMock(return_value=hl()),
    )
    default_conf.update({"hyperopt_loss": "SharpeHyperOptLossDaily"})
    x = HyperOptLossResolver.load_hyperoptloss(default_conf)
    assert hasattr(x, "hyperopt_loss_function")


def test_hyperoptlossresolver_wrongname(default_conf) -> None:
    default_conf.update({"hyperopt_loss": "NonExistingLossClass"})

    with pytest.raises(OperationalException, match=r"无法加载超参数优化损失类.*"):
        HyperOptLossResolver.load_hyperoptloss(default_conf)


def test_loss_calculation_prefer_correct_trade_count(hyperopt_conf, hyperopt_results) -> None:
    hyperopt_conf.update({"hyperopt_loss": "ShortTradeDurHyperOptLoss"})
    hl = HyperOptLossResolver.load_hyperoptloss(hyperopt_conf)
    correct = hl.hyperopt_loss_function(
        results=hyperopt_results,
        trade_count=600,
        min_date=datetime(2019, 1, 1),
        max_date=datetime(2019, 5, 1),
        config=hyperopt_conf,
        processed=None,
        backtest_stats={"profit_total": hyperopt_results["profit_abs"].sum()},
        starting_balance=hyperopt_conf["dry_run_wallet"],
    )
    over = hl.hyperopt_loss_function(
        results=hyperopt_results,
        trade_count=600 + 100,
        min_date=datetime(2019, 1, 1),
        max_date=datetime(2019, 5, 1),
        config=hyperopt_conf,
        processed=None,
        backtest_stats={"profit_total": hyperopt_results["profit_abs"].sum()},
        starting_balance=hyperopt_conf["dry_run_wallet"],
    )
    under = hl.hyperopt_loss_function(
        results=hyperopt_results,
        trade_count=600 - 100,
        min_date=datetime(2019, 1, 1),
        max_date=datetime(2019, 5, 1),
        config=hyperopt_conf,
        processed=None,
        backtest_stats={"profit_total": hyperopt_results["profit_abs"].sum()},
        starting_balance=hyperopt_conf["dry_run_wallet"],
    )
    assert over > correct
    assert under > correct


def test_loss_calculation_prefer_shorter_trades(hyperopt_conf, hyperopt_results) -> None:
    resultsb = hyperopt_results.copy()
    resultsb.loc[1, "trade_duration"] = 20

    hyperopt_conf.update({"hyperopt_loss": "ShortTradeDurHyperOptLoss"})
    hl = HyperOptLossResolver.load_hyperoptloss(hyperopt_conf)
    longer = hl.hyperopt_loss_function(
        results=hyperopt_results,
        trade_count=100,
        min_date=datetime(2019, 1, 1),
        max_date=datetime(2019, 5, 1),
        config=hyperopt_conf,
        processed=None,
        backtest_stats={"profit_total": hyperopt_results["profit_abs"].sum()},
        starting_balance=hyperopt_conf["dry_run_wallet"],
    )
    shorter = hl.hyperopt_loss_function(
        results=resultsb,
        trade_count=100,
        min_date=datetime(2019, 1, 1),
        max_date=datetime(2019, 5, 1),
        config=hyperopt_conf,
        processed=None,
        backtest_stats={"profit_total": resultsb["profit_abs"].sum()},
        starting_balance=hyperopt_conf["dry_run_wallet"],
    )
    assert shorter < longer


def test_loss_calculation_has_limited_profit(hyperopt_conf, hyperopt_results) -> None:
    results_over = hyperopt_results.copy()
    results_over["profit_ratio"] = hyperopt_results["profit_ratio"] * 2
    results_under = hyperopt_results.copy()
    results_under["profit_ratio"] = hyperopt_results["profit_ratio"] / 2

    hyperopt_conf.update({"hyperopt_loss": "ShortTradeDurHyperOptLoss"})
    hl = HyperOptLossResolver.load_hyperoptloss(hyperopt_conf)
    correct = hl.hyperopt_loss_function(
        results=hyperopt_results,
        trade_count=600,
        min_date=datetime(2019, 1, 1),
        max_date=datetime(2019, 5, 1),
        config=hyperopt_conf,
        processed=None,
        backtest_stats={"profit_total": hyperopt_results["profit_abs"].sum()},
        starting_balance=hyperopt_conf["dry_run_wallet"],
    )
    over = hl.hyperopt_loss_function(
        results=results_over,
        trade_count=600,
        min_date=datetime(2019, 1, 1),
        max_date=datetime(2019, 5, 1),
        config=hyperopt_conf,
        processed=None,
        backtest_stats={"profit_total": results_over["profit_abs"].sum()},
        starting_balance=hyperopt_conf["dry_run_wallet"],
    )
    under = hl.hyperopt_loss_function(
        results=results_under,
        trade_count=600,
        min_date=datetime(2019, 1, 1),
        max_date=datetime(2019, 5, 1),
        config=hyperopt_conf,
        processed=None,
        backtest_stats={"profit_total": results_under["profit_abs"].sum()},
        starting_balance=hyperopt_conf["dry_run_wallet"],
    )
    assert over < correct
    assert under > correct


@pytest.mark.parametrize(
    "lossfunction",
    [
        "OnlyProfitHyperOptLoss",
        "SortinoHyperOptLoss",
        "SortinoHyperOptLossDaily",
        "SharpeHyperOptLoss",
        "SharpeHyperOptLossDaily",
        "MaxDrawDownHyperOptLoss",
        "MaxDrawDownRelativeHyperOptLoss",
        "MaxDrawDownPerPairHyperOptLoss",
        "CalmarHyperOptLoss",
        "ProfitDrawDownHyperOptLoss",
        "MultiMetricHyperOptLoss",
    ],
)
def test_loss_functions_better_profits(default_conf, hyperopt_results, lossfunction) -> None:
    results_over = hyperopt_results.copy()
    results_over["profit_abs"] = hyperopt_results["profit_abs"] * 2 + 0.2
    results_over["profit_ratio"] = hyperopt_results["profit_ratio"] * 2
    results_under = hyperopt_results.copy()
    results_under["profit_abs"] = hyperopt_results["profit_abs"] / 2 - 0.2
    results_under["profit_ratio"] = hyperopt_results["profit_ratio"] / 2
    pair_results = [
        {
            "key": "ETH/USDT",
            "max_drawdown_abs": 50.0,
            "profit_total_abs": 100.0,
        },
        {
            "key": "BTC/USDT",
            "max_drawdown_abs": 50.0,
            "profit_total_abs": 100.0,
        },
    ]
    pair_results_over = [
        {
            **p,
            "max_drawdown_abs": p["max_drawdown_abs"] * 0.5,
            "profit_total_abs": p["profit_total_abs"] * 2,
        }
        for p in pair_results
    ]
    pair_results_under = [
        {
            **p,
            "max_drawdown_abs": p["max_drawdown_abs"] * 2,
            "profit_total_abs": p["profit_total_abs"] * 0.5,
        }
        for p in pair_results
    ]

    default_conf.update({"hyperopt_loss": lossfunction})
    hl = HyperOptLossResolver.load_hyperoptloss(default_conf)
    correct = hl.hyperopt_loss_function(
        results=hyperopt_results,
        trade_count=len(hyperopt_results),
        min_date=datetime(2019, 1, 1),
        max_date=datetime(2019, 5, 1),
        config=default_conf,
        processed=None,
        backtest_stats={
            "profit_total": hyperopt_results["profit_abs"].sum(),
            "results_per_pair": pair_results,
        },
        starting_balance=default_conf["dry_run_wallet"],
    )
    over = hl.hyperopt_loss_function(
        results=results_over,
        trade_count=len(results_over),
        min_date=datetime(2019, 1, 1),
        max_date=datetime(2019, 5, 1),
        config=default_conf,
        processed=None,
        backtest_stats={
            "profit_total": results_over["profit_abs"].sum(),
            "results_per_pair": pair_results_over,
        },
        starting_balance=default_conf["dry_run_wallet"],
    )
    under = hl.hyperopt_loss_function(
        results=results_under,
        trade_count=len(results_under),
        min_date=datetime(2019, 1, 1),
        max_date=datetime(2019, 5, 1),
        config=default_conf,
        processed=None,
        backtest_stats={
            "profit_total": results_under["profit_abs"].sum(),
            "results_per_pair": pair_results_under,
        },
        starting_balance=default_conf["dry_run_wallet"],
    )
    assert over < correct
    assert under > correct