import logging

from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


def hyperopt_filter_epochs(epochs: list, filteroptions: dict, log: bool = True) -> list:
    """
    从超参数优化结果列表中过滤项目
    """
    if filteroptions["only_best"]:
        epochs = [x for x in epochs if x["is_best"]]
    if filteroptions["only_profitable"]:
        epochs = [x for x in epochs if x["results_metrics"].get("profit_total", 0) > 0]

    epochs = _hyperopt_filter_epochs_trade_count(epochs, filteroptions)

    epochs = _hyperopt_filter_epochs_duration(epochs, filteroptions)

    epochs = _hyperopt_filter_epochs_profit(epochs, filteroptions)

    epochs = _hyperopt_filter_epochs_objective(epochs, filteroptions)
    if log:
        logger.info(
            f"{len(epochs)} "
            + ("最优的 " if filteroptions["only_best"] else "")
            + ("盈利的 " if filteroptions["only_profitable"] else "")
            + "周期已找到。"
        )
    return epochs


def _hyperopt_filter_epochs_trade(epochs: list, trade_count: int):
    """
    过滤交易次数大于指定值的周期
    """
    return [x for x in epochs if x["results_metrics"].get("total_trades", 0) > trade_count]


def _hyperopt_filter_epochs_trade_count(epochs: list, filteroptions: dict) -> list:
    if filteroptions["filter_min_trades"] > 0:
        epochs = _hyperopt_filter_epochs_trade(epochs, filteroptions["filter_min_trades"])

    if filteroptions["filter_max_trades"] > 0:
        epochs = [
            x
            for x in epochs
            if x["results_metrics"].get("total_trades") < filteroptions["filter_max_trades"]
        ]
    return epochs


def _hyperopt_filter_epochs_duration(epochs: list, filteroptions: dict) -> list:
    def get_duration_value(x):
        # 以分钟为单位的时长...
        if "holding_avg_s" in x["results_metrics"]:
            avg = x["results_metrics"]["holding_avg_s"]
            return avg // 60
        raise OperationalException(
            "持仓平均时长不可用。请省略平均时长过滤，或使用此版本重新运行超参数优化。"
        )

    if filteroptions["filter_min_avg_time"] is not None:
        epochs = _hyperopt_filter_epochs_trade(epochs, 0)
        epochs = [x for x in epochs if get_duration_value(x) > filteroptions["filter_min_avg_time"]]
    if filteroptions["filter_max_avg_time"] is not None:
        epochs = _hyperopt_filter_epochs_trade(epochs, 0)
        epochs = [x for x in epochs if get_duration_value(x) < filteroptions["filter_max_avg_time"]]

    return epochs


def _hyperopt_filter_epochs_profit(epochs: list, filteroptions: dict) -> list:
    if filteroptions["filter_min_avg_profit"] is not None:
        epochs = _hyperopt_filter_epochs_trade(epochs, 0)
        epochs = [
            x
            for x in epochs
            if x["results_metrics"].get("profit_mean", 0) * 100
            > filteroptions["filter_min_avg_profit"]
        ]
    if filteroptions["filter_max_avg_profit"] is not None:
        epochs = _hyperopt_filter_epochs_trade(epochs, 0)
        epochs = [
            x
            for x in epochs
            if x["results_metrics"].get("profit_mean", 0) * 100
            < filteroptions["filter_max_avg_profit"]
        ]
    if filteroptions["filter_min_total_profit"] is not None:
        epochs = _hyperopt_filter_epochs_trade(epochs, 0)
        epochs = [
            x
            for x in epochs
            if x["results_metrics"].get("profit_total_abs", 0)
            > filteroptions["filter_min_total_profit"]
        ]
    if filteroptions["filter_max_total_profit"] is not None:
        epochs = _hyperopt_filter_epochs_trade(epochs, 0)
        epochs = [
            x
            for x in epochs
            if x["results_metrics"].get("profit_total_abs", 0)
            < filteroptions["filter_max_total_profit"]
        ]
    return epochs


def _hyperopt_filter_epochs_objective(epochs: list, filteroptions: dict) -> list:
    if filteroptions["filter_min_objective"] is not None:
        epochs = _hyperopt_filter_epochs_trade(epochs, 0)

        epochs = [x for x in epochs if x["loss"] < filteroptions["filter_min_objective"]]
    if filteroptions["filter_max_objective"] is not None:
        epochs = _hyperopt_filter_epochs_trade(epochs, 0)

        epochs = [x for x in epochs if x["loss"] > filteroptions["filter_max_objective"]]

    return epochs