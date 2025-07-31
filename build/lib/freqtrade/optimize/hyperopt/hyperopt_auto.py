"""
HyperOptAuto 类。
此模块实现了一个便捷的自动超参数优化类，可与实现 IHyperStrategy 接口的策略一起使用。
"""

import logging
from collections.abc import Callable
from contextlib import suppress

from freqtrade.exceptions import OperationalException


with suppress(ImportError):
    from freqtrade.optimize.space import Dimension

from freqtrade.optimize.hyperopt.hyperopt_interface import EstimatorType, IHyperOpt


logger = logging.getLogger(__name__)


def _format_exception_message(space: str, ignore_missing_space: bool) -> None:
    msg = (
        f"'{space}' 空间已包含在超参数优化中 "
        f"但在您的策略中未找到该空间的参数。 "
    )
    if ignore_missing_space:
        logger.warning(msg + "此空间将被忽略。")
    else:
        raise OperationalException(
            msg + f"请确保已启用此空间的参数进行优化 "
            f"或从超参数优化中移除 '{space}' 空间。"
        )


class HyperOptAuto(IHyperOpt):
    """
    此类将功能委托给 Strategy(IHyperStrategy) 和 Strategy.HyperOpt 类。
    大多数情况下，Strategy.HyperOpt 类只实现 indicator_space 和 sell_indicator_space 方法，
    但也可以重写其他超参数优化方法。
    """

    def _get_func(self, name) -> Callable:
        """
        返回在 Strategy.HyperOpt 类中定义的函数，或在 super() 类中定义的函数。
        :param name: 函数名称。
        :return: 请求的函数。
        """
        hyperopt_cls = getattr(self.strategy, "HyperOpt", None)
        default_func = getattr(super(), name)
        if hyperopt_cls:
            return getattr(hyperopt_cls, name, default_func)
        else:
            return default_func

    def _generate_indicator_space(self, category):
        for attr_name, attr in self.strategy.enumerate_parameters(category):
            if attr.optimize:
                yield attr.get_space(attr_name)

    def _get_indicator_space(self, category) -> list:
        # TODO: 这是否必要，或者我们可以直接调用 "generate_space"？
        indicator_space = list(self._generate_indicator_space(category))
        if len(indicator_space) > 0:
            return indicator_space
        else:
            _format_exception_message(
                category, self.config.get("hyperopt_ignore_missing_space", False)
            )
            return []

    def buy_indicator_space(self) -> list["Dimension"]:
        return self._get_indicator_space("buy")

    def sell_indicator_space(self) -> list["Dimension"]:
        return self._get_indicator_space("sell")

    def protection_space(self) -> list["Dimension"]:
        return self._get_indicator_space("protection")

    def generate_roi_table(self, params: dict) -> dict[int, float]:
        return self._get_func("generate_roi_table")(params)

    def roi_space(self) -> list["Dimension"]:
        return self._get_func("roi_space")()

    def stoploss_space(self) -> list["Dimension"]:
        return self._get_func("stoploss_space")()

    def generate_trailing_params(self, params: dict) -> dict:
        return self._get_func("generate_trailing_params")(params)

    def trailing_space(self) -> list["Dimension"]:
        return self._get_func("trailing_space")()

    def max_open_trades_space(self) -> list["Dimension"]:
        return self._get_func("max_open_trades_space")()

    def generate_estimator(self, dimensions: list["Dimension"], **kwargs) -> EstimatorType:
        return self._get_func("generate_estimator")(dimensions=dimensions,** kwargs)