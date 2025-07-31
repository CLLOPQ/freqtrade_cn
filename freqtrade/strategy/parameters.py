"""
IHyperStrategy接口，可超参数优化的参数类。
本模块定义了用于自动超参数优化策略的基类。
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import suppress
from typing import Any, Union

from freqtrade.enums import HyperoptState
from freqtrade.optimize.hyperopt_tools import HyperoptStateContainer


with suppress(ImportError):
    from freqtrade.optimize.space import (
        Categorical,
        Integer,
        Real,
        SKDecimal,
    )

from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


class BaseParameter(ABC):
    """
    定义可通过超参数优化的参数。
    """

    category: str | None
    default: Any
    value: Any
    in_space: bool = False
    name: str

    def __init__(
        self,
        *,
        default: Any,
        space: str | None = None,
        optimize: bool = True,
        load: bool = True,
        **kwargs,
    ):
        """
        初始化可超参数优化的参数。
        :param space: 参数类别。可以是'buy'或'sell'。如果参数字段名以'buy_'或'sell_'为前缀，则此参数可选。
        :param optimize: 将参数包含在超参数优化中。
        :param load: 从{space}_params加载参数值。
        :param kwargs: optuna.distributions的额外参数。
                (IntDistribution|FloatDistribution|CategoricalDistribution)。
        """
        if "name" in kwargs:
            raise OperationalException(
                "Name is determined by parameter field name and can not be specified manually."
            )
        self.category = space
        self._space_params = kwargs
        self.value = default
        self.optimize = optimize
        self.load = load

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"

    @abstractmethod
    def get_space(self, name: str) -> Union["Integer", "Real", "SKDecimal", "Categorical"]:
        """
        获取空间 - 将由超参数优化器用于获取超参数优化空间
        """

    def can_optimize(self):
        return (
            self.in_space
            and self.optimize
            and HyperoptStateContainer.state != HyperoptState.OPTIMIZE
        )


class NumericParameter(BaseParameter):
    """用于数值目的的内部参数"""

    float_or_int = int | float
    default: float_or_int
    value: float_or_int

    def __init__(
        self,
        low: float_or_int | Sequence[float_or_int],
        high: float_or_int | None = None,
        *,
        default: float_or_int,
        space: str | None = None,
        optimize: bool = True,
        load: bool = True,
        **kwargs,
    ):
        """
        初始化可超参数优化的数值参数。
        不能实例化，但为其他数值参数提供验证
        :param low: 优化空间的低端（包含）或[low, high]。
        :param high: 优化空间的高端（包含）。如果整个范围通过第一个参数传递，则必须为None。
        :param default: 默认值。
        :param space: 参数类别。可以是'buy'或'sell'。如果参数字段名以'buy_'或'sell_'为前缀，则此参数可选。
        :param optimize: 将参数包含在超参数优化中。
        :param load: 从{space}_params加载参数值。
        :param kwargs: optuna.distributions.*的额外参数。
        """
        if high is not None and isinstance(low, Sequence):
            raise OperationalException(f"{self.__class__.__name__} space invalid.")
        if high is None or isinstance(low, Sequence):
            if not isinstance(low, Sequence) or len(low) != 2:
                raise OperationalException(f"{self.__class__.__name__} space must be [low, high]")
            self.low, self.high = low
        else:
            self.low = low
            self.high = high

        super().__init__(default=default, space=space, optimize=optimize, load=load, **kwargs)


class IntParameter(NumericParameter):
    default: int
    value: int
    low: int
    high: int

    def __init__(
        self,
        low: int | Sequence[int],
        high: int | None = None,
        *,
        default: int,
        space: str | None = None,
        optimize: bool = True,
        load: bool = True,
        **kwargs,
    ):
        """
        初始化可超参数优化的整数参数。
        :param low: 优化空间的低端（包含）或[low, high]。
        :param high: 优化空间的高端（包含）。如果整个范围通过第一个参数传递，则必须为None。
        :param default: 默认值。
        :param space: 参数类别。可以是'buy'或'sell'。如果参数字段名以'buy_'或'sell_'为前缀，则此参数可选。
        :param optimize: 将参数包含在超参数优化中。
        :param load: 从{space}_params加载参数值。
        :param kwargs: optuna.distributions.IntDistribution的额外参数。
        """

        super().__init__(
            low=low, high=high, default=default, space=space, optimize=optimize, load=load, **kwargs
        )

    def get_space(self, name: str) -> "Integer":
        """
        创建optuna分布空间。
        :param name: 参数字段的名称。
        """
        return Integer(low=self.low, high=self.high, name=name, **self._space_params)

    @property
    def range(self):
        """
        以列表形式获取此空间中的每个值。
        在超参数优化模式下，返回从low到high（包含）的列表。
        在“非超参数优化”模式下，返回包含1个项（`value`）的列表，以避免计算大量指标。
        """
        if self.can_optimize():
            # optuna分布范围是“包含”的，而Python的range是“不包含”的
            return range(self.low, self.high + 1)
        else:
            return range(self.value, self.value + 1)


class RealParameter(NumericParameter):
    default: float
    value: float

    def __init__(
        self,
        low: float | Sequence[float],
        high: float | None = None,
        *,
        default: float,
        space: str | None = None,
        optimize: bool = True,
        load: bool = True,
        **kwargs,
    ):
        """
        初始化具有无限精度的可超参数优化浮点参数。
        :param low: 优化空间的低端（包含）或[low, high]。
        :param high: 优化空间的高端（包含）。如果整个范围通过第一个参数传递，则必须为None。
        :param default: 默认值。
        :param space: 参数类别。可以是'buy'或'sell'。如果参数字段名以'buy_'或'sell_'为前缀，则此参数可选。
        :param optimize: 将参数包含在超参数优化中。
        :param load: 从{space}_params加载参数值。
        :param kwargs: optuna.distributions.FloatDistribution的额外参数。
        """
        super().__init__(
            low=low, high=high, default=default, space=space, optimize=optimize, load=load, **kwargs
        )

    def get_space(self, name: str) -> "Real":
        """
        创建优化空间。
        :param name: 参数字段的名称。
        """
        return Real(low=self.low, high=self.high, name=name, **self._space_params)


class DecimalParameter(NumericParameter):
    default: float
    value: float

    def __init__(
        self,
        low: float | Sequence[float],
        high: float | None = None,
        *,
        default: float,
        decimals: int = 3,
        space: str | None = None,
        optimize: bool = True,
        load: bool = True,
        **kwargs,
    ):
        """
        初始化具有有限精度的可超参数优化小数参数。
        :param low: 优化空间的低端（包含）或[low, high]。
        :param high: 优化空间的高端（包含）。如果整个范围通过第一个参数传递，则必须为None。
        :param default: 默认值。
        :param decimals: 测试中包含的小数点后的小数位数。
        :param space: 参数类别。可以是'buy'或'sell'。如果参数字段名以'buy_'或'sell_'为前缀，则此参数可选。
        :param optimize: 将参数包含在超参数优化中。
        :param load: 从{space}_params加载参数值。
        :param kwargs: optuna的NumericParameter的额外参数。
        """
        self._decimals = decimals
        default = round(default, self._decimals)

        super().__init__(
            low=low, high=high, default=default, space=space, optimize=optimize, load=load, **kwargs
        )

    def get_space(self, name: str) -> "SKDecimal":
        """
        创建优化空间。
        :param name: 参数字段的名称。
        """
        return SKDecimal(
            low=self.low, high=self.high, decimals=self._decimals, name=name, **self._space_params
        )

    @property
    def range(self):
        """
        以列表形式获取此空间中的每个值。
        在超参数优化模式下，返回从low到high（包含）的列表。
        在“非超参数优化”模式下，返回包含1个项（`value`）的列表，以避免计算大量指标。
        """
        if self.can_optimize():
            low = int(self.low * pow(10, self._decimals))
            high = int(self.high * pow(10, self._decimals)) + 1
            return [round(n * pow(0.1, self._decimals), self._decimals) for n in range(low, high)]
        else:
            return [self.value]


class CategoricalParameter(BaseParameter):
    default: Any
    value: Any
    opt_range: Sequence[Any]

    def __init__(
        self,
        categories: Sequence[Any],
        *,
        default: Any | None = None,
        space: str | None = None,
        optimize: bool = True,
        load: bool = True,
        **kwargs,
    ):
        """
        初始化可超参数优化的参数。
        :param categories: 优化空间，[a, b, ...]。
        :param default: 默认值。如果未指定，则使用指定空间中的第一项。
        :param space: 参数类别。可以是'buy'或'sell'。如果参数字段名以'buy_'或'sell_'为前缀，则此参数可选。
        :param optimize: 将参数包含在超参数优化中。
        :param load: 从{space}_params加载参数值。
        :param kwargs: 兼容性。Optuna的CategoricalDistribution不接受额外参数。
        """
        if len(categories) < 2:
            raise OperationalException(
                "CategoricalParameter space must be [a, b, ...] (at least two parameters)"
            )
        self.opt_range = categories
        super().__init__(default=default, space=space, optimize=optimize, load=load, **kwargs)

    def get_space(self, name: str) -> "Categorical":
        """
        创建optuna分布空间。
        :param name: 参数字段的名称。
        """
        return Categorical(self.opt_range, name=name)

    @property
    def range(self):
        """
        以列表形式获取此空间中的每个值。
        在超参数优化模式下，返回类别列表。
        在“非超参数优化”模式下，返回包含1个项（`value`）的列表，以避免计算大量指标。
        """
        if self.can_optimize():
            return self.opt_range
        else:
            return [self.value]


class BooleanParameter(CategoricalParameter):
    def __init__(
        self,
        *,
        default: Any | None = None,
        space: str | None = None,
        optimize: bool = True,
        load: bool = True,
        **kwargs,
    ):
        """
        初始化可超参数优化的布尔型参数。
        这是`CategoricalParameter([True, False])`的快捷方式。
        :param default: 默认值。如果未指定，则使用指定空间中的第一项。
        :param space: 参数类别。可以是'buy'或'sell'。如果参数字段名以'buy_'或'sell_'为前缀，则此参数可选。
        :param optimize: 将参数包含在超参数优化中。
        :param load: 从{space}_params加载参数值。
        :param kwargs: optuna.distributions.CategoricalDistribution的额外参数。
        """

        categories = [True, False]
        super().__init__(
            categories=categories,
            default=default,
            space=space,
            optimize=optimize,
            load=load,
            **kwargs,
        )