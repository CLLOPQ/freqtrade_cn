from collections.abc import Sequence
from typing import Any, Protocol

from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution


class DimensionProtocol(Protocol):
    """维度协议，定义维度对象必须包含name属性"""
    name: str


class ft_CategoricalDistribution(CategoricalDistribution):
    """
    自定义的分类分布类，继承自Optuna的CategoricalDistribution
    增加了name属性用于标识该分布的名称
    """
    def __init__(
        self,
        categories: Sequence[Any],
        name: str,
        **kwargs,
    ):
        self.name = name  # 分布名称
        self.categories = categories  # 分类选项列表
        # if len(categories) <= 1:
        #     raise Exception(f"{name}需要至少2个分类选项")
        super().__init__(categories,** kwargs)

    def __repr__(self):
        """返回对象的字符串表示"""
        return f"CategoricalDistribution({self.categories})"


class ft_IntDistribution(IntDistribution):
    """
    自定义的整数分布类，继承自Optuna的IntDistribution
    增加了name属性用于标识该分布的名称
    """
    def __init__(
        self,
        low: int | float,
        high: int | float,
        name: str,
        **kwargs,
    ):
        self.name = name  # 分布名称
        self.low = int(low)  # 最小值（整数）
        self.high = int(high)  # 最大值（整数）
        super().__init__(self.low, self.high, **kwargs)

    def __repr__(self):
        """返回对象的字符串表示"""
        return f"IntDistribution(low={self.low}, high={self.high})"


class ft_FloatDistribution(FloatDistribution):
    """
    自定义的浮点数分布类，继承自Optuna的FloatDistribution
    增加了name属性用于标识该分布的名称
    """
    def __init__(
        self,
        low: float,
        high: float,
        name: str,
        **kwargs,
    ):
        self.name = name  # 分布名称
        self.low = low  # 最小值（浮点数）
        self.high = high  # 最大值（浮点数）
        super().__init__(low, high, **kwargs)

    def __repr__(self):
        """返回对象的字符串表示"""
        return f"FloatDistribution(low={self.low}, high={self.high}, step={self.step})"