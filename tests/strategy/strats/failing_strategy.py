# 因不存在的依赖而无法加载的策略
import nonexiting_module  # noqa

from freqtrade.strategy.interface import IStrategy


class TestStrategyLegacyV1(IStrategy):
    pass
