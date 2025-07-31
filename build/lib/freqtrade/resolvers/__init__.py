# flake8: noqa: F401
# isort: off
from freqtrade.resolvers.iresolver import IResolver
from freqtrade.resolvers.exchange_resolver import ExchangeResolver

# isort: on
# 不要导入 HyperoptResolver 以避免加载整个优化模块树
# from freqtrade.resolvers.hyperopt_resolver import HyperOptResolver
from freqtrade.resolvers.pairlist_resolver import PairListResolver
from freqtrade.resolvers.protection_resolver import ProtectionResolver
from freqtrade.resolvers.strategy_resolver import StrategyResolver