"""
交易对列表处理器基类
"""

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from typing import Any, Literal, TypedDict

from freqtrade.constants import Config
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import Exchange, market_is_active
from freqtrade.exchange.exchange_types import Ticker, Tickers
from freqtrade.mixins import LoggingMixin


logger = logging.getLogger(__name__)


class __PairlistParameterBase(TypedDict):
    description: str  # 描述
    help: str  # 帮助


class __NumberPairlistParameter(__PairlistParameterBase):
    type: Literal["number"]  # 类型
    default: int | float | None  # 默认值


class __StringPairlistParameter(__PairlistParameterBase):
    type: Literal["string"]  # 类型
    default: str | None  # 默认值


class __OptionPairlistParameter(__PairlistParameterBase):
    type: Literal["option"]  # 类型
    default: str | None  # 默认值
    options: list[str]  # 选项列表


class __ListPairListParamenter(__PairlistParameterBase):
    type: Literal["list"]  # 类型
    default: list[str] | None  # 默认值


class __BoolPairlistParameter(__PairlistParameterBase):
    type: Literal["boolean"]  # 类型
    default: bool | None  # 默认值


PairlistParameter = (
    __NumberPairlistParameter
    | __StringPairlistParameter
    | __OptionPairlistParameter
    | __BoolPairlistParameter
    | __ListPairListParamenter
)


class SupportsBacktesting(str, Enum):
    """
    枚举，用于指示交易对列表处理器是否支持回测。
    """

    YES = "yes"  # 支持
    NO = "no"    # 不支持
    NO_ACTION = "no_action"  # 无操作
    BIASED = "biased"  # 有偏差


class IPairList(LoggingMixin, ABC):
    is_pairlist_generator = False
    supports_backtesting: SupportsBacktesting = SupportsBacktesting.NO  # 支持回测

    def __init__(
        self,
        exchange: Exchange,
        pairlistmanager,
        config: Config,
        pairlistconfig: dict[str, Any],
        pairlist_pos: int,
    ) -> None:
        """
        :param exchange: 交易所实例
        :param pairlistmanager: 已实例化的交易对列表管理器
        :param config: 全局机器人配置
        :param pairlistconfig: 此交易对列表处理器的配置 - 可以为空。
        :param pairlist_pos: 交易对列表处理器在链中的位置
        """
        self._enabled = True  # 是否启用

        self._exchange: Exchange = exchange  # 交易所实例
        self._pairlistmanager = pairlistmanager  # 交易对列表管理器
        self._config = config  # 全局配置
        self._pairlistconfig = pairlistconfig  # 当前处理器配置
        self._pairlist_pos = pairlist_pos  # 处理器位置
        self.refresh_period = self._pairlistconfig.get("refresh_period", 1800)  # 刷新周期
        LoggingMixin.__init__(self, logger, self.refresh_period)  # 初始化日志

    @property
    def name(self) -> str:
        """
        获取类名
        -> 无需在子类中重写
        """
        return self.__class__.__name__  # 类名

    @property
    @abstractmethod
    def needstickers(self) -> bool:
        """
        布尔属性，定义是否需要行情数据。
        如果没有交易对列表需要行情数据，则将空字典作为行情数据参数传递给filter_pairlist方法
        """
        return False  # 是否需要行情数据

    @staticmethod
    @abstractmethod
    def description() -> str:
        """
        返回此交易对列表处理器的描述
        -> 请在子类中重写
        """
        return ""  # 描述

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        """
        返回此交易对列表处理器使用的参数及其类型
        包含一个以参数名为键，以包含类型和默认值的字典为值的字典。
        -> 请在子类中重写
        """
        return {}  # 参数

    @staticmethod
    def refresh_period_parameter() -> dict[str, PairlistParameter]:
        return {
            "refresh_period": {
                "type": "number",  # 类型
                "default": 1800,  # 默认值
                "description": "刷新周期",  # 描述
                "help": "刷新周期（秒）",  # 帮助
            }
        }

    @abstractmethod
    def short_desc(self) -> str:
        """
        简短的白名单方法描述 - 用于启动消息
        -> 请在子类中重写
        """

    def _validate_pair(self, pair: str, ticker: Ticker | None) -> bool:
        """
        根据交易对列表处理器的特定条件检查单个交易对。

        可以在交易对列表处理器中实现此方法，或者重写通用的filter_pairlist()方法。

        :param pair: 当前正在验证的交易对
        :param ticker: 从ccxt.fetch_ticker返回的行情字典
        :return: 如果交易对可以保留则返回True，否则返回False
        """
        raise NotImplementedError()  # 未实现

    def gen_pairlist(self, tickers: Tickers) -> list[str]:
        """
        生成交易对列表。

        此方法由交易对列表管理器在refresh_pairlist()方法中调用一次，为交易对列表处理器链提供起始交易对列表。
        交易对列表过滤器（那些不能在链的第一个位置使用的交易对列表处理器）不应重写此基础实现——如果在链的第一个位置使用交易对列表处理器，此实现将引发异常。

        :param tickers: 行情数据（来自exchange.get_tickers）。可能已缓存。
        :return: 交易对列表
        """
        raise OperationalException(
            "此交易对列表处理器不应被"
            "用作交易对列表处理器列表中的第一个位置。"
        )

    def filter_pairlist(self, pairlist: list[str], tickers: Tickers) -> list[str]:
        """
        过滤并排序交易对列表，然后再次返回白名单。

        在每个机器人迭代时调用——如果需要，请使用内部缓存。
        此通用实现对交易对列表中的每个交易对调用self._validate_pair()。

        有些交易对列表处理器会重写此通用实现并使用自己的过滤方法。

        :param pairlist: 要过滤或排序的交易对列表
        :param tickers: 行情数据（来自exchange.get_tickers）。可能已缓存。
        :return: 新的白名单
        """
        if self._enabled:
            # 复制列表，因为我们要修改此列表
            for p in deepcopy(pairlist):
                # 过滤资产
                if not self._validate_pair(p, tickers[p] if p in tickers else None):
                    pairlist.remove(p)

        return pairlist

    def verify_blacklist(self, pairlist: list[str], logmethod) -> list[str]:
        """
        代理方法，用于verify_blacklist，方便子类访问。
        :param pairlist: 要验证的交易对列表
        :param logmethod: 将被调用的函数，`logger.info` 或 `logger.warning`。
        :return: 交易对列表 - 已被列入黑名单的交易对
        """
        return self._pairlistmanager.verify_blacklist(pairlist, logmethod)

    def verify_whitelist(
        self, pairlist: list[str], logmethod, keep_invalid: bool = False
    ) -> list[str]:
        """
        代理方法，用于verify_whitelist，方便子类访问。
        :param pairlist: 要验证的交易对列表
        :param logmethod: 将被调用的函数，`logger.info` 或 `logger.warning`
        :param keep_invalid: 如果设置为True，则在扩展正则表达式时静默丢弃无效交易对。
        :return: 交易对列表 - 已被列入白名单的交易对
        """
        return self._pairlistmanager.verify_whitelist(pairlist, logmethod, keep_invalid)

    def _whitelist_for_active_markets(self, pairlist: list[str]) -> list[str]:
        """
        检查可用市场，如果有必要，从白名单中移除交易对
        :param pairlist: 用户可能想要交易的已排序交易对列表
        :return: 用户想要交易的交易对列表，不包含不可用或已被列入黑名单的交易对
        """
        markets = self._exchange.markets
        if not markets:
            raise OperationalException(
                "市场未加载。请确保交易所已正确初始化。"
            )

        sanitized_whitelist: list[str] = []
        for pair in pairlist:
            # 交易对不在生成的动态市场中或 stake 货币错误
            if pair not in markets:
                self.log_once(
                    f"交易对 {pair} 与交易所 {self._exchange.name} 不兼容。从白名单中移除它..",
                    logger.warning,
                    True,
                )
                continue

            if not self._exchange.market_is_tradable(markets[pair]):
                self.log_once(
                    f"交易对 {pair} 无法与Freqtrade交易。从白名单中移除它..",
                    logger.warning,
                    True,
                )
                continue

            if self._exchange.get_pair_quote_currency(pair) != self._config["stake_currency"]:
                self.log_once(
                    f"交易对 {pair} 与您的 stake 货币 {self._config['stake_currency']} 不兼容。从白名单中移除它..",
                    logger.warning,
                    True,
                )
                continue

            # 检查市场是否激活
            market = markets[pair]
            if not market_is_active(market):
                self.log_once(
                    f"从白名单中忽略 {pair}。市场未激活。",
                    logger.info,
                    True,
                )
                continue
            if pair not in sanitized_whitelist:
                sanitized_whitelist.append(pair)

        # 需要移除未知交易对
        return sanitized_whitelist