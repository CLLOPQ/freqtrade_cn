"""
此模块用于加载自定义交易所
"""

import logging
from inspect import isclass
from typing import Any

import freqtrade.exchange as exchanges
from freqtrade.constants import Config, ExchangeConfig
from freqtrade.exchange import MAP_EXCHANGE_CHILDCLASS, Exchange
from freqtrade.resolvers.iresolver import IResolver


logger = logging.getLogger(__name__)


class ExchangeResolver(IResolver):
    """
    此类包含加载自定义交易所类的所有逻辑
    """

    object_type = Exchange

    @staticmethod
    def load_exchange(
        config: Config,
        *,
        exchange_config: ExchangeConfig | None = None,
        validate: bool = True,
        load_leverage_tiers: bool = False,
    ) -> Exchange:
        """
        从配置参数加载自定义类
        :param exchange_name: 要加载的交易所名称
        :param config: 配置字典
        """
        exchange_name: str = config["exchange"]["name"]
        # 映射交易所名称以避免相同交易所的重复类
        exchange_name = MAP_EXCHANGE_CHILDCLASS.get(exchange_name, exchange_name)
        exchange_name = exchange_name.title()
        exchange = None
        try:
            exchange = ExchangeResolver._load_exchange(
                exchange_name,
                kwargs={
                    "config": config,
                    "validate": validate,
                    "exchange_config": exchange_config,
                    "load_leverage_tiers": load_leverage_tiers,
                },
            )
        except ImportError:
            logger.info(
                f"未找到{exchange_name}特定的子类。改用通用类。"
            )
        if not exchange:
            exchange = Exchange(
                config,
                validate=validate,
                exchange_config=exchange_config,
            )
        return exchange

    @staticmethod
    def _load_exchange(exchange_name: str, kwargs: dict) -> Exchange:
        """
        加载指定的交易所。
        仅检查在freqtrade.exchanges中导出的交易所
        :param exchange_name: 要导入的模块名称
        :return: Exchange实例或None
        """

        try:
            ex_class = getattr(exchanges, exchange_name)

            exchange = ex_class(**kwargs)
            if exchange:
                logger.info(f"正在使用已解析的交易所'{exchange_name}'...")
                return exchange
        except AttributeError:
            # 传递并改为引发ImportError
            pass

        raise ImportError(
            f"无法加载交易所'{exchange_name}'。此类不存在 "
            "或包含Python代码错误。"
        )

    @classmethod
    def search_all_objects(
        cls, config: Config, enum_failed: bool, recursive: bool = False
    ) -> list[dict[str, Any]]:
        """
        搜索有效对象
        :param config: 配置对象
        :param enum_failed: 如果为True，将为失败的模块返回None。否则，跳过失败的模块。
            否则，失败的模块将被跳过。
        :param recursive: 递归遍历目录树以搜索策略
        :return: 包含'name'、'class'和'location'条目的字典列表
        """
        result = []
        for exchange_name in dir(exchanges):
            exchange = getattr(exchanges, exchange_name)
            if isclass(exchange) and issubclass(exchange, Exchange):
                result.append(
                    {
                        "name": exchange_name,
                        "class": exchange,
                        "location": exchange.__module__,
                        "location_rel: ": exchange.__module__.replace("freqtrade.", ""),
                    }
                )
        return result