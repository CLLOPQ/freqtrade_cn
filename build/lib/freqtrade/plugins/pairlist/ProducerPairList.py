"""
外部交易对列表提供器
从Leader数据提供交易对列表
"""

import logging

from freqtrade.exceptions import OperationalException
from freqtrade.exchange.exchange_types import Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting


logger = logging.getLogger(__name__)


class ProducerPairList(IPairList):
    """
    与外部消息消费者配合使用的交易对列表插件。
    将使用来自Leader数据的交易对。

    使用示例：
        "pairlists": [
            {
                "method": "ProducerPairList",
                "number_assets": 5,
                "producer_name": "default",
            }
        ],
    """

    is_pairlist_generator = True
    supports_backtesting = SupportsBacktesting.NO

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._num_assets: int = self._pairlistconfig.get("number_assets", 0)
        self._producer_name = self._pairlistconfig.get("producer_name", "default")
        if not self._config.get("external_message_consumer", {}).get("enabled"):
            raise OperationalException(
                "ProducerPairList要求外部消息消费者已启用。"
            )

    @property
    def needstickers(self) -> bool:
        """
        定义是否需要行情数据的布尔属性。
        如果没有交易对列表需要行情数据，则将空字典作为行情数据参数传递给filter_pairlist方法
        """
        return False

    def short_desc(self) -> str:
        """
        简短的白名单方法描述 - 用于启动消息
        -> 请在子类中重写
        """
        return f"{self.name} - {self._producer_name}"

    @staticmethod
    def description() -> str:
        return "从上游机器人获取交易对列表。"

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "number_assets": {
                "type": "number",
                "default": 0,
                "description": "资产数量",
                "help": "从交易对列表中使用的资产数量",
            },
            "producer_name": {
                "type": "string",
                "default": "default",
                "description": "生产者名称",
                "help": (
                    "要使用的生产者名称。需要额外的"
                    "外部消息消费者配置。"
                ),
            },
        }

    def _filter_pairlist(self, pairlist: list[str] | None):
        upstream_pairlist = self._pairlistmanager._dataprovider.get_producer_pairs(
            self._producer_name
        )

        if pairlist is None:
            pairlist = self._pairlistmanager._dataprovider.get_producer_pairs(self._producer_name)

        pairs = list(dict.fromkeys(pairlist + upstream_pairlist))
        if self._num_assets:
            pairs = pairs[: self._num_assets]

        return pairs

    def gen_pairlist(self, tickers: Tickers) -> list[str]:
        """
        生成交易对列表
        :param tickers: 行情数据（来自交易所.get_tickers()）。可能已缓存。
        :return: 交易对列表
        """
        pairs = self._filter_pairlist(None)
        self.log_once(f"Received pairs: {pairs}", logger.debug)
        pairs = self._whitelist_for_active_markets(self.verify_whitelist(pairs, logger.info))
        return pairs

    def filter_pairlist(self, pairlist: list[str], tickers: Tickers) -> list[str]:
        """
        过滤并排序交易对列表，然后返回白名单。
        在每个机器人迭代时调用 - 如果需要，请使用内部缓存
        :param pairlist: 要过滤或排序的交易对列表
        :param tickers: 行情数据（来自交易所.get_tickers()）。可能已缓存。
        :return: 新的白名单
        """
        return self._filter_pairlist(pairlist)