"""
精度交易对列表过滤器
"""

import logging

from freqtrade.exceptions import OperationalException
from freqtrade.exchange import ROUND_UP
from freqtrade.exchange.exchange_types import Ticker
from freqtrade.plugins.pairlist.IPairList import IPairList, SupportsBacktesting


logger = logging.getLogger(__name__)


class PrecisionFilter(IPairList):
    supports_backtesting = SupportsBacktesting.BIASED

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if "stoploss" not in self._config:
            raise OperationalException(
                "PrecisionFilter 只能在定义了止损的情况下工作。请在配置中添加 stoploss 键（会覆盖策略中可能设置的止损值）。"
            )
        self._stoploss = self._config["stoploss"]
        self._enabled = self._stoploss != 0

        # 预计算经过处理的止损值，避免对每个交易对重复计算
        self._stoploss = 1 - abs(self._stoploss)

    @property
    def needstickers(self) -> bool:
        """
        定义是否需要行情数据的布尔属性。
        如果没有交易对列表需要行情数据，将传递一个空字典作为行情数据参数给 filter_pairlist 方法
        """
        return True

    def short_desc(self) -> str:
        """
        简短的白名单方法描述 - 用于启动消息
        """
        return f"{self.name} - 过滤不可交易的交易对。"

    @staticmethod
    def description() -> str:
        return "过滤低价值的币种，这些币种不允许设置止损。"

    def _validate_pair(self, pair: str, ticker: Ticker | None) -> bool:
        """
        检查交易对是否有足够的空间添加止损，以避免购买极低价值的币种导致“无法卖出”。
        :param pair: 当前正在验证的交易对
        :param ticker: 从 ccxt.fetch_ticker 返回的行情数据字典
        :return: 如果交易对可以保留则返回 True，否则返回 False
        """
        if not ticker or ticker.get("last", None) is None:
            self.log_once(
                f"将 {pair} 从白名单中移除，因为 ticker['last'] 为空（通常表示过去24小时内没有交易）。",
                logger.info,
            )
            return False
        stop_price = ticker["last"] * self._stoploss

        # 根据交易对的精度调整止损价格（向上取整）
        sp = self._exchange.price_to_precision(pair, stop_price, rounding_mode=ROUND_UP)

        # 计算止损价的99%并根据交易对精度调整（向上取整）
        stop_gap_price = self._exchange.price_to_precision(
            pair, stop_price * 0.99, rounding_mode=ROUND_UP
        )
        logger.debug(f"{pair} - {sp} : {stop_gap_price}")

        if sp <= stop_gap_price:
            self.log_once(
                f"将 {pair} 从白名单中移除，因为止损价格 {sp} 将小于等于止损限制 {stop_gap_price}",
                logger.info,
            )
            return False

        return True