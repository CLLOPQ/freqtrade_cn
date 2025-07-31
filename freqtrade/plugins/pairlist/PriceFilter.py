"""
价格交易对列表过滤器
"""

import logging

from freqtrade.exceptions import OperationalException
from freqtrade.exchange.exchange_types import Ticker
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting


logger = logging.getLogger(__name__)


class PriceFilter(IPairList):
    supports_backtesting = SupportsBacktesting.BIASED

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._low_price_ratio = self._pairlistconfig.get("low_price_ratio", 0)
        if self._low_price_ratio < 0:
            raise OperationalException("PriceFilter 要求 low_price_ratio 大于等于 0")
        self._min_price = self._pairlistconfig.get("min_price", 0)
        if self._min_price < 0:
            raise OperationalException("PriceFilter 要求 min_price 大于等于 0")
        self._max_price = self._pairlistconfig.get("max_price", 0)
        if self._max_price < 0:
            raise OperationalException("PriceFilter 要求 max_price 大于等于 0")
        self._max_value = self._pairlistconfig.get("max_value", 0)
        if self._max_value < 0:
            raise OperationalException("PriceFilter 要求 max_value 大于等于 0")
        self._enabled = (
            (self._low_price_ratio > 0)
            or (self._min_price > 0)
            or (self._max_price > 0)
            or (self._max_value > 0)
        )

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
        active_price_filters = []
        if self._low_price_ratio != 0:
            active_price_filters.append(f"低于 {self._low_price_ratio:.1%}")
        if self._min_price != 0:
            active_price_filters.append(f"低于 {self._min_price:.8f}")
        if self._max_price != 0:
            active_price_filters.append(f"高于 {self._max_price:.8f}")
        if self._max_value != 0:
            active_price_filters.append(f"价值高于 {self._max_value:.8f}")

        if len(active_price_filters):
            return f"{self.name} - 过滤价格在 {' 或 '.join(active_price_filters)} 的交易对。"

        return f"{self.name} - 未配置价格过滤条件。"

    @staticmethod
    def description() -> str:
        return "按价格过滤交易对。"

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "low_price_ratio": {
                "type": "number",
                "default": 0,
                "description": "低价格比率",
                "help": (
                    "移除价格变动1个价格单位（点）超过此比率的交易对。"
                ),
            },
            "min_price": {
                "type": "number",
                "default": 0,
                "description": "最低价格",
                "help": "移除价格低于此值的交易对。",
            },
            "max_price": {
                "type": "number",
                "default": 0,
                "description": "最高价格",
                "help": "移除价格高于此值的交易对。",
            },
            "max_value": {
                "type": "number",
                "default": 0,
                "description": "最大价值",
                "help": "移除价值（价格*数量）高于此值的交易对。",
            },
        }

    def _validate_pair(self, pair: str, ticker: Ticker | None) -> bool:
        """
        检查一个价格单位（点）是否超过某个阈值。
        :param pair: 当前正在验证的交易对
        :param ticker: 从 ccxt.fetch_ticker 返回的行情数据字典
        :return: 如果交易对可以保留则返回 True，否则返回 False
        """
        if ticker and "last" in ticker and ticker["last"] is not None and ticker.get("last") != 0:
            price: float = ticker["last"]
        else:
            self.log_once(
                f"从白名单中移除 {pair}，因为 ticker['last'] 为空（通常过去24小时内没有交易）。",
                logger.info,
            )
            return False

        # 执行低价格比率检查。
        if self._low_price_ratio != 0:
            compare = self._exchange.price_get_one_pip(pair, price)
            changeperc = compare / price
            if changeperc > self._low_price_ratio:
                self.log_once(
                    f"从白名单中移除 {pair}，因为 1 个单位为 {changeperc:.3%}",
                    logger.info,
                )
                return False

        # 执行低数量检查
        if self._max_value != 0:
            market = self._exchange.markets[pair]
            limits = market["limits"]
            if limits["amount"]["min"] is not None:
                min_amount = limits["amount"]["min"]
                min_precision = market["precision"]["amount"]

                min_value = min_amount * price
                if self._exchange.precisionMode == 4:
                    # 价格点
                    next_value = (min_amount + min_precision) * price
                else:
                    # 小数位数
                    min_precision = pow(0.1, min_precision)
                    next_value = (min_amount + min_precision) * price
                diff = next_value - min_value

                if diff > self._max_value:
                    self.log_once(
                        f"从白名单中移除 {pair}，因为最小价值变化 {diff} 大于 {self._max_value}。",
                        logger.info,
                    )
                    return False

        # 执行最低价格检查。
        if self._min_price != 0:
            if price < self._min_price:
                self.log_once(
                    f"从白名单中移除 {pair}，因为最后价格 < {self._min_price:.8f}",
                    logger.info,
                )
                return False

        # 执行最高价格检查。
        if self._max_price != 0:
            if price > self._max_price:
                self.log_once(
                    f"从白名单中移除 {pair}，因为最后价格 > {self._max_price:.8f}",
                    logger.info,
                )
                return False

        return True