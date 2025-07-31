"""
定义将加密货币转换为法币的类的模块，例如 BTC 转换为 USD
"""

import logging
from datetime import datetime
from typing import Any

from cachetools import TTLCache
from requests.exceptions import RequestException

from freqtrade.constants import SUPPORTED_FIAT, Config
from freqtrade.mixins.logging_mixin import LoggingMixin
from freqtrade.util.coin_gecko import FtCoinGeckoApi


logger = logging.getLogger(__name__)


# 手动为一些常见的、在CoinGecko有重复条目的硬币映射符号到ID
coingecko_mapping = {
    "eth": "ethereum",
    "bnb": "binancecoin",
    "sol": "solana",
    "usdt": "tether",
    "busd": "binance-usd",
    "tusd": "true-usd",
    "usdc": "usd-coin",
    "btc": "bitcoin",
}


class CryptoToFiatConverter(LoggingMixin):
    """
    用于初始化加密货币到法币转换的主类。
    该对象包含一个加密货币-法币对的列表。
    该对象也是单例模式
    """

    __instance = None

    _coinlistings: list[dict] = []
    _backoff: float = 0.0

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        """
        单例模式，确保只创建一个实例。
        """
        if not cls.__instance:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self, config: Config) -> None:
        # 超时时间：6小时
        self._pair_price: TTLCache = TTLCache(maxsize=500, ttl=6 * 60 * 60)

        _coingecko_config = config.get("coingecko", {})
        self._coingecko = FtCoinGeckoApi(
            api_key=_coingecko_config.get("api_key", ""),
            is_demo=_coingecko_config.get("is_demo", True),
            retries=1,
        )
        LoggingMixin.__init__(self, logger, 3600)
        self._load_cryptomap()

    def _load_cryptomap(self) -> None:
        try:
            # 使用列表推导式确保我们得到一个列表。
            self._coinlistings = [x for x in self._coingecko.get_coins_list()]
        except RequestException as request_exception:
            if "429" in str(request_exception):
                logger.warning(
                    "CoinGecko API请求次数过多，正在退避并稍后重试。"
                )
                # 将退避时间戳设置为60秒后
                self._backoff = datetime.now().timestamp() + 60
                return
            # 如果请求不是429错误，我们希望正常抛出错误
            logger.error(
                "无法加载法币加密货币映射，原因如下："
                f"{request_exception}"
            )
        except Exception as exception:
            logger.error(
                f"无法加载法币加密货币映射，原因如下：{exception}"
            )

    def _get_gecko_id(self, crypto_symbol):
        if not self._coinlistings:
            if self._backoff <= datetime.now().timestamp():
                self._load_cryptomap()
                # 仍然未加载成功
                if not self._coinlistings:
                    return None
            else:
                return None
        found = [x for x in self._coinlistings if x["symbol"].lower() == crypto_symbol]

        if crypto_symbol in coingecko_mapping.keys():
            found = [x for x in self._coinlistings if x["id"] == coingecko_mapping[crypto_symbol]]

        if len(found) == 1:
            return found[0]["id"]

        if len(found) > 0:
            # 错误！
            logger.warning(f"在CoinGecko中找到多个{crypto_symbol}的映射。")
            return None

    def convert_amount(self, crypto_amount: float, crypto_symbol: str, fiat_symbol: str) -> float:
        """
        将一定数量的加密货币转换为法币
        :param crypto_amount: 要转换的加密货币数量
        :param crypto_symbol: 使用的加密货币
        :param fiat_symbol: 要转换到的法币
        :return: float，加密货币数量对应的法币价值
        """
        if crypto_symbol == fiat_symbol:
            return float(crypto_amount)
        price = self.get_price(crypto_symbol=crypto_symbol, fiat_symbol=fiat_symbol)
        return float(crypto_amount) * float(price)

    def get_price(self, crypto_symbol: str, fiat_symbol: str) -> float:
        """
        返回加密货币在法币中的价格
        :param crypto_symbol: 要转换的加密货币（例如 BTC）
        :param fiat_symbol: 要转换到的法币（例如 USD）
        :return: 法币中的价格
        """
        crypto_symbol = crypto_symbol.lower()
        fiat_symbol = fiat_symbol.lower()
        inverse = False

        if crypto_symbol == "usd":
            # usd在CoinGecko中对应"uniswap-state-dollar"。
            # 因此，我们需要"交换"货币
            logger.info(f"交换汇率 {crypto_symbol}，{fiat_symbol}")
            crypto_symbol = fiat_symbol
            fiat_symbol = "usd"
            inverse = True

        symbol = f"{crypto_symbol}/{fiat_symbol}"
        # 检查要转换的法币是否受支持
        if not self._is_supported_fiat(fiat=fiat_symbol):
            raise ValueError(f"法币 {fiat_symbol} 不受支持。")

        price = self._pair_price.get(symbol, None)

        if not price:
            price = self._find_price(crypto_symbol=crypto_symbol, fiat_symbol=fiat_symbol)
            if inverse and price != 0.0:
                price = 1 / price
            self._pair_price[symbol] = price

        return price

    def _is_supported_fiat(self, fiat: str) -> bool:
        """
        检查要转换到的法币是否受支持
        :param fiat: 要检查的法币（例如 USD）
        :return: bool，True表示受支持，False表示不支持
        """

        return fiat.upper() in SUPPORTED_FIAT

    def _find_price(self, crypto_symbol: str, fiat_symbol: str) -> float:
        """
        调用CoinGecko API获取法币中的价格
        :param crypto_symbol: 要转换的加密货币（例如 btc）
        :param fiat_symbol: 要转换到的法币（例如 usd）
        :return: float，加密货币在法币中的价格
        """
        # 检查要转换的法币是否受支持
        if not self._is_supported_fiat(fiat=fiat_symbol):
            raise ValueError(f"法币 {fiat_symbol} 不受支持。")

        # 如果加密货币和法币相同，则无需转换
        if crypto_symbol == fiat_symbol:
            return 1.0

        _gecko_id = self._get_gecko_id(crypto_symbol)

        if not _gecko_id:
            # 对于不支持的质押货币返回0（法币转换不应导致机器人中断）
            self.log_once(
                f"不支持的加密货币符号 {crypto_symbol.upper()} - 返回 0.0", logger.warning
            )
            return 0.0

        try:
            return float(
                self._coingecko.get_price(ids=_gecko_id, vs_currencies=fiat_symbol)[_gecko_id][
                    fiat_symbol
                ]
            )
        except Exception as exception:
            logger.error("获取价格时出错: %s", exception)
            return 0.0