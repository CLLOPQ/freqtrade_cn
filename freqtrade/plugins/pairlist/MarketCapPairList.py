"""
市值交易对列表提供器

基于市值提供动态交易对列表
"""

import logging
import math

from cachetools import TTLCache

from freqtrade.exceptions import OperationalException
from freqtrade.exchange.exchange_types import Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting
from freqtrade.util.coin_gecko import FtCoinGeckoApi


logger = logging.getLogger(__name__)


class MarketCapPairList(IPairList):
    is_pairlist_generator = True
    supports_backtesting = SupportsBacktesting.BIASED

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if "number_assets" not in self._pairlistconfig:
            raise OperationalException(
                "`number_assets`未指定。请检查您的配置中是否存在“pairlist.config.number_assets”"
            )

        self._stake_currency = self._config["stake_currency"]
        self._number_assets = self._pairlistconfig["number_assets"]
        self._max_rank = self._pairlistconfig.get("max_rank", 30)
        self._refresh_period = self._pairlistconfig.get("refresh_period", 86400)
        self._categories = self._pairlistconfig.get("categories", [])
        self._marketcap_cache: TTLCache = TTLCache(maxsize=1, ttl=self._refresh_period)
        self._def_candletype = self._config["candle_type_def"]

        _coingecko_config = self._config.get("coingecko", {})

        self._coingecko: FtCoinGeckoApi = FtCoinGeckoApi(
            api_key=_coingecko_config.get("api_key", ""),
            is_demo=_coingecko_config.get("is_demo", True),
        )

        if self._categories:
            categories = self._coingecko.get_coins_categories_list()
            category_ids = [cat["category_id"] for cat in categories]

            for category in self._categories:
                if category not in category_ids:
                    raise OperationalException(
                        f"分类{category}不在CoinGecko的分类列表中。您可以从{category_ids}中选择"
                    )

        if self._max_rank > 250:
            self.logger.warning(
                f"您设置的max_rank值({self._max_rank})较高。这可能导致CoinGecko API的速率限制问题。请确保此值对您的使用场景是必要的。",
            )

    @property
    def needstickers(self) -> bool:
        """
        定义是否需要行情数据的布尔属性。如果没有交易对列表需要行情数据，则将空字典作为行情数据参数传递给filter_pairlist方法
        """
        return False

    def short_desc(self) -> str:
        """
        简短的白名单方法描述 - 用于启动消息
        """
        num = self._number_assets
        rank = self._max_rank
        msg = f"{self.name} - 在市值排名前{rank}的交易对中选择{num}个交易对。"
        return msg

    @staticmethod
    def description() -> str:
        return "基于CoinGecko的市值排名提供交易对列表。"

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "number_assets": {
                "type": "number",
                "default": 30,
                "description": "资产数量",
                "help": "从交易对列表中使用的资产数量",
            },
            "max_rank": {
                "type": "number",
                "default": 30,
                "description": "资产最大排名",
                "help": "从交易对列表中使用的资产最大排名",
            },
            "categories": {
                "type": "list",
                "default": [],
                "description": "币种分类",
                "help": (
                    "币种的分类，例如Layer-1，默认为空列表 "
                    "(https://www.coingecko.com/zh-cn/categories)"
                ),
            },
            "refresh_period": {
                "type": "number",
                "default": 86400,
                "description": "刷新周期",
                "help": "刷新周期（秒）",
            },
        }

    def gen_pairlist(self, tickers: Tickers) -> list[str]:
        """
        生成交易对列表
        :param tickers: 行情数据（来自交易所.get_tickers()）。可能已缓存。
        :return: 交易对列表
        """
        # 生成动态白名单
        # 如果此交易对列表是第一个，则必须始终运行。
        pairlist = self._marketcap_cache.get("pairlist_mc")
        if pairlist:
            # 找到项目 - 无需刷新
            return pairlist.copy()
        else:
            # 使用新的交易对列表
            # 检查交易对计价货币是否等于持仓货币。
            _pairlist = [
                k
                for k in self._exchange.get_markets(
                    quote_currencies=[self._stake_currency], tradable_only=True, active_only=True
                ).keys()
            ]
            # 测试黑名单没有意义...
            _pairlist = self.verify_blacklist(_pairlist, logger.info)

            pairlist = self.filter_pairlist(_pairlist, tickers)
            self._marketcap_cache["pairlist_mc"] = pairlist.copy()

        return pairlist

    def filter_pairlist(self, pairlist: list[str], tickers: dict) -> list[str]:
        """
        过滤并排序交易对列表，然后再次返回白名单。在每个机器人迭代时调用 - 如果需要，请使用内部缓存
        :param pairlist: 要过滤或排序的交易对列表
        :param tickers: 行情数据（来自交易所.get_tickers()）。可能已缓存。
        :return: 新的白名单
        """
        marketcap_list = self._marketcap_cache.get("marketcap")

        default_kwargs = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": "250",
            "page": "1",
            "sparkline": "false",
            "locale": "en",
        }

        if marketcap_list is None:
            data = []

            if not self._categories:
                pages_required = math.ceil(self._max_rank / 250)
                for page in range(1, pages_required + 1):
                    default_kwargs["page"] = str(page)
                    page_data = self._coingecko.get_coins_markets(**default_kwargs)
                    data.extend(page_data)
            else:
                for category in self._categories:
                    category_data = self._coingecko.get_coins_markets(
                        **default_kwargs, **({"category": category} if category else {})
                    )
                    data += category_data

            data.sort(key=lambda d: float(d.get("market_cap") or 0.0), reverse=True)

            if data:
                marketcap_list = [row["symbol"] for row in data]
                self._marketcap_cache["marketcap"] = marketcap_list

        if marketcap_list:
            filtered_pairlist = []

            market = self._config["trading_mode"]
            pair_format = f"{self._stake_currency.upper()}"
            if market == "futures":
                pair_format += f":{self._stake_currency.upper()}"

            top_marketcap = marketcap_list[: self._max_rank :]

            for mc_pair in top_marketcap:
                test_pair = f"{mc_pair.upper()}/{pair_format}"
                if test_pair in pairlist and test_pair not in filtered_pairlist:
                    filtered_pairlist.append(test_pair)
                    if len(filtered_pairlist) == self._number_assets:
                        break

            if len(filtered_pairlist) > 0:
                return filtered_pairlist

        # 如果未找到交易对，返回原始交易对列表
        return []