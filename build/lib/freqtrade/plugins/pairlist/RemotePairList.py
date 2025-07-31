"""
远程交易对列表提供器

提供从远程源获取的交易对列表
"""

import logging
from pathlib import Path
from typing import Any

import rapidjson
import requests
from cachetools import TTLCache

from freqtrade import __version__
from freqtrade.configuration.load_config import CONFIG_PARSE_MODE
from freqtrade.exceptions import OperationalException
from freqtrade.exchange.exchange_types import Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting
from freqtrade.plugins.pairlist.pairlist_helpers import expand_pairlist


logger = logging.getLogger(__name__)


class RemotePairList(IPairList):
    is_pairlist_generator = True
    # 潜在的优胜者偏差
    supports_backtesting = SupportsBacktesting.BIASED

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if "number_assets" not in self._pairlistconfig:
            raise OperationalException(
                "`number_assets`未指定。请检查您的配置中是否有`pairlist.config.number_assets`"
            )

        if "pairlist_url" not in self._pairlistconfig:
            raise OperationalException(
                "`pairlist_url`未指定。请检查您的配置中是否有`pairlist.config.pairlist_url`"
            )

        self._mode = self._pairlistconfig.get("mode", "whitelist")
        self._processing_mode = self._pairlistconfig.get("processing_mode", "filter")
        self._number_pairs = self._pairlistconfig["number_assets"]
        self._refresh_period: int = self._pairlistconfig.get("refresh_period", 1800)
        self._keep_pairlist_on_failure = self._pairlistconfig.get("keep_pairlist_on_failure", True)
        self._pair_cache: TTLCache = TTLCache(maxsize=1, ttl=self._refresh_period)
        self._pairlist_url = self._pairlistconfig.get("pairlist_url", "")
        self._read_timeout = self._pairlistconfig.get("read_timeout", 60)
        self._bearer_token = self._pairlistconfig.get("bearer_token", "")
        self._init_done = False
        self._save_to_file = self._pairlistconfig.get("save_to_file", None)
        self._last_pairlist: list[Any] = list()

        if self._mode not in ["whitelist", "blacklist"]:
            raise OperationalException(
                '`mode`配置不正确。支持的模式为"whitelist"、"blacklist"'
            )

        if self._processing_mode not in ["filter", "append"]:
            raise OperationalException(
                '`processing_mode`配置不正确。支持的模式为"filter"、"append"'
            )

        if self._pairlist_pos == 0 and self._mode == "blacklist":
            raise OperationalException(
                "黑名单模式的RemotePairList不能位于您的交易对列表的第一个位置。"
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
        """
        return f"{self.name} - 来自远程交易对列表的{self._pairlistconfig['number_assets']}个交易对。"

    @staticmethod
    def description() -> str:
        return "从远程API或本地文件检索交易对。"

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "pairlist_url": {
                "type": "string",
                "default": "",
                "description": "用于获取交易对列表的URL",
                "help": "用于获取交易对列表的URL",
            },
            "number_assets": {
                "type": "number",
                "default": 30,
                "description": "资产数量",
                "help": "从交易对列表中使用的资产数量。",
            },
            "mode": {
                "type": "option",
                "default": "whitelist",
                "options": ["whitelist", "blacklist"],
                "description": "交易对列表模式",
                "help": "此交易对列表应作为白名单还是黑名单运行？",
            },
            "processing_mode": {
                "type": "option",
                "default": "filter",
                "options": ["filter", "append"],
                "description": "处理模式",
                "help": "将交易对追加到传入的交易对列表还是过滤它们？",
            },
            **IPairList.refresh_period_parameter(),
            "keep_pairlist_on_failure": {
                "type": "boolean",
                "default": True,
                "description": "失败时保留上次交易对列表",
                "help": "失败时保留上次交易对列表",
            },
            "read_timeout": {
                "type": "number",
                "default": 60,
                "description": "读取超时",
                "help": "远程交易对列表的请求超时时间",
            },
            "bearer_token": {
                "type": "string",
                "default": "",
                "description": "Bearer令牌",
                "help": "Bearer令牌 - 用于对上游服务进行身份验证。",
            },
            "save_to_file": {
                "type": "string",
                "default": "",
                "description": "保存已处理交易对列表的文件名。",
                "help": "指定文件名，以JSON格式保存已处理的交易对列表。",
            },
        }

    def process_json(self, jsonparse) -> list[str]:
        pairlist = jsonparse.get("pairs", [])
        remote_refresh_period = int(jsonparse.get("refresh_period", self._refresh_period))

        if self._refresh_period < remote_refresh_period:
            self.log_once(
                f"刷新周期已从{self._refresh_period}增加到远程指定的最小允许值：{remote_refresh_period}。",
                logger.info,
            )

            self._refresh_period = remote_refresh_period
            self._pair_cache = TTLCache(maxsize=1, ttl=remote_refresh_period)

        self._init_done = True

        return pairlist

    def return_last_pairlist(self) -> list[str]:
        if self._keep_pairlist_on_failure:
            pairlist = self._last_pairlist
            self.log_once("保留上次获取的交易对列表", logger.info)
        else:
            pairlist = []

        return pairlist

    def fetch_pairlist(self) -> tuple[list[str], float]:
        headers = {"User-Agent": "Freqtrade/" + __version__ + " Remotepairlist"}

        if self._bearer_token:
            headers["Authorization"] = f"Bearer {self._bearer_token}"

        try:
            response = requests.get(self._pairlist_url, headers=headers, timeout=self._read_timeout)
            content_type = response.headers.get("content-type")
            time_elapsed = response.elapsed.total_seconds()

            if "application/json" in str(content_type):
                jsonparse = response.json()

                try:
                    pairlist = self.process_json(jsonparse)
                except Exception as e:
                    pairlist = self._handle_error(f"处理JSON数据失败：{type(e)}")
            else:
                pairlist = self._handle_error(
                    f"远程交易对列表不是JSON类型。{self._pairlist_url}"
                )

        except requests.exceptions.RequestException:
            pairlist = self._handle_error(
                f"无法从以下地址获取交易对列表：{self._pairlist_url}"
            )

            time_elapsed = 0

        return pairlist, time_elapsed

    def _handle_error(self, error: str) -> list[str]:
        if self._init_done:
            self.log_once("错误：" + error, logger.info)
            return self.return_last_pairlist()
        else:
            raise OperationalException(error)

    def gen_pairlist(self, tickers: Tickers) -> list[str]:
        """
        生成交易对列表
        :param tickers: 行情数据（来自exchange.get_tickers）。可能已缓存。
        :return: 交易对列表
        """

        if self._init_done:
            pairlist = self._pair_cache.get("pairlist")
            if pairlist == [None]:
                # 有效但为空的交易对列表。
                return []
        else:
            pairlist = []

        time_elapsed = 0.0

        if pairlist:
            # 已找到项目 - 无需刷新
            return pairlist.copy()
        else:
            if self._pairlist_url.startswith("file:///"):
                filename = self._pairlist_url.split("file:///", 1)[1]
                file_path = Path(filename)

                if file_path.exists():
                    with file_path.open() as json_file:
                        try:
                            # 将JSON数据加载到字典中
                            jsonparse = rapidjson.load(json_file, parse_mode=CONFIG_PARSE_MODE)
                            pairlist = self.process_json(jsonparse)
                        except Exception as e:
                            pairlist = self._handle_error(f"处理JSON数据：{type(e)}")
                else:
                    pairlist = self._handle_error(f"{self._pairlist_url}不存在。")

            else:
                # 从远程URL获取交易对列表
                pairlist, time_elapsed = self.fetch_pairlist()

        self.log_once(f"获取的交易对：{pairlist}", logger.debug)

        pairlist = expand_pairlist(pairlist, list(self._exchange.get_markets().keys()))
        pairlist = self._whitelist_for_active_markets(pairlist)
        pairlist = pairlist[: self._number_pairs]

        if pairlist:
            self._pair_cache["pairlist"] = pairlist.copy()
        else:
            # 如果交易对列表为空，设置一个虚拟值以避免再次获取
            self._pair_cache["pairlist"] = [None]

        if time_elapsed != 0.0:
            self.log_once(f"交易对列表在{time_elapsed}秒内获取完成。", logger.info)
        else:
            self.log_once("交易对列表已获取。", logger.info)

        self._last_pairlist = list(pairlist)

        if self._save_to_file:
            self.save_pairlist(pairlist, self._save_to_file)

        return pairlist

    def save_pairlist(self, pairlist: list[str], filename: str) -> None:
        pairlist_data = {"pairs": pairlist}
        try:
            file_path = Path(filename)
            with file_path.open("w") as json_file:
                rapidjson.dump(pairlist_data, json_file)
                logger.info(f"已处理的交易对列表已保存到{filename}")
        except Exception as e:
            logger.error(f"将已处理的交易对列表保存到{filename}时出错：{e}")

    def filter_pairlist(self, pairlist: list[str], tickers: dict) -> list[str]:
        """
        过滤并排序交易对列表，然后再次返回白名单。
        在每个机器人迭代时调用 - 如果需要，请使用内部缓存
        :param pairlist: 要过滤或排序的交易对列表
        :param tickers: 行情数据（来自exchange.get_tickers）。可能已缓存。
        :return: 新的白名单
        """
        rpl_pairlist = self.gen_pairlist(tickers)
        merged_list = []
        filtered = []

        if self._mode == "whitelist":
            if self._processing_mode == "filter":
                merged_list = [pair for pair in pairlist if pair in rpl_pairlist]
            elif self._processing_mode == "append":
                merged_list = pairlist + rpl_pairlist
            merged_list = sorted(set(merged_list), key=merged_list.index)
        else:
            for pair in pairlist:
                if pair not in rpl_pairlist:
                    merged_list.append(pair)
                else:
                    filtered.append(pair)
            if filtered:
                self.log_once(f"黑名单 - 过滤掉的交易对：{filtered}", logger.info)

        merged_list = merged_list[: self._number_pairs]
        return merged_list