"""
交易对列表管理器类
"""

import logging
from functools import partial

from cachetools import TTLCache, cached

from freqtrade.constants import Config, ListPairsWithTimeframes
from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import CandleType
from freqtrade.enums.runmode import RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange.exchange_types import Tickers
from freqtrade.mixins import LoggingMixin
from freqtrade.plugins.pairlist.IPairList import IPairList, SupportsBacktesting
from freqtrade.plugins.pairlist.pairlist_helpers import expand_pairlist
from freqtrade.resolvers import PairListResolver


logger = logging.getLogger(__name__)


class PairListManager(LoggingMixin):
    def __init__(self, exchange, config: Config, dataprovider: DataProvider | None = None) -> None:
        self._exchange = exchange
        self._config = config
        self._whitelist = self._config["exchange"].get("pair_whitelist")
        self._blacklist = self._config["exchange"].get("pair_blacklist", [])
        self._pairlist_handlers: list[IPairList] = []
        self._tickers_needed = False
        self._dataprovider: DataProvider | None = dataprovider
        for pairlist_handler_config in self._config.get("pairlists", []):
            pairlist_handler = PairListResolver.load_pairlist(
                pairlist_handler_config["method"],
                exchange=exchange,
                pairlistmanager=self,
                config=config,
                pairlistconfig=pairlist_handler_config,
                pairlist_pos=len(self._pairlist_handlers),
            )
            self._tickers_needed |= pairlist_handler.needstickers
            self._pairlist_handlers.append(pairlist_handler)

        if not self._pairlist_handlers:
            raise OperationalException("未定义交易对列表处理器")

        if self._tickers_needed and not self._exchange.exchange_has("fetchTickers"):
            invalid = ". ".join([p.name for p in self._pairlist_handlers if p.needstickers])

            raise OperationalException(
                "交易所不支持fetchTickers，因此以下交易对列表无法使用。请编辑您的配置并重启机器人。\n"
                f"{invalid}。"
            )

        self._check_backtest()

        refresh_period = config.get("pairlist_refresh_period", 3600)
        LoggingMixin.__init__(self, logger, refresh_period)

    def _check_backtest(self) -> None:
        """
        检查是否在回测或超参数优化模式
        """
        if self._config["runmode"] not in (RunMode.BACKTEST, RunMode.HYPEROPT):
            return

        pairlist_errors: list[str] = []
        noaction_pairlists: list[str] = []
        biased_pairlists: list[str] = []
        for pairlist_handler in self._pairlist_handlers:
            if pairlist_handler.supports_backtesting == SupportsBacktesting.NO:
                pairlist_errors.append(pairlist_handler.name)
            if pairlist_handler.supports_backtesting == SupportsBacktesting.NO_ACTION:
                noaction_pairlists.append(pairlist_handler.name)
            if pairlist_handler.supports_backtesting == SupportsBacktesting.BIASED:
                biased_pairlists.append(pairlist_handler.name)

        if noaction_pairlists:
            logger.warning(
                f"交易对列表处理器 {', '.join(noaction_pairlists)} 在回测期间不会生成"
                "任何变化。虽然启用它们是安全的，但它们的行为将与模拟/实盘模式不同。 "
            )

        if biased_pairlists:
            logger.warning(
                f"交易对列表处理器 {', '.join(biased_pairlists)} 将为您的回测结果引入前瞻偏差"
                "，因为它们使用今日数据 - 这固有地存在‘胜利者偏差’。"
            )
        if pairlist_errors:
            raise OperationalException(
                f"交易对列表处理器 {', '.join(pairlist_errors)} 不支持回测。"
            )

    @property
    def whitelist(self) -> list[str]:
        """当前的交易对白名单"""
        return self._whitelist

    @property
    def blacklist(self) -> list[str]:
        """
        当前的交易对黑名单
        -> 无需在子类中重写
        """
        return self._blacklist

    @property
    def expanded_blacklist(self) -> list[str]:
        """已展开的黑名单（包括通配符扩展）"""
        return expand_pairlist(self._blacklist, self._exchange.get_markets().keys())

    @property
    def name_list(self) -> list[str]:
        """获取已加载的交易对列表处理器名称列表"""
        return [p.name for p in self._pairlist_handlers]

    def short_desc(self) -> list[dict]:
        """每个交易对列表处理器的简短描述列表"""
        return [{p.name: p.short_desc()} for p in self._pairlist_handlers]

    @cached(TTLCache(maxsize=1, ttl=1800))
    def _get_cached_tickers(self) -> Tickers:
        return self._exchange.get_tickers()

    def refresh_pairlist(self) -> None:
        """通过所有已配置的交易对列表处理器运行交易对列表。"""
        # 为避免每次调用都请求交易所，应缓存 tickers
        tickers: dict = {}
        if self._tickers_needed:
            tickers = self._get_cached_tickers()

        # 使用链中的第一个交易对列表处理器生成交易对列表
        pairlist = self._pairlist_handlers[0].gen_pairlist(tickers)

        # 处理链中除第一个生成器外的所有交易对列表处理器
        for pairlist_handler in self._pairlist_handlers[1:]:
            pairlist = pairlist_handler.filter_pairlist(pairlist, tickers)

        # 在交易对列表处理器链之后进行黑名单验证
        # 以确保黑名单被正确遵守。
        pairlist = self.verify_blacklist(pairlist, logger.warning)

        self.log_once(f"白名单包含 {len(pairlist)} 个交易对：{pairlist}", logger.info)

        self._whitelist = pairlist

    def verify_blacklist(self, pairlist: list[str], logmethod) -> list[str]:
        """
        验证并从交易对列表中移除项目，返回过滤后的交易对列表。
        根据`aswarning`决定记录警告或信息。
        显式使用此方法的交易对列表处理器应使用
        `logmethod=logger.info`以避免警告信息过多
        :param pairlist: 要验证的交易对列表
        :param logmethod: 将被调用的函数，`logger.info`或`logger.warning`。
        :return: 交易对列表 - 已被黑名单过滤的交易对
        """
        try:
            blacklist = self.expanded_blacklist
        except ValueError as err:
            logger.error(f"交易对黑名单包含无效的通配符：{err}")
            return []
        log_once = partial(self.log_once, logmethod=logmethod)
        for pair in pairlist.copy():
            if pair in blacklist:
                log_once(f"交易对 {pair} 在您的黑名单中。将其从白名单中移除...")
                pairlist.remove(pair)
        return pairlist

    def verify_whitelist(
        self, pairlist: list[str], logmethod, keep_invalid: bool = False
    ) -> list[str]:
        """
        验证并从交易对列表中移除项目，返回过滤后的交易对列表。
        根据`aswarning`决定记录警告或信息。
        显式使用此方法的交易对列表处理器应使用
        `logmethod=logger.info`以避免警告信息过多
        :param pairlist: 要验证的交易对列表
        :param logmethod: 将被调用的函数，`logger.info`或`logger.warning`
        :param keep_invalid: 如果设为True，在扩展正则表达式时静默丢弃无效交易对。
        :return: 交易对列表 - 已被白名单过滤的交易对
        """
        try:
            whitelist = expand_pairlist(pairlist, self._exchange.get_markets().keys(), keep_invalid)
        except ValueError as err:
            logger.error(f"交易对白名单包含无效的通配符：{err}")
            return []
        return whitelist

    def create_pair_list(
        self, pairs: list[str], timeframe: str | None = None
    ) -> ListPairsWithTimeframes:
        """创建包含（交易对，时间周期）元组的列表"""
        return [
            (
                pair,
                timeframe or self._config["timeframe"],
                self._config.get("candle_type_def", CandleType.SPOT),
            )
            for pair in pairs
        ]