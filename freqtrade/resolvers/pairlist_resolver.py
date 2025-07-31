# pragma pylint: disable=attribute-defined-outside-init

"""
这个模块加载自定义的交易对列表
"""

import logging
from pathlib import Path

from freqtrade.constants import Config
from freqtrade.plugins.pairlist.IPairList import IPairList
from freqtrade.resolvers import IResolver


logger = logging.getLogger(__name__)


class PairListResolver(IResolver):
    """
    这个类包含加载自定义交易对列表类的所有逻辑
    """

    object_type = IPairList
    object_type_str = "交易对列表"
    user_subdir = None
    initial_search_path = Path(__file__).parent.parent.joinpath("plugins/pairlist").resolve()

    @staticmethod
    def load_pairlist(
        pairlist_name: str,
        exchange,
        pairlistmanager,
        config: Config,
        pairlistconfig: dict,
        pairlist_pos: int,
    ) -> IPairList:
        """
        使用交易对列表名称加载交易对列表
        :param pairlist_name: 交易对列表的类名
        :param exchange: 已初始化的交易所类
        :param pairlistmanager: 已初始化的交易对列表管理器
        :param config: 配置字典
        :param pairlistconfig: 专用于此交易对列表的配置
        :param pairlist_pos: 交易对列表在交易对列表列表中的位置
        :return: 已初始化的交易对列表类
        """
        return PairListResolver.load_object(
            pairlist_name,
            config,
            kwargs={
                "exchange": exchange,
                "pairlistmanager": pairlistmanager,
                "config": config,
                "pairlistconfig": pairlistconfig,
                "pairlist_pos": pairlist_pos,
            },
        )