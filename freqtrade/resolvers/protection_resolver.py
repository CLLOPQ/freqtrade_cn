"""
这个模块用于加载自定义的保护插件
"""

import logging
from pathlib import Path

from freqtrade.constants import Config
from freqtrade.plugins.protections import IProtection
from freqtrade.resolvers import IResolver


logger = logging.getLogger(__name__)


class ProtectionResolver(IResolver):
    """
    这个类包含加载自定义保护类的所有逻辑
    """

    object_type = IProtection
    object_type_str = "Protection"
    user_subdir = None
    initial_search_path = Path(__file__).parent.parent.joinpath("plugins/protections").resolve()

    @staticmethod
    def load_protection(
        protection_name: str, config: Config, protection_config: dict
    ) -> IProtection:
        """
        使用保护名称加载保护
        :param protection_name: 保护类的类名
        :param config: 配置字典
        :param protection_config: 专用于此保护类的配置
        :return: 初始化的保护类实例
        """
        return ProtectionResolver.load_object(
            protection_name,
            config,
            kwargs={
                "config": config,
                "protection_config": protection_config,
            },
        )