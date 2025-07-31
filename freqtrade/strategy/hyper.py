"""
IHyperStrategy接口，可进行超参数优化的参数类。
本模块定义了自动超参数优化策略的基类。
"""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from freqtrade.constants import Config
from freqtrade.exceptions import OperationalException
from freqtrade.misc import deep_merge_dicts
from freqtrade.optimize.hyperopt_tools import HyperoptTools
from freqtrade.strategy.parameters import BaseParameter


logger = logging.getLogger(__name__)


class HyperStrategyMixin:
    """
    一个辅助基类，允许HyperOptAuto类复用买入/卖出策略逻辑的实现。
    """

    def __init__(self, config: Config, *args, **kwargs):
        """
        初始化可超参数优化的策略混合类。
        """
        self.config = config
        self.ft_buy_params: list[BaseParameter] = []
        self.ft_sell_params: list[BaseParameter] = []
        self.ft_protection_params: list[BaseParameter] = []

        params = self.load_params_from_file()
        params = params.get("params", {})
        self._ft_params_from_file = params
        # 参数的初始化/加载在ft_bot_start()中完成。

    def enumerate_parameters(
        self, category: str | None = None
    ) -> Iterator[tuple[str, BaseParameter]]:
        """
        查找所有可优化参数并返回（名称，属性）迭代器。
        :param category: 参数类别
        :return: 迭代器
        """
        if category not in ("buy", "sell", "protection", None):
            raise OperationalException(
                '类别必须是以下之一："buy"、"sell"、"protection"或None。'
            )

        if category is None:
            params = self.ft_buy_params + self.ft_sell_params + self.ft_protection_params
        else:
            params = getattr(self, f"ft_{category}_params")

        for par in params:
            yield par.name, par

    @classmethod
    def detect_all_parameters(cls) -> dict:
        """检测所有参数并以列表形式返回它们"""
        params: dict[str, Any] = {
            "buy": list(detect_parameters(cls, "buy")),
            "sell": list(detect_parameters(cls, "sell")),
            "protection": list(detect_parameters(cls, "protection")),
        }
        params.update({"count": len(params["buy"] + params["sell"] + params["protection"])})

        return params

    def ft_load_params_from_file(self) -> None:
        """
        从参数文件加载参数
        应/必须在策略解析器中加载配置值之前运行。
        """
        if self._ft_params_from_file:
            # 从超参数优化结果文件中设置参数
            params = self._ft_params_from_file
            self.minimal_roi = params.get("roi", getattr(self, "minimal_roi", {}))

            self.stoploss = params.get("stoploss", {}).get(
                "stoploss", getattr(self, "stoploss", -0.1)
            )
            self.max_open_trades = params.get("max_open_trades", {}).get(
                "max_open_trades", getattr(self, "max_open_trades", -1)
            )
            trailing = params.get("trailing", {})
            self.trailing_stop = trailing.get(
                "trailing_stop", getattr(self, "trailing_stop", False)
            )
            self.trailing_stop_positive = trailing.get(
                "trailing_stop_positive", getattr(self, "trailing_stop_positive", None)
            )
            self.trailing_stop_positive_offset = trailing.get(
                "trailing_stop_positive_offset", getattr(self, "trailing_stop_positive_offset", 0)
            )
            self.trailing_only_offset_is_reached = trailing.get(
                "trailing_only_offset_is_reached",
                getattr(self, "trailing_only_offset_is_reached", 0.0),
            )

    def ft_load_hyper_params(self, hyperopt: bool = False) -> None:
        """
        加载可超参数优化的参数
        优先级：
        * 参数文件中的参数
        * 参数对象中定义的参数（buy_params、sell_params、...）
        * 参数默认值
        """

        buy_params = deep_merge_dicts(
            self._ft_params_from_file.get("buy", {}), getattr(self, "buy_params", {})
        )
        sell_params = deep_merge_dicts(
            self._ft_params_from_file.get("sell", {}), getattr(self, "sell_params", {})
        )
        protection_params = deep_merge_dicts(
            self._ft_params_from_file.get("protection", {}), getattr(self, "protection_params", {})
        )

        self._ft_load_params(buy_params, "buy", hyperopt)
        self._ft_load_params(sell_params, "sell", hyperopt)
        self._ft_load_params(protection_params, "protection", hyperopt)

    def load_params_from_file(self) -> dict:
        filename_str = getattr(self, "__file__", "")
        if not filename_str:
            return {}
        filename = Path(filename_str).with_suffix(".json")

        if filename.is_file():
            logger.info(f"从文件 {filename} 加载参数")
            try:
                params = HyperoptTools.load_params(filename)
                if params.get("strategy_name") != self.__class__.__name__:
                    raise OperationalException("提供的参数文件无效。")
                return params
            except ValueError:
                logger.warning("参数文件格式无效。")
                return {}
        logger.info("未找到参数文件。")

        return {}

    def _ft_load_params(self, params: dict, space: str, hyperopt: bool = False) -> None:
        """
        设置可优化参数的值。
        :param params: 包含新参数值的字典。
        """
        if not params:
            logger.info(f"未找到 {space} 的参数，使用默认值。")
        param_container: list[BaseParameter] = getattr(self, f"ft_{space}_params")

        for attr_name, attr in detect_parameters(self, space):
            attr.name = attr_name
            attr.in_space = hyperopt and HyperoptTools.has_space(self.config, space)
            if not attr.category:
                attr.category = space

            param_container.append(attr)

            if params and attr_name in params:
                if attr.load:
                    attr.value = params[attr_name]
                    logger.info(f"策略参数：{attr_name} = {attr.value}")
                else:
                    logger.warning(
                        f'参数 "{attr_name}" 存在，但已禁用。 '
                        f'使用默认值 "{attr.value}"。'
                    )
            else:
                logger.info(f"策略参数(默认)：{attr_name} = {attr.value}")

    def get_no_optimize_params(self) -> dict[str, dict]:
        """
        返回不属于当前优化任务的参数列表
        """
        params: dict[str, dict] = {
            "buy": {},
            "sell": {},
            "protection": {},
        }
        for name, p in self.enumerate_parameters():
            if p.category and (not p.optimize or not p.in_space):
                params[p.category][name] = p.value
        return params


def detect_parameters(
    obj: HyperStrategyMixin | type[HyperStrategyMixin], category: str
) -> Iterator[tuple[str, BaseParameter]]:
    """
    为“obj”检测“category”的所有参数
    :param obj: 策略对象或类
    :param category: 参数类别 - 通常为`'buy'、'sell'、'protection'、...
    """
    for attr_name in dir(obj):
        if not attr_name.startswith("__"):  # 忽略内部属性，非必需。
            attr = getattr(obj, attr_name)
            if issubclass(attr.__class__, BaseParameter):
                if (
                    attr_name.startswith(category + "_")
                    and attr.category is not None
                    and attr.category != category
                ):
                    raise OperationalException(
                        f"参数名 {attr_name} 不明确，类别：{attr.category}。"
                    )

                if category == attr.category or (
                    attr_name.startswith(category + "_") and attr.category is None
                ):
                    yield attr_name, attr