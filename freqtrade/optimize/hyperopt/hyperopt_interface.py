"""
IHyperOpt 接口
此模块定义了用于超优化（hyperopt）必须应用的接口
"""

import logging
import math
from abc import ABC
from typing import TypeAlias

from optuna.samplers import BaseSampler

from freqtrade.constants import Config
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.misc import round_dict
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
from freqtrade.strategy import IStrategy


logger = logging.getLogger(__name__)

EstimatorType: TypeAlias = BaseSampler | str


class IHyperOpt(ABC):
    """
    freqtrade 超参数优化的接口
    定义了任何自定义超参数优化必须遵循的强制结构

    您可以使用的类属性：
        timeframe -> int: 策略使用的时间框架值
    """

    timeframe: str
    strategy: IStrategy

    def __init__(self, config: Config) -> None:
        self.config = config

        # 分配将在超参数优化中使用的时间框架
        IHyperOpt.timeframe = str(config["timeframe"])

    def generate_estimator(self, dimensions: list[Dimension], **kwargs) -> EstimatorType:
        """
        返回基础估计器。
        可以是 "TPESampler"、"GPSampler"、"CmaEsSampler"、"NSGAIISampler"、
        "NSGAIIISampler"、"QMCSampler" 中的任何一个，或者是继承自 BaseSampler 的类的实例
        （来自 optuna.samplers）。
        """
        return "NSGAIIISampler"

    def generate_roi_table(self, params: dict) -> dict[int, float]:
        """
        创建 ROI 表。

        生成将被 Hyperopt 使用的 ROI 表。
        您可以在自定义的 Hyperopt 类中重写它。
        """
        roi_table = {}
        roi_table[0] = params["roi_p1"] + params["roi_p2"] + params["roi_p3"]
        roi_table[params["roi_t3"]] = params["roi_p1"] + params["roi_p2"]
        roi_table[params["roi_t3"] + params["roi_t2"]] = params["roi_p1"]
        roi_table[params["roi_t3"] + params["roi_t2"] + params["roi_t1"]] = 0

        return roi_table

    def roi_space(self) -> list[Dimension]:
        """
        创建 ROI 空间。

        定义每个 ROI 步骤要搜索的值。

        此方法实现了自适应的 ROI 超空间，具有变化的参数范围，可自动适应所使用的时间框架。

        如果没有定义自定义的 roi_space 方法，Freqtrade 会默认使用它。
        """

        # ROI 超空间的默认缩放系数。可以更改以调整 ROI 表的结果范围。
        # 如果需要更宽的 ROI 超空间范围，则增加；如果需要更短的范围，则减少。
        roi_t_alpha = 1.0
        roi_p_alpha = 1.0

        timeframe_min = timeframe_to_minutes(self.timeframe)

        # 我们在这里定义了 ROI 空间参数的限制，自动适应机器人使用的时间框架：
        #
        # * 'roi_t'（ROI 表中的时间间隔限制）组件是线性缩放的。
        # * 'roi_p'（ROI 值步骤的限制）组件是对数缩放的。
        #
        # 缩放设计为对于 5m 时间框架，它与传统的 Freqtrade roi_space() 方法完全映射。
        roi_t_scale = timeframe_min / 5
        roi_p_scale = math.log1p(timeframe_min) / math.log1p(5)
        roi_limits = {
            "roi_t1_min": int(10 * roi_t_scale * roi_t_alpha),
            "roi_t1_max": int(120 * roi_t_scale * roi_t_alpha),
            "roi_t2_min": int(10 * roi_t_scale * roi_t_alpha),
            "roi_t2_max": int(60 * roi_t_scale * roi_t_alpha),
            "roi_t3_min": int(10 * roi_t_scale * roi_t_alpha),
            "roi_t3_max": int(40 * roi_t_scale * roi_t_alpha),
            "roi_p1_min": 0.01 * roi_p_scale * roi_p_alpha,
            "roi_p1_max": 0.04 * roi_p_scale * roi_p_alpha,
            "roi_p2_min": 0.01 * roi_p_scale * roi_p_alpha,
            "roi_p2_max": 0.07 * roi_p_scale * roi_p_alpha,
            "roi_p3_min": 0.01 * roi_p_scale * roi_p_alpha,
            "roi_p3_max": 0.20 * roi_p_scale * roi_p_alpha,
        }
        logger.debug(f"使用 ROI 空间限制：{roi_limits}")
        p = {
            "roi_t1": roi_limits["roi_t1_min"],
            "roi_t2": roi_limits["roi_t2_min"],
            "roi_t3": roi_limits["roi_t3_min"],
            "roi_p1": roi_limits["roi_p1_min"],
            "roi_p2": roi_limits["roi_p2_min"],
            "roi_p3": roi_limits["roi_p3_min"],
        }
        logger.info(f"最小 ROI 表：{round_dict(self.generate_roi_table(p), 3)}")
        p = {
            "roi_t1": roi_limits["roi_t1_max"],
            "roi_t2": roi_limits["roi_t2_max"],
            "roi_t3": roi_limits["roi_t3_max"],
            "roi_p1": roi_limits["roi_p1_max"],
            "roi_p2": roi_limits["roi_p2_max"],
            "roi_p3": roi_limits["roi_p3_max"],
        }
        logger.info(f"最大 ROI 表：{round_dict(self.generate_roi_table(p), 3)}")

        return [
            Integer(roi_limits["roi_t1_min"], roi_limits["roi_t1_max"], name="roi_t1"),
            Integer(roi_limits["roi_t2_min"], roi_limits["roi_t2_max"], name="roi_t2"),
            Integer(roi_limits["roi_t3_min"], roi_limits["roi_t3_max"], name="roi_t3"),
            SKDecimal(
                roi_limits["roi_p1_min"], roi_limits["roi_p1_max"], decimals=3, name="roi_p1"
            ),
            SKDecimal(
                roi_limits["roi_p2_min"], roi_limits["roi_p2_max"], decimals=3, name="roi_p2"
            ),
            SKDecimal(
                roi_limits["roi_p3_min"], roi_limits["roi_p3_max"], decimals=3, name="roi_p3"
            ),
        ]

    def stoploss_space(self) -> list[Dimension]:
        """
        创建止损空间。

        定义要搜索的止损值范围。
        您可以在自定义的 Hyperopt 类中重写它。
        """
        return [
            SKDecimal(-0.35, -0.02, decimals=3, name="stoploss"),
        ]

    def generate_trailing_params(self, params: dict) -> dict:
        """
        创建带有追踪止损参数的字典。
        """
        return {
            "trailing_stop": params["trailing_stop"],
            "trailing_stop_positive": params["trailing_stop_positive"],
            "trailing_stop_positive_offset": (
                params["trailing_stop_positive"] + params["trailing_stop_positive_offset_p1"]
            ),
            "trailing_only_offset_is_reached": params["trailing_only_offset_is_reached"],
        }

    def trailing_space(self) -> list[Dimension]:
        """
        创建追踪止损空间。

        您可以在自定义的 Hyperopt 类中重写它。
        """
        return [
            # 已决定如果使用 'trailing' 超空间，总是将 trailing_stop 设置为 True。
            # 否则，超参数优化将改变其他参数，如果 trailing_stop 设置为 False，这些参数将不会生效。
            # 此参数包含在超空间维度中，而不是在代码中显式分配，以便与其他 'trailing' 超空间参数一起打印在结果中。
            Categorical([True], name="trailing_stop"),
            SKDecimal(0.01, 0.35, decimals=3, name="trailing_stop_positive"),
            # 'trailing_stop_positive_offset' 应该大于 'trailing_stop_positive'，
            # 因此这个中间参数用作它们之间差异的值。'trailing_stop_positive_offset' 的值在
            # generate_trailing_params() 方法中构造。
            # 这类似于用于构造 ROI 表的超空间维度。
            SKDecimal(0.001, 0.1, decimals=3, name="trailing_stop_positive_offset_p1"),
            Categorical([True, False], name="trailing_only_offset_is_reached"),
        ]

    def max_open_trades_space(self) -> list[Dimension]:
        """
        创建最大开仓交易空间。

        您可以在自定义的 Hyperopt 类中重写它。
        """
        return [
            Integer(-1, 10, name="max_open_trades"),
        ]

    # 这是正确反序列化类属性 timeframe 所必需的，该属性由解析器设置为实际值。
    # 为什么在现代 Python 中我还需要这种不可思议的咒语？
    def __getstate__(self):
        state = self.__dict__.copy()
        state["timeframe"] = self.timeframe
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        IHyperOpt.timeframe = state["timeframe"]