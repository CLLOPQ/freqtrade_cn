"""
策略接口
此模块定义了适用于策略的接口
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from math import isinf, isnan

from pandas import DataFrame
from pydantic import ValidationError

from freqtrade.constants import CUSTOM_TAG_MAX_LENGTH, Config, IntOrInf, ListPairsWithTimeframes
from freqtrade.data.converter import populate_dataframe_with_trades
from freqtrade.data.converter.converter import reduce_dataframe_footprint
from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import (
    CandleType,
    ExitCheckTuple,
    ExitType,
    MarketDirection,
    RunMode,
    SignalDirection,
    SignalTagType,
    SignalType,
    TradingMode,
)
from freqtrade.exceptions import OperationalException, StrategyError
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_next_date, timeframe_to_seconds
from freqtrade.ft_types import AnnotationType
from freqtrade.misc import remove_entry_exit_signals
from freqtrade.persistence import Order, PairLocks, Trade
from freqtrade.strategy.hyper import HyperStrategyMixin
from freqtrade.strategy.informative_decorator import (
    InformativeData,
    PopulateIndicators,
    _create_and_merge_informative_pair,
    _format_pair_name,
)
from freqtrade.strategy.strategy_validation import StrategyResultValidator
from freqtrade.strategy.strategy_wrapper import strategy_safe_wrapper
from freqtrade.util import dt_now
from freqtrade.wallets import Wallets


logger = logging.getLogger(__name__)


class IStrategy(ABC, HyperStrategyMixin):
    """
    Freqtrade 策略接口
    定义了任何自定义策略必须遵循的强制结构

    可使用的属性:
        minimal_roi -> Dict: 为策略设计的最小投资回报率
        stoploss -> float: 为策略设计的最佳止损
        timeframe -> str: 策略使用的时间周期值
    """

    # 策略接口版本
    # 默认为版本 2
    # 版本 1 是没有元数据字典的初始接口 - 已弃用且不再支持。
    # 版本 2 populate_* 包含元数据字典
    # 版本 3 - 第一个支持做空和杠杆的版本
    INTERFACE_VERSION: int = 3

    _ft_params_from_file: dict
    # 关联的最小投资回报率
    minimal_roi: dict = {}
    use_custom_roi: bool = False

    # 关联的止损
    stoploss: float

    # 策略最大开仓交易数
    max_open_trades: IntOrInf

    # 追踪止损
    trailing_stop: bool = False
    trailing_stop_positive: float | None = None
    trailing_stop_positive_offset: float = 0.0
    trailing_only_offset_is_reached = False
    use_custom_stoploss: bool = False

    # 该策略能否做空?
    can_short: bool = False

    # 关联的时间周期
    timeframe: str

    # 可选订单类型
    order_types: dict = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "limit",
        "stoploss_on_exchange": False,
        "stoploss_on_exchange_interval": 60,
    }

    # 可选时间有效性
    order_time_in_force: dict = {
        "entry": "GTC",
        "exit": "GTC",
    }

    # 仅在新蜡烛图上运行 "populate_indicators"
    process_only_new_candles: bool = True

    use_exit_signal: bool
    exit_profit_only: bool
    exit_profit_offset: float
    ignore_roi_if_entry_signal: bool

    # 仓位调整默认禁用
    position_adjustment_enable: bool = False
    max_entry_position_adjustment: int = -1

    # 蜡烛图过期后，不再产生买入信号的秒数
    ignore_buying_expired_candle_after: int = 0

    # 禁用数据帧检查 (将错误转换为警告消息)
    disable_dataframe_checks: bool = False

    # 策略生成有效信号前所需的蜡烛图数量
    startup_candle_count: int = 0

    # 保护措施
    protections: list = []

    # 类级别变量 (有意为之) 包含
    # dataprovider (dp) (访问其他蜡烛图、历史数据等)
    # 和 wallets - 访问当前余额。
    dp: DataProvider
    wallets: Wallets | None = None
    # 从配置中填充
    stake_currency: str
    # 策略源代码的容器变量
    __source__: str = ""
    __file__: str = ""

    # plot_config 的定义。详情请参阅绘图文档。
    plot_config: dict = {}

    # 一个表示市场方向的自设参数。从配置中填充
    market_direction: MarketDirection = MarketDirection.NONE

    # 全局缓存字典
    _cached_grouped_trades_per_pair: dict[str, DataFrame] = {}

    def __init__(self, config: Config) -> None:
        self.config = config
        # 用于判断是否需要分析的字典
        self._last_candle_seen_per_pair: dict[str, datetime] = {}
        super().__init__(config)

        # 从 @informative 装饰器修饰的方法中收集信息对。
        self._ft_informative: list[tuple[InformativeData, PopulateIndicators]] = []
        for attr_name in dir(self.__class__):
            cls_method = getattr(self.__class__, attr_name)
            if not callable(cls_method):
                continue
            informative_data_list = getattr(cls_method, "_ft_informative", None)
            if not isinstance(informative_data_list, list):
                # 需要进行类型检查，因为 mocker 会返回一个评估为 True 的模拟对象，导致代码混淆。
                continue
            strategy_timeframe_minutes = timeframe_to_minutes(self.timeframe)
            for informative_data in informative_data_list:
                if timeframe_to_minutes(informative_data.timeframe) < strategy_timeframe_minutes:
                    raise OperationalException(
                        "信息时间周期必须等于或高于策略时间周期!"
                    )
                if not informative_data.candle_type:
                    informative_data.candle_type = config["candle_type_def"]
                self._ft_informative.append((informative_data, cls_method))

    def load_freqAI_model(self) -> None:
        if self.config.get("freqai", {}).get("enabled", False):
            # 如果 freqAI 未启用，则在此处导入以避免导入
            from freqtrade.freqai.utils import download_all_data_for_training
            from freqtrade.resolvers.freqaimodel_resolver import FreqaiModelResolver

            self.freqai = FreqaiModelResolver.load_freqaimodel(self.config)
            self.freqai_info = self.config["freqai"]

            # 在 dry/live 模式下下载所需数据
            if self.config.get("runmode") in (RunMode.DRY_RUN, RunMode.LIVE):
                logger.info(
                    "正在为白名单和 corr_pairlist 中的所有交易对下载所有训练数据，"
                    "如果数据尚未在磁盘上，这可能需要一些时间。"
                )
                download_all_data_for_training(self.dp, self.config)
        else:
            # 如果 freqAI 被禁用但调用了 "start"，则进行优雅失败。
            class DummyClass:
                def start(self, *args, **kwargs):
                    raise OperationalException(
                        "freqAI 未启用。请在您的配置中启用它以使用此策略。"
                    )

                def shutdown(self, *args, **kwargs):
                    pass

            self.freqai = DummyClass()  # type: ignore

    def ft_bot_start(self, **kwargs) -> None:
        """
        策略初始化 - 在 dataprovider 添加后运行。
        必须调用 bot_start()
        """
        self.load_freqAI_model()

        strategy_safe_wrapper(self.bot_start)()

        self.ft_load_hyper_params(self.config.get("runmode") == RunMode.HYPEROPT)

    def ft_bot_cleanup(self) -> None:
        """
        清理 FreqAI 和子线程
        """
        self.freqai.shutdown()

    @abstractmethod
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        填充将在买入、卖出、做空、平仓策略中使用的指标
        :param dataframe: 包含来自交易所的数据的 DataFrame
        :param metadata: 附加信息，例如当前交易对
        :return: 包含策略所有强制性指标的 Dataframe
        """
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        已弃用 - 请迁移到 populate_entry_trend
        :param dataframe: DataFrame
        :param metadata: 附加信息，例如当前交易对
        :return: 带有买入列的 DataFrame
        """
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        根据技术分析指标，填充给定数据帧的入场信号
        :param dataframe: DataFrame
        :param metadata: 附加信息，例如当前交易对
        :return: 填充了入场列的 DataFrame
        """
        return self.populate_buy_trend(dataframe, metadata)

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        已弃用 - 请迁移到 populate_exit_trend
        根据技术分析指标，填充给定数据帧的卖出信号
        :param dataframe: DataFrame
        :param metadata: 附加信息，例如当前交易对
        :return: 带有卖出列的 DataFrame
        """
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        根据技术分析指标，填充给定数据帧的出场信号
        :param dataframe: DataFrame
        :param metadata: 附加信息，例如当前交易对
        :return: 填充了出场列的 DataFrame
        """
        return self.populate_sell_trend(dataframe, metadata)

    def bot_start(self, **kwargs) -> None:
        """
        仅在机器人实例化后调用一次。
        :param **kwargs: 确保保留此参数，以便更新不会破坏您的策略。
        """
        pass

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """
        在机器人迭代（一个循环）开始时调用。
        可能用于执行与交易对无关的任务
        （例如，为比较收集一些远程资源）
        :param current_time: datetime 对象，包含当前日期时间
        :param **kwargs: 确保保留此参数，以便更新不会破坏您的策略。
        """
        pass

    def check_buy_timeout(
        self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs
    ) -> bool:
        """
        已弃用: 请改用 `check_entry_timeout`。
        """
        return False

    def check_entry_timeout(
        self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs
    ) -> bool:
        """
        检查入场超时回调函数。
        此方法可用于覆盖入场超时。
        当限价入场订单已创建但尚未完全成交时调用。
        `unfilledtimeout` 中的配置选项将在此之前验证，
        因此请确保将这些超时设置得足够高。

        如果策略未实现此方法，则简单返回 False。
        :param pair: 交易对
        :param trade: Trade 对象。
        :param order: Order 对象。
        :param current_time: datetime 对象，包含当前日期时间
        :param **kwargs: 确保保留此参数，以便更新不会破坏您的策略。
        :return bool: 当返回 True 时，入场订单将被取消。
        """
        return self.check_buy_timeout(
            pair=pair, trade=trade, order=order, current_time=current_time
        )

    def check_sell_timeout(
        self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs
    ) -> bool:
        """
        已弃用: 请改用 `check_exit_timeout`。
        """
        return False

    def check_exit_timeout(
        self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs
    ) -> bool:
        """
        检查出场超时回调函数。
        此方法可用于覆盖出场超时。
        当限价出场订单已创建但尚未完全成交时调用。
        `unfilledtimeout` 中的配置选项将在此之前验证，
        因此请确保将这些超时设置得足够高。

        如果策略未实现此方法，则简单返回 False。
        :param pair: 交易对
        :param trade: Trade 对象。
        :param order: Order 对象
        :param current_time: datetime 对象，包含当前日期时间
        :param **kwargs: 确保保留此参数，以便更新不会破坏您的策略。
        :return bool: 当返回 True 时，出场订单将被取消。
        """
        return self.check_sell_timeout(
            pair=pair, trade=trade, order=order, current_time=current_time
        )

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        """
        在下入场订单之前调用。
        此函数的时间点至关重要，因此避免在此方法中执行繁重的计算或网络请求。

        完整的文档请访问 https://www.freqtrade.io/en/latest/strategy-advanced/

        如果策略未实现此方法，则返回 True（始终确认）。

        :param pair: 即将买入/做空的交易对。
        :param order_type: 订单类型（根据 order_types 配置）。通常是限价或市价。
        :param amount: 目标（基础）货币中将要交易的数量。
        :param rate: 限价订单将使用的价格，或市价订单的当前价格。
        :param time_in_force: 时间有效性。默认为 GTC (Good-til-cancelled)。
        :param current_time: datetime 对象，包含当前日期时间
        :param entry_tag: 可选的入场标签 (buy_tag)，如果随买入信号提供。
        :param side: 'long' (多头) 或 'short' (空头) - 表示建议交易的方向
        :param **kwargs: 确保保留此参数，以便更新不会破坏您的策略。
        :return bool: 当返回 True 时，买入订单将在交易所下单。
            False 则中止该过程
        """
        return True

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        """
        在下常规出场订单之前调用。
        此函数的时间点至关重要，因此避免在此方法中执行繁重的计算或网络请求。

        完整的文档请访问 https://www.freqtrade.io/en/latest/strategy-advanced/

        如果策略未实现此方法，则返回 True（始终确认）。

        :param pair: 即将出场的交易对。
        :param trade: 交易对象。
        :param order_type: 订单类型（根据 order_types 配置）。通常是限价或市价。
        :param amount: 基础货币的数量。
        :param rate: 限价订单将使用的价格，或市价订单的当前价格。
        :param time_in_force: 时间有效性。默认为 GTC (Good-til-cancelled)。
        :param exit_reason: 出场原因。
            可以是 ['roi', 'stop_loss', 'stoploss_on_exchange', 'trailing_stop_loss',
            'exit_signal', 'force_exit', 'emergency_exit'] 中的任何一个。
        :param current_time: datetime 对象，包含当前日期时间
        :param **kwargs: 确保保留此参数，以便更新不会破坏您的策略。
        :return bool: 当返回 True 时，出场订单将在交易所下单。
            False 则中止该过程
        """
        return True

    def order_filled(
        self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs
    ) -> None:
        """
        订单成交后立即调用。
        将为所有订单类型（入场、出场、止损、仓位调整）调用。
        :param pair: 交易对
        :param trade: 交易对象。
        :param order: 订单对象。
        :param current_time: datetime 对象，包含当前日期时间
        :param **kwargs: 确保保留此参数，以便更新不会破坏您的策略。
        """
        pass

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        """
        自定义止损逻辑，返回相对于 current_rate 的新距离（作为比率）。
        例如，返回 -0.05 将在 current_rate 下方创建 5% 的止损。
        自定义止损永远不能低于 self.stoploss，后者作为硬性最大亏损。

        完整的文档请访问 https://www.freqtrade.io/en/latest/strategy-advanced/

        如果策略未实现此方法，则返回初始止损值。
        仅当 use_custom_stoploss 设置为 True 时调用。

        :param pair: 当前分析的交易对
        :param trade: 交易对象。
        :param current_time: datetime 对象，包含当前日期时间
        :param current_rate: 根据 exit_pricing 中的定价设置计算出的价格。
        :param current_profit: 当前利润（作为比率），根据 current_rate 计算。
        :param after_fill: 如果止损在订单成交后调用，则为 True。
        :param **kwargs: 确保保留此参数，以便更新不会破坏您的策略。
        :return float: 新的止损值，相对于 current_rate
        """
        return self.stoploss

    def custom_roi(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        trade_duration: int,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float | None:
        """
        自定义投资回报率 (ROI) 逻辑，返回一个新的最小 ROI 阈值（作为比率，例如，+5% 为 0.05）。
        仅当 use_custom_roi 设置为 True 时调用。

        如果与 minimal_roi 同时使用，当达到较低阈值时将触发退出。
        示例：如果 minimal_roi = {"0": 0.01} 且 custom_roi 返回 0.05，
        则当利润达到 5% 时将触发退出。

        :param pair: 当前分析的交易对。
        :param trade: 交易对象。
        :param current_time: datetime 对象，包含当前日期时间。
        :param trade_duration: 当前交易持续时间（分钟）。
        :param entry_tag: 可选的入场标签 (buy_tag)，如果随买入信号提供。
        :param side: 'long' (多头) 或 'short' (空头) - 表示当前交易的方向。
        :param **kwargs: 确保保留此参数，以便更新不会破坏您的策略。
        :return float: 新的 ROI 值作为比率，或 None 以回退到 minimal_roi 逻辑。
        """
        return None

    def custom_entry_price(
        self,
        pair: str,
        trade: Trade | None,
        current_time: datetime,
        proposed_rate: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        自定义入场价格逻辑，返回新的入场价格。

        完整的文档请访问 https://www.freqtrade.io/en/latest/strategy-advanced/

        如果策略未实现此方法，则返回 None，将使用订单簿设置入场价格

        :param pair: 当前分析的交易对
        :param trade: 交易对象（初始入场时为 None）。
        :param current_time: datetime 对象，包含当前日期时间
        :param proposed_rate: 根据 exit_pricing 中的定价设置计算出的价格。
        :param entry_tag: 可选的入场标签 (buy_tag)，如果随买入信号提供。
        :param side: 'long' (多头) 或 'short' (空头) - 表示建议交易的方向
        :param **kwargs: 确保保留此参数，以便更新不会破坏您的策略。
        :return float: 如果提供了，则返回新的入场价格值
        """
        return proposed_rate

    def custom_exit_price(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        proposed_rate: float,
        current_profit: float,
        exit_tag: str | None,
        **kwargs,
    ) -> float:
        """
        自定义出场价格逻辑，返回新的出场价格。

        完整的文档请访问 https://www.freqtrade.io/en/latest/strategy-advanced/

        如果策略未实现此方法，则返回 None，将使用订单簿设置出场价格

        :param pair: 当前分析的交易对
        :param trade: 交易对象。
        :param current_time: datetime 对象，包含当前日期时间
        :param proposed_rate: 根据 exit_pricing 中的定价设置计算出的价格。
        :param current_profit: 当前利润（作为比率），根据 current_rate 计算。
        :param exit_tag: 出场原因。
        :param **kwargs: 确保保留此参数，以便更新不会破坏您的策略。
        :return float: 如果提供了，则返回新的出场价格值
        """
        return proposed_rate

    def custom_sell(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool | None:
        """
        已弃用 - 请改用 custom_exit。
        自定义出场信号逻辑，指示应卖出指定仓位。从此方法返回字符串或 True
        等同于在指定时间点的蜡烛图上设置出场信号。当出场信号已设置时，此方法不会被调用。

        应覆盖此方法以创建依赖于交易参数的出场信号。例如，您可以实现
        相对于交易开仓时蜡烛图的出场，或自定义的 1:2 风险回报率。

        自定义出场原因的最大长度为 64。超出部分的字符将被移除。

        :param pair: 当前分析的交易对
        :param trade: 交易对象。
        :param current_time: datetime 对象，包含当前日期时间
        :param current_rate: 根据 exit_pricing 中的定价设置计算出的价格。
        :param current_profit: 当前利润（作为比率），根据 current_rate 计算。
        :param **kwargs: 确保保留此参数，以便更新不会破坏您的策略。
        :return: 要执行出场，返回带有自定义出场原因的字符串或 True。否则返回 None 或 False。
        """
        return None

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool | None:
        """
        自定义出场信号逻辑，指示应卖出指定仓位。从此方法返回字符串或 True
        等同于在指定时间点的蜡烛图上设置出场信号。当出场信号已设置时，此方法不会被调用。

        应覆盖此方法以创建依赖于交易参数的出场信号。例如，您可以实现
        相对于交易开仓时蜡烛图的出场，或自定义的 1:2 风险回报率。

        自定义出场原因的最大长度为 64。超出部分的字符将被移除。

        :param pair: 当前分析的交易对
        :param trade: 交易对象。
        :param current_time: datetime 对象，包含当前日期时间
        :param current_rate: 根据 exit_pricing 中的定价设置计算出的价格。
        :param current_profit: 当前利润（作为比率），根据 current_rate 计算。
        :param **kwargs: 确保保留此参数，以便更新不会破坏您的策略。
        :return: 要执行出场，返回带有自定义出场原因的字符串或 True。否则返回 None 或 False。
        """
        return self.custom_sell(pair, trade, current_time, current_rate, current_profit, **kwargs)

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        为每笔新交易自定义仓位大小。

        :param pair: 当前分析的交易对
        :param current_time: datetime 对象，包含当前日期时间
        :param current_rate: 根据 exit_pricing 中的定价设置计算出的价格。
        :param proposed_stake: 机器人建议的仓位金额。
        :param min_stake: 交易所允许的最小仓位大小。
        :param max_stake: 可用于交易的余额。
        :param leverage: 为此交易选择的杠杆。
        :param entry_tag: 可选的入场标签 (buy_tag)，如果随买入信号提供。
        :param side: 'long' (多头) 或 'short' (空头) - 表示建议交易的方向
        :return: 介于 min_stake 和 max_stake 之间的仓位大小。
        """
        return proposed_stake

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> float | None | tuple[float | None, str | None]:
        """
        自定义交易仓位调整逻辑，返回应增加或减少交易的仓位金额。
        这意味着额外的入场或出场订单将产生额外的费用。
        仅当 `position_adjustment_enable` 设置为 True 时调用。

        完整的文档请访问 https://www.freqtrade.io/en/latest/strategy-advanced/

        如果策略未实现此方法，则返回 None

        :param trade: 交易对象。
        :param current_time: datetime 对象，包含当前日期时间
        :param current_rate: 当前入场价格（与 current_entry_profit 相同）
        :param current_profit: 当前利润（作为比率），根据 current_rate 计算
                               （与 current_entry_profit 相同）。
        :param min_stake: 交易所允许的最小仓位大小（适用于入场和出场）
        :param max_stake: 允许的最大仓位（通过余额或交易所限制）。
        :param current_entry_rate: 使用入场定价的当前价格。
        :param current_exit_rate: 使用出场定价的当前价格。
        :param current_entry_profit: 使用入场定价的当前利润。
        :param current_exit_profit: 使用出场定价的当前利润。
        :param **kwargs: 确保保留此参数，以便更新不会破坏您的策略。
        :return float: 用于调整交易的仓位金额，
                       正值表示增加仓位，负值表示减少仓位。
                       返回 None 表示不采取任何行动。
                       可选地，返回一个包含第二个元素（订单原因）的元组
        """
        return None

    def adjust_entry_price(
        self,
        trade: Trade,
        order: Order | None,
        pair: str,
        current_time: datetime,
        proposed_rate: float,
        current_order_rate: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float | None:
        """
        入场价格重新调整逻辑，返回用户所需的限价。
        仅当订单已下单、仍处于开放状态（完全或部分未成交），
        且在入场触发后的后续蜡烛图中未超时时执行。

        完整的文档请访问 https://www.freqtrade.io/en/latest/strategy-callbacks/

        如果策略未实现此方法，则默认返回 current_order_rate。
        如果返回 current_order_rate，则保留现有订单。
        如果返回 None，则订单被取消，但不会被新订单替换。

        :param pair: 当前分析的交易对
        :param trade: Trade 对象。
        :param order: Order 对象
        :param current_time: datetime 对象，包含当前日期时间
        :param proposed_rate: 根据 entry_pricing 中的定价设置计算出的价格。
        :param current_order_rate: 现有订单的价格。
        :param entry_tag: 可选的入场标签 (buy_tag)，如果随买入信号提供。
        :param side: 'long' (多头) 或 'short' (空头) - 表示建议交易的方向
        :param **kwargs: 确保保留此参数，以便更新不会破坏您的策略。
        :return float 或 None: 如果提供了，则返回新的入场价格值

        """
        return current_order_rate

    def adjust_exit_price(
        self,
        trade: Trade,
        order: Order | None,
        pair: str,
        current_time: datetime,
        proposed_rate: float,
        current_order_rate: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float | None:
        """
        出场价格重新调整逻辑，返回用户所需的限价。
        仅当订单已下单、仍处于开放状态（完全或部分未成交），
        且在入场触发后的后续蜡烛图中未超时时执行。

        完整的文档请访问 https://www.freqtrade.io/en/latest/strategy-callbacks/

        如果策略未实现此方法，则默认返回 current_order_rate。
        如果返回 current_order_rate，则保留现有订单。
        如果返回 None，则订单被取消，但不会被新订单替换。

        :param pair: 当前分析的交易对
        :param trade: Trade 对象。
        :param order: Order 对象
        :param current_time: datetime 对象，包含当前日期时间
        :param proposed_rate: 根据 entry_pricing 中的定价设置计算出的价格。
        :param current_order_rate: 现有订单的价格。
        :param entry_tag: 可选的入场标签 (buy_tag)，如果随买入信号提供。
        :param side: 'long' (多头) 或 'short' (空头) - 表示建议交易的方向
        :param **kwargs: 确保保留此参数，以便更新不会破坏您的策略。
        :return float 或 None: 如果提供了，则返回新的出场价格值

        """
        return current_order_rate

    def adjust_order_price(
        self,
        trade: Trade,
        order: Order | None,
        pair: str,
        current_time: datetime,
        proposed_rate: float,
        current_order_rate: float,
        entry_tag: str | None,
        side: str,
        is_entry: bool,
        **kwargs,
    ) -> float | None:
        """
        出场和入场订单价格重新调整逻辑，返回用户所需的限价。
        仅当订单已下单、仍处于开放状态（完全或部分未成交），
        且在入场触发后的后续蜡烛图中未超时时执行。

        完整的文档请访问 https://www.freqtrade.io/en/latest/strategy-callbacks/

        如果策略未实现此方法，则默认返回 current_order_rate。
        如果返回 current_order_rate，则保留现有订单。
        如果返回 None，则订单被取消，但不会被新订单替换。

        :param pair: 当前分析的交易对
        :param trade: Trade 对象。
        :param order: Order 对象
        :param current_time: datetime 对象，包含当前日期时间
        :param proposed_rate: 根据 entry_pricing 中的定价设置计算出的价格。
        :param current_order_rate: 现有订单的价格。
        :param entry_tag: 可选的入场标签 (buy_tag)，如果随买入信号提供。
        :param side: 'long' (多头) 或 'short' (空头) - 表示建议交易的方向
        :param is_entry: 如果订单是入场订单则为 True，如果出场订单则为 False。
        :param **kwargs: 确保保留此参数，以便更新不会破坏您的策略。
        :return float 或 None: 如果提供了，则返回新的入场价格值
        """
        if is_entry:
            return self.adjust_entry_price(
                trade=trade,
                order=order,
                pair=pair,
                current_time=current_time,
                proposed_rate=proposed_rate,
                current_order_rate=current_order_rate,
                entry_tag=entry_tag,
                side=side,
                **kwargs,
            )
        else:
            return self.adjust_exit_price(
                trade=trade,
                order=order,
                pair=pair,
                current_time=current_time,
                proposed_rate=proposed_rate,
                current_order_rate=current_order_rate,
                entry_tag=entry_tag,
                side=side,
                **kwargs,
            )

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        为每笔新交易自定义杠杆。此方法仅在期货模式下调用。

        :param pair: 当前分析的交易对
        :param current_time: datetime 对象，包含当前日期时间
        :param current_rate: 根据 exit_pricing 中的定价设置计算出的价格。
        :param proposed_leverage: 机器人建议的杠杆。
        :param max_leverage: 该交易对允许的最大杠杆。
        :param entry_tag: 可选的入场标签 (buy_tag)，如果随买入信号提供。
        :param side: 'long' (多头) 或 'short' (空头) - 表示建议交易的方向
        :return: 介于 1.0 和 max_leverage 之间的杠杆金额。
        """
        return 1.0

    def informative_pairs(self) -> ListPairsWithTimeframes:
        """
        定义额外的信息性交易对/时间周期组合，以便从交易所缓存。
        这些交易对/时间周期组合不可交易，除非它们也包含在白名单中。
        有关更多信息，请查阅文档。
        :return: 格式为 (pair, interval) 的元组列表
            示例: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def version(self) -> str | None:
        """
        返回策略版本。
        """
        return None

    def plot_annotations(
        self, pair: str, start_date: datetime, end_date: datetime, dataframe: DataFrame, **kwargs
    ) -> list[AnnotationType]:
        """
        检索图表的区域标注。
        必须以数组形式返回，包含 type, label, color, start, end, y_start, y_end。
        除 type 外，所有设置都是可选的 - 但通常包含 "start and end" 或 "y_start and y_end"
        用于水平或垂直绘图（或全部 4 个用于方框）才有意义。
        :param pair: 当前分析的交易对
        :param start_date: 请求的图表数据的开始日期
        :param end_date: 请求的图表数据的结束日期
        :param dataframe: 包含图表分析数据的 DataFrame
        :param **kwargs: 确保保留此参数，以便更新不会破坏您的策略。
        :return: AnnotationType 对象的列表
        """
        return []

    def populate_any_indicators(
        self,
        pair: str,
        df: DataFrame,
        tf: str,
        informative: DataFrame | None = None,
        set_generalized_indicators: bool = False,
    ) -> DataFrame:
        """
        已弃用 - 请改用特征工程函数
        此函数旨在根据用户在配置文件中指定的时间周期自动生成、命名和合并特征。
        用户可以在此处添加额外特征，但必须遵循命名约定。
        此方法*仅*在 FreqaiDataKitchen 类中使用，因此仅当 FreqAI 处于活动状态时才调用。
        :param pair: 用作信息对的交易对
        :param df: 策略数据帧，将接收来自信息数据帧的合并
        :param tf: 将修改特征名称的数据帧时间周期
        :param informative: 与信息对关联的数据帧
        """
        return df

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        *仅适用于启用 FreqAI 的策略*
        此函数将自动扩展配置中定义的特征：`indicator_periods_candles`、
        `include_timeframes`、`include_shifted_candles` 和 `include_corr_pairs`。
        换句话说，此函数中定义的单个特征将自动扩展为总计
        `indicator_periods_candles` * `include_timeframes` * `include_shifted_candles` *
        `include_corr_pairs` 个添加到模型的特征。

        所有特征必须以 `%` 开头，以便 FreqAI 内部识别。

        有关这些配置参数如何加速特征工程的更多详细信息，请参阅文档：

        https://www.freqtrade.io/en/latest/freqai-parameter-table/#feature-parameters

        https://www.freqtrade.io/en/latest/freqai-feature-engineering/#defining-the-features

        :param dataframe: 将接收特征的策略数据帧
        :param period: 指标的周期 - 用法示例:
        :param metadata: 当前交易对的元数据
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        """
        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        *仅适用于启用 FreqAI 的策略*
        此函数将自动扩展配置中定义的特征：`include_timeframes`、
        `include_shifted_candles` 和 `include_corr_pairs`。
        换句话说，此函数中定义的单个特征将自动扩展为总计
        `include_timeframes` * `include_shifted_candles` * `include_corr_pairs`
        个添加到模型的特征。

        此处定义的特征不会自动在用户定义的 `indicator_periods_candles` 上重复。

        所有特征必须以 `%` 开头，以便 FreqAI 内部识别。

        有关这些配置参数如何加速特征工程的更多详细信息，请参阅文档：

        https://www.freqtrade.io/en/latest/freqai-parameter-table/#feature-parameters

        https://www.freqtrade.io/en/latest/freqai-feature-engineering/#defining-the-features

        :param dataframe: 将接收特征的策略数据帧
        :param metadata: 当前交易对的元数据
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-ema-200"] = ta.EMA(dataframe, timeperiod=200)
        """
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        *仅适用于启用 FreqAI 的策略*
        此可选函数将使用基础时间周期的数据帧调用一次。
        这是最后一个被调用的函数，这意味着进入此函数的数据帧将包含所有由其他
        freqai_feature_engineering_* 函数创建的特征和列。

        此函数是进行自定义奇异特征提取（例如 tsfresh）的好地方。
        此函数也是任何不应自动扩展的特征（例如，星期几）的好地方。

        所有特征必须以 `%` 开头，以便 FreqAI 内部识别。

        有关特征工程的更多详细信息，请参阅：

        https://www.freqtrade.io/en/latest/freqai-feature-engineering

        :param dataframe: 将接收特征的策略数据帧
        :param metadata: 当前交易对的元数据
        用法示例: dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        """
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        *仅适用于启用 FreqAI 的策略*
        设置模型目标所需的函数。
        所有目标必须以 `&` 开头，以便 FreqAI 内部识别。

        有关特征工程的更多详细信息，请参阅：

        https://www.freqtrade.io/en/latest/freqai-feature-engineering

        :param dataframe: 将接收目标的策略数据帧
        :param metadata: 当前交易对的元数据
        用法示例: dataframe["&-target"] = dataframe["close"].shift(-1) / dataframe["close"]
        """
        return dataframe

    ###
    # 结束 - 意图由策略覆盖
    ###

    _ft_stop_uses_after_fill = False

    def _adjust_trade_position_internal(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> tuple[float | None, str]:
        """
        围绕 adjust_trade_position 的包装器，用于处理返回值
        """
        resp = strategy_safe_wrapper(
            self.adjust_trade_position, default_retval=(None, ""), supress_error=True
        )(
            trade=trade,
            current_time=current_time,
            current_rate=current_rate,
            current_profit=current_profit,
            min_stake=min_stake,
            max_stake=max_stake,
            current_entry_rate=current_entry_rate,
            current_exit_rate=current_exit_rate,
            current_entry_profit=current_entry_profit,
            current_exit_profit=current_exit_profit,
            **kwargs,
        )
        order_tag = ""
        if isinstance(resp, tuple):
            if len(resp) >= 1:
                stake_amount = resp[0]
            if len(resp) > 1:
                order_tag = resp[1] or ""
        else:
            stake_amount = resp
        return stake_amount, order_tag

    def __informative_pairs_freqai(self) -> ListPairsWithTimeframes:
        """
        创建 FreqAI 所需的信息对
        """
        if self.config.get("freqai", {}).get("enabled", False):
            whitelist_pairs = self.dp.current_whitelist()
            candle_type = self.config.get("candle_type_def", CandleType.SPOT)
            corr_pairs = self.config["freqai"]["feature_parameters"]["include_corr_pairlist"]
            informative_pairs = []
            for tf in self.config["freqai"]["feature_parameters"]["include_timeframes"]:
                for pair in set(whitelist_pairs + corr_pairs):
                    informative_pairs.append((pair, tf, candle_type))
            return informative_pairs

        return []

    def gather_informative_pairs(self) -> ListPairsWithTimeframes:
        """
        内部方法，用于收集所有信息对（用户或自动定义）。
        """
        informative_pairs = self.informative_pairs()
        # 兼容 2 元组信息对的代码
        informative_pairs = [
            (
                p[0],
                p[1],
                (
                    CandleType.from_string(p[2])
                    if len(p) > 2 and p[2] != ""
                    else self.config.get("candle_type_def", CandleType.SPOT)
                ),
            )
            for p in informative_pairs
        ]
        for inf_data, _ in self._ft_informative:
            # 如果未明确提供，获取默认蜡烛图类型。
            candle_type = (
                inf_data.candle_type
                if inf_data.candle_type
                else self.config.get("candle_type_def", CandleType.SPOT)
            )
            if inf_data.asset:
                if any(s in inf_data.asset for s in ("{BASE}", "{base}")):
                    for pair in self.dp.current_whitelist():
                        pair_tf = (
                            _format_pair_name(self.config, inf_data.asset, self.dp.market(pair)),
                            inf_data.timeframe,
                            candle_type,
                        )
                        informative_pairs.append(pair_tf)

                else:
                    pair_tf = (
                        _format_pair_name(self.config, inf_data.asset),
                        inf_data.timeframe,
                        candle_type,
                    )
                    informative_pairs.append(pair_tf)
            else:
                for pair in self.dp.current_whitelist():
                    informative_pairs.append((pair, inf_data.timeframe, candle_type))
        informative_pairs.extend(self.__informative_pairs_freqai())
        return list(set(informative_pairs))

    def get_strategy_name(self) -> str:
        """
        返回策略类名
        """
        return self.__class__.__name__

    def lock_pair(
        self, pair: str, until: datetime, reason: str | None = None, side: str = "*"
    ) -> None:
        """
        锁定交易对，直到给定时间戳。
        锁定的交易对不进行分析，并阻止开新仓。
        锁定时间只能增加（允许用户锁定交易对更长时间）。
        要从交易对中移除锁定，请使用 `unlock_pair()`
        :param pair: 要锁定的交易对
        :param until: UTC 时间，直到该时间交易对将被阻止开新仓。
                需要是时区感知的 `datetime.now(timezone.utc)`
        :param reason: 可选字符串，解释为什么交易对被锁定。
        :param side: 要检查的方向，可以是 long, short 或 '*' (所有方向)
        """
        PairLocks.lock_pair(pair, until, reason, side=side)

    def unlock_pair(self, pair: str) -> None:
        """
        解锁先前使用 lock_pair 锁定的交易对。
        不被 freqtrade 本身使用，但旨在供用户在策略内部手动锁定交易对时使用，
        以提供一种轻松解锁交易对的方法。
        :param pair: 解锁交易对以允许再次交易
        """
        PairLocks.unlock_pair(pair, datetime.now(timezone.utc))

    def unlock_reason(self, reason: str) -> None:
        """
        解锁所有先前使用 lock_pair 锁定并指定了原因的交易对。
        不被 freqtrade 本身使用，但旨在供用户在策略内部手动锁定交易对时使用，
        以提供一种轻松解锁交易对的方法。
        :param reason: 解锁交易对以允许再次交易
        """
        PairLocks.unlock_reason(reason, datetime.now(timezone.utc))

    def is_pair_locked(
        self, pair: str, *, candle_date: datetime | None = None, side: str = "*"
    ) -> bool:
        """
        检查交易对当前是否被锁定
        第二个可选参数确保锁定应用到新蜡烛图到来之时，
        而不是在 14:00:00 停止锁定 - 而下一根蜡烛图在 14:00:02 到来，
        留下了 2 秒的空白时间，可能导致旧信号上的入场订单。
        :param pair: “要检查的交易对”
        :param candle_date: 最后一根蜡烛图的日期。可选，默认为当前日期
        :param side: 要检查的方向，可以是 long, short 或 '*'
        :returns: 相关交易对的锁定状态。
        """

        if not candle_date:
            # 简单调用...
            return PairLocks.is_pair_locked(pair, side=side)
        else:
            lock_time = timeframe_to_next_date(self.timeframe, candle_date)
            return PairLocks.is_pair_locked(pair, lock_time, side=side)

    def analyze_ticker(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        解析给定的蜡烛图 (OHLCV) 数据并返回一个填充好的 DataFrame
        向其中添加多个技术分析指标和入场订单信号
        应仅在实盘中使用。
        :param dataframe: 包含来自交易所数据的 DataFrame
        :param metadata: 包含附加数据（例如 'pair'）的元数据字典
        :return: 包含指标数据和添加了信号的蜡烛图 (OHLCV) 数据 DataFrame
        """
        logger.debug("技术分析启动")
        dataframe = self.advise_indicators(dataframe, metadata)
        dataframe = self.advise_entry(dataframe, metadata)
        dataframe = self.advise_exit(dataframe, metadata)
        logger.debug("技术分析结束")
        return dataframe

    def _analyze_ticker_internal(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        解析给定的蜡烛图 (OHLCV) 数据并返回一个填充好的 DataFrame
        向其中添加多个技术分析指标和买入信号
        警告: 仅内部使用，如果设置了 `process_only_new_candles`，可能会跳过分析。
        :param dataframe: 包含来自交易所数据的 DataFrame
        :param metadata: 包含附加数据（例如 'pair'）的元数据字典
        :return: 包含指标数据和添加了信号的蜡烛图 (OHLCV) 数据 DataFrame
        """
        pair = str(metadata.get("pair"))

        new_candle = self._last_candle_seen_per_pair.get(pair, None) != dataframe.iloc[-1]["date"]
        # 测试是否见过该交易对和之前的最后一根蜡烛图。
        # 如果 process_only_new_candles 设置为 false，则始终运行
        if not self.process_only_new_candles or new_candle:
            # 仅在新的蜡烛图数据上进行更改的定义。
            dataframe = self.analyze_ticker(dataframe, metadata)

            self._last_candle_seen_per_pair[pair] = dataframe.iloc[-1]["date"]

            candle_type = self.config.get("candle_type_def", CandleType.SPOT)
            self.dp._set_cached_df(pair, self.timeframe, dataframe, candle_type=candle_type)
            self.dp._emit_df((pair, self.timeframe, candle_type), dataframe, new_candle)

        else:
            logger.debug("跳过对已分析蜡烛图的技术分析")
            dataframe = remove_entry_exit_signals(dataframe)

        logger.debug("循环分析启动")

        return dataframe

    def analyze_pair(self, pair: str) -> None:
        """
        从 dataprovider 获取此交易对的数据并进行分析。
        将数据帧存储到 dataprovider 中。
        然后可以通过 `dp.get_analyzed_dataframe()` 访问分析后的数据帧。
        :param pair: 要分析的交易对。
        """
        dataframe = self.dp.ohlcv(
            pair, self.timeframe, candle_type=self.config.get("candle_type_def", CandleType.SPOT)
        )
        if not isinstance(dataframe, DataFrame) or dataframe.empty:
            logger.warning("交易对 %s 的蜡烛图 (OHLCV) 数据为空", pair)
            return

        try:
            validator = StrategyResultValidator(
                dataframe, warn_only=not self.disable_dataframe_checks
            )

            dataframe = strategy_safe_wrapper(self._analyze_ticker_internal, message="")(
                dataframe, {"pair": pair}
            )

            validator.assert_df(dataframe)
        except StrategyError as error:
            logger.warning(f"无法分析交易对 {pair} 的蜡烛图 (OHLCV) 数据: {error}")
            return

        if dataframe.empty:
            logger.warning("交易对 %s 的数据帧为空", pair)
            return

    def analyze(self, pairs: list[str]) -> None:
        """
        使用 analyze_pair() 分析所有交易对。
        :param pairs: 要分析的交易对列表
        """
        for pair in pairs:
            self.analyze_pair(pair)

    def get_latest_candle(
        self,
        pair: str,
        timeframe: str,
        dataframe: DataFrame,
    ) -> tuple[DataFrame | None, datetime | None]:
        """
        根据数据帧的入场订单或出场订单列计算当前信号。
        由机器人用于获取入场或出场信号。
        :param pair: 格式为 ANT/BTC 的交易对
        :param timeframe: 使用的时间周期
        :param dataframe: 用于获取信号的分析数据帧。
        :return: (None, None) 或 (Dataframe, latest_date) - 对应于最后一根蜡烛图
        """
        if not isinstance(dataframe, DataFrame) or dataframe.empty:
            logger.warning(f"交易对 {pair} 的蜡烛图 (OHLCV) 数据为空")
            return None, None

        try:
            latest_date_pd = dataframe["date"].max()
            latest = dataframe.loc[dataframe["date"] == latest_date_pd].iloc[-1]
        except Exception as e:
            logger.warning(f"无法获取交易对 {pair} 的最新蜡烛图 (OHLCV) 数据 - {e}")
            return None, None
        # 显式转换为 datetime 对象，以确保下面的比较不会失败
        latest_date: datetime = latest_date_pd.to_pydatetime()

        # 检查数据帧是否过时
        timeframe_minutes = timeframe_to_minutes(timeframe)
        offset = self.config.get("exchange", {}).get("outdated_offset", 5)
        if latest_date < (dt_now() - timedelta(minutes=timeframe_minutes * 2 + offset)):
            logger.warning(
                "交易对 %s 的历史数据已过时。最新行情已过时 %s 分钟",
                pair,
                int((dt_now() - latest_date).total_seconds() // 60),
            )
            return None, None
        return latest, latest_date

    def get_exit_signal(
        self, pair: str, timeframe: str, dataframe: DataFrame, is_short: bool | None = None
    ) -> tuple[bool, bool, str | None]:
        """
        根据数据帧的列计算当前出场信号。
        由机器人用于获取出场信号。
        根据 is_short，查看 "short" 或 "long" 列。
        :param pair: 格式为 ANT/BTC 的交易对
        :param timeframe: 使用的时间周期
        :param dataframe: 用于获取信号的分析数据帧。
        :param is_short: 指示现有交易方向。
        :return: (enter, exit) 包含入场/出场布尔值的元组。
        """
        latest, _latest_date = self.get_latest_candle(pair, timeframe, dataframe)
        if latest is None:
            return False, False, None

        if is_short:
            enter = latest.get(SignalType.ENTER_SHORT.value, 0) == 1
            exit_ = latest.get(SignalType.EXIT_SHORT.value, 0) == 1

        else:
            enter = latest.get(SignalType.ENTER_LONG.value, 0) == 1
            exit_ = latest.get(SignalType.EXIT_LONG.value, 0) == 1
        exit_tag = latest.get(SignalTagType.EXIT_TAG.value, None)
        # 标签可以是 None，这不会解析为 False。
        exit_tag = exit_tag if isinstance(exit_tag, str) and exit_tag != "nan" else None

        logger.debug(f"出场触发: {latest['date']} (交易对={pair}) 入场={enter} 出场={exit_}")

        return enter, exit_, exit_tag

    def get_entry_signal(
        self,
        pair: str,
        timeframe: str,
        dataframe: DataFrame,
    ) -> tuple[SignalDirection | None, str | None]:
        """
        根据数据帧信号的列计算当前入场信号。
        由机器人用于获取入场交易信号。
        :param pair: 格式为 ANT/BTC 的交易对
        :param timeframe: 使用的时间周期
        :param dataframe: 用于获取信号的分析数据帧。
        :return: (SignalDirection, entry_tag)
        """
        latest, latest_date = self.get_latest_candle(pair, timeframe, dataframe)
        if latest is None or latest_date is None:
            return None, None

        enter_long = latest.get(SignalType.ENTER_LONG.value, 0) == 1
        exit_long = latest.get(SignalType.EXIT_LONG.value, 0) == 1
        enter_short = latest.get(SignalType.ENTER_SHORT.value, 0) == 1
        exit_short = latest.get(SignalType.EXIT_SHORT.value, 0) == 1

        enter_signal: SignalDirection | None = None
        enter_tag: str | None = None
        if enter_long == 1 and not any([exit_long, enter_short]):
            enter_signal = SignalDirection.LONG
            enter_tag = latest.get(SignalTagType.ENTER_TAG.value, None)
        if (
            self.config.get("trading_mode", TradingMode.SPOT) != TradingMode.SPOT
            and self.can_short
            and enter_short == 1
            and not any([exit_short, enter_long])
        ):
            enter_signal = SignalDirection.SHORT
            enter_tag = latest.get(SignalTagType.ENTER_TAG.value, None)

        enter_tag = enter_tag if isinstance(enter_tag, str) and enter_tag != "nan" else None

        timeframe_seconds = timeframe_to_seconds(timeframe)

        if self.ignore_expired_candle(
            latest_date=latest_date,
            current_time=dt_now(),
            timeframe_seconds=timeframe_seconds,
            enter=bool(enter_signal),
        ):
            return None, enter_tag

        logger.debug(
            f"入场触发: {latest['date']} (交易对={pair}) "
            f"入场={enter_long} 入场标签值={enter_tag}"
        )
        return enter_signal, enter_tag

    def ignore_expired_candle(
        self, latest_date: datetime, current_time: datetime, timeframe_seconds: int, enter: bool
    ):
        if self.ignore_buying_expired_candle_after and enter:
            time_delta = current_time - (latest_date + timedelta(seconds=timeframe_seconds))
            return time_delta.total_seconds() > self.ignore_buying_expired_candle_after
        else:
            return False

    def should_exit(
        self,
        trade: Trade,
        rate: float,
        current_time: datetime,
        *,
        enter: bool,
        exit_: bool,
        low: float | None = None,
        high: float | None = None,
        force_stoploss: float = 0,
    ) -> list[ExitCheckTuple]:
        """
        此函数评估是否已达到触发出场订单的条件之一，可以是止损、投资回报率或出场信号。
        :param low: 仅在回测期间用于模拟（多头）止损/（空头）投资回报率
        :param high: 仅在回测期间用于模拟（空头）止损/（多头）投资回报率
        :param force_stoploss: 外部提供的止损
        :return: 出场原因列表 - 或空列表。
        """
        exits: list[ExitCheckTuple] = []
        current_rate = rate
        current_profit = trade.calc_profit_ratio(current_rate)
        current_profit_best = current_profit
        if low is not None or high is not None:
            # 在回测中，将当前价格设置为高点以进行投资回报率退出
            current_rate_best = (low if trade.is_short else high) or rate
            current_profit_best = trade.calc_profit_ratio(current_rate_best)

        trade.adjust_min_max_rates(high or current_rate, low or current_rate)

        stoplossflag = self.ft_stoploss_reached(
            current_rate=current_rate,
            trade=trade,
            current_time=current_time,
            current_profit=current_profit,
            force_stoploss=force_stoploss,
            low=low,
            high=high,
        )

        # 如果有入场信号并且 ignore_roi 被设置，则无需评估 min_roi。
        roi_reached = not (enter and self.ignore_roi_if_entry_signal) and self.min_roi_reached(
            trade=trade, current_profit=current_profit_best, current_time=current_time
        )

        exit_signal = ExitType.NONE
        custom_reason = ""

        if self.use_exit_signal:
            if exit_ and not enter:
                exit_signal = ExitType.EXIT_SIGNAL
            else:
                reason_cust = strategy_safe_wrapper(self.custom_exit, default_retval=False)(
                    pair=trade.pair,
                    trade=trade,
                    current_time=current_time,
                    current_rate=current_rate,
                    current_profit=current_profit,
                )
                if reason_cust:
                    exit_signal = ExitType.CUSTOM_EXIT
                    if isinstance(reason_cust, str):
                        custom_reason = reason_cust
                        if len(reason_cust) > CUSTOM_TAG_MAX_LENGTH:
                            logger.warning(
                                f"从 custom_exit 返回的自定义出场原因过长，已被截断到 "
                                f"{CUSTOM_TAG_MAX_LENGTH} 个字符。"
                            )
                            custom_reason = reason_cust[:CUSTOM_TAG_MAX_LENGTH]
                    else:
                        custom_reason = ""
            if exit_signal == ExitType.CUSTOM_EXIT or (
                exit_signal == ExitType.EXIT_SIGNAL
                and (not self.exit_profit_only or current_profit > self.exit_profit_offset)
            ):
                logger.debug(
                    f"{trade.pair} - 收到卖出信号。 "
                    f"出场类型=ExitType.{exit_signal.name}"
                    + (f", 自定义原因={custom_reason}" if custom_reason else "")
                )
                exits.append(ExitCheckTuple(exit_type=exit_signal, exit_reason=custom_reason))

        # 序列:
        # 出场信号
        # 止损
        # 投资回报率
        # 追踪止损

        if stoplossflag.exit_type in (ExitType.STOP_LOSS, ExitType.LIQUIDATION):
            logger.debug(f"{trade.pair} - 止损被触发。出场类型={stoplossflag.exit_type}")
            exits.append(stoplossflag)

        if roi_reached:
            logger.debug(f"{trade.pair} - 达到所需利润。出场类型=ExitType.ROI")
            exits.append(ExitCheckTuple(exit_type=ExitType.ROI))

        if stoplossflag.exit_type == ExitType.TRAILING_STOP_LOSS:
            logger.debug(f"{trade.pair} - 追踪止损被触发。")
            exits.append(stoplossflag)

        return exits

    def ft_stoploss_adjust(
        self,
        current_rate: float,
        trade: Trade,
        current_time: datetime,
        current_profit: float,
        force_stoploss: float,
        low: float | None = None,
        high: float | None = None,
        after_fill: bool = False,
    ) -> None:
        """
        如果配置允许，动态调整止损。
        :param current_profit: 当前利润比率
        :param low: 蜡烛图的低点值，仅在回测中设置
        :param high: 蜡烛图的高点值，仅在回测中设置
        """
        if after_fill and not self._ft_stop_uses_after_fill:
            # 如果策略不支持成交后调整，则跳过。
            return

        stop_loss_value = force_stoploss if force_stoploss else self.stoploss

        # 使用 open_rate 初始化止损。如果止损已设置，则不执行任何操作。
        trade.adjust_stop_loss(trade.open_rate, stop_loss_value, initial=True)

        dir_correct = (
            trade.stop_loss < (low or current_rate)
            if not trade.is_short
            else trade.stop_loss > (high or current_rate)
        )

        # 确保在回测中使用高点计算 current_profit。
        bound = low if trade.is_short else high
        bound_profit = current_profit if not bound else trade.calc_profit_ratio(bound)
        if self.use_custom_stoploss and dir_correct:
            stop_loss_value_custom = strategy_safe_wrapper(
                self.custom_stoploss, default_retval=None, supress_error=True
            )(
                pair=trade.pair,
                trade=trade,
                current_time=current_time,
                current_rate=(bound or current_rate),
                current_profit=bound_profit,
                after_fill=after_fill,
            )
            # 健全性检查 - 错误情况下将返回 None
            if stop_loss_value_custom and not (
                isnan(stop_loss_value_custom) or isinf(stop_loss_value_custom)
            ):
                stop_loss_value = stop_loss_value_custom
                trade.adjust_stop_loss(
                    bound or current_rate, stop_loss_value, allow_refresh=after_fill
                )
            else:
                logger.debug("CustomStoploss 函数未返回有效止损")

        if self.trailing_stop and dir_correct:
            # 追踪止损处理
            sl_offset = self.trailing_stop_positive_offset
            # 确保在回测中使用高点计算 current_profit。

            # 如果 trailing_only_offset_is_reached 为 true，则不更新止损。
            if not (self.trailing_only_offset_is_reached and bound_profit < sl_offset):
                # 对 trailing_stop_positive 的特殊处理
                if self.trailing_stop_positive is not None and bound_profit > sl_offset:
                    stop_loss_value = self.trailing_stop_positive
                    logger.debug(
                        f"{trade.pair} - 使用正向止损: {stop_loss_value} "
                        f"偏移: {sl_offset:.4g} 利润: {bound_profit:.2%}"
                    )

                trade.adjust_stop_loss(bound or current_rate, stop_loss_value)

    def ft_stoploss_reached(
        self,
        current_rate: float,
        trade: Trade,
        current_time: datetime,
        current_profit: float,
        force_stoploss: float,
        low: float | None = None,
        high: float | None = None,
    ) -> ExitCheckTuple:
        """
        根据交易的当前利润和配置的（追踪）止损，决定是否出场。
        :param current_profit: 当前利润比率
        :param low: 蜡烛图的低点值，仅在回测中设置
        :param high: 蜡烛图的高点值，仅在回测中设置
        """
        self.ft_stoploss_adjust(
            current_rate, trade, current_time, current_profit, force_stoploss, low, high
        )

        sl_higher_long = trade.stop_loss >= (low or current_rate) and not trade.is_short
        sl_lower_short = trade.stop_loss <= (high or current_rate) and trade.is_short
        liq_higher_long = (
            trade.liquidation_price
            and trade.liquidation_price >= (low or current_rate)
            and not trade.is_short
        )
        liq_lower_short = (
            trade.liquidation_price
            and trade.liquidation_price <= (high or current_rate)
            and trade.is_short
        )

        # 如果止损不在交易所，则评估止损是否被触发
        # 在 Dry-Run 模式下，这也处理止损逻辑，因为逻辑与常规止损处理没有区别。
        if (sl_higher_long or sl_lower_short) and (
            not self.order_types.get("stoploss_on_exchange") or self.config["dry_run"]
        ):
            exit_type = ExitType.STOP_LOSS

            # 如果初始止损与当前止损不同，则表示正在追踪止损。
            if trade.is_stop_loss_trailing:
                exit_type = ExitType.TRAILING_STOP_LOSS
                logger.debug(
                    f"{trade.pair} - 触发止损: 当前价格在 "
                    f"{((high if trade.is_short else low) or current_rate):.6f}, "
                    f"止损为 {trade.stop_loss:.6f}, "
                    f"初始止损为 {trade.initial_stop_loss:.6f}, "
                    f"交易开仓价为 {trade.open_rate:.6f}"
                )

            return ExitCheckTuple(exit_type=exit_type)

        if liq_higher_long or liq_lower_short:
            logger.debug(f"{trade.pair} - 触发爆仓价格。出场类型=ExitType.LIQUIDATION")
            return ExitCheckTuple(exit_type=ExitType.LIQUIDATION)

        return ExitCheckTuple(exit_type=ExitType.NONE)

    def min_roi_reached_entry(
        self,
        trade: Trade,
        trade_dur: int,
        current_time: datetime,
    ) -> tuple[int | None, float | None]:
        """
        根据交易持续时间定义可能已达到的投资回报率 (ROI) 条目。
        :param trade_dur: 交易持续时间（分钟）
        :return: 最小投资回报率条目值，如果没有找到合适的投资回报率条目则返回 None。
        """

        # 如果 use_custom_roi 设置为 True，获取自定义投资回报率
        custom_roi = None
        if self.use_custom_roi:
            custom_roi = strategy_safe_wrapper(
                self.custom_roi, default_retval=None, supress_error=True
            )(
                pair=trade.pair,
                trade=trade,
                current_time=current_time,
                trade_duration=trade_dur,
                entry_tag=trade.enter_tag,
                side=trade.trade_direction,
            )
            if custom_roi is None or isnan(custom_roi) or isinf(custom_roi):
                custom_roi = None
                logger.debug(f"自定义投资回报率函数未为 {trade.pair} 返回有效投资回报率")

        # 获取 ROI 字典中键值 <= 交易持续时间的最大条目
        roi_list = [x for x in self.minimal_roi.keys() if x <= trade_dur]
        if roi_list:
            roi_entry = max(roi_list)
            min_roi = self.minimal_roi[roi_entry]
        else:
            roi_entry = None
            min_roi = None

        # 使用最低可用值来触发退出。
        if custom_roi is not None and (min_roi is None or custom_roi < min_roi):
            return trade_dur, custom_roi
        else:
            return roi_entry, min_roi

    def min_roi_reached(self, trade: Trade, current_profit: float, current_time: datetime) -> bool:
        """
        根据交易持续时间、交易当前利润和投资回报率配置，
        决定机器人是否应该退出。
        :param current_profit: 当前利润比率
        :return: 如果机器人应该以当前价格退出，则返回 True
        """
        # 检查时间是否匹配且当前价格是否高于阈值
        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        _, roi = self.min_roi_reached_entry(trade, trade_dur, current_time)
        if roi is None:
            return False
        else:
            return current_profit > roi

    def ft_check_timed_out(self, trade: Trade, order: Order, current_time: datetime) -> bool:
        """
        FT 内部方法。
        检查超时是否激活，以及订单是否仍处于开放状态且已超时。
        """
        side = "entry" if order.ft_order_side == trade.entry_side else "exit"

        timeout = self.config.get("unfilledtimeout", {}).get(side)
        if timeout is not None:
            timeout_unit = self.config.get("unfilledtimeout", {}).get("unit", "minutes")
            timeout_kwargs = {timeout_unit: -timeout}
            timeout_threshold = current_time + timedelta(**timeout_kwargs)
            timedout = order.status == "open" and order.order_date_utc < timeout_threshold
            if timedout:
                return True
        time_method = (
            self.check_exit_timeout
            if order.ft_order_side == trade.exit_side
            else self.check_entry_timeout
        )

        return strategy_safe_wrapper(time_method, default_retval=False)(
            pair=trade.pair, trade=trade, order=order, current_time=current_time
        )

    def advise_all_indicators(self, data: dict[str, DataFrame]) -> dict[str, DataFrame]:
        """
        为给定的蜡烛图 (OHLCV) 数据（适用于多个交易对）填充指标
        不运行 advise_entry 或 advise_exit！
        仅在优化操作中使用，不在实盘/模拟运行中使用。
        使用 .copy() 获取数据帧的全新副本，用于每次策略运行。
        输出时也进行复制，以避免 pandas 1.3.0 开始显示的 PerformanceWarnings。
        无论出于何种原因，这都对内存使用产生积极影响 - 即使只使用一个策略也是如此。
        """
        res = {}
        for pair, pair_data in data.items():
            validator = StrategyResultValidator(
                pair_data, warn_only=not self.disable_dataframe_checks
            )
            res[pair] = self.advise_indicators(pair_data.copy(), {"pair": pair}).copy()
            validator.assert_df(res[pair])
        return res

    def ft_advise_signals(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        调用 advise_entry 和 advise_exit 并返回结果数据帧。
        :param dataframe: 包含交易所数据以及预计算指标的数据帧
        :param metadata: 包含附加信息（例如当前交易对）的元数据字典
        :return: 包含指标数据和添加了信号的蜡烛图 (OHLCV) 数据 DataFrame

        """

        dataframe = self.advise_entry(dataframe, metadata)
        dataframe = self.advise_exit(dataframe, metadata)
        return dataframe

    def _if_enabled_populate_trades(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        use_public_trades = self.config.get("exchange", {}).get("use_public_trades", False)
        if use_public_trades:
            pair = metadata["pair"]
            trades = self.dp.trades(pair=pair, copy=False)

            # TODO: 将交易切片到数据帧的大小，以便更快地回测
            cached_grouped_trades: DataFrame | None = self._cached_grouped_trades_per_pair.get(pair)
            dataframe, cached_grouped_trades = populate_dataframe_with_trades(
                cached_grouped_trades, self.config, dataframe, trades
            )

            # 解除旧缓存引用
            if pair in self._cached_grouped_trades_per_pair:
                del self._cached_grouped_trades_per_pair[pair]
            self._cached_grouped_trades_per_pair[pair] = cached_grouped_trades

            logger.debug("已使用交易数据填充数据帧。")
        return dataframe

    def advise_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        填充将在买入、卖出、做空、平仓策略中使用的指标
        此方法不应被覆盖。
        :param dataframe: 包含来自交易所数据的数据帧
        :param metadata: 附加信息，例如当前交易对
        :return: 包含策略所有强制性指标的数据帧
        """
        logger.debug(f"正在填充交易对 {metadata.get('pair')} 的指标。")

        # 调用带有 @informative 装饰器的 populate_indicators_Nm() 方法。
        for inf_data, populate_fn in self._ft_informative:
            dataframe = _create_and_merge_informative_pair(
                self, dataframe, metadata, inf_data, populate_fn
            )

        dataframe = self._if_enabled_populate_trades(dataframe, metadata)
        dataframe = self.populate_indicators(dataframe, metadata)
        if self.config.get("reduce_df_footprint", False) and self.config.get("runmode") not in [
            RunMode.DRY_RUN,
            RunMode.LIVE,
        ]:
            dataframe = reduce_dataframe_footprint(dataframe)
        return dataframe

    def advise_entry(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        根据技术分析指标，填充给定数据帧的入场订单信号
        此方法不应被覆盖。
        :param dataframe: DataFrame
        :param metadata: 附加信息字典，包含当前交易对等详细信息
        :return: 带有买入列的 DataFrame
        """

        logger.debug(f"正在填充交易对 {metadata.get('pair')} 的入场信号。")
        # 初始化列以解决 Pandas bug #56503。
        dataframe.loc[:, "enter_tag"] = ""
        df = self.populate_entry_trend(dataframe, metadata)
        if "enter_long" not in df.columns:
            df = df.rename({"buy": "enter_long", "buy_tag": "enter_tag"}, axis="columns")

        return df

    def advise_exit(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        根据技术分析指标，填充给定数据帧的出场订单信号
        此方法不应被覆盖。
        :param dataframe: DataFrame
        :param metadata: 附加信息字典，包含当前交易对等详细信息
        :return: 带有出场列的 DataFrame
        """
        # 初始化列以解决 Pandas bug #56503。
        dataframe.loc[:, "exit_tag"] = ""
        logger.debug(f"正在填充交易对 {metadata.get('pair')} 的出场信号。")
        df = self.populate_exit_trend(dataframe, metadata)
        if "exit_long" not in df.columns:
            df = df.rename({"sell": "exit_long"}, axis="columns")
        return df

    def ft_plot_annotations(self, pair: str, dataframe: DataFrame) -> list[AnnotationType]:
        """
        plot_dataframe 的内部包装器
        """
        if len(dataframe) > 0:
            annotations = strategy_safe_wrapper(self.plot_annotations)(
                pair=pair,
                dataframe=dataframe,
                start_date=dataframe.iloc[0]["date"].to_pydatetime(),
                end_date=dataframe.iloc[-1]["date"].to_pydatetime(),
            )

            from freqtrade.ft_types.plot_annotation_type import AnnotationTypeTA

            annotations_new: list[AnnotationType] = []
            for annotation in annotations:
                if isinstance(annotation, dict):
                    # 转换为 AnnotationType
                    try:
                        AnnotationTypeTA.validate_python(annotation)
                        annotations_new.append(annotation)
                    except ValidationError as e:
                        logger.error(f"无效的标注数据: {annotation}。错误: {e}")
                else:
                    # 已经是 AnnotationType
                    annotations_new.append(annotation)

            return annotations_new
        return []
