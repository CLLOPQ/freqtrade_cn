"""
抽象数据处理器接口。
其子类负责从磁盘处理和存储数据。
"""

import logging
import re
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from pandas import DataFrame, to_datetime

from freqtrade import misc
from freqtrade.configuration import TimeRange
from freqtrade.constants import DEFAULT_TRADES_COLUMNS, ListPairsWithTimeframes
from freqtrade.data.converter import (
    clean_ohlcv_dataframe,
    trades_convert_types,
    trades_df_remove_duplicates,
    trim_dataframe,
)
from freqtrade.enums import CandleType, TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_seconds


logger = logging.getLogger(__name__)


class IDataHandler(ABC):
    _OHLCV_REGEX = r"^([a-zA-Z_\d-]+)\-(\d+[a-zA-Z]{1,2})\-?([a-zA-Z_]*)?(?=\.)"
    _TRADES_REGEX = r"^([a-zA-Z_\d-]+)\-(trades)?(?=\.)"

    def __init__(self, datadir: Path) -> None:
        self._datadir = datadir

    @classmethod
    def _get_file_extension(cls) -> str:
        """
        获取此特定数据处理器的文件扩展名
        """
        raise NotImplementedError()

    @classmethod
    def ohlcv_get_available_data(
        cls, datadir: Path, trading_mode: TradingMode
    ) -> ListPairsWithTimeframes:
        """
        返回此数据目录中所有有 OHLCV 数据的交易对列表
        :param datadir: 用于搜索 OHLCV 文件的目录
        :param trading_mode: 要使用的交易模式
        :return: 元组列表 (交易对, 时间框架, 蜡烛类型)
        """
        if trading_mode == TradingMode.FUTURES:
            datadir = datadir.joinpath("futures")
        _tmp = [
            re.search(cls._OHLCV_REGEX, p.name)
            for p in datadir.glob(f"*.{cls._get_file_extension()}")
        ]
        return [
            (
                cls.rebuild_pair_from_filename(match[1]),
                cls.rebuild_timeframe_from_filename(match[2]),
                CandleType.from_string(match[3]),
            )
            for match in _tmp
            if match and len(match.groups()) > 1
        ]

    @classmethod
    def ohlcv_get_pairs(cls, datadir: Path, timeframe: str, candle_type: CandleType) -> list[str]:
        """
        返回此数据目录中指定时间框架下所有有 OHLCV 数据的交易对列表
        :param datadir: 用于搜索 OHLCV 文件的目录
        :param timeframe: 要搜索交易对的时间框架
        :param candle_type: CandleType 枚举值之一（必须与交易模式匹配！）
        :return: 交易对列表
        """
        candle = ""
        if candle_type != CandleType.SPOT:
            datadir = datadir.joinpath("futures")
            candle = f"-{candle_type}"
        ext = cls._get_file_extension()
        _tmp = [
            re.search(r"^(\S+)(?=\-" + timeframe + candle + f".{ext})", p.name)
            for p in datadir.glob(f"*{timeframe}{candle}.{ext}")
        ]
        # 检查正则表达式是否找到匹配项，只返回这些结果
        return [cls.rebuild_pair_from_filename(match[0]) for match in _tmp if match]

    @abstractmethod
    def ohlcv_store(
        self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType
    ) -> None:
        """
        存储 OHLCV 数据。
        :param pair: 交易对 - 用于生成文件名
        :param timeframe: 时间框架 - 用于生成文件名
        :param data: 包含 OHLCV 数据的数据框
        :param candle_type: CandleType 枚举值之一（必须与交易模式匹配！）
        :return: 无
        """

    def ohlcv_data_min_max(
        self, pair: str, timeframe: str, candle_type: CandleType
    ) -> tuple[datetime, datetime, int]:
        """
        返回给定交易对和时间框架的最小和最大时间戳。
        :param pair: 要获取最小/最大时间戳的交易对
        :param timeframe: 要获取最小/最大时间戳的时间框架
        :param candle_type: CandleType 枚举值之一（必须与交易模式匹配！）
        :return: (最小时间, 最大时间, 数据长度)
        """
        df = self._ohlcv_load(pair, timeframe, None, candle_type)
        if df.empty:
            return (
                datetime.fromtimestamp(0, tz=timezone.utc),
                datetime.fromtimestamp(0, tz=timezone.utc),
                0,
            )
        return df.iloc[0]["date"].to_pydatetime(), df.iloc[-1]["date"].to_pydatetime(), len(df)

    @abstractmethod
    def _ohlcv_load(
        self, pair: str, timeframe: str, timerange: TimeRange | None, candle_type: CandleType
    ) -> DataFrame:
        """
        用于从磁盘加载一个交易对数据的内部方法。
        实现数据加载并转换为 Pandas 数据框。
        时间范围修剪和数据框验证在该方法之外进行。
        :param pair: 要加载数据的交易对
        :param timeframe: 时间框架（例如 "5m"）
        :param timerange: 将加载的数据限制在此时间范围内。
                        子类可选择性实现以避免在可能的情况下加载所有数据。
        :param candle_type: CandleType 枚举值之一（必须与交易模式匹配！）
        :return: 包含 OHLCV 数据的数据框，或空数据框
        """

    def ohlcv_purge(self, pair: str, timeframe: str, candle_type: CandleType) -> bool:
        """
        删除此交易对的数据
        :param pair: 要删除数据的交易对
        :param timeframe: 时间框架（例如 "5m"）
        :param candle_type: CandleType 枚举值之一（必须与交易模式匹配！）
        :return: 如果删除成功则为 True，如果文件不存在则为 False
        """
        filename = self._pair_data_filename(self._datadir, pair, timeframe, candle_type)
        if filename.exists():
            filename.unlink()
            return True
        return False

    @abstractmethod
    def ohlcv_append(
        self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType
    ) -> None:
        """
        将数据追加到现有数据结构
        :param pair: 交易对
        :param timeframe: 此 OHLCV 数据对应的时间框架
        :param data: 要追加的数据
        :param candle_type: CandleType 枚举值之一（必须与交易模式匹配！）
        """

    @classmethod
    def trades_get_available_data(cls, datadir: Path, trading_mode: TradingMode) -> list[str]:
        """
        返回此数据目录中所有有交易数据的交易对列表
        :param datadir: 用于搜索交易文件的目录
        :param trading_mode: 要使用的交易模式
        :return: 元组列表 (交易对, 时间框架, 蜡烛类型)
        """
        if trading_mode == TradingMode.FUTURES:
            datadir = datadir.joinpath("futures")
        _tmp = [
            re.search(cls._TRADES_REGEX, p.name)
            for p in datadir.glob(f"*.{cls._get_file_extension()}")
        ]
        return [
            cls.rebuild_pair_from_filename(match[1])
            for match in _tmp
            if match and len(match.groups()) > 1
        ]

    def trades_data_min_max(
        self,
        pair: str,
        trading_mode: TradingMode,
    ) -> tuple[datetime, datetime, int]:
        """
        返回给定交易对的交易数据的最小和最大时间戳。
        :param pair: 要获取最小/最大时间戳的交易对
        :param trading_mode: 要使用的交易模式（用于确定文件名）
        :return: (最小时间, 最大时间, 数据长度)
        """
        df = self._trades_load(pair, trading_mode)
        if df.empty:
            return (
                datetime.fromtimestamp(0, tz=timezone.utc),
                datetime.fromtimestamp(0, tz=timezone.utc),
                0,
            )
        return (
            to_datetime(df.iloc[0]["timestamp"], unit="ms", utc=True).to_pydatetime(),
            to_datetime(df.iloc[-1]["timestamp"], unit="ms", utc=True).to_pydatetime(),
            len(df),
        )

    @classmethod
    def trades_get_pairs(cls, datadir: Path) -> list[str]:
        """
        返回所有有交易数据的交易对列表
        :param datadir: 用于搜索交易文件的目录
        :return: 交易对列表
        """
        _ext = cls._get_file_extension()
        _tmp = [
            re.search(r"^(\S+)(?=\-trades." + _ext + ")", p.name)
            for p in datadir.glob(f"*trades.{_ext}")
        ]
        # 检查正则表达式是否找到匹配项，只返回这些结果以避免异常
        return [cls.rebuild_pair_from_filename(match[0]) for match in _tmp if match]

    @abstractmethod
    def _trades_store(self, pair: str, data: DataFrame, trading_mode: TradingMode) -> None:
        """
        将交易数据（字典列表）存储到文件
        :param pair: 交易对 - 用于生成文件名
        :param data: 包含交易的数据框
                     列顺序与 DEFAULT_TRADES_COLUMNS 一致
        :param trading_mode: 要使用的交易模式（用于确定文件名）
        """

    @abstractmethod
    def trades_append(self, pair: str, data: DataFrame):
        """
        将数据追加到现有文件
        :param pair: 交易对 - 用于生成文件名
        :param data: 包含交易的数据框
                     列顺序与 DEFAULT_TRADES_COLUMNS 一致
        """

    @abstractmethod
    def _trades_load(
        self, pair: str, trading_mode: TradingMode, timerange: TimeRange | None = None
    ) -> DataFrame:
        """
        从文件加载交易对数据，可以是 .json.gz 或 .json 格式
        :param pair: 要加载交易数据的交易对
        :param trading_mode: 要使用的交易模式（用于确定文件名）
        :param timerange: 要加载的交易时间范围 - 目前未实现
        :return: 包含交易数据的数据框
        """

    def trades_store(self, pair: str, data: DataFrame, trading_mode: TradingMode) -> None:
        """
        将交易数据（字典列表）存储到文件
        :param pair: 交易对 - 用于生成文件名
        :param data: 包含交易的数据框
                     列顺序与 DEFAULT_TRADES_COLUMNS 一致
        :param trading_mode: 要使用的交易模式（用于确定文件名）
        """
        # 过滤预期的列（将移除实际的日期列）
        self._trades_store(pair, data[DEFAULT_TRADES_COLUMNS], trading_mode)

    def trades_purge(self, pair: str, trading_mode: TradingMode) -> bool:
        """
        删除此交易对的数据
        :param pair: 要删除数据的交易对
        :param trading_mode: 要使用的交易模式（用于确定文件名）
        :return: 如果删除成功则为 True，如果文件不存在则为 False
        """
        filename = self._pair_trades_filename(self._datadir, pair, trading_mode)
        if filename.exists():
            filename.unlink()
            return True
        return False

    def trades_load(
        self, pair: str, trading_mode: TradingMode, timerange: TimeRange | None = None
    ) -> DataFrame:
        """
        从文件加载交易对数据，可以是 .json.gz 或 .json 格式
        在此过程中移除重复项
        :param pair: 要加载交易数据的交易对
        :param trading_mode: 要使用的交易模式（用于确定文件名）
        :param timerange: 要加载的交易时间范围 - 目前未实现
        :return: 交易列表
        """
        try:
            trades = self._trades_load(pair, trading_mode, timerange=timerange)
        except Exception:
            logger.exception(f"加载 {pair} 的交易数据时出错")
            return DataFrame(columns=DEFAULT_TRADES_COLUMNS)

        trades = trades_df_remove_duplicates(trades)

        trades = trades_convert_types(trades)
        return trades

    @classmethod
    def create_dir_if_needed(cls, datadir: Path):
        """
        必要时创建数据目录
        目前只需要为 "futures" 模式创建目录
        """
        if not datadir.parent.is_dir():
            datadir.parent.mkdir()

    @classmethod
    def _pair_data_filename(
        cls,
        datadir: Path,
        pair: str,
        timeframe: str,
        candle_type: CandleType,
        no_timeframe_modify: bool = False,
    ) -> Path:
        pair_s = misc.pair_to_filename(pair)
        candle = ""
        if not no_timeframe_modify:
            timeframe = cls.timeframe_to_file(timeframe)

        if candle_type != CandleType.SPOT:
            datadir = datadir.joinpath("futures")
            candle = f"-{candle_type}"
        filename = datadir.joinpath(f"{pair_s}-{timeframe}{candle}.{cls._get_file_extension()}")
        return filename

    @classmethod
    def _pair_trades_filename(cls, datadir: Path, pair: str, trading_mode: TradingMode) -> Path:
        pair_s = misc.pair_to_filename(pair)
        if trading_mode == TradingMode.FUTURES:
            # 期货交易对...
            datadir = datadir.joinpath("futures")

        filename = datadir.joinpath(f"{pair_s}-trades.{cls._get_file_extension()}")
        return filename

    @staticmethod
    def timeframe_to_file(timeframe: str):
        return timeframe.replace("M", "Mo")

    @staticmethod
    def rebuild_timeframe_from_filename(timeframe: str) -> str:
        """
        将磁盘中的时间框架转换为文件格式
        将 mo 替换为 M（以避免在不区分大小写的文件系统上出现问题）
        """
        return re.sub("1mo", "1M", timeframe, flags=re.IGNORECASE)

    @staticmethod
    def rebuild_pair_from_filename(pair: str) -> str:
        """
        从文件名重建交易对名称
        假设资产名称最长为 7 个字符，以支持 BTC-PERP 和 BTC-PERP:USD 等名称
        """
        res = re.sub(r"^(([A-Za-z\d]{1,10})|^([A-Za-z\-]{1,6}))(_)", r"\g<1>/", pair, count=1)
        res = re.sub("_", ":", res, count=1)
        return res

    def ohlcv_load(
        self,
        pair,
        timeframe: str,
        candle_type: CandleType,
        *,
        timerange: TimeRange | None = None,
        fill_missing: bool = True,
        drop_incomplete: bool = False,
        startup_candles: int = 0,
        warn_no_data: bool = True,
    ) -> DataFrame:
        """
        加载给定交易对的缓存蜡烛图（OHLCV）数据。

        :param pair: 要加载数据的交易对
        :param timeframe: 时间框架（例如 "5m"）
        :param timerange: 将加载的数据限制在此时间范围内
        :param fill_missing: 用 "无操作" 蜡烛填充缺失值
        :param drop_incomplete: 丢弃最后一根蜡烛，假设它可能不完整
        :param startup_candles: 要在周期开始时加载的额外蜡烛数量
        :param warn_no_data: 当没有找到数据时记录警告消息
        :param candle_type: CandleType 枚举值之一（必须与交易模式匹配！）
        :return: 包含 OHLCV 数据的数据框，或空数据框
        """
        # 修正启动周期
        timerange_startup = deepcopy(timerange)
        if startup_candles > 0 and timerange_startup:
            timerange_startup.subtract_start(timeframe_to_seconds(timeframe) * startup_candles)

        pairdf = self._ohlcv_load(
            pair, timeframe, timerange=timerange_startup, candle_type=candle_type
        )
        if self._check_empty_df(pairdf, pair, timeframe, candle_type, warn_no_data):
            return pairdf
        else:
            enddate = pairdf.iloc[-1]["date"]
            if timerange_startup:
                self._validate_pairdata(pair, pairdf, timeframe, candle_type, timerange_startup)
                pairdf = trim_dataframe(pairdf, timerange_startup)
                if self._check_empty_df(pairdf, pair, timeframe, candle_type, warn_no_data, True):
                    return pairdf

            # 不完整的蜡烛只应在我们没有预先修剪结尾的情况下被丢弃
            pairdf = clean_ohlcv_dataframe(
                pairdf,
                timeframe,
                pair=pair,
                fill_missing=fill_missing,
                drop_incomplete=(drop_incomplete and enddate == pairdf.iloc[-1]["date"]),
            )
            self._check_empty_df(pairdf, pair, timeframe, candle_type, warn_no_data)
            return pairdf

    def _check_empty_df(
        self,
        pairdf: DataFrame,
        pair: str,
        timeframe: str,
        candle_type: CandleType,
        warn_no_data: bool,
        warn_price: bool = False,
    ) -> bool:
        """
        为空数据框发出警告
        """
        if pairdf.empty:
            if warn_no_data:
                logger.warning(
                    f"未找到 {pair}，{candle_type}，{timeframe} 的历史数据。"
                    "使用 `freqtrade download-data` 下载数据"
                )
            return True
        elif warn_price:
            candle_price_gap = 0
            if (
                candle_type in (CandleType.SPOT, CandleType.FUTURES)
                and not pairdf.empty
                and "close" in pairdf.columns
                and "open" in pairdf.columns
            ):
                # 检测前一根收盘价和当前开盘价之间的差距
                gaps = (pairdf["open"] - pairdf["close"].shift(1)) / pairdf["close"].shift(1)
                gaps = gaps.dropna()
                if len(gaps):
                    candle_price_gap = max(abs(gaps))
            if candle_price_gap > 0.1:
                logger.info(
                    f"在 {pair}，{timeframe}，{candle_type} 的两根蜡烛之间检测到 "
                    f"{candle_price_gap:.2%} 的价格跳空。"
                )

        return False

    def _validate_pairdata(
        self,
        pair,
        pairdata: DataFrame,
        timeframe: str,
        candle_type: CandleType,
        timerange: TimeRange,
    ):
        """
        验证交易对数据在开始和结束时是否有缺失数据，并记录警告。
        :param pairdata: 要验证的数据框
        :param timerange: 指定开始和结束日期的时间范围
        """

        if timerange.starttype == "date":
            if pairdata.iloc[0]["date"] > timerange.startdt:
                logger.warning(
                    f"{pair}，{candle_type}，{timeframe}，"
                    f"数据开始于 {pairdata.iloc[0]['date']:%Y-%m-%d %H:%M:%S}"
                )
        if timerange.stoptype == "date":
            if pairdata.iloc[-1]["date"] < timerange.stopdt:
                logger.warning(
                    f"{pair}，{candle_type}，{timeframe}，"
                    f"数据结束于 {pairdata.iloc[-1]['date']:%Y-%m-%d %H:%M:%S}"
                )

    def rename_futures_data(
        self, pair: str, new_pair: str, timeframe: str, candle_type: CandleType
    ):
        """
        临时方法，用于将数据从旧命名迁移到新命名（BTC/USDT -> BTC/USDT:USDT）
        仅用于币安，以支持币安期货命名统一
        """

        file_old = self._pair_data_filename(self._datadir, pair, timeframe, candle_type)
        file_new = self._pair_data_filename(self._datadir, new_pair, timeframe, candle_type)
        # print(file_old, file_new)
        if file_new.exists():
            logger.warning(f"{file_new} 已存在，无法迁移 {pair}。")
            return
        file_old.rename(file_new)

    def fix_funding_fee_timeframe(self, ff_timeframe: str):
        """
        临时方法，用于将数据从旧的资金费率时间框架迁移到正确的时间框架
        适用于 bybit 和 okx，其中资金费率和标记蜡烛有不同的时间框架
        """
        paircombs = self.ohlcv_get_available_data(self._datadir, TradingMode.FUTURES)
        funding_rate_combs = [
            f for f in paircombs if f[2] == CandleType.FUNDING_RATE and f[1] != ff_timeframe
        ]

        if funding_rate_combs:
            logger.warning(
                f"正在将 {len(funding_rate_combs)} 个资金费率迁移到正确的时间框架。"
            )

        for pair, timeframe, candletype in funding_rate_combs:
            old_name = self._pair_data_filename(self._datadir, pair, timeframe, candletype)
            new_name = self._pair_data_filename(self._datadir, pair, ff_timeframe, candletype)

            if not Path(old_name).exists():
                logger.warning(f"{old_name} 不存在，跳过。")
                continue

            if Path(new_name).exists():
                logger.warning(f"{new_name} 已存在，正在删除。")
                Path(new_name).unlink()

            Path(old_name).rename(new_name)


def get_datahandlerclass(datatype: str) -> type[IDataHandler]:
    """
    获取数据处理器类。
    可以使用解析器来完成，但由于这可能被频繁调用且解析器成本较高，
    直接这样做应该可以提高性能。
    :param datatype: 要使用的数据类型
    :return: 数据处理器类
    """

    if datatype == "json":
        from .jsondatahandler import JsonDataHandler

        return JsonDataHandler
    elif datatype == "jsongz":
        from .jsondatahandler import JsonGzDataHandler

        return JsonGzDataHandler
    elif datatype == "hdf5":
        raise OperationalException(
            "已弃用：hdf5 数据格式已被弃用，并已在 2025.1 中移除。"
            "请降级到 2024.12 并使用 convert-data 命令将您的数据转换为支持的格式。"
            "我们建议使用 feather 格式，因为它更快且更节省空间。"
        )

    elif datatype == "feather":
        from .featherdatahandler import FeatherDataHandler

        return FeatherDataHandler
    elif datatype == "parquet":
        from .parquetdatahandler import ParquetDataHandler

        return ParquetDataHandler
    else:
        raise ValueError(f"没有可用的 {datatype} 数据类型的数据处理器。")


def get_datahandler(
    datadir: Path, data_format: str | None = None, data_handler: IDataHandler | None = None
) -> IDataHandler:
    """
    :param datadir: 保存数据的文件夹
    :param data_format: 要使用的数据格式
    :param data_handler: 如果存在则返回此数据处理器，否则初始化一个新的
    """

    if not data_handler:
        HandlerClass = get_datahandlerclass(data_format or "feather")
        data_handler = HandlerClass(datadir)
    return data_handler
