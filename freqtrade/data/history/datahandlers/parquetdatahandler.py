import logging

from pandas import DataFrame, read_parquet, to_datetime

from freqtrade.configuration import TimeRange
from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS, DEFAULT_TRADES_COLUMNS
from freqtrade.enums import CandleType, TradingMode

from .idatahandler import IDataHandler


logger = logging.getLogger(__name__)


class ParquetDataHandler(IDataHandler):
    _columns = DEFAULT_DATAFRAME_COLUMNS

    def ohlcv_store(
        self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType
    ) -> None:
        """
        以 Parquet 格式存储数据。
            格式如下：
            [[<date>,<open>,<high>,<low>,<close>]]
        :param pair: 交易对 - 用于生成文件名
        :param timeframe: 时间框架 - 用于生成文件名
        :param data: 包含 OHLCV 数据的数据框
        :param candle_type: CandleType 枚举值之一（必须与交易模式匹配！）
        :return: 无
        """
        filename = self._pair_data_filename(self._datadir, pair, timeframe, candle_type)
        self.create_dir_if_needed(filename)

        data.reset_index(drop=True).loc[:, self._columns].to_parquet(filename)

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
        filename = self._pair_data_filename(self._datadir, pair, timeframe, candle_type=candle_type)
        if not filename.exists():
            # 1M 文件的 fallback 模式
            filename = self._pair_data_filename(
                self._datadir, pair, timeframe, candle_type=candle_type, no_timeframe_modify=True
            )
            if not filename.exists():
                return DataFrame(columns=self._columns)
        try:
            pairdata = read_parquet(filename)
            pairdata.columns = self._columns
            pairdata = pairdata.astype(
                dtype={
                    "open": "float",
                    "high": "float",
                    "low": "float",
                    "close": "float",
                    "volume": "float",
                }
            )
            pairdata["date"] = to_datetime(pairdata["date"], unit="ms", utc=True)
            return pairdata
        except Exception as e:
            logger.exception(
                f"从 {filename} 加载数据时出错。异常：{e}。返回空数据框。"
            )
            return DataFrame(columns=self._columns)

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
        raise NotImplementedError()

    def _trades_store(self, pair: str, data: DataFrame, trading_mode: TradingMode) -> None:
        """
        将交易数据（字典列表）存储到文件
        :param pair: 交易对 - 用于生成文件名
        :param data: 包含交易的数据框
                     列顺序与 DEFAULT_TRADES_COLUMNS 一致
        :param trading_mode: 要使用的交易模式（用于确定文件名）
        """
        filename = self._pair_trades_filename(self._datadir, pair, trading_mode)
        self.create_dir_if_needed(filename)
        data.reset_index(drop=True).to_parquet(filename)

    def trades_append(self, pair: str, data: DataFrame):
        """
        将数据追加到现有文件
        :param pair: 交易对 - 用于生成文件名
        :param data: 包含交易的数据框
                     列顺序与 DEFAULT_TRADES_COLUMNS 一致
        """
        raise NotImplementedError()

    def _trades_load(
        self, pair: str, trading_mode: TradingMode, timerange: TimeRange | None = None
    ) -> DataFrame:
        """
        从文件加载交易对数据，可以是 .json.gz 或 .json 格式
        # TODO: 考虑时间范围...
        :param pair: 要加载交易数据的交易对
        :param trading_mode: 要使用的交易模式（用于确定文件名）
        :param timerange: 要加载的交易时间范围 - 目前未实现
        :return: 交易列表
        """
        filename = self._pair_trades_filename(self._datadir, pair, trading_mode)
        if not filename.exists():
            return DataFrame(columns=DEFAULT_TRADES_COLUMNS)

        tradesdata = read_parquet(filename)

        return tradesdata

    @classmethod
    def _get_file_extension(cls):
        return "parquet"