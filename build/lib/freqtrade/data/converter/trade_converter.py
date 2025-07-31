"""
用于将数据从一种格式转换为另一种格式的函数
"""

import logging
from pathlib import Path

import pandas as pd
from pandas import DataFrame, to_datetime

from freqtrade.configuration import TimeRange
from freqtrade.constants import (
    DEFAULT_DATAFRAME_COLUMNS,
    DEFAULT_TRADES_COLUMNS,
    TRADES_DTYPES,
    Config,
    TradeList,
)
from freqtrade.enums import CandleType, TradingMode
from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


def trades_df_remove_duplicates(trades: pd.DataFrame) -> pd.DataFrame:
    """
    从交易数据框中移除重复项。
    使用 pandas.DataFrame.drop_duplicates 基于 'timestamp' 列移除重复项。
    :param trades: 包含 constants.DEFAULT_TRADES_COLUMNS 列的数据框
    :return: 基于 'timestamp' 列移除重复项后的数据框
    """
    return trades.drop_duplicates(subset=["timestamp", "id"])


def trades_dict_to_list(trades: list[dict]) -> TradeList:
    """
    将 fetch_trades 结果转换为列表（以提高内存效率）。
    :param trades: 交易列表，如 ccxt.fetch_trades 返回的格式
    :return: 列表的列表，以 constants.DEFAULT_TRADES_COLUMNS 作为列
    """
    return [[t[col] for col in DEFAULT_TRADES_COLUMNS] for t in trades]


def trades_convert_types(trades: DataFrame) -> DataFrame:
    """
    转换交易数据类型并添加 'date' 列
    """
    trades = trades.astype(TRADES_DTYPES)
    trades["date"] = to_datetime(trades["timestamp"], unit="ms", utc=True)
    return trades


def trades_list_to_df(trades: TradeList, convert: bool = True):
    """
    将交易列表转换为数据框
    :param trades: 列表的列表，以 constants.DEFAULT_TRADES_COLUMNS 作为列
    """
    if not trades:
        df = DataFrame(columns=DEFAULT_TRADES_COLUMNS)
    else:
        df = DataFrame(trades, columns=DEFAULT_TRADES_COLUMNS)

    if convert:
        df = trades_convert_types(df)

    return df


def trades_to_ohlcv(trades: DataFrame, timeframe: str) -> DataFrame:
    """
    将交易列表转换为 OHLCV 列表
    :param trades: 交易列表，如 ccxt.fetch_trades 返回的格式
    :param timeframe: 重采样数据的时间框架
    :return: OHLCV 数据框
    :raises: 如果没有提供交易数据则抛出 ValueError
    """
    from freqtrade.exchange import timeframe_to_resample_freq

    if trades.empty:
        raise ValueError("交易列表为空。")
    df = trades.set_index("date", drop=True)
    resample_interval = timeframe_to_resample_freq(timeframe)
    df_new = df["price"].resample(resample_interval).ohlc()
    df_new["volume"] = df["amount"].resample(resample_interval).sum()
    df_new["date"] = df_new.index
    # 删除成交量为 0 的行
    df_new = df_new.dropna()
    return df_new.loc[:, DEFAULT_DATAFRAME_COLUMNS]


def convert_trades_to_ohlcv(
    pairs: list[str],
    timeframes: list[str],
    datadir: Path,
    timerange: TimeRange,
    erase: bool,
    data_format_ohlcv: str,
    data_format_trades: str,
    candle_type: CandleType,
) -> None:
    """
    将存储的交易数据转换为 OHLCV 数据
    """
    from freqtrade.data.history import get_datahandler

    data_handler_trades = get_datahandler(datadir, data_format=data_format_trades)
    data_handler_ohlcv = get_datahandler(datadir, data_format=data_format_ohlcv)

    logger.info(
        f"即将转换交易对：'{', '.join(pairs)}'，"
        f"时间间隔：'{', '.join(timeframes)}' 到 {datadir}"
    )
    trading_mode = TradingMode.FUTURES if candle_type != CandleType.SPOT else TradingMode.SPOT
    for pair in pairs:
        trades = data_handler_trades.trades_load(pair, trading_mode)
        for timeframe in timeframes:
            if erase:
                if data_handler_ohlcv.ohlcv_purge(pair, timeframe, candle_type=candle_type):
                    logger.info(f"正在删除交易对 {pair}，时间间隔 {timeframe} 的现有数据。")
            try:
                ohlcv = trades_to_ohlcv(trades, timeframe)
                # 存储 OHLCV
                data_handler_ohlcv.ohlcv_store(pair, timeframe, data=ohlcv, candle_type=candle_type)
            except ValueError:
                logger.warning(f"无法将 {pair} 转换为 OHLCV。")


def convert_trades_format(config: Config, convert_from: str, convert_to: str, erase: bool):
    """
    将交易数据从一种格式转换为另一种格式。
    :param config: 配置字典
    :param convert_from: 源格式
    :param convert_to: 目标格式
    :param erase: 擦除源数据（如果源格式和目标格式相同则不适用）
    """
    if convert_from == "kraken_csv":
        if config["exchange"]["name"] != "kraken":
            raise OperationalException(
                "仅支持从 Kraken 的 CSV 格式转换。"
                "有关此特殊模式的详细信息，请参阅文档。"
            )
        from freqtrade.data.converter.trade_converter_kraken import import_kraken_trades_from_csv

        import_kraken_trades_from_csv(config, convert_to)
        return

    from freqtrade.data.history import get_datahandler

    src = get_datahandler(config["datadir"], convert_from)
    trg = get_datahandler(config["datadir"], convert_to)

    if "pairs" not in config:
        config["pairs"] = src.trades_get_pairs(config["datadir"])
    logger.info(f"正在转换 {config['pairs']} 的交易数据")
    trading_mode: TradingMode = config.get("trading_mode", TradingMode.SPOT)
    for pair in config["pairs"]:
        data = src.trades_load(pair, trading_mode)
        logger.info(f"正在转换 {pair} 的 {len(data)} 笔交易")
        trg.trades_store(pair, data, trading_mode)

        if erase and convert_from != convert_to:
            logger.info(f"正在删除 {pair} 的源交易数据。")
            src.trades_purge(pair, trading_mode)