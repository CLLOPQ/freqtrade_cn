"""
处理历史数据（OHLCV）。

包括：
* 从磁盘加载一个交易对（或交易对列表）的数据
* 从交易所下载数据并存储到磁盘
"""

# flake8: noqa: F401
import logging
import operator
from datetime import datetime, timedelta
from pathlib import Path

from pandas import DataFrame, concat

from freqtrade.configuration import TimeRange
from freqtrade.constants import DATETIME_PRINT_FORMAT, DL_DATA_TIMEFRAMES, DOCS_LINK, Config
from freqtrade.data.converter import (
    clean_ohlcv_dataframe,
    convert_trades_to_ohlcv,
    trades_df_remove_duplicates,
    trades_list_to_df,
)
from freqtrade.data.history.datahandlers import IDataHandler, get_datahandler
from freqtrade.enums import CandleType, TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import Exchange
from freqtrade.plugins.pairlist.pairlist_helpers import dynamic_expand_pairlist
from freqtrade.util import dt_now, dt_ts, format_ms_time, format_ms_time_det
from freqtrade.util.migrations import migrate_data
from freqtrade.util.progress_tracker import CustomProgress, retrieve_progress_tracker


logger = logging.getLogger(__name__)


def load_pair_history(
    pair: str,
    timeframe: str,
    datadir: Path,
    *,
    timerange: TimeRange | None = None,
    fill_up_missing: bool = True,
    drop_incomplete: bool = False,
    startup_candles: int = 0,
    data_format: str | None = None,
    data_handler: IDataHandler | None = None,
    candle_type: CandleType = CandleType.SPOT,
) -> DataFrame:
    """
    加载给定交易对的缓存 OHLCV 历史数据。

    :param pair: 要加载数据的交易对
    :param timeframe: 时间框架（例如 "5m"）
    :param datadir: 数据存储位置的路径
    :param data_format: 数据格式。如果设置了 data_handler，则忽略此参数
    :param timerange: 将加载的数据限制在此时间范围内
    :param fill_up_missing: 用 "无操作" 蜡烛填充缺失值
    :param drop_incomplete: 丢弃最后一根蜡烛，假设它可能不完整
    :param startup_candles: 要在周期开始时加载的额外蜡烛数量
    :param data_handler: 要使用的已初始化数据处理器
                         如果未设置，将从 data_format 初始化
    :param candle_type: CandleType 枚举值之一（必须与交易模式匹配！）
    :return: 包含 OHLCV 数据的 DataFrame，或空 DataFrame
    """
    data_handler = get_datahandler(datadir, data_format, data_handler)

    return data_handler.ohlcv_load(
        pair=pair,
        timeframe=timeframe,
        timerange=timerange,
        fill_missing=fill_up_missing,
        drop_incomplete=drop_incomplete,
        startup_candles=startup_candles,
        candle_type=candle_type,
    )


def load_data(
    datadir: Path,
    timeframe: str,
    pairs: list[str],
    *,
    timerange: TimeRange | None = None,
    fill_up_missing: bool = True,
    startup_candles: int = 0,
    fail_without_data: bool = False,
    data_format: str = "feather",
    candle_type: CandleType = CandleType.SPOT,
    user_futures_funding_rate: int | None = None,
) -> dict[str, DataFrame]:
    """
    加载交易对列表的 OHLCV 历史数据。

    :param datadir: 数据存储位置的路径
    :param timeframe: 时间框架（例如 "5m"）
    :param pairs: 要加载的交易对列表
    :param timerange: 将加载的数据限制在此时间范围内
    :param fill_up_missing: 用 "无操作" 蜡烛填充缺失值
    :param startup_candles: 要在周期开始时加载的额外蜡烛数量
    :param fail_without_data: 如果未找到数据，则引发 OperationalException
    :param data_format: 应使用的数据格式。默认为 json
    :param candle_type: CandleType 枚举值之一（必须与交易模式匹配！）
    :return: 字典，键为交易对，值为对应的 DataFrame
    """
    result: dict[str, DataFrame] = {}
    if startup_candles > 0 and timerange:
        logger.info(f"使用指标启动周期：{startup_candles} ...")

    data_handler = get_datahandler(datadir, data_format)

    for pair in pairs:
        hist = load_pair_history(
            pair=pair,
            timeframe=timeframe,
            datadir=datadir,
            timerange=timerange,
            fill_up_missing=fill_up_missing,
            startup_candles=startup_candles,
            data_handler=data_handler,
            candle_type=candle_type,
        )
        if not hist.empty:
            result[pair] = hist
        else:
            if candle_type is CandleType.FUNDING_RATE and user_futures_funding_rate is not None:
                logger.warning(f"{pair} 使用用户指定的 [{user_futures_funding_rate}]")
            elif candle_type not in (CandleType.SPOT, CandleType.FUTURES):
                result[pair] = DataFrame(columns=["date", "open", "close", "high", "low", "volume"])

    if fail_without_data and not result:
        raise OperationalException("未找到数据。终止。")
    return result


def refresh_data(
    *,
    datadir: Path,
    timeframe: str,
    pairs: list[str],
    exchange: Exchange,
    data_format: str | None = None,
    timerange: TimeRange | None = None,
    candle_type: CandleType,
) -> None:
    """
    刷新交易对列表的 OHLCV 历史数据。

    :param datadir: 数据存储位置的路径
    :param timeframe: 时间框架（例如 "5m"）
    :param pairs: 要加载的交易对列表
    :param exchange: 交易所对象
    :param data_format: 要使用的数据格式
    :param timerange: 将加载的数据限制在此时间范围内
    :param candle_type: CandleType 枚举值之一（必须与交易模式匹配！）
    """
    data_handler = get_datahandler(datadir, data_format)
    for pair in pairs:
        _download_pair_history(
            pair=pair,
            timeframe=timeframe,
            datadir=datadir,
            timerange=timerange,
            exchange=exchange,
            data_handler=data_handler,
            candle_type=candle_type,
        )


def _load_cached_data_for_updating(
    pair: str,
    timeframe: str,
    timerange: TimeRange | None,
    data_handler: IDataHandler,
    candle_type: CandleType,
    prepend: bool = False,
) -> tuple[DataFrame, int | None, int | None]:
    """
    加载缓存数据以下载更多数据。
    如果传入了时间范围，检查是否将下载存储数据之前的数据。
    如果是这种情况，则应完全覆盖可用数据。
    否则，下载始终从可用数据的末尾开始，以避免数据间隙。
    注意：仅由 download_pair_history() 使用。
    """
    start = None
    end = None
    if timerange:
        if timerange.starttype == "date":
            start = timerange.startdt
        if timerange.stoptype == "date":
            end = timerange.stopdt

    # 有意不传入时间范围 - 因为我们需要加载完整的数据集
    data = data_handler.ohlcv_load(
        pair,
        timeframe=timeframe,
        timerange=None,
        fill_missing=False,
        drop_incomplete=True,
        warn_no_data=False,
        candle_type=candle_type,
    )
    if not data.empty:
        if prepend:
            end = data.iloc[0]["date"]
        else:
            if start and start < data.iloc[0]["date"]:
                # 请求的开始日期早于现有数据，更新开始日期
                logger.info(
                    f"{pair}, {timeframe}, {candle_type}: "
                    f"请求的开始日期 {start:{DATETIME_PRINT_FORMAT}} 早于本地数据开始日期 "
                    f"{data.iloc[0]['date']:{DATETIME_PRINT_FORMAT}}。 "
                    f"使用 `--prepend` 下载 {data.iloc[0]['date']:{DATETIME_PRINT_FORMAT}} 之前的数据，或 "
                    "`--erase` 重新下载所有数据。"
                )
            start = data.iloc[-1]["date"]

    start_ms = int(start.timestamp() * 1000) if start else None
    end_ms = int(end.timestamp() * 1000) if end else None
    return data, start_ms, end_ms


def _download_pair_history(
    pair: str,
    *,
    datadir: Path,
    exchange: Exchange,
    timeframe: str = "5m",
    new_pairs_days: int = 30,
    data_handler: IDataHandler | None = None,
    timerange: TimeRange | None = None,
    candle_type: CandleType,
    erase: bool = False,
    prepend: bool = False,
) -> bool:
    """
    从交易所下载传入参数中的交易对和时间框架的最新蜡烛图
    数据从缓存中存在的最后正确数据开始下载。如果时间范围早于缓存中的数据，
    则将重新下载完整数据

    :param pair: 要下载的交易对
    :param timeframe: 时间框架（例如 "5m"）
    :param timerange: 要下载的时间范围
    :param candle_type: CandleType 枚举值之一（必须与交易模式匹配！）
    :param erase: 清除现有数据
    :return: 表示成功状态的布尔值
    """
    data_handler = get_datahandler(datadir, data_handler=data_handler)

    try:
        if erase:
            if data_handler.ohlcv_purge(pair, timeframe, candle_type=candle_type):
                logger.info(f"删除交易对 {pair}，{timeframe}，{candle_type} 的现有数据。")

        data, since_ms, until_ms = _load_cached_data_for_updating(
            pair,
            timeframe,
            timerange,
            data_handler=data_handler,
            candle_type=candle_type,
            prepend=prepend,
        )

        logger.info(
            f'下载 "{pair}"，{timeframe}，{candle_type} 的历史数据并存储在 {datadir} 中。 '
            f"从 {format_ms_time(since_ms) if since_ms else '开始'} 到 "
            f"{format_ms_time(until_ms) if until_ms else '现在'}"
        )

        logger.debug(
            "当前开始时间: %s",
            f"{data.iloc[0]['date']:{DATETIME_PRINT_FORMAT}}" if not data.empty else "无",
        )
        logger.debug(
            "当前结束时间: %s",
            f"{data.iloc[-1]['date']:{DATETIME_PRINT_FORMAT}}" if not data.empty else "无",
        )

        # 如果没有给出 since_ms，默认设置为 30 天前
        new_dataframe = exchange.get_historic_ohlcv(
            pair=pair,
            timeframe=timeframe,
            since_ms=(
                since_ms
                if since_ms
                else int((datetime.now() - timedelta(days=new_pairs_days)).timestamp()) * 1000
            ),
            is_new_pair=data.empty,
            candle_type=candle_type,
            until_ms=until_ms if until_ms else None,
        )
        logger.info(f"为 {pair} 下载了 {len(new_dataframe)} 条数据。")
        if data.empty:
            data = new_dataframe
        else:
            # 再次运行清理以确保没有重复的蜡烛
            # 特别是在现有数据和新数据之间
            data = clean_ohlcv_dataframe(
                concat([data, new_dataframe], axis=0),
                timeframe,
                pair,
                fill_missing=False,
                drop_incomplete=False,
            )

        logger.debug(
            "新开始时间: %s",
            f"{data.iloc[0]['date']:{DATETIME_PRINT_FORMAT}}" if not data.empty else "无",
        )
        logger.debug(
            "新结束时间: %s",
            f"{data.iloc[-1]['date']:{DATETIME_PRINT_FORMAT}}" if not data.empty else "无",
        )

        data_handler.ohlcv_store(pair, timeframe, data=data, candle_type=candle_type)
        return True

    except Exception:
        logger.exception(
            f'下载交易对 "{pair}"，时间框架 {timeframe} 的历史数据失败。'
        )
        return False


def refresh_backtest_ohlcv_data(
    exchange: Exchange,
    pairs: list[str],
    timeframes: list[str],
    datadir: Path,
    trading_mode: str,
    timerange: TimeRange | None = None,
    new_pairs_days: int = 30,
    erase: bool = False,
    data_format: str | None = None,
    prepend: bool = False,
    progress_tracker: CustomProgress | None = None,
) -> list[str]:
    """
    刷新用于回测和超参数优化操作的存储的 OHLCV 数据。
    由 freqtrade download-data 子命令使用。
    :return: 不可用的交易对列表
    """
    progress_tracker = retrieve_progress_tracker(progress_tracker)

    pairs_not_available = []
    data_handler = get_datahandler(datadir, data_format)
    candle_type = CandleType.get_default(trading_mode)
    with progress_tracker as progress:
        tf_length = len(timeframes) if trading_mode != "futures" else len(timeframes) + 2
        timeframe_task = progress.add_task("时间框架", total=tf_length)
        pair_task = progress.add_task("正在下载数据...", total=len(pairs))

        for pair in pairs:
            progress.update(pair_task, description=f"正在下载 {pair}")
            progress.update(timeframe_task, completed=0)

            if pair not in exchange.markets:
                pairs_not_available.append(f"{pair}: 交易对在交易所不可用。")
                logger.info(f"跳过交易对 {pair}...")
                continue
            for timeframe in timeframes:
                progress.update(timeframe_task, description=f"时间框架 {timeframe}")
                logger.debug(f"下载交易对 {pair}，{candle_type}，间隔 {timeframe}。")
                _download_pair_history(
                    pair=pair,
                    datadir=datadir,
                    exchange=exchange,
                    timerange=timerange,
                    data_handler=data_handler,
                    timeframe=str(timeframe),
                    new_pairs_days=new_pairs_days,
                    candle_type=candle_type,
                    erase=erase,
                    prepend=prepend,
                )
                progress.update(timeframe_task, advance=1)
            if trading_mode == "futures":
                # 取决于交易所的预定义蜡烛类型（和时间框架）
                # 下载基于期货数据回测所需的数据
                tf_mark = exchange.get_option("mark_ohlcv_timeframe")
                tf_funding_rate = exchange.get_option("funding_fee_timeframe")

                fr_candle_type = CandleType.from_string(exchange.get_option("mark_ohlcv_price"))
                # 所有交易所的期货交易都需要 FundingRate
                # 时间框架与标记价格时间框架对齐
                combs = ((CandleType.FUNDING_RATE, tf_funding_rate), (fr_candle_type, tf_mark))
                for candle_type_f, tf in combs:
                    logger.debug(f"下载交易对 {pair}，{candle_type_f}，间隔 {tf}。")
                    _download_pair_history(
                        pair=pair,
                        datadir=datadir,
                        exchange=exchange,
                        timerange=timerange,
                        data_handler=data_handler,
                        timeframe=str(tf),
                        new_pairs_days=new_pairs_days,
                        candle_type=candle_type_f,
                        erase=erase,
                        prepend=prepend,
                    )
                    progress.update(
                        timeframe_task, advance=1, description=f"时间框架 {candle_type_f}，{tf}"
                    )

            progress.update(pair_task, advance=1)
            progress.update(timeframe_task, description="时间框架")

    return pairs_not_available


def _download_trades_history(
    exchange: Exchange,
    pair: str,
    *,
    new_pairs_days: int = 30,
    timerange: TimeRange | None = None,
    data_handler: IDataHandler,
    trading_mode: TradingMode,
) -> bool:
    """
    从交易所下载交易历史。
    追加到先前下载的交易数据。
    """
    until = None
    since = 0
    if timerange:
        if timerange.starttype == "date":
            since = timerange.startts * 1000
        if timerange.stoptype == "date":
            until = timerange.stopts * 1000

    trades = data_handler.trades_load(pair, trading_mode)

    # TradesList 列在 constants.DEFAULT_TRADES_COLUMNS 中定义
    # DEFAULT_TRADES_COLUMNS: 0 -> timestamp
    # DEFAULT_TRADES_COLUMNS: 1 -> id

    if not trades.empty and since > 0 and (since + 1000) < trades.iloc[0]["timestamp"]:
        # since 早于第一笔交易
        raise ValueError(
            f"开始时间 {format_ms_time_det(since)} 早于可用数据 "
            f"({format_ms_time_det(trades.iloc[0]['timestamp'])}). "
            f"如果您想重新下载 {pair}，请使用 `--erase`。"
        )

    from_id = trades.iloc[-1]["id"] if not trades.empty else None
    if not trades.empty and since < trades.iloc[-1]["timestamp"]:
        # 将 since 重置为最后一个可用点
        # - 5 秒（确保我们获取所有交易）
        since = int(trades.iloc[-1]["timestamp"] - (5 * 1000))
        logger.info(
            f"使用最后一笔交易日期 -5s - 下载 {pair} 的交易，从 {format_ms_time(since)} 开始。"
        )

    if not since:
        since = dt_ts(dt_now() - timedelta(days=new_pairs_days))

    logger.debug(
        "当前开始时间: %s",
        "无" if trades.empty else f"{trades.iloc[0]['date']:{DATETIME_PRINT_FORMAT}}",
    )
    logger.debug(
        "当前结束时间: %s",
        "无" if trades.empty else f"{trades.iloc[-1]['date']:{DATETIME_PRINT_FORMAT}}",
    )
    logger.info(f"当前交易数量: {len(trades)}")

    new_trades = exchange.get_historic_trades(
        pair=pair,
        since=since,
        until=until,
        from_id=from_id,
    )
    new_trades_df = trades_list_to_df(new_trades[1])
    trades = concat([trades, new_trades_df], axis=0)
    # 移除重复项以确保我们不会存储不需要的数据
    trades = trades_df_remove_duplicates(trades)
    data_handler.trades_store(pair, trades, trading_mode)

    logger.debug(
        "新开始时间: %s",
        "无" if trades.empty else f"{trades.iloc[0]['date']:{DATETIME_PRINT_FORMAT}}",
    )
    logger.debug(
        "新结束时间: %s",
        "无" if trades.empty else f"{trades.iloc[-1]['date']:{DATETIME_PRINT_FORMAT}}",
    )
    logger.info(f"新交易数量: {len(trades)}")
    return True


def refresh_backtest_trades_data(
    exchange: Exchange,
    pairs: list[str],
    datadir: Path,
    timerange: TimeRange,
    trading_mode: TradingMode,
    new_pairs_days: int = 30,
    erase: bool = False,
    data_format: str = "feather",
    progress_tracker: CustomProgress | None = None,
) -> list[str]:
    """
    刷新用于回测和超参数优化操作的存储的交易数据。
    由 freqtrade download-data 子命令使用。
    :return: 不可用的交易对列表
    """
    progress_tracker = retrieve_progress_tracker(progress_tracker)
    pairs_not_available = []
    data_handler = get_datahandler(datadir, data_format=data_format)
    with progress_tracker as progress:
        pair_task = progress.add_task("正在下载数据...", total=len(pairs))
        for pair in pairs:
            progress.update(pair_task, description=f"正在下载交易 [{pair}]")
            if pair not in exchange.markets:
                pairs_not_available.append(f"{pair}: 交易对在交易所不可用。")
                logger.info(f"跳过交易对 {pair}...")
                continue

            if erase:
                if data_handler.trades_purge(pair, trading_mode):
                    logger.info(f"删除交易对 {pair} 的现有数据。")

            logger.info(f"下载交易对 {pair} 的交易。")
            try:
                _download_trades_history(
                    exchange=exchange,
                    pair=pair,
                    new_pairs_days=new_pairs_days,
                    timerange=timerange,
                    data_handler=data_handler,
                    trading_mode=trading_mode,
                )
            except ValueError as e:
                pairs_not_available.append(f"{pair}: {str(e)}")
            except Exception:
                logger.exception(
                    f'下载和存储交易对 "{pair}" 的历史交易失败。 '
                )

            progress.update(pair_task, advance=1)

    return pairs_not_available


def get_timerange(data: dict[str, DataFrame]) -> tuple[datetime, datetime]:
    """
    获取给定回测数据的最大公共时间范围。

    :param data: 预处理的回测数据字典
    :return: 包含最小日期、最大日期的元组
    """
    timeranges = [
        (frame["date"].min().to_pydatetime(), frame["date"].max().to_pydatetime())
        for frame in data.values()
    ]
    return (
        min(timeranges, key=operator.itemgetter(0))[0],
        max(timeranges, key=operator.itemgetter(1))[1],
    )


def validate_backtest_data(
    data: DataFrame, pair: str, min_date: datetime, max_date: datetime, timeframe_min: int
) -> bool:
    """
    验证预处理的回测数据是否存在缺失值，并显示相关警告。

    :param data: 预处理的回测数据（DataFrame 形式）
    :param pair: 用于日志输出的交易对
    :param min_date: 数据的开始日期
    :param max_date: 数据的结束日期
    :param timeframe_min: 时间框架（分钟）
    """
    # 总差异（分钟）/ 时间框架（分钟）
    expected_frames = int((max_date - min_date).total_seconds() // 60 // timeframe_min)
    found_missing = False
    dflen = len(data)
    if dflen < expected_frames:
        found_missing = True
        logger.warning(
            "%s 存在缺失的框架：预期 %s，实际 %s，缺失 %s 个值",
            pair,
            expected_frames,
            dflen,
            expected_frames - dflen,
        )
    return found_missing


def download_data_main(config: Config) -> None:
    from freqtrade.resolvers.exchange_resolver import ExchangeResolver

    exchange = ExchangeResolver.load_exchange(config, validate=False)

    download_data(config, exchange)


def download_data(
    config: Config,
    exchange: Exchange,
    *,
    progress_tracker: CustomProgress | None = None,
) -> None:
    """
    数据下载函数。用于命令行界面和 API。
    """
    timerange = TimeRange()
    if "days" in config and config["days"] is not None:
        time_since = (datetime.now() - timedelta(days=config["days"])).strftime("%Y%m%d")
        timerange = TimeRange.parse_timerange(f"{time_since}-")

    if "timerange" in config:
        timerange = TimeRange.parse_timerange(config["timerange"])

    # 移除基准货币以跳过与数据下载无关的检查
    config["stake_currency"] = ""

    pairs_not_available: list[str] = []

    available_pairs = [
        p
        for p in exchange.get_markets(
            tradable_only=True, active_only=not config.get("include_inactive")
        ).keys()
    ]

    expanded_pairs = dynamic_expand_pairlist(config, available_pairs)
    if "timeframes" not in config:
        config["timeframes"] = DL_DATA_TIMEFRAMES

    if len(expanded_pairs) == 0:
        logger.warning(
            "没有可下载的交易对。请确保您为所选的交易模式使用了正确的交易对命名。 \n"
            f"更多信息：{DOCS_LINK}/bot-basics/#pair-naming"
        )
        return

    logger.info(
        f"即将下载交易对：{expanded_pairs}，时间间隔：{config['timeframes']} 到 {config['datadir']}"
    )

    for timeframe in config["timeframes"]:
        exchange.validate_timeframes(timeframe)

    # 开始下载
    try:
        if config.get("download_trades"):
            if not exchange.get_option("trades_has_history", True):
                raise OperationalException(
                    f"{exchange.name} 不支持交易历史。您不能为此交易所使用 --dl-trades。"
                )
            pairs_not_available = refresh_backtest_trades_data(
                exchange,
                pairs=expanded_pairs,
                datadir=config["datadir"],
                timerange=timerange,
                new_pairs_days=config["new_pairs_days"],
                erase=bool(config.get("erase")),
                data_format=config["dataformat_trades"],
                trading_mode=config.get("trading_mode", TradingMode.SPOT),
                progress_tracker=progress_tracker,
            )

            if config.get("convert_trades") or not exchange.get_option("ohlcv_has_history", True):
                # 将下载的交易数据转换为不同的时间框架
                # 仅对没有历史 K 线的交易所自动转换

                convert_trades_to_ohlcv(
                    pairs=expanded_pairs,
                    timeframes=config["timeframes"],
                    datadir=config["datadir"],
                    timerange=timerange,
                    erase=bool(config.get("erase")),
                    data_format_ohlcv=config["dataformat_ohlcv"],
                    data_format_trades=config["dataformat_trades"],
                    candle_type=config.get("candle_type_def", CandleType.SPOT),
                )
        else:
            if not exchange.get_option("ohlcv_has_history", True):
                if not exchange.get_option("trades_has_history", True):
                    raise OperationalException(
                        f"{exchange.name} 不支持历史数据。{exchange.name} 不支持下载交易或 OHLCV 数据。"
                    )
                else:
                    raise OperationalException(
                        f"{exchange.name} 不支持历史 K 线。请为此交易所使用 `--dl-trades` "
                        "（不幸的是，这将需要很长时间）。"
                    )
            migrate_data(config, exchange)
            pairs_not_available = refresh_backtest_ohlcv_data(
                exchange,
                pairs=expanded_pairs,
                timeframes=config["timeframes"],
                datadir=config["datadir"],
                timerange=timerange,
                new_pairs_days=config["new_pairs_days"],
                erase=bool(config.get("erase")),
                data_format=config["dataformat_ohlcv"],
                trading_mode=config.get("trading_mode", "spot"),
                prepend=config.get("prepend_data", False),
                progress_tracker=progress_tracker,
            )
    finally:
        if pairs_not_available:
            errors = "\n" + ("\n".join(pairs_not_available))
            logger.warning(
                f"从 {exchange.name} 下载以下交易对时遇到问题：{errors}"
            )