"""
从 https://data.binance.vision/ 获取每日归档的 OHLCV 数据
文档可参考 https://github.com/binance/binance-public-data
"""

import asyncio
import logging
import zipfile
from datetime import date, timedelta
from io import BytesIO
from typing import Any

import aiohttp
import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.constants import DEFAULT_TRADES_COLUMNS
from freqtrade.enums import CandleType
from freqtrade.misc import chunks
from freqtrade.util.datetime_helpers import dt_from_ts, dt_now


logger = logging.getLogger(__name__)


class Http404(Exception):
    def __init__(self, msg, date, url):
        super().__init__(msg)
        self.date = date
        self.url = url


class BadHttpStatus(Exception):
    """非 200/404 状态码"""

    pass


async def download_archive_ohlcv(
    candle_type: CandleType,
    pair: str,
    timeframe: str,
    *,
    since_ms: int,
    until_ms: int | None,
    markets: dict[str, Any],
    stop_on_404: bool = True,
) -> DataFrame:
    """
    从 https://data.binance.vision 获取 OHLCV 数据
    函数会尽力下载时间范围 [`since_ms`, `until_ms`] 内的数据
    包含 `since_ms`，但不包含 `until_ms`。
    如果 `stop_one_404` 为 True，返回的 DataFrame 保证从 `since_ms` 开始且数据无间隙。

    :candle_type: 目前仅支持现货和期货
    :pair: CCXT 格式的交易对名称
    :since_ms: 数据的开始时间戳（包含）
    :until_ms: 数据的结束时间戳（不包含）
    :param until_ms: `None` 表示最新可用数据的时间戳
    :markets: CCXT 市场字典，为 None 时，函数将从新的 `ccxt.binance` 实例加载市场数据
    :param stop_on_404: 当返回 404 时停止下载后续数据
    :return: 日期范围在 [since_ms, until_ms) 之间，若时间范围内无可用数据则返回空 DataFrame
    """
    try:
        symbol = markets[pair]["id"]

        start = dt_from_ts(since_ms)
        end = dt_from_ts(until_ms) if until_ms else dt_now()

        # 我们使用两天前作为最后可用日期，因为每日归档是每日上传且有几小时延迟
        last_available_date = dt_now() - timedelta(days=2)
        end = min(end, last_available_date)
        if start >= end:
            return DataFrame()
        df = await _download_archive_ohlcv(
            symbol, pair, timeframe, candle_type, start, end, stop_on_404
        )
        logger.debug(
            f"已从 https://data.binance.vision 下载 {pair} 的数据，长度为 {len(df)}。"
        )
    except Exception as e:
        logger.warning(
            "从 Binance 快速下载时发生异常，将回退到较慢的 REST API，这可能需要更多时间。",
            exc_info=e,
        )
        df = DataFrame()

    if not df.empty:
        # 仅返回请求时间范围内的数据
        return df.loc[(df["date"] >= start) & (df["date"] < end)]
    else:
        return df


def concat_safe(dfs) -> DataFrame:
    if all(df is None for df in dfs):
        return DataFrame()
    else:
        return pd.concat(dfs)


async def _download_archive_ohlcv(
    symbol: str,
    pair: str,
    timeframe: str,
    candle_type: CandleType,
    start: date,
    end: date,
    stop_on_404: bool,
) -> DataFrame:
    # 每日数据帧，`None` 表示当天数据缺失（当 `stop_on_404` 为 False 时）
    dfs: list[DataFrame | None] = []
    # 当前处理的天数，从 1 开始
    current_day = 0

    connector = aiohttp.TCPConnector(limit=100)
    async with aiohttp.ClientSession(connector=connector, trust_env=True) as session:
        # HTTP 连接已被 TCPConnector 限制
        for dates in chunks(list(date_range(start, end)), 1000):
            tasks = [
                asyncio.create_task(get_daily_ohlcv(symbol, timeframe, candle_type, date, session))
                for date in dates
            ]
            for task in tasks:
                current_day += 1
                try:
                    df = await task
                except Http404 as e:
                    if stop_on_404:
                        logger.debug(f"因 404 错误无法下载 {e.url}。")

                        # 第一天出现 404 错误表示 https://data.binance.vision 上缺少数据，
                        # 我们提供警告和建议。
                        # https://github.com/freqtrade/freqtrade/blob/acc53065e5fa7ab5197073276306dc9dc3adbfa3/tests/exchange_online/test_binance_compare_ohlcv.py#L7
                        if current_day == 1:
                            logger.warning(
                                f"因缺少数据，快速下载不可用：{e.url}。将回退到较慢的 REST API，这可能需要更多时间。"
                            )
                            if pair in ["BTC/USDT:USDT", "ETH/USDT:USDT", "BCH/USDT:USDT"]:
                                logger.warning(
                                    f"为避免延迟，您可以先使用 `--timerange <开始日期>-20200101` 下载 {pair}，"
                                    "然后使用 `--timerange 20200101-<结束日期>` 下载剩余数据。"
                                )
                        else:
                            logger.warning(
                                f"{pair} 的 Binance 快速下载在 {e.date} 因缺少数据而停止：{e.url}，"
                                "将回退到 REST API 下载剩余数据，这可能需要更多时间。"
                            )
                        await cancel_and_await_tasks(tasks[tasks.index(task) + 1 :])
                        return concat_safe(dfs)
                    else:
                        dfs.append(None)
                except Exception as e:
                    logger.warning(f"发生异常：{e}")
                    # 直接返回现有数据，不允许数据中存在间隙
                    await cancel_and_await_tasks(tasks[tasks.index(task) + 1 :])
                    return concat_safe(dfs)
                else:
                    dfs.append(df)
    return concat_safe(dfs)


async def cancel_and_await_tasks(unawaited_tasks):
    """取消并等待任务完成"""
    logger.debug("尝试取消未完成的下载任务。")
    for task in unawaited_tasks:
        task.cancel()
    await asyncio.gather(*unawaited_tasks, return_exceptions=True)
    logger.debug("所有下载任务均已处理完毕。")


def date_range(start: date, end: date):
    date = start
    while date <= end:
        yield date
        date += timedelta(days=1)


def binance_vision_zip_name(symbol: str, timeframe: str, date: date) -> str:
    return f"{symbol}-{timeframe}-{date.strftime('%Y-%m-%d')}.zip"


def candle_type_to_url_segment(candle_type: CandleType) -> str:
    if candle_type == CandleType.SPOT:
        return "spot"
    elif candle_type == CandleType.FUTURES:
        return "futures/um"
    else:
        raise ValueError(f"不支持的 CandleType：{candle_type}")


def binance_vision_ohlcv_zip_url(
    symbol: str, timeframe: str, candle_type: CandleType, date: date
) -> str:
    """
    示例 URL：
    https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1s/BTCUSDT-1s-2023-10-27.zip
    https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/1h/BTCUSDT-1h-2023-10-27.zip
    """
    asset_type_url_segment = candle_type_to_url_segment(candle_type)
    url = (
        f"https://data.binance.vision/data/{asset_type_url_segment}/daily/klines/{symbol}"
        f"/{timeframe}/{binance_vision_zip_name(symbol, timeframe, date)}"
    )
    return url


def binance_vision_trades_zip_url(symbol: str, candle_type: CandleType, date: date) -> str:
    """
    示例 URL：
    https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2023-10-27.zip
    https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2023-10-27.zip
    """
    asset_type_url_segment = candle_type_to_url_segment(candle_type)
    url = (
        f"https://data.binance.vision/data/{asset_type_url_segment}/daily/aggTrades/{symbol}"
        f"/{symbol}-aggTrades-{date.strftime('%Y-%m-%d')}.zip"
    )
    return url


async def get_daily_ohlcv(
    symbol: str,
    timeframe: str,
    candle_type: CandleType,
    date: date,
    session: aiohttp.ClientSession,
    retry_count: int = 3,
    retry_delay: float = 0.0,
) -> DataFrame:
    """
    从 https://data.binance.vision 获取每日 OHLCV 数据
    参见 https://github.com/binance/binance-public-data

    :symbol: binance 符号名称，例如 BTCUSDT
    :timeframe: 例如 1m, 1h
    :candle_type: SPOT 或 FUTURES
    :date: 返回的 DataFrame 将覆盖 UTC 时间的 `date` 全天
    :session: aiohttp.ClientSession 实例
    :retry_count: 返回异常前的重试次数
    :retry_delay: 每次重试前的等待时间
    :return: 包含 date,open,high,low,close,volume 列的数据帧
    """

    url = binance_vision_ohlcv_zip_url(symbol, timeframe, candle_type, date)

    logger.debug(f"从 binance 下载数据：{url}")

    retry = 0
    while True:
        if retry > 0:
            sleep_secs = retry * retry_delay
            logger.debug(
                f"[{retry}/{retry_count}] 在 {sleep_secs} 秒后重试下载 {url}"
            )
            await asyncio.sleep(sleep_secs)
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    logger.debug(f"成功下载 {url}")
                    with zipfile.ZipFile(BytesIO(content)) as zipf:
                        with zipf.open(zipf.namelist()[0]) as csvf:
                            # https://github.com/binance/binance-public-data/issues/283
                            first_byte = csvf.read(1)[0]
                            if chr(first_byte).isdigit():
                                header = None
                            else:
                                header = 0
                            csvf.seek(0)

                            df = pd.read_csv(
                                csvf,
                                usecols=[0, 1, 2, 3, 4, 5],
                                names=["date", "open", "high", "low", "close", "volume"],
                                header=header,
                            )
                            df["date"] = pd.to_datetime(
                                np.where(df["date"] > 1e13, df["date"] // 1000, df["date"]),
                                unit="ms",
                                utc=True,
                            )
                            return df
                elif resp.status == 404:
                    logger.debug(f"下载 {url} 失败")
                    raise Http404(f"404: {url}", date, url)
                else:
                    raise BadHttpStatus(f"{resp.status} - {resp.reason}")
        except Exception as e:
            retry += 1
            if isinstance(e, Http404) or retry > retry_count:
                logger.debug(f"从 {url} 获取数据失败：{e}")
                raise


async def download_archive_trades(
    candle_type: CandleType,
    pair: str,
    *,
    since_ms: int,
    until_ms: int | None,
    markets: dict[str, Any],
    stop_on_404: bool = True,
) -> tuple[str, list[list]]:
    try:
        symbol = markets[pair]["id"]

        last_available_date = dt_now() - timedelta(days=2)

        start = dt_from_ts(since_ms)
        end = dt_from_ts(until_ms) if until_ms else dt_now()
        end = min(end, last_available_date)
        if start >= end:
            return pair, []
        result_list = await _download_archive_trades(
            symbol, pair, candle_type, start, end, stop_on_404
        )
        return pair, result_list

    except Exception as e:
        logger.warning(
            "从 Binance 快速下载交易数据时发生异常，将回退到较慢的 REST API，这可能需要更多时间。",
            exc_info=e,
        )
        return pair, []


def parse_trades_from_zip(csvf):
    # https://github.com/binance/binance-public-data/issues/283
    first_byte = csvf.read(1)[0]
    if chr(first_byte).isdigit():
        # 现货
        header = None
        names = [
            "id",
            "price",
            "amount",
            "first_trade_id",
            "last_trade_id",
            "timestamp",
            "is_buyer_maker",
            "is_best_match",
        ]
    else:
        # 期货
        header = 0
        names = [
            "id",
            "price",
            "amount",
            "first_trade_id",
            "last_trade_id",
            "timestamp",
            "is_buyer_maker",
        ]
    csvf.seek(0)

    df = pd.read_csv(
        csvf,
        names=names,
        header=header,
    )
    df.loc[:, "cost"] = df["price"] * df["amount"]
    # 基于 ccxt parseTrade 逻辑，故意反转方向
    df.loc[:, "side"] = np.where(df["is_buyer_maker"], "sell", "buy")
    df.loc[:, "type"] = None
    # 将时间戳转换为毫秒
    df.loc[:, "timestamp"] = np.where(
        df["timestamp"] > 1e13,
        df["timestamp"] // 1000,
        df["timestamp"],
    )
    return df.loc[:, DEFAULT_TRADES_COLUMNS].to_records(index=False).tolist()


async def get_daily_trades(
    symbol: str,
    candle_type: CandleType,
    date: date,
    session: aiohttp.ClientSession,
    retry_count: int = 3,
    retry_delay: float = 0.0,
) -> list[list]:
    """
    从 https://data.binance.vision 获取每日交易数据
    参见 https://github.com/binance/binance-public-data

    :symbol: binance 符号名称，例如 BTCUSDT
    :candle_type: SPOT 或 FUTURES
    :date: 返回的数据将覆盖 UTC 时间的 `date` 全天
    :session: aiohttp.ClientSession 实例
    :retry_count: 返回异常前的重试次数
    :retry_delay: 每次重试前的等待时间
    :return: 包含 DEFAULT_TRADES_COLUMNS 格式交易数据的列表
    """

    url = binance_vision_trades_zip_url(symbol, candle_type, date)

    logger.debug(f"从 binance 下载交易数据：{url}")

    retry = 0
    while True:
        if retry > 0:
            sleep_secs = retry * retry_delay
            logger.debug(
                f"[{retry}/{retry_count}] 在 {sleep_secs} 秒后重试下载 {url}"
            )
            await asyncio.sleep(sleep_secs)
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    logger.debug(f"成功下载 {url}")
                    with zipfile.ZipFile(BytesIO(content)) as zipf:
                        with zipf.open(zipf.namelist()[0]) as csvf:
                            return parse_trades_from_zip(csvf)
                elif resp.status == 404:
                    logger.debug(f"下载 {url} 失败")
                    raise Http404(f"404: {url}", date, url)
                else:
                    raise BadHttpStatus(f"{resp.status} - {resp.reason}")
        except Exception as e:
            logger.info("下载每日交易数据时发生：%s", e)
            retry += 1
            if isinstance(e, Http404) or retry > retry_count:
                logger.debug(f"从 {url} 获取数据失败：{e}")
                raise


async def _download_archive_trades(
    symbol: str,
    pair: str,
    candle_type: CandleType,
    start: date,
    end: date,
    stop_on_404: bool,
) -> list[list]:
    # 每日数据列表，`None` 表示当天数据缺失（当 `stop_on_404` 为 False 时）
    results: list[list] = []
    # 当前处理的天数，从 1 开始
    current_day = 0

    connector = aiohttp.TCPConnector(limit=100)
    async with aiohttp.ClientSession(connector=connector, trust_env=True) as session:
        # HTTP 连接已被 TCPConnector 限制
        for dates in chunks(list(date_range(start, end)), 30):
            tasks = [
                asyncio.create_task(get_daily_trades(symbol, candle_type, date, session))
                for date in dates
            ]
            for task in tasks:
                current_day += 1
                try:
                    result = await task
                except Http404 as e:
                    if stop_on_404:
                        logger.debug(f"因 404 错误无法下载 {e.url}。")

                        # 第一天出现 404 错误表示 https://data.binance.vision 上缺少数据，
                        # 我们提供警告和建议。
                        # https://github.com/freqtrade/freqtrade/blob/acc53065e5fa7ab5197073276306dc9dc3adbfa3/tests/exchange_online/test_binance_compare_ohlcv.py#L7
                        if current_day == 1:
                            logger.warning(
                                f"因缺少数据，快速下载不可用：{e.url}。将回退到较慢的 REST API，这可能需要更多时间。"
                            )
                            if pair in ["BTC/USDT:USDT", "ETH/USDT:USDT", "BCH/USDT:USDT"]:
                                logger.warning(
                                    f"为避免延迟，您可以先使用 `--timerange <开始日期>-20200101` 下载 {pair}，"
                                    "然后使用 `--timerange 20200101-<结束日期>` 下载剩余数据。"
                                )
                        else:
                            logger.warning(
                                f"{pair} 的 Binance 快速下载在 {e.date} 因缺少数据而停止：{e.url}，"
                                "将回退到 REST API 下载剩余数据，这可能需要更多时间。"
                            )
                        await cancel_and_await_tasks(tasks[tasks.index(task) + 1 :])
                        return results
                except Exception as e:
                    logger.warning(f"发生异常：{e}")
                    # 直接返回现有数据，不允许数据中存在间隙
                    await cancel_and_await_tasks(tasks[tasks.index(task) + 1 :])
                    return results
                else:
                    # 正常情况
                    results.extend(result)

    return results