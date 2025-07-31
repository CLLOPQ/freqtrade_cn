import asyncio
import logging
import time
from copy import deepcopy
from functools import partial
from threading import Thread

import ccxt

from freqtrade.constants import Config, PairWithTimeframe
from freqtrade.enums.candletype import CandleType
from freqtrade.exceptions import TemporaryError
from freqtrade.exchange.common import retrier
from freqtrade.exchange.exchange import timeframe_to_seconds
from freqtrade.exchange.exchange_types import OHLCVResponse
from freqtrade.util import dt_ts, format_ms_time, format_ms_time_det


logger = logging.getLogger(__name__)


class ExchangeWS:
    def __init__(self, config: Config, ccxt_object: ccxt.Exchange) -> None:
        self.config = config
        self._ccxt_object = ccxt_object
        self._background_tasks: set[asyncio.Task] = set()

        self._klines_watching: set[PairWithTimeframe] = set()
        self._klines_scheduled: set[PairWithTimeframe] = set()
        self.klines_last_refresh: dict[PairWithTimeframe, float] = {}
        self.klines_last_request: dict[PairWithTimeframe, float] = {}
        self._thread = Thread(name="ccxt_ws", target=self._start_forever)
        self._thread.start()
        self.__cleanup_called = False

    def _start_forever(self) -> None:
        self._loop = asyncio.new_event_loop()
        try:
            self._loop.run_forever()
        finally:
            if self._loop.is_running():
                self._loop.stop()

    def cleanup(self) -> None:
        logger.debug("调用清理 - 正在停止")
        self._klines_watching.clear()
        for task in self._background_tasks:
            task.cancel()
        if hasattr(self, "_loop") and not self._loop.is_closed():
            self.reset_connections()

            self._loop.call_soon_threadsafe(self._loop.stop)
            time.sleep(0.1)
            if not self._loop.is_closed():
                self._loop.close()

        self._thread.join()
        logger.debug("已停止")

    def reset_connections(self) -> None:
        """
        重置所有连接 - 避免约9天后发生的"连接重置"错误
        """
        if hasattr(self, "_loop") and not self._loop.is_closed():
            logger.info("正在重置WS连接。")
            asyncio.run_coroutine_threadsafe(self._cleanup_async(), loop=self._loop)
            while not self.__cleanup_called:
                time.sleep(0.1)
        self.__cleanup_called = False

    async def _cleanup_async(self) -> None:
        try:
            await self._ccxt_object.close()
            # 清除缓存
            # 不这样做会在动态交易对列表启动时导致问题
            self._ccxt_object.ohlcvs.clear()
        except Exception:
            logger.exception("_cleanup_async 中的异常")
        finally:
            self.__cleanup_called = True

    def _pop_history(self, paircomb: PairWithTimeframe) -> None:
        """
        从ccxt缓存中移除交易对/时间框架组合的历史
        """
        self._ccxt_object.ohlcvs.get(paircomb[0], {}).pop(paircomb[1], None)
        self.klines_last_refresh.pop(paircomb, None)

    @retrier(retries=3)
    def ohlcvs(self, pair: str, timeframe: str) -> list[list]:
        """
        返回交易对/时间框架组合的K线副本
        注意：这将只包含从WebSocket接收的数据
            因此数据将随着时间的推移而积累
        """
        try:
            return deepcopy(self._ccxt_object.ohlcvs.get(pair, {}).get(timeframe, []))
        except RuntimeError as e:
            # 捕获运行时错误并重试
            # TemporaryError不会导致退避 - 所以我们实际上是立即重试
            raise TemporaryError(f"深拷贝错误: {e}") from e

    def cleanup_expired(self) -> None:
        """
        如果在最后一个时间框架(+偏移量)内没有请求过交易对，则从监视列表中移除
        """
        changed = False
        for p in list(self._klines_watching):
            _, timeframe, _ = p
            timeframe_s = timeframe_to_seconds(timeframe)
            last_refresh = self.klines_last_request.get(p, 0)
            if last_refresh > 0 and (dt_ts() - last_refresh) > ((timeframe_s + 20) * 1000):
                logger.info(f"从WebSocket监视列表中移除 {p}。")
                self._klines_watching.discard(p)
                # 移除历史以避免获取过时数据
                self._pop_history(p)
                changed = True
        if changed:
            logger.info(f"移除完成: 新的监视列表 ({len(self._klines_watching)})")

    async def _schedule_while_true(self) -> None:
        # 对于我们应该监视的交易对
        for p in self._klines_watching:
            # 检查它们是否已被调度
            if p not in self._klines_scheduled:
                self._klines_scheduled.add(p)
                pair, timeframe, candle_type = p
                task = asyncio.create_task(
                    self._continuously_async_watch_ohlcv(pair, timeframe, candle_type)
                )
                self._background_tasks.add(task)
                task.add_done_callback(
                    partial(
                        self._continuous_stopped,
                        pair=pair,
                        timeframe=timeframe,
                        candle_type=candle_type,
                    )
                )

    async def _unwatch_ohlcv(self, pair: str, timeframe: str, candle_type: CandleType) -> None:
        try:
            await self._ccxt_object.un_watch_ohlcv_for_symbols([[pair, timeframe]])
        except ccxt.NotSupported as e:
            logger.debug("un_watch_ohlcv_for_symbols 不支持: %s", e)
            pass
        except Exception:
            logger.exception("_unwatch_ohlcv 中的异常")

    def _continuous_stopped(
        self, task: asyncio.Task, pair: str, timeframe: str, candle_type: CandleType
    ):
        self._background_tasks.discard(task)
        result = "完成"
        if task.cancelled():
            result = "已取消"
        else:
            if (result1 := task.result()) is not None:
                result = str(result1)

        logger.info(f"{pair}, {timeframe}, {candle_type} - 任务完成 - {result}")
        asyncio.run_coroutine_threadsafe(
            self._unwatch_ohlcv(pair, timeframe, candle_type), loop=self._loop
        )

        self._klines_scheduled.discard((pair, timeframe, candle_type))
        self._pop_history((pair, timeframe, candle_type))

    async def _continuously_async_watch_ohlcv(
        self, pair: str, timeframe: str, candle_type: CandleType
    ) -> None:
        try:
            while (pair, timeframe, candle_type) in self._klines_watching:
                start = dt_ts()
                data = await self._ccxt_object.watch_ohlcv(pair, timeframe)
                self.klines_last_refresh[(pair, timeframe, candle_type)] = dt_ts()
                logger.debug(
                    f"监视完成 {pair}, {timeframe}, 数据 {len(data)} "
                    f"耗时 {(dt_ts() - start) / 1000:.3f}秒"
                )
        except ccxt.ExchangeClosedByUser:
            logger.debug("交易所连接被用户关闭")
        except ccxt.BaseError:
            logger.exception(f"continuously_async_watch_ohlcv 中 {pair}, {timeframe} 的异常")
        finally:
            self._klines_watching.discard((pair, timeframe, candle_type))

    def schedule_ohlcv(self, pair: str, timeframe: str, candle_type: CandleType) -> None:
        """
        调度要监视的交易对/时间框架组合
        """
        self._klines_watching.add((pair, timeframe, candle_type))
        self.klines_last_request[(pair, timeframe, candle_type)] = dt_ts()
        # asyncio.run_coroutine_threadsafe(self.schedule_schedule(), loop=self._loop)
        asyncio.run_coroutine_threadsafe(self._schedule_while_true(), loop=self._loop)
        self.cleanup_expired()

    async def get_ohlcv(
        self,
        pair: str,
        timeframe: str,
        candle_type: CandleType,
        candle_ts: int,
    ) -> OHLCVResponse:
        """
        从ccxt的"watch"缓存返回缓存的K线。
        :param candle_ts: 我们期望的蜡烛结束时间的时间戳。
        """
        # 深拷贝响应 - 因为随着新消息的到达，它可能在后台被修改
        candles = self.ohlcvs(pair, timeframe)
        refresh_date = self.klines_last_refresh[(pair, timeframe, candle_type)]
        received_ts = candles[-1][0] if candles else 0
        drop_hint = received_ts >= candle_ts
        if received_ts > refresh_date:
            logger.warning(
                f"{pair}, {timeframe} - 蜡烛日期 > 最后刷新 "
                f"({format_ms_time(received_ts)} > {format_ms_time_det(refresh_date)}). "
                "这通常表明时间同步有问题。"
            )
        logger.debug(
            f"{pair}, {timeframe} 的监视结果，长度 {len(candles)}, "
            f"r_ts={format_ms_time(received_ts)}, "
            f"lref={format_ms_time_det(refresh_date)}, "
            f"candle_ts={format_ms_time(candle_ts)}, {drop_hint=}"
        )
        return pair, timeframe, candle_type, candles, drop_hint