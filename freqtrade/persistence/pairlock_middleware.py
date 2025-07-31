import logging
from collections.abc import Sequence
from datetime import datetime, timezone

from sqlalchemy import select

from freqtrade.exchange import timeframe_to_next_date
from freqtrade.persistence.models import PairLock


logger = logging.getLogger(__name__)


class PairLocks:
    """
    交易对锁定中间件类
    将数据库层抽象出来，使其成为可选的——这对于未来支持回测和参数优化是必要的。
    """

    use_db = True
    locks: list[PairLock] = []

    timeframe: str = ""

    @staticmethod
    def reset_locks() -> None:
        """
        重置所有锁定。仅在回测模式下有效。
        """
        if not PairLocks.use_db:
            PairLocks.locks = []

    @staticmethod
    def lock_pair(
        pair: str,
        until: datetime,
        reason: str | None = None,
        *,
        now: datetime | None = None,
        side: str = "*",
    ) -> PairLock:
        """
        从现在到“until”创建交易对锁定。
        默认使用数据库，除非将PairLocks.use_db设为False，此时将维护一个列表。
        :param pair: 要锁定的交易对。使用 '*' 锁定所有交易对
        :param until: 锁定的结束时间。将向上舍入到下一个K线周期
        :param reason: 将作为锁定原因显示的原因字符串
        :param now: 当前时间戳。用于确定锁定开始时间
        :param side: 锁定交易对的方向，可以是 'long'（多）、'short'（空）或 '*'（所有方向）
        """
        lock = PairLock(
            pair=pair,
            lock_time=now or datetime.now(timezone.utc),
            lock_end_time=timeframe_to_next_date(PairLocks.timeframe, until),
            reason=reason,
            side=side,
            active=True,
        )
        if PairLocks.use_db:
            PairLock.session.add(lock)
            PairLock.session.commit()
        else:
            PairLocks.locks.append(lock)
        return lock

    @staticmethod
    def get_pair_locks(
        pair: str | None, now: datetime | None = None, side: str | None = None
    ) -> Sequence[PairLock]:
        """
        获取此交易对当前所有有效的锁定
        :param pair: 要检查的交易对。如果pair为空，则返回所有当前锁定
        :param now: 日期时间对象（通过datetime.now(timezone.utc)生成）。默认为datetime.now(timezone.utc)
        :param side: 要获取锁定的方向，可以是 'long'（多）、'short'（空）、'*'（所有方向）或None
        """
        if not now:
            now = datetime.now(timezone.utc)

        if PairLocks.use_db:
            return PairLock.query_pair_locks(pair, now, side).all()
        else:
            locks = [
                lock
                for lock in PairLocks.locks
                if (
                    lock.lock_end_time >= now
                    and lock.active is True
                    and (pair is None or lock.pair == pair)
                    and (side is None or lock.side == "*" or lock.side == side)
                )
            ]
            return locks

    @staticmethod
    def get_pair_longest_lock(
        pair: str, now: datetime | None = None, side: str = "*"
    ) -> PairLock | None:
        """
        获取给定交易对中过期时间最晚的锁定。
        """
        locks = PairLocks.get_pair_locks(pair, now, side=side)
        locks = sorted(locks, key=lambda lock: lock.lock_end_time, reverse=True)
        return locks[0] if locks else None

    @staticmethod
    def unlock_pair(pair: str, now: datetime | None = None, side: str = "*") -> None:
        """
        释放此交易对的所有锁定。
        :param pair: 要解锁的交易对
        :param now: 日期时间对象（通过datetime.now(timezone.utc)生成）。默认为datetime.now(timezone.utc)
        """
        if not now:
            now = datetime.now(timezone.utc)

        logger.info(f"释放 {pair} 的所有锁定。")
        locks = PairLocks.get_pair_locks(pair, now, side=side)
        for lock in locks:
            lock.active = False
        if PairLocks.use_db:
            PairLock.session.commit()

    @staticmethod
    def unlock_reason(reason: str, now: datetime | None = None) -> None:
        """
        释放具有此原因的所有锁定。
        :param reason: 要解锁的原因
        :param now: 日期时间对象（通过datetime.now(timezone.utc)生成）。默认为datetime.now(timezone.utc)
        """
        if not now:
            now = datetime.now(timezone.utc)

        if PairLocks.use_db:
            # 用于实盘模式
            logger.info(f"释放具有原因 '{reason}' 的所有锁定：")
            filters = [
                PairLock.lock_end_time > now,
                PairLock.active.is_(True),
                PairLock.reason == reason,
            ]
            locks = PairLock.session.scalars(select(PairLock).filter(*filters)).all()
            for lock in locks:
                logger.info(f"释放 {lock.pair} 的锁定，原因是 '{reason}'。")
                lock.active = False
            PairLock.session.commit()
        else:
            # 用于回测模式；为提高速度，不显示日志消息
            locksb = PairLocks.get_pair_locks(None)
            for lock in locksb:
                if lock.reason == reason:
                    lock.active = False

    @staticmethod
    def is_global_lock(now: datetime | None = None, side: str = "*") -> bool:
        """
        :param now: 日期时间对象（通过datetime.now(timezone.utc)生成）。默认为datetime.now(timezone.utc)
        """
        if not now:
            now = datetime.now(timezone.utc)

        return len(PairLocks.get_pair_locks("*", now, side)) > 0

    @staticmethod
    def is_pair_locked(pair: str, now: datetime | None = None, side: str = "*") -> bool:
        """
        :param pair: 要检查的交易对
        :param now: 日期时间对象（通过datetime.now(timezone.utc)生成）。默认为datetime.now(timezone.utc)
        """
        if not now:
            now = datetime.now(timezone.utc)

        return len(PairLocks.get_pair_locks(pair, now, side)) > 0 or PairLocks.is_global_lock(
            now, side
        )

    @staticmethod
    def get_all_locks() -> Sequence[PairLock]:
        """
        返回所有锁定，包括已过期的锁定
        """
        if PairLocks.use_db:
            return PairLock.get_all_locks().all()
        else:
            return PairLocks.locks