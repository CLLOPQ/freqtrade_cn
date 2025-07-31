from datetime import datetime, timezone
from typing import Any, ClassVar

from sqlalchemy import ScalarResult, String, or_, select
from sqlalchemy.orm import Mapped, mapped_column

from freqtrade.constants import DATETIME_PRINT_FORMAT
from freqtrade.persistence.base import ModelBase, SessionType


class PairLock(ModelBase):
    """
    交易对锁定数据库模型。
    """

    __tablename__ = "pairlocks"
    session: ClassVar[SessionType]

    id: Mapped[int] = mapped_column(primary_key=True)

    pair: Mapped[str] = mapped_column(String(25), nullable=False, index=True)
    # 锁定方向 - 多仓、空仓或 *（两者都锁定）
    side: Mapped[str] = mapped_column(String(25), nullable=False, default="*")
    reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # 锁定交易对的时间（开始时间）
    lock_time: Mapped[datetime] = mapped_column(nullable=False)
    # 锁定交易对的时间（结束时间）
    lock_end_time: Mapped[datetime] = mapped_column(nullable=False, index=True)

    active: Mapped[bool] = mapped_column(nullable=False, default=True, index=True)

    def __repr__(self) -> str:
        lock_time = self.lock_time.strftime(DATETIME_PRINT_FORMAT)
        lock_end_time = self.lock_end_time.strftime(DATETIME_PRINT_FORMAT)
        return (
            f"交易对锁定(id={self.id}, 交易对={self.pair}, 方向={self.side}, 锁定时间={lock_time}, "
            f"锁定结束时间={lock_end_time}, 原因={self.reason}, 激活状态={self.active})"
        )

    @staticmethod
    def query_pair_locks(
        pair: str | None, now: datetime, side: str | None = None
    ) -> ScalarResult["PairLock"]:
        """
        获取此交易对当前所有激活的锁定
        :param pair: 要检查的交易对。如果pair为空，则返回所有当前锁定
        :param now: 日期时间对象（通过datetime.now(timezone.utc)生成）。
        """
        filters = [
            PairLock.lock_end_time > now,
            # 仅激活的锁定
            PairLock.active.is_(True),
        ]
        if pair:
            filters.append(PairLock.pair == pair)
        if side is not None and side != "*":
            filters.append(or_(PairLock.side == side, PairLock.side == "*"))
        elif side is not None:
            filters.append(PairLock.side == "*")

        return PairLock.session.scalars(select(PairLock).filter(*filters))

    @staticmethod
    def get_all_locks() -> ScalarResult["PairLock"]:
        """获取所有锁定"""
        return PairLock.session.scalars(select(PairLock))

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "交易对": self.pair,
            "锁定时间": self.lock_time.strftime(DATETIME_PRINT_FORMAT),
            "锁定时间戳": int(self.lock_time.replace(tzinfo=timezone.utc).timestamp() * 1000),
            "锁定结束时间": self.lock_end_time.strftime(DATETIME_PRINT_FORMAT),
            "锁定结束时间戳": int(
                self.lock_end_time.replace(tzinfo=timezone.utc).timestamp() * 1000
            ),
            "原因": self.reason,
            "方向": self.side,
            "激活状态": self.active,
        }