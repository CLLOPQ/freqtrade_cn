import json
import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any, ClassVar

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, select
from sqlalchemy.orm import Mapped, mapped_column, relationship

from freqtrade.constants import DATETIME_PRINT_FORMAT
from freqtrade.persistence.base import ModelBase, SessionType
from freqtrade.util import dt_now


logger = logging.getLogger(__name__)


class _CustomData(ModelBase):
    """
    自定义数据数据库模型
    作为键值存储记录交易或全局持久化值的元数据
    与交易(Trades)存在一对多关系：
      - 一个交易可以有多个元数据条目
      - 一个元数据条目只能与一个交易关联
    """

    __tablename__ = "trade_custom_data"
    __allow_unmapped__ = True
    session: ClassVar[SessionType]

    # 唯一性应通过交易对(pair)、订单ID(order_id)来确保，在某些交易所，订单ID可能对每个交易对是唯一的。
    __table_args__ = (UniqueConstraint("ft_trade_id", "cd_key", name="_trade_id_cd_key"),)

    id = mapped_column(Integer, primary_key=True)
    ft_trade_id = mapped_column(Integer, ForeignKey("trades.id"), index=True)

    trade = relationship("Trade", back_populates="custom_data")

    cd_key: Mapped[str] = mapped_column(String(255), nullable=False)
    cd_type: Mapped[str] = mapped_column(String(25), nullable=False)
    cd_value: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=dt_now)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # 空容器值 - 不持久化，但在查询时填充cd_value
    value: Any = None

    def __repr__(self):
        create_time = (
            self.created_at.strftime(DATETIME_PRINT_FORMAT) if self.created_at is not None else None
        )
        update_time = (
            self.updated_at.strftime(DATETIME_PRINT_FORMAT) if self.updated_at is not None else None
        )
        return (
            f"自定义数据(id={self.id}, key={self.cd_key}, type={self.cd_type}, "
            + f"value={self.cd_value}, trade_id={self.ft_trade_id}, created={create_time}, "
            + f"updated={update_time})"
        )

    @classmethod
    def query_cd(
        cls, key: str | None = None, trade_id: int | None = None
    ) -> Sequence["_CustomData"]:
        """
        获取所有自定义数据，如果未指定交易ID，则返回与交易无关的通用值
        :param trade_id: 交易ID
        """
        filters = []
        if trade_id is not None:
            filters.append(_CustomData.ft_trade_id == trade_id)
        if key is not None:
            filters.append(_CustomData.cd_key.ilike(key))

        return _CustomData.session.scalars(select(_CustomData).filter(*filters)).all()


class CustomDataWrapper:
    """
    自定义数据中间件类
    抽象数据库层使其成为可选 - 这对于未来支持回测和参数优化是必要的。
    """

    use_db = True
    custom_data: list[_CustomData] = []
    unserialized_types = ["bool", "float", "int", "str"]

    @staticmethod
    def _convert_custom_data(data: _CustomData) -> _CustomData:
        if data.cd_type in CustomDataWrapper.unserialized_types:
            data.value = data.cd_value
            if data.cd_type == "bool":
                data.value = data.cd_value.lower() == "true"
            elif data.cd_type == "int":
                data.value = int(data.cd_value)
            elif data.cd_type == "float":
                data.value = float(data.cd_value)
        else:
            data.value = json.loads(data.cd_value)
        return data

    @staticmethod
    def reset_custom_data() -> None:
        """
        重置所有键值对。仅在回测模式下有效。
        """
        if not CustomDataWrapper.use_db:
            CustomDataWrapper.custom_data = []

    @staticmethod
    def delete_custom_data(trade_id: int) -> None:
        _CustomData.session.query(_CustomData).filter(_CustomData.ft_trade_id == trade_id).delete()
        _CustomData.session.commit()

    @staticmethod
    def get_custom_data(*, trade_id: int, key: str | None = None) -> list[_CustomData]:
        """
        获取特定交易和/或键的自定义数据条目
        """
        if CustomDataWrapper.use_db:
            filters = [
                _CustomData.ft_trade_id == trade_id,
            ]
            if key is not None:
                filters.append(_CustomData.cd_key.ilike(key))
            filtered_custom_data = _CustomData.session.scalars(
                select(_CustomData).filter(*filters)
            ).all()

        else:
            filtered_custom_data = [
                data_entry
                for data_entry in CustomDataWrapper.custom_data
                if (data_entry.ft_trade_id == trade_id)
            ]
            if key is not None:
                filtered_custom_data = [
                    data_entry
                    for data_entry in filtered_custom_data
                    if (data_entry.cd_key.casefold() == key.casefold())
                ]
        return [CustomDataWrapper._convert_custom_data(d) for d in filtered_custom_data]

    @staticmethod
    def set_custom_data(trade_id: int, key: str, value: Any) -> None:
        value_type = type(value).__name__

        if value_type not in CustomDataWrapper.unserialized_types:
            try:
                value_db = json.dumps(value)
            except TypeError as e:
                logger.warning(f"无法序列化{key}的值，原因是{e}")
                return
        else:
            value_db = str(value)

        if trade_id is None:
            trade_id = 0

        custom_data = CustomDataWrapper.get_custom_data(trade_id=trade_id, key=key)
        if custom_data:
            data_entry = custom_data[0]
            data_entry.cd_value = value_db
            data_entry.updated_at = dt_now()
        else:
            data_entry = _CustomData(
                ft_trade_id=trade_id,
                cd_key=key,
                cd_type=value_type,
                cd_value=value_db,
                created_at=dt_now(),
            )
        data_entry.value = value

        if CustomDataWrapper.use_db and value_db is not None:
            _CustomData.session.add(data_entry)
            _CustomData.session.commit()
        else:
            if not custom_data:
                CustomDataWrapper.custom_data.append(data_entry)
            # Existing data will have updated interactively.