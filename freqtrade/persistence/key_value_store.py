from datetime import datetime, timezone
from enum import Enum
from typing import ClassVar, Literal

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from freqtrade.persistence.base import ModelBase, SessionType


ValueTypes = str | datetime | float | int


class ValueTypesEnum(str, Enum):
    STRING = "字符串"  # 枚举成员值翻译：字符串
    DATETIME = "日期时间"  # 枚举成员值翻译：日期时间
    FLOAT = "浮点数"  # 枚举成员值翻译：浮点数
    INT = "整数"  # 枚举成员值翻译：整数


KeyStoreKeys = Literal[
    "机器人启动时间",  # 键名翻译：机器人启动时间
    "启动时间",  # 键名翻译：启动时间
    "币安迁移",  # 键名翻译：币安迁移
]


class _KeyValueStoreModel(ModelBase):
    """
    交易对锁定数据库模型。  # 类文档字符串翻译
    """

    __tablename__ = "KeyValueStore"  # 表名保持不变
    session: ClassVar[SessionType]  # 类变量保持不变

    id: Mapped[int] = mapped_column(primary_key=True)  # 列名保持不变

    key: Mapped[KeyStoreKeys] = mapped_column(String(25), nullable=False, index=True)  # 列名保持不变

    value_type: Mapped[ValueTypesEnum] = mapped_column(String(20), nullable=False)  # 列名保持不变

    string_value: Mapped[str | None] = mapped_column(String(255), nullable=True)  # 列名保持不变
    datetime_value: Mapped[datetime | None]  # 列名保持不变
    float_value: Mapped[float | None]  # 列名保持不变
    int_value: Mapped[int | None]  # 列名保持不变


class KeyValueStore:
    """
    通用的全机器人持久键值存储  # 类文档字符串翻译
    可用于存储通用值，例如机器人首次启动时间。  # 类文档字符串翻译
    支持的类型包括字符串、日期时间、浮点数和整数。  # 类文档字符串翻译
    """

    @staticmethod
    def store_value(key: KeyStoreKeys, value: ValueTypes) -> None:
        """
        为给定的键存储给定的值。  # 方法文档字符串翻译
        :param key: 用于存储值的键 - 可在get-value中用于检索该键  # 参数文档翻译
        :param value: 要存储的值 - 可以是字符串、日期时间、浮点数或整数  # 参数文档翻译
        """
        kv = (
            _KeyValueStoreModel.session.query(_KeyValueStoreModel)
            .filter(_KeyValueStoreModel.key == key)
            .first()
        )
        if kv is None:
            kv = _KeyValueStoreModel(key=key)
        if isinstance(value, str):
            kv.value_type = ValueTypesEnum.STRING
            kv.string_value = value
        elif isinstance(value, datetime):
            kv.value_type = ValueTypesEnum.DATETIME
            kv.datetime_value = value
        elif isinstance(value, float):
            kv.value_type = ValueTypesEnum.FLOAT
            kv.float_value = value
        elif isinstance(value, int):
            kv.value_type = ValueTypesEnum.INT
            kv.int_value = value
        else:
            raise ValueError(f"未知的值类型 {kv.value_type}")
        _KeyValueStoreModel.session.add(kv)
        _KeyValueStoreModel.session.commit()

    @staticmethod
    def delete_value(key: KeyStoreKeys) -> None:
        """
        删除给定键的值。  # 方法文档字符串翻译
        :param key: 要删除值的键  # 参数文档翻译
        """
        kv = (
            _KeyValueStoreModel.session.query(_KeyValueStoreModel)
            .filter(_KeyValueStoreModel.key == key)
            .first()
        )
        if kv is not None:
            _KeyValueStoreModel.session.delete(kv)
            _KeyValueStoreModel.session.commit()

    @staticmethod
    def get_value(key: KeyStoreKeys) -> ValueTypes | None:
        """
        获取给定键的值。  # 方法文档字符串翻译
        :param key: 要获取值的键  # 参数文档翻译
        """
        kv = (
            _KeyValueStoreModel.session.query(_KeyValueStoreModel)
            .filter(_KeyValueStoreModel.key == key)
            .first()
        )
        if kv is None:
            return None
        if kv.value_type == ValueTypesEnum.STRING:
            return kv.string_value
        if kv.value_type == ValueTypesEnum.DATETIME and kv.datetime_value is not None:
            return kv.datetime_value.replace(tzinfo=timezone.utc)
        if kv.value_type == ValueTypesEnum.FLOAT:
            return kv.float_value
        if kv.value_type == ValueTypesEnum.INT:
            return kv.int_value
        # This should never happen unless someone messed with the database manually
        raise ValueError(f"未知的值类型 {kv.value_type}")  # pragma: no cover

    @staticmethod
    def get_string_value(key: KeyStoreKeys) -> str | None:
        """
        获取给定键的值。  # 方法文档字符串翻译
        :param key: 要获取值的键  # 参数文档翻译
        """
        kv = (
            _KeyValueStoreModel.session.query(_KeyValueStoreModel)
            .filter(
                _KeyValueStoreModel.key == key,
                _KeyValueStoreModel.value_type == ValueTypesEnum.STRING,
            )
            .first()
        )
        if kv is None:
            return None
        return kv.string_value

    @staticmethod
    def get_datetime_value(key: KeyStoreKeys) -> datetime | None:
        """
        获取给定键的值。  # 方法文档字符串翻译
        :param key: 要获取值的键  # 参数文档翻译
        """
        kv = (
            _KeyValueStoreModel.session.query(_KeyValueStoreModel)
            .filter(
                _KeyValueStoreModel.key == key,
                _KeyValueStoreModel.value_type == ValueTypesEnum.DATETIME,
            )
            .first()
        )
        if kv is None or kv.datetime_value is None:
            return None
        return kv.datetime_value.replace(tzinfo=timezone.utc)

    @staticmethod
    def get_float_value(key: KeyStoreKeys) -> float | None:
        """
        获取给定键的值。  # 方法文档字符串翻译
        :param key: 要获取值的键  # 参数文档翻译
        """
        kv = (
            _KeyValueStoreModel.session.query(_KeyValueStoreModel)
            .filter(
                _KeyValueStoreModel.key == key,
                _KeyValueStoreModel.value_type == ValueTypesEnum.FLOAT,
            )
            .first()
        )
        if kv is None:
            return None
        return kv.float_value

    @staticmethod
    def get_int_value(key: KeyStoreKeys) -> int | None:
        """
        获取给定键的值。  # 方法文档字符串翻译
        :param key: 要获取值的键  # 参数文档翻译
        """
        kv = (
            _KeyValueStoreModel.session.query(_KeyValueStoreModel)
            .filter(
                _KeyValueStoreModel.key == key, _KeyValueStoreModel.value_type == ValueTypesEnum.INT
            )
            .first()
        )
        if kv is None:
            return None
        return kv.int_value


def set_startup_time() -> None:
    """
    将bot_start_time设置为首次交易的开盘日期 - 或在新数据库上设置为“现在”。  # 函数文档字符串翻译
    将startup_time设置为“现在”  # 函数文档字符串翻译
    """
    st = KeyValueStore.get_value("bot_start_time")
    if st is None:
        from freqtrade.persistence import Trade

        t = Trade.session.query(Trade).order_by(Trade.open_date.asc()).first()
        if t is not None:
            KeyValueStore.store_value("bot_start_time", t.open_date_utc)
        else:
            KeyValueStore.store_value("bot_start_time", datetime.now(timezone.utc))
    KeyValueStore.store_value("startup_time", datetime.now(timezone.utc))