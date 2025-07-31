"""
此模块包含用于将交易持久化到SQLite的类
"""

import functools
import logging
import threading
from contextvars import ContextVar
from typing import Any, Final

from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import NoSuchModuleError
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.pool import StaticPool

from freqtrade.exceptions import OperationalException
from freqtrade.persistence.base import ModelBase
from freqtrade.persistence.custom_data import _CustomData
from freqtrade.persistence.key_value_store import _KeyValueStoreModel
from freqtrade.persistence.migrations import check_migrate
from freqtrade.persistence.pairlock import PairLock
from freqtrade.persistence.trade_model import Order, Trade


logger = logging.getLogger(__name__)


REQUEST_ID_CTX_KEY: Final[str] = "request_id"
_request_id_ctx_var: ContextVar[str | None] = ContextVar(REQUEST_ID_CTX_KEY, default=None)


def get_request_or_thread_id() -> str | None:
    """
    辅助方法，用于获取异步上下文（适用于FastAPI请求）或线程ID
    """
    request_id = _request_id_ctx_var.get()
    if request_id is None:
        # 当不在请求上下文中时 - 使用线程ID
        request_id = str(threading.current_thread().ident)

    return request_id


_SQL_DOCS_URL = "http://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls"


def init_db(db_url: str) -> None:
    """
    使用给定的配置初始化此模块，
    注册所有已知的命令处理程序
    并开始轮询消息更新
    :param db_url: 使用的数据库
    :return: None
    """
    kwargs: dict[str, Any] = {}

    if db_url == "sqlite:///":
        raise OperationalException(
            f"错误的数据库URL {db_url}。对于内存数据库，请使用 `sqlite://`。"
        )
    if db_url == "sqlite://":
        kwargs.update(
            {
                "poolclass": StaticPool,
            }
        )
    # 处理线程所有权
    if db_url.startswith("sqlite://"):
        kwargs.update(
            {
                "connect_args": {"check_same_thread": False},
            }
        )

    try:
        engine = create_engine(db_url, future=True, **kwargs)
    except NoSuchModuleError:
        raise OperationalException(
            f"给定的db_url值 '{db_url}' 不是有效的数据库URL！（参见 {_SQL_DOCS_URL}）"
        )

    # https://docs.sqlalchemy.org/en/13/orm/contextual.html#thread-local-scope
    # 作用域会话将请求代理到适当的线程本地会话。
    # 由于我们还使用FastAPI，也需要使其感知请求ID
    Trade.session = scoped_session(
        sessionmaker(bind=engine, autoflush=False), scopefunc=get_request_or_thread_id
    )
    Order.session = Trade.session
    PairLock.session = Trade.session
    _KeyValueStoreModel.session = Trade.session
    _CustomData.session = scoped_session(
        sessionmaker(bind=engine, autoflush=True), scopefunc=get_request_or_thread_id
    )

    previous_tables = inspect(engine).get_table_names()
    ModelBase.metadata.create_all(engine)
    check_migrate(engine, decl_base=ModelBase, previous_tables=previous_tables)


def custom_data_rpc_wrapper(func):
    """
    使用custom_data时的RPC方法包装器
    行为类似于deps.get_rpc() - 但仅限于custom_data。
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            _CustomData.session.rollback()
            return func(*args, **kwargs)
        finally:
            _CustomData.session.rollback()
            # 确保使用后移除会话
            _CustomData.session.remove()

    return wrapper