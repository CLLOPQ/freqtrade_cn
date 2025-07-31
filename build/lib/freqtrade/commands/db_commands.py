import logging
from typing import Any

from freqtrade.enums import RunMode


logger = logging.getLogger(__name__)


def start_convert_db(args: dict[str, Any]) -> None:
    """转换数据库（迁移数据）"""
    from sqlalchemy import func, select
    from sqlalchemy.orm import make_transient

    from freqtrade.configuration.config_setup import setup_utils_configuration
    from freqtrade.persistence import Order, Trade, init_db
    from freqtrade.persistence.migrations import set_sequence_ids
    from freqtrade.persistence.pairlock import PairLock

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    # 初始化目标数据库
    init_db(config["db_url"])
    session_target = Trade.session
    # 初始化源数据库
    init_db(config["db_url_from"])
    logger.info("开始数据库迁移。")

    trade_count = 0  # 交易计数
    pairlock_count = 0  # 交易对锁定计数
    
    # 迁移交易数据
    for trade in Trade.get_trades():
        trade_count += 1
        # 移除对象的持久化标识，使其成为临时对象
        make_transient(trade)
        # 处理该交易下的所有订单
        for o in trade.orders:
            make_transient(o)

        # 将交易添加到目标会话
        session_target.add(trade)

    # 提交交易数据
    session_target.commit()

    # 迁移交易对锁定数据
    for pairlock in PairLock.get_all_locks():
        pairlock_count += 1
        make_transient(pairlock)
        session_target.add(pairlock)
    # 提交交易对锁定数据
    session_target.commit()

    # 更新序列ID，确保自增ID正确
    max_trade_id = session_target.scalar(select(func.max(Trade.id)))
    max_order_id = session_target.scalar(select(func.max(Order.id)))
    max_pairlock_id = session_target.scalar(select(func.max(PairLock.id)))

    set_sequence_ids(
        session_target.get_bind(),
        trade_id=max_trade_id,
        order_id=max_order_id,
        pairlock_id=max_pairlock_id,
    )

    logger.info(f"已迁移 {trade_count} 笔交易和 {pairlock_count} 个交易对锁定记录。")