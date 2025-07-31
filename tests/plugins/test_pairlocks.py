from datetime import datetime, timedelta, timezone

import pytest

from freqtrade.persistence import PairLocks
from freqtrade.persistence.models import PairLock
from freqtrade.util import dt_now


@pytest.mark.parametrize("use_db", (False, True))
@pytest.mark.usefixtures("init_persistence")
def test_PairLocks(use_db):
    PairLocks.timeframe = "5m"
    PairLocks.use_db = use_db
    # 不应该有锁存在
    if use_db:
        assert len(PairLock.get_all_locks().all()) == 0

    assert PairLocks.use_db == use_db

    pair = "ETH/BTC"
    assert not PairLocks.is_pair_locked(pair)
    PairLocks.lock_pair(pair, dt_now() + timedelta(minutes=4))
    # ETH/BTC 被锁定4分钟（多空双向）
    assert PairLocks.is_pair_locked(pair)
    assert PairLocks.is_pair_locked(pair, side="long")
    assert PairLocks.is_pair_locked(pair, side="short")

    pair = "BNB/BTC"
    PairLocks.lock_pair(pair, dt_now() + timedelta(minutes=4), side="long")
    assert not PairLocks.is_pair_locked(pair)
    assert PairLocks.is_pair_locked(pair, side="long")
    assert not PairLocks.is_pair_locked(pair, side="short")

    pair = "BNB/USDT"
    PairLocks.lock_pair(pair, dt_now() + timedelta(minutes=4), side="short")
    assert not PairLocks.is_pair_locked(pair)
    assert not PairLocks.is_pair_locked(pair, side="long")
    assert PairLocks.is_pair_locked(pair, side="short")
    assert len(PairLocks.get_pair_locks(pair)) == 1

    # XRP/BTC 现在不应该被锁定
    pair = "XRP/BTC"
    assert not PairLocks.is_pair_locked(pair)
    # 解锁一个未锁定的交易对不应该引发错误
    PairLocks.unlock_pair(pair)

    PairLocks.lock_pair(pair, dt_now() + timedelta(minutes=4))
    assert PairLocks.is_pair_locked(pair)

    # 获取上面所有的锁
    locks = PairLocks.get_pair_locks(None)
    assert len(locks) == 4

    assert len(PairLocks.get_pair_locks(None, side="*")) == 2

    # 解锁原始交易对
    pair = "ETH/BTC"
    PairLocks.unlock_pair(pair)
    assert not PairLocks.is_pair_locked(pair)
    assert not PairLocks.is_global_lock()

    pair = "BTC/USDT"
    # 锁定到14:30
    lock_time = datetime(2020, 5, 1, 14, 30, 0, tzinfo=timezone.utc)
    PairLocks.lock_pair(pair, lock_time)

    assert not PairLocks.is_pair_locked(pair)
    assert PairLocks.is_pair_locked(pair, lock_time + timedelta(minutes=-10))
    assert not PairLocks.is_global_lock(lock_time + timedelta(minutes=-10))
    assert PairLocks.is_pair_locked(pair, lock_time + timedelta(minutes=-50))
    assert not PairLocks.is_global_lock(lock_time + timedelta(minutes=-50))

    # 时间过期后不应该被锁定
    assert not PairLocks.is_pair_locked(pair, lock_time + timedelta(minutes=10))

    locks = PairLocks.get_pair_locks(pair, lock_time + timedelta(minutes=-2))
    assert len(locks) == 1
    assert "PairLock" in str(locks[0])

    # 解锁所有
    PairLocks.unlock_pair(pair, lock_time + timedelta(minutes=-2))
    assert not PairLocks.is_global_lock(lock_time + timedelta(minutes=-50))

    # 全局锁定
    PairLocks.lock_pair("*", lock_time)
    assert PairLocks.is_global_lock(lock_time + timedelta(minutes=-50))
    # 全局锁定也会分别锁定每个交易对
    assert PairLocks.is_pair_locked(pair, lock_time + timedelta(minutes=-50))
    assert PairLocks.is_pair_locked("XRP/USDT", lock_time + timedelta(minutes=-50))

    if use_db:
        locks = PairLocks.get_all_locks()
        locks_db = PairLock.get_all_locks().all()
        assert len(locks) == len(locks_db)
        assert len(locks_db) > 0
    else:
        # 没有数据被推送到数据库
        assert len(PairLocks.get_all_locks()) > 0
        assert len(PairLock.get_all_locks().all()) == 0
    # 重置use-db变量
    PairLocks.reset_locks()
    PairLocks.use_db = True


@pytest.mark.parametrize("use_db", (False, True))
@pytest.mark.usefixtures("init_persistence")
def test_PairLocks_getlongestlock(use_db):
    PairLocks.timeframe = "5m"
    # 不应该有锁存在
    PairLocks.use_db = use_db
    if use_db:
        assert len(PairLock.get_all_locks().all()) == 0

    assert PairLocks.use_db == use_db

    pair = "ETH/BTC"
    assert not PairLocks.is_pair_locked(pair)
    PairLocks.lock_pair(pair, dt_now() + timedelta(minutes=4))
    # ETH/BTC 被锁定4分钟
    assert PairLocks.is_pair_locked(pair)
    lock = PairLocks.get_pair_longest_lock(pair)

    assert lock.lock_end_time.replace(tzinfo=timezone.utc) > dt_now() + timedelta(minutes=3)
    assert lock.lock_end_time.replace(tzinfo=timezone.utc) < dt_now() + timedelta(minutes=14)

    PairLocks.lock_pair(pair, dt_now() + timedelta(minutes=15))
    assert PairLocks.is_pair_locked(pair)

    lock = PairLocks.get_pair_longest_lock(pair)
    # 必须比上面的锁定时间更长
    assert lock.lock_end_time.replace(tzinfo=timezone.utc) > dt_now() + timedelta(minutes=14)

    PairLocks.reset_locks()
    PairLocks.use_db = True


@pytest.mark.parametrize("use_db", (False, True))
@pytest.mark.usefixtures("init_persistence")
def test_PairLocks_reason(use_db):
    PairLocks.timeframe = "5m"
    PairLocks.use_db = use_db
    # 不应该有锁存在
    if use_db:
        assert len(PairLock.get_all_locks().all()) == 0

    assert PairLocks.use_db == use_db

    PairLocks.lock_pair("XRP/USDT", dt_now() + timedelta(minutes=4), "TestLock1")
    PairLocks.lock_pair("ETH/USDT", dt_now() + timedelta(minutes=4), "TestLock2")

    assert PairLocks.is_pair_locked("XRP/USDT")
    assert PairLocks.is_pair_locked("ETH/USDT")

    PairLocks.unlock_reason("TestLock1")
    assert not PairLocks.is_pair_locked("XRP/USDT")
    assert PairLocks.is_pair_locked("ETH/USDT")

    PairLocks.reset_locks()
    PairLocks.use_db = True