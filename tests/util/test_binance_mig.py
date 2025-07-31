import shutil

import pytest

from freqtrade.persistence import Trade
from freqtrade.util.migrations import migrate_binance_futures_data, migrate_data
from freqtrade.util.migrations.binance_mig import migrate_binance_futures_names
from tests.conftest import create_mock_trades_usdt, log_has


def test_binance_mig_data_conversion(default_conf_usdt, tmp_path, testdatadir):
    """测试币安合约数据迁移功能"""
    # 在现货模式下调用，应不执行任何操作
    migrate_binance_futures_data(default_conf_usdt)
    
    # 切换到合约模式
    default_conf_usdt["trading_mode"] = "futures"
    # 旧的交易对格式和新的统一格式
    pair_old = "XRP_USDT"
    pair_unified = "XRP_USDT_USDT"
    
    # 准备测试数据路径
    futures_src = testdatadir / "futures"
    futures_dst = tmp_path / "futures"
    futures_dst.mkdir()
    
    # 需要迁移的文件列表
    files = [
        "-1h-mark.feather",
        "-1h-futures.feather",
        "-8h-funding_rate.feather",
        "-8h-mark.feather",
    ]

    # 复制文件到临时目录并使用旧命名方式
    for file in files:
        fn_after = futures_dst / f"{pair_old}{file}"
        shutil.copy(futures_src / f"{pair_unified}{file}", fn_after)

    # 设置数据目录并执行迁移
    default_conf_usdt["datadir"] = tmp_path
    migrate_binance_futures_data(default_conf_usdt)

    # 验证所有文件是否已迁移到新的命名格式
    for file in files:
        fn_after = futures_dst / f"{pair_unified}{file}"
        assert fn_after.exists()


@pytest.mark.usefixtures("init_persistence")
def test_binance_mig_db_conversion(default_conf_usdt, fee, caplog):
    """测试币安合约数据库数据迁移功能"""
    # 在现货模式下调用，应不执行任何操作
    migrate_binance_futures_names(default_conf_usdt)

    # 创建模拟交易数据
    create_mock_trades_usdt(fee, None)

    # 将所有交易标记为币安合约交易
    for t in Trade.get_trades():
        t.trading_mode = "FUTURES"
        t.exchange = "binance"
    Trade.commit()

    # 切换到合约模式并执行数据库迁移
    default_conf_usdt["trading_mode"] = "futures"
    migrate_binance_futures_names(default_conf_usdt)
    
    # 验证迁移日志是否正确记录
    assert log_has("Migrating binance futures pairs in database.", caplog)


def test_migration_wrapper(default_conf_usdt, mocker):
    """测试迁移包装函数是否正确调用各个迁移函数"""
    # 设置为合约模式
    default_conf_usdt["trading_mode"] = "futures"
    
    # 模拟迁移函数
    binmock = mocker.patch("freqtrade.util.migrations.migrate_binance_futures_data")
    funding_mock = mocker.patch("freqtrade.util.migrations.migrate_funding_fee_timeframe")
    
    # 调用迁移包装函数
    migrate_data(default_conf_usdt)

    # 验证迁移函数是否被正确调用
    assert binmock.call_count == 1
    assert funding_mock.call_count == 1