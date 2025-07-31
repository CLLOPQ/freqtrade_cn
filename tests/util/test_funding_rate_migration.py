from shutil import copytree

from freqtrade.util.migrations import migrate_funding_fee_timeframe


def test_migrate_funding_rate_timeframe(default_conf_usdt, tmp_path, testdatadir):
    """测试资金费率时间框架迁移功能"""
    # 复制测试数据到临时目录
    copytree(testdatadir / "futures", tmp_path / "futures")
    
    # 定义测试文件路径
    file_4h = tmp_path / "futures" / "XRP_USDT_USDT-4h-funding_rate.feather"
    file_8h = tmp_path / "futures" / "XRP_USDT_USDT-8h-funding_rate.feather"
    file_1h = tmp_path / "futures" / "XRP_USDT_USDT-1h-futures.feather"
    
    # 将8小时资金费率文件重命名为4小时，模拟需要迁移的情况
    file_8h.rename(file_4h)
    
    # 验证初始文件状态
    assert file_1h.exists()  # 1小时期货数据文件应存在
    assert file_4h.exists()  # 重命名后的4小时资金费率文件应存在
    assert not file_8h.exists()  # 原始8小时文件应不存在

    # 设置数据目录
    default_conf_usdt["datadir"] = tmp_path

    # 在现货交易模式下迁移应不执行任何操作
    migrate_funding_fee_timeframe(default_conf_usdt, None)

    # 切换到期货交易模式
    default_conf_usdt["trading_mode"] = "futures"

    # 执行资金费率时间框架迁移
    migrate_funding_fee_timeframe(default_conf_usdt, None)

    # 验证迁移结果
    assert not file_4h.exists()  # 4小时文件应已被迁移
    assert file_8h.exists()  # 8小时文件应已创建
    assert file_1h.exists()  # 期货数据文件应保持不变