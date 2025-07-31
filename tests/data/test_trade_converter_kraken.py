from datetime import datetime, timezone
from shutil import copytree
from unittest.mock import PropertyMock

import pytest

from freqtrade.data.converter.trade_converter_kraken import import_kraken_trades_from_csv
from freqtrade.data.history import get_datahandler
from freqtrade.enums import TradingMode
from freqtrade.exceptionsceptions import OperationalException
from tests.conftest import EXMS, log_has, log_has_re, patch_exchange


def test_import_kraken_trades_from_csv(testdatadir, tmp_path, caplog, default_conf_usdt, mocker):
    # 测试非Kraken交易所时应抛出异常
    with pytest.raises(OperationalException, match="此函数仅适用于Kraken交易所"):
        import_kraken_trades_from_csv(default_conf_usdt, "feather")

    # 将交易所设置为Kraken
    default_conf_usdt["exchange"]["name"] = "kraken"

    # 模拟Kraken交易所
    patch_exchange(mocker, exchange="kraken")
    mocker.patch(
        f"{EXMS}.markets",
        PropertyMock(
            return_value={
                "BCH/EUR": {"symbol": "BCH/EUR", "id": "BCHEUR", "altname": "BCHEUR"},
            }
        ),
    )
    
    # 目标文件路径
    dstfile = tmp_path / "BCH_EUR-trades.feather"
    # 初始状态下目标文件不应存在
    assert not dstfile.is_file()
    
    # 设置数据目录
    default_conf_usdt["datadir"] = tmp_path
    
    # 测试数据说明：
    # 此目录树中有2个文件，共包含2天的数据
    # tests/testdata/kraken/
    # └── trades_csv
    #     ├── BCHEUR.csv       <-- 2023-01-01的数据
    #     └── incremental_q2
    #         └── BCHEUR.csv   <-- 2023-01-02的数据

    # 复制测试数据到临时目录
    copytree(testdatadir / "kraken/trades_csv", tmp_path / "trades_csv")

    # 执行Kraken交易数据导入
    import_kraken_trades_from_csv(default_conf_usdt, "feather")
    
    # 验证日志输出
    assert log_has("找到BCHEUR的csv文件。", caplog)
    assert log_has("正在转换交易对: BCH/EUR。", caplog)
    assert log_has_re(r"BCH/EUR: 340笔交易.* 2023-01-01.* 2023-01-02.*", caplog)

    # 验证目标文件已创建
    assert dstfile.is_file()

    # 加载并验证导入的数据
    dh = get_datahandler(tmp_path, "feather")
    trades = dh.trades_load("BCH_EUR", TradingMode.SPOT)
    # 验证交易数量
    assert len(trades) == 340

    # 验证时间范围
    assert trades["date"].min().to_pydatetime() == datetime(
        2023, 1, 1, 0, 3, 56, tzinfo=timezone.utc
    )
    assert trades["date"].max().to_pydatetime() == datetime(
        2023, 1, 2, 23, 17, 3, tzinfo=timezone.utc
    )
    # 验证ID列未填充
    assert len(trades.loc[trades["id"] != ""]) == 0

    # 测试过滤不存在的交易对
    caplog.clear()
    default_conf_usdt["pairs"] = ["XRP/EUR"]
    import_kraken_trades_from_csv(default_conf_usdt, "feather")
    # 应找到BCHEUR文件但过滤后无数据
    assert log_has("找到BCHEUR的csv文件。", caplog)
    assert log_has("未找到交易对XRP/EUR的数据。", caplog)