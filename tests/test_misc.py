# pragma pylint: disable=missing-docstring,C0103

from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from freqtrade.misc import (
    dataframe_to_json,
    deep_merge_dicts,
    file_dump_json,
    file_load_json,
    is_file_in_dir,
    json_to_dataframe,
    pair_to_filename,
    parse_db_uri_for_logging,
    plural,
    safe_value_fallback,
    safe_value_fallback2,
)


def test_file_dump_json(mocker) -> None:
    """测试JSON文件写入功能"""
    # 测试普通JSON文件写入
    file_open = mocker.patch("freqtrade.misc.Path.open", MagicMock())
    json_dump = mocker.patch("rapidjson.dump", MagicMock())
    file_dump_json(Path("somefile"), [1, 2, 3])
    assert file_open.call_count == 1
    assert json_dump.call_count == 1
    
    # 测试压缩JSON文件写入
    file_open = mocker.patch("freqtrade.misc.gzip.open", MagicMock())
    json_dump = mocker.patch("rapidjson.dump", MagicMock())
    file_dump_json(Path("somefile"), [1, 2, 3], True)
    assert file_open.call_count == 1
    assert json_dump.call_count == 1


def test_file_load_json(mocker, testdatadir) -> None:
    """测试JSON文件读取功能"""
    # 测试不存在的.json文件
    ret = file_load_json(testdatadir / "UNITTEST_BTC-7m.json")
    assert not ret
    
    # 测试存在的.json文件（无对应的.gz文件）
    ret = file_load_json(testdatadir / "UNITTEST_BTC-1m.json")
    assert ret
    
    # 测试空的.json文件（会加载对应的.gz文件）
    ret = file_load_json(testdatadir / "UNITTEST_BTC-8m.json")
    assert ret


def test_is_file_in_dir(tmp_path):
    """测试文件是否在指定目录中"""
    # 创建临时目录和文件
    dir_path = tmp_path / "subdir"
    dir_path.mkdir()
    file_path = dir_path / "test.txt"
    file_path.touch()

    # 测试文件在目录中时返回True
    assert is_file_in_dir(file_path, dir_path) is True

    # 测试文件不在目录中时返回False
    assert is_file_in_dir(file_path, tmp_path) is False

    # 测试相对路径的情况
    file_path2 = tmp_path / "../../test2.txt"
    assert is_file_in_dir(file_path2, tmp_path) is False


@pytest.mark.parametrize(
    "pair,expected_result",
    [
        ("ETH/BTC", "ETH_BTC"),
        ("ETH/USDT", "ETH_USDT"),
        ("ETH/USDT:USDT", "ETH_USDT_USDT"),  # 以USDT为结算货币的swap
        ("ETH/USD:USD", "ETH_USD_USD"),  # 以USD为结算货币的swap
        ("AAVE/USD:USD", "AAVE_USD_USD"),  # 以USDT为结算货币的swap
        ("ETH/USDT:USDT-210625", "ETH_USDT_USDT-210625"),  # 到期期货
        ("Fabric Token/ETH", "Fabric_Token_ETH"),
        ("ETHH20", "ETHH20"),
        (".XBTBON2H", "_XBTBON2H"),
        ("ETHUSD.d", "ETHUSD_d"),
        ("ADA-0327", "ADA-0327"),
        ("BTC-USD-200110", "BTC-USD-200110"),
        ("BTC-PERP:USDT", "BTC-PERP_USDT"),
        ("F-AKRO/USDT", "F-AKRO_USDT"),
        ("LC+/ETH", "LC__ETH"),
        ("CMT@18/ETH", "CMT_18_ETH"),
        ("LBTC:1022/SAI", "LBTC_1022_SAI"),
        ("$PAC/BTC", "_PAC_BTC"),
        ("ACC_OLD/BTC", "ACC_OLD_BTC"),
    ],
)
def test_pair_to_filename(pair, expected_result):
    """测试交易对转换为文件名的功能"""
    pair_s = pair_to_filename(pair)
    assert pair_s == expected_result


def test_safe_value_fallback():
    """测试安全获取字典值并提供备选的功能"""
    dict1 = {"keya": None, "keyb": 2, "keyc": 5, "keyd": None}
    
    # 测试基本功能
    assert safe_value_fallback(dict1, "keya", "keyb") == 2
    assert safe_value_fallback(dict1, "keyb", "keya") == 2
    assert safe_value_fallback(dict1, "keyb", "keyc") == 2
    assert safe_value_fallback(dict1, "keya", "keyc") == 5
    assert safe_value_fallback(dict1, "keyc", "keyb") == 5
    assert safe_value_fallback(dict1, "keya", "keyd") is None
    
    # 测试默认值
    assert safe_value_fallback(dict1, "keyNo", "keyNo") is None
    assert safe_value_fallback(dict1, "keyNo", "keyNo", 55) == 55
    assert safe_value_fallback(dict1, "keyNo", default_value=55) == 55
    assert safe_value_fallback(dict1, "keyNo", None, default_value=55) == 55


def test_safe_value_fallback2():
    """测试从两个字典中安全获取值并提供备选的功能"""
    dict1 = {"keya": None, "keyb": 2, "keyc": 5, "keyd": None}
    dict2 = {"keya": 20, "keyb": None, "keyc": 6, "keyd": None}
    
    # 测试基本功能
    assert safe_value_fallback2(dict1, dict2, "keya", "keya") == 20
    assert safe_value_fallback2(dict2, dict1, "keya", "keya") == 20
    assert safe_value_fallback2(dict1, dict2, "keyb", "keyb") == 2
    assert safe_value_fallback2(dict2, dict1, "keyb", "keyb") == 2
    assert safe_value_fallback2(dict1, dict2, "keyc", "keyc") == 5
    assert safe_value_fallback2(dict2, dict1, "keyc", "keyc") == 6
    
    # 测试默认值
    assert safe_value_fallback2(dict1, dict2, "keyd", "keyd") is None
    assert safe_value_fallback2(dict2, dict1, "keyd", "keyd") is None
    assert safe_value_fallback2(dict2, dict1, "keyd", "keyd", 1234) == 1234
    assert safe_value_fallback2(dict1, dict2, "keyNo", "keyNo") is None
    assert safe_value_fallback2(dict2, dict1, "keyNo", "keyNo") is None
    assert safe_value_fallback2(dict2, dict1, "keyNo", "keyNo", 1234) == 1234


def test_plural() -> None:
    """测试复数形式转换功能"""
    # 测试默认复数形式（加s）
    assert plural(0, "page") == "pages"
    assert plural(0.0, "page") == "pages"
    assert plural(1, "page") == "page"
    assert plural(1.0, "page") == "page"
    assert plural(2, "page") == "pages"
    assert plural(2.0, "page") == "pages"
    assert plural(-1, "page") == "page"
    assert plural(-1.0, "page") == "page"
    assert plural(-2, "page") == "pages"
    assert plural(-2.0, "page") == "pages"
    assert plural(0.5, "page") == "pages"
    assert plural(1.5, "page") == "pages"
    assert plural(-0.5, "page") == "pages"
    assert plural(-1.5, "page") == "pages"

    # 测试自定义复数形式
    assert plural(0, "ox", "oxen") == "oxen"
    assert plural(0.0, "ox", "oxen") == "oxen"
    assert plural(1, "ox", "oxen") == "ox"
    assert plural(1.0, "ox", "oxen") == "ox"
    assert plural(2, "ox", "oxen") == "oxen"
    assert plural(2.0, "ox", "oxen") == "oxen"
    assert plural(-1, "ox", "oxen") == "ox"
    assert plural(-1.0, "ox", "oxen") == "ox"
    assert plural(-2, "ox", "oxen") == "oxen"
    assert plural(-2.0, "ox", "oxen") == "oxen"
    assert plural(0.5, "ox", "oxen") == "oxen"
    assert plural(1.5, "ox", "oxen") == "oxen"
    assert plural(-0.5, "ox", "oxen") == "oxen"
    assert plural(-1.5, "ox", "oxen") == "oxen"


@pytest.mark.parametrize(
    "conn_url,expected",
    [
        (
            "postgresql+psycopg2://scott123:scott123@host:1245/dbname",
            "postgresql+psycopg2://scott123:*****@host:1245/dbname",
        ),
        (
            "postgresql+psycopg2://scott123:scott123@host.name.com/dbname",
            "postgresql+psycopg2://scott123:*****@host.name.com/dbname",
        ),
        (
            "mariadb+mariadbconnector://app_user:Password123!@127.0.0.1:3306/company",
            "mariadb+mariadbconnector://app_user:*****@127.0.0.1:3306/company",
        ),
        (
            "mysql+pymysql://user:pass@some_mariadb/dbname?charset=utf8mb4",
            "mysql+pymysql://user:*****@some_mariadb/dbname?charset=utf8mb4",
        ),
        (
            "sqlite:////freqtrade/user_data/tradesv3.sqlite",
            "sqlite:////freqtrade/user_data/tradesv3.sqlite",
        ),
    ],
)
def test_parse_db_uri_for_logging(conn_url, expected) -> None:
    """测试数据库连接URI的日志格式化（隐藏密码）"""
    assert parse_db_uri_for_logging(conn_url) == expected


def test_deep_merge_dicts():
    """测试字典深度合并功能"""
    a = {"first": {"rows": {"pass": "dog", "number": "1", "test": None}}}
    b = {"first": {"rows": {"fail": "cat", "number": "5", "test": "asdf"}}}
    res = {"first": {"rows": {"pass": "dog", "fail": "cat", "number": "5", "test": "asdf"}}}
    res2 = {"first": {"rows": {"pass": "dog", "fail": "cat", "number": "1", "test": None}}}
    
    # 测试b合并到a
    assert deep_merge_dicts(b, deepcopy(a)) == res

    # 测试a合并到b
    assert deep_merge_dicts(a, deepcopy(b)) == res2

    # 测试不允许null覆盖
    res2["first"]["rows"]["test"] = "asdf"
    assert deep_merge_dicts(a, deepcopy(b), allow_null_overrides=False) == res2


def test_dataframe_json(ohlcv_history):
    """测试DataFrame与JSON之间的转换"""
    from pandas.testing import assert_frame_equal

    # 转换为JSON再转换回DataFrame
    json = dataframe_to_json(ohlcv_history)
    dataframe = json_to_dataframe(json)

    # 验证转换后的数据与原始数据一致
    assert list(ohlcv_history.columns) == list(dataframe.columns)
    assert len(ohlcv_history) == len(dataframe)
    assert_frame_equal(ohlcv_history, dataframe)
    
    # 测试包含NaT值的情况
    ohlcv_history.at[1, "date"] = pd.NaT
    json = dataframe_to_json(ohlcv_history)
    dataframe = json_to_dataframe(json)