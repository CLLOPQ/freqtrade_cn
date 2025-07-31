# pragma pylint: disable=missing-docstring, protected-access, invalid-name
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from freqtrade.configuration.directory_operations import (
    chown_user_directory,
    copy_sample_files,
    create_datadir,
    create_userdata_dir,
)
from freqtrade.exceptions import OperationalException
from tests.conftest import log_has, log_has_re


def test_create_datadir(mocker, default_conf, caplog) -> None:
    """测试创建数据目录功能"""
    # 模拟目录不存在
    mocker.patch.object(Path, "is_dir", MagicMock(return_value=False))
    # 模拟创建目录的方法
    md = mocker.patch.object(Path, "mkdir", MagicMock())

    # 调用创建数据目录的函数
    create_datadir(default_conf, "/foo/bar")
    # 验证是否创建了目录，且是否创建了父目录
    assert md.call_args[1]["parents"] is True
    # 验证是否有正确的日志输出
    assert log_has("已创建数据目录: /foo/bar", caplog)


def test_create_userdata_dir(mocker, tmp_path, caplog) -> None:
    """测试创建用户数据目录功能"""
    # 模拟目录不存在
    mocker.patch.object(Path, "is_dir", MagicMock(return_value=False))
    # 模拟创建目录的方法
    md = mocker.patch.object(Path, "mkdir", MagicMock())

    # 调用创建用户数据目录的函数
    x = create_userdata_dir(tmp_path / "bar", create_dir=True)
    # 验证是否创建了所有必要的子目录
    assert md.call_count == 10
    # 验证是否不创建父目录
    assert md.call_args[1]["parents"] is False
    # 验证是否有正确的日志输出
    assert log_has(f"已创建用户数据目录: {tmp_path / 'bar'}", caplog)
    # 验证返回值是否为Path对象且路径正确
    assert isinstance(x, Path)
    assert str(x) == str(tmp_path / "bar")


def test_create_userdata_dir_and_chown(mocker, tmp_path, caplog) -> None:
    """测试创建用户数据目录并更改所有者功能"""
    # 模拟子进程调用
    sp_mock = mocker.patch("subprocess.check_output")
    path = tmp_path / "bar"
    # 验证目录初始不存在
    assert not path.is_dir()

    # 创建用户数据目录
    x = create_userdata_dir(str(path), create_dir=True)
    # 验证初始没有调用子进程(没有设置环境变量)
    assert sp_mock.call_count == 0
    # 验证日志和返回值
    assert log_has(f"已创建用户数据目录: {path}", caplog)
    assert isinstance(x, Path)
    # 验证目录已创建
    assert path.is_dir()
    assert (path / "data").is_dir()

    # 设置Docker环境变量，测试更改所有者
    os.environ["FT_APP_ENV"] = "docker"
    chown_user_directory(path / "data")
    # 验证子进程被调用了一次
    assert sp_mock.call_count == 1
    # 清理环境变量
    del os.environ["FT_APP_ENV"]


def test_create_userdata_dir_exists(mocker, tmp_path) -> None:
    """测试当用户数据目录已存在时的行为"""
    # 模拟目录已存在
    mocker.patch.object(Path, "is_dir", MagicMock(return_value=True))
    # 模拟创建目录的方法
    md = mocker.patch.object(Path, "mkdir", MagicMock())

    # 调用创建用户数据目录的函数
    create_userdata_dir(f"{tmp_path}/bar")
    # 验证没有尝试创建目录
    assert md.call_count == 0


def test_create_userdata_dir_exists_exception(mocker, tmp_path) -> None:
    """测试当用户数据目录不存在且不允许创建时的异常"""
    # 模拟目录不存在
    mocker.patch.object(Path, "is_dir", MagicMock(return_value=False))
    # 模拟创建目录的方法
    md = mocker.patch.object(Path, "mkdir", MagicMock())

    # 验证当不允许创建目录且目录不存在时，抛出异常
    with pytest.raises(OperationalException, match=r"目录 `.*.{1,2}bar` 不存在.*"):
        create_userdata_dir(f"{tmp_path}/bar", create_dir=False)
    # 验证没有尝试创建目录
    assert md.call_count == 0


def test_copy_sample_files(mocker, tmp_path) -> None:
    """测试复制示例文件功能"""
    # 模拟目录存在
    mocker.patch.object(Path, "is_dir", MagicMock(return_value=True))
    # 模拟文件不存在
    mocker.patch.object(Path, "exists", MagicMock(return_value=False))
    # 模拟复制文件的方法
    copymock = mocker.patch("shutil.copy", MagicMock())

    # 调用复制示例文件的函数
    copy_sample_files(Path(f"{tmp_path}/bar"))
    # 验证复制了3个示例文件
    assert copymock.call_count == 3
    # 验证复制的目标路径正确
    assert copymock.call_args_list[0][0][1] == str(tmp_path / "bar/strategies/sample_strategy.py")
    assert copymock.call_args_list[1][0][1] == str(
        tmp_path / "bar/hyperopts/sample_hyperopt_loss.py"
    )
    assert copymock.call_args_list[2][0][1] == str(
        tmp_path / "bar/notebooks/strategy_analysis_example.ipynb"
    )


def test_copy_sample_files_errors(mocker, tmp_path, caplog) -> None:
    """测试复制示例文件时的错误处理"""
    # 模拟目录不存在
    mocker.patch.object(Path, "is_dir", MagicMock(return_value=False))
    # 模拟文件不存在
    mocker.patch.object(Path, "exists", MagicMock(return_value=False))
    # 模拟复制文件的方法
    mocker.patch("shutil.copy", MagicMock())
    
    # 验证当目录不存在时抛出异常
    with pytest.raises(OperationalException, match=r"目录 `.*.{1,2}bar` 不存在\."):
        copy_sample_files(Path(f"{tmp_path}/bar"))

    # 模拟策略目录不存在
    mocker.patch.object(Path, "is_dir", MagicMock(side_effect=[True, False]))

    # 验证当策略目录不存在时抛出异常
    with pytest.raises(
        OperationalException,
        match=r"目录 `.*.{1,2}bar.{1,2}strategies` 不存在\.",
    ):
        copy_sample_files(Path(f"{tmp_path}/bar"))
    
    # 模拟目录存在，文件已存在
    mocker.patch.object(Path, "is_dir", MagicMock(return_value=True))
    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    
    # 验证当文件已存在且不覆盖时，有正确的日志
    copy_sample_files(Path(f"{tmp_path}/bar"))
    assert log_has_re(r"文件 `.*` 已存在，不部署示例文件\.", caplog)
    
    # 清除日志
    caplog.clear()
    
    # 验证当文件已存在且覆盖时，有正确的日志
    copy_sample_files(Path(f"{tmp_path}/bar"), overwrite=True)
    assert log_has_re(r"文件 `.*` 已存在，正在覆盖\.", caplog)