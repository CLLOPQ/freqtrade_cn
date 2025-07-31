import hashlib
from copy import deepcopy
from pathlib import Path

import rapidjson


def get_strategy_run_id(strategy) -> str:
    """
    为回测运行生成唯一标识哈希。相同的配置和策略文件将始终返回相同的哈希。
    :param strategy: 策略对象。
    :return: 十六进制字符串ID。
    """
    digest = hashlib.sha1()  # noqa: S324
    config = deepcopy(strategy.config)

    # 对单个回测结果没有影响的选项。
    not_important_keys = ("strategy_list", "original_config", "telegram", "api_server")
    for k in not_important_keys:
        if k in config:
            del config[k]

    # 显式允许NaN值（例如max_open_trades）。因为这对获取哈希没有影响。
    digest.update(
        rapidjson.dumps(config, default=str, number_mode=rapidjson.NM_NAN).encode("utf-8")
    )
    # 包含 _ft_params_from_file - 因此更改参数文件会导致缓存失效
    digest.update(
        rapidjson.dumps(
            strategy._ft_params_from_file, default=str, number_mode=rapidjson.NM_NAN
        ).encode("utf-8")
    )
    with Path(strategy.__file__).open("rb") as fp:
        digest.update(fp.read())
    return digest.hexdigest().lower()


def get_backtest_metadata_filename(filename: Path | str) -> Path:
    """返回指定回测结果文件的元数据文件名。"""
    filename = Path(filename)
    return filename.parent / Path(f"{filename.stem}.meta.json")