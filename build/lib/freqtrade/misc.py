"""
Freqtrade和脚本的各种工具函数
"""

import gzip
import logging
from collections.abc import Iterator, Mapping
from io import StringIO
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlparse

import pandas as pd
import rapidjson

from freqtrade.enums import SignalTagType, SignalType


logger = logging.getLogger(__name__)


def dump_json_to_file(file_obj: TextIO, data: Any) -> None:
    """
    将JSON数据转储到文件对象中
    :param file_obj: 要写入的文件对象
    :param data: 要保存的JSON数据
    """
    rapidjson.dump(data, file_obj, default=str, number_mode=rapidjson.NM_NATIVE)


def file_dump_json(filename: Path, data: Any, is_zip: bool = False, log: bool = True) -> None:
    """
    将JSON数据转储到文件中
    :param filename: 要创建的文件
    :param is_zip: 文件是否应为压缩格式
    :param data: 要保存的JSON数据
    :return:
    """

    if is_zip:
        if filename.suffix != ".gz":
            filename = filename.with_suffix(".gz")
        if log:
            logger.info(f'将JSON数据转储到“{filename}”')

        with gzip.open(filename, "wt", encoding="utf-8") as fpz:
            dump_json_to_file(fpz, data)
    else:
        if log:
            logger.info(f'将JSON数据转储到“{filename}”')
        with filename.open("w") as fp:
            dump_json_to_file(fp, data)

    logger.debug(f'已完成JSON到“{filename}”的转储')


def json_load(datafile: TextIO) -> Any:
    """
    使用rapidjson加载数据
    为了获得一致的体验，请使用此函数，将number_mode设置为"NM_NATIVE"以获得最佳性能
    """
    return rapidjson.load(datafile, number_mode=rapidjson.NM_NATIVE)


def file_load_json(file: Path):
    if file.suffix != ".gz":
        gzipfile = file.with_suffix(file.suffix + ".gz")
    else:
        gzipfile = file
    # 优先尝试gzip文件，否则尝试常规json文件。
    if gzipfile.is_file():
        logger.debug(f"从文件{gzipfile}加载历史数据")
        with gzip.open(gzipfile, "rt", encoding="utf-8") as datafile:
            pairdata = json_load(datafile)
    elif file.is_file():
        logger.debug(f"从文件{file}加载历史数据")
        with file.open() as datafile:
            pairdata = json_load(datafile)
    else:
        return None
    return pairdata


def is_file_in_dir(file: Path, directory: Path) -> bool:
    """
    辅助函数，用于检查文件是否在目录中。
    """
    return file.is_file() and file.parent.samefile(directory)


def pair_to_filename(pair: str) -> str:
    for ch in ["/", " ", ".", "@", "$", "+", ":"]:
        pair = pair.replace(ch, "_")
    return pair


def deep_merge_dicts(source, destination, allow_null_overrides: bool = True):
    """
    源中的值会覆盖目标中的值，返回目标（并对其进行修改！！）
    示例：
    >>> a = { 'first' : { 'rows' : { 'pass' : 'dog', 'number' : '1' } } }
    >>> b = { 'first' : { 'rows' : { 'fail' : 'cat', 'number' : '5' } } }
    >>> merge(b, a) == { 'first' : { 'rows' : { 'pass' : 'dog', 'fail' : 'cat', 'number' : '5' } } }
    True
    """
    for key, value in source.items():
        if isinstance(value, dict):
            # 获取节点或创建节点
            node = destination.setdefault(key, {})
            deep_merge_dicts(value, node, allow_null_overrides)
        elif value is not None or allow_null_overrides:
            destination[key] = value

    return destination


def round_dict(d, n):
    """
    将字典中的浮点数值四舍五入到小数点后n位。
    """
    return {k: (round(v, n) if isinstance(v, float) else v) for k, v in d.items()}


DictMap = dict[str, Any] | Mapping[str, Any]


def safe_value_fallback(obj: DictMap, key1: str, key2: str | None = None, default_value=None):
    """
    在obj中搜索值，如果该值不为None则返回。
    然后在obj中搜索key2，如果key2不为None则返回，否则使用default_value。
    否则返回None。
    """
    if key1 in obj and obj[key1] is not None:
        return obj[key1]
    else:
        if key2 and key2 in obj and obj[key2] is not None:
            return obj[key2]
    return default_value


def safe_value_fallback2(dict1: DictMap, dict2: DictMap, key1: str, key2: str, default_value=None):
    """
    在dict1中搜索值，如果该值不为None则返回。
    回退到dict2，如果dict2中的key2不为None则返回。
    否则返回None。

    """
    if key1 in dict1 and dict1[key1] is not None:
        return dict1[key1]
    else:
        if key2 in dict2 and dict2[key2] is not None:
            return dict2[key2]
    return default_value


def plural(num: float, singular: str, plural: str | None = None) -> str:
    return singular if (num == 1 or num == -1) else plural or singular + "s"


def chunks(lst: list[Any], n: int) -> Iterator[list[Any]]:
    """
    将列表拆分为大小为n的块。
    :param lst: 要拆分的列表
    :param n: 每个块的最大元素数
    :return: None
    """
    for chunk in range(0, len(lst), n):
        yield (lst[chunk : chunk + n])


def parse_db_uri_for_logging(uri: str):
    """
    辅助方法，用于解析数据库URI，如果其中包含密码，则返回带有密码被屏蔽的相同数据库URI。否则，返回未修改的数据库URI
    :param uri: 用于日志记录的数据库URI
    """
    parsed_db_uri = urlparse(uri)
    if not parsed_db_uri.netloc:  # 无需屏蔽，因为未提供密码
        return uri
    pwd = parsed_db_uri.netloc.split(":")[1].split("@")[0]
    return parsed_db_uri.geturl().replace(f":{pwd}@", ":*****@")


def dataframe_to_json(dataframe: pd.DataFrame) -> str:
    """
    使用JSON序列化DataFrame以通过网络传输
    :param dataframe: pandas DataFrame对象
    :returns: pandas DataFrame的JSON字符串
    """
    return dataframe.to_json(orient="split")


def json_to_dataframe(data: str) -> pd.DataFrame:
    """
    将JSON反序列化为DataFrame
    :param data: JSON字符串
    :returns: 从JSON字符串转换的pandas DataFrame
    """
    dataframe = pd.read_json(StringIO(data), orient="split")
    if "date" in dataframe.columns:
        dataframe["date"] = pd.to_datetime(dataframe["date"], unit="ms", utc=True)

    return dataframe


def remove_entry_exit_signals(dataframe: pd.DataFrame):
    """
    从DataFrame中移除入场和出场信号

    :param dataframe: 要从中移除信号的DataFrame
    """
    dataframe[SignalType.ENTER_LONG.value] = 0
    dataframe[SignalType.EXIT_LONG.value] = 0
    dataframe[SignalType.ENTER_SHORT.value] = 0
    dataframe[SignalType.EXIT_SHORT.value] = 0
    dataframe[SignalTagType.ENTER_TAG.value] = None
    dataframe[SignalTagType.EXIT_TAG.value] = None

    return dataframe


def append_candles_to_dataframe(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """
    将`right` DataFrame附加到`left` DataFrame

    :param left: 要附加到的完整DataFrame
    :param right: 包含要附加数据的新DataFrame
    :returns: 包含right数据的DataFrame
    """
    if left.iloc[-1]["date"] != right.iloc[-1]["date"]:
        left = pd.concat([left, right])

    # 仅在内存中保留最后1500根K线
    left = left[-1500:] if len(left) > 1500 else left
    left.reset_index(drop=True, inplace=True)

    return left