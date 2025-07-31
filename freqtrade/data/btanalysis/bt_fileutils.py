"""
分析回测数据时的辅助工具
"""

import logging
import zipfile
from copy import copy
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from freqtrade.constants import LAST_BT_RESULT_FN
from freqtrade.exceptions import ConfigurationError, OperationalException
from freqtrade.ft_types import BacktestHistoryEntryType, BacktestResultType
from freqtrade.misc import file_dump_json, json_load
from freqtrade.optimize.backtest_caching import get_backtest_metadata_filename
from freqtrade.persistence import LocalTrade, Trade, init_db


logger = logging.getLogger(__name__)

# 最新格式
BT_DATA_COLUMNS = [
    "pair",
    "stake_amount",
    "max_stake_amount",
    "amount",
    "open_date",
    "close_date",
    "open_rate",
    "close_rate",
    "fee_open",
    "fee_close",
    "trade_duration",
    "profit_ratio",
    "profit_abs",
    "exit_reason",
    "initial_stop_loss_abs",
    "initial_stop_loss_ratio",
    "stop_loss_abs",
    "stop_loss_ratio",
    "min_rate",
    "max_rate",
    "is_open",
    "enter_tag",
    "leverage",
    "is_short",
    "open_timestamp",
    "close_timestamp",
    "orders",
    "funding_fees",
]


def get_latest_optimize_filename(directory: Path | str, variant: str) -> str:
    """
    基于 '.last_result.json' 获取最新的回测导出文件。
    :param directory: 用于搜索最新结果的目录
    :param variant: 'backtest' 或 'hyperopt' - 要返回的方法
    :return: 包含最新回测结果文件名的字符串
    :raises: 在以下情况下引发 ValueError:
        * 目录不存在
        * `directory/.last_result.json` 不存在
        * `directory/.last_result.json` 内容错误
    """
    if isinstance(directory, str):
        directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"目录 '{directory}' 不存在。")
    filename = directory / LAST_BT_RESULT_FN

    if not filename.is_file():
        raise ValueError(
            f"目录 '{directory}' 似乎尚未包含回测统计信息。"
        )

    with filename.open() as file:
        data = json_load(file)

    if f"latest_{variant}" not in data:
        raise ValueError(f"'{LAST_BT_RESULT_FN}' 格式无效。")

    return data[f"latest_{variant}"]


def get_latest_backtest_filename(directory: Path | str) -> str:
    """
    基于 '.last_result.json' 获取最新的回测导出文件。
    :param directory: 用于搜索最新结果的目录
    :return: 包含最新回测结果文件名的字符串
    :raises: 在以下情况下引发 ValueError:
        * 目录不存在
        * `directory/.last_result.json` 不存在
        * `directory/.last_result.json` 内容错误
    """
    return get_latest_optimize_filename(directory, "backtest")


def get_latest_hyperopt_filename(directory: Path | str) -> str:
    """
    基于 '.last_result.json' 获取最新的超参数优化导出文件。
    :param directory: 用于搜索最新结果的目录
    :return: 包含最新超参数优化结果文件名的字符串
    :raises: 在以下情况下引发 ValueError:
        * 目录不存在
        * `directory/.last_result.json` 不存在
        * `directory/.last_result.json` 内容错误
    """
    try:
        return get_latest_optimize_filename(directory, "hyperopt")
    except ValueError:
        # 返回默认（传统）的 pickle 文件名
        return "hyperopt_results.pickle"


def get_latest_hyperopt_file(directory: Path | str, predef_filename: str | None = None) -> Path:
    """
    基于 '.last_result.json' 获取最新的超参数优化导出文件。
    :param directory: 用于搜索最新结果的目录
    :return: 包含最新超参数优化结果文件名的字符串
    :raises: 在以下情况下引发 ValueError:
        * 目录不存在
        * `directory/.last_result.json` 不存在
        * `directory/.last_result.json` 内容错误
    """
    if isinstance(directory, str):
        directory = Path(directory)
    if predef_filename:
        if Path(predef_filename).is_absolute():
            raise ConfigurationError(
                "--hyperopt-filename 仅接受文件名，不接受绝对路径。"
            )
        return directory / predef_filename
    return directory / get_latest_hyperopt_filename(directory)


def load_backtest_metadata(filename: Path | str) -> dict[str, Any]:
    """
    从回测结果文件中读取元数据字典，无需读取和反序列化整个文件。
    :param filename: 回测结果文件的路径。
    :return: 元数据字典，如果元数据不存在则为 None。
    """
    filename = get_backtest_metadata_filename(filename)
    try:
        with filename.open() as fp:
            return json_load(fp)
    except FileNotFoundError:
        return {}
    except Exception as e:
        raise OperationalException("加载回测元数据时发生意外错误。") from e


def load_backtest_stats(filename: Path | str) -> BacktestResultType:
    """
    加载回测统计文件。
    :param filename: pathlib.Path 对象或指向文件的字符串。
    :return: 包含结果文件的字典。
    """
    if isinstance(filename, str):
        filename = Path(filename)
    if filename.is_dir():
        filename = filename / get_latest_backtest_filename(filename)
    if not filename.is_file():
        raise ValueError(f"文件 {filename} 不存在。")
    logger.info(f"从 {filename} 加载回测结果")

    if filename.suffix == ".zip":
        data = json_load(
            StringIO(
                load_file_from_zip(filename, filename.with_suffix(".json").name).decode("utf-8")
            )
        )
    else:
        with filename.open() as file:
            data = json_load(file)

    # 传统列表格式不包含元数据。
    if isinstance(data, dict):
        data["metadata"] = load_backtest_metadata(filename)
    return data


def load_and_merge_backtest_result(strategy_name: str, filename: Path, results: dict[str, Any]):
    """
    从多策略结果中加载一个策略并将其与结果合并
    :param strategy_name: 结果中包含的策略名称
    :param filename: 要加载的回测结果文件名
    :param results: 要合并结果的字典。
    """
    bt_data = load_backtest_stats(filename)
    k: Literal["metadata", "strategy"]
    for k in ("metadata", "strategy"):
        results[k][strategy_name] = bt_data[k][strategy_name]
    results["metadata"][strategy_name]["filename"] = filename.stem
    comparison = bt_data["strategy_comparison"]
    for i in range(len(comparison)):
        if comparison[i]["key"] == strategy_name:
            results["strategy_comparison"].append(comparison[i])
            break


def _get_backtest_files(dirname: Path) -> list[Path]:
    # 分别获取 json 和 zip 文件并合并结果
    json_files = dirname.glob("backtest-result-*-[0-9][0-9]*.json")
    zip_files = dirname.glob("backtest-result-*-[0-9][0-9]*.zip")
    return list(reversed(sorted(list(json_files) + list(zip_files))))


def _extract_backtest_result(filename: Path) -> list[BacktestHistoryEntryType]:
    metadata = load_backtest_metadata(filename)
    return [
        {
            "filename": filename.stem,
            "strategy": s,
            "run_id": v["run_id"],
            "notes": v.get("notes", ""),
            # 回测"运行"时间
            "backtest_start_time": v["backtest_start_time"],
            # 回测时间范围
            "backtest_start_ts": v.get("backtest_start_ts", None),
            "backtest_end_ts": v.get("backtest_end_ts", None),
            "timeframe": v.get("timeframe", None),
            "timeframe_detail": v.get("timeframe_detail", None),
        }
        for s, v in metadata.items()
    ]


def get_backtest_result(filename: Path) -> list[BacktestHistoryEntryType]:
    """
    从元数据文件中读取回测结果
    """
    return _extract_backtest_result(filename)


def get_backtest_resultlist(dirname: Path) -> list[BacktestHistoryEntryType]:
    """
    从元数据文件中读取回测结果列表
    """
    return [
        result
        for filename in _get_backtest_files(dirname)
        for result in _extract_backtest_result(filename)
    ]


def delete_backtest_result(file_abs: Path):
    """
    删除回测结果文件和相应的元数据文件。
    """
    # *.meta.json
    logger.info(f"删除回测结果文件: {file_abs.name}")

    for file in file_abs.parent.glob(f"{file_abs.stem}*"):
        logger.info(f"删除文件: {file}")
        file.unlink()


def update_backtest_metadata(filename: Path, strategy: str, content: dict[str, Any]):
    """
    用新内容更新回测元数据文件。
    :raises: 如果元数据文件不存在或策略不在此文件中，则引发 ValueError。
    """
    metadata = load_backtest_metadata(filename)
    if not metadata:
        raise ValueError("文件不存在。")
    if strategy not in metadata:
        raise ValueError("策略不在元数据中。")
    metadata[strategy].update(content)
    # 再次写入数据。
    file_dump_json(get_backtest_metadata_filename(filename), metadata)


def get_backtest_market_change(filename: Path, include_ts: bool = True) -> pd.DataFrame:
    """
    读取回测市场变化文件。
    """
    if filename.suffix == ".zip":
        data = load_file_from_zip(filename, f"{filename.stem}_market_change.feather")
        df = pd.read_feather(BytesIO(data))
    else:
        df = pd.read_feather(filename)
    if include_ts:
        df.loc[:, "__date_ts"] = df.loc[:, "date"].astype(np.int64) // 1000 // 1000
    return df


def find_existing_backtest_stats(
    dirname: Path | str, run_ids: dict[str, str], min_backtest_date: datetime | None = None
) -> dict[str, Any]:
    """
    查找与指定运行 ID 匹配的现有回测统计信息并加载它们。
    :param dirname: pathlib.Path 对象或指向文件的字符串。
    :param run_ids: {策略名称: id字符串} 字典。
    :param min_backtest_date: 不加载早于指定日期的回测。
    :return: 结果字典。
    """
    # 复制以便我们可以修改此字典而不影响父作用域。
    run_ids = copy(run_ids)
    dirname = Path(dirname)
    results: dict[str, Any] = {
        "metadata": {},
        "strategy": {},
        "strategy_comparison": [],
    }

    for filename in _get_backtest_files(dirname):
        metadata = load_backtest_metadata(filename)
        if not metadata:
            # 文件按从新到旧排序。当遇到没有元数据的文件时，可以安全地假设旧文件也不会有任何元数据。
            break

        for strategy_name, run_id in list(run_ids.items()):
            strategy_metadata = metadata.get(strategy_name, None)
            if not strategy_metadata:
                # 此策略不存在于已分析的回测中。
                continue

            if min_backtest_date is not None:
                backtest_date = strategy_metadata["backtest_start_time"]
                backtest_date = datetime.fromtimestamp(backtest_date, tz=timezone.utc)
                if backtest_date < min_backtest_date:
                    # 不为此策略使用缓存结果，因为第一个结果太旧。
                    del run_ids[strategy_name]
                    continue

            if strategy_metadata["run_id"] == run_id:
                del run_ids[strategy_name]
                load_and_merge_backtest_result(strategy_name, filename, results)

        if len(run_ids) == 0:
            break
    return results


def _load_backtest_data_df_compatibility(df: pd.DataFrame) -> pd.DataFrame:
    """
    对旧回测数据的兼容性支持。
    """
    df["open_date"] = pd.to_datetime(df["open_date"], utc=True)
    df["close_date"] = pd.to_datetime(df["close_date"], utc=True)
    # 对 pre short 列的兼容性支持
    if "is_short" not in df.columns:
        df["is_short"] = False
    if "leverage" not in df.columns:
        df["leverage"] = 1.0
    if "enter_tag" not in df.columns:
        df["enter_tag"] = df["buy_tag"]
        df = df.drop(["buy_tag"], axis=1)
    if "max_stake_amount" not in df.columns:
        df["max_stake_amount"] = df["stake_amount"]
    if "orders" not in df.columns:
        df["orders"] = None
    if "funding_fees" not in df.columns:
        df["funding_fees"] = 0.0
    return df


def load_backtest_data(filename: Path | str, strategy: str | None = None) -> pd.DataFrame:
    """
    加载回测数据文件。
    :param filename: pathlib.Path 对象或指向文件或目录的字符串
    :param strategy: 要加载的策略 - 主要与多策略回测相关
                     也可以作为加载正确结果的保护。
    :return: 带有分析结果的数据框
    :raise: 如果加载出错则引发 ValueError。
    """
    data = load_backtest_stats(filename)
    if not isinstance(data, list):
        # 新的嵌套格式
        if "strategy" not in data:
            raise ValueError("未知的数据格式。")

        if not strategy:
            if len(data["strategy"]) == 1:
                strategy = next(iter(data["strategy"].keys()))
            else:
                raise ValueError(
                    "检测到包含多个策略的回测结果。请指定一个策略。"
                )

        if strategy not in data["strategy"]:
            raise ValueError(
                f"策略 {strategy} 不在回测结果中。可用策略为 '{','.join(data['strategy'].keys())}'"
            )

        data = data["strategy"][strategy]["trades"]
        df = pd.DataFrame(data)
        if not df.empty:
            df = _load_backtest_data_df_compatibility(df)

    else:
        # 旧格式 - 仅包含列表。
        raise OperationalException(
            "仅包含交易数据的回测结果不再受支持。"
        )
    if not df.empty:
        df = df.sort_values("open_date").reset_index(drop=True)
    return df


def load_file_from_zip(zip_path: Path, filename: str) -> bytes:
    """
    从 zip 文件中加载文件
    :param zip_path: zip 文件的路径
    :param filename: 要加载的文件的名称
    :return: 文件的字节
    :raises: 如果加载出错则引发 ValueError。
    """
    try:
        with zipfile.ZipFile(zip_path) as zipf:
            try:
                with zipf.open(filename) as file:
                    return file.read()
            except KeyError:
                logger.error(f"在 zip 文件 {zip_path} 中未找到文件 {filename}")
                raise ValueError(f"在 zip 文件 {zip_path} 中未找到文件 {filename}") from None
    except FileNotFoundError:
        raise ValueError(f"未找到 zip 文件 {zip_path}。")
    except zipfile.BadZipFile:
        logger.error(f"损坏的 zip 文件: {zip_path}。")
        raise ValueError(f"损坏的 zip 文件: {zip_path}。") from None


def load_backtest_analysis_data(backtest_dir: Path, name: str):
    """
    从 pickle 文件或 zip 文件中加载回测分析数据
    :param backtest_dir: 包含回测结果的目录
    :param name: 要加载的分析数据的名称（signals、rejected、exited）
    :return: 分析数据
    """
    import joblib

    if backtest_dir.is_dir():
        lbf = Path(get_latest_backtest_filename(backtest_dir))
        zip_path = backtest_dir / lbf
    else:
        zip_path = backtest_dir

    if zip_path.suffix == ".zip":
        # 从 zip 文件加载
        analysis_name = f"{zip_path.stem}_{name}.pkl"
        data = load_file_from_zip(zip_path, analysis_name)
        if not data:
            return None
        loaded_data = joblib.load(BytesIO(data))
        

        logger.info(f"从 zip 文件加载 {name} 蜡烛图: {str(zip_path)}:{analysis_name}")
        return loaded_data

    else:
        # 从单独的 pickle 文件加载
        if backtest_dir.is_dir():
            scpf = Path(backtest_dir, f"{zip_path.stem}_{name}.pkl")
        else:
            scpf = Path(backtest_dir.parent / f"{backtest_dir.stem}_{name}.pkl")

        try:
            with scpf.open("rb") as scp:
                loaded_data = joblib.load(scp)
                logger.info(f"加载 {name} 蜡烛图: {str(scpf)}")
                return loaded_data
        except Exception:
            logger.exception(f"无法从 pickle 结果中加载 {name} 数据。")
            return None


def load_rejected_signals(backtest_dir: Path):
    """
    从回测目录加载被拒绝的信号
    """
    return load_backtest_analysis_data(backtest_dir, "rejected")


def load_signal_candles(backtest_dir: Path):
    """
    从回测目录加载信号蜡烛图
    """
    return load_backtest_analysis_data(backtest_dir, "signals")


def load_exit_signal_candles(backtest_dir: Path) -> dict[str, dict[str, pd.DataFrame]]:
    """
    从回测目录加载退出信号蜡烛图
    """
    return load_backtest_analysis_data(backtest_dir, "exited")


def trade_list_to_dataframe(trades: list[Trade] | list[LocalTrade]) -> pd.DataFrame:
    """
    将交易对象列表转换为 pandas 数据框
    :param trades: 交易对象列表
    :return: 包含 BT_DATA_COLUMNS 的数据框
    """
    df = pd.DataFrame.from_records([t.to_json(True) for t in trades], columns=BT_DATA_COLUMNS)
    if len(df) > 0:
        df["close_date"] = pd.to_datetime(df["close_date"], utc=True)
        df["open_date"] = pd.to_datetime(df["open_date"], utc=True)
        df["close_rate"] = df["close_rate"].astype("float64")
    return df


def load_trades_from_db(db_url: str, strategy: str | None = None) -> pd.DataFrame:
    """
    从数据库加载交易（使用 dburl）
    :param db_url: sqlite 格式的 url（默认格式 sqlite:///tradesv3.dry-run.sqlite）
    :param strategy: 要加载的策略 - 主要与多策略回测相关
                     也可以作为加载正确结果的保护。
    :return: 包含交易的数据框
    """
    init_db(db_url)

    filters = []
    if strategy:
        filters.append(Trade.strategy == strategy)
    trades = trade_list_to_dataframe(list(Trade.get_trades(filters).all()))

    return trades


def load_trades(
    source: str,
    db_url: str,
    exportfilename: Path,
    no_trades: bool = False,
    strategy: str | None = None,
) -> pd.DataFrame:
    """
    基于配置选项 'trade_source':
    * 从数据库加载数据（使用 `db_url`）
    * 从回测文件加载数据（使用 `exportfilename`）
    :param source: "DB" 或 "file" - 指定要加载的来源
    :param db_url: sqlalchemy 格式的数据库 url
    :param exportfilename: 由回测生成的 Json 文件
    :param no_trades: 跳过使用交易，仅返回回测数据列
    :return: 包含交易的数据框
    """
    if no_trades:
        df = pd.DataFrame(columns=BT_DATA_COLUMNS)
        return df

    if source == "DB":
        return load_trades_from_db(db_url)
    elif source == "file":
        return load_backtest_data(exportfilename, strategy)


def extract_trades_of_period(
    dataframe: pd.DataFrame, trades: pd.DataFrame, date_index=False
) -> pd.DataFrame:
    """
    比较交易和回测对数据框，以获取在回测期间执行的交易
    :return: 期间交易的数据框
    """
    if date_index:
        trades_start = dataframe.index[0]
        trades_stop = dataframe.index[-1]
    else:
        trades_start = dataframe.iloc[0]["date"]
        trades_stop = dataframe.iloc[-1]["date"]
    trades = trades.loc[
        (trades["open_date"] >= trades_start) & (trades["close_date"] <= trades_stop)
    ]
    return trades
