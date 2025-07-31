import logging
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from pandas import DataFrame

from freqtrade.configuration import sanitize_config
from freqtrade.constants import LAST_BT_RESULT_FN
from freqtrade.enums.runmode import RunMode
from freqtrade.ft_types import BacktestResultType
from freqtrade.misc import dump_json_to_file, file_dump_json
from freqtrade.optimize.backtest_caching import get_backtest_metadata_filename


logger = logging.getLogger(__name__)


def file_dump_joblib(file_obj: BytesIO, data: Any, log: bool = True) -> None:
    """
    将对象数据转储到文件中
    :param filename: 要创建的文件
    :param data: 要保存的对象数据
    :return:
    """
    import joblib

    joblib.dump(data, file_obj)


def _generate_filename(recordfilename: Path, appendix: str, suffix: str) -> Path:
    """
    根据提供的参数生成文件名。
    :param recordfilename: Path对象，可以是文件名或目录。
    :param appendix: 用于文件名。例如：backtest-result-<datetime>
    :param suffix: 文件的后缀，例如：.json、.pkl
    :return: 生成的文件名（Path对象）
    """
    if recordfilename.is_dir():
        filename = (recordfilename / f"backtest-result-{appendix}").with_suffix(suffix)
    else:
        filename = Path.joinpath(
            recordfilename.parent, f"{recordfilename.stem}-{appendix}"
        ).with_suffix(suffix)
    return filename


def store_backtest_results(
    config: dict,
    stats: BacktestResultType,
    dtappendix: str,
    *,
    market_change_data: DataFrame | None = None,
    analysis_results: dict[str, dict[str, DataFrame]] | None = None,
    strategy_files: dict[str, str] | None = None,
) -> Path:
    """
    将回测结果和分析数据存储到zip文件中，元数据单独存储以方便使用。
    :param config: 配置字典
    :param stats: 包含回测统计数据的数据框
    :param dtappendix: 用于文件名的日期时间
    :param market_change_data: 包含市场变化数据的数据框
    :param analysis_results: 包含分析结果的字典
    """
    recordfilename: Path = config["exportfilename"]
    zip_filename = _generate_filename(recordfilename, dtappendix, ".zip")
    base_filename = _generate_filename(recordfilename, dtappendix, "")
    json_filename = _generate_filename(recordfilename, dtappendix, ".json")

    # 将元数据以.json扩展名单独存储
    file_dump_json(get_backtest_metadata_filename(json_filename), stats["metadata"])

    # 将最新的回测信息单独存储
    latest_filename = Path.joinpath(zip_filename.parent, LAST_BT_RESULT_FN)
    file_dump_json(latest_filename, {"latest_backtest": str(zip_filename.name)}, log=False)

    # 创建zip文件并添加文件
    with ZipFile(zip_filename, "w", ZIP_DEFLATED) as zipf:
        # 存储统计数据
        stats_copy = {
            "strategy": stats["strategy"],
            "strategy_comparison": stats["strategy_comparison"],
        }
        stats_buf = StringIO()
        dump_json_to_file(stats_buf, stats_copy)
        zipf.writestr(json_filename.name, stats_buf.getvalue())

        config_buf = StringIO()
        dump_json_to_file(config_buf, sanitize_config(config["original_config"]))
        zipf.writestr(f"{base_filename.stem}_配置.json", config_buf.getvalue())

        for strategy_name, strategy_file in (strategy_files or {}).items():
            # 存储策略文件及其参数
            strategy_buf = BytesIO()
            strategy_path = Path(strategy_file)
            if not strategy_path.is_file():
                logger.warning(f"策略文件'{strategy_path}'不存在。已跳过。")
                continue
            with strategy_path.open("rb") as strategy_file_obj:
                strategy_buf.write(strategy_file_obj.read())
            strategy_buf.seek(0)
            zipf.writestr(f"{base_filename.stem}_{strategy_name}_策略.py", strategy_buf.getvalue())
            strategy_params = strategy_path.with_suffix(".json")
            if strategy_params.is_file():
                strategy_params_buf = BytesIO()
                with strategy_params.open("rb") as strategy_params_obj:
                    strategy_params_buf.write(strategy_params_obj.read())
                strategy_params_buf.seek(0)
                zipf.writestr(
                    f"{base_filename.stem}_{strategy_name}_参数.json",
                    strategy_params_buf.getvalue(),
                )

        # 如果存在，添加市场变化数据
        if market_change_data is not None:
            market_change_name = f"{base_filename.stem}_市场变化.feather"
            market_change_buf = BytesIO()
            market_change_data.reset_index().to_feather(
                market_change_buf, compression_level=9, compression="lz4"
            )
            market_change_buf.seek(0)
            zipf.writestr(market_change_name, market_change_buf.getvalue())

        # 如果存在分析结果且以回测模式运行，则添加分析结果
        if (
            config.get("export", "none") == "signals"
            and analysis_results is not None
            and config.get("runmode", RunMode.OTHER) == RunMode.BACKTEST
        ):
            for name in ["signals", "rejected", "exited"]:
                if name in analysis_results:
                    analysis_name = f"{base_filename.stem}_{name}_分析结果.pkl"
                    analysis_buf = BytesIO()
                    file_dump_joblib(analysis_buf, analysis_results[name])
                    analysis_buf.seek(0)
                    zipf.writestr(analysis_name, analysis_buf.getvalue())

    return zip_filename