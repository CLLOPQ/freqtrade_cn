import logging
import shutil
from pathlib import Path

from freqtrade.configuration.detect_environment import running_in_docker
from freqtrade.constants import (
    USER_DATA_FILES,
    USERPATH_FREQAIMODELS,
    USERPATH_HYPEROPTS,
    USERPATH_NOTEBOOKS,
    USERPATH_STRATEGIES,
    Config,
)
from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


def create_datadir(config: Config, datadir: str | None = None) -> Path:
    """
    创建数据目录
    """
    folder = Path(datadir) if datadir else Path(f"{config['user_data_dir']}/data")
    if not datadir:
        # 设置数据目录
        exchange_name = config.get("exchange", {}).get("name", "").lower()
        folder = folder.joinpath(exchange_name)

    if not folder.is_dir():
        folder.mkdir(parents=True)
        logger.info(f"已创建数据目录：{datadir}")
    return folder


def chown_user_directory(directory: Path) -> None:
    """
    必要时使用Sudo更改主目录的权限
    仅在docker中运行时适用！
    """
    if running_in_docker():
        try:
            import subprocess  # noqa: S404, RUF100

            subprocess.check_output(["sudo", "chown", "-R", "ftuser:", str(directory.resolve())])
        except Exception:
            logger.warning(f"无法更改 {directory} 的所有者")


def create_userdata_dir(directory: str, create_dir: bool = False) -> Path:
    """
    创建用户数据目录结构。
    如果create_dir为True，则当父目录不存在时将创建它。
    如果父目录存在，子目录将始终被创建。
    如果给定的目录不存在，将引发OperationalException。
    :param directory: 要检查的目录
    :param create_dir: 如果目录不存在则创建它。
    :return: 包含目录的Path对象
    """
    sub_dirs = [
        "backtest_results",
        "data",
        USERPATH_HYPEROPTS,
        "hyperopt_results",
        "logs",
        USERPATH_NOTEBOOKS,
        "plot",
        USERPATH_STRATEGIES,
        USERPATH_FREQAIMODELS,
    ]
    folder = Path(directory)
    chown_user_directory(folder)
    if not folder.is_dir():
        if create_dir:
            folder.mkdir(parents=True)
            logger.info(f"已创建用户数据目录：{folder}")
        else:
            raise OperationalException(
                f"目录 `{folder}` 不存在。"
                "请使用 `freqtrade create-userdir` 创建用户目录"
            )

    # 创建所需的子目录
    for f in sub_dirs:
        subfolder = folder / f
        if not subfolder.is_dir():
            if subfolder.exists() or subfolder.is_symlink():
                raise OperationalException(
                    f"文件 `{subfolder}` 已存在且不是目录。"
                    "Freqtrade要求这是一个目录。"
                )
            subfolder.mkdir(parents=False)
    return folder


def copy_sample_files(directory: Path, overwrite: bool = False) -> None:
    """
    将文件从模板复制到用户数据目录。
    :param directory: 要复制数据到的目录
    :param overwrite: 是否覆盖现有的示例文件
    """
    if not directory.is_dir():
        raise OperationalException(f"目录 `{directory}` 不存在。")
    sourcedir = Path(__file__).parents[1] / "templates"
    for source, target in USER_DATA_FILES.items():
        targetdir = directory / target
        if not targetdir.is_dir():
            raise OperationalException(f"目录 `{targetdir}` 不存在。")
        targetfile = targetdir / source
        if targetfile.exists():
            if not overwrite:
                logger.warning(f"文件 `{targetfile}` 已存在，不部署示例文件。")
                continue
            logger.warning(f"文件 `{targetfile}` 已存在，正在覆盖。")
        shutil.copy(str(sourcedir / source), str(targetfile))