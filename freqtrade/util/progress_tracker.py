from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from freqtrade.util.rich_progress import CustomProgress


def retrieve_progress_tracker(pt: CustomProgress | None) -> CustomProgress:
    """
    获取进度跟踪器。
    如果传入的跟踪器为None，则返回默认的进度跟踪器。
    """
    if pt is None:
        return get_progress_tracker()
    return pt


def get_progress_tracker(**kwargs) -> CustomProgress:
    """
    获取带有自定义列的进度条。
    """
    from freqtrade.loggers import error_console

    return CustomProgress(
        TextColumn("[进度描述]{任务描述}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
        expand=True,
        console=error_console,** kwargs,
    )