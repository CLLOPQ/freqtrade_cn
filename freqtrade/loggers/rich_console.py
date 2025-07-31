import sys
from shutil import get_terminal_size

from rich.console import Console


def console_width() -> int | None:
    """
    获取控制台宽度
    """
    if any(module in ["pytest", "ipykernel"] for module in sys.modules):
        return 200

    width, _ = get_terminal_size((1, 24))
    # 如果无法获取终端大小，则回退到200
    # 通过假设一个不合理的1字符宽度来判断，这种情况不太可能发生
    w = None if width > 1 else 200
    return w


def get_rich_console(**kwargs) -> Console:
    """
    获取具有默认设置的rich控制台
    """
    kwargs["width"] = kwargs.get("width", console_width())
    return Console(** kwargs)