from datetime import datetime
from logging import Handler

from rich._null_file import NullFile
from rich.console import Console
from rich.text import Text


class FtRichHandler(Handler):
    """
    使用 Rich 的基本彩色日志处理器。
    不支持标准日志处理器的所有功能，并且使用硬编码的日志格式
    """

    def __init__(self, console: Console, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._console = console

    def emit(self, record):
        try:
            msg = self.format(record)
            # 格式化日志消息
            log_time = Text(
                datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
                if record.created
                else "N/A",
            )
            name = Text(record.name, style="violet")
            log_level = Text(record.levelname, style=f"logging.level.{record.levelname.lower()}")
            gray_sep = Text(" - ", style="gray46")

            if isinstance(self._console.file, NullFile):
                # 处理 pythonw 情况，此时 stdout/stderr 为空，我们从 Console.file 返回 NullFile 实例
                # 在这种情况下，即使我们不会向文件写入任何内容，我们仍然希望生成日志记录
                self.handleError(record)
                return

            self._console.print(
                Text() + log_time + gray_sep + name + gray_sep + log_level + gray_sep + msg
            )

        except RecursionError:
            raise
        except ImportError:
            # 关闭控制台时出错...
            pass
        except Exception:
            self.handleError(record)