import sys
from logging import Handler


class FTStdErrStreamHandler(Handler):
    def flush(self):
        """
        重写刷新行为 - 我们保持配置容量的一半
        否则，会出现"空"日志的时刻。
        """
        self.acquire()
        try:
            sys.stderr.flush()
        finally:
            self.release()

    def emit(self, record):
        try:
            msg = self.format(record)
            # 不保留对stderr的引用 - 这在使用进度条时可能会有问题
            sys.stderr.write(msg + "\n")
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)