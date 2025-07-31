from logging.handlers import BufferingHandler


class FTBufferingHandler(BufferingHandler):
    def flush(self):
        """
        重写刷新行为 - 我们保留配置容量的一半
        否则，会出现"空"日志的时刻。
        """
        self.acquire()
        try:
            # 在缓冲区中保留一半的记录
            records_to_keep = -int(self.capacity / 2)
            self.buffer = self.buffer[records_to_keep:]
        finally:
            self.release()