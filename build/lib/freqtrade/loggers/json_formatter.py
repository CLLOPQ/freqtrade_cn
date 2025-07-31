import json
import logging


class JsonFormatter(logging.Formatter):
    """
    解析 LogRecord 后输出 JSON 字符串的格式化器。

    @param dict fmt_dict: 键值对为日志格式属性。默认为 {"message": "message"}。
    @param str time_format: time.strftime() 格式字符串。默认："%Y-%m-%dT%H:%M:%S"
    @param str msec_format: 毫秒格式。附加在末尾。默认："%s.%03dZ"
    """

    def __init__(
        self,
        fmt_dict: dict | None = None,
        time_format: str = "%Y-%m-%dT%H:%M:%S",
        msec_format: str = "%s.%03dZ",
    ):
        self.fmt_dict = (
            fmt_dict
            if fmt_dict is not None
            else {
                "timestamp": "asctime",
                "level": "levelname",
                "logger": "name",
                "message": "message",
            }
        )
        self.default_time_format = time_format
        self.default_msec_format = msec_format
        self.datefmt = None

    def usesTime(self) -> bool:
        """
        在格式字典值中查找属性，而不是在格式字符串中。
        """
        return "asctime" in self.fmt_dict.values()

    def formatMessage(self, record) -> str:
        raise NotImplementedError()

    def formatMessageDict(self, record) -> dict:
        """
        返回相关 LogRecord 属性的字典而不是字符串。
        如果在 fmt_dict 中提供了未知属性，则会引发 KeyError。
        """
        return {fmt_key: record.__dict__[fmt_val] for fmt_key, fmt_val in self.fmt_dict.items()}

    def format(self, record) -> str:
        """
        与父类的方法大致相同，不同之处在于操作字典并将其序列化为 JSON 而不是字符串。
        """
        record.message = record.getMessage()

        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)

        message_dict = self.formatMessageDict(record)

        if record.exc_info:
            # 缓存回溯文本以避免多次转换
            # （反正它是不变的）
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)

        if record.exc_text:
            message_dict["exc_info"] = record.exc_text

        if record.stack_info:
            message_dict["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(message_dict, default=str)