from freqtrade.enums.exittype import ExitType


class ExitCheckTuple:
    """
    用于存储出场类型和原因的元组类
    """

    exit_type: ExitType
    exit_reason: str = ""

    def __init__(self, exit_type: ExitType, exit_reason: str = ""):
        self.exit_type = exit_type
        self.exit_reason = exit_reason or exit_type.value

    @property
    def exit_flag(self):
        """返回是否需要出场的标志"""
        return self.exit_type != ExitType.NONE

    def __eq__(self, other):
        """判断两个ExitCheckTuple是否相等"""
        return self.exit_type == other.exit_type and self.exit_reason == other.exit_reason

    def __repr__(self):
        """返回对象的字符串表示"""
        return f"ExitCheckTuple({self.exit_type}, {self.exit_reason})"