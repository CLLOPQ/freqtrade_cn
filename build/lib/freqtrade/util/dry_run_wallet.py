from freqtrade.constants import Config


def get_dry_run_wallet(config: Config) -> int | float:
    """
    从配置中返回以交易货币计算的模拟交易钱包余额。
    此设置还支持模拟交易钱包的字典模式。
    """
    if isinstance(_start_cap := config["dry_run_wallet"], float | int):
        return _start_cap
    else:
        return _start_cap.get("交易货币")