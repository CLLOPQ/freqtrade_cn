import logging

from freqtrade.constants import Config
from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import available_exchanges, is_exchange_known_ccxt, validate_exchange
from freqtrade.exchange.common import MAP_EXCHANGE_CHILDCLASS, SUPPORTED_EXCHANGES


logger = logging.getLogger(__name__)


def check_exchange(config: Config, check_for_bad: bool = True) -> bool:
    """
    检查配置文件中的交易所名称是否受Freqtrade支持
    :param check_for_bad: 如果为True，检查交易所是否在已知的"问题"交易所列表中
    :return: 如果交易所是"问题"交易所（已知存在严重问题或完全无法使用、会导致崩溃等），则返回False。否则返回True。
             如果交易所不受ccxt支持，因此不为Freqtrade所知，则引发异常
    """

    if config["runmode"] in [
        RunMode.PLOT,
        RunMode.UTIL_NO_EXCHANGE,
        RunMode.OTHER,
    ] and not config.get("exchange", {}).get("name"):
        # 在绘图模式下跳过检查交易所，因为它不需要交易所
        return True
    logger.info("正在检查交易所...")

    exchange = config.get("exchange", {}).get("name", "").lower()
    if not exchange:
        raise OperationalException(
            "此命令需要配置交易所。您应该使用`--exchange <交易所名称>`或通过`--config`指定配置文件。\n"
            f"Freqtrade支持以下交易所: {', '.join(available_exchanges())}"
        )

    if not is_exchange_known_ccxt(exchange):
        raise OperationalException(
            f'交易所 "{exchange}" 不为ccxt库所知，因此无法用于机器人。\n'
            f"Freqtrade支持以下交易所: {', '.join(available_exchanges())}"
        )

    valid, reason, _ = validate_exchange(exchange)
    if not valid:
        if check_for_bad:
            raise OperationalException(
                f'交易所 "{exchange}" 无法与Freqtrade一起使用。原因: {reason}'
            )
        else:
            logger.warning(f'交易所 "{exchange}" 无法与Freqtrade一起使用。原因: {reason}')

    if MAP_EXCHANGE_CHILDCLASS.get(exchange, exchange) in SUPPORTED_EXCHANGES:
        logger.info(
            f'交易所 "{exchange}" 得到Freqtrade开发团队的官方支持。'
        )
    else:
        logger.warning(
            f'交易所 "{exchange}" 为ccxt库所知，可用于机器人，但未得到'
            f"Freqtrade开发团队的官方支持。"
            f"它可能完美运行（请反馈）或存在严重问题。"
            f"请自行决定是否使用。"
        )

    return True