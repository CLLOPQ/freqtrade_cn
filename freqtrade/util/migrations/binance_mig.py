import logging

from packaging import version
from sqlalchemy import select

from freqtrade.constants import DOCS_LINK, Config
from freqtrade.enums import TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.persistence import KeyValueStore, Trade
from freqtrade.persistence.pairlock import PairLock


logger = logging.getLogger(__name__)


def migrate_binance_futures_names(config: Config):
    if not (
        config.get("trading_mode", TradingMode.SPOT) == TradingMode.FUTURES
        and config["exchange"]["name"] == "binance"
    ):
        # 仅对新的期货交易生效
        return
    if KeyValueStore.get_int_value("binance_migration"):
        # 已迁移
        return
    import ccxt

    if version.parse("2.6.26") > version.parse(ccxt.__version__):
        raise OperationalException(
            "请按照文档中的更新说明操作 "
            f"({DOCS_LINK}/updating/) 以安装兼容的ccxt版本。"
        )
    _migrate_binance_futures_db(config)
    migrate_binance_futures_data(config)
    KeyValueStore.store_value("binance_migration", 1)


def _migrate_binance_futures_db(config: Config):
    logger.info("正在数据库中迁移币安期货交易对。")
    trades = Trade.get_trades([Trade.exchange == "binance", Trade.trading_mode == "FUTURES"]).all()
    for trade in trades:
        if ":" in trade.pair:
            # 已迁移
            continue
        new_pair = f"{trade.pair}:{trade.stake_currency}"
        trade.pair = new_pair

        for order in trade.orders:
            order.ft_pair = new_pair
            # 是否也需要迁移合约代码？
            # order.symbol = new_pair
    Trade.commit()
    pls = PairLock.session.scalars(select(PairLock).filter(PairLock.pair.notlike("%:%"))).all()
    for pl in pls:
        pl.pair = f"{pl.pair}:{config['stake_currency']}"
    # print(pls)
    # pls.update({'pair': concat(PairLock.pair,':USDT')})
    Trade.commit()
    logger.info("币安期货交易对数据库迁移完成。")


def migrate_binance_futures_data(config: Config):
    if not (
        config.get("trading_mode", TradingMode.SPOT) == TradingMode.FUTURES
        and config["exchange"]["name"] == "binance"
    ):
        # 仅对新的期货交易生效
        return

    from freqtrade.data.history import get_datahandler

    dhc = get_datahandler(config["datadir"], config["dataformat_ohlcv"])

    paircombs = dhc.ohlcv_get_available_data(
        config["datadir"], config.get("trading_mode", TradingMode.SPOT)
    )

    for pair, timeframe, candle_type in paircombs:
        if ":" in pair:
            # 已迁移
            continue
        new_pair = f"{pair}:{config['stake_currency']}"
        dhc.rename_futures_data(pair, new_pair, timeframe, candle_type)