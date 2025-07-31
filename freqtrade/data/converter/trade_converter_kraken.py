import logging
from pathlib import Path

import pandas as pd

from freqtrade.constants import DATETIME_PRINT_FORMAT, DEFAULT_TRADES_COLUMNS, Config
from freqtrade.data.converter.trade_converter import (
    trades_convert_types,
    trades_df_remove_duplicates,
)
from freqtrade.data.history import get_datahandler
from freqtrade.enums import TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.plugins.pairlist.pairlist_helpers import expand_pairlist
from freqtrade.resolvers import ExchangeResolver


logger = logging.getLogger(__name__)

KRAKEN_CSV_TRADE_COLUMNS = ["timestamp", "price", "amount"]


def import_kraken_trades_from_csv(config: Config, convert_to: str):
    """
    从 CSV 导入 Kraken 交易数据
    """
    if config["exchange"]["name"] != "kraken":
        raise OperationalException("此函数仅适用于 Kraken 交易所。")

    datadir: Path = config["datadir"]
    data_handler = get_datahandler(datadir, data_format=convert_to)

    tradesdir: Path = config["datadir"] / "trades_csv"
    exchange = ExchangeResolver.load_exchange(config, validate=False)
    # 遍历此目录中的文件
    data_symbols = {p.stem for p in tradesdir.rglob("*.csv")}

    # 创建交易对/文件名映射
    markets = {
        (m["symbol"], m["altname"])
        for m in exchange.markets.values()
        if m.get("altname") in data_symbols
    }
    logger.info(f"找到以下交易对的 CSV 文件：{', '.join(data_symbols)}。")

    if pairs_raw := config.get("pairs"):
        pairs = expand_pairlist(pairs_raw, [m[0] for m in markets])
        markets = {m for m in markets if m[0] in pairs}
        if not markets:
            logger.info(f"未找到 {', '.join(pairs_raw)} 交易对的数据。")
            return
    logger.info(f"正在转换交易对：{', '.join(m[0] for m in markets)}。")

    for pair, name in markets:
        logger.debug(f"正在转换交易对 {pair}，文件 */{name}.csv")
        dfs = []
        # 加载并合并此交易对的所有 CSV 文件
        for f in tradesdir.rglob(f"{name}.csv"):
            df = pd.read_csv(f, names=KRAKEN_CSV_TRADE_COLUMNS)
            if not df.empty:
                dfs.append(df)

        # 加载现有交易数据
        if not dfs:
            # 边缘情况，仅当在上述 glob 操作和此处之间文件被删除时才会发生
            logger.info(f"未找到交易对 {pair} 的数据")
            continue

        trades = pd.concat(dfs, ignore_index=True)
        del dfs

        # 删除 timestamp 列中不是数字的行
        timestamp_numeric = pd.to_numeric(trades["timestamp"], errors="coerce")
        trades = trades[timestamp_numeric.notna()]

        trades.loc[:, "timestamp"] = trades["timestamp"] * 1e3
        trades.loc[:, "cost"] = trades["price"] * trades["amount"]
        for col in DEFAULT_TRADES_COLUMNS:
            if col not in trades.columns:
                trades.loc[:, col] = ""
        trades = trades[DEFAULT_TRADES_COLUMNS]
        trades = trades_convert_types(trades)

        trades_df = trades_df_remove_duplicates(trades)
        del trades
        logger.info(
            f"{pair}：{len(trades_df)} 笔交易，时间范围从 "
            f"{trades_df['date'].min():{DATETIME_PRINT_FORMAT}} 到 "
            f"{trades_df['date'].max():{DATETIME_PRINT_FORMAT}}"
        )

        data_handler.trades_store(pair, trades_df, TradingMode.SPOT)