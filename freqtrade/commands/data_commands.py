import logging
import sys
from collections import defaultdict
from typing import Any

from freqtrade.constants import DATETIME_PRINT_FORMAT, DL_DATA_TIMEFRAMES, Config
from freqtrade.enums import CandleType, RunMode, TradingMode
from freqtrade.exceptions import ConfigurationError
from freqtrade.plugins.pairlist.pairlist_helpers import dynamic_expand_pairlist


logger = logging.getLogger(__name__)


def _check_data_config_download_sanity(config: Config) -> None:
    """检查数据下载配置的合理性"""
    if "days" in config and "timerange" in config:
        raise ConfigurationError(
            "--days 和 --timerange 是互斥的。您只能指定其中一个。"
        )

    if "pairs" not in config:
        raise ConfigurationError(
            "下载数据需要交易对列表。"
            "请查看文档了解如何配置。"
        )


def start_download_data(args: dict[str, Any]) -> None:
    """
    下载数据（原 download_backtest_data.py 脚本）
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.data.history import download_data_main

    config = setup_utils_configuration(args, RunMode.UTIL_EXCHANGE)

    _check_data_config_download_sanity(config)

    try:
        download_data_main(config)

    except KeyboardInterrupt:
        sys.exit("收到 SIGINT 信号，正在中止...")


def start_convert_trades(args: dict[str, Any]) -> None:
    """将交易数据转换为OHLCV数据"""
    from freqtrade.configuration import TimeRange, setup_utils_configuration
    from freqtrade.data.converter import convert_trades_to_ohlcv
    from freqtrade.resolvers import ExchangeResolver

    config = setup_utils_configuration(args, RunMode.UTIL_EXCHANGE)

    timerange = TimeRange()

    # 移除基础货币以跳过与数据下载无关的检查
    config["stake_currency"] = ""

    if "timeframes" not in config:
        config["timeframes"] = DL_DATA_TIMEFRAMES

    # 初始化交易所
    exchange = ExchangeResolver.load_exchange(config, validate=False)
    # 相关设置的手动验证

    for timeframe in config["timeframes"]:
        exchange.validate_timeframes(timeframe)
    available_pairs = [
        p
        for p in exchange.get_markets(
            tradable_only=True, active_only=not config.get("include_inactive")
        ).keys()
    ]

    expanded_pairs = dynamic_expand_pairlist(config, available_pairs)

    # 将下载的交易数据转换为不同的时间周期
    convert_trades_to_ohlcv(
        pairs=expanded_pairs,
        timeframes=config["timeframes"],
        datadir=config["datadir"],
        timerange=timerange,
        erase=bool(config.get("erase")),
        data_format_ohlcv=config["dataformat_ohlcv"],
        data_format_trades=config["dataformat_trades"],
        candle_type=config.get("candle_type_def", CandleType.SPOT),
    )


def start_convert_data(args: dict[str, Any], ohlcv: bool = True) -> None:
    """
    将数据从一种格式转换为另一种格式
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.data.converter import convert_ohlcv_format, convert_trades_format
    from freqtrade.util.migrations import migrate_data

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)
    if ohlcv:
        migrate_data(config)
        convert_ohlcv_format(
            config,
            convert_from=args["format_from"],
            convert_to=args["format_to"],
            erase=args["erase"],
        )
    else:
        convert_trades_format(
            config,
            convert_from=args["format_from_trades"],
            convert_to=args["format_to"],
            erase=args["erase"],
        )


def start_list_data(args: dict[str, Any]) -> None:
    """
    列出可用的OHLCV数据
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.exchange import timeframe_to_minutes
    from freqtrade.util import print_rich_table

    if args["trades"]:
        start_list_trades_data(args)
        return

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    from freqtrade.data.history import get_datahandler

    dhc = get_datahandler(config["datadir"], config["dataformat_ohlcv"])

    paircombs = dhc.ohlcv_get_available_data(
        config["datadir"], config.get("trading_mode", TradingMode.SPOT)
    )
    if args["pairs"]:
        paircombs = [comb for comb in paircombs if comb[0] in args["pairs"]]
    title = f"找到 {len(paircombs)} 个交易对/时间周期组合。"
    if not config.get("show_timerange"):
        groupedpair = defaultdict(list)
        for pair, timeframe, candle_type in sorted(
            paircombs, key=lambda x: (x[0], timeframe_to_minutes(x[1]), x[2])
        ):
            groupedpair[(pair, candle_type)].append(timeframe)

        if groupedpair:
            print_rich_table(
                [
                    (pair, ", ".join(timeframes), candle_type)
                    for (pair, candle_type), timeframes in groupedpair.items()
                ],
                ("交易对", "时间周期", "类型"),
                title,
                table_kwargs={"min_width": 50},
            )
    else:
        paircombs1 = [
            (pair, timeframe, candle_type, *dhc.ohlcv_data_min_max(pair, timeframe, candle_type))
            for pair, timeframe, candle_type in paircombs
        ]
        print_rich_table(
            [
                (
                    pair,
                    timeframe,
                    candle_type,
                    start.strftime(DATETIME_PRINT_FORMAT),
                    end.strftime(DATETIME_PRINT_FORMAT),
                    str(length),
                )
                for pair, timeframe, candle_type, start, end, length in sorted(
                    paircombs1, key=lambda x: (x[0], timeframe_to_minutes(x[1]), x[2])
                )
            ],
            ("交易对", "时间周期", "类型", "起始时间", "结束时间", "K线数量"),
            summary=title,
            table_kwargs={"min_width": 50},
        )


def start_list_trades_data(args: dict[str, Any]) -> None:
    """
    列出可用的交易数据
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.misc import plural
    from freqtrade.util import print_rich_table

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    from freqtrade.data.history import get_datahandler

    dhc = get_datahandler(config["datadir"], config["dataformat_trades"])

    paircombs = dhc.trades_get_available_data(
        config["datadir"], config.get("trading_mode", TradingMode.SPOT)
    )

    if args["pairs"]:
        paircombs = [comb for comb in paircombs if comb in args["pairs"]]

    title = f"找到 {len(paircombs)} 个{plural(len(paircombs), '交易对')}的交易数据。"
    if not config.get("show_timerange"):
        print_rich_table(
            [(pair, config.get("candle_type_def", CandleType.SPOT)) for pair in sorted(paircombs)],
            ("交易对", "类型"),
            title,
            table_kwargs={"min_width": 50},
        )
    else:
        paircombs1 = [
            (pair, *dhc.trades_data_min_max(pair, config.get("trading_mode", TradingMode.SPOT)))
            for pair in paircombs
        ]
        print_rich_table(
            [
                (
                    pair,
                    config.get("candle_type_def", CandleType.SPOT),
                    start.strftime(DATETIME_PRINT_FORMAT),
                    end.strftime(DATETIME_PRINT_FORMAT),
                    str(length),
                )
                for pair, start, end, length in sorted(paircombs1, key=lambda x: (x[0]))
            ],
            ("交易对", "类型", "起始时间", "结束时间", "交易数量"),
            summary=title,
            table_kwargs={"min_width": 50},
        )