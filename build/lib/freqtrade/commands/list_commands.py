import csv
import logging
import sys
import json
from typing import Any

from freqtrade.enums import RunMode
from freqtrade.exceptions import ConfigurationError, OperationalException


logger = logging.getLogger(__name__)


def start_list_exchanges(args: dict[str, Any]) -> None:
    """
    打印可用的交易所
    :param args: 来自Arguments()的命令行参数
    :return: None
    """
    from rich.table import Table
    from rich.text import Text

    from freqtrade.exchange import list_available_exchanges
    from freqtrade.ft_types import ValidExchangesType
    from freqtrade.loggers.rich_console import get_rich_console

    available_exchanges: list[ValidExchangesType] = list_available_exchanges(
        args["list_exchanges_all"]
    )

    if args["print_one_column"]:
        print("\n".join([e["classname"] for e in available_exchanges]))
    else:
        if args["list_exchanges_all"]:
            title = (
                f"ccxt库支持的所有交易所 "
                f"({len(available_exchanges)} 个交易所):"
            )
        else:
            available_exchanges = [e for e in available_exchanges if e["valid"] is not False]
            title = f"Freqtrade可用的交易所 ({len(available_exchanges)} 个交易所):"

        table = Table(title=title)

        table.add_column("交易所名称")
        table.add_column("类名")
        table.add_column("市场类型")
        table.add_column("说明")

        for exchange in available_exchanges:
            name = Text(exchange["name"])
            if exchange["supported"]:
                name.append(" (支持)", style="italic")
                name.stylize("green bold")
            classname = Text(exchange["classname"])
            if exchange["is_alias"]:
                name.stylize("strike")
                classname.stylize("strike")
                classname.append(f" (使用 {exchange['alias_for']})", style="italic")

            trade_modes = Text(
                ", ".join(
                    (f"{a.get('margin_mode', '')} {a['trading_mode']}").lstrip()
                    for a in exchange["trade_modes"]
                ),
                style="",
            )
            if exchange["dex"]:
                trade_modes = Text("DEX: ") + trade_modes
                trade_modes.stylize("bold", 0, 3)

            table.add_row(
                name,
                classname,
                trade_modes,
                exchange["comment"],
                style=None if exchange["valid"] else "red",
            )

        console = get_rich_console()
        console.print(table)


def _print_objs_tabular(objs: list, print_colorized: bool) -> None:
    from rich.table import Table
    from rich.text import Text

    from freqtrade.loggers.rich_console import get_rich_console

    names = [s["name"] for s in objs]
    objs_to_print: list[dict[str, Text | str]] = [
        {
            "name": Text(s["name"] if s["name"] else "--"),
            "location": s["location_rel"],
            "status": (
                Text("加载失败", style="bold red")
                if s["class"] is None
                else Text("正常", style="bold green")
                if names.count(s["name"]) == 1
                else Text("名称重复", style="bold yellow")
            ),
        }
        for s in objs
    ]
    for idx, s in enumerate(objs):
        if "hyperoptable" in s:
            objs_to_print[idx].update(
                {
                    "hyperoptable": "是" if s["hyperoptable"]["count"] > 0 else "否",
                    "买入参数": str(len(s["hyperoptable"].get("buy", []))),
                    "卖出参数": str(len(s["hyperoptable"].get("sell", []))),
                }
            )
    table = Table()

    for header in objs_to_print[0].keys():
        table.add_column(header.capitalize(), justify="right")

    for row in objs_to_print:
        table.add_row(*[row[header] for header in objs_to_print[0].keys()])

    console = get_rich_console(color_system="auto" if print_colorized else None)
    console.print(table)


def start_list_strategies(args: dict[str, Any]) -> None:
    """
    打印目录中可用的策略自定义类文件
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.resolvers import StrategyResolver

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    strategy_objs = StrategyResolver.search_all_objects(
        config, not args["print_one_column"], config.get("recursive_strategy_search", False)
    )
    # 按字母顺序排序
    strategy_objs = sorted(strategy_objs, key=lambda x: x["name"])
    for obj in strategy_objs:
        if obj["class"]:
            obj["hyperoptable"] = obj["class"].detect_all_parameters()
        else:
            obj["hyperoptable"] = {"count": 0}

    if args["print_one_column"]:
        print("\n".join([s["name"] for s in strategy_objs]))
    else:
        _print_objs_tabular(strategy_objs, config.get("print_colorized", False))


def start_list_freqAI_models(args: dict[str, Any]) -> None:
    """
    打印目录中可用的FreqAI模型自定义类文件
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.resolvers.freqaimodel_resolver import FreqaiModelResolver

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    model_objs = FreqaiModelResolver.search_all_objects(config, not args["print_one_column"])
    # 按字母顺序排序
    model_objs = sorted(model_objs, key=lambda x: x["name"])
    if args["print_one_column"]:
        print("\n".join([s["name"] for s in model_objs]))
    else:
        _print_objs_tabular(model_objs, config.get("print_colorized", False))


def start_list_hyperopt_loss_functions(args: dict[str, Any]) -> None:
    """
    打印目录中可用的超参数优化损失函数类文件
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.resolvers.hyperopt_resolver import HyperOptLossResolver

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    model_objs = HyperOptLossResolver.search_all_objects(config, not args["print_one_column"])
    # 按字母顺序排序
    model_objs = sorted(model_objs, key=lambda x: x["name"])
    if args["print_one_column"]:
        print("\n".join([s["name"] for s in model_objs]))
    else:
        _print_objs_tabular(model_objs, config.get("print_colorized", False))


def start_list_timeframes(args: dict[str, Any]) -> None:
    """
    打印交易所可用的时间周期
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.resolvers import ExchangeResolver

    config = setup_utils_configuration(args, RunMode.UTIL_EXCHANGE)
    # 不使用配置中设置的时间周期
    config["timeframe"] = None

    # 初始化交易所
    exchange = ExchangeResolver.load_exchange(config, validate=False)

    if args["print_one_column"]:
        print("\n".join(exchange.timeframes))
    else:
        print(
            f"交易所 `{exchange.name}` 可用的时间周期: "
            f"{', '.join(exchange.timeframes)}"
        )


def start_list_markets(args: dict[str, Any], pairs_only: bool = False) -> None:
    """
    打印交易所上的交易对/市场
    :param args: 来自Arguments()的命令行参数
    :param pairs_only: 如果为True，只打印交易对，否则打印所有工具(市场)
    :return: None
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.exchange import market_is_active
    from freqtrade.misc import plural, safe_value_fallback
    from freqtrade.resolvers import ExchangeResolver
    from freqtrade.util import print_rich_table

    config = setup_utils_configuration(args, RunMode.UTIL_EXCHANGE)

    # 初始化交易所
    exchange = ExchangeResolver.load_exchange(config, validate=False)

    # 默认只显示活跃的交易对/市场
    active_only = not args.get("list_pairs_all", False)

    base_currencies = args.get("base_currencies", [])
    quote_currencies = args.get("quote_currencies", [])

    try:
        pairs = exchange.get_markets(
            base_currencies=base_currencies,
            quote_currencies=quote_currencies,
            tradable_only=pairs_only,
            active_only=active_only,
        )
        # 按符号排序交易对/市场
        pairs = dict(sorted(pairs.items()))
    except Exception as e:
        raise OperationalException(f"无法获取市场数据。原因: {e}") from e

    tickers = exchange.get_tickers()

    summary_str = (
        (f"交易所 {exchange.name} 有 {len(pairs)} 个")
        + ("活跃的 " if active_only else "")
        + (plural(len(pairs), "交易对" if pairs_only else "市场"))
        + (
            f"，以 {', '.join(base_currencies)} 为基础"
            f"{plural(len(base_currencies), '货币', '货币')}"
            if base_currencies
            else ""
        )
        + (" 和" if base_currencies and quote_currencies else "")
        + (
            f"，以 {', '.join(quote_currencies)} 为计价"
            f"{plural(len(quote_currencies), '货币', '货币')}"
            if quote_currencies
            else ""
        )
    )

    headers = [
        "ID",
        "交易对",
        "基础货币",
        "计价货币",
        "活跃",
        "现货",
        "保证金",
        "期货",
        "杠杆",
        "最小持仓",
    ]

    tabular_data = [
        {
            "ID": v["id"],
            "交易对": v["symbol"],
            "基础货币": v["base"],
            "计价货币": v["quote"],
            "活跃": market_is_active(v),
            "现货": "现货" if exchange.market_is_spot(v) else "",
            "保证金": "保证金" if exchange.market_is_margin(v) else "",
            "期货": "期货" if exchange.market_is_future(v) else "",
            "杠杆": exchange.get_max_leverage(v["symbol"], 20),
            "最小持仓": round(
                exchange.get_min_pair_stake_amount(
                    v["symbol"],
                    safe_value_fallback(tickers.get(v["symbol"], {}), "last", "ask", 0.0),
                    0.0,
                )
                or 0.0,
                8,
            ),
        }
        for _, v in pairs.items()
    ]

    if (
        args.get("print_one_column", False)
        or args.get("list_pairs_print_json", False)
        or args.get("print_csv", False)
    ):
        # 对于机器可读的常规格式，在日志中打印摘要字符串
        logger.info(f"{summary_str}。")
    else:
        # 对于人类可读的格式，打印空字符串分隔前导日志和输出
        print()

    if pairs:
        if args.get("print_list", False):
            # 以列表形式打印数据，并附带人类可读的摘要
            print(f"{summary_str}: {', '.join(pairs.keys())}。")
        elif args.get("print_one_column", False):
            print("\n".join(pairs.keys()))
        elif args.get("list_pairs_print_json", False):
            import rapidjson

            print(rapidjson.dumps(list(pairs.keys()), default=str))
        elif args.get("print_csv", False):
            writer = csv.DictWriter(sys.stdout, fieldnames=headers)
            writer.writeheader()
            writer.writerows(tabular_data)
        else:
            print_rich_table(tabular_data, headers, summary_str)
    elif not (
        args.get("print_one_column", False)
        or args.get("list_pairs_print_json", False)
        or args.get("print_csv", False)
    ):
        print(f"{summary_str}。")


def start_show_trades(args: dict[str, Any]) -> None:
    """
    显示交易记录
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.misc import parse_db_uri_for_logging
    from freqtrade.persistence import Trade, init_db

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    if "db_url" not in config:
        raise ConfigurationError("此命令需要 --db-url 参数。")

    logger.info(f'使用数据库: "{parse_db_uri_for_logging(config["db_url"])}"')
    init_db(config["db_url"])
    tfilter = []

    if config.get("trade_ids"):
        tfilter.append(Trade.id.in_(config["trade_ids"]))

    trades = Trade.get_trades(tfilter).all()
    logger.info(f"显示 {len(trades)} 笔交易: ")
    if config.get("print_json", False):
        print(json.dumps([trade.to_json() for trade in trades], indent=4))
    else:
        for trade in trades:
            print(trade)