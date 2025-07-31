from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pandas import DataFrame

from freqtrade.enums import CandleType
from freqtrade.exceptions import OperationalException
from freqtrade.strategy.strategy_helper import merge_informative_pair


PopulateIndicators = Callable[[Any, DataFrame, dict], DataFrame]


@dataclass
class InformativeData:
    """用于为装饰器存储信息性数据的数据类。"""
    asset: str | None
    timeframe: str
    fmt: str | Callable[[Any], str] | None
    ffill: bool
    candle_type: CandleType | None


def informative(
    timeframe: str,
    asset: str = "",
    fmt: str | Callable[[Any], str] | None = None,
    *,
    candle_type: CandleType | str | None = None,
    ffill: bool = True,
) -> Callable[[PopulateIndicators], PopulateIndicators]:
    """
    用于populate_indicators_Nn(self, dataframe, metadata)的装饰器，允许这些函数定义信息性指标。

    使用示例：

        @informative('1h')
        def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
            return dataframe

    :param timeframe: 信息性时间框架。必须始终等于或高于策略时间框架。
    :param asset: 信息性资产，例如 BTC、BTC/USDT、ETH/BTC。不指定则使用当前交易对。也支持有限的交易对格式字符串（见下文）
    :param fmt: 列格式（字符串）或列格式化器（可调用对象(name, asset, timeframe)）。未指定时，默认为：
    * 如果指定了资产：{base}_{quote}_{column}_{timeframe}
    * 如果未指定资产：{column}_{timeframe}
    交易对格式支持以下变量：
    * {base} - 基础货币，小写，例如 'eth'。
    * {BASE} - 同{base}，但大写。
    * {quote} - 计价货币，小写，例如 'usdt'。
    * {QUOTE} - 同{quote}，但大写。
    格式字符串还支持以下变量。
    * {asset} - 资产全名，例如 'BTC/USDT'。
    * {column} - 数据框列名。
    * {timeframe} - 信息性数据框的时间框架。
    :param ffill: 合并信息性交易对后对数据框进行前向填充。
    :param candle_type:  '', mark, index, premiumIndex, 或 funding_rate
    """
    _asset = asset
    _timeframe = timeframe
    _fmt = fmt
    _ffill = ffill
    _candle_type = CandleType.from_string(candle_type) if candle_type else None

    def decorator(fn: PopulateIndicators):
        informative_pairs = getattr(fn, "_ft_informative", [])
        informative_pairs.append(InformativeData(_asset, _timeframe, _fmt, _ffill, _candle_type))
        setattr(fn, "_ft_informative", informative_pairs)  # noqa: B010
        return fn

    return decorator


def __get_pair_formats(market: dict[str, Any] | None) -> dict[str, str]:
    """返回交易对格式变量的字典。"""
    if not market:
        return {}
    base = market["base"]
    quote = market["quote"]
    return {
        "base": base.lower(),
        "BASE": base.upper(),
        "quote": quote.lower(),
        "QUOTE": quote.upper(),
    }


def _format_pair_name(config, pair: str, market: dict[str, Any] | None = None) -> str:
    return pair.format(
        stake_currency=config["stake_currency"],
        stake=config["stake_currency"],
        **__get_pair_formats(market),
    ).upper()


def _create_and_merge_informative_pair(
    strategy,
    dataframe: DataFrame,
    metadata: dict,
    inf_data: InformativeData,
    populate_indicators: PopulateIndicators,
):
    """创建信息性数据框并合并到主数据框中。"""
    asset = inf_data.asset or ""
    timeframe = inf_data.timeframe
    fmt = inf_data.fmt
    candle_type = inf_data.candle_type

    config = strategy.config

    if asset:
        # 如果需要，插入计价货币。
        market1 = strategy.dp.market(metadata["pair"])
        asset = _format_pair_name(config, asset, market1)
    else:
        # 不指定资产将为当前交易对定义信息性数据框。
        asset = metadata["pair"]

    market = strategy.dp.market(asset)
    if market is None:
        raise OperationalException(f"Market {asset} is not available.")

    # 默认格式。这优化了常见场景：使用相同计价货币的信息性交易对。当计价货币与策略计价货币匹配时，列名将省略基础货币。
    # 这允许轻松重新配置策略以使用不同的基础货币。在极少数情况下
    # 如果希望始终在列名中保留计价货币，用户应指定
    # fmt='{base}_{quote}_{column}_{timeframe}' 格式或类似格式。
    if not fmt:
        fmt = "{column}_{timeframe}"  # 当前交易对的信息性数据
        if inf_data.asset:
            fmt = "{base}_{quote}_" + fmt  # 其他交易对的信息性数据

    inf_metadata = {"pair": asset, "timeframe": timeframe}
    inf_dataframe = strategy.dp.get_pair_dataframe(asset, timeframe, candle_type)
    inf_dataframe = populate_indicators(strategy, inf_dataframe, inf_metadata)

    formatter: Any = None
    if callable(fmt):
        formatter = fmt  # 自定义用户指定的格式化器函数。
    else:
        formatter = fmt.format  # 默认字符串格式化器。

    fmt_args = {
        **__get_pair_formats(market),
        "asset": asset,
        "timeframe": timeframe,
    }
    inf_dataframe.rename(columns=lambda column: formatter(column=column, **fmt_args), inplace=True)

    date_column = formatter(column="date", **fmt_args)
    if date_column in dataframe.columns:
        raise OperationalException(
            f"数据框中已存在重复列名 {date_column}！确保列名唯一！"
        )
    dataframe = merge_informative_pair(
        dataframe,
        inf_dataframe,
        strategy.timeframe,
        timeframe,
        ffill=inf_data.ffill,
        append_timeframe=False,
        date_column=date_column,
    )
    return dataframe