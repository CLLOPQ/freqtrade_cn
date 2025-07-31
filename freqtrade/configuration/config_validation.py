import logging
from collections import Counter
from copy import deepcopy
from typing import Any

from jsonschema import Draft4Validator, validators
from jsonschema.exceptions import ValidationError, best_match

from freqtrade.config_schema.config_schema import (
    CONF_SCHEMA,
    SCHEMA_BACKTEST_REQUIRED,
    SCHEMA_BACKTEST_REQUIRED_FINAL,
    SCHEMA_MINIMAL_REQUIRED,
    SCHEMA_MINIMAL_WEBSERVER,
    SCHEMA_TRADE_REQUIRED,
)
from freqtrade.configuration.deprecated_settings import process_deprecated_setting
from freqtrade.constants import UNLIMITED_STAKE_AMOUNT
from freqtrade.enums import RunMode, TradingMode
from freqtrade.exceptions import ConfigurationError


logger = logging.getLogger(__name__)


def _extend_validator(validator_class):
    """
    为Freqtrade配置JSON模式扩展验证器。
    目前仅处理子模式的默认值。
    """
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for prop, subschema in properties.items():
            if "default" in subschema:
                instance.setdefault(prop, subschema["default"])

        yield from validate_properties(validator, properties, instance, schema)

    return validators.extend(validator_class, {"properties": set_defaults})


FreqtradeValidator = _extend_validator(Draft4Validator)


def validate_config_schema(conf: dict[str, Any], preliminary: bool = False) -> dict[str, Any]:
    """
    验证配置是否符合配置模式
    :param conf: JSON格式的配置
    :return: 如果有效则返回配置，否则抛出异常
    """
    conf_schema = deepcopy(CONF_SCHEMA)
    if conf.get("runmode", RunMode.OTHER) in (RunMode.DRY_RUN, RunMode.LIVE):
        conf_schema["required"] = SCHEMA_TRADE_REQUIRED
    elif conf.get("runmode", RunMode.OTHER) in (RunMode.BACKTEST, RunMode.HYPEROPT):
        if preliminary:
            conf_schema["required"] = SCHEMA_BACKTEST_REQUIRED
        else:
            conf_schema["required"] = SCHEMA_BACKTEST_REQUIRED_FINAL
    elif conf.get("runmode", RunMode.OTHER) == RunMode.WEBSERVER:
        conf_schema["required"] = SCHEMA_MINIMAL_WEBSERVER
    else:
        conf_schema["required"] = SCHEMA_MINIMAL_REQUIRED
    try:
        FreqtradeValidator(conf_schema).validate(conf)
        return conf
    except ValidationError as e:
        logger.critical(f"配置无效。原因：{e}")
        raise ValidationError(best_match(Draft4Validator(conf_schema).iter_errors(conf)).message)


def validate_config_consistency(conf: dict[str, Any], *, preliminary: bool = False) -> None:
    """
    验证配置的一致性。
    应在加载配置和策略之后运行，
    因为策略也可能设置某些配置项。
    :param conf: JSON格式的配置
    :return: 如果一切正常则返回None，否则抛出ConfigurationError
    """

    # 验证追踪止损
    _validate_trailing_stoploss(conf)
    _validate_price_config(conf)
    _validate_edge(conf)
    _validate_whitelist(conf)
    _validate_unlimited_amount(conf)
    _validate_ask_orderbook(conf)
    _validate_freqai_hyperopt(conf)
    _validate_freqai_backtest(conf)
    _validate_freqai_include_timeframes(conf, preliminary=preliminary)
    _validate_consumers(conf)
    validate_migrated_strategy_settings(conf)
    _validate_orderflow(conf)

    # 返回前验证配置
    logger.info("正在验证配置...")
    validate_config_schema(conf, preliminary=preliminary)


def _validate_unlimited_amount(conf: dict[str, Any]) -> None:
    """
    必须设置max_open_trades或stake_amount中的一个。
    :raise: 如果配置验证失败则抛出ConfigurationError
    """
    if (
        conf.get("max_open_trades") == float("inf") or conf.get("max_open_trades") == -1
    ) and conf.get("stake_amount") == UNLIMITED_STAKE_AMOUNT:
        raise ConfigurationError("`max_open_trades`和`stake_amount`不能同时设置为无限制。")


def _validate_price_config(conf: dict[str, Any]) -> None:
    """
    使用市价单时，价格方向必须使用价格的"另一侧"
    """
    # TODO: 以下可以是使用市价单时的强制设置
    if conf.get("order_types", {}).get("entry") == "market" and conf.get("entry_pricing", {}).get(
        "price_side"
    ) not in ("ask", "other"):
        raise ConfigurationError('市价入场订单要求entry_pricing.price_side = "other"。')

    if conf.get("order_types", {}).get("exit") == "market" and conf.get("exit_pricing", {}).get(
        "price_side"
    ) not in ("bid", "other"):
        raise ConfigurationError('市价出场订单要求exit_pricing.price_side = "other"。')


def _validate_trailing_stoploss(conf: dict[str, Any]) -> None:
    if conf.get("stoploss") == 0.0:
        raise ConfigurationError(
            "配置中的止损必须不同于0，以避免卖单出现问题。"
        )
    # 如果未激活追踪止损则跳过
    if not conf.get("trailing_stop", False):
        return

    tsl_positive = float(conf.get("trailing_stop_positive", 0))
    tsl_offset = float(conf.get("trailing_stop_positive_offset", 0))
    tsl_only_offset = conf.get("trailing_only_offset_is_reached", False)

    if tsl_only_offset:
        if tsl_positive == 0.0:
            raise ConfigurationError(
                "配置中的trailing_only_offset_is_reached需要"
                "配置中的trailing_stop_positive_offset大于0。"
            )
    if tsl_positive > 0 and 0 < tsl_offset <= tsl_positive:
        raise ConfigurationError(
            "配置中的trailing_stop_positive_offset需要"
            "大于配置中的trailing_stop_positive。"
        )

    # 再次获取，不使用默认值
    if "trailing_stop_positive" in conf and float(conf["trailing_stop_positive"]) == 0.0:
        raise ConfigurationError(
            "配置中的trailing_stop_positive必须不同于0"
            "以避免卖单出现问题。"
        )


def _validate_edge(conf: dict[str, Any]) -> None:
    """
    Edge和动态白名单不应同时启用，因为edge会覆盖动态白名单。
    """

    if conf.get("edge", {}).get("enabled"):
        raise ConfigurationError(
            "Edge已不再受支持，并已在Freqtrade 2025.6版本中移除。"
        )


def _validate_whitelist(conf: dict[str, Any]) -> None:
    """
    动态白名单不需要设置pair_whitelist，但StaticWhitelist需要。
    """
    if conf.get("runmode", RunMode.OTHER) in [
        RunMode.OTHER,
        RunMode.PLOT,
        RunMode.UTIL_NO_EXCHANGE,
        RunMode.UTIL_EXCHANGE,
    ]:
        return

    for pl in conf.get("pairlists", [{"method": "StaticPairList"}]):
        if (
            isinstance(pl, dict)
            and pl.get("method") == "StaticPairList"
            and not conf.get("exchange", {}).get("pair_whitelist")
        ):
            raise ConfigurationError("StaticPairList需要设置pair_whitelist。")


def _validate_ask_orderbook(conf: dict[str, Any]) -> None:
    ask_strategy = conf.get("exit_pricing", {})
    ob_min = ask_strategy.get("order_book_min")
    ob_max = ask_strategy.get("order_book_max")
    if ob_min is not None and ob_max is not None and ask_strategy.get("use_order_book"):
        if ob_min != ob_max:
            raise ConfigurationError(
                "在exit_pricing中使用order_book_max != order_book_min已不再支持。"
                "请选择一个值，并在未来使用`order_book_top`。"
            )
        else:
            # 将值移至order_book_top
            ask_strategy["order_book_top"] = ob_min
            logger.warning(
                "已过时："
                "请在`exit_pricing`配置中使用`order_book_top`代替`order_book_min`和`order_book_max`。"
            )


def validate_migrated_strategy_settings(conf: dict[str, Any]) -> None:
    _validate_time_in_force(conf)
    _validate_order_types(conf)
    _validate_unfilledtimeout(conf)
    _validate_pricing_rules(conf)
    _strategy_settings(conf)


def _validate_time_in_force(conf: dict[str, Any]) -> None:
    time_in_force = conf.get("order_time_in_force", {})
    if "buy" in time_in_force or "sell" in time_in_force:
        if conf.get("trading_mode", TradingMode.SPOT) != TradingMode.SPOT:
            raise ConfigurationError(
                "请将您的time_in_force设置迁移为使用'entry'和'exit'。"
            )
        else:
            logger.warning(
                "已过时：使用'buy'和'sell'作为time_in_force已过时。"
                "请将您的time_in_force设置迁移为使用'entry'和'exit'。"
            )
            process_deprecated_setting(
                conf, "order_time_in_force", "buy", "order_time_in_force", "entry"
            )

            process_deprecated_setting(
                conf, "order_time_in_force", "sell", "order_time_in_force", "exit"
            )


def _validate_order_types(conf: dict[str, Any]) -> None:
    order_types = conf.get("order_types", {})
    old_order_types = [
        "buy",
        "sell",
        "emergencysell",
        "forcebuy",
        "forcesell",
        "emergencyexit",
        "forceexit",
        "forceentry",
    ]
    if any(x in order_types for x in old_order_types):
        if conf.get("trading_mode", TradingMode.SPOT) != TradingMode.SPOT:
            raise ConfigurationError(
                "请将您的order_types设置迁移为使用新的命名方式。"
            )
        else:
            logger.warning(
                "已过时：使用'buy'和'sell'作为order_types已过时。"
                "请将您的order_types设置迁移为使用'entry'和'exit'命名方式。"
            )
            for o, n in [
                ("buy", "entry"),
                ("sell", "exit"),
                ("emergencysell", "emergency_exit"),
                ("forcesell", "force_exit"),
                ("forcebuy", "force_entry"),
                ("emergencyexit", "emergency_exit"),
                ("forceexit", "force_exit"),
                ("forceentry", "force_entry"),
            ]:
                process_deprecated_setting(conf, "order_types", o, "order_types", n)


def _validate_unfilledtimeout(conf: dict[str, Any]) -> None:
    unfilledtimeout = conf.get("unfilledtimeout", {})
    if any(x in unfilledtimeout for x in ["buy", "sell"]):
        if conf.get("trading_mode", TradingMode.SPOT) != TradingMode.SPOT:
            raise ConfigurationError(
                "请将您的unfilledtimeout设置迁移为使用新的命名方式。"
            )
        else:
            logger.warning(
                "已过时：使用'buy'和'sell'作为unfilledtimeout已过时。"
                "请将您的unfilledtimeout设置迁移为使用'entry'和'exit'命名方式。"
            )
            for o, n in [
                ("buy", "entry"),
                ("sell", "exit"),
            ]:
                process_deprecated_setting(conf, "unfilledtimeout", o, "unfilledtimeout", n)


def _validate_pricing_rules(conf: dict[str, Any]) -> None:
    if conf.get("ask_strategy") or conf.get("bid_strategy"):
        if conf.get("trading_mode", TradingMode.SPOT) != TradingMode.SPOT:
            raise ConfigurationError("请将您的定价设置迁移为使用新的命名方式。")
        else:
            logger.warning(
                "已过时：使用'ask_strategy'和'bid_strategy'已过时。"
                "请将您的设置迁移为使用'entry_pricing'和'exit_pricing'。"
            )
            conf["entry_pricing"] = {}
            for obj in list(conf.get("bid_strategy", {}).keys()):
                if obj == "ask_last_balance":
                    process_deprecated_setting(
                        conf, "bid_strategy", obj, "entry_pricing", "price_last_balance"
                    )
                else:
                    process_deprecated_setting(conf, "bid_strategy", obj, "entry_pricing", obj)
            del conf["bid_strategy"]

            conf["exit_pricing"] = {}
            for obj in list(conf.get("ask_strategy", {}).keys()):
                if obj == "bid_last_balance":
                    process_deprecated_setting(
                        conf, "ask_strategy", obj, "exit_pricing", "price_last_balance"
                    )
                else:
                    process_deprecated_setting(conf, "ask_strategy", obj, "exit_pricing", obj)
            del conf["ask_strategy"]


def _validate_freqai_hyperopt(conf: dict[str, Any]) -> None:
    freqai_enabled = conf.get("freqai", {}).get("enabled", False)
    analyze_per_epoch = conf.get("analyze_per_epoch", False)
    if analyze_per_epoch and freqai_enabled:
        raise ConfigurationError(
            "使用analyze-per-epoch参数不支持FreqAI策略。"
        )


def _validate_freqai_include_timeframes(conf: dict[str, Any], preliminary: bool) -> None:
    freqai_enabled = conf.get("freqai", {}).get("enabled", False)
    if freqai_enabled:
        main_tf = conf.get("timeframe", "5m")
        freqai_include_timeframes = (
            conf.get("freqai", {}).get("feature_parameters", {}).get("include_timeframes", [])
        )

        from freqtrade.exchange import timeframe_to_seconds

        main_tf_s = timeframe_to_seconds(main_tf)
        offending_lines = []
        for tf in freqai_include_timeframes:
            tf_s = timeframe_to_seconds(tf)
            if tf_s < main_tf_s:
                offending_lines.append(tf)
        if offending_lines:
            raise ConfigurationError(
                f"主时间框架{main_tf}必须小于或等于FreqAI的"
                f"`include_timeframes`。违规的包含时间框架：{', '.join(offending_lines)}"
            )

        # 确保基础时间框架包含在include_timeframes列表中
        if not preliminary and main_tf not in freqai_include_timeframes:
            feature_parameters = conf.get("freqai", {}).get("feature_parameters", {})
            include_timeframes = [main_tf, *freqai_include_timeframes]
            conf.get("freqai", {}).get("feature_parameters", {}).update(
                {**feature_parameters, "include_timeframes": include_timeframes}
            )


def _validate_freqai_backtest(conf: dict[str, Any]) -> None:
    if conf.get("runmode", RunMode.OTHER) == RunMode.BACKTEST:
        freqai_enabled = conf.get("freqai", {}).get("enabled", False)
        timerange = conf.get("timerange")
        freqai_backtest_live_models = conf.get("freqai_backtest_live_models", False)
        if freqai_backtest_live_models and freqai_enabled and timerange:
            raise ConfigurationError(
                "使用timerange参数不支持"
                "--freqai-backtest-live-models参数。"
            )

        if freqai_backtest_live_models and not freqai_enabled:
            raise ConfigurationError(
                "使用--freqai-backtest-live-models参数仅"
                "支持FreqAI策略。"
            )

        if freqai_enabled and not freqai_backtest_live_models and not timerange:
            raise ConfigurationError(
                "如果您打算使用FreqAI进行回测，请传递--timerange参数。"
            )


def _validate_consumers(conf: dict[str, Any]) -> None:
    emc_conf = conf.get("external_message_consumer", {})
    if emc_conf.get("enabled", False):
        if len(emc_conf.get("producers", [])) < 1:
            raise ConfigurationError("您必须指定至少一个要连接的生产者。")

        producer_names = [p["name"] for p in emc_conf.get("producers", [])]
        duplicates = [item for item, count in Counter(producer_names).items() if count > 1]
        if duplicates:
            raise ConfigurationError(
                f"生产者名称必须唯一。重复项：{', '.join(duplicates)}"
            )
        if conf.get("process_only_new_candles", True):
            # 这里是警告还是要求？
            logger.warning(
                "为获得外部数据的最佳性能，"
                "请将`process_only_new_candles`设置为False"
            )


def _validate_orderflow(conf: dict[str, Any]) -> None:
    if conf.get("exchange", {}).get("use_public_trades"):
        if "orderflow" not in conf:
            raise ConfigurationError(
                "当使用公共交易数据时，orderflow是必需的配置项。"
            )


def _strategy_settings(conf: dict[str, Any]) -> None:
    process_deprecated_setting(conf, None, "use_sell_signal", None, "use_exit_signal")
    process_deprecated_setting(conf, None, "sell_profit_only", None, "exit_profit_only")
    process_deprecated_setting(conf, None, "sell_profit_offset", None, "exit_profit_offset")
    process_deprecated_setting(
        conf, None, "ignore_roi_if_buy_signal", None, "ignore_roi_if_entry_signal"
    )