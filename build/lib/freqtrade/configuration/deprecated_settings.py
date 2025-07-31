"""
处理已弃用设置的函数
"""

import logging

from freqtrade.constants import Config
from freqtrade.exceptions import ConfigurationError, OperationalException


logger = logging.getLogger(__name__)


def check_conflicting_settings(
    config: Config,
    section_old: str | None,
    name_old: str,
    section_new: str | None,
    name_new: str,
) -> None:
    section_new_config = config.get(section_new, {}) if section_new else config
    section_old_config = config.get(section_old, {}) if section_old else config
    if name_new in section_new_config and name_old in section_old_config:
        new_name = f"{section_new}.{name_new}" if section_new else f"{name_new}"
        old_name = f"{section_old}.{name_old}" if section_old else f"{name_old}"
        raise OperationalException(
            f"配置文件中检测到冲突的设置 `{new_name}` 和 `{old_name}`（已弃用）。"
            "这个已弃用的设置将在Freqtrade的下一个版本中移除。"
            f"请从您的配置中删除它，并改用 `{new_name}` 设置。"
        )


def process_removed_setting(
    config: Config, section1: str, name1: str, section2: str | None, name2: str
) -> None:
    """
    :param section1: 已移除的部分
    :param name1: 已移除的设置名称
    :param section2: 此键的新部分
    :param name2: 新的设置名称
    """
    section1_config = config.get(section1, {})
    if name1 in section1_config:
        section_2 = f"{section2}.{name2}" if section2 else f"{name2}"
        raise ConfigurationError(
            f"设置 `{section1}.{name1}` 已移至 `{section_2}`。"
            f"请从您的配置中删除它，并改用 `{section_2}` 设置。"
        )


def process_deprecated_setting(
    config: Config,
    section_old: str | None,
    name_old: str,
    section_new: str | None,
    name_new: str,
) -> None:
    check_conflicting_settings(config, section_old, name_old, section_new, name_new)
    section_old_config = config.get(section_old, {}) if section_old else config

    if name_old in section_old_config:
        section_1 = f"{section_old}.{name_old}" if section_old else f"{name_old}"
        section_2 = f"{section_new}.{name_new}" if section_new else f"{name_new}"
        logger.warning(
            "已弃用："
            f"`{section_1}` 设置已弃用，"
            "将在Freqtrade的下一个版本中移除。"
            f"请在您的配置中改用 `{section_2}` 设置。"
        )

        section_new_config = config.get(section_new, {}) if section_new else config
        section_new_config[name_new] = section_old_config[name_old]
        del section_old_config[name_old]


def process_temporary_deprecated_settings(config: Config) -> None:
    # 为未来的已弃用/移动的设置保留
    # check_conflicting_settings(config, 'ask_strategy', 'use_sell_signal',
    #                            'experimental', 'use_sell_signal')

    process_deprecated_setting(
        config,
        "ask_strategy",
        "ignore_buying_expired_candle_after",
        None,
        "ignore_buying_expired_candle_after",
    )

    process_deprecated_setting(config, None, "forcebuy_enable", None, "force_entry_enable")

    # 新设置
    if config.get("telegram"):
        process_deprecated_setting(
            config["telegram"], "notification_settings", "sell", "notification_settings", "exit"
        )
        process_deprecated_setting(
            config["telegram"],
            "notification_settings",
            "sell_fill",
            "notification_settings",
            "exit_fill",
        )
        process_deprecated_setting(
            config["telegram"],
            "notification_settings",
            "sell_cancel",
            "notification_settings",
            "exit_cancel",
        )
        process_deprecated_setting(
            config["telegram"], "notification_settings", "buy", "notification_settings", "entry"
        )
        process_deprecated_setting(
            config["telegram"],
            "notification_settings",
            "buy_fill",
            "notification_settings",
            "entry_fill",
        )
        process_deprecated_setting(
            config["telegram"],
            "notification_settings",
            "buy_cancel",
            "notification_settings",
            "entry_cancel",
        )
    if config.get("webhook"):
        process_deprecated_setting(config, "webhook", "webhookbuy", "webhook", "webhookentry")
        process_deprecated_setting(
            config, "webhook", "webhookbuycancel", "webhook", "webhookentrycancel"
        )
        process_deprecated_setting(
            config, "webhook", "webhookbuyfill", "webhook", "webhookentryfill"
        )
        process_deprecated_setting(config, "webhook", "webhooksell", "webhook", "webhookexit")
        process_deprecated_setting(
            config, "webhook", "webhooksellcancel", "webhook", "webhookexitcancel"
        )
        process_deprecated_setting(
            config, "webhook", "webhooksellfill", "webhook", "webhookexitfill"
        )

    # 旧方式 - 将它们放在experimental中...

    process_removed_setting(config, "experimental", "use_sell_signal", None, "use_exit_signal")
    process_removed_setting(config, "experimental", "sell_profit_only", None, "exit_profit_only")
    process_removed_setting(
        config, "experimental", "ignore_roi_if_buy_signal", None, "ignore_roi_if_entry_signal"
    )

    process_removed_setting(config, "ask_strategy", "use_sell_signal", None, "use_exit_signal")
    process_removed_setting(config, "ask_strategy", "sell_profit_only", None, "exit_profit_only")
    process_removed_setting(
        config, "ask_strategy", "sell_profit_offset", None, "exit_profit_offset"
    )
    process_removed_setting(
        config, "ask_strategy", "ignore_roi_if_buy_signal", None, "ignore_roi_if_entry_signal"
    )
    if "ticker_interval" in config:
        raise ConfigurationError(
            "已弃用：检测到 'ticker_interval'。"
            "请使用 'timeframe' 代替 'ticker_interval'。"
        )

    if "protections" in config:
        raise ConfigurationError(
            "已弃用：在配置中设置 'protections' 已过时。"
        )