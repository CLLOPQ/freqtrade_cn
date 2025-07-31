from copy import deepcopy

from freqtrade.constants import Config, ExchangeConfig


_SENSITIVE_KEYS = [
    "exchange.key",
    "exchange.api_key",
    "exchange.apiKey",
    "exchange.secret",
    "exchange.password",
    "exchange.uid",
    "exchange.account_id",
    "exchange.accountId",
    "exchange.wallet_address",
    "exchange.walletAddress",
    "exchange.private_key",
    "exchange.privateKey",
    "telegram.token",
    "telegram.chat_id",
    "discord.webhook_url",
    "api_server.password",
    "webhook.url",
]


def sanitize_config(config: Config, *, show_sensitive: bool = False) -> Config:
    """
    从配置中移除敏感信息。
    :param config: 配置信息
    :param show_sensitive: 是否显示敏感信息
    :return: 处理后的配置信息
    """
    if show_sensitive:
        return config
    config = deepcopy(config)
    for key in _SENSITIVE_KEYS:
        if "." in key:
            nested_keys = key.split(".")
            nested_config = config
            for nested_key in nested_keys[:-1]:
                nested_config = nested_config.get(nested_key, {})
            if nested_keys[-1] in nested_config:
                nested_config[nested_keys[-1]] = "已屏蔽"
        else:
            if key in config:
                config[key] = "已屏蔽"

    return config


def remove_exchange_credentials(exchange_config: ExchangeConfig, dry_run: bool) -> None:
    """
    从配置中移除交易所密钥并指定模拟运行模式
    用于回测/超参数优化和工具。
    会修改输入的字典！
    :param exchange_config: 交易所配置
    :param dry_run: 如果为True，从交易所配置中移除敏感密钥
    """
    if not dry_run:
        return

    for key in [k for k in _SENSITIVE_KEYS if k.startswith("exchange.")]:
        if "." in key:
            key1 = key.removeprefix("exchange.")
            if key1 in exchange_config:
                exchange_config[key1] = ""