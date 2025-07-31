import logging
import os
from typing import Any

import rapidjson

from freqtrade.constants import ENV_VAR_PREFIX
from freqtrade.misc import deep_merge_dicts


logger = logging.getLogger(__name__)


def _get_var_typed(val):
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            if val.lower() in ("t", "true"):
                return True
            elif val.lower() in ("f", "false"):
                return False
            # 尝试从json转换
            try:
                value = rapidjson.loads(val)
                # 目前仅支持列表
                if isinstance(value, list):
                    return value
            except rapidjson.JSONDecodeError:
                pass
    # 保持为字符串
    return val


def _flat_vars_to_nested_dict(env_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    """
    环境变量必须以FREQTRADE为前缀。
    格式为 FREQTRADE__{section}__{key}
    :param env_dict: 要验证的字典 - 通常是os.environ
    :param prefix: 要考虑的前缀（通常是FREQTRADE__）
    :return: 基于可用相关变量的嵌套字典。
    """
    no_convert = ["CHAT_ID", "PASSWORD"]
    ccxt_config_keys = ["ccxt_config", "ccxt_sync_config", "ccxt_async_config"]
    relevant_vars: dict[str, Any] = {}

    for env_var, val in sorted(env_dict.items()):
        if env_var.startswith(prefix):
            logger.info(f"加载变量 '{env_var}'")
            key = env_var.replace(prefix, "")
            key_parts = key.split("__")
            logger.info("键部分: %s", key_parts)

            # 检查任何ccxt配置键是否在键部分中
            preserve_case = key_parts[0].lower() == "exchange" and any(
                ccxt_key in [part.lower() for part in key_parts] for ccxt_key in ccxt_config_keys
            )

            for i, k in enumerate(reversed(key_parts)):
                # 如果涉及ccxt配置，保留最终键的大小写
                key_name = k if preserve_case and i == 0 else k.lower()

                val = {
                    key_name: (
                        _get_var_typed(val)
                        if not isinstance(val, dict) and k not in no_convert
                        else val
                    )
                }
            relevant_vars = deep_merge_dicts(val, relevant_vars)
    return relevant_vars


def enironment_vars_to_dict() -> dict[str, Any]:
    """
    读取环境变量并返回相关变量的嵌套字典
    相关变量必须遵循 FREQTRADE__{section}__{key} 模式
    :return: 基于可用相关变量的嵌套字典。
    """
    return _flat_vars_to_nested_dict(os.environ.copy(), ENV_VAR_PREFIX)