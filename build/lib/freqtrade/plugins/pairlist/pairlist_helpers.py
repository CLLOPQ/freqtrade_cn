import re

from freqtrade.constants import Config


def expand_pairlist(
    wildcardpl: list[str], available_pairs: list[str], keep_invalid: bool = False
) -> list[str]:
    """
    基于可用市场扩展可能包含通配符的交易对列表。这将隐式过滤通配符列表中不在可用交易对中的所有交易对。
    :param wildcardpl: 交易对列表，可能包含正则表达式
    :param available_pairs: 所有可用交易对列表（`exchange.get_markets().keys()`）
    :param keep_invalid: 如果设为True，在扩展正则表达式时会静默丢弃无效交易对
    :return: 扩展后的交易对列表，其中应用了wildcardpl中的正则表达式以匹配所有可用交易对。
    :raises: 值错误：如果通配符无效（例如'*/BTC'——应改为`.*/BTC`）
    """
    result = []
    if keep_invalid:
        for pair_wc in wildcardpl:
            try:
                comp = re.compile(pair_wc, re.IGNORECASE)
                result_partial = [pair for pair in available_pairs if re.fullmatch(comp, pair)]
                # 添加所有匹配的交易对。如果没有匹配的交易对（交易对不在交易所中），则保留该通配符。
                result += result_partial or [pair_wc]
            except re.error as err:
                raise ValueError(f"Wildcard error in {pair_wc}, {err}")

        # 移除没有匹配项的通配符交易对。
        result = [element for element in result if re.fullmatch(r"^[A-Za-z0-9:/-]+$", element)]

    else:
        for pair_wc in wildcardpl:
            try:
                comp = re.compile(pair_wc, re.IGNORECASE)
                result += [pair for pair in available_pairs if re.fullmatch(comp, pair)]
            except re.error as err:
                raise ValueError(f"Wildcard error in {pair_wc}, {err}")
    return result


def dynamic_expand_pairlist(config: Config, markets: list[str]) -> list[str]:
    expanded_pairs = expand_pairlist(config["pairs"], markets)
    if config.get("freqai", {}).get("enabled", False):
        corr_pairlist = config["freqai"]["feature_parameters"]["include_corr_pairlist"]
        expanded_pairs += [pair for pair in corr_pairlist if pair not in config["pairs"]]

    return expanded_pairs