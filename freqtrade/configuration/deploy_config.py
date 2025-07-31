import logging
import secrets
from pathlib import Path
from typing import Any

from questionary import Separator, prompt

from freqtrade.constants import UNLIMITED_STAKE_AMOUNT
from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


def validate_is_int(val):
    try:
        _ = int(val)
        return True
    except Exception:
        return False


def validate_is_float(val):
    try:
        _ = float(val)
        return True
    except Exception:
        return False


def ask_user_overwrite(config_path: Path) -> bool:
    questions = [
        {
            "type": "confirm",
            "name": "overwrite",
            "message": f"文件 {config_path} 已存在。是否覆盖？",
            "default": False,
        },
    ]
    answers = prompt(questions)
    return answers["overwrite"]


def ask_user_config() -> dict[str, Any]:
    """
    向用户询问一些问题以构建配置。
    使用 https://github.com/tmbo/questionary 构建交互式问题
    :returns: 要放入模板的键值字典
    """

    from freqtrade.configuration.detect_environment import running_in_docker
    from freqtrade.exchange import available_exchanges

    questions: list[dict[str, Any]] = [
        {
            "type": "confirm",
            "name": "dry_run",
            "message": "是否要启用模拟交易（Dry-run）？",
            "default": True,
        },
        {
            "type": "text",
            "name": "stake_currency",
            "message": "请输入您的交易货币：",
            "default": "USDT",
        },
        {
            "type": "text",
            "name": "stake_amount",
            "message": f"请输入您的交易金额（数字或 '{UNLIMITED_STAKE_AMOUNT}' 表示无限制）：",
            "default": "unlimited",
            "validate": lambda val: val == UNLIMITED_STAKE_AMOUNT or validate_is_float(val),
            "filter": lambda val: (
                '"' + UNLIMITED_STAKE_AMOUNT + '"' if val == UNLIMITED_STAKE_AMOUNT else val
            ),
        },
        {
            "type": "text",
            "name": "max_open_trades",
            "message": "请输入最大开放交易数（整数，-1 表示无限制）：",
            "default": "3",
            "validate": lambda val: validate_is_int(val),
        },
        {
            "type": "select",
            "name": "timeframe_in_config",
            "message": "时间框架设置",
            "choices": ["由策略定义时间框架。", "在配置中覆盖设置。"],
        },
        {
            "type": "text",
            "name": "timeframe",
            "message": "请输入您想要的时间框架（例如 5m）：",
            "default": "5m",
            "when": lambda x: x["timeframe_in_config"] == "在配置中覆盖设置。",
        },
        {
            "type": "text",
            "name": "fiat_display_currency",
            "message": (
                "请输入报告中使用的法币显示货币 "
                "（留空以禁用法币转换）："
            ),
            "default": "USD",
        },
        {
            "type": "select",
            "name": "exchange_name",
            "message": "选择交易所",
            "choices": [
                "binance",
                "binanceus",
                "bingx",
                "gate",
                "htx",
                "kraken",
                "kucoin",
                "okx",
                Separator("------------------"),
                "其他",
            ],
        },
        {
            "type": "confirm",
            "name": "trading_mode",
            "message": "是否要交易永续合约（Perpetual Swaps）？",
            "default": False,
            "filter": lambda val: "futures" if val else "spot",
            "when": lambda x: x["exchange_name"] in ["binance", "gate", "okx", "bybit"],
        },
        {
            "type": "autocomplete",
            "name": "exchange_name",
            "message": "输入您的交易所名称（必须受ccxt支持）",
            "choices": available_exchanges(),
            "when": lambda x: x["exchange_name"] == "其他",
        },
        {
            "type": "password",
            "name": "exchange_key",
            "message": "输入交易所API Key",
            "when": lambda x: not x["dry_run"],
        },
        {
            "type": "password",
            "name": "exchange_secret",
            "message": "输入交易所API Secret",
            "when": lambda x: not x["dry_run"],
        },
        {
            "type": "password",
            "name": "exchange_key_password",
            "message": "输入交易所API Key密码",
            "when": lambda x: not x["dry_run"] and x["exchange_name"] in ("kucoin", "okx"),
        },
        {
            "type": "confirm",
            "name": "telegram",
            "message": "是否要启用Telegram通知？",
            "default": False,
        },
        {
            "type": "password",
            "name": "telegram_token",
            "message": "输入Telegram机器人令牌",
            "when": lambda x: x["telegram"],
        },
        {
            "type": "password",
            "name": "telegram_chat_id",
            "message": "输入Telegram聊天ID",
            "when": lambda x: x["telegram"],
        },
        {
            "type": "confirm",
            "name": "api_server",
            "message": "是否要启用REST API（包括FreqUI）？",
            "default": False,
        },
        {
            "type": "text",
            "name": "api_server_listen_addr",
            "message": (
                "输入API服务器监听地址（Docker使用0.0.0.0，"
                "其他情况建议保持默认）"
            ),
            "default": "127.0.0.1" if not running_in_docker() else "0.0.0.0",  # noqa: S104
            "when": lambda x: x["api_server"],
        },
        {
            "type": "text",
            "name": "api_server_username",
            "message": "输入API服务器用户名",
            "default": "freqtrader",
            "when": lambda x: x["api_server"],
        },
        {
            "type": "password",
            "name": "api_server_password",
            "message": "输入API服务器密码",
            "when": lambda x: x["api_server"],
        },
    ]
    answers = prompt(questions)

    if not answers:
        # 中断的问题会话返回空字典
        raise OperationalException("用户中断了交互式问题。")
    # 确保非合约交易所的默认值已设置
    answers["trading_mode"] = answers.get("trading_mode", "spot")
    answers["margin_mode"] = "isolated" if answers.get("trading_mode") == "futures" else ""
    # 强制JWT令牌为随机字符串
    answers["api_server_jwt_key"] = secrets.token_hex()
    answers["api_server_ws_token"] = secrets.token_urlsafe(25)

    return answers


def deploy_new_config(config_path: Path, selections: dict[str, Any]) -> None:
    """
    将选择应用到模板并将结果写入config_path
    :param config_path: 新配置文件的Path对象。不应已存在
    :param selections: 包含用户选择的字典
    """
    from jinja2.exceptions import TemplateNotFound

    from freqtrade.exchange import MAP_EXCHANGE_CHILDCLASS
    from freqtrade.util import render_template

    try:
        exchange_template = MAP_EXCHANGE_CHILDCLASS.get(
            selections["exchange_name"], selections["exchange_name"]
        )

        selections["exchange"] = render_template(
            templatefile=f"subtemplates/exchange_{exchange_template}.j2", arguments=selections
        )
    except TemplateNotFound:
        selections["exchange"] = render_template(
            templatefile="subtemplates/exchange_generic.j2", arguments=selections
        )

    config_text = render_template(templatefile="base_config.json.j2", arguments=selections)

    logger.info(f"将配置写入 `{config_path}`。")
    logger.info(
        "请务必检查配置内容并根据您的需求调整设置。"
    )

    config_path.write_text(config_text)