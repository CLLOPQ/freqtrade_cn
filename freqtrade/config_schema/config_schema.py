from typing import Any

from freqtrade.constants import (
    AVAILABLE_DATAHANDLERS,
    AVAILABLE_PAIRLISTS,
    BACKTEST_BREAKDOWNS,
    DRY_RUN_WALLET,
    EXPORT_OPTIONS,
    MARGIN_MODES,
    ORDERTIF_POSSIBILITIES,
    ORDERTYPE_POSSIBILITIES,
    PRICING_SIDES,
    REQUIRED_ORDERTIF,
    STOPLOSS_PRICE_TYPES,
    SUPPORTED_FIAT,
    TELEGRAM_SETTING_OPTIONS,
    TIMEOUT_UNITS,
    TRADING_MODES,
    UNLIMITED_STAKE_AMOUNT,
    WEBHOOK_FORMAT_OPTIONS,
)
from freqtrade.enums import RPCMessageType


__MESSAGE_TYPE_DICT: dict[str, dict[str, str]] = {x: {"type": "object"} for x in RPCMessageType}

__IN_STRATEGY = "\n通常在策略中指定，而在配置中缺失。"

__VIA_ENV = "建议通过环境变量设置"

CONF_SCHEMA = {
    "type": "object",
    "properties": {
        "max_open_trades": {
            "description": "最大开仓交易数量。-1表示无限制。",
            "type": ["integer", "number"],
            "minimum": -1,
        },
        "timeframe": {
            "description": (
                f"使用的时间周期（例如 `1m`、`5m`、`15m`、`30m`、`1h` 等）。{__IN_STRATEGY}"
            ),
            "type": "string",
        },
        "proxy_coin": {
            "description": "代理货币 - 特定期货模式下必须使用（例如 BNFCR）",
            "type": "string",
        },
        "stake_currency": {
            "description": "用于质押的货币。",
            "type": "string",
        },
        "stake_amount": {
            "description": "每笔交易的质押金额。",
            "type": ["number", "string"],
            "minimum": 0.0001,
            "pattern": UNLIMITED_STAKE_AMOUNT,
        },
        "tradable_balance_ratio": {
            "description": "可交易的余额比例。",
            "type": "number",
            "minimum": 0.0,
            "maximum": 1,
            "default": 0.99,
        },
        "available_capital": {
            "description": "可用于交易的总资金。",
            "type": "number",
            "minimum": 0,
        },
        "amend_last_stake_amount": {
            "description": "是否修正最后一笔质押金额。",
            "type": "boolean",
            "default": False,
        },
        "last_stake_amount_min_ratio": {
            "description": "最后一笔质押金额的最小比例。",
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.5,
        },
        "fiat_display_currency": {
            "description": "用于显示的法定货币。",
            "type": "string",
            "enum": SUPPORTED_FIAT,
        },
        "dry_run": {
            "description": "启用或禁用模拟交易模式。",
            "type": "boolean",
        },
        "dry_run_wallet": {
            "description": "模拟交易模式下的初始钱包余额。",
            "type": ["number", "object"],
            "default": DRY_RUN_WALLET,
            "patternProperties": {r"^[a-zA-Z0-9]+$": {"type": "number"}},
            "additionalProperties": False,
        },
        "cancel_open_orders_on_exit": {
            "description": "退出时取消未成交订单。",
            "type": "boolean",
            "default": False,
        },
        "process_only_new_candles": {
            "description": "仅处理新的K线。",
            "type": "boolean",
        },
        "minimal_roi": {
            "description": f"最小投资回报率。{__IN_STRATEGY}",
            "type": "object",
            "patternProperties": {"^[0-9.]+$": {"type": "number"}},
        },
        "amount_reserve_percent": {
            "description": "要保留的金额百分比。",
            "type": "number",
            "minimum": 0.0,
            "maximum": 0.5,
        },
        "stoploss": {
            "description": f"用作止损值的值（比率形式）。{__IN_STRATEGY}",
            "type": "number",
            "maximum": 0,
            "exclusiveMaximum": True,
        },
        "trailing_stop": {
            "description": f"启用或禁用追踪止损。{__IN_STRATEGY}",
            "type": "boolean",
        },
        "trailing_stop_positive": {
            "description": f"追踪止损的正向偏移量。{__IN_STRATEGY}",
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "trailing_stop_positive_offset": {
            "description": f"激活追踪止损的偏移量。{__IN_STRATEGY}",
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "trailing_only_offset_is_reached": {
            "description": f"仅当达到偏移量时使用追踪止损。{__IN_STRATEGY}",
            "type": "boolean",
        },
        "use_exit_signal": {
            "description": f"对交易使用退出信号。{__IN_STRATEGY}",
            "type": "boolean",
        },
        "exit_profit_only": {
            "description": (
                "仅在盈利时退出。只要利润 < exit_profit_offset，退出信号将被忽略。"
                f"{__IN_STRATEGY}"
            ),
            "type": "boolean",
        },
        "exit_profit_offset": {
            "description": f"盈利退出的偏移量。{__IN_STRATEGY}",
            "type": "number",
        },
        "fee": {
            "description": "交易手续费百分比。可用于在回测中模拟滑点",
            "type": "number",
            "minimum": 0,
            "maximum": 0.1,
        },
        "ignore_roi_if_entry_signal": {
            "description": f"如果存在入场信号，则忽略ROI。{__IN_STRATEGY}",
            "type": "boolean",
        },
        "ignore_buying_expired_candle_after": {
            "description": f"蜡烛过期后忽略买入。{__IN_STRATEGY}",
            "type": "number",
        },
        "trading_mode": {
            "description": "交易模式（例如现货、保证金）。",
            "type": "string",
            "enum": TRADING_MODES,
        },
        "margin_mode": {
            "description": "交易的保证金模式。",
            "type": "string",
            "enum": MARGIN_MODES,
        },
        "reduce_df_footprint": {
            "description": "通过将列转换为float32/int32来减少DataFrame占用空间。",
            "type": "boolean",
            "default": False,
        },
        # 前瞻分析部分
        "minimum_trade_amount": {
            "description": "交易的最小金额 - 仅用于前瞻分析",
            "type": "number",
            "default": 10,
        },
        "targeted_trade_amount": {
            "description": "前瞻分析的目标交易金额。",
            "type": "number",
            "default": 20,
        },
        "lookahead_analysis_exportfilename": {
            "description": "前瞻分析导出的csv文件名。",
            "type": "string",
        },
        "startup_candle": {
            "description": "启动蜡烛配置。",
            "type": "array",
            "uniqueItems": True,
            "default": [199, 399, 499, 999, 1999],
        },
        "liquidation_buffer": {
            "description": "清算的缓冲比率。",
            "type": "number",
            "minimum": 0.0,
            "maximum": 0.99,
        },
        "backtest_breakdown": {
            "description": "回测的细分配置。",
            "type": "array",
            "items": {"type": "string", "enum": BACKTEST_BREAKDOWNS},
        },
        "bot_name": {
            "description": "交易机器人的名称。通过API传递给客户端。",
            "type": "string",
        },
        "unfilledtimeout": {
            "description": f"未成交订单的超时配置。{__IN_STRATEGY}",
            "type": "object",
            "properties": {
                "entry": {
                    "description": "入场订单的超时时间（单位）。",
                    "type": "number",
                    "minimum": 1,
                },
                "exit": {
                    "description": "出场订单的超时时间（单位）。",
                    "type": "number",
                    "minimum": 1,
                },
                "exit_timeout_count": {
                    "description": "放弃前重试出场订单的次数。",
                    "type": "number",
                    "minimum": 0,
                    "default": 0,
                },
                "unit": {
                    "description": "超时的时间单位（例如秒、分钟）。",
                    "type": "string",
                    "enum": TIMEOUT_UNITS,
                    "default": "minutes",
                },
            },
        },
        "entry_pricing": {
            "description": "入场定价配置。",
            "type": "object",
            "properties": {
                "price_last_balance": {
                    "description": "最新价格的平衡比率。",
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "exclusiveMaximum": False,
                },
                "price_side": {
                    "description": "使用的价格方向（例如买一、卖一、相同）。",
                    "type": "string",
                    "enum": PRICING_SIDES,
                    "default": "same",
                },
                "use_order_book": {
                    "description": "是否使用订单簿进行定价。",
                    "type": "boolean",
                },
                "order_book_top": {
                    "description": "要考虑的订单簿前N个层级。",
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                },
                "check_depth_of_market": {
                    "description": "检查市场深度的配置。",
                    "type": "object",
                    "properties": {
                        "enabled": {
                            "description": "启用或禁用市场深度检查。",
                            "type": "boolean",
                        },
                        "bids_to_ask_delta": {
                            "description": "要考虑的买卖价差。",
                            "type": "number",
                            "minimum": 0,
                        },
                    },
                },
            },
            "required": ["price_side"],
        },
        "exit_pricing": {
            "description": "出场定价配置。",
            "type": "object",
            "properties": {
                "price_side": {
                    "description": "使用的价格方向（例如买一、卖一、相同）。",
                    "type": "string",
                    "enum": PRICING_SIDES,
                    "default": "same",
                },
                "price_last_balance": {
                    "description": "最新价格的平衡比率。",
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "exclusiveMaximum": False,
                },
                "use_order_book": {
                    "description": "是否使用订单簿进行定价。",
                    "type": "boolean",
                },
                "order_book_top": {
                    "description": "要考虑的订单簿前N个层级。",
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                },
            },
            "required": ["price_side"],
        },
        "custom_price_max_distance_ratio": {
            "description": "当前价格与自定义入场或出场价格之间的最大距离比率。",
            "type": "number",
            "minimum": 0.0,
            "maximum": 1,
            "default": 0.02,
        },
        "order_types": {
            "description": f"订单类型配置。{__IN_STRATEGY}",
            "type": "object",
            "properties": {
                "entry": {
                    "description": "入场订单类型（例如限价单、市价单）。",
                    "type": "string",
                    "enum": ORDERTYPE_POSSIBILITIES,
                },
                "exit": {
                    "description": "出场订单类型（例如限价单、市价单）。",
                    "type": "string",
                    "enum": ORDERTYPE_POSSIBILITIES,
                },
                "force_exit": {
                    "description": "强制出场订单类型（例如限价单、市价单）。",
                    "type": "string",
                    "enum": ORDERTYPE_POSSIBILITIES,
                },
                "force_entry": {
                    "description": "强制入场订单类型（例如限价单、市价单）。",
                    "type": "string",
                    "enum": ORDERTYPE_POSSIBILITIES,
                },
                "emergency_exit": {
                    "description": "紧急出场订单类型（例如限价单、市价单）。",
                    "type": "string",
                    "enum": ORDERTYPE_POSSIBILITIES,
                    "default": "market",
                },
                "stoploss": {
                    "description": "止损订单类型（例如限价单、市价单）。",
                    "type": "string",
                    "enum": ORDERTYPE_POSSIBILITIES,
                },
                "stoploss_on_exchange": {
                    "description": "是否在交易所放置止损单。",
                    "type": "boolean",
                },
                "stoploss_price_type": {
                    "description": "止损的价格类型（例如最新价、标记价、指数价）。",
                    "type": "string",
                    "enum": STOPLOSS_PRICE_TYPES,
                },
                "stoploss_on_exchange_interval": {
                    "description": "交易所止损的间隔时间（秒）。",
                    "type": "number",
                },
                "stoploss_on_exchange_limit_ratio": {
                    "description": "交易所止损的限价比率。",
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            },
            "required": ["entry", "exit", "stoploss", "stoploss_on_exchange"],
        },
        "order_time_in_force": {
            "description": f"订单的有效时间配置。{__IN_STRATEGY}",
            "type": "object",
            "properties": {
                "entry": {
                    "description": "入场订单的有效时间。",
                    "type": "string",
                    "enum": ORDERTIF_POSSIBILITIES,
                },
                "exit": {
                    "description": "出场订单的有效时间。",
                    "type": "string",
                    "enum": ORDERTIF_POSSIBILITIES,
                },
            },
            "required": REQUIRED_ORDERTIF,
        },
        "coingecko": {
            "description": "CoinGecko API配置。",
            "type": "object",
            "properties": {
                "is_demo": {
                    "description": "是否在演示模式下使用CoinGecko。",
                    "type": "boolean",
                    "default": True,
                },
                "api_key": {"description": "访问CoinGecko的API密钥。", "type": "string"},
            },
            "required": ["is_demo", "api_key"],
        },
        "exchange": {
            "description": "交易所配置。",
            "$ref": "#/definitions/exchange",
        },
        "log_config": {
            "description": "日志配置。",
            "$ref": "#/definitions/logging",
        },
        "freqai": {
            "description": "FreqAI配置。",
            "$ref": "#/definitions/freqai",
        },
        "external_message_consumer": {
            "description": "外部消息消费者配置。",
            "$ref": "#/definitions/external_message_consumer",
        },
        "experimental": {
            "description": "实验性配置。",
            "type": "object",
            "properties": {"block_bad_exchanges": {"type": "boolean"}},
        },
        "pairlists": {
            "description": "交易对列表配置。",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "method": {
                        "description": "用于生成交易对列表的方法。",
                        "type": "string",
                        "enum": AVAILABLE_PAIRLISTS,
                    },
                },
                "required": ["method"],
            },
        },
        # RPC部分
        "telegram": {
            "description": "Telegram设置。",
            "type": "object",
            "properties": {
                "enabled": {
                    "description": "启用Telegram通知。",
                    "type": "boolean",
                },
                "token": {"description": "Telegram机器人令牌。", "type": "string"},
                "chat_id": {
                    "description": (
                        f"Telegram聊天或群组ID。{__VIA_ENV} FREQTRADE__TELEGRAM__CHAT_ID"
                    ),
                    "type": "string",
                },
                "topic_id": {
                    "description": (
                        "Telegram主题ID - 仅适用于群组聊天。"
                        f"{__VIA_ENV} FREQTRADE__TELEGRAM__TOPIC_ID"
                    ),
                    "type": "string",
                },
                "authorized_users": {
                    "description": "机器人的授权用户。",
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
                },
                "allow_custom_messages": {
                    "description": "允许从策略发送自定义消息。",
                    "type": "boolean",
                    "default": True,
                },
                "balance_dust_level": {
                    "description": "视为灰尘的最小余额水平。",
                    "type": "number",
                    "minimum": 0.0,
                },
                "notification_settings": {
                    "description": "不同类型通知的设置。",
                    "type": "object",
                    "default": {},
                    "properties": {
                        "status": {
                            "description": "状态更新的Telegram设置。",
                            "type": "string",
                            "enum": TELEGRAM_SETTING_OPTIONS,
                        },
                        "warning": {
                            "description": "警告的Telegram设置。",
                            "type": "string",
                            "enum": TELEGRAM_SETTING_OPTIONS,
                        },
                        "startup": {
                            "description": "启动消息的Telegram设置。",
                            "type": "string",
                            "enum": TELEGRAM_SETTING_OPTIONS,
                        },
                        "entry": {
                            "description": "入场信号的Telegram设置。",
                            "type": "string",
                            "enum": TELEGRAM_SETTING_OPTIONS,
                        },
                        "entry_fill": {
                            "description": "入场成交信号的Telegram设置。",
                            "type": "string",
                            "enum": TELEGRAM_SETTING_OPTIONS,
                            "default": "off",
                        },
                        "entry_cancel": {
                            "description": "入场取消信号的Telegram设置。",
                            "type": "string",
                            "enum": TELEGRAM_SETTING_OPTIONS,
                        },
                        "exit": {
                            "description": "出场信号的Telegram设置。",
                            "type": ["string", "object"],
                            "additionalProperties": {
                                "type": "string",
                                "enum": TELEGRAM_SETTING_OPTIONS,
                            },
                        },
                        "exit_fill": {
                            "description": "出场成交信号的Telegram设置。",
                            "type": ["string", "object"],
                            "additionalProperties": {
                                "type": "string",
                                "enum": TELEGRAM_SETTING_OPTIONS,
                            },
                            "default": "on",
                        },
                        "exit_cancel": {
                            "description": "出场取消信号的Telegram设置。",
                            "type": "string",
                            "enum": TELEGRAM_SETTING_OPTIONS,
                        },
                        "protection_trigger": {
                            "description": "保护触发的Telegram设置。",
                            "type": "string",
                            "enum": TELEGRAM_SETTING_OPTIONS,
                            "default": "on",
                        },
                        "protection_trigger_global": {
                            "description": "全局保护触发的Telegram设置。",
                            "type": "string",
                            "enum": TELEGRAM_SETTING_OPTIONS,
                            "default": "on",
                        },
                    },
                },
                "reload": {
                    "description": "在特定消息中添加重新加载按钮。",
                    "type": "boolean",
                },
            },
            "required": ["enabled", "token", "chat_id"],
        },
        "webhook": {
            "description": "Webhook设置。",
            "type": "object",
            "properties": {
                "enabled": {"description": "启用webhook通知。", "type": "boolean"},
                "url": {
                    "description": f"Webhook URL。{__VIA_ENV} FREQTRADE__WEBHOOK__URL",
                    "type": "string",
                },
                "format": {"type": "string", "enum": WEBHOOK_FORMAT_OPTIONS, "default": "form"},
                "retries": {"type": "integer", "minimum": 0},
                "retry_delay": {"type": "number", "minimum": 0},
                **__MESSAGE_TYPE_DICT,
            },
        },
        "discord": {
            "description": "Discord设置。",
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "webhook_url": {
                    "description": (
                        f"Discord webhook URL。{__VIA_ENV} FREQTRADE__DISCORD__WEBHOOK_URL"
                    ),
                    "type": "string",
                },
                "exit_fill": {
                    "type": "array",
                    "items": {"type": "object"},
                    "default": [
                        {"交易ID": "{trade_id}"},
                        {"交易所": "{exchange}"},
                        {"交易对": "{pair}"},
                        {"方向": "{direction}"},
                        {"开仓价格": "{open_rate}"},
                        {"平仓价格": "{close_rate}"},
                        {"数量": "{amount}"},
                        {"开仓时间": "{open_date:%Y-%m-%d %H:%M:%S}"},
                        {"平仓时间": "{close_date:%Y-%m-%d %H:%M:%S}"},
                        {"利润": "{profit_amount} {stake_currency}"},
                        {"盈利率": "{profit_ratio:.2%}"},
                        {"入场标签": "{enter_tag}"},
                        {"出场原因": "{exit_reason}"},
                        {"策略": "{strategy}"},
                        {"时间周期": "{timeframe}"},
                    ],
                },
                "entry_fill": {
                    "type": "array",
                    "items": {"type": "object"},
                    "default": [
                        {"交易ID": "{trade_id}"},
                        {"交易所": "{exchange}"},
                        {"交易对": "{pair}"},
                        {"方向": "{direction}"},
                        {"开仓价格": "{open_rate}"},
                        {"数量": "{amount}"},
                        {"开仓时间": "{open_date:%Y-%m-%d %H:%M:%S}"},
                        {"入场标签": "{enter_tag}"},
                        {"策略": "{strategy} {timeframe}"},
                    ],
                },
            },
        },
        "api_server": {
            "description": "API服务器设置。",
            "type": "object",
            "properties": {
                "enabled": {"description": "API服务器是否启用。", "type": "boolean"},
                "listen_ip_address": {
                    "description": "API服务器监听的IP地址。",
                    "format": "ipv4",
                },
                "listen_port": {
                    "description": "API服务器监听的端口。",
                    "type": "integer",
                    "minimum": 1024,
                    "maximum": 65535,
                },
                "username": {
                    "description": "API服务器认证的用户名。",
                    "type": "string",
                },
                "password": {
                    "description": "API服务器认证的密码。",
                    "type": "string",
                },
                "ws_token": {
                    "description": "API服务器的WebSocket令牌。",
                    "type": ["string", "array"],
                    "items": {"type": "string"},
                },
                "jwt_secret_key": {
                    "description": "JWT认证的密钥。",
                    "type": "string",
                },
                "CORS_origins": {
                    "description": "允许的CORS来源列表。",
                    "type": "array",
                    "items": {"type": "string"},
                },
                "verbosity": {
                    "description": "日志详细程度级别。",
                    "type": "string",
                    "enum": ["error", "info"],
                },
            },
            "required": ["enabled", "listen_ip_address", "listen_port", "username", "password"],
        },
        # RPC部分结束
        "db_url": {
            "description": "数据库连接URL。",
            "type": "string",
        },
        "export": {
            "description": "要导出的数据类型。",
            "type": "string",
            "enum": EXPORT_OPTIONS,
            "default": "trades",
        },
        "disableparamexport": {
            "description": "禁用参数导出。",
            "type": "boolean",
        },
        "initial_state": {
            "description": "系统的初始状态。",
            "type": "string",
            "enum": ["running", "paused", "stopped"],
        },
        "force_entry_enable": {
            "description": "强制启用入场。",
            "type": "boolean",
        },
        "disable_dataframe_checks": {
            "description": "禁用对数据帧的检查。",
            "type": "boolean",
        },
        "internals": {
            "description": "内部设置。",
            "type": "object",
            "default": {},
            "properties": {
                "process_throttle_secs": {
                    "description": "一次机器人迭代的最小循环持续时间（秒）。",
                    "type": "integer",
                },
                "interval": {
                    "description": "间隔时间（秒）。",
                    "type": "integer",
                },
                "sd_notify": {
                    "description": "启用systemd通知。",
                    "type": "boolean",
                },
            },
        },
        "dataformat_ohlcv": {
            "description": "OHLCV数据的数据格式。",
            "type": "string",
            "enum": AVAILABLE_DATAHANDLERS,
            "default": "feather",
        },
        "dataformat_trades": {
            "description": "交易数据的数据格式。",
            "type": "string",
            "enum": AVAILABLE_DATAHANDLERS,
            "default": "feather",
        },
        "position_adjustment_enable": {
            "description": f"启用仓位调整。{__IN_STRATEGY}",
            "type": "boolean",
        },
        # 下载数据部分
        "new_pairs_days": {
            "description": "下载新交易对指定天数的数据",
            "type": "integer",
            "default": 30,
        },
        "download_trades": {
            "description": "默认下载交易数据（而不是ohlcv数据）。",
            "type": "boolean",
        },
        "max_entry_position_adjustment": {
            "description": f"允许的最大入场仓位调整。{__IN_STRATEGY}",
            "type": ["integer", "number"],
            "minimum": -1,
        },
        "add_config_files": {
            "description": "要加载的额外配置文件。",
            "type": "array",
            "items": {"type": "string"},
        },
        "orderflow": {
            "description": "与订单流相关的设置。",
            "type": "object",
            "properties": {
                "cache_size": {
                    "description": "订单流数据的缓存大小。",
                    "type": "number",
                    "minimum": 1,
                    "default": 1500,
                },
                "max_candles": {
                    "description": "要考虑的最大K线数量。",
                    "type": "number",
                    "minimum": 1,
                    "default": 1500,
                },
                "scale": {
                    "description": "订单流数据的缩放因子。",
                    "type": "number",
                    "minimum": 0.0,
                },
                "stacked_imbalance_range": {
                    "description": "堆叠不平衡的范围。",
                    "type": "number",
                    "minimum": 0,
                },
                "imbalance_volume": {
                    "description": "不平衡的交易量阈值。",
                    "type": "number",
                    "minimum": 0,
                },
                "imbalance_ratio": {
                    "description": "不平衡的比率阈值。",
                    "type": "number",
                    "minimum": 0.0,
                },
            },
            "required": [
                "max_candles",
                "scale",
                "stacked_imbalance_range",
                "imbalance_volume",
                "imbalance_ratio",
            ],
        },
    },
    "definitions": {
        "exchange": {
            "description": "交易所配置设置。",
            "type": "object",
            "properties": {
                "name": {"description": "交易所名称。", "type": "string"},
                "key": {
                    "description": (
                        f"交易所的API密钥。{__VIA_ENV} FREQTRADE__EXCHANGE__KEY"
                    ),
                    "type": "string",
                    "default": "",
                },
                "secret": {
                    "description": (
                        f"交易所的API密钥。{__VIA_ENV} FREQTRADE__EXCHANGE__SECRET"
                    ),
                    "type": "string",
                    "default": "",
                },
                "password": {
                    "description": (
                        "交易所的密码（如果需要）。"
                        f"{__VIA_ENV} FREQTRADE__EXCHANGE__PASSWORD"
                    ),
                    "type": "string",
                    "default": "",
                },
                "uid": {
                    "description": (
                        "交易所的用户ID（如果需要）。"
                        f"{__VIA_ENV} FREQTRADE__EXCHANGE__UID"
                    ),
                    "type": "string",
                },
                "account_id": {
                    "description": (
                        "交易所的账户ID（如果需要）。"
                        f"{__VIA_ENV} FREQTRADE__EXCHANGE__ACCOUNT_ID"
                    ),
                    "type": "string",
                },
                "wallet_address": {
                    "description": (
                        "交易所的钱包地址（如果需要）。"
                        "通常用于去中心化交易所。"
                        f"{__VIA_ENV} FREQTRADE__EXCHANGE__WALLET_ADDRESS"
                    ),
                    "type": "string",
                },
                "private_key": {
                    "description": (
                        "交易所的私钥（如果需要）。通常用于去中心化交易所。"
                        f"{__VIA_ENV} FREQTRADE__EXCHANGE__PRIVATE_KEY"
                    ),
                    "type": "string",
                },
                "pair_whitelist": {
                    "description": "白名单交易对列表。",
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
                },
                "pair_blacklist": {
                    "description": "黑名单交易对列表。",
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
                },
                "log_responses": {
                    "description": (
                        "记录来自交易所的响应。"
                        "调试订单处理问题时有用/必需。"
                    ),
                    "type": "boolean",
                    "default": False,
                },
                "enable_ws": {
                    "description": "启用与交易所的WebSocket连接。",
                    "type": "boolean",
                    "default": True,
                },
                "unknown_fee_rate": {
                    "description": "未知市场的费率。",
                    "type": "number",
                },
                "outdated_offset": {
                    "description": "过时数据的偏移量（分钟）。",
                    "type": "integer",
                    "minimum": 1,
                },
                "markets_refresh_interval": {
                    "description": "刷新市场数据的间隔（分钟）。",
                    "type": "integer",
                    "default": 60,
                },
                "ccxt_config": {"description": "CCXT配置设置。", "type": "object"},
                "ccxt_async_config": {
                    "description": (
                        "CCXT异步配置设置。"
                        "通常应使用ccxt_config代替。"
                    ),
                    "type": "object",
                },
                "ccxt_sync_config": {
                    "description": (
                        "CCXT同步配置设置。"
                        "通常应使用ccxt_config代替。"
                    ),
                    "type": "object",
                },
            },
            "required": ["name"],
        },
        "logging": {
            "type": "object",
            "properties": {
                "version": {"type": "number", "const": 1},
                "formatters": {
                    "type": "object",
                    # 理论上如下，但可以更灵活
                    # 基于logging.config文档
                    # "additionalProperties": {
                    #     "type": "object",
                    #     "properties": {
                    #         "format": {"type": "string"},
                    #         "datefmt": {"type": "string"},
                    #     },
                    #     "required": ["format"],
                    # },
                },
                "handlers": {"type": "object"},
                "root": {"type": "object"},
            },
            "required": ["version", "formatters", "handlers", "root"],
        },
        "external_message_consumer": {
            "description": "外部消息消费者配置。",
            "type": "object",
            "properties": {
                "enabled": {
                    "description": "外部消息消费者是否启用。",
                    "type": "boolean",
                    "default": False,
                },
                "producers": {
                    "description": "外部消息消费者的生产者列表。",
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "description": "生产者名称。",
                                "type": "string",
                            },
                            "host": {
                                "description": "生产者主机。",
                                "type": "string",
                            },
                            "port": {
                                "description": "生产者端口。",
                                "type": "integer",
                                "default": 8080,
                                "minimum": 0,
                                "maximum": 65535,
                            },
                            "secure": {
                                "description": "是否使用SSL连接到生产者。",
                                "type": "boolean",
                                "default": False,
                            },
                            "ws_token": {
                                "description": "生产者的WebSocket令牌。",
                                "type": "string",
                            },
                        },
                        "required": ["name", "host", "ws_token"],
                    },
                },
                "wait_timeout": {
                    "description": "等待超时时间（秒）。",
                    "type": "integer",
                    "minimum": 0,
                },
                "sleep_time": {
                    "description": "重试连接前的睡眠时间（秒）。",
                    "type": "integer",
                    "minimum": 0,
                },
                "ping_timeout": {
                    "description": "Ping超时时间（秒）。",
                    "type": "integer",
                    "minimum": 0,
                },
                "remove_entry_exit_signals": {
                    "description": "从数据帧中移除信号列（将它们设置为0）",
                    "type": "boolean",
                    "default": False,
                },
                "initial_candle_limit": {
                    "description": "初始K线限制。",
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 1500,
                    "default": 1500,
                },
                "message_size_limit": {
                    "description": "消息大小限制（兆字节）。",
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 8,
                },
            },
            "required": ["producers"],
        },
        "freqai": {
            "type": "object",
            "properties": {
                "enabled": {
                    "description": "freqAI是否启用。",
                    "type": "boolean",
                    "default": False,
                },
                "identifier": {
                    "description": (
                        "当前模型的唯一ID。"
                        "修改特征时必须更改。"
                    ),
                    "type": "string",
                    "default": "example",
                },
                "write_metrics_to_disk": {
                    "description": "将指标写入磁盘？",
                    "type": "boolean",
                    "default": False,
                },
                "purge_old_models": {
                    "description": "在磁盘上保留的模型数量。",
                    "type": ["boolean", "number"],
                    "default": 2,
                },
                "conv_width": {
                    "description": "神经网络输入张量的宽度。",
                    "type": "integer",
                    "default": 1,
                },
                "train_period_days": {
                    "description": (
                        "用于训练数据的天数（滑动窗口的宽度）"
                    ),
                    "type": "integer",
                    "default": 0,
                },
                "backtest_period_days": {
                    "description": (
                        "在滑动`train_period_days`窗口之前，从训练模型推断的天数"
                    ),
                    "type": "number",
                    "default": 7,
                },
                "live_retrain_hours": {
                    "description": "模拟/实盘运行期间的再训练频率。",
                    "type": "number",
                    "default": 0,
                },
                "expiration_hours": {
                    "description": (
                        "如果模型超过`expiration_hours`天，则避免进行预测。默认为0（永不过期）。"
                    ),
                    "type": "number",
                    "default": 0,
                },
                "save_backtest_models": {
                    "description": "运行回测时将模型保存到磁盘。",
                    "type": "boolean",
                    "default": False,
                },
                "fit_live_predictions_candles": {
                    "description": (
                        "用于从预测数据（而非训练数据集）计算目标（标签）统计信息的历史K线数量。"
                    ),
                    "type": "integer",
                },
                "data_kitchen_thread_count": {
                    "description": (
                        "指定用于数据处理的线程数（异常值方法、归一化等）。"
                    ),
                    "type": "integer",
                },
                "activate_tensorboard": {
                    "description": "指示是否激活tensorboard",
                    "type": "boolean",
                    "default": True,
                },
                "wait_for_training_iteration_on_reload": {
                    "description": (
                        "在/reload或ctrl+c后等待下一个训练迭代完成。"
                    ),
                    "type": "boolean",
                    "default": True,
                },
                "continual_learning": {
                    "description": (
                        "使用最近训练模型的最终状态作为新模型的起点，允许"
                        "增量学习。"
                    ),
                    "type": "boolean",
                    "default": False,
                },
                "keras": {
                    "description": "使用Keras进行模型训练。",
                    "type": "boolean",
                    "default": False,
                },
                "feature_parameters": {
                    "description": "用于构建特征集的参数",
                    "type": "object",
                    "properties": {
                        "include_corr_pairlist": {
                            "description": "要包含在特征中的相关交易对列表。",
                            "type": "array",
                        },
                        "include_timeframes": {
                            "description": (
                                "`feature_engineering_expand_*()`中所有指标将为之创建的时间周期列表。"
                            ),
                            "type": "array",
                        },
                        "label_period_candles": {
                            "description": (
                                "用于标记周期的未来K线数量。"
                                "可在`set_freqai_targets()`中使用。"
                            ),
                            "type": "integer",
                        },
                        "include_shifted_candles": {
                            "description": (
                                "将先前K线的特征添加到后续K线，目的是添加历史信息。"
                            ),
                            "type": "integer",
                            "default": 0,
                        },
                        "DI_threshold": {
                            "description": (
                                "当设置为>0时，激活使用差异指数进行"
                                "异常值检测。"
                            ),
                            "type": "number",
                            "default": 0,
                        },
                        "weight_factor": {
                            "description": (
                                "根据数据点的新近度对训练数据点进行加权。"
                            ),
                            "type": "number",
                            "default": 0,
                        },
                        "principal_component_analysis": {
                            "description": (
                                "使用主成分分析自动降低数据集的维度"
                            ),
                            "type": "boolean",
                            "default": False,
                        },
                        "indicator_periods_candles": {
                            "description": (
                                "计算指标的时间周期。"
                                "指标被添加到基础指标数据集中。"
                            ),
                            "type": "array",
                            "items": {"type": "number", "minimum": 1},
                        },
                        "use_SVM_to_remove_outliers": {
                            "description": "使用SVM从特征中移除异常值。",
                            "type": "boolean",
                            "default": False,
                        },
                        "plot_feature_importances": {
                            "description": "为每个模型创建特征重要性图。",
                            "type": "integer",
                            "default": 0,
                        },
                        "svm_params": {
                            "description": (
                                "Sklearn的`SGDOneClassSVM()`中可用的所有参数。"
                            ),
                            "type": "object",
                            "properties": {
                                "shuffle": {
                                    "description": "应用SVM前是否打乱数据。",
                                    "type": "boolean",
                                    "default": False,
                                },
                                "nu": {
                                    "type": "number",
                                    "default": 0.1,
                                },
                            },
                        },
                        "shuffle_after_split": {
                            "description": (
                                "将数据拆分为训练集和测试集，然后分别打乱"
                                "两个集合。"
                            ),
                            "type": "boolean",
                            "default": False,
                        },
                        "buffer_train_data_candles": {
                            "description": (
                                "在填充指标*之后*，从训练数据的开始和结束处裁剪`buffer_train_data_candles`。"
                            ),
                            "type": "integer",
                            "default": 0,
                        },
                    },
                    "required": [
                        "include_timeframes",
                        "include_corr_pairlist",
                    ],
                },
                "data_split_parameters": {
                    "descriptions": (
                        "scikit-learn的test_train_split()函数的附加参数。"
                    ),
                    "type": "object",
                    "properties": {
                        "test_size": {"type": "number"},
                        "random_state": {"type": "integer"},
                        "shuffle": {"type": "boolean", "default": False},
                    },
                },
                "model_training_parameters": {
                    "description": (
                        "包含所选模型库可用的所有参数的灵活字典。"
                    ),
                    "type": "object",
                },
                "rl_config": {
                    "type": "object",
                    "properties": {
                        "drop_ohlc_from_features": {
                            "description": (
                                "不在特征集中包含归一化的ohlc数据。"
                            ),
                            "type": "boolean",
                            "default": False,
                        },
                        "train_cycles": {
                            "description": "要执行的训练周期数。",
                            "type": "integer",
                        },
                        "max_trade_duration_candles": {
                            "description": (
                                "指导代理训练将交易保持在期望长度以下。"
                            ),
                            "type": "integer",
                        },
                        "add_state_info": {
                            "description": (
                                "在特征集中包含状态信息用于"
                                "训练和推理。"
                            ),
                            "type": "boolean",
                            "default": False,
                        },
                        "max_training_drawdown_pct": {
                            "description": "训练期间允许的最大回撤百分比。",
                            "type": "number",
                            "default": 0.02,
                        },
                        "cpu_count": {
                            "description": "用于训练的线程/CPU数量。",
                            "type": "integer",
                            "default": 1,
                        },
                        "model_type": {
                            "description": "来自stable_baselines3或SBcontrib的模型字符串。",
                            "type": "string",
                            "default": "PPO",
                        },
                        "policy_type": {
                            "description": (
                                "来自stable_baselines3的可用策略类型之一。"
                            ),
                            "type": "string",
                            "default": "MlpPolicy",
                        },
                        "net_arch": {
                            "description": "神经网络的架构。",
                            "type": "array",
                            "default": [128, 128],
                        },
                        "randomize_starting_position": {
                            "description": (
                                "随机化每个情节的起点以避免过拟合。"
                            ),
                            "type": "boolean",
                            "default": False,
                        },
                        "progress_bar": {
                            "description": "显示带有当前进度的进度条。",
                            "type": "boolean",
                            "default": True,
                        },
                        "model_reward_parameters": {
                            "description": "用于配置奖励模型的参数。",
                            "type": "object",
                            "properties": {
                                "rr": {
                                    "type": "number",
                                    "default": 1,
                                    "description": "奖励比率参数。",
                                },
                                "profit_aim": {
                                    "type": "number",
                                    "default": 0.025,
                                    "description": "利润目标参数。",
                                },
                            },
                        },
                    },
                },
            },
            "required": [
                "enabled",
                "train_period_days",
                "backtest_period_days",
                "identifier",
                "feature_parameters",
                "data_split_parameters",
            ],
        },
    },
}

SCHEMA_TRADE_REQUIRED = [
    "exchange",
    "timeframe",
    "max_open_trades",
    "stake_currency",
    "stake_amount",
    "tradable_balance_ratio",
    "last_stake_amount_min_ratio",
    "dry_run",
    "dry_run_wallet",
    "exit_pricing",
    "entry_pricing",
    "stoploss",
    "minimal_roi",
    "internals",
    "dataformat_ohlcv",
    "dataformat_trades",
]

SCHEMA_BACKTEST_REQUIRED = [
    "exchange",
    "stake_currency",
    "stake_amount",
    "dry_run_wallet",
    "dataformat_ohlcv",
    "dataformat_trades",
]
SCHEMA_BACKTEST_REQUIRED_FINAL = [
    *SCHEMA_BACKTEST_REQUIRED,
    "stoploss",
    "minimal_roi",
    "max_open_trades",
]

SCHEMA_MINIMAL_REQUIRED = [
    "exchange",
    "dry_run",
    "dataformat_ohlcv",
    "dataformat_trades",
]
SCHEMA_MINIMAL_WEBSERVER = [*SCHEMA_MINIMAL_REQUIRED, "api_server"]