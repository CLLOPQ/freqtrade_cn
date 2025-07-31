#!/usr/bin/env python3
import json
import os
from pathlib import Path

import ccxt


# 从环境变量获取交易所API密钥和密钥
key = os.environ.get("FREQTRADE__EXCHANGE__KEY")
secret = os.environ.get("FREQTRADE__EXCHANGE__SECRET")

# 从环境变量获取代理设置（如果有）
proxy = os.environ.get("CI_WEB_PROXY")

# 初始化Binance交易所连接，配置为期货交易
exchange = ccxt.binance(
    {
        "apiKey": key,
        "secret": secret,
        "httpsProxy": proxy,
        "options": {"defaultType": "swap"},  # 设置为永续合约
    }
)
# 加载市场数据（获取交易对信息等）
_ = exchange.load_markets()

# 获取所有交易对的杠杆层级信息
lev_tiers = exchange.fetch_leverage_tiers()

# 假设脚本在仓库根目录运行，定义保存文件路径
file = Path("freqtrade/exchange/binance_leverage_tiers.json")
# 将杠杆层级信息排序后保存为JSON文件，带缩进格式
json.dump(dict(sorted(lev_tiers.items())), file.open("w"), indent=2)
