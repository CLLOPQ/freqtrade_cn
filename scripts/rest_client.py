#!/usr/bin/env python3
"""
简单的RPC命令行客户端
可作为Telegram的替代工具

不应从freqtrade导入任何内容，
因此它可以用作独立脚本。
"""

from freqtrade_client.ft_client import main


if __name__ == "__main__":
    main()