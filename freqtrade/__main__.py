#!/usr/bin/env python3
"""
Freqtrade的__main__.py文件
用于作为模块启动Freqtrade

> python -m freqtrade（需要Python >= 3.10）
"""

from freqtrade import main


if __name__ == "__main__":
    main.main()