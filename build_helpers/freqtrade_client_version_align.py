#!/usr/bin/env python3
from freqtrade import __version__ as ft_version
from freqtrade_client import __version__ as client_version


def main():
    # 比较Freqtrade核心版本与客户端版本
    if ft_version != client_version:
        # 版本不匹配时输出详细信息并以错误码退出
        print(f"版本不匹配: \n核心: {ft_version} \n客户端: {client_version}")
        exit(1)
    # 版本匹配时输出确认信息并正常退出
    print(f"版本匹配: 核心: {ft_version}, 客户端: {client_version}")
    exit(0)


if __name__ == "__main__":
    main()