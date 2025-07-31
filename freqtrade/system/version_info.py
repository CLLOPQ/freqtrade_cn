from freqtrade import __version__


def print_version_info():
    """打印freqtrade及其关键依赖项的版本信息。"""
    import platform
    import sys

    import ccxt

    print(f"操作系统:\t{platform.platform()}")
    print(f"Python版本:\t\tPython {sys.version.split(' ')[0]}")
    print(f"CCXT版本:\t\t{ccxt.__version__}")
    print()
    print(f"Freqtrade版本:\tfreqtrade {__version__}")