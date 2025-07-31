#!/usr/bin/env python3
"""
Freqtrade主程序脚本。
请阅读文档以了解所需的命令行参数。
"""

import logging
import sys
from typing import Any


# 检查最小Python版本
if sys.version_info < (3, 10):  # pragma: no cover  # noqa: UP036
    sys.exit("Freqtrade需要Python版本 >= 3.10")

from freqtrade import __version__
from freqtrade.commands import Arguments
from freqtrade.constants import DOCS_LINK
from freqtrade.exceptions import ConfigurationError, FreqtradeException, OperationalException
from freqtrade.loggers import setup_logging_pre
from freqtrade.system import asyncio_setup, gc_set_threshold, print_version_info


logger = logging.getLogger("freqtrade")


def main(sysargv: list[str] | None = None) -> None:
    """
    此函数将初始化交易机器人并启动交易循环。
    :return: None
    """

    return_code: Any = 1
    try:
        setup_logging_pre()
        asyncio_setup()
        arguments = Arguments(sysargv)
        args = arguments.get_parsed_arg()

        # 调用子命令
        if args.get("version") or args.get("version_main"):
            print_version_info()
            return_code = 0
        elif "func" in args:
            logger.info(f"Freqtrade {__version__}")
            gc_set_threshold()
            return_code = args["func"](args)
        else:
            # 未指定子命令
            raise OperationalException(
                "使用Freqtrade需要指定一个子命令。\n"
                "要让机器人在实盘/模拟盘模式下执行交易，请根据配置文件中`dry_run`设置的值，使用`freqtrade trade [options...]`命令运行Freqtrade。\n"
                "要查看可用的完整选项列表，请使用`freqtrade --help`或`freqtrade <command> --help`。"
            )

    except SystemExit as e:  # pragma: no cover
        return_code = e
    except KeyboardInterrupt:
        logger.info("收到SIGINT信号，正在中止...")
        return_code = 0
    except ConfigurationError as e:
        logger.error(
            f"配置错误：{e}\n"
            f"请确保查阅{DOCS_LINK}处的文档。"
        )
    except FreqtradeException as e:
        logger.error(str(e))
        return_code = 2
    except Exception:
        logger.exception("致命异常！")
    finally:
        sys.exit(return_code)


if __name__ == "__main__":  # pragma: no cover
    main()