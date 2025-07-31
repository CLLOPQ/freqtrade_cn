import logging
from collections.abc import Callable
from copy import deepcopy
from functools import wraps
from typing import Any, TypeVar, cast

from freqtrade.exceptions import StrategyError


logger = logging.getLogger(__name__)


F = TypeVar("F", bound=Callable[..., Any])


def strategy_safe_wrapper(f: F, message: str = "", default_retval=None, supress_error=False) -> F:
    """
    围绕用户提供的方法和函数的包装器。
    缓存所有异常，并返回默认返回值（如果不为None）或抛出StrategyError异常，该异常随后需要由调用方法处理。
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            if not (getattr(f, "__qualname__", "")).startswith("IStrategy."):
                # 如果函数未在用户策略中实现，则不要深拷贝。``
                if "trade" in kwargs:
                    # 防止策略内部的意外修改
                    kwargs["trade"] = deepcopy(kwargs["trade"])
            return f(*args, **kwargs)
        except ValueError as error:
            logger.warning(f"{message}策略导致以下异常：{error}{f}")
            if default_retval is None and not supress_error:
                raise StrategyError(str(error)) from error
            return default_retval
        except Exception as error:
            logger.exception(f"{message}调用{f}时发生意外错误：{error}")
            if default_retval is None and not supress_error:
                raise StrategyError(str(error)) from error
            return default_retval

    return cast(F, wrapper)