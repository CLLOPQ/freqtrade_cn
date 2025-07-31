# pragma pylint: disable=attribute-defined-outside-init

"""
此模块用于加载自定义对象
"""

import importlib.util
import inspect
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from freqtrade.constants import Config
from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


class PathModifier:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        """注入路径以允许使用相对导入进行导入。"""
        sys.path.insert(0, str(self.path))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """撤销本地路径的插入。"""
        str_path = str(self.path)
        if str_path in sys.path:
            sys.path.remove(str_path)


class IResolver:
    """
    此类包含加载自定义类的所有逻辑
    """

    # 子类需要重写此属性
    object_type: type[Any]
    object_type_str: str
    user_subdir: str | None = None
    initial_search_path: Path | None = None
    # 包含路径的可选配置设置（strategy_path, freqaimodel_path）
    extra_path: str | None = None

    @classmethod
    def build_search_paths(
        cls,
        config: Config,
        user_subdir: str | None = None,
        extra_dirs: list[str] | None = None,
    ) -> list[Path]:
        abs_paths: list[Path] = []
        if cls.initial_search_path:
            abs_paths.append(cls.initial_search_path)

        if user_subdir:
            abs_paths.insert(0, config["user_data_dir"].joinpath(user_subdir))

        # 将额外目录添加到搜索路径的顶部
        if extra_dirs:
            for directory in extra_dirs:
                abs_paths.insert(0, Path(directory).resolve())

        if cls.extra_path and (extra := config.get(cls.extra_path)):
            abs_paths.insert(0, Path(extra).resolve())

        return abs_paths

    @classmethod
    def _get_valid_object(
        cls, module_path: Path, object_name: str | None, enum_failed: bool = False
    ) -> Iterator[Any]:
        """
        生成器返回路径中与对象类型和对象名称匹配的对象。
        :param module_path: 模块的绝对路径
        :param object_name: 对象的类名
        :param enum_failed: 如果为True，则对失败的模块返回None。否则，跳过失败的模块。
        :return: 包含匹配对象的生成器
             元组格式: [对象, 源]
        """

        # 根据绝对路径生成模块规范
        # 将object_name作为第一个参数，以便日志打印合理的名称。
        with PathModifier(module_path.parent):
            module_name = module_path.stem or ""
            spec = importlib.util.spec_from_file_location(module_name, str(module_path))
            if not spec:
                return iter([None])

            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)  # type: ignore # importlib不使用类型提示
            except (
                AttributeError,
                ModuleNotFoundError,
                SyntaxError,
                ImportError,
                NameError,
            ) as err:
                # 捕获特定模块未安装时的错误
                logger.warning(f"无法导入 {module_path}，原因是 '{err}'")
                if enum_failed:
                    return iter([None])

            def is_valid_class(obj):
                try:
                    return (
                        inspect.isclass(obj)
                        and issubclass(obj, cls.object_type)
                        and obj is not cls.object_type
                        and obj.__module__ == module_name
                    )
                except TypeError:
                    return False

            valid_objects_gen = (
                (obj, inspect.getsource(module))
                for name, obj in inspect.getmembers(module, is_valid_class)
                if (object_name is None or object_name == name)
            )
            # __module__检查确保只使用在此文件夹中定义的对象。
            return valid_objects_gen

    @classmethod
    def _search_object(
        cls, directory: Path, *, object_name: str, add_source: bool = False
    ) -> tuple[Any, Path] | tuple[None, None]:
        """
        在给定目录中搜索对象名称
        :param directory: 相对或绝对目录路径
        :param object_name: 要加载的对象的类名
        :return: 对象类
        """
        logger.debug(f"在 '{directory}' 中搜索 {cls.object_type.__name__} {object_name}")
        for entry in directory.iterdir():
            # 只考虑Python文件
            if entry.suffix != ".py":
                logger.debug("忽略 %s", entry)
                continue
            if entry.is_symlink() and not entry.is_file():
                logger.debug("忽略损坏的符号链接 %s", entry)
                continue
            module_path = entry.resolve()

            obj = next(cls._get_valid_object(module_path, object_name), None)

            if obj:
                obj[0].__file__ = str(entry)
                if add_source:
                    obj[0].__source__ = obj[1]
                return (obj[0], module_path)
        return (None, None)

    @classmethod
    def _load_object(
        cls, paths: list[Path], *, object_name: str, add_source: bool = False, kwargs: dict
    ) -> Any | None:
        """
        尝试从路径列表加载对象。
        """

        for _path in paths:
            try:
                (module, module_path) = cls._search_object(
                    directory=_path, object_name=object_name, add_source=add_source
                )
                if module:
                    logger.info(
                        f"从 '{module_path}' 使用解析后的 {cls.object_type.__name__.lower()[1:]} {object_name}..."
                    )
                    return module(**kwargs)
            except FileNotFoundError:
                logger.warning('路径 "%s" 不存在。', _path.resolve())

        return None

    @classmethod
    def load_object(
        cls, object_name: str, config: Config, *, kwargs: dict, extra_dir: str | None = None
    ) -> Any:
        """
        根据子类中配置的方式搜索并加载指定的对象。
        :param object_name: 要导入的模块名称
        :param config: 配置字典
        :param extra_dir: 用于搜索给定对象列表的附加目录
        :raises: OperationalException 如果类无效或不存在。
        :return: 对象实例或None
        """

        extra_dirs: list[str] = []
        if extra_dir:
            extra_dirs.append(extra_dir)

        abs_paths = cls.build_search_paths(
            config, user_subdir=cls.user_subdir, extra_dirs=extra_dirs
        )

        found_object = cls._load_object(paths=abs_paths, object_name=object_name, kwargs=kwargs)
        if found_object:
            return found_object
        raise OperationalException(
            f"无法加载 {cls.object_type_str} '{object_name}'。此类不存在 "
            "或包含Python代码错误。"
        )

    @classmethod
    def search_all_objects(
        cls, config: Config, enum_failed: bool, recursive: bool = False
    ) -> list[dict[str, Any]]:
        """
        搜索有效对象
        :param config: 配置对象
        :param enum_failed: 如果为True，则对失败的模块返回None。否则，跳过失败的模块。
        :param recursive: 递归遍历目录树以搜索对象
        :return: 包含'name'、'class'和'location'条目的字典列表
        """
        result = []

        abs_paths = cls.build_search_paths(config, user_subdir=cls.user_subdir)
        for path in abs_paths:
            result.extend(cls._search_all_objects(path, enum_failed, recursive))
        return result

    @classmethod
    def _build_rel_location(cls, directory: Path, entry: Path) -> str:
        builtin = cls.initial_search_path == directory
        return (
            f"<内置>/{entry.relative_to(directory)}"
            if builtin
            else str(entry.relative_to(directory))
        )

    @classmethod
    def _search_all_objects(
        cls,
        directory: Path,
        enum_failed: bool,
        recursive: bool = False,
        basedir: Path | None = None,
    ) -> list[dict[str, Any]]:
        """
        在目录中搜索有效对象
        :param directory: 要搜索的路径
        :param enum_failed: 如果为True，则对失败的模块返回None。否则，跳过失败的模块。
        :param recursive: 递归遍历目录树以搜索对象
        :return: 包含'name'、'class'和'location'条目的字典列表
        """
        logger.debug(f"在 '{directory}' 中搜索 {cls.object_type.__name__}")
        objects: list[dict[str, Any]] = []
        if not directory.is_dir():
            logger.info(f"'{directory}' 不是目录，已跳过。")
            return objects
        for entry in directory.iterdir():
            if (
                recursive
                and entry.is_dir()
                and not entry.name.startswith("__")
                and not entry.name.startswith(".")
            ):
                objects.extend(
                    cls._search_all_objects(entry, enum_failed, recursive, basedir or directory)
                )
            # 只考虑Python文件
            if entry.suffix != ".py":
                logger.debug("忽略 %s", entry)
                continue
            module_path = entry.resolve()
            logger.debug(f"路径 {module_path}")
            for obj in cls._get_valid_object(
                module_path, object_name=None, enum_failed=enum_failed
            ):
                objects.append(
                    {
                        "name": obj[0].__name__ if obj is not None else "",
                        "class": obj[0] if obj is not None else None,
                        "location": entry,
                        "location_rel": cls._build_rel_location(basedir or directory, entry),
                    }
                )
        return objects