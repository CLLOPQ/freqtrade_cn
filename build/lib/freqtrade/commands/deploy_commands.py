import logging
import sys
from pathlib import Path
from typing import Any

from freqtrade.constants import USERPATH_STRATEGIES
from freqtrade.enums import RunMode
from freqtrade.exceptions import ConfigurationError, OperationalException


logger = logging.getLogger(__name__)


# 请求超时时间
req_timeout = 30


def start_create_userdir(args: dict[str, Any]) -> None:
    """
    创建"user_data"目录，用于存放用户数据策略、超参数优化等
    :param args: 来自Arguments()的命令行参数
    :return: None
    """
    from freqtrade.configuration.directory_operations import copy_sample_files, create_userdata_dir

    if user_data_dir := args.get("user_data_dir"):
        userdir = create_userdata_dir(user_data_dir, create_dir=True)
        copy_sample_files(userdir, overwrite=args["reset"])
    else:
        logger.warning("`create-userdir` 需要设置 --userdir。")
        sys.exit(1)


def deploy_new_strategy(strategy_name: str, strategy_path: Path, subtemplate: str) -> None:
    """
    从模板部署新策略到strategy_path
    """
    from freqtrade.util import render_template, render_template_with_fallback

    fallback = "full"
    # 渲染策略属性模板
    attributes = render_template_with_fallback(
        templatefile=f"strategy_subtemplates/strategy_attributes_{subtemplate}.j2",
        templatefallbackfile=f"strategy_subtemplates/strategy_attributes_{fallback}.j2",
    )
    # 渲染指标模板
    indicators = render_template_with_fallback(
        templatefile=f"strategy_subtemplates/indicators_{subtemplate}.j2",
        templatefallbackfile=f"strategy_subtemplates/indicators_{fallback}.j2",
    )
    # 渲染买入趋势模板
    buy_trend = render_template_with_fallback(
        templatefile=f"strategy_subtemplates/buy_trend_{subtemplate}.j2",
        templatefallbackfile=f"strategy_subtemplates/buy_trend_{fallback}.j2",
    )
    # 渲染卖出趋势模板
    sell_trend = render_template_with_fallback(
        templatefile=f"strategy_subtemplates/sell_trend_{subtemplate}.j2",
        templatefallbackfile=f"strategy_subtemplates/sell_trend_{fallback}.j2",
    )
    # 渲染绘图配置模板
    plot_config = render_template_with_fallback(
        templatefile=f"strategy_subtemplates/plot_config_{subtemplate}.j2",
        templatefallbackfile=f"strategy_subtemplates/plot_config_{fallback}.j2",
    )
    # 渲染额外方法模板
    additional_methods = render_template_with_fallback(
        templatefile=f"strategy_subtemplates/strategy_methods_{subtemplate}.j2",
        templatefallbackfile="strategy_subtemplates/strategy_methods_empty.j2",
    )

    # 渲染基础策略模板
    strategy_text = render_template(
        templatefile="base_strategy.py.j2",
        arguments={
            "strategy": strategy_name,
            "attributes": attributes,
            "indicators": indicators,
            "buy_trend": buy_trend,
            "sell_trend": sell_trend,
            "plot_config": plot_config,
            "additional_methods": additional_methods,
        },
    )

    logger.info(f"正在将策略写入 `{strategy_path}`。")
    strategy_path.write_text(strategy_text)


def start_new_strategy(args: dict[str, Any]) -> None:
    """创建新策略"""
    from freqtrade.configuration import setup_utils_configuration

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    if strategy := args.get("strategy"):
        if strategy_path := args.get("strategy_path"):
            strategy_dir = Path(strategy_path)
        else:
            strategy_dir = config["user_data_dir"] / USERPATH_STRATEGIES
        if not strategy_dir.is_dir():
            logger.info(f"正在创建策略目录 {strategy_dir}")
            strategy_dir.mkdir(parents=True)
        new_path = strategy_dir / (strategy + ".py")

        if new_path.exists():
            raise OperationalException(
                f"`{new_path}` 已存在。请选择其他策略名称。"
            )

        deploy_new_strategy(strategy, new_path, args["template"])

    else:
        raise ConfigurationError("`new-strategy` 需要设置 --strategy。")


def start_install_ui(args: dict[str, Any]) -> None:
    """安装用户界面"""
    from freqtrade.commands.deploy_ui import (
        clean_ui_subdir,
        download_and_install_ui,
        get_ui_download_url,
        read_ui_version,
    )

    dest_folder = Path(__file__).parents[1] / "rpc/api_server/ui/installed/"
    # 首先确保资源已被移除
    dl_url, latest_version = get_ui_download_url(
        args.get("ui_version"), args.get("ui_prerelease", False)
    )

    curr_version = read_ui_version(dest_folder)
    if curr_version == latest_version and not args.get("erase_ui_only"):
        logger.info(f"UI 已经是最新版本，FreqUI 版本 {curr_version}。")
        return

    clean_ui_subdir(dest_folder)
    if args.get("erase_ui_only"):
        logger.info("已清除 UI 目录内容。不下载新版本。")
    else:
        # 下载新版本
        download_and_install_ui(dest_folder, dl_url, latest_version)