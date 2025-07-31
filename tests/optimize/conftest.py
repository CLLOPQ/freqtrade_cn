from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from freqtrade.enums import ExitType, RunMode
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.optimize.hyperopt import Hyperopt
from tests.conftest import patch_exchange


@pytest.fixture(scope="function")
def hyperopt_conf(default_conf):
    """
    创建超参数优化的配置
    基于默认配置进行 deepcopy并更新超参特定的配置项
    """
    hyperconf = deepcopy(default_conf)
    hyperconf.update(
        {
            "datadir": Path(default_conf["datadir"]),  # 数据目录
            "runmode": RunMode.HYPEROPT,  # 运行模式：超参数优化
            "strategy": "HyperoptableStrategy",  # 使用的策略
            "hyperopt_loss": "ShortTradeDurHyperOptLoss",  # 超参数优化的损失函数
            "hyperopt_path": str(Path(__file__).parent / "hyperopts"),  # 超参数优化文件路径
            "epochs": 1,  # 优化迭代次数
            "timerange": None,  # 时间范围，None表示使用所有数据
            "spaces": ["default"],  # 优化空间
            "hyperopt_jobs": 1,  # 超参数优化的并行进程数
            "hyperopt_min_trades": 1,  # 最小交易次数要求
        }
    )
    return hyperconf


@pytest.fixture(autouse=True)
def backtesting_cleanup():
    """
    回测试清理装置
    在每个个测试函数执行后运行，清理回测相关资源
    """
    yield None  # 测试函数执行处

    Backtesting.cleanup()  # 清理理回测相关资源


@pytest.fixture(scope="function")
def hyperopt(hyperopt_conf, mocker):
    """
    创建超参数优化实例的装置
    打用提供的配置和模拟的交易所创建Hyperopt实例
    """
    patch_exchange(mocker)  # 模拟交易所
    return Hyperopt(hyperopt_conf)  # 创建并返回Hyperopt实例


@pytest.fixture(scope="function")
def hyperopt_results():
    """
    创建建建超参数优化结果的示例数据
    返回包含交易结果的DataFrame，用于测试
    """
    return pd.DataFrame(
        {
            "pair": ["ETH/USDT", "ETH/USDT", "ETH/USDT", "ETH/USDT"],  # 交易对
            "profit_ratio": [-0.1, 0.2, -0.12, 0.3],  # 利润率
            "profit_abs": [-0.2, 0.4, -0.21, 0.6],  # 绝对利润
            "trade_duration": [10, 30, 10, 10],  # 交易持续时间
            "amount": [0.1, 0.1, 0.1, 0.1],  # 交易数量
            "exit_reason": [ExitType.STOP_LOSS, ExitType.ROI, ExitType.STOP_LOSS, ExitType.ROI],  # 退出原因
            "open_date": [  # 开仓时间
                datetime(2019, 1, 1, 9, 15, 0),
                datetime(2019, 1, 2, 8, 55, 0),
                datetime(2019, 1, 3, 9, 15, 0),
                datetime(2019, 1, 4, 9, 15, 0),
            ],
            "close_date": [  # 平仓时间
                datetime(2019, 1, 1, 9, 25, 0),
                datetime(2019, 1, 2, 9, 25, 0),
                datetime(2019, 1, 3, 9, 25, 0),
                datetime(2019, 1, 4, 9, 25, 0),
            ],
        }
    )