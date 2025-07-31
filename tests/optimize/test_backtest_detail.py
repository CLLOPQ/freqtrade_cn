# pragma pylint: disable=missing-docstring, W0212, line-too-long, C0103, C0330, unused-argument
import logging
from unittest.mock import MagicMock

import pytest

from freqtrade.data.history import get_timerange
from freqtrade.enums import ExitType, TradingMode
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.persistence.trade_model import LocalTrade
from tests.conftest import EXMS, patch_exchange
from tests.optimize import (
    BTContainer,
    BTrade,
    _build_backtest_dataframe,
    _get_frame_time_from_offset,
    tests_timeframe,
)


# 测试 0: 在第3根K线发出卖出信号
# 止损设置为 1%
tc0 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
        [1, 5000, 5025, 4975, 4987, 6172, 0, 0],  # 进入交易（最后一根K线的信号）
        [2, 4987, 5012, 4986, 4986, 6172, 0, 0],  # 退出，触发止损
        [3, 5010, 5010, 4980, 5010, 6172, 0, 1],
        [4, 5010, 5011, 4977, 4995, 6172, 0, 0],
        [5, 4995, 4995, 4950, 4950, 6172, 0, 0],
    ],
    stop_loss=-0.01,
    roi={"0": 1},
    profit_perc=0.002,
    use_exit_signal=True,
    trades=[BTrade(exit_reason=ExitType.EXIT_SIGNAL, open_tick=1, close_tick=4)],
)

# 测试 1: 触发止损，1% 损失
# 止损设置为 1%
tc1 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
        [1, 5000, 5025, 4975, 4987, 6172, 0, 0],  # 进入交易（最后一根K线的信号）
        [2, 4987, 5012, 4600, 4600, 6172, 0, 0],  # 退出，触发止损
        [3, 4975, 5000, 4975, 4977, 6172, 0, 0],
        [4, 4977, 4995, 4977, 4995, 6172, 0, 0],
        [5, 4995, 4995, 4950, 4950, 6172, 0, 0],
    ],
    stop_loss=-0.01,
    roi={"0": 1},
    profit_perc=-0.01,
    trades=[BTrade(exit_reason=ExitType.STOP_LOSS, open_tick=1, close_tick=2)],
)


# 测试 2: 最低价下跌 4%，收盘价下跌 1%
# 止损设置为 3%
tc2 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
        [1, 5000, 5025, 4975, 4987, 6172, 0, 0],  # 进入交易（最后一根K线的信号）
        [2, 4987, 5012, 4962, 4975, 6172, 0, 0],
        [3, 4975, 5000, 4800, 4962, 6172, 0, 0],  # 退出，触发止损
        [4, 4962, 4987, 4937, 4950, 6172, 0, 0],
        [5, 4950, 4975, 4925, 4950, 6172, 0, 0],
    ],
    stop_loss=-0.03,
    roi={"0": 1},
    profit_perc=-0.03,
    trades=[BTrade(exit_reason=ExitType.STOP_LOSS, open_tick=1, close_tick=3)],
)


# 测试 3: 多笔交易
#         K线下跌 4%，恢复 1%
#         满足入场条件
#         K线下跌 20%
#  交易A: 触发止损，2% 损失
#  交易B: 触发止损，2% 损失
tc3 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
        [1, 5000, 5025, 4975, 4987, 6172, 0, 0],  # 进入交易（最后一根K线的信号）
        [2, 4987, 5012, 4800, 4975, 6172, 0, 0],  # 退出，触发止损
        [3, 4975, 5000, 4950, 4962, 6172, 1, 0],
        [4, 4975, 5000, 4950, 4962, 6172, 0, 0],  # 进入交易2（最后一根K线的信号）
        [5, 4962, 4987, 4000, 4000, 6172, 0, 0],  # 退出，触发止损
        [6, 4950, 4975, 4950, 4950, 6172, 0, 0],
    ],
    stop_loss=-0.02,
    roi={"0": 1},
    profit_perc=-0.04,
    trades=[
        BTrade(exit_reason=ExitType.STOP_LOSS, open_tick=1, close_tick=2),
        BTrade(exit_reason=ExitType.STOP_LOSS, open_tick=4, close_tick=5),
    ],
)

# 测试 4: 下跌 3% / 恢复 +15%
# K线数据：K线下跌3%，收盘上涨15%
# 止损设置为 2%，投资回报率 6%
# 触发止损，2% 损失
tc4 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
        [1, 5000, 5025, 4975, 4987, 6172, 0, 0],  # 进入交易（最后一根K线的信号）
        [2, 4987, 5750, 4850, 5750, 6172, 0, 0],  # 退出，触发止损
        [3, 4975, 5000, 4950, 4962, 6172, 0, 0],
        [4, 4962, 4987, 4937, 4950, 6172, 0, 0],
        [5, 4950, 4975, 4925, 4950, 6172, 0, 0],
    ],
    stop_loss=-0.02,
    roi={"0": 0.06},
    profit_perc=-0.02,
    trades=[BTrade(exit_reason=ExitType.STOP_LOSS, open_tick=1, close_tick=2)],
)

# 测试 5: 下跌 0.5%，收盘上涨 20%，ROI 触发 3% 收益
# 止损：1%，ROI：3%
tc5 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5025, 4980, 4987, 6172, 1, 0],
        [1, 5000, 5025, 4980, 4987, 6172, 0, 0],  # 进入交易（最后一根K线的信号）
        [2, 4987, 5025, 4975, 4987, 6172, 0, 0],
        [3, 4975, 6000, 4975, 6000, 6172, 0, 0],  # ROI
        [4, 4962, 4987, 4962, 4972, 6172, 0, 0],
        [5, 4950, 4975, 4925, 4950, 6172, 0, 0],
    ],
    stop_loss=-0.01,
    roi={"0": 0.03},
    profit_perc=0.03,
    trades=[BTrade(exit_reason=ExitType.ROI, open_tick=1, close_tick=3)],
)

# 测试 6: 下跌 3% / 恢复 6% 为正 / 收盘 1% 为正，止损触发 2% 损失
# 止损：2%，ROI：5%
tc6 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
        [1, 5000, 5025, 4975, 4987, 6172, 0, 0],  # 进入交易（最后一根K线的信号）
        [2, 4987, 5300, 4850, 5050, 6172, 0, 0],  # 退出，触发止损
        [3, 4975, 5000, 4950, 4962, 6172, 0, 0],
        [4, 4962, 4987, 4950, 4950, 6172, 0, 0],
        [5, 4950, 4975, 4925, 4950, 6172, 0, 0],
    ],
    stop_loss=-0.02,
    roi={"0": 0.05},
    profit_perc=-0.02,
    trades=[BTrade(exit_reason=ExitType.STOP_LOSS, open_tick=1, close_tick=2)],
)

# 测试 7: 6% 正收益 / 1% 负收益 / 收盘 1% 正收益，ROI 触发 3% 收益
# 止损：2%，ROI：3%
tc7 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
        [1, 5000, 5025, 4975, 4987, 6172, 0, 0],
        [2, 4987, 5300, 4950, 5050, 6172, 0, 0],
        [3, 4975, 5000, 4950, 4962, 6172, 0, 0],
        [4, 4962, 4987, 4950, 4950, 6172, 0, 0],
        [5, 4950, 4975, 4925, 4950, 6172, 0, 0],
    ],
    stop_loss=-0.02,
    roi={"0": 0.03},
    profit_perc=0.03,
    trades=[BTrade(exit_reason=ExitType.ROI, open_tick=1, close_tick=2)],
)


# 测试 8: 移动止损应该上调，导致第3根K线触发止损
# 止损：10%，ROI：10%（不应触发），在第2根K线调整止损
tc8 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
        [1, 5000, 5050, 4950, 5000, 6172, 0, 0],
        [2, 5000, 5250, 4750, 4850, 6172, 0, 0],
        [3, 4850, 5050, 4650, 4750, 6172, 0, 0],
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.10},
    profit_perc=-0.055,
    trailing_stop=True,
    trades=[BTrade(exit_reason=ExitType.TRAILING_STOP_LOSS, open_tick=1, close_tick=3)],
)


# 测试 9: 移动止损应该上调 - 同一根K线内的最高价和最低价
# 止损：10%，ROI：10%（不应触发），在第3根K线调整止损
tc9 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
        [1, 5000, 5050, 4950, 5000, 6172, 0, 0],
        [2, 5000, 5050, 4950, 5000, 6172, 0, 0],
        [3, 5000, 5200, 4550, 4850, 6172, 0, 0],
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.10},
    profit_perc=-0.064,
    trailing_stop=True,
    trades=[BTrade(exit_reason=ExitType.TRAILING_STOP_LOSS, open_tick=1, close_tick=3)],
)

# 测试 10: 移动止损应该上调，导致第3根K线触发止损
# 不应用正向移动止损，因为止损偏移量为10%
# 止损：10%，ROI：10%（不应触发），在第2根K线调整止损
tc10 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
        [1, 5000, 5100, 4950, 5100, 6172, 0, 0],
        [2, 5100, 5251, 5100, 5100, 6172, 0, 0],
        [3, 4850, 5050, 4650, 4750, 6172, 0, 0],
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.10},
    profit_perc=-0.1,
    trailing_stop=True,
    trailing_only_offset_is_reached=True,
    trailing_stop_positive_offset=0.10,
    trailing_stop_positive=0.03,
    trades=[BTrade(exit_reason=ExitType.STOP_LOSS, open_tick=1, close_tick=4)],
)

# 测试 11: 移动止损应该上调，导致第3根K线触发止损
# 应用3%的正向移动止损，因为达到了止损正偏移量
# 止损：10%，ROI：10%（不应触发），在第2根K线调整止损
tc11 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
        [1, 5000, 5100, 4950, 5100, 6172, 0, 0],
        [2, 5100, 5251, 5100, 5100, 6172, 0, 0],
        [3, 5000, 5150, 4650, 4750, 6172, 0, 0],
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.10},
    profit_perc=0.019,
    trailing_stop=True,
    trailing_only_offset_is_reached=True,
    trailing_stop_positive_offset=0.05,
    trailing_stop_positive=0.03,
    trades=[BTrade(exit_reason=ExitType.TRAILING_STOP_LOSS, open_tick=1, close_tick=3)],
)

# 测试 12: 移动止损应该在第2根K线上调并在同一根K线内触发止损
# 应用3%的正向移动止损，因为达到了止损正偏移量
# 止损：10%，ROI：10%（不应触发），在第2根K线调整止损
tc12 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
        [1, 5000, 5100, 4950, 5100, 6172, 0, 0],
        [2, 5100, 5251, 4650, 5100, 6172, 0, 0],
        [3, 4850, 5050, 4650, 4750, 6172, 0, 0],
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.10},
    profit_perc=0.019,
    trailing_stop=True,
    trailing_only_offset_is_reached=True,
    trailing_stop_positive_offset=0.05,
    trailing_stop_positive=0.03,
    trades=[BTrade(exit_reason=ExitType.TRAILING_STOP_LOSS, open_tick=1, close_tick=2)],
)

# 测试 13: 在同一根K线内买入和卖出（ROI）
# 止损：10%（不应触发），ROI：1%
tc13 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
        [1, 5000, 5100, 4950, 5100, 6172, 0, 0],
        [2, 5100, 5251, 4850, 5100, 6172, 0, 0],
        [3, 4850, 5050, 4750, 4750, 6172, 0, 0],
        [4, 4750, 4950, 4750, 4750, 6172, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.01},
    profit_perc=0.01,
    trades=[BTrade(exit_reason=ExitType.ROI, open_tick=1, close_tick=1)],
)

# 测试 14 - 在同一根K线内买入和止损
# 止损：5%，ROI：10%（不应触发）
tc14 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
        [1, 5000, 5100, 4600, 5100, 6172, 0, 0],
        [2, 5100, 5251, 4850, 5100, 6172, 0, 0],
        [3, 4850, 5050, 4750, 4750, 6172, 0, 0],
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0],
    ],
    stop_loss=-0.05,
    roi={"0": 0.10},
    profit_perc=-0.05,
    trades=[BTrade(exit_reason=ExitType.STOP_LOSS, open_tick=1, close_tick=1)],
)


# 测试 15 - 在同一根K线内买入和ROI，下一根K线买入和止损
# 止损：5%，ROI：10%（不应触发）
tc15 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
        [1, 5000, 5100, 4900, 5100, 6172, 1, 0],
        [2, 5100, 5251, 4650, 5100, 6172, 0, 0],
        [3, 4850, 5050, 4750, 4750, 6172, 0, 0],
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0],
    ],
    stop_loss=-0.05,
    roi={"0": 0.01},
    profit_perc=-0.04,
    trades=[
        BTrade(exit_reason=ExitType.ROI, open_tick=1, close_tick=1),
        BTrade(exit_reason=ExitType.STOP_LOSS, open_tick=2, close_tick=2),
    ],
)

# 测试 16: 买入，持有65分钟，然后使用roi=-1强制退出
# 即使卖出原因是ROI，也会导致负收益
# 止损：10%，ROI：10%（不应触发），65分钟后-100%（限制交易持续时间）
tc16 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
        [1, 5000, 5025, 4975, 4987, 6172, 0, 0],
        [2, 4987, 5300, 4950, 5050, 6172, 0, 0],
        [3, 4975, 5000, 4940, 4962, 6172, 0, 0],  # ROI强制退出（roi=-1）
        [4, 4962, 4987, 4950, 4950, 6172, 0, 0],
        [5, 4950, 4975, 4925, 4950, 6172, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.10, "65": -1},
    profit_perc=-0.012,
    trades=[BTrade(exit_reason=ExitType.ROI, open_tick=1, close_tick=3)],
)

# 测试 17: 买入，持有120分钟，然后使用roi=-1强制退出
# 即使卖出原因是ROI，也会导致负收益
# 止损：10%，ROI：10%（不应触发），100分钟后-100%（限制交易持续时间）
# 使用开盘价作为卖出价（特殊情况）- 因为roi时间是时间框架的倍数
tc17 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
        [1, 5000, 5025, 4975, 4987, 6172, 0, 0],
        [2, 4987, 5300, 4950, 5050, 6172, 0, 0],
        [3, 4980, 5000, 4940, 4962, 6172, 0, 0],  # ROI强制退出（roi=-1）
        [4, 4962, 4987, 4950, 4950, 6172, 0, 0],
        [5, 4950, 4975, 4925, 4950, 6172, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.10, "120": -1},
    profit_perc=-0.004,
    trades=[BTrade(exit_reason=ExitType.ROI, open_tick=1, close_tick=3)],
)


# 测试 18: 买入，持有120分钟，然后将ROI降至1%，导致在第3根K线卖出
# 止损：10%，ROI：10%（不应触发），100分钟后-100%（限制交易持续时间）
# 使用开盘价作为卖出价
tc18 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
        [1, 5000, 5025, 4975, 4987, 6172, 0, 0],
        [2, 4987, 5300, 4950, 5200, 6172, 0, 0],
        [3, 5200, 5220, 4940, 4962, 6172, 0, 0],  # ROI卖出（在开盘时卖出）
        [4, 4962, 4987, 4950, 4950, 6172, 0, 0],
        [5, 4950, 4975, 4925, 4950, 6172, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.10, "120": 0.01},
    profit_perc=0.04,
    trades=[BTrade(exit_reason=ExitType.ROI, open_tick=1, close_tick=3)],
)

# 测试 19: 买入，持有119分钟，然后将ROI降至1%，导致在第3根K线卖出
# 止损：10%，ROI：10%（不应触发），100分钟后-100%（限制交易持续时间）
# 使用计算的ROI（1%）作为卖出价，其他与tc18相同
tc19 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
        [1, 5000, 5025, 4975, 4987, 6172, 0, 0],
        [2, 4987, 5300, 4950, 5200, 6172, 0, 0],
        [3, 5000, 5300, 4940, 4962, 6172, 0, 0],  # ROI卖出
        [4, 4962, 4987, 4950, 4950, 6172, 0, 0],
        [5, 4550, 4975, 4550, 4950, 6172, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.10, "120": 0.01},
    profit_perc=0.01,
    trades=[BTrade(exit_reason=ExitType.ROI, open_tick=1, close_tick=3)],
)

# 测试 20: 买入，持有119分钟，然后将ROI降至1%，导致在第3根K线卖出
# 止损：10%，ROI：10%（不应触发），100分钟后-100%（限制交易持续时间）
# 使用计算的ROI（1%）作为卖出价，其他与tc18相同
tc20 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
        [1, 5000, 5025, 4975, 4987, 6172, 0, 0],
        [2, 4987, 5300, 4950, 5200, 6172, 0, 0],
        [3, 5200, 5300, 4940, 4962, 6172, 0, 0],  # ROI卖出
        [4, 4962, 4987, 4950, 4950, 6172, 0, 0],
        [5, 4925, 4975, 4925, 4950, 6172, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.10, "119": 0.01},
    profit_perc=0.01,
    trades=[BTrade(exit_reason=ExitType.ROI, open_tick=1, close_tick=3)],
)

# 测试 21: 移动止损与ROI冲突
# ROI应该在移动止损之前触发 - 否则移动止损的收益可能大于ROI
# 这在现实中是不可能的
# 止损：10%，ROI：4%，在卖出K线调整移动止损
tc21 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
        [1, 5000, 5100, 4950, 5100, 6172, 0, 0],
        [2, 5100, 5251, 4650, 5100, 6172, 0, 0],
        [3, 4850, 5050, 4650, 4750, 6172, 0, 0],
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.04},
    profit_perc=0.04,
    trailing_stop=True,
    trailing_only_offset_is_reached=True,
    trailing_stop_positive_offset=0.05,
    trailing_stop_positive=0.03,
    trades=[BTrade(exit_reason=ExitType.ROI, open_tick=1, close_tick=2)],
)

# 测试 22: 移动止损在第2根K线上调 - 但同时应用ROI
# 应用3%的正向移动止损 - ROI应该在移动止损之前应用
# 止损：10%，ROI：4%，在第2根K线调整止损
tc22 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
        [1, 5000, 5100, 4950, 5100, 6172, 0, 0],
        [2, 5100, 5251, 5100, 5100, 6172, 0, 0],
        [3, 4850, 5050, 4650, 4750, 6172, 0, 0],
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.04},
    profit_perc=0.04,
    trailing_stop=True,
    trailing_only_offset_is_reached=True,
    trailing_stop_positive_offset=0.05,
    trailing_stop_positive=0.03,
    trades=[BTrade(exit_reason=ExitType.ROI, open_tick=1, close_tick=2)],
)


# 测试 23: 移动止损在第2根K线上调 - 但同时应用ROI
# 应用3%的正向移动止损 - ROI应该在移动止损之前应用
# 止损：10%，ROI：4%，在第2根K线调整止损
tc23 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 0, 0, 1, 0],
        [1, 5000, 5050, 4900, 4900, 6172, 0, 0, 0, 0],
        [2, 4900, 4900, 4749, 4900, 6172, 0, 0, 0, 0],
        [3, 4850, 5050, 4650, 4750, 6172, 0, 0, 0, 0],
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.04},
    profit_perc=0.04,
    trailing_stop=True,
    trailing_only_offset_is_reached=True,
    trailing_stop_positive_offset=0.05,
    trailing_stop_positive=0.03,
    trades=[BTrade(exit_reason=ExitType.ROI, open_tick=1, close_tick=2, is_short=True)],
)

# 测试 24: 移动止损在第2根K线上调（未触发）
# 应用3%的正向移动止损，因为达到了止损正偏移量
# ROI在此之后更改为4%，使ROI低于trailing_stop_positive，导致在上调止损K线后的K线内卖出，卖出原因为ROI
# 止损在此K线也会触发，但不再相关
# 止损：10%，ROI：4%，在第2根K线调整止损，在第3根K线调整ROI（导致卖出）
tc24 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
        [1, 5000, 5100, 4950, 5100, 6172, 0, 0],
        [2, 5100, 5251, 5100, 5100, 6172, 0, 0],
        [3, 4850, 5251, 4650, 4750, 6172, 0, 0],
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.1, "119": 0.03},
    profit_perc=0.03,
    trailing_stop=True,
    trailing_only_offset_is_reached=True,
    trailing_stop_positive_offset=0.05,
    trailing_stop_positive=0.03,
    trades=[BTrade(exit_reason=ExitType.ROI, open_tick=1, close_tick=3)],
)

# 测试 25: 在第3根K线发出卖出信号（此K线也触发止损）
# 止损为1%
# 止损优先于卖出信号（因为卖出信号在下一根K线执行）
tc25 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
        [1, 5000, 5025, 4975, 4987, 6172, 0, 0],  # 进入交易（最后一根K线的信号）
        [2, 4987, 5012, 4986, 4986, 6172, 0, 0],
        [3, 5010, 5010, 4855, 5010, 6172, 0, 1],  # 触发止损 + 卖出信号
        [4, 5010, 5010, 4977, 4995, 6172, 0, 0],
        [5, 4995, 4995, 4950, 4950, 6172, 0, 0],
    ],
    stop_loss=-0.01,
    roi={"0": 1},
    profit_perc=-0.01,
    use_exit_signal=True,
    trades=[BTrade(exit_reason=ExitType.STOP_LOSS, open_tick=1, close_tick=3)],
)

# 测试 26: 在第3根K线发出卖出信号（此K线也触发止损）
# 止损为1%
# 卖出信号优先于止损
tc26 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
        [1, 5000, 5025, 4975, 4987, 6172, 0, 0],  # 进入交易（最后一根K线的信号）
        [2, 4987, 5012, 4986, 4986, 6172, 0, 0],
        [3, 5010, 5010, 4986, 5010, 6172, 0, 1],
        [4, 5010, 5010, 4855, 4995, 6172, 0, 0],  # 触发止损 + 执行卖出信号
        [5, 4995, 4995, 4950, 4950, 6172, 0, 0],
    ],
    stop_loss=-0.01,
    roi={"0": 1},
    profit_perc=0.002,
    use_exit_signal=True,
    trades=[BTrade(exit_reason=ExitType.EXIT_SIGNAL, open_tick=1, close_tick=4)],
)

# 测试 27: （test26的杠杆版本）
# 在第3根K线发出卖出信号（此K线也触发止损）
# 止损为1%
# 卖出信号优先于止损
tc27 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
        [1, 5000, 5025, 4975, 4987, 6172, 0, 0],  # 进入交易（最后一根K线的信号）
        [2, 4987, 5012, 4986, 4986, 6172, 0, 0],
        [3, 5010, 5010, 4986, 5010, 6172, 0, 1],
        [4, 5010, 5010, 4855, 4995, 6172, 0, 0],  # 触发止损 + 执行卖出信号
        [5, 4995, 4995, 4950, 4950, 6172, 0, 0],
    ],
    stop_loss=-0.05,
    roi={"0": 1},
    profit_perc=0.002 * 5.0,
    use_exit_signal=True,
    leverage=5.0,
    trades=[BTrade(exit_reason=ExitType.EXIT_SIGNAL, open_tick=1, close_tick=4)],
)

# 测试 28: （test26的杠杆和做空版本）
# 在第3根K线发出卖出信号（此K线也触发止损）
# 止损为1%
# 卖出信号优先于止损
tc28 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5025, 4975, 4987, 6172, 0, 0, 1, 0],
        [1, 5000, 5025, 4975, 4987, 6172, 0, 0, 0, 0],  # 进入交易（最后一根K线的信号）
        [2, 4987, 5012, 4986, 4986, 6172, 0, 0, 0, 0],
        [3, 5010, 5010, 4986, 5010, 6172, 0, 0, 0, 1],
        [4, 4990, 5010, 4855, 4995, 6172, 0, 0, 0, 0],  # 触发止损 + 执行卖出信号
        [5, 4995, 4995, 4950, 4950, 6172, 0, 0, 0, 0],
    ],
    stop_loss=-0.05,
    roi={"0": 1},
    profit_perc=0.002 * 5.0,
    use_exit_signal=True,
    leverage=5.0,
    trades=[BTrade(exit_reason=ExitType.EXIT_SIGNAL, open_tick=1, close_tick=4, is_short=True)],
)
# 测试 29: 在第3根K线发出卖出信号（在信号K线触发ROI）
# 止损为10%（无关），ROI为5%（将触发）
# 卖出信号优先于止损
tc29 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
        [1, 5000, 5025, 4975, 4987, 6172, 0, 0],  # 进入交易（最后一根K线的信号）
        [2, 4987, 5012, 4986, 4986, 6172, 0, 0],
        [3, 5010, 5251, 4986, 5010, 6172, 0, 1],  # 触发ROI，卖出信号
        [4, 5010, 5010, 4855, 4995, 6172, 0, 0],
        [5, 4995, 4995, 4950, 4950, 6172, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.05},
    profit_perc=0.05,
    use_exit_signal=True,
    trades=[BTrade(exit_reason=ExitType.ROI, open_tick=1, close_tick=3)],
)

# 测试 30: 在第3根K线发出卖出信号（在信号K线触发ROI）
# 止损为10%（无关），ROI为5%（将触发）- 优先于卖出信号
tc30 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
        [1, 5000, 5025, 4975, 4987, 6172, 0, 0],  # 进入交易（最后一根K线的信号）
        [2, 4987, 5012, 4986, 4986, 6172, 0, 0],
        [3, 5010, 5012, 4986, 5010, 6172, 0, 1],  # 卖出信号
        [4, 5010, 5251, 4855, 4995, 6172, 0, 0],  # 触发ROI，执行卖出信号
        [5, 4995, 4995, 4950, 4950, 6172, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.05},
    profit_perc=0.002,
    use_exit_signal=True,
    trades=[BTrade(exit_reason=ExitType.EXIT_SIGNAL, open_tick=1, close_tick=4)],
)

# 测试 31: 移动止损应该上调，导致第3根K线触发止损
# 与tc11相同的情况 - 但第3根K线"跳空下跌" - 止损将高于K线
# 因此将使用"开盘价"
# 止损：10%，ROI：10%（不应触发），在第2根K线调整止损
tc31 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
        [1, 5000, 5100, 4950, 5100, 6172, 0, 0],
        [2, 5100, 5251, 5100, 5100, 6172, 0, 0],
        [3, 4850, 5050, 4650, 4750, 6172, 0, 0],
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.10},
    profit_perc=-0.03,
    trailing_stop=True,
    trailing_only_offset_is_reached=True,
    trailing_stop_positive_offset=0.05,
    trailing_stop_positive=0.03,
    trades=[BTrade(exit_reason=ExitType.TRAILING_STOP_LOSS, open_tick=1, close_tick=3)],
)

# 测试 32: （test 31的做空版本）移动止损应该上调，导致第3根K线触发止损
# 与tc11相同的情况 - 但第3根K线"跳空下跌" - 止损将高于K线
# 因此将使用"开盘价"
# 止损：10%，ROI：10%（不应触发），在第2根K线调整止损
tc32 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 0, 0, 1, 0],
        [1, 5000, 5050, 4890, 4890, 6172, 0, 0, 0, 0],
        [2, 4890, 4890, 4749, 4890, 6172, 0, 0, 0, 0],
        [3, 5150, 5350, 4950, 4950, 6172, 0, 0, 0, 0],
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.10},
    profit_perc=-0.03,
    trailing_stop=True,
    trailing_only_offset_is_reached=True,
    trailing_stop_positive_offset=0.05,
    trailing_stop_positive=0.03,
    trades=[
        BTrade(exit_reason=ExitType.TRAILING_STOP_LOSS, open_tick=1, close_tick=3, is_short=True)
    ],
)

# 测试 33: 移动止损应该被下一根K线的最低价触发，而不使用止损K线的最高价调整止损
# 止损：10%，ROI：10%（不应触发）
tc33 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
        [1, 5000, 5050, 5000, 5000, 6172, 0, 0],  # 进入交易（最后一根K线的信号）
        [2, 4900, 5250, 4500, 5100, 6172, 0, 0],  # 触发移动止损
        [3, 5100, 5100, 4650, 4750, 6172, 0, 0],
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.10},
    profit_perc=-0.02,
    trailing_stop=True,
    trailing_stop_positive=0.03,
    trades=[BTrade(exit_reason=ExitType.TRAILING_STOP_LOSS, open_tick=1, close_tick=2)],
)

# 测试 34: 移动止损应该在交易开盘K线立即触发
# 止损：10%，ROI：10%（不应触发）
tc34 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
        [1, 5000, 5500, 4900, 4900, 6172, 0, 0],  # 进入交易（最后一根K线的信号）并止损
        [2, 4900, 5250, 4500, 5100, 6172, 0, 0],
        [3, 5100, 5100, 4650, 4750, 6172, 0, 0],
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.10},
    profit_perc=-0.01,
    trailing_stop=True,
    trailing_stop_positive=0.01,
    trades=[BTrade(exit_reason=ExitType.TRAILING_STOP_LOSS, open_tick=1, close_tick=1)],
)

# 测试 35: 移动止损应该在交易开盘K线立即触发
# 止损：10%，ROI：10%（不应触发）
tc35 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
        [1, 5000, 5500, 4900, 4900, 6172, 0, 0],  # 进入交易（最后一根K线的信号）并止损
        [2, 4900, 5250, 4500, 5100, 6172, 0, 0],
        [3, 5100, 5100, 4650, 4750, 6172, 0, 0],
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.10},
    profit_perc=0.01,
    trailing_stop=True,
    trailing_only_offset_is_reached=True,
    trailing_stop_positive_offset=0.02,
    trailing_stop_positive=0.01,
    trades=[BTrade(exit_reason=ExitType.TRAILING_STOP_LOSS, open_tick=1, close_tick=1)],
)

# 测试 36: 移动止损应该在交易开盘K线立即触发
# 止损：1%，ROI：10%（不应触发）
tc36 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
        [1, 5000, 5500, 4951, 5000, 6172, 0, 0],  # 进入交易并止损
        [2, 4900, 5250, 4500, 5100, 6172, 0, 0],
        [3, 5100, 5100, 4650, 4750, 6172, 0, 0],
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0],
    ],
    stop_loss=-0.01,
    roi={"0": 0.10},
    profit_perc=-0.01,
    trailing_stop=True,
    trailing_only_offset_is_reached=True,
    trailing_stop_positive_offset=0.02,
    trailing_stop_positive=0.01,
    use_custom_stoploss=True,
    trades=[BTrade(exit_reason=ExitType.TRAILING_STOP_LOSS, open_tick=1, close_tick=1)],
)

# 测试 37: 移动止损应该在交易开盘K线立即触发
# 止损：1%，ROI：10%（不应触发）
tc37 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0, 0, 0, "buy_signal_01"],
        [1, 5000, 5500, 4951, 5000, 6172, 0, 0, 0, 0, None],  # 进入交易并止损
        [2, 4900, 5250, 4500, 5100, 6172, 0, 0, 0, 0, None],
        [3, 5100, 5100, 4650, 4750, 6172, 0, 0, 0, 0, None],
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0, 0, 0, None],
    ],
    stop_loss=-0.01,
    roi={"0": 0.10},
    profit_perc=-0.01,
    trailing_stop=True,
    trailing_only_offset_is_reached=True,
    trailing_stop_positive_offset=0.02,
    trailing_stop_positive=0.01,
    use_custom_stoploss=True,
    trades=[
        BTrade(
            exit_reason=ExitType.TRAILING_STOP_LOSS,
            open_tick=1,
            close_tick=1,
            enter_tag="buy_signal_01",
        )
    ],
)
# 测试 38: 移动止损应该在交易开盘K线立即触发
# Test37的做空版本
# 止损：1%，ROI：10%（不应触发）
tc38 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 0, 0, 1, 0, "short_signal_01"],
        [1, 5000, 5049, 4500, 5000, 6172, 0, 0, 0, 0, None],  # 进入交易并止损
        [2, 4900, 5250, 4500, 5100, 6172, 0, 0, 0, 0, None],
        [3, 5100, 5100, 4650, 4750, 6172, 0, 0, 0, 0, None],
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0, 0, 0, None],
    ],
    stop_loss=-0.01,
    roi={"0": 0.10},
    profit_perc=-0.01,
    trailing_stop=True,
    trailing_only_offset_is_reached=True,
    trailing_stop_positive_offset=0.02,
    trailing_stop_positive=0.01,
    use_custom_stoploss=True,
    trades=[
        BTrade(
            exit_reason=ExitType.TRAILING_STOP_LOSS,
            open_tick=1,
            close_tick=1,
            enter_tag="short_signal_01",
            is_short=True,
        )
    ],
)

# 测试 39: 自定义入场价低于所有K线应该超时 - 所以不会发生交易
tc39 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
        [1, 5000, 5500, 4951, 5000, 6172, 0, 0],  # 超时
        [2, 4900, 5250, 4500, 5100, 6172, 0, 0],
        [3, 5100, 5100, 4650, 4750, 6172, 0, 0],
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0],
    ],
    stop_loss=-0.01,
    roi={"0": 0.10},
    profit_perc=0.0,
    custom_entry_price=4200,
    trades=[],
)

# 测试 40: 自定义入场价高于所有K线应该将价格调整为"入场K线最高价"
tc40 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
        [1, 5000, 5500, 4951, 5000, 6172, 0, 0],  # 超时
        [2, 4900, 5250, 4500, 5100, 6172, 0, 0],
        [3, 5100, 5100, 4650, 4750, 6172, 0, 0],
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0],
    ],
    stop_loss=-0.01,
    roi={"0": 0.10},
    profit_perc=-0.01,
    custom_entry_price=7200,
    trades=[BTrade(exit_reason=ExitType.STOP_LOSS, open_tick=1, close_tick=1)],
)

# 测试 41: 自定义入场价高于所有K线应该将价格调整为"入场K线最高价"
tc41 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 0, 0, 1, 0],
        [1, 5000, 5500, 4951, 5000, 6172, 0, 0, 0, 0],  # 超时
        [2, 4900, 5250, 4500, 5100, 6172, 0, 0, 0, 0],
        [3, 5100, 5100, 4650, 4750, 6172, 0, 0, 0, 0],
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0, 0, 0],
    ],
    stop_loss=-0.01,
    roi={"0": 0.10},
    profit_perc=-0.01,
    custom_entry_price=4000,
    trades=[BTrade(exit_reason=ExitType.STOP_LOSS, open_tick=1, close_tick=1, is_short=True)],
)

# 测试 42: 自定义入场价在K线最低价附近
# 会导致立即ROI退出，但由于交易是在开盘价以下进入的
# 我们将此视为作弊，并将卖出延迟1根K线
# 详情：https://github.com/freqtrade/freqtrade/issues/6261
tc42 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
        [1, 5000, 5500, 4951, 4999, 6172, 0, 0],  # 进入并立即ROI
        [2, 4900, 5250, 4500, 5100, 6172, 0, 0],
        [3, 5100, 5100, 4650, 4750, 6172, 0, 0],
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.01},
    profit_perc=0.01,
    custom_entry_price=4952,
    trades=[BTrade(exit_reason=ExitType.ROI, open_tick=1, close_tick=2)],
)

# 测试 43: 自定义入场价在K线最低价附近
# 会导致在收盘价以下立即ROI退出
# 详情：https://github.com/freqtrade/freqtrade/issues/6261
tc43 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
        [1, 5400, 5500, 4951, 5100, 6172, 0, 0],  # 进入并立即ROI
        [2, 4900, 5250, 4500, 5100, 6172, 0, 0],
        [3, 5100, 5100, 4650, 4750, 6172, 0, 0],
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.01},
    profit_perc=0.01,
    custom_entry_price=4952,
    trades=[BTrade(exit_reason=ExitType.ROI, open_tick=1, close_tick=1)],
)

# 测试 44: 自定义退出价低于所有K线
# 价格调整为K线最低价
tc44 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
        [1, 5000, 5500, 4951, 5000, 6172, 0, 0],
        [2, 4900, 5250, 4900, 5100, 6172, 0, 1],  # 退出 - 但超时
        [3, 5100, 5100, 4950, 4950, 6172, 0, 0],
        [4, 5000, 5100, 4950, 4950, 6172, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.10},
    profit_perc=-0.01,
    use_exit_signal=True,
    custom_exit_price=4552,
    trades=[BTrade(exit_reason=ExitType.EXIT_SIGNAL, open_tick=1, close_tick=3)],
)

# 测试 45: 自定义退出价高于所有K线
# 导致卖出信号超时
tc45 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
        [1, 5000, 5500, 4951, 5000, 6172, 0, 0],
        [2, 4950, 5250, 4900, 5100, 6172, 0, 1],  # 退出 - 入场超时
        [3, 5100, 5100, 4950, 4950, 6172, 0, 0],
        [4, 5000, 5100, 4950, 4950, 6172, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.10},
    profit_perc=0.0,
    use_exit_signal=True,
    custom_exit_price=6052,
    trades=[BTrade(exit_reason=ExitType.FORCE_EXIT, open_tick=1, close_tick=4)],
)

# 测试 46: （tc45的做空版本）自定义做空退出价低于K线
# 导致卖出信号超时
tc46 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 0, 0, 1, 0],
        [1, 5000, 5000, 4951, 5000, 6172, 0, 0, 0, 0],
        [2, 4910, 5150, 4910, 5100, 6172, 0, 0, 0, 1],  # 退出 - 入场超时
        [3, 5100, 5100, 4950, 4950, 6172, 0, 0, 0, 0],
        [4, 5000, 5100, 4950, 4950, 6172, 0, 0, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.10},
    profit_perc=0.0,
    use_exit_signal=True,
    custom_exit_price=4700,
    trades=[BTrade(exit_reason=ExitType.FORCE_EXIT, open_tick=1, close_tick=4, is_short=True)],
)

# 测试 47: 多头和空头信号冲突
tc47 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0, 1, 0],
        [1, 5000, 5500, 4951, 5000, 6172, 0, 0, 0, 0],
        [2, 4900, 5250, 4900, 5100, 6172, 0, 0, 0, 0],
        [3, 5100, 5100, 4950, 4950, 6172, 0, 0, 0, 0],
        [4, 5000, 5100, 4950, 4950, 6172, 0, 0, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.10},
    profit_perc=0.0,
    use_exit_signal=True,
    trades=[],
)

# 测试 48: 自定义入场价低于所有K线 - 重新调整订单
tc48 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
        [1, 5000, 5500, 4951, 5000, 6172, 0, 0],  # 超时
        [2, 4900, 5250, 4500, 5100, 6172, 0, 0],  # 订单重新调整
        [3, 5100, 5100, 4650, 4750, 6172, 0, 1],
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0],
    ],
    stop_loss=-0.2,
    roi={"0": 0.10},
    profit_perc=-0.087,
    use_exit_signal=True,
    timeout=1000,
    custom_entry_price=4200,
    adjust_entry_price=5200,
    trades=[BTrade(exit_reason=ExitType.EXIT_SIGNAL, open_tick=1, close_tick=4, is_short=False)],
)


# 测试 49: 自定义入场价做空高于所有K线 - 重新调整订单
tc49 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 0, 0, 1, 0],
        [1, 5000, 5200, 4951, 5000, 6172, 0, 0, 0, 0],  # 超时
        [2, 4900, 5250, 4900, 5100, 6172, 0, 0, 0, 0],  # 订单重新调整
        [3, 5100, 5100, 4650, 4750, 6172, 0, 0, 0, 1],
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0, 0, 0],
    ],
    stop_loss=-0.2,
    roi={"0": 0.10},
    profit_perc=0.05,
    use_exit_signal=True,
    timeout=1000,
    custom_entry_price=5300,
    adjust_entry_price=5000,
    trades=[BTrade(exit_reason=ExitType.EXIT_SIGNAL, open_tick=1, close_tick=4, is_short=True)],
)

# 测试 50: 自定义入场价低于所有K线 - 重新调整订单取消订单
tc50 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0],  # 进入多头 - 下单
        [1, 5000, 5500, 4951, 5000, 6172, 0, 0],  # 订单重新调整 - 取消订单
        [2, 4900, 5250, 4500, 5100, 6172, 0, 0],
        [3, 5100, 5100, 4650, 4750, 6172, 0, 0],
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0],
    ],
    stop_loss=-0.01,
    roi={"0": 0.10},
    profit_perc=0.0,
    use_exit_signal=True,
    timeout=1000,
    custom_entry_price=4200,
    adjust_entry_price=None,
    trades=[],
)

# 测试 51: 自定义入场价低于所有K线 - 重新调整订单保持订单并超时
tc51 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0],  # 进入多头 - 下单
        [1, 5000, 5500, 4951, 5000, 6172, 0, 0],  # 订单重新调整 - 替换订单
        [2, 4900, 5250, 4500, 5100, 6172, 0, 0],  # 订单重新调整 - 保持订单
        [3, 5100, 5100, 4650, 4750, 6172, 0, 0],  # 超时
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0],
    ],
    stop_loss=-0.01,
    roi={"0": 0.10},
    profit_perc=0.0,
    use_exit_signal=True,
    timeout=60,
    custom_entry_price=4200,
    adjust_entry_price=4100,
    trades=[],
)

# 测试 52: 自定义入场价低于所有K线 - 重新调整订单 - 止损
tc52 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
        [1, 5000, 5500, 4951, 5000, 6172, 0, 0],  # 进入交易（最后一根K线的信号）
        [2, 4900, 5250, 4500, 5100, 6172, 0, 0],  # 订单重新调整
        [3, 5100, 5100, 4650, 4750, 6172, 0, 0],  # 触发止损？
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0],
    ],
    stop_loss=-0.03,
    roi={},
    profit_perc=-0.03,
    use_exit_signal=True,
    timeout=1000,
    custom_entry_price=4200,
    adjust_entry_price=5200,
    trades=[BTrade(exit_reason=ExitType.STOP_LOSS, open_tick=1, close_tick=2, is_short=False)],
)


# 测试 53: 自定义入场价做空高于所有K线 - 重新调整订单 - 止损
tc53 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 0, 0, 1, 0],
        [1, 5000, 5200, 4951, 5000, 6172, 0, 0, 0, 0],  # 进入交易（最后一根K线的信号）
        [2, 4900, 5250, 4900, 5100, 6172, 0, 0, 0, 0],  # 订单重新调整
        [3, 5100, 5100, 4650, 4750, 6172, 0, 0, 0, 1],  # 触发止损？
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0, 0, 0],
    ],
    stop_loss=-0.03,
    roi={"0": 0.10},
    profit_perc=-0.03,
    use_exit_signal=True,
    timeout=1000,
    custom_entry_price=5300,
    adjust_entry_price=5000,
    trades=[BTrade(exit_reason=ExitType.STOP_LOSS, open_tick=1, close_tick=2, is_short=True)],
)

# 测试 54: 从多头切换到空头
tc54 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0, 0, 0],
        [1, 5000, 5000, 4951, 5000, 6172, 0, 0, 0, 0],
        [2, 4910, 5150, 4910, 5100, 6172, 0, 0, 1, 0],  # 进入空头信号被忽略
        [3, 5100, 5100, 4950, 4950, 6172, 0, 1, 1, 0],  # 退出 - 重新进入空头
        [4, 5000, 5100, 4950, 4950, 6172, 0, 0, 0, 1],
        [5, 5000, 5100, 4950, 4950, 6172, 0, 0, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.10},
    profit_perc=0.00,
    use_exit_signal=True,
    trades=[
        BTrade(exit_reason=ExitType.EXIT_SIGNAL, open_tick=1, close_tick=4, is_short=False),
        BTrade(exit_reason=ExitType.EXIT_SIGNAL, open_tick=4, close_tick=5, is_short=True),
    ],
)

# 测试 55: 从空头切换到多头
tc55 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 0, 0, 1, 0],
        [1, 5000, 5000, 4951, 5000, 6172, 1, 0, 0, 0],  # 进入多头信号被忽略
        [2, 4910, 5150, 4910, 5100, 6172, 1, 0, 0, 1],  # 退出 - 重新进入多头
        [3, 5100, 5100, 4950, 4950, 6172, 0, 0, 0, 0],
        [4, 5000, 5100, 4950, 4950, 6172, 0, 1, 0, 0],
        [5, 5000, 5100, 4950, 4950, 6172, 0, 0, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 0.10},
    profit_perc=-0.04,
    use_exit_signal=True,
    trades=[
        BTrade(exit_reason=ExitType.EXIT_SIGNAL, open_tick=1, close_tick=3, is_short=True),
        BTrade(exit_reason=ExitType.EXIT_SIGNAL, open_tick=3, close_tick=5, is_short=False),
    ],
)

# 测试 56: 从多头切换到空头
tc56 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0, 0, 0],
        [1, 5000, 5000, 4951, 5000, 6172, 0, 0, 0, 0],
        [2, 4910, 5150, 4910, 5100, 6172, 0, 0, 1, 0],  # 止损退出 - 重新进入空头
        [3, 5100, 5100, 4888, 4950, 6172, 0, 0, 0, 0],
        [4, 5000, 5100, 4950, 4950, 6172, 0, 0, 0, 1],
        [5, 5000, 5100, 4950, 4950, 6172, 0, 0, 0, 0],
    ],
    stop_loss=-0.02,
    roi={"0": 0.10},
    profit_perc=-0.0,
    use_exit_signal=True,
    trades=[
        BTrade(exit_reason=ExitType.STOP_LOSS, open_tick=1, close_tick=3, is_short=False),
        BTrade(exit_reason=ExitType.EXIT_SIGNAL, open_tick=3, close_tick=5, is_short=True),
    ],
)


# 测试 57: 自定义入场价用于仓位调整但不会成交
# 导致负调整取消未成交订单并部分退出
tc57 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0, 0, 0],
        [1, 4598, 5200, 4498, 5000, 6172, 0, 0, 0, 0],
        [2, 4900, 5250, 4900, 5100, 6172, 0, 0, 0, 0],  # 增强仓位，但不会成交
        [3, 5100, 5100, 4650, 4750, 6172, 0, 0, 0, 0],
        [4, 4750, 4950, 4650, 4750, 6172, 0, 0, 0, 0],
        [5, 4750, 4950, 4650, 4750, 6172, 0, 1, 0, 0],
        [6, 4750, 4950, 4650, 4750, 6172, 0, 0, 0, 0],
    ],
    stop_loss=-0.2,
    roi={"0": 0.50},
    profit_perc=0.033,
    use_exit_signal=True,
    timeout=1000,
    custom_entry_price=4600,
    adjust_trade_position=[
        None,
        0.001,
        None,
        -0.0001,  # 取消上述未成交订单并部分退出
        None,
        None,
    ],
    trades=[
        BTrade(exit_reason=ExitType.EXIT_SIGNAL, open_tick=1, close_tick=6, is_short=False),
    ],
)

# 测试 58: 自定义退出价做空 - 低于所有K线
tc58 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 0, 0, 1, 0],
        [1, 5000, 5200, 4951, 5000, 6172, 0, 0, 0, 0],  # 进入交易（最后一根K线的信号）
        [2, 4900, 5250, 4900, 5100, 6172, 0, 0, 0, 1],  # 退出 - 延迟
        [3, 5100, 5100, 4650, 4750, 6172, 0, 0, 0, 0],  #
        [4, 4750, 5100, 4350, 4750, 6172, 0, 0, 0, 0],
    ],
    stop_loss=-0.10,
    roi={"0": 1.00},
    profit_perc=-0.01,
    use_exit_signal=True,
    timeout=1000,
    custom_exit_price=4300,
    adjust_exit_price=5050,
    trades=[BTrade(exit_reason=ExitType.EXIT_SIGNAL, open_tick=1, close_tick=4, is_short=True)],
)

# 测试 59: 自定义退出价高于所有K线 - 重新调整订单
tc59 = BTContainer(
    data=[
        # D   O     H     L     C    V    EL XL ES Xs  BT
        [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
        [1, 5000, 5500, 4951, 5000, 6172, 0, 0],
        [2, 4900, 5250, 4500, 5100, 6172, 0, 1],  # 退出
        [3, 5100, 5100, 4650, 4750, 6172, 0, 0],  # 订单重新调整
        [4, 4750, 4950, 4350, 4750, 6172, 0, 0],
    ],
    stop_loss=-0.2,
    roi={"0": 0.10},
    profit_perc=-0.02,
    use_exit_signal=True,
    timeout=1000,
    custom_exit_price=5300,
    adjust_exit_price=4900,
    trades=[BTrade(exit_reason=ExitType.EXIT_SIGNAL, open_tick=1, close_tick=4, is_short=False)],
)


TESTS = [
    tc0,
    tc1,
    tc2,
    tc3,
    tc4,
    tc5,
    tc6,
    tc7,
    tc8,
    tc9,
    tc10,
    tc11,
    tc12,
    tc13,
    tc14,
    tc15,
    tc16,
    tc17,
    tc18,
    tc19,
    tc20,
    tc21,
    tc22,
    tc23,
    tc24,
    tc25,
    tc26,
    tc27,
    tc28,
    tc29,
    tc30,
    tc31,
    tc32,
    tc33,
    tc34,
    tc35,
    tc36,
    tc37,
    tc38,
    tc39,
    tc40,
    tc41,
    tc42,
    tc43,
    tc44,
    tc45,
    tc46,
    tc47,
    tc48,
    tc49,
    tc50,
    tc51,
    tc52,
    tc53,
    tc54,
    tc55,
    tc56,
    tc57,
    tc58,
    tc59,
]


@pytest.mark.parametrize("data", TESTS)
def test_backtest_results(default_conf, mocker, caplog, data: BTContainer) -> None:
    """
    运行功能测试
    """
    default_conf["stoploss"] = data.stop_loss
    default_conf["minimal_roi"] = data.roi
    default_conf["timeframe"] = tests_timeframe
    default_conf["trailing_stop"] = data.trailing_stop
    default_conf["trailing_only_offset_is_reached"] = data.trailing_only_offset_is_reached
    if data.timeout:
        default_conf["unfilledtimeout"].update(
            {
                "entry": data.timeout,
                "exit": data.timeout,
            }
        )
    # 仅在必要时将其添加到配置中
    if data.trailing_stop_positive is not None:
        default_conf["trailing_stop_positive"] = data.trailing_stop_positive
    default_conf["trailing_stop_positive_offset"] = data.trailing_stop_positive_offset
    default_conf["use_exit_signal"] = data.use_exit_signal
    default_conf["max_open_trades"] = 10

    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.get_fee", return_value=0.0)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    mocker.patch(f"{EXMS}.get_max_leverage", return_value=100)
    mocker.patch(f"{EXMS}.calculate_funding_fees", return_value=0)
    frame = _build_backtest_dataframe(data.data)
    backtesting = Backtesting(default_conf)
    # TODO: 我们应该正确初始化这个吗？？
    backtesting.trading_mode = TradingMode.MARGIN
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting._can_short = True
    backtesting.required_startup = 0
    backtesting.strategy.advise_entry = lambda a, m: frame
    backtesting.strategy.advise_exit = lambda a, m: frame
    if data.custom_entry_price:
        backtesting.strategy.custom_entry_price = MagicMock(return_value=data.custom_entry_price)
    if data.custom_exit_price:
        backtesting.strategy.custom_exit_price = MagicMock(return_value=data.custom_exit_price)
    if data.adjust_trade_position:
        backtesting.strategy.position_adjustment_enable = True
        backtesting.strategy.adjust_trade_position = MagicMock(
            side_effect=data.adjust_trade_position
        )
    if data.adjust_entry_price:
        backtesting.strategy.adjust_entry_price = MagicMock(return_value=data.adjust_entry_price)
    if data.adjust_exit_price:
        backtesting.strategy.adjust_exit_price = MagicMock(return_value=data.adjust_exit_price)

    backtesting.strategy.use_custom_stoploss = data.use_custom_stoploss
    backtesting.strategy.leverage = lambda **kwargs: data.leverage
    caplog.set_level(logging.DEBUG)

    pair = "UNITTEST/BTC"
    # 虚拟数据，因为我们模拟了分析函数
    data_processed = {pair: frame.copy()}
    min_date, max_date = get_timerange({pair: frame})
    result = backtesting.backtest(
        processed=data_processed,
        start_date=min_date,
        end_date=max_date,
    )

    results = result["results"]
    assert len(results) == len(data.trades)
    assert round(results["profit_ratio"].sum(), 3) == round(data.profit_perc, 3)

    for c, trade in enumerate(data.trades):
        res: BTrade = results.iloc[c]
        assert res.exit_reason == trade.exit_reason.value
        assert res.enter_tag == trade.enter_tag
        assert res.open_date == _get_frame_time_from_offset(trade.open_tick)
        assert res.close_date == _get_frame_time_from_offset(trade.close_tick)
        assert res.is_short == trade.is_short
    assert len(LocalTrade.bt_trades) == len(data.trades)
    assert len(LocalTrade.bt_trades_open) == 0, "剩余未平仓交易"
    backtesting.cleanup()
    del backtesting