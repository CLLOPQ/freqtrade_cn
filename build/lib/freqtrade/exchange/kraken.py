"""Kraken 交易所子类"""

import logging
from datetime import datetime
from typing import Any

import ccxt
from pandas import DataFrame

from freqtrade.constants import BuySell
from freqtrade.enums import MarginMode, TradingMode
from freqtrade.exceptions import DDosProtection, OperationalException, TemporaryError
from freqtrade.exchange import Exchange
from freqtrade.exchange.common import retrier
from freqtrade.exchange.exchange_types import CcxtBalances, FtHas


logger = logging.getLogger(__name__)


class Kraken(Exchange):
    _params: dict = {"trading_agreement": "agree"}  # 交易协议参数
    _ft_has: FtHas = {
        "stoploss_on_exchange": True,  # 支持交易所止损
        "stop_price_param": "stopLossPrice",  # 止损价格参数名
        "stop_price_prop": "stopLossPrice",  # 止损价格属性名
        "stoploss_order_types": {"limit": "limit", "market": "market"},  # 止损订单类型映射
        "order_time_in_force": ["GTC", "IOC", "PO"],  # 订单有效期类型
        "ohlcv_has_history": False,  # 不支持OHLCV历史数据
        "trades_pagination": "id",  # 交易历史分页方式为ID
        "trades_pagination_arg": "since",  # 交易历史分页参数名
        "trades_pagination_overlap": False,  # 分页无重叠
        "trades_has_history": True,  # 支持交易历史
        "mark_ohlcv_timeframe": "4h",  # 标记价格K线时间框架
    }

    _supported_trading_mode_margin_pairs: list[tuple[TradingMode, MarginMode]] = [
        # TradingMode.SPOT始终受支持，无需在此列表中
        # (TradingMode.MARGIN, MarginMode.CROSS),
        # (TradingMode.FUTURES, MarginMode.CROSS)
    ]

    def market_is_tradable(self, market: dict[str, Any]) -> bool:
        """
        检查市场交易对是否可被Freqtrade交易。
        除默认检查外，还检查交易对是否为暗池交易对。
        """
        父类检查结果 = super().market_is_tradable(market)

        return 父类检查结果 and market.get("darkpool", False) is False

    def consolidate_balances(self, balances: CcxtBalances) -> CcxtBalances:
        """
        合并相同货币的余额。
        如果启用了奖励功能，Kraken会返回".F"后缀的余额。
        """
        合并后的余额: CcxtBalances = {}
        for 货币, 余额 in balances.items():
            基础货币 = 货币[:-2] if 货币.endswith(".F") else 货币

            if 基础货币 in 合并后的余额:
                合并后的余额[基础货币]["free"] += 余额["free"]
                合并后的余额[基础货币]["used"] += 余额["used"]
                合并后的余额[基础货币]["total"] += 余额["total"]
            else:
                合并后的余额[基础货币] = 余额
        return 合并后的余额

    @retrier
    def get_balances(self) -> CcxtBalances:
        if self._config["dry_run"]:
            return {}

        try:
            余额 = self._api.fetch_balance()
            # 移除ccxt结果中的额外信息
            余额.pop("info", None)
            余额.pop("free", None)
            余额.pop("total", None)
            余额.pop("used", None)
            self._log_exchange_response("fetch_balances", 余额)

            # 合并余额
            余额 = self.consolidate_balances(余额)

            订单列表 = self._api.fetch_open_orders()
            订单信息 = [
                (
                    x["symbol"].split("/")[0 if x["side"] == "sell" else 1],
                    x["remaining"] if x["side"] == "sell" else x["remaining"] * x["price"],
                    # 不要删除下面的注释，这对调试很重要
                    # x["side"], x["amount"],
                )
                for x in 订单列表
                if x["remaining"] is not None and (x["side"] == "sell" or x["price"] is not None)
            ]
            for 货币 in 余额:
                if not isinstance(余额[货币], dict):
                    continue
                余额[货币]["used"] = sum(订单[1] for 订单 in 订单信息 if 订单[0] == 货币)
                余额[货币]["free"] = 余额[货币]["total"] - 余额[货币]["used"]

            self._log_exchange_response("fetch_balances2", 余额)
            return 余额
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"由于 {e.__class__.__name__} 无法获取余额。消息: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def _set_leverage(
        self,
        leverage: float,
        pair: str | None = None,
        accept_fail: bool = False,
    ):
        """
        Kraken将杠杆设置为订单对象中的一个选项，因此我们需要
        将其添加到参数中
        """
        return

    def _get_params(
        self,
        side: BuySell,
        ordertype: str,
        leverage: float,
        reduceOnly: bool,
        time_in_force: str = "GTC",
    ) -> dict:
        params = super()._get_params(
            side=side,
            ordertype=ordertype,
            leverage=leverage,
            reduceOnly=reduceOnly,
            time_in_force=time_in_force,
        )
        if leverage > 1.0:
            params["leverage"] = round(leverage)
        if time_in_force == "PO":
            params.pop("timeInForce", None)
            params["postOnly"] = True
        return params

    def calculate_funding_fees(
        self,
        df: DataFrame,
        amount: float,
        is_short: bool,
        open_date: datetime,
        close_date: datetime,
        time_in_ratio: float | None = None,
    ) -> float:
        """
        # ! 当Freqtrade运行此方法时，它总是会出错，因为time_in_ratio从未
        # ! 传递给_get_funding_fee。要使Kraken期货在回测和模拟交易中工作
        # ! 必须添加功能，在使用Kraken时将参数time_in_ratio传递给
        # ! _get_funding_fee
        计算期货交易期间该交易对产生的所有资金费用的总和
        :param df: 包含资金费率和标记价格的DataFrame
                   分别作为`open_fund`和`open_mark`。
        :param amount: 交易数量
        :param is_short: 交易方向
        :param open_date: 交易开始的日期和时间
        :param close_date: 交易结束的日期和时间
        :param time_in_ratio: 大多数交易所类不使用
        """
        if not time_in_ratio:
            raise OperationalException(
                f"{self.name}._get_funding_fee 需要 time_in_ratio 参数"
            )
        费用: float = 0

        if not df.empty:
            df = df[(df["date"] >= open_date) & (df["date"] <= close_date)]
            费用 = sum(df["open_fund"] * df["open_mark"] * amount * time_in_ratio)

        return 费用 if is_short else -费用

    def _get_trade_pagination_next_value(self, trades: list[dict]):
        """
        提取下一个"from_id"值的分页ID
        仅适用于按ID获取交易历史。
        """
        if len(trades) > 0:
            if isinstance(trades[-1].get("info"), list) and len(trades[-1].get("info", [])) > 7:
                # 交易响应中的"last"值。
                return trades[-1].get("info", [])[-1]
            # 如果info为空，回退到时间戳。
            return trades[-1].get("timestamp")
        return None

    def _valid_trade_pagination_id(self, pair: str, from_id: str) -> bool:
        """
        验证交易分页ID是否有效。
        解决Kraken偶尔出现的ID错误问题。
        """
        # 常规ID采用时间戳格式 1705443695120072285
        # 如果ID长度小于19个字符，则不是有效的时间戳。
        if len(from_id) >= 19:
            return True
        logger.debug(f"{pair} - 交易分页ID无效。回退到时间戳。")
        return False