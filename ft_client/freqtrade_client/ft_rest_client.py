"""
Freqtrade机器人的REST客户端

不应从freqtrade导入任何内容，
因此它可以用作独立脚本，并且可以独立安装。
"""

import json
import logging
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError as RequestConnectionError


logger = logging.getLogger("ft_rest_client")

ParamsT = dict[str, Any] | None
PostDataT = dict[str, Any] | list[dict[str, Any]] | None


class FtRestClient:
    def __init__(
        self,
        serverurl,
        username=None,
        password=None,
        *,
        pool_connections=10,
        pool_maxsize=10,
        timeout=10,
    ):
        self._serverurl = serverurl
        self._session = requests.Session()
        self._timeout = timeout

        # 允许配置连接池
        adapter = HTTPAdapter(pool_connections=pool_connections, pool_maxsize=pool_maxsize)
        self._session.mount("http://", adapter)

        if username and password:
            self._session.auth = (username, password)

    def _call(self, method, apipath, params: dict | None = None, data=None, files=None):
        if str(method).upper() not in ("GET", "POST", "PUT", "DELETE"):
            raise ValueError(f"无效的方法 <{method}>")
        basepath = f"{self._serverurl}/api/v1/{apipath}"

        hd = {"Accept": "application/json", "Content-Type": "application/json"}

        # 拆分URL
        schema, netloc, path, par, query, fragment = urlparse(basepath)
        # URL编码查询字符串
        query = urlencode(params) if params else ""
        # 重组URL
        url = urlunparse((schema, netloc, path, par, query, fragment))

        try:
            resp = self._session.request(
                method, url, headers=hd, timeout=self._timeout, data=json.dumps(data)
            )
            return resp.json()
        except RequestConnectionError:
            logger.warning(f"连接错误 - 无法连接到 {netloc}。")

    def _get(self, apipath, params: ParamsT = None):
        return self._call("GET", apipath, params=params)

    def _delete(self, apipath, params: ParamsT = None):
        return self._call("DELETE", apipath, params=params)

    def _post(self, apipath, params: ParamsT = None, data: PostDataT = None):
        return self._call("POST", apipath, params=params, data=data)

    def start(self):
        """如果机器人处于停止状态，则启动它。

        :return: json对象
        """
        return self._post("start")

    def stop(self):
        """停止机器人。使用`start`重新启动。

        :return: json对象
        """
        return self._post("stop")

    def stopbuy(self):
        """停止买入（但正常处理卖出）。使用`reload_config`重置。

        :return: json对象
        """
        return self._post("stopbuy")

    def reload_config(self):
        """重新加载配置。

        :return: json对象
        """
        return self._post("reload_config")

    def balance(self):
        """获取账户余额。

        :return: json对象
        """
        return self._get("balance")

    def count(self):
        """返回当前未平仓交易的数量。

        :return: json对象
        """
        return self._get("count")

    def entries(self, pair=None):
        """返回基于买入标签表现的所有交易的字典列表
        可以是所有交易对的平均值，或指定的特定交易对

        :return: json对象
        """
        return self._get("entries", params={"pair": pair} if pair else None)

    def exits(self, pair=None):
        """返回基于卖出原因表现的所有交易的字典列表
        可以是所有交易对的平均值，或指定的特定交易对

        :return: json对象
        """
        return self._get("exits", params={"pair": pair} if pair else None)

    def mix_tags(self, pair=None):
        """返回基于入场标签+出场原因表现的所有交易的字典列表
        可以是所有交易对的平均值，或指定的特定交易对

        :return: json对象
        """
        return self._get("mix_tags", params={"pair": pair} if pair else None)

    def locks(self):
        """返回当前的交易对锁定情况

        :return: json对象
        """
        return self._get("locks")

    def delete_lock(self, lock_id):
        """从数据库中删除（禁用）锁定。

        :param lock_id: 要删除的锁定ID
        :return: json对象
        """
        return self._delete(f"locks/{lock_id}")

    def lock_add(self, pair: str, until: str, side: str = "*", reason: str = ""):
        """锁定交易对

        :param pair: 要锁定的交易对
        :param until: 锁定到该日期（格式 "2024-03-30 16:00:00Z"）
        :param side: 要锁定的方向（long, short, *）
        :param reason: 锁定原因
        :return: json对象
        """
        data = [{"pair": pair, "until": until, "side": side, "reason": reason}]
        return self._post("locks", data=data)

    def daily(self, days=None):
        """返回每天的利润和交易数量。

        :return: json对象
        """
        return self._get("daily", params={"timescale": days} if days else None)

    def weekly(self, weeks=None):
        """返回每周的利润和交易数量。

        :return: json对象
        """
        return self._get("weekly", params={"timescale": weeks} if weeks else None)

    def monthly(self, months=None):
        """返回每月的利润和交易数量。

        :return: json对象
        """
        return self._get("monthly", params={"timescale": months} if months else None)

    def profit(self):
        """返回利润摘要。

        :return: json对象
        """
        return self._get("profit")

    def stats(self):
        """返回统计报告（持续时间、卖出原因）。

        :return: json对象
        """
        return self._get("stats")

    def performance(self):
        """返回不同币种的表现。

        :return: json对象
        """
        return self._get("performance")

    def status(self):
        """获取未平仓交易的状态。

        :return: json对象
        """
        return self._get("status")

    def version(self):
        """返回机器人的版本。

        :return: 包含版本的json对象
        """
        return self._get("version")

    def show_config(self):
        """返回部分配置，与交易操作相关。
        :return: 包含配置的json对象
        """
        return self._get("show_config")

    def ping(self):
        """简单的心跳检测"""
        configstatus = self.show_config()
        if not configstatus:
            return {"status": "未运行"}
        elif configstatus["state"] == "running":
            return {"status": "pong"}
        else:
            return {"status": "未运行"}

    def logs(self, limit=None):
        """显示最新日志。

        :param limit: 将日志消息限制为最后<limit>条。不限制则获取全部日志。
        :return: json对象
        """
        return self._get("logs", params={"limit": limit} if limit else {})

    def trades(self, limit=None, offset=None, order_by_id=True):
        """返回交易历史，按id排序（如果order_by_id=False则按最新时间戳排序）

        :param limit: 将交易限制为最近的X笔。最多500笔交易。
        :param offset: 按此数量的交易偏移。
        :param order_by_id: 按id排序交易（默认：True）。如果为False，则按最新时间戳排序。
        :return: json对象
        """
        params = {}
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        if not order_by_id:
            params["order_by_id"] = False
        return self._get("trades", params)

    def list_open_trades_custom_data(self, key=None, limit=100, offset=0):
        """列出运行中机器人的未平仓交易自定义数据。

        :param key: str, 可选 - 自定义数据的键
        :param limit: 交易数量限制
        :param offset: 分页的交易偏移量
        :return: json对象
        """
        params = {}
        params["limit"] = limit
        params["offset"] = offset
        if key is not None:
            params["key"] = key

        return self._get("trades/open/custom-data", params=params)

    def list_custom_data(self, trade_id, key=None):
        """列出运行中机器人特定交易的自定义数据。

        :param trade_id: 交易ID
        :param key: str, 可选 - 自定义数据的键
        :return: JSON对象
        """
        params = {}
        params["trade_id"] = trade_id
        if key is not None:
            params["key"] = key

        return self._get(f"trades/{trade_id}/custom-data", params=params)

    def trade(self, trade_id):
        """返回特定交易

        :param trade_id: 指定要获取的交易。
        :return: json对象
        """
        return self._get(f"trade/{trade_id}")

    def delete_trade(self, trade_id):
        """从数据库中删除交易。
        尝试关闭未平仓订单。需要手动处理交易所上的该资产。

        :param trade_id: 从数据库中删除具有此ID的交易。
        :return: json对象
        """
        return self._delete(f"trades/{trade_id}")

    def cancel_open_order(self, trade_id):
        """取消交易的未平仓订单。

        :param trade_id: 取消此交易的未平仓订单。
        :return: json对象
        """
        return self._delete(f"trades/{trade_id}/open-order")

    def whitelist(self):
        """显示当前的白名单。

        :return: json对象
        """
        return self._get("whitelist")

    def blacklist(self, *args):
        """显示当前的黑名单。

        :param add: 要添加的币种列表（例如："BNB/BTC"）
        :return: json对象
        """
        if not args:
            return self._get("blacklist")
        else:
            return self._post("blacklist", data={"blacklist": args})

    def forcebuy(self, pair, price=None):
        """购买资产。

        :param pair: 要购买的交易对（ETH/BTC）
        :param price: 可选 - 购买价格
        :return: 交易的json对象
        """
        data = {"pair": pair, "price": price}
        return self._post("forcebuy", data=data)

    def forceenter(
        self,
        pair,
        side,
        price=None,
        *,
        order_type=None,
        stake_amount=None,
        leverage=None,
        enter_tag=None,
    ):
        """强制进入交易

        :param pair: 要购买的交易对（ETH/BTC）
        :param side: 'long' 或 'short'
        :param price: 可选 - 购买价格
        :param order_type: 可选关键字参数 - 'limit' 或 'market'
        :param stake_amount: 可选关键字参数 - 投注金额（作为浮点数）
        :param leverage: 可选关键字参数 - 杠杆（作为浮点数）
        :param enter_tag: 可选关键字参数 - 入场标签（作为字符串，默认：'force_enter'）
        :return: 交易的json对象
        """
        data = {
            "pair": pair,
            "side": side,
        }

        if price:
            data["price"] = price

        if order_type:
            data["ordertype"] = order_type

        if stake_amount:
            data["stakeamount"] = stake_amount

        if leverage:
            data["leverage"] = leverage

        if enter_tag:
            data["entry_tag"] = enter_tag

        return self._post("forceenter", data=data)

    def forceexit(self, tradeid, ordertype=None, amount=None):
        """强制退出交易。

        :param tradeid: 交易ID（可通过status命令获取）
        :param ordertype: 要使用的订单类型（必须是market或limit）
        :param amount: 要卖出的数量。不指定则全部卖出
        :return: json对象
        """

        return self._post(
            "forceexit",
            data={
                "tradeid": tradeid,
                "ordertype": ordertype,
                "amount": amount,
            },
        )

    def strategies(self):
        """列出可用策略

        :return: json对象
        """
        return self._get("strategies")

    def strategy(self, strategy):
        """获取策略详情

        :param strategy: 策略类名
        :return: json对象
        """
        return self._get(f"strategy/{strategy}")

    def pairlists_available(self):
        """列出可用的交易对列表提供者

        :return: json对象
        """
        return self._get("pairlists/available")

    def plot_config(self):
        """如果策略定义了绘图配置，则返回它。

        :return: json对象
        """
        return self._get("plot_config")

    def available_pairs(self, timeframe=None, stake_currency=None):
        """基于时间框架/基础货币选择返回可用交易对（回测数据）

        :param timeframe: 仅返回有此时间框架的交易对。
        :param stake_currency: 仅返回包含此基础货币的交易对
        :return: json对象
        """
        return self._get(
            "available_pairs",
            params={
                "stake_currency": stake_currency if timeframe else "",
                "timeframe": timeframe if timeframe else "",
            },
        )

    def pair_candles(self, pair, timeframe, limit=None, columns=None):
        """返回<交易对><时间框架>的实时数据框。

        :param pair: 要获取数据的交易对
        :param timeframe: 时间框架
        :param limit: 将结果限制为最后n根K线。
        :param columns: 要返回的数据框列列表。空列表将返回OHLCV。
        :return: json对象
        """
        params = {
            "pair": pair,
            "timeframe": timeframe,
        }
        if limit:
            params["limit"] = limit

        if columns is not None:
            params["columns"] = columns
            return self._post("pair_candles", data=params)

        return self._get("pair_candles", params=params)

    def pair_history(self, pair, timeframe, strategy, timerange=None, freqaimodel=None):
        """返回历史分析数据框

        :param pair: 要获取数据的交易对
        :param timeframe: 时间框架
        :param strategy: 用于分析和获取值的策略
        :param freqaimodel: 用于分析的FreqAI模型
        :param timerange: 要获取数据的时间范围（与--timerange端点相同的格式）
        :return: json对象
        """
        return self._get(
            "pair_history",
            params={
                "pair": pair,
                "timeframe": timeframe,
                "strategy": strategy,
                "freqaimodel": freqaimodel,
                "timerange": timerange if timerange else "",
            },
        )

    def sysinfo(self):
        """提供系统信息（CPU、RAM使用情况）

        :return: json对象
        """
        return self._get("sysinfo")

    def health(self):
        """提供运行中机器人的快速健康检查。

        :return: json对象
        """
        return self._get("health")