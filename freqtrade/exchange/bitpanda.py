"""Bitpanda 交易所子类"""

import logging
from datetime import datetime, timezone

from freqtrade.exchange import Exchange


logger = logging.getLogger(__name__)


class Bitpanda(Exchange):
    """
    Bitpanda 交易所类。包含使 Freqtrade 能够与该交易所正常工作所需的调整。
    """

    def get_trades_for_order(
        self, order_id: str, pair: str, since: datetime, params: dict | None = None
    ) -> list:
        """
        使用 "fetch_my_trades" 端点获取订单，并按订单ID过滤。
        传入的 "since" 参数来自数据库，是带时区的UTC datetime对象。
        Python文档说明：
            >  naive datetime实例被假定为表示本地时间
        因此，调用 "since.timestamp()" 将获得UTC时间戳，这是在从本地时区转换到UTC之后。
        这对于UTC+时区有效，因为结果将包含过去几小时的交易而不是过去5秒的交易，
        但对于UTC-时区会失败，因为此时我们会请求未来时间的交易。

        :param order_id: 创建订单时给出的订单ID
        :param pair: 订单对应的交易对
        :param since: 订单创建时间的datetime对象。假设该对象为UTC时间。
        """
        params = {"to": int(datetime.now(timezone.utc).timestamp() * 1000)}
        return super().get_trades_for_order(order_id, pair, since, params)