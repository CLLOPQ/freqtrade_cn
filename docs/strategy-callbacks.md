在超参数优化期间，此函数仅在启动时运行一次。

## 机器人循环开始

一个简单的回调函数，在实盘/模拟交易模式下的每次机器人节流迭代开始时调用一次（大约每5秒，除非配置不同），或在回测/超参数优化模式下每个K线调用一次。
可用于执行与交易对无关的计算（适用于所有交易对）、加载外部数据等。
class AwesomeStrategy(IStrategy):

    # ... populate_* 方法

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool, 
                        **kwargs) -> float | None:

        # 确保最长的时间间隔条件放在最前面 - 这些条件是从上到下依次评估的。
        if current_time - timedelta(minutes=120) > trade.open_date_utc:
            return -0.05 * trade.leverage
        elif current_time - timedelta(minutes=60) > trade.open_date_utc:
            return -0.10 * trade.leverage
        return None

class AwesomeStrategy(IStrategy):

    # ... populate_* 方法

    # 将未成交超时设置为25小时，因为下方的最大超时为24小时。
    unfilledtimeout = {
        "entry": 60 * 25,
        "exit": 60 * 25
    }

    def check_entry_timeout(self, pair: str, trade: Trade, order: Order,
                            current_time: datetime, **kwargs) -> bool:
        ob = self.dp.orderbook(pair, 1)
        current_price = ob["bids"][0][0]
        # 如果当前价格比订单价格高2%以上，则取消买入订单。
        if current_price > order.price * 1.02:
            return True
        return False


    def check_exit_timeout(self, pair: str, trade: Trade, order: Order,
                           current_time: datetime, **kwargs) -> bool:
        ob = self.dp.orderbook(pair, 1)
        current_price = ob["asks"][0][0]
        # 如果当前价格比订单价格低2%以上，则取消卖出订单。
        if current_price < order.price * 0.98:
            return True
        return False


---

## 机器人订单确认

确认交易入场/出场。
这是下单前最后调用的方法。

### 交易入场（买入订单）确认

`confirm_trade_entry()` 可用于在最后一刻中止交易入场（可能因为价格不符合预期）。
:param current_profit: 当前利润（比率），基于current_rate计算（与current_entry_profit相同）。
        :param min_stake: 交易所允许的最小持仓金额（适用于入场和出场）
        :param max_stake: 允许的最大持仓金额（受余额或交易所限制）。
        :param current_entry_rate: 使用入场定价的当前价格。
        :param current_exit_rate: 使用出场定价的当前价格。
        :param current_entry_profit: 使用入场定价计算的当前利润。
        :param current_exit_profit: 使用出场定价计算的当前利润。
        :param **kwargs: 确保保留此项，以便后续更新不会破坏策略。
        :return float: 调整交易的持仓金额，
                       正值表示增加头寸，负值表示减少头寸。
                       返回None表示不执行操作。
                       可选地，返回一个元组，第二个元素为订单原因
        """
        if trade.has_open_orders:
            # 仅在没有未成交订单时操作
            return

        if current_profit > 0.05 and trade.nr_of_successful_exits == 0:
            # 在盈利+5%时获利了结一半
            return -(trade.stake_amount / 2), "half_profit_5%"

        if current_profit > -0.05:
            return None

        # 获取交易对的 dataframe（仅展示如何访问）
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        # 仅在价格未持续下跌时买入。
        last_candle = dataframe.iloc[-1].squeeze()
        previous_candle = dataframe.iloc[-2].squeeze()
        if last_candle["close"] < previous_candle["close"]:
            return None

        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries
        # 允许最多3次额外的、逐步增大的买入（总共4次）
        # 初始买入为1倍
        # 如果跌至-5%利润，买入1.25倍更多，平均利润应大致增加到-2.2%
        # 如果再次跌至-5%，买入1.5倍更多
        # 如果再次跌至-5%，买入1.75倍更多
        # 此交易的总持仓金额为初始允许持仓的1 + 1.25 + 1.5 + 1.75 = 5.5倍
        # 这就是为什么max_dca_multiplier设为5.5
        # 希望你有足够深的钱包！
        try:
            # 返回首次订单的持仓金额
            stake_amount = filled_entries[0].stake_amount_filled
            # 计算当前加仓订单的金额
            stake_amount = stake_amount * (1 + (count_of_entries * 0.25))
            return stake_amount, "1/3rd_increase"
        except Exception as exception:
            return None

        return None
策略随后可以返回一个 `AnnotationType` 对象列表，以便在图表上显示。根据返回的内容，图表可以显示水平区域、垂直区域或方框。

完整的对象结构如下：