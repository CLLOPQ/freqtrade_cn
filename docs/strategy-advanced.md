from freqtrade.persistence import Trade
from datetime import timedelta

class AwesomeStrategy(IStrategy):

    def bot_loop_start(self, **kwargs) -> None:
        # 遍历所有有未结订单的交易
        for trade in Trade.get_open_order_trades():
            # 选择交易入场方向的已成交订单
            fills = trade.select_filled_orders(trade.entry_side)
            if trade.pair == 'ETH/USDT':
                # 获取交易的入场类型自定义数据
                trade_entry_type = trade.get_custom_data(key='entry_type')
                if trade_entry_type is None:
                    # 如果入场标签包含'entry_1'，则入场类型为'突破'，否则为'回调'
                    trade_entry_type = 'breakout' if 'entry_1' in trade.enter_tag else 'dip'
                elif fills > 1:
                    # 如果成交次数大于1，入场类型为'追涨'
                    trade_entry_type = 'buy_up'
                # 设置交易的入场类型自定义数据
                trade.set_custom_data(key='entry_type', value=trade_entry_type)
        return super().bot_loop_start(**kwargs)

    def adjust_entry_price(self, trade: Trade, order: Order | None, pair: str,
                           current_time: datetime, proposed_rate: float, current_order_rate: float,
                           entry_tag: str | None, side: str, **kwargs) -> float:
        # 对于BTC/USDT交易对，在入场触发后的前10分钟内，限价单使用并跟随SMA200作为价格目标。
        if (
            pair == 'BTC/USDT' 
            and entry_tag == 'long_sma200' 
            and side == 'long' 
            and (current_time - timedelta(minutes=10)) > trade.open_date_utc 
            and order.filled == 0.0
        ):
            # 获取分析后的交易对数据框
            dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
            # 获取最后一根K线
            current_candle = dataframe.iloc[-1].squeeze()
            # 存储入场价格调整次数的信息
            existing_count = trade.get_custom_data('num_entry_adjustments', default=0)
            if not existing_count:
                existing_count = 1
            else:
                existing_count += 1
            trade.set_custom_data(key='num_entry_adjustments', value=existing_count)

            # 调整订单价格
            return current_candle['sma_200']

        # 默认：维持现有订单价格
        return current_order_rate

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs):

        entry_adjustment_count = trade.get_custom_data(key='num_entry_adjustments')
        trade_entry_type = trade.get_custom_data(key='entry_type')
        if entry_adjustment_count is None:
            # 如果没有入场调整次数数据，且利润大于1%且交易已开仓超过100分钟，则退出
            if current_profit > 0.01 and (current_time - timedelta(minutes=100) > trade.open_date_utc):
                return True, 'exit_1'
        else:
            # 如果入场调整次数大于0且利润大于5%，则退出
            if entry_adjustment_count > 0 and current_profit > 0.05:
                return True, 'exit_2'
            # 如果入场类型为突破且利润大于10%，则退出
            if trade_entry_type == 'breakout' and current_profit > 0.1:
                return True, 'exit_3'

        return False, None