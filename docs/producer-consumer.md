# 生产者/消费者模式

freqtrade提供了一种机制，通过消息WebSocket，一个实例（也称为`consumer`，消费者）可以监听上游freqtrade实例（也称为`producer`，生产者）发送的消息。主要包括`analyzed_df`（已分析数据框）和`whitelist`（白名单）消息。这允许在多个机器人中重用为交易对计算的指标（和信号），而无需多次计算。

有关设置消息WebSocket的`api_server`配置（即生产者配置），请参见REST API文档中的[消息WebSocket](rest-api.md#message-websocket)部分。

!!! 注意
    我们强烈建议将`ws_token`设置为随机且仅自己知道的值，以避免对您的机器人的未授权访问。

## 配置

要启用订阅实例，请在消费者的配置文件中添加`external_message_consumer`部分。