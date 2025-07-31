# 订单流数据

本指南将逐步介绍如何在Freqtrade中利用公共交易数据进行高级订单流分析。

!!! Warning "实验性功能"
    订单流功能目前处于测试阶段（beta），未来版本可能会有变更。如有任何问题或反馈，请在 [Freqtrade GitHub 仓库](https://github.com/freqtrade/freqtrade/issues) 报告。
    目前此功能尚未与freqAI一起测试 - 现阶段不建议将这两个功能结合使用。

!!! Warning "性能"
    订单流需要原始交易数据。此类数据量较大，当Freqtrade需要下载过去X根K线的交易数据时，可能会导致初始启动变慢。此外，启用此功能会增加内存使用量。请确保有足够的系统资源。

## 开始使用

### 启用公共交易数据

在您的 `config.json` 文件中，将 `exchange` 部分下的 `use_public_trades` 选项设置为 true。