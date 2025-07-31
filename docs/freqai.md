![freqai-logo](assets/freqai_doc_logo.svg)

# FreqAI

## 简介

FreqAI 是一款旨在自动化与训练预测性机器学习模型相关的各种任务的软件，该模型可根据一组输入信号生成市场预测。总体而言，FreqAI 旨在成为一个沙盒，用于在实时数据上轻松部署强大的机器学习库（[详情](#freqai在开源机器学习领域的定位)）。

!!! Note
    FreqAI 现在是、将来也永远是一个非盈利的开源项目。FreqAI **没有**加密代币，**不**出售信号，并且除了当前的 [freqtrade 文档](https://www.freqtrade.io/en/latest/freqai/) 之外没有其他域名。

功能包括：

* **自适应再训练** - 在[实盘部署](freqai-running.md#live-deployments)期间重新训练模型，以监督方式自适应市场变化
* **快速特征工程** - 基于用户创建的简单策略创建大型丰富的[特征集](freqai-feature-engineering.md#feature-engineering)（10k+ 特征）
* **高性能** - 多线程允许在单独的线程（或在 GPU 可用时使用 GPU）上进行自适应模型再训练，与模型推理（预测）和机器人交易操作分开。最新模型和数据保存在 RAM 中以实现快速推理
* **真实回测** - 使用[回测模块](freqai-running.md#backtesting)在历史数据上模拟自适应训练，该模块可自动执行再训练
* **可扩展性** - 通用且强大的架构允许整合 Python 中可用的任何[机器学习库/方法](freqai-configuration.md#using-different-prediction-models)。目前提供八个示例，包括分类器、回归器和卷积神经网络
* **智能异常值移除** - 使用各种[异常值检测技术](freqai-feature-engineering.md#outlier-detection)从训练和预测数据集中移除异常值
* **崩溃恢复能力** - 将训练好的模型存储到磁盘，以便从崩溃中快速轻松地重新加载，并[清除过时文件](freqai-running.md#purging-old-model-data)以支持持续的模拟/实盘运行
* **自动数据归一化** - 以智能且统计安全的方式[归一化数据](freqai-feature-engineering.md#building-the-data-pipeline)
* **自动数据下载** - 计算数据下载的时间范围并更新历史数据（在实盘部署中）
* **输入数据清洗** - 在训练和模型推理前安全处理 NaN 值
* **降维** - 通过[主成分分析](freqai-feature-engineering.md#data-dimensionality-reduction-with-principal-component-analysis)减小训练数据的大小
* **部署机器人集群** - 设置一个机器人训练模型，同时让一组[消费者](producer-consumer.md)使用信号。

## 快速开始

快速测试 FreqAI 的最简单方法是使用以下命令在模拟模式下运行：