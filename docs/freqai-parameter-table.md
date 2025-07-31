# 参数表

下表列出了FreqAI可用的所有配置参数。部分参数在`config_examples/config_freqai.example.json`中有示例。

必填参数标记为**必填项**，必须通过建议的方式之一进行设置。

### 常规配置参数

| 参数 | 描述 |
|------------|-------------|
|  |  **`config.freqai` 树中的常规配置参数**
| `freqai` | **必填项。** <br> 包含所有用于控制FreqAI的参数的父字典。 <br> **数据类型：** 字典。
| `train_period_days` | **必填项。** <br> 用于训练数据的天数（滑动窗口的宽度）。 <br> **数据类型：** 正整数。
| `backtest_period_days` | **必填项。** <br> 在回测期间，从训练好的模型进行推理的天数，之后滑动上述定义的`train_period_days`窗口并重新训练模型（更多信息见[这里](freqai-running.md#backtesting)）。可以是分数天，但请注意，提供的`timerange`将除以该数值，以得出完成回测所需的训练次数。 <br> **数据类型：** 浮点数。
| `identifier` | **必填项。** <br> 当前模型的唯一ID。如果模型保存到磁盘，`identifier`允许重新加载特定的预训练模型/数据。 <br> **数据类型：** 字符串。
| `live_retrain_hours` | 模拟/实盘运行期间的重训练频率。 <br> **数据类型：** 大于0的浮点数。 <br> 默认值：`0`（模型尽可能频繁地重训练）。
| `expiration_hours` | 如果模型超过`expiration_hours`小时，则避免进行预测。 <br> **数据类型：** 正整数。 <br> 默认值：`0`（模型永不过期）。
| `purge_old_models` | 保留在磁盘上的模型数量（与回测无关）。默认值为2，意味着模拟/实盘运行将在磁盘上保留最新的2个模型。设置为0则保留所有模型。此参数也接受布尔值以保持向后兼容性。 <br> **数据类型：** 整数。 <br> 默认值：`2`。
| `save_backtest_models` | 运行回测时将模型保存到磁盘。回测通过保存预测数据并直接在后续运行中重用（当您希望调整入场/出场参数时）来实现最高效率。将回测模型保存到磁盘还允许使用相同的模型文件启动具有相同模型`identifier`的模拟/实盘实例。 <br> **数据类型：** 布尔值。 <br> 默认值：`False`（不保存模型）。
| `fit_live_predictions_candles` | 用于从预测数据计算目标（标签）统计信息的历史K线数量，而非从训练数据集（更多信息见[这里](freqai-configuration.md#creating-a-dynamic-target-threshold)）。 <br> **数据类型：** 正整数。
| `continual_learning` | 使用最近训练模型的最终状态作为新模型的起点，允许增量学习（更多信息见[这里](freqai-running.md#continual-learning)）。请注意，这目前是一种朴素的增量学习方法，当市场偏离您的模型时，很可能会过拟合/陷入局部最小值。我们在此提供连接主要是为了实验目的，并为在加密市场等混沌系统中更成熟的持续学习方法做好准备。 <br> **数据类型：** 布尔值。 <br> 默认值：`False`。
| `write_metrics_to_disk` | 收集训练时间、推理时间和CPU使用率到json文件中。 <br> **数据类型：** 布尔值。 <br> 默认值：`False`
| `data_kitchen_thread_count` | <br> 指定用于数据处理（异常值方法、归一化等）的线程数。这对训练使用的线程数没有影响。如果用户未设置（默认），FreqAI将使用最大线程数减2（为Freqtrade机器人和FreqUI留下1个物理核心） <br> **数据类型：** 正整数。
| `activate_tensorboard` | <br> 指示是否为支持TensorBoard的模块（当前为强化学习、XGBoost、Catboost和PyTorch）激活TensorBoard。TensorBoard需要安装Torch，这意味着您需要torch/RL docker镜像，或者在安装时回答“是”以安装Torch。 <br> **数据类型：** 布尔值。 <br> 默认值：`True`。
| `wait_for_training_iteration_on_reload` | <br> 使用/reload或ctrl-c时，等待当前训练迭代完成后再优雅关闭。如果设置为`False`，FreqAI将中断当前训练迭代，允许更快地优雅关闭，但会丢失当前训练迭代。 <br> **数据类型：** 布尔值。 <br> 默认值：`True`。

### 特征参数

| 参数 | 描述 |
|------------|-------------|
|  |  **`freqai.feature_parameters` 子字典中的特征参数**
| `feature_parameters` | 包含用于构建特征集的参数的字典。详细信息和示例见[这里](freqai-feature-engineering.md)。 <br> **数据类型：** 字典。
| `include_timeframes` | `feature_engineering_expand_*()` 中所有指标将为之创建的时间框架列表。该列表将作为特征添加到基础指标数据集中。 <br> **数据类型：** 时间框架列表（字符串）。
| `include_corr_pairlist` | 相关币种列表，FreqAI将为所有`pair_whitelist`币种添加这些币种作为额外特征。特征工程期间（详见[这里](freqai-feature-engineering.md)）在`feature_engineering_expand_*()`中设置的所有指标将为每个相关币种创建。相关币种的特征将添加到基础指标数据集中。 <br> **数据类型：** 资产列表（字符串）。
| `label_period_candles` | 为其创建标签的未来K线数量。可在`set_freqai_targets()`中使用（详见`templates/FreqaiExampleStrategy.py`中的用法）。此参数不一定是必填项，您可以创建自定义标签并选择是否使用此参数。请参见`templates/FreqaiExampleStrategy.py`查看示例用法。 <br> **数据类型：** 正整数。
| `include_shifted_candles` | 将前几根K线的特征添加到后续K线中，目的是添加历史信息。如果使用，FreqAI将复制并移动`include_shifted_candles`根前序K线的所有特征，以便后续K线可以使用这些信息。 <br> **数据类型：** 正整数。
| `weight_factor` | 根据数据点的新近度对训练数据点进行加权（详见[这里](freqai-feature-engineering.md#weighting-features-for-temporal-importance)）。 <br> **数据类型：** 正浮点数（通常<1）。
| `indicator_max_period_candles` | **不再使用（#7325）**。由策略中设置的`startup_candle_count`替换（见[freqai-configuration.md#building-a-freqai-strategy](freqai-configuration.md#building-a-freqai-strategy)）。`startup_candle_count`与时间框架无关，定义了`feature_engineering_*()`中用于指标创建的最大周期。FreqAI使用此参数以及`include_time_frames`中的最大时间框架来计算需要下载的数据点数量，以确保第一个数据点不包含NaN。 <br> **数据类型：** 正整数。
| `indicator_periods_candles` | 用于计算指标的时间周期。指标将添加到基础指标数据集中。 <br> **数据类型：** 正整数列表。
| `principal_component_analysis` | 使用主成分分析自动降低数据集的维度。详见其工作原理[这里](freqai-feature-engineering.md#data-dimensionality-reduction-with-principal-component-analysis) <br> **数据类型：** 布尔值。 <br> 默认值：`False`。
| `plot_feature_importances` | 为每个模型创建特征重要性图，显示`plot_feature_importances`数量的顶部/底部特征。图表存储在`user_data/models/<identifier>/sub-train-<COIN>_<timestamp>.html`。 <br> **数据类型：** 整数。 <br> 默认值：`0`。
| `DI_threshold` | 当设置为>0时，激活使用差异指数（Dissimilarity Index）进行异常值检测。详见其工作原理[这里](freqai-feature-engineering.md#identifying-outliers-with-the-dissimilarity-index-di)。 <br> **数据类型：** 正浮点数（通常<1）。
| `use_SVM_to_remove_outliers` | 训练支持向量机以检测并从训练数据集以及传入数据点中移除异常值。详见其工作原理[这里](freqai-feature-engineering.md#identifying-outliers-using-a-support-vector-machine-svm)。 <br> **数据类型：** 布尔值。
| `svm_params` | Sklearn的`SGDOneClassSVM()`中可用的所有参数。详见部分选定参数[这里](freqai-feature-engineering.md#identifying-outliers-using-a-support-vector-machine-svm)。 <br> **数据类型：** 字典。
| `use_DBSCAN_to_remove_outliers` | 使用DBSCAN算法对数据进行聚类，以识别并从训练和预测数据中移除异常值。详见其工作原理[这里](freqai-feature-engineering.md#identifying-outliers-with-dbscan)。 <br> **数据类型：** 布尔值。 
| `noise_standard_deviation` | 如果设置，FreqAI会向训练特征添加噪声，目的是防止过拟合。FreqAI从具有`noise_standard_deviation`标准差的高斯分布生成随机偏差，并将其添加到所有数据点。`noise_standard_deviation`应相对于归一化空间设置，即介于-1和1之间。换句话说，由于FreqAI中的数据始终归一化到-1和1之间，`noise_standard_deviation: 0.05`将导致32%的数据随机增加/减少超过2.5%（即落在第一个标准差内的数据百分比）。 <br> **数据类型：** 整数。 <br> 默认值：`0`。
| `outlier_protection_percentage` | 启用以防止异常值检测方法丢弃过多数据。如果SVM或DBSCAN检测到超过`outlier_protection_percentage`%的点为异常值，FreqAI将记录警告消息并忽略异常值检测，即保留原始数据集。如果触发异常值保护，将不会基于该训练数据集进行预测。 <br> **数据类型：** 浮点数。 <br> 默认值：`30`。
| `reverse_train_test_order` | 分割特征数据集（见下文），并使用最新的数据分割进行训练，在历史数据分割上进行测试。这允许模型训练到最新的数据点，同时避免过拟合。但是，在使用此参数之前，您应该了解其非传统性质。 <br> **数据类型：** 布尔值。 <br> 默认值：`False`（不反转）。
| `shuffle_after_split` | 将数据分割为训练集和测试集，然后分别打乱两个集合。 <br> **数据类型：** 布尔值。 <br> 默认值：`False`。
| `buffer_train_data_candles` | 在指标填充后，从训练数据的开头和结尾切除`buffer_train_data_candles`根K线。主要示例用途是在预测最大值和最小值时，argrelextrema函数无法知道时间范围边缘的最大值/最小值。为提高模型准确性，最好在完整时间范围内计算argrelextrema，然后使用此函数通过内核切除边缘（缓冲区）。在另一种情况下，如果目标设置为偏移价格变动，则此缓冲区是不必要的，因为时间范围末尾的偏移K线将为NaN，FreqAI会自动从训练数据集中切除这些K线。<br> **数据类型：** 整数。 <br> 默认值：`0`。

### 数据分割参数

| 参数 | 描述 |
|------------|-------------|
|  |  **`freqai.data_split_parameters` 子字典中的数据分割参数**
| `data_split_parameters` | 包含scikit-learn `test_train_split()`的任何其他可用参数，详见[这里](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)（外部网站）。 <br> **数据类型：** 字典。
| `test_size` | 应用于测试而非训练的数据比例。 <br> **数据类型：** 正浮点数<1。
| `shuffle` | 训练期间打乱训练数据点。通常，为了不破坏时间序列预测中的数据时间顺序，此参数设置为`False`。 <br> **数据类型：** 布尔值。 <br> 默认值：`False`。

### 模型训练参数

| 参数 | 描述 |
|------------|-------------|
|  |  **`freqai.model_training_parameters` 子字典中的模型训练参数**
| `model_training_parameters` | 包含所选模型库可用的所有参数的灵活字典。例如，如果使用`LightGBMRegressor`，此字典可以包含`LightGBMRegressor`的任何参数（见[这里](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html)（外部网站））。如果选择不同的模型，此字典可以包含该模型的任何参数。当前可用模型列表见[这里](freqai-configuration.md#using-different-prediction-models)。  <br> **数据类型：** 字典。
| `n_estimators` | 模型训练中拟合的提升树数量。 <br> **数据类型：** 整数。
| `learning_rate` | 模型训练期间的提升学习率。 <br> **数据类型：** 浮点数。
| `n_jobs`, `thread_count`, `task_type` | 设置并行处理的线程数和`task_type`（`gpu`或`cpu`）。不同的模型库使用不同的参数名称。 <br> **数据类型：** 浮点数。

### 强化学习参数

| 参数 | 描述 |
|------------|-------------|
|  |  **`freqai.rl_config` 子字典中的强化学习参数**
| `rl_config` | 包含强化学习模型控制参数的字典。 <br> **数据类型：** 字典。
| `train_cycles` | 训练时间步长将基于`train_cycles * 训练数据点数量`设置。 <br> **数据类型：** 整数。
| `max_trade_duration_candles`| 指导智能体训练以将交易保持在期望长度以下。示例用法见`prediction_models/ReinforcementLearner.py`中的可自定义`calculate_reward()`函数。 <br> **数据类型：** int。
| `model_type` | 来自stable_baselines3或SBcontrib的模型字符串。可用字符串包括：`'TRPO', 'ARS', 'RecurrentPPO', 'MaskablePPO', 'PPO', 'A2C', 'DQN'`。用户应确保`model_training_parameters`与对应的stable_baselines3模型的可用参数匹配，可查阅其文档。[PPO文档](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)（外部网站） <br> **数据类型：** string。
| `policy_type` | 来自stable_baselines3的可用策略类型之一 <br> **数据类型：** string。
| `max_training_drawdown_pct` | 智能体在训练期间允许经历的最大回撤。 <br> **数据类型：** float。 <br> 默认值：0.8
| `cpu_count` | 专用于强化学习训练过程的线程/CPU数量（取决于是否选择`ReinforcementLearning_multiproc`）。建议保持默认值，默认情况下，此值设置为物理核心总数减1。 <br> **数据类型：** int。 
| `model_reward_parameters` | `ReinforcementLearner.py`中可自定义的`calculate_reward()`函数内使用的参数 <br> **数据类型：** int。
| `add_state_info` | 告诉FreqAI在训练和推理的特征集中包含状态信息。当前状态变量包括交易持续时间、当前利润、交易仓位。仅在模拟/实盘运行中可用，回测时自动切换为false。 <br> **数据类型：** bool。 <br> 默认值：`False`。
| `net_arch` | 网络架构，在[`stable_baselines3`文档](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#examples)中有详细描述。概括地说：`[<共享层>, dict(vf=[<非共享值网络层>], pi=[<非共享策略网络层>])]`。默认设置为`[128, 128]`，定义了2个共享隐藏层，每个层有128个单元。
| `randomize_starting_position` | 随机化每个回合的起始点以避免过拟合。 <br> **数据类型：** bool。 <br> 默认值：`False`。
| `drop_ohlc_from_features` | 训练期间不将归一化的ohlc数据包含在传递给智能体的特征集中（在所有情况下，ohlc仍将用于驱动环境） <br> **数据类型：** Boolean。 <br> **默认值：** `False`
| `progress_bar` | 显示进度条，包含当前进度、已用时间和估计剩余时间。 <br> **数据类型：** Boolean。 <br> 默认值：`False`。

### PyTorch参数

#### general

| 参数 | 描述 |
|------------|-------------|
|  |  **`freqai.model_training_parameters` 子字典中的模型训练参数**
| `learning_rate` | 传递给优化器的学习率。 <br> **数据类型：** float。 <br> 默认值：`3e-4`。
| `model_kwargs` | 传递给模型类的参数。 <br> **数据类型：** dict。 <br> 默认值：`{}`。
| `trainer_kwargs` | 传递给训练器类的参数。 <br> **数据类型：** dict。 <br> 默认值：`{}`。

#### trainer_kwargs

| 参数 | 描述 |
|--------------|-------------|
|              |  **`freqai.model_training_parameters.model_kwargs` 子字典中的模型训练参数**
| `n_epochs`   | `n_epochs`参数是PyTorch训练循环中的关键设置，决定了整个训练数据集用于更新模型参数的次数。一个epoch表示完整遍历整个训练数据集一次。覆盖`n_steps`。必须设置`n_epochs`或`n_steps`之一。 <br><br> **数据类型：** int. 可选。 <br> 默认值：`10`。
| `n_steps`    | 设置`n_epochs`的另一种方式——运行的训练迭代次数。此处的迭代指调用`optimizer.step()`的次数。如果设置了`n_epochs`，则忽略此参数。简化函数如下： <br><br> n_epochs = n_steps / (n_obs / batch_size) <br><br> 此处的目的是`n_steps`更容易优化，并且在不同n_obs（数据点数量）之间保持稳定。  <br> <br> **数据类型：** int. 可选。 <br> 默认值：`None`。
| `batch_size` | 训练期间使用的批次大小。 <br><br> **数据类型：** int. <br> 默认值：`64`。


### 其他参数

| 参数 | 描述 |
|------------|-------------|
|  |  **额外参数**
| `freqai.keras` | 如果所选模型使用Keras（通常用于基于TensorFlow的预测模型），需要激活此标志，以便模型的保存/加载遵循Keras标准。 <br> **数据类型：** Boolean。 <br> 默认值：`False`。
| `freqai.conv_width` | 神经网络输入张量的宽度。通过将历史数据点作为张量的第二维度输入，取代了移动K线（`include_shifted_candles`）的需求。从技术上讲，此参数也可用于回归器，但只会增加计算开销，不会改变模型训练/预测。 <br> **数据类型：** Integer。 <br> 默认值：`2`。
| `freqai.reduce_df_footprint` | 将所有数值列重新转换为float32/int32，目的是减少内存/磁盘使用量并缩短训练/推理时间。此参数在Freqtrade配置文件的主级别设置（不在FreqAI内部）。 <br> **数据类型：** Boolean。 <br> 默认值：`False`。