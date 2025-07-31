上述代码片段创建名为`ft_userdata`的新目录，下载最新的compose文件并拉取freqtrade镜像。
片段中的最后两步创建带有`user_data`的目录，并（交互式地）基于您的选择创建默认配置。

!!! Question "如何编辑机器人配置？"
    您可以随时编辑配置，使用上述配置时，配置文件位于`ft_userdata`目录下的`user_data/config.json`。

    您也可以通过编辑`docker-compose.yml`文件的命令部分来更改策略和命令。

#### 添加自定义策略

1. 配置文件现在位于`user_data/config.json`
2. 将自定义策略复制到`user_data/strategies/`目录
3. 将策略类名添加到`docker-compose.yml`文件中

默认运行`SampleStrategy`。

!!! Danger "`SampleStrategy`只是一个演示！"
    `SampleStrategy`仅作为参考，为您提供自定义策略的思路。
    请务必对您的策略进行回测，并在投入真实资金前使用干运行模式测试一段时间！
    您可以在[策略自定义文档](strategy-customization.md)中找到更多关于策略开发的信息。

完成上述步骤后，您就可以启动机器人进入交易模式（干运行或实盘交易，取决于您在上述问题中的回答）。