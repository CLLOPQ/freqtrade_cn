# 如何更新

要更新您的Freqtrade安装，请使用以下对应于您安装方式的方法之一。

!!! Note "跟踪变更"
    重大变更/行为变更将记录在每个版本附带的更新日志中。
    对于develop分支，请关注PR以避免对变更感到意外。

## Docker

!!! Note "使用`master`镜像的旧版安装"
    我们将发布镜像从master切换到stable - 请调整您的docker-file并将`freqtradeorg/freqtrade:master`替换为`freqtradeorg/freqtrade:stable`