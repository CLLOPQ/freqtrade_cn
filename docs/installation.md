# 安装

本页面介绍如何准备运行机器人的环境。

Freqtrade文档描述了多种安装方式：

* [Docker镜像](docker_quickstart.md)（单独页面）
* [脚本安装](#脚本安装)
* [手动安装](#手动安装)
* [使用Conda安装](#使用conda安装)

评估Freqtrade功能时，建议使用预构建的[Docker镜像](docker_quickstart.md)以快速开始。

------

## 重要信息

Windows系统安装请参考[Windows安装指南](windows_installation.md)。

安装和运行Freqtrade最简单的方法是克隆机器人的Github仓库，然后运行`./setup.sh`脚本（如果你的平台支持）。

!!! 注意 "版本说明"
    克隆仓库时，默认工作分支名称为`develop`。该分支包含所有最新功能（由于自动化测试，可以认为相对稳定）。
    `stable`分支包含最新发布版本的代码（通常每月发布一次，基于`develop`分支约一周前的快照，以防止打包错误，因此可能更稳定）。

!!! 注意
    假设已安装Python3.10或更高版本以及对应的`pip`。如果未满足，安装脚本会发出警告并停止。还需要`git`来克隆Freqtrade仓库。  
    此外，必须安装Python头文件（`python<你的版本>-dev` / `python<你的版本>-devel`）才能成功完成安装。

!!! 警告 "时钟同步"
    运行机器人的系统时钟必须准确，需通过NTP服务器频繁同步，以避免与交易所通信时出现问题。

------

## 系统要求

这些要求同时适用于[脚本安装](#脚本安装)和[手动安装](#手动安装)。

!!! 注意 "ARM64系统"
    如果你运行的是ARM64系统（如MacOS M1或Oracle VM），请使用[Docker](docker_quickstart.md)运行Freqtrade。
    虽然通过手动操作可以进行本地安装，但目前不提供支持。

### 安装指南

* [Python >= 3.10](http://docs.python-guide.org/en/latest/starting/installation/)
* [pip](https://pip.pypa.io/en/stable/installing/)
* [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
* [virtualenv](https://virtualenv.pypa.io/en/stable/installation.html)（推荐）
* [TA-Lib](https://ta-lib.github.io/ta-lib-python/)（安装说明[下文](#安装-ta-lib)）

### 安装代码

我们提供了Ubuntu、MacOS和Windows系统的安装说明。这些是指导方针，在其他发行版上的成功情况可能有所不同。
首先列出特定系统的步骤，下方的通用部分对所有系统都是必需的。

!!! 注意
    假设已安装Python3.10或更高版本以及对应的pip。

=== "Debian/Ubuntu系统"
    #### 安装必要的依赖项