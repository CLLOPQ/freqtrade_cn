# ![freqtrade](https://raw.githubusercontent.com/freqtrade/freqtrade/develop/docs/assets/freqtrade_poweredby.svg)

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

Freqtrade 是一个免费开源的加密货币交易机器人，采用 Python 编写。它支持所有主流交易所，可通过 Telegram 或 webUI 进行控制。包含回测、绘图和资金管理工具，以及通过机器学习优化策略的功能。

![freqtrade](https://raw.githubusercontent.com/freqtrade/freqtrade/develop/docs/assets/freqtrade-screenshot.png)

## 免责声明

本软件仅用于教育目的。请勿用你无法承受损失的资金进行冒险。使用本软件的风险由你自行承担。作者及所有关联方不对你的交易结果负责。

始终先在模拟交易（Dry-run）模式下运行交易机器人，在了解其工作原理和预期盈亏之前，不要投入真实资金。

我们强烈建议你具备编程和 Python 知识。请务必阅读源代码，了解此机器人的运行机制。

## 支持的交易所市场

请阅读[交易所特定说明](docs/exchanges.md)，了解每个交易所可能需要的特殊配置。

- [X] [币安（Binance）](https://www.binance.com/)
- [X] [Bitmart](https://bitmart.com/)
- [X] [BingX](https://bingx.com/invite/0EM9RX)
- [X] [Bybit](https://bybit.com/)
- [X] [Gate.io](https://www.gate.io/ref/6266643)
- [X] [HTX](https://www.htx.com/)
- [X] [Hyperliquid](https://hyperliquid.xyz/)（去中心化交易所，简称 DEX）
- [X] [Kraken](https://kraken.com/)
- [X] [OKX](https://okx.com/)
- [X] [MyOKX](https://okx.com/)（OKX 欧洲经济区版本）
- [ ] [可能还有许多其他交易所](https://github.com/ccxt/ccxt/)。_（我们不能保证它们都能正常工作）_

### 支持的期货交易所（实验性）

- [X] [币安（Binance）](https://www.binance.com/)
- [X] [Gate.io](https://www.gate.io/ref/6266643)
- [X] [Hyperliquid](https://hyperliquid.xyz/)（去中心化交易所，简称 DEX）
- [X] [OKX](https://okx.com/)
- [X] [Bybit](https://bybit.com/)

请务必在开始前阅读[交易所特定说明](docs/exchanges.md)以及[杠杆交易](docs/leverage.md)文档。

### 社区测试通过

经社区确认可正常工作的交易所：

- [X] [Bitvavo](https://bitvavo.com/)
- [X] [Kucoin](https://www.kucoin.com/)

## 文档

我们建议你阅读机器人文档，以确保了解机器人的工作原理。

完整文档请参见 [freqtrade 官网](https://www.freqtrade.io)。

## 功能

- [x] **基于 Python 3.10+**：可在任何操作系统上运行——Windows、macOS 和 Linux。
- [x] **数据持久化**：通过 sqlite 实现数据持久化。
- [x] **模拟交易（Dry-run）**：无需实际投入资金即可运行机器人。
- [x] **回测**：对你的买入/卖出策略进行模拟运行。
- [x] **通过机器学习优化策略**：利用机器学习，结合真实交易所数据优化你的买入/卖出策略参数。
- [X] **自适应预测建模**：使用 FreqAI 构建智能策略，通过自适应机器学习方法对市场进行自我训练。[了解更多](https://www.freqtrade.io/en/stable/freqai/)
- [x] **加密货币白名单**：选择你想要交易的加密货币，或使用动态白名单。
- [x] **加密货币黑名单**：选择你想要避开的加密货币。
- [x] **内置 WebUI**：内置网页界面，方便管理机器人。
- [x] **通过 Telegram 管理**：可通过 Telegram 管理机器人。
- [x] **以法定货币显示盈亏**：以法定货币显示你的盈亏情况。
- [x] **业绩状态报告**：提供当前交易的业绩状态。

## 快速开始

请参考[Docker 快速开始文档](https://www.freqtrade.io/en/stable/docker_quickstart/)了解如何快速入门。

关于其他（原生）安装方法，请参考[安装文档页面](https://www.freqtrade.io/en/stable/installation/)。

## 基本用法

### 机器人命令

```
usage: freqtrade [-h] [-V]
                 {trade,create-userdir,new-config,show-config,new-strategy,download-data,convert-data,convert-trade-data,trades-to-ohlcv,list-data,backtesting,backtesting-show,backtesting-analysis,edge,hyperopt,hyperopt-list,hyperopt-show,list-exchanges,list-markets,list-pairs,list-strategies,list-hyperoptloss,list-freqaimodels,list-timeframes,show-trades,test-pairlist,convert-db,install-ui,plot-dataframe,plot-profit,webserver,strategy-updater,lookahead-analysis,recursive-analysis}
                 ...

免费开源的加密货币交易机器人

位置参数：
  {trade,create-userdir,new-config,show-config,new-strategy,download-data,convert-data,convert-trade-data,trades-to-ohlcv,list-data,backtesting,backtesting-show,backtesting-analysis,edge,hyperopt,hyperopt-list,hyperopt-show,list-exchanges,list-markets,list-pairs,list-strategies,list-hyperoptloss,list-freqaimodels,list-timeframes,show-trades,test-pairlist,convert-db,install-ui,plot-dataframe,plot-profit,webserver,strategy-updater,lookahead-analysis,recursive-analysis}
    trade               交易模块。
    create-userdir      创建用户数据目录。
    new-config          创建新配置
    show-config         显示解析后的配置
    new-strategy        创建新策略
    download-data       下载回测数据。
    convert-data        将K线（OHLCV）数据从一种格式转换为另一种。
    convert-trade-data  将交易数据从一种格式转换为另一种。
    trades-to-ohlcv     将交易数据转换为OHLCV数据。
    list-data           列出已下载的数据。
    backtesting         回测模块。
    backtesting-show    显示过去的回测结果
    backtesting-analysis
                        回测分析模块。
    hyperopt            超参数优化模块。
    hyperopt-list       列出超参数优化结果
    hyperopt-show       显示超参数优化结果的详情
    list-exchanges      打印可用的交易所。
    list-markets        打印交易所的市场。
    list-pairs          打印交易所的交易对。
    list-strategies     打印可用的策略。
    list-hyperoptloss   打印可用的超参数优化损失函数。
    list-freqaimodels   打印可用的freqAI模型。
    list-timeframes     打印交易所可用的时间周期。
    show-trades         显示交易。
    test-pairlist       测试你的交易对列表配置。
    convert-db          将数据库迁移到不同的系统
    install-ui          安装FreqUI
    plot-dataframe      绘制带有指标的K线图。
    plot-profit         生成显示利润的图表。
    webserver           网页服务器模块。
    strategy-updater    将过时的策略文件更新到当前版本
    lookahead-analysis  检查潜在的前瞻性偏差。
    recursive-analysis  检查潜在的递归公式问题。

选项：
  -h, --help            显示帮助信息并退出
  -V, --version         显示程序的版本号并退出
```

### Telegram 远程控制命令

Telegram 并非必需，但它是控制机器人的好方法。更多详情和完整命令列表请参见[文档](https://www.freqtrade.io/en/latest/telegram-usage/)

- `/start`：启动交易程序。
- `/stop`：停止交易程序。
- `/stopentry`：停止进入新交易。
- `/status <trade_id>|[table]`：列出所有或特定的未平仓交易。
- `/profit [<n>]`：列出所有已完成交易的累计利润，过去 n 天的数据。
- `/forceexit <trade_id>|all`：立即平仓指定交易（忽略 `minimum_roi`）。
- `/fx <trade_id>|all`：`/forceexit` 的别名
- `/performance`：按交易对显示每个已完成交易的表现
- `/balance`：按货币显示账户余额。
- `/daily <n>`：显示过去 n 天每天的盈亏情况。
- `/help`：显示帮助信息。
- `/version`：显示版本信息。

## 开发分支

该项目目前主要有两个分支：

- `develop` - 这个分支经常有新功能，但也可能包含破坏性变更。我们努力保持这个分支的稳定性。
- `stable` - 这个分支包含最新的稳定版本。这个分支通常经过充分测试。
- `feat/*` - 这些是功能分支，正在进行大量开发。除非你想测试特定功能，否则请不要使用这些分支。

## 支持

### 帮助 / Discord

如果有文档未涵盖的问题，或需要更多关于机器人的信息，或者只是想与志同道合的人交流，我们建议你加入 Freqtrade 的 [discord 服务器](https://discord.gg/p7nuUNVfP7)。

### [漏洞 / 问题](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue)

如果你发现机器人有漏洞，请先[搜索问题跟踪器](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue)。如果尚未报告，请[创建一个新问题](https://github.com/freqtrade/freqtrade/issues/new/choose)，并确保遵循模板指南，以便团队能尽快为你提供帮助。

对于创建的每个[问题](https://github.com/freqtrade/freqtrade/issues/new/choose)，请跟进并在问题得到解决时标记满意度或提醒关闭问题。

--遵守 github 的[社区政策](https://docs.github.com/en/site-policy/github-terms/github-community-code-of-conduct)--

### [功能请求](https://github.com/freqtrade/freqtrade/labels/enhancement)

你有改善机器人的好主意想分享吗？请先搜索该功能是否已被[讨论过](https://github.com/freqtrade/freqtrade/labels/enhancement)。如果尚未被请求，请[创建一个新请求](https://github.com/freqtrade/freqtrade/issues/new/choose)，并确保遵循模板指南，以免在错误报告中丢失。

### [拉取请求](https://github.com/freqtrade/freqtrade/pulls)

你觉得机器人缺少某个功能吗？我们欢迎你的拉取请求！

请阅读[贡献文档](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md)，了解提交拉取请求的要求。

贡献不一定需要编码——或许可以从改进文档开始？标记为 [good first issue](https://github.com/freqtrade/freqtrade/labels/good%20first%20issue) 的问题可能是很好的初次贡献，有助于你熟悉代码库。

**注意** 在开始任何重大的新功能开发之前，*请打开一个问题描述你计划做什么*，或在 [discord](https://discord.gg/p7nuUNVfP7) 上与我们交流（请使用 #dev 频道）。这将确保相关人员能对该功能提供有价值的反馈，并让其他人知道你正在开发它。

**重要提示**：请始终针对 `develop` 分支创建拉取请求，而不是 `stable` 分支。

## 要求

### 最新时钟

时钟必须准确，频繁同步到 NTP 服务器，以避免与交易所通信出现问题。

### 最低硬件要求

要运行此机器人，我们建议你使用云服务器，最低配置为：

- 最低（建议）系统要求：2GB 内存，1GB 磁盘空间，2 核 CPU

### 软件要求

- [Python >= 3.10](http://docs.python-guide.org/en/latest/starting/installation/)
- [pip](https://pip.pypa.io/en/stable/installing/)
- [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- [TA-Lib](https://ta-lib.github.io/ta-lib-python/)
- [virtualenv](https://virtualenv.pypa.io/en/stable/installation.html)（推荐）
- [Docker](https://www.docker.com/products/docker)（推荐）