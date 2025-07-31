# FreqUI

Freqtrade提供了一个内置Web服务器，可用于运行[FreqUI](https://github.com/freqtrade/frequi)——Freqtrade的前端界面。

默认情况下，UI会作为安装过程的一部分自动安装（脚本、Docker方式）。也可以使用 `freqtrade install-ui` 命令手动安装freqUI。此命令也可用于将freqUI更新到新版本。

一旦机器人以交易/回测模式启动（使用 `freqtrade trade` 命令），UI将在配置的API端口下可用（默认地址为 `http://127.0.0.1:8080`）。

??? 注意 "想要为freqUI贡献代码？"
    开发者不应使用此方法，而应克隆相应的仓库，并使用 [freqUI 仓库](https://github.com/freqtrade/frequi) 中描述的方法获取freqUI的源代码。构建前端需要安装有效的node环境。

!!! 提示 "运行freqtrade不需要freqUI"
    freqUI是freqtrade的可选组件，并非运行机器人所必需。它是一个可用于监控机器人和与之交互的前端界面——但即使没有它，freqtrade本身也能完美运行。

## 配置

FreqUI没有自己的配置文件，但其运行依赖于已正确设置的 [rest-api](rest-api.md)。请参考相应的文档页面来完成freqUI的设置。

## UI

FreqUI是一个现代化、响应式的Web应用，可用于监控机器人并与之交互。

FreqUI提供浅色和深色两种主题。可以通过页面顶部的显眼按钮轻松切换主题。本页面截图的主题会根据所选的文档主题进行适配，因此若要查看深色（或浅色）版本，请切换文档的主题。

### 登录

下面的截图显示了freqUI的登录界面。

![FreqUI - login](assets/frequi-login-CORS.png#only-dark)
![FreqUI - login](assets/frequi-login-CORS-light.png#only-light)

!!! 提示 "CORS"
    此截图中显示的CORS错误是由于UI与API运行在不同端口，且 [CORS](#cors) 尚未正确设置所致。

### 交易视图

交易视图允许您可视化机器人正在进行的交易并与机器人交互。在此页面上，您还可以通过启动和停止机器人来与之交互，并且（如果已配置）可以强制交易的进入和退出。

![FreqUI - trade view](assets/freqUI-trade-pane-dark.png#only-dark)
![FreqUI - trade view](assets/freqUI-trade-pane-light.png#only-light)

### 图表配置器

FreqUI图表可以通过策略中的 `plot_config` 配置对象（可通过“来自策略”按钮加载）或通过UI进行配置。可以创建多个图表配置并随意切换，从而灵活地以不同视图查看图表。

可以通过交易视图右上角的“图表配置器”（齿轮图标）按钮访问图表配置。

![FreqUI - plot configuration](assets/freqUI-plot-configurator-dark.png#only-dark)
![FreqUI - plot configuration](assets/freqUI-plot-configurator-light.png#only-light)

### 设置

通过访问设置页面，可以更改多个与UI相关的设置。

您可以更改的设置包括（但不限于）：

* UI的时区
* 在网站图标（浏览器标签）中显示未平仓交易
* K线颜色（上涨/下跌 -> 红色/绿色）
* 启用/禁用应用内通知类型

![FreqUI - Settings view](assets/frequi-settings-dark.png#only-dark)
![FreqUI - Settings view](assets/frequi-settings-light.png#only-light)

## 回测

当freqtrade以 [Web服务器模式](utils.md#webserver-mode) 启动（使用 `freqtrade webserver` 命令启动）时，回测视图将可用。此视图允许您回测策略并可视化结果。

您还可以加载和可视化以前的回测结果，并将结果相互比较。

![FreqUI - Backtesting](assets/freqUI-backtesting-dark.png#only-dark)
![FreqUI - Backtesting](assets/freqUI-backtesting-light.png#only-light)


--8<-- "includes/cors.md"