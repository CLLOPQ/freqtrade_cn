这将启动一个本地服务器（通常在端口8000上），您可以查看内容是否符合预期。

## 开发环境设置

要配置开发环境，您可以使用提供的[开发容器](#devcontainer-setup)，或使用`setup.sh`脚本并在被问及“Do you want to install dependencies for dev [y/N]? ”时回答“y”。
或者（例如，如果您的系统不支持setup.sh脚本），按照手动安装流程运行`pip3 install -r requirements-dev.txt`，然后运行`pip3 install -e .[all]`。

这将安装所有开发所需的工具，包括`pytest`、`ruff`、`mypy`和`coveralls`。

然后通过运行`pre-commit install`安装git钩子脚本，这样您的更改在提交前会在本地进行验证。
这避免了大量等待CI的时间，因为一些基本的格式检查会在您的本地机器上完成。

在创建拉取请求之前，请熟悉我们的[贡献指南](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md)。

### 开发容器设置

开始开发的最快最简单方法是使用[VSCode](https://code.visualstudio.com/)和Remote Container扩展。
这使开发者能够启动包含所有必要依赖的机器人，*无需*在本地机器上安装任何Freqtrade特定的依赖。

#### 开发容器依赖项

* [VSCode](https://code.visualstudio.com/)
* [docker](https://docs.docker.com/install/)
* [Remote Container扩展文档](https://code.visualstudio.com/docs/remote)

有关[Remote Container扩展](https://code.visualstudio.com/docs/remote)的更多信息，请查阅官方文档。

### 测试

新代码应包含基本的单元测试。根据功能的复杂性，审核者可能会要求更深入的单元测试。
如有必要，Freqtrade团队可以协助并指导编写良好的测试（但请不要期望有人为您编写测试）。

#### 如何运行测试

在根文件夹中使用`pytest`运行所有可用的测试用例，并确认您的本地环境设置正确。

!!! 注意 "功能分支"
    测试应在`develop`和`stable`分支上通过。其他分支可能是正在开发中的工作，测试可能尚未通过。

#### 在测试中检查日志内容

Freqtrade使用两种主要方法在测试中检查日志内容：`log_has()`和`log_has_re()`（使用正则表达式检查动态日志消息）。
这些方法可从`conftest.py`获取，并可导入到任何测试模块中。

示例检查如下：