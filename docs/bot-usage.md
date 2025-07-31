# 启动机器人

本页解释机器人的不同参数以及如何运行它。

!!! Note
    如果您使用了 `setup.sh`，在运行 freqtrade 命令之前不要忘记激活虚拟环境（`source .venv/bin/activate`）。

!!! Warning "时钟同步"
    运行机器人的系统时钟必须准确，并足够频繁地同步到 NTP 服务器，以避免与交易所通信时出现问题。

## 机器人命令

--8<-- "commands/main.md"

### 机器人交易命令

--8<-- "commands/trade.md"

### 如何指定要使用的配置文件？

机器人允许您通过 `-c/--config` 命令行选项选择要使用的配置文件：