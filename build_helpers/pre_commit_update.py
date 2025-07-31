# 用于CI流程的文件，确保pre-commit依赖保持最新状态

import sys
from pathlib import Path

import yaml


pre_commit配置文件 = Path(".pre-commit-config.yaml")
开发依赖文件 = Path("requirements-dev.txt")
依赖文件 = Path("requirements.txt")

# 读取开发依赖文件内容
with 开发依赖文件.open("r") as rfile:
    依赖列表 = rfile.readlines()

# 读取普通依赖文件内容并合并
with 依赖文件.open("r") as rfile:
    依赖列表.extend(rfile.readlines())

# 提取类型相关的依赖
类型依赖 = [
    r.strip("\n") for r in 依赖列表 if r.startswith("types-") or r.startswith("SQLAlchemy")
]

# 读取pre-commit配置文件
with pre_commit配置文件.open("r") as file:
    配置内容 = yaml.load(file, Loader=yaml.SafeLoader)


# 找到mypy相关的仓库配置
mypy仓库 = [
    repo for repo in 配置内容["repos"] if repo["repo"] == "https://github.com/pre-commit/mirrors-mypy"
]

# 获取钩子的额外依赖
钩子依赖 = mypy仓库[0]["hooks"][0]["additional_dependencies"]

# 检查依赖一致性
错误列表 = []
for 钩子 in 钩子依赖:
    if 钩子 not in 类型依赖:
        错误列表.append(f"{钩子} 在requirements-dev.txt中缺失。")

for 依赖 in 类型依赖:
    if 依赖 not in 钩子依赖:
        错误列表.append(f"{依赖} 在pre-config文件中缺失。")


# 输出错误并退出
if 错误列表:
    for 错误 in 错误列表:
        print(错误)
    sys.exit(1)

sys.exit(0)