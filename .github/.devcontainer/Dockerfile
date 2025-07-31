# 基于Freqtrade官方开发镜像构建，包含freqairl支持
FROM freqtradeorg/freqtrade:develop_freqairl

# 切换到root用户以执行系统级操作
USER root

# 将本地的requirements-dev.txt文件复制到容器内的/freqtrade目录
COPY requirements-dev.txt /freqtrade/

# 定义用户名变量，默认为ftuser
ARG USERNAME=ftuser

# 安装系统依赖和开发工具
RUN apt-get update \
    && apt-get -y install --no-install-recommends \
        apt-utils \           # 高级包管理工具
        dialog \              # 终端交互工具
        git \                 # 版本控制工具
        ssh \                 # SSH客户端，用于远程访问
        vim \                 # 文本编辑器
        build-essential \     # 编译工具链
        zsh \                 # Zsh shell
    && apt-get clean \        # 清理APT缓存，减小镜像体积
    # 创建VS Code服务器所需目录
    && mkdir -p /home/${USERNAME}/.vscode-server /home/${USERNAME}/.vscode-server-insiders /home/${USERNAME}/commandhistory \
    # 设置目录权限，确保ftuser用户有访问权限
    && chown ${USERNAME}:${USERNAME} -R /home/${USERNAME}/.local/ \
    && chown ${USERNAME}: -R /home/${USERNAME}/

# 切换回ftuser用户，避免以root身份运行后续命令
USER ftuser

# 安装Python开发依赖
RUN pip install --user \
    autopep8 \                  # Python代码格式化工具
    -r docs/requirements-docs.txt \  # 文档生成所需依赖
    -r requirements-dev.txt \       # 开发环境所需依赖
    --no-cache-dir              # 不缓存安装包，减小镜像体积

# 清空ENTRYPOINT，允许执行任意命令（默认镜像可能有固定的启动命令）
ENTRYPOINT []