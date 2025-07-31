#!/usr/bin/env bash
#encoding=utf8

function echo_block() {
    echo "----------------------------"
    echo $1
    echo "----------------------------"
}

function check_installed_pip() {
   ${PYTHON} -m pip > /dev/null
   if [ $? -ne 0 ]; then
        echo_block "为 ${PYTHON} 安装 Pip"
        curl https://bootstrap.pypa.io/get-pip.py -s -o get-pip.py
        ${PYTHON} get-pip.py
        rm get-pip.py
   fi
}

# 检查已安装的 python 版本
function check_installed_python() {
    if [ -n "${VIRTUAL_ENV}" ]; then
        echo "运行 setup.sh 前请先退出虚拟环境。"
        echo "你可以通过运行 'deactivate' 来退出。"
        exit 2
    fi

    for v in 13 12 11 10
    do
        PYTHON="python3.${v}"
        which $PYTHON
        if [ $? -eq 0 ]; then
            echo "使用 ${PYTHON}"
            check_installed_pip
            return
        fi
    done

    echo "未找到可用的 python。请确保已安装 python3.10 或更高版本。"
    exit 1
}

function updateenv() {
    echo_block "更新你的虚拟环境"
    if [ ! -f .venv/bin/activate ]; then
        echo "出现错误，未找到虚拟环境。"
        exit 1
    fi
    source .venv/bin/activate
    SYS_ARCH=$(uname -m)
    echo "正在安装 pip，请稍候..."
    ${PYTHON} -m pip install --upgrade pip wheel setuptools
    REQUIREMENTS_HYPEROPT=""
    REQUIREMENTS_PLOT=""
    REQUIREMENTS_FREQAI=""
    REQUIREMENTS_FREQAI_RL=""
    REQUIREMENTS=requirements.txt

    read -p "是否要安装开发所需的依赖（将完整安装所有依赖）[y/N]? "
    dev=$REPLY
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        REQUIREMENTS=requirements-dev.txt
    else
        # requirements-dev.txt 已包含以下所有依赖，因此无需进一步询问
        read -p "是否要安装绘图依赖（plotly）[y/N]? "
        if [[ $REPLY =~ ^[Yy]$ ]]
        then
            REQUIREMENTS_PLOT="-r requirements-plot.txt"
        fi
        if [ "${SYS_ARCH}" == "armv7l" ] || [ "${SYS_ARCH}" == "armv6l" ]; then
            echo "检测到树莓派，正在安装 cython，跳过 hyperopt 安装。"
            ${PYTHON} -m pip install --upgrade cython
        else
            # 非树莓派设备
            read -p "是否要安装 hyperopt 依赖 [y/N]? "
            if [[ $REPLY =~ ^[Yy]$ ]]
            then
                REQUIREMENTS_HYPEROPT="-r requirements-hyperopt.txt"
            fi
        fi

        read -p "是否要安装 freqai 依赖 [y/N]? "
        if [[ $REPLY =~ ^[Yy]$ ]]
        then
            REQUIREMENTS_FREQAI="-r requirements-freqai.txt --use-pep517"
            read -p "是否还需要 freqai-rl 或 PyTorch 的依赖（需额外约700MB存储空间）[y/N]? "
            if [[ $REPLY =~ ^[Yy]$ ]]
            then
                REQUIREMENTS_FREQAI="-r requirements-freqai-rl.txt"
            fi
        fi
    fi
    install_talib

    ${PYTHON} -m pip install --upgrade -r ${REQUIREMENTS} ${REQUIREMENTS_HYPEROPT} ${REQUIREMENTS_PLOT} ${REQUIREMENTS_FREQAI} ${REQUIREMENTS_FREQAI_RL}
    if [ $? -ne 0 ]; then
        echo "依赖安装失败"
        exit 1
    fi
    ${PYTHON} -m pip install -e .
    if [ $? -ne 0 ]; then
        echo "Freqtrade 安装失败"
        exit 1
    fi

    echo "安装 freqUI"
    freqtrade install-ui

    echo "pip 安装完成"
    echo
    if [[ $dev =~ ^[Yy]$ ]]; then
        ${PYTHON} -m pre_commit install
        if [ $? -ne 0 ]; then
            echo "pre-commit 安装失败"
            exit 1
        fi
    fi
}

# 安装 ta-lib
function install_talib() {
    if [ -f /usr/local/lib/libta_lib.a ] || [ -f /usr/local/lib/libta_lib.so ] || [ -f /usr/lib/libta_lib.so ]; then
        echo "ta-lib 已安装，跳过"
        return
    fi

    cd build_helpers && ./install_ta-lib.sh

    if [ $? -ne 0 ]; then
        echo "退出。请先修复上述错误再继续。"
        cd ..
        exit 1
    fi;

    cd ..
}


# 在 MacOS 上安装机器人
function install_macos() {
    if [ ! -x "$(command -v brew)" ]
    then
        echo_block "安装 Brew"
        /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    fi

    brew install gettext libomp

    # 获取 python 版本的小数部分
    version=$(egrep -o 3.\[0-9\]+ <<< $PYTHON | sed 's/3.//g')
}

# 在 Debian/Ubuntu 上安装机器人
function install_debian() {
    sudo apt-get update
    sudo apt-get install -y gcc build-essential autoconf libtool pkg-config make wget git curl $(echo lib${PYTHON}-dev ${PYTHON}-venv)
}

# 在 RedHat/CentOS 上安装机器人
function install_redhat() {
    sudo yum update
    sudo yum install -y gcc gcc-c++ make autoconf libtool pkg-config wget git $(echo ${PYTHON}-devel | sed 's/\.//g')
}

# 升级机器人
function update() {
    git pull
    if [ -f .env/bin/activate  ]; then
        # 发现旧环境 - 更新到新环境
        recreate_environments
    fi
    updateenv
    echo "更新完成。"
    echo_block "别忘了用 'source .venv/bin/activate' 激活你的虚拟环境！"

}

function check_git_changes() {
    if [ -z "$(git status --porcelain)" ]; then
        echo "git 目录中无更改"
        return 1
    else
        echo "git 目录中有更改"
        return 0
    fi
}

function recreate_environments() {
    if [ -d ".env" ]; then
        # 删除旧的虚拟环境
        echo "- 删除你之前的虚拟环境"
        echo "警告：你的新环境将位于 .venv！"
        rm -rf .env
    fi
    if [ -d ".venv" ]; then
        echo "- 删除你之前的虚拟环境"
        rm -rf .venv
    fi

    echo
    ${PYTHON} -m venv .venv
    if [ $? -ne 0 ]; then
        echo "无法创建虚拟环境。现在退出"
        exit 1
    fi

}

# 重置 Develop 或 Stable 分支
function reset() {
    echo_block "重置分支和虚拟环境"

    if [ "1" == $(git branch -vv |grep -cE "\* develop|\* stable") ]
    then
        if check_git_changes; then
            read -p "保留你的本地更改吗？（否则将删除你所做的所有更改！）[Y/n]? "
            if [[ $REPLY =~ ^[Nn]$ ]]; then

                git fetch -a

                if [ "1" == $(git branch -vv | grep -c "* develop") ]
                then
                    echo "- 硬重置 'develop' 分支。"
                    git reset --hard origin/develop
                elif [ "1" == $(git branch -vv | grep -c "* stable") ]
                then
                    echo "- 硬重置 'stable' 分支。"
                    git reset --hard origin/stable
                fi
            fi
        fi
    else
        echo "重置被忽略，因为你不在 'stable' 或 'develop' 分支上。"
    fi
    recreate_environments

    updateenv
}

function config() {
    echo_block "请使用 'freqtrade new-config -c user_data/config.json' 生成新的配置文件。"
}

function install() {

    echo_block "安装必要的依赖"

    if [ "$(uname -s)" == "Darwin" ]; then
        echo "检测到 macOS。正在为此系统进行设置"
        install_macos
    elif [ -x "$(command -v apt-get)" ]; then
        echo "检测到 Debian/Ubuntu。正在为此系统进行设置"
        install_debian
    elif [ -x "$(command -v yum)" ]; then
        echo "检测到 Red Hat/CentOS。正在为此系统进行设置"
        install_redhat
    else
        echo "此脚本不支持你的操作系统。"
        echo "如果你已安装 Python 3.10 - 3.13、pip、virtualenv、ta-lib，则可以继续。"
        echo "等待 10 秒以继续下一步安装，或使用 ctrl+c 中断此 shell。"
        sleep 10
    fi
    echo
    reset
    config
    echo_block "运行机器人！"
    echo "现在你可以通过执行 'source .venv/bin/activate; freqtrade <子命令>' 来使用机器人。"
    echo "你可以通过执行 'source .venv/bin/activate; freqtrade --help' 查看可用的机器人子命令列表。"
    echo "你可以通过运行 'source .venv/bin/activate; freqtrade --version' 验证 freqtrade 是否安装成功。"
}

function plot() {
    echo_block "安装绘图脚本的依赖"
    ${PYTHON} -m pip install plotly --upgrade
}

function help() {
    echo "用法："
    echo "	-i,--install    从头开始安装 freqtrade"
    echo "	-u,--update     执行 git pull 进行更新"
    echo "	-r,--reset      硬重置你的 develop/stable 分支"
    echo "	-c,--config     简易配置生成器（将覆盖你现有的文件）"
    echo "	-p,--plot       安装绘图脚本的依赖"
}

# 验证是否安装了 3.10+ 版本
check_installed_python

case $* in
--install|-i)
install
;;
--config|-c)
config
;;
--update|-u)
update
;;
--reset|-r)
reset
;;
--plot|-p)
plot
;;
*)
help
;;
esac
exit 0