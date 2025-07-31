# 如果未提供第一个参数，则使用默认安装路径
if [ -z "$1" ]; then
  安装路径=/usr/local
else
  安装路径=${1}
fi

echo "将安装到 ${安装路径}"

# 如果提供了第二个参数，或者未找到已安装的库文件，则执行安装流程
if [ -n "$2" ] || [ ! -f "${安装路径}/lib/libta_lib.a" ]; then
  # 解压源码包
  tar zxvf ta-lib-0.4.0-src.tar.gz
  
  # 进入源码目录并执行编译前准备工作
  cd ta-lib \
  && sed -i.bak "s|0.00000001|0.000000000000000001 |g" src/ta_func/ta_utility.h \
  && echo "正在下载gcc的config.guess和config.sub文件" \
  && curl -s 'https://raw.githubusercontent.com/gcc-mirror/gcc/master/config.guess' -o config.guess \
  && curl -s 'https://raw.githubusercontent.com/gcc-mirror/gcc/master/config.sub' -o config.sub \
  && ./configure --prefix=${安装路径}/ \
  && make

  # 检查编译是否成功
  if [ $? -ne 0 ]; then
    echo "编译ta-lib失败。"
    cd .. && rm -rf ./ta-lib/
    exit 1
  fi

  # 根据是否提供第二个参数决定安装方式
  if [ -z "$2" ]; then
    # 尝试使用sudo权限安装
    which sudo && sudo make install || make install
    
    # 如果是Debian/Ubuntu系统，更新库缓存
    if [ -x "$(command -v apt-get)" ]; then
      echo "使用ldconfig更新库路径"
      sudo ldconfig
    fi
  else
    # 不使用sudo权限安装
    make install
  fi

  # 清理安装文件
  cd .. && rm -rf ./ta-lib/
else
  echo "TA-lib已安装，跳过安装流程"
fi