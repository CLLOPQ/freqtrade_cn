# 通过 https://github.com/xmatthias/ta-lib-python/tree/ta_bundled_040 编译的 vendored Wheels

# 升级pip工具
python -m pip install --upgrade pip

# 打印当前Python版本号（主版本.次版本）
python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"

# 升级wheel工具并安装指定版本的numpy
pip install -U wheel "numpy<3.0"

# 安装ta-lib，仅使用二进制包，从build_helpers目录查找
pip install --only-binary ta-lib --find-links=build_helpers ta-lib

# 安装开发环境所需依赖
pip install -r requirements-dev.txt

# 以可编辑模式安装当前项目
pip install -e .