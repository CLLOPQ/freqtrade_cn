import os
from pathlib import Path


# 更改目录
# 修改此单元格以确保输出显示正确的路径。
# 相对于单元格输出中显示的项目根目录定义所有路径
project_root = "somedir/freqtrade"
i = 0
try:
    os.chdir(project_root)
    if not Path("LICENSE").is_file():
        i = 0
        while i < 4 and (not Path("LICENSE").is_file()):
            os.chdir(Path(Path.cwd(), "../"))
            i += 1
        project_root = Path.cwd()
except FileNotFoundError:
    print("请定义相对于当前目录的项目根目录")
print(Path.cwd())