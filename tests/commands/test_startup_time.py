import subprocess  # noqa: S404, RUF100
import time

from tests.conftest import is_arm, is_mac


MAXIMUM_STARTUP_TIME = 0.7 if is_mac() and not is_arm() else 0.5


def test_startup_time():
    # 预热以生成pyc文件
    subprocess.run(["freqtrade", "-h"])

    start = time.time()
    subprocess.run(["freqtrade", "-h"])
    elapsed = time.time() - start
    assert elapsed < MAXIMUM_STARTUP_TIME, (
        "启动时间过长，请尝试在命令入口函数中使用延迟导入"
        f" (最大允许 {MAXIMUM_STARTUP_TIME}秒, 实际 {elapsed}秒)"
    )
