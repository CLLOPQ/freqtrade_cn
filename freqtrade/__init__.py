"""Freqtrade 机器人"""

__version__ = "2025.6"

if "dev" in __version__:
    from pathlib import Path

    try:
        import subprocess  # noqa: S404, RUF100

        freqtrade_basedir = Path(__file__).parent

        __version__ = (
            __version__
            + "-"
            + subprocess.check_output(
                ["git", "log", '--format="%h"', "-n 1"],
                stderr=subprocess.DEVNULL,
                cwd=freqtrade_basedir,
            )
            .decode("utf-8")
            .rstrip()
            .strip('"')
        )

    except Exception:  # pragma: no cover
        # Git不可用，忽略
        try:
            # 尝试回退到freqtrade_commit文件（由CI在构建Docker镜像时创建）
            versionfile = Path("./freqtrade_commit")
            if versionfile.is_file():
                __version__ = f"docker-{__version__}-{versionfile.read_text()[:8]}"
        except Exception:  # noqa: S110
            pass