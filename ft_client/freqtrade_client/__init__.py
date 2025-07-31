from freqtrade_client.ft_rest_client import FtRestClient


__version__ = "2025.6"

if "dev" in __version__:
    from pathlib import Path

    try:
        import subprocess  # noqa: S404, RUF100

        freqtrade_base_dir = Path(__file__).parent

        __version__ = (
            __version__
            + "-"
            + subprocess.check_output(
                ["git", "log", '--format="%h"', "-n 1"],
                stderr=subprocess.DEVNULL,
                cwd=freqtrade_base_dir,
            )
            .decode("utf-8")
            .rstrip()
            .strip('"')
        )

    except Exception:  # pragma: no cover
        # 没有git环境，忽略错误
        try:
            # 尝试回退到freqtrade_commit文件（由CI在构建docker镜像时创建）
            version_file = Path("./freqtrade_commit")
            if version_file.is_file():
                __version__ = f"docker-{__version__}-{version_file.read_text()[:8]}"
        except Exception:  # noqa: S110
            pass

__all__ = ["FtRestClient"]