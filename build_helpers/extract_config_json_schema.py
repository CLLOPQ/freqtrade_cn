"""Script to extract the configuration json schema from config_schema.py file."""

from pathlib import Path

import rapidjson


def extract_config_json_schema():
    try:
        # 尝试从已安装的包中导入
        from freqtrade.config_schema import CONF_SCHEMA
    except ImportError:
        # 如果未安装freqtrade，将父目录添加到sys.path
        # 以便直接从源代码导入
        import sys

        script_dir = Path(__file__).parent
        freqtrade_dir = script_dir.parent
        sys.path.insert(0, str(freqtrade_dir))

        # 现在尝试从源代码导入
        from freqtrade.config_schema import CONF_SCHEMA

    # 定义输出的schema文件路径
    schema_filename = Path(__file__).parent / "schema.json"
    # 将配置模式以缩进格式写入JSON文件
    with schema_filename.open("w") as f:
        rapidjson.dump(CONF_SCHEMA, f, indent=2)


if __name__ == "__main__":
    extract_config_json_schema()
