from collections.abc import Sequence
from typing import Any, TypeAlias

from pandas import DataFrame
from rich.table import Column, Table
from rich.text import Text

from freqtrade.loggers.rich_console import get_rich_console


TextOrString: TypeAlias = str | Text


def print_rich_table(
    tabular_data: Sequence[dict[str, Any] | Sequence[TextOrString]],
    headers: Sequence[str],
    summary: str | None = None,
    *,
    justify="right",
    table_kwargs: dict[str, Any] | None = None,
) -> None:
    # 创建表格
    table = Table(
        *[c if isinstance(c, Column) else Column(c, justify=justify) for c in headers],
        title=summary,  # 表格标题
        **(table_kwargs or {}),  # 其他表格参数
    )

    # 遍历数据行
    for row in tabular_data:
        if isinstance(row, dict):
            # 字典行：提取对应列的值并转换为文本
            table.add_row(
                *[
                    row[header] if isinstance(row[header], Text) else str(row[header])
                    for header in headers
                ]
            )

        else:
            # 序列行：将每个元素转换为文本或Text对象
            row_to_add: list[str | Text] = [r if isinstance(r, Text) else str(r) for r in row]
            table.add_row(*row_to_add)

    # 获取富控制台并打印表格
    console = get_rich_console()
    console.print(table)


def _format_value(value: Any, *, floatfmt: str) -> str:
    """格式化值为字符串，浮点数按指定格式显示"""
    if isinstance(value, float):
        return f"{value:{floatfmt}}"  # 按指定格式格式化浮点数
    return str(value)  # 非浮点数直接转换为字符串


def print_df_rich_table(
    tabular_data: DataFrame,
    headers: Sequence[str],
    summary: str | None = None,
    *,
    show_index=False,
    index_name: str | None = None,
    table_kwargs: dict[str, Any] | None = None,
) -> None:
    # 创建表格
    table = Table(title=summary, **(table_kwargs or {}))

    # 如果显示索引，添加索引列
    if show_index:
        index_name = str(index_name) if index_name else tabular_data.index.name
        table.add_column(index_name)

    # 添加数据列
    for header in headers:
        table.add_column(header, justify="right")

    # 遍历DataFrame行，格式化后添加到表格
    for value_list in tabular_data.itertuples(index=show_index):
        row = [_format_value(x, floatfmt=".3f") for x in value_list]  # 浮点数保留3位小数
        table.add_row(*row)

    # 获取富控制台并打印表格
    console = get_rich_console()
    console.print(table)