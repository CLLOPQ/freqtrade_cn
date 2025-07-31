import shutil
from pathlib import Path

import ast_comments

from freqtrade.constants import Config


class StrategyUpdater:
    name_mapping = {
        "ticker_interval": "timeframe",
        "buy": "enter_long",
        "sell": "exit_long",
        "buy_tag": "enter_tag",
        "sell_reason": "exit_reason",
        "sell_signal": "exit_signal",
        "custom_sell": "custom_exit",
        "force_sell": "force_exit",
        "emergency_sell": "emergency_exit",
        # Strategy/config settings:
        "use_sell_signal": "use_exit_signal",
        "sell_profit_only": "exit_profit_only",
        "sell_profit_offset": "exit_profit_offset",
        "ignore_roi_if_buy_signal": "ignore_roi_if_entry_signal",
        "forcebuy_enable": "force_entry_enable",
    }

    function_mapping = {
        "populate_buy_trend": "populate_entry_trend",
        "populate_sell_trend": "populate_exit_trend",
        "custom_sell": "custom_exit",
        "check_buy_timeout": "check_entry_timeout",
        "check_sell_timeout": "check_exit_timeout",
        # '': '',
    }
    # order_time_in_force, order_types, unfilledtimeout
    otif_ot_unfilledtimeout = {
        "buy": "entry",
        "sell": "exit",
    }

    # create a dictionary that maps the old column names to the new ones
    rename_dict = {"buy": "enter_long", "sell": "exit_long", "buy_tag": "enter_tag"}

    def start(self, config: Config, strategy_obj: dict) -> None:
        """
        运行策略更新器
        它借助ast模块将策略更新到v3版本
        :return: None
        """

        source_file = strategy_obj["location"]
        strategies_backup_folder = Path.joinpath(config["user_data_dir"], "strategies_orig_updater")
        target_file = Path.joinpath(strategies_backup_folder, strategy_obj["location_rel"])

        # 读取文件
        with Path(source_file).open("r") as f:
            old_code = f.read()
        if not strategies_backup_folder.is_dir():
            Path(strategies_backup_folder).mkdir(parents=True, exist_ok=True)

        # 备份原始文件
        # => 当前文件名后没有日期，如果此操作被触发两次，可能会被快速覆盖！目前文件夹和文件名总是相同的。
        shutil.copy(source_file, target_file)

        # 更新代码
        new_code = self.update_code(old_code)
        # 将修改后的代码写入目标文件夹
        with Path(source_file).open("w") as f:
            f.write(new_code)

    # 定义用于更新代码的函数
    def update_code(self, code):
        # 将代码解析为AST
        tree = ast_comments.parse(code)

        # 使用AST更新代码
        updated_code = self.modify_ast(tree)

        # 返回修改后的代码（不执行）
        return updated_code

    # 使用ast模块更新代码的函数
    def modify_ast(self, tree):
        # 使用访问器更新AST中的名称和函数
        NameUpdater().visit(tree)

        # 首先修复注释，以便它能正确理解多行注释中的"\n"
        ast_comments.fix_missing_locations(tree)
        ast_comments.increment_lineno(tree, n=1)

        # 从更新后的AST生成新代码
        # 不使用{}参数的话，会直接连在一起写。

        # ast_comments非常棒，因为这是唯一能保留注释的解决方案，但目前它没有unparse函数，希望未来能有！
        # return ast_comments.unparse(tree)

        return ast_comments.unparse(tree)


# 这里遍历每个相应的节点、切片、元素、键...以替换过时的条目。
class NameUpdater(ast_comments.NodeTransformer):
    def generic_visit(self, node):
        # 空间尚未从买入/卖出转移到入场/出场，因此必须跳过。
        if isinstance(node, ast_comments.keyword):
            if node.arg == "space":
                return node

        # 从这里开始是原始函数。
        for field, old_value in ast_comments.iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast_comments.AST):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, ast_comments.AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast_comments.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node

    def visit_Expr(self, node):
        if hasattr(node.value, "left") and hasattr(node.value.left, "id"):
            node.value.left.id = self.check_dict(StrategyUpdater.name_mapping, node.value.left.id)
            self.visit(node.value)
        return node

    # 将元素重命名（如果存在于字典中）。
    @staticmethod
    def check_dict(current_dict: dict, element: str):
        if element in current_dict:
            element = current_dict[element]
        return element

    def visit_arguments(self, node):
        if isinstance(node.args, list):
            for arg in node.args:
                arg.arg = self.check_dict(StrategyUpdater.name_mapping, arg.arg)
        return node

    def visit_Name(self, node):
        # 如果名称在映射中，则更新它
        node.id = self.check_dict(StrategyUpdater.name_mapping, node.id)
        return node

    def visit_Import(self, node):
        # 不更新import语句中的名称
        return node

    def visit_ImportFrom(self, node):
        # if hasattr(node, "module"):
        #    if node.module == "freqtrade.strategy.hyper":
        #        node.module = "freqtrade.strategy"
        return node

    def visit_If(self, node: ast_comments.If):
        for child in ast_comments.iter_child_nodes(node):
            self.visit(child)
        return node

    def visit_FunctionDef(self, node):
        node.name = self.check_dict(StrategyUpdater.function_mapping, node.name)
        self.generic_visit(node)
        return node

    def visit_Attribute(self, node):
        if (
            isinstance(node.value, ast_comments.Name)
            and node.value.id == "trade"
            and node.attr == "nr_of_successful_buys"
        ):
            node.attr = "nr_of_successful_entries"
        return node

    def visit_ClassDef(self, node):
        # 检查该类是否继承自IStrategy
        if any(
            isinstance(base, ast_comments.Name) and base.id == "IStrategy" for base in node.bases
        ):
            # 检查INTERFACE_VERSION变量是否存在
            has_interface_version = any(
                isinstance(child, ast_comments.Assign)
                and isinstance(child.targets[0], ast_comments.Name)
                and child.targets[0].id == "INTERFACE_VERSION"
                for child in node.body
            )

            # 如果INTERFACE_VERSION变量不存在，则将其作为第一个子节点添加
            if not has_interface_version:
                node.body.insert(0, ast_comments.parse("INTERFACE_VERSION = 3").body[0])
            # 否则，将其值更新为3
            else:
                for child in node.body:
                    if (
                        isinstance(child, ast_comments.Assign)
                        and isinstance(child.targets[0], ast_comments.Name)
                        and child.targets[0].id == "INTERFACE_VERSION"
                    ):
                        child.value = ast_comments.parse("3").body[0].value
        self.generic_visit(node)
        return node

    def visit_Subscript(self, node):
        if isinstance(node.slice, ast_comments.Constant):
            if node.slice.value in StrategyUpdater.rename_dict:
                # 将切片属性替换为rename_dict中的值
                node.slice.value = StrategyUpdater.rename_dict[node.slice.value]
        if hasattr(node.slice, "elts"):
            self.visit_elts(node.slice.elts)
        if hasattr(node.slice, "value"):
            if hasattr(node.slice.value, "elts"):
                self.visit_elts(node.slice.value.elts)
        return node

    # elts可能包含elts（从技术上讲是递归的）
    def visit_elts(self, elts):
        if isinstance(elts, list):
            for elt in elts:
                self.visit_elt(elt)
        else:
            self.visit_elt(elts)
        return elts

    # 由于结构本身高度灵活，需要再次使用子函数...
    def visit_elt(self, elt):
        if isinstance(elt, ast_comments.Constant) and elt.value in StrategyUpdater.rename_dict:
            elt.value = StrategyUpdater.rename_dict[elt.value]
        if hasattr(elt, "elts"):
            self.visit_elts(elt.elts)
        if hasattr(elt, "args"):
            if isinstance(elt.args, ast_comments.arguments):
                self.visit_elts(elt.args)
            else:
                for arg in elt.args:
                    self.visit_elts(arg)
        return elt

    def visit_Constant(self, node):
        node.value = self.check_dict(StrategyUpdater.otif_ot_unfilledtimeout, node.value)
        node.value = self.check_dict(StrategyUpdater.name_mapping, node.value)
        return node