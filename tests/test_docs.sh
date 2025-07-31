#!/bin/bash
# 测试文档格式 - 检查文档中特定标记的格式问题
# !!! <TYPE>: 这种格式是不允许的！
# !!! <TYPE> "title" - 标题需要用引号括起来！
# ??? 标记也有相同的要求

# 查找不符合格式的标记：
# 1. 匹配 "!!! 类型:" 或 "??? 类型:" 这种带冒号的格式
# 2. 匹配 "!!! 类型 标题" 或 "??? 类型 标题" 这种标题没有引号的格式
grep -Er '^(!{3}|\?{3})\s\S+:|^(!{3}|\?{3})\s\S+\s[^"]' docs/*
format_issues=$?  # 记录grep命令的退出状态，0表示未找到匹配项，1表示找到匹配项

failed=0  # 用于记录整体测试结果，0表示成功，1表示失败

# 检查文件中"!!!"或"???"标记的格式
# 这些标记后面的非空行必须以4个空格开头
while read -r file; do
    # 使用awk处理每个文件，检查标记后的格式
    awk -v fname="$file" '
    /^(!!!|\?\?\?)/ {  # 匹配以!!!或???开头的行
        current_line_number = NR  # 记录当前行号
        current_line = $0         # 记录当前行内容
        found_next_content = 0    # 标记是否找到后续内容
        
        # 读取下一行内容
        while (getline nextLine > 0) {
            # 跳过空行
            if (nextLine ~ /^$/) {
                continue
            }
            
            found_next_content = 1  # 找到非空内容行
            
            # 检查下一行是否以4个空格开头
            if (nextLine !~ /^    /) {
                print "文件:", fname
                print "行", current_line_number, "错误: 期望下一个非空行以4个空格开头"
                print ">>", current_line
                print "------"
                exit 1  # 退出awk，返回错误状态
            }
            break  # 找到正确格式的行，退出循环
        }
        
        # 如果没有找到后续内容
        if (!found_next_content) {
            print "文件:", fname
            print "行", current_line_number, "错误: 找到标记但没有后续内容"
            print ">>", current_line
            print "------"
            exit 1  # 退出awk，返回错误状态
        }
    }
    ' "$file"

    # 检查awk命令的退出状态，如果非0则表示有错误
    if [ $? -ne 0 ]; then
        failed=1  # 设置整体测试结果为失败
    fi
done < <(find . -type f -name "*.md")  # 查找所有.md文件作为输入

# 判断测试结果：如果格式检查和内容检查都通过
if  [ $format_issues -eq 1 ] && [ $failed -eq 0 ]; then
    echo "文档测试成功。"
    exit 0
fi

# 否则测试失败
echo "文档测试失败。"
exit 1