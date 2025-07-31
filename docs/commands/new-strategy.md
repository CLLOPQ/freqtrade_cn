usage: freqtrade new-strategy [-h] [--userdir PATH] [-s NAME]
                              [--strategy-path PATH]
                              [--template {full,minimal,advanced}]

options:
  -h, --help            显示此帮助消息并退出
  --userdir PATH, --user-data-dir PATH
                        用户数据目录的路径。
  -s NAME, --strategy NAME
                        指定机器人将使用的策略类名称。
  --strategy-path PATH  指定额外的策略查找路径。
  --template {full,minimal,advanced}
                        使用模板，可选 `minimal`（最小化）、`full`（包含多个示例指标）或 `advanced`（高级）。默认值：`full`。