# 贡献指南

## 为 freqtrade 做贡献

你觉得我们的机器人缺少某项功能吗？我们欢迎你的拉取请求（Pull Requests）！

标记为 [good first issue](https://github.com/freqtrade/freqtrade/labels/good%20first%20issue) 的问题非常适合作为初次贡献，这些问题能帮助你熟悉代码库。

贡献的几点提示：
- 请针对 `develop` 分支创建拉取请求，而非 `stable` 分支。
- 新功能需要包含单元测试，必须符合 PEP8 规范（最大行长度为 100），并且应在引入 PR 时提供文档说明。
- PR 可以标记为 `[WIP]`——表示这是正在进行中的拉取请求（尚未完成）。

如果你不确定，可先在我们的 [discord 服务器](https://discord.gg/p7nuUNVfP7) 上或通过 [issue](https://github.com/freqtrade/freqtrade/issues) 讨论该功能，再提交拉取请求。

## 入门指南

最好先阅读 [文档](https://www.freqtrade.io/)，了解机器人的功能，或者直接查看 [开发者文档](https://www.freqtrade.io/en/latest/developer/)（开发中），它会帮助你入门。

## 提交 PR 前的准备

### 1. 运行单元测试

所有单元测试必须通过。如果某个单元测试失败，需修改你的代码使其通过，因为这意味着你引入了回归问题。

#### 测试整个项目
```bash
pytest
```

#### 仅测试一个文件
```bash
pytest tests/test_<文件名>.py
```

#### 仅测试一个文件中的某个方法
```bash
pytest tests/test_<文件名>.py::test_<方法名>
```

### 2. 测试代码是否符合 PEP8 规范

#### 运行 Ruff
```bash
ruff check .
```

我们收到很多不符合 `ruff` 检查的代码。为避免这种情况，建议你安装 git 预提交钩子，当你尝试提交不符合检查的代码时，它会发出警告。

你可以使用 `pre-commit run -a` 手动运行预提交检查。

##### 额外的样式要求
- 所有公共方法都要有文档字符串（docstrings）
- 文档字符串使用双引号
- 多行文档字符串的缩进应与第一个引号的层级一致
- 文档字符串应遵循 reST 格式（`：param xxx：...`，`：return：...`，`：raises KeyError：...`）

### 3. 测试所有类型提示是否正确

#### 运行 mypy
```bash
mypy freqtrade
```

### 4. 确保格式正确

#### 运行 ruff
```bash
ruff format .
```

## （核心）提交者指南

### 拉取请求处理流程

拉取请求的优先级从高到低如下：
1. 修复失败的测试。“失败”指在任何支持的平台或 Python 版本上失败。
2. 补充测试以覆盖边缘情况。
3. 文档的小修改。
4. 错误修复。
5. 文档的重大修改。
6. 新功能。

确保每个拉取请求都符合贡献文档中的所有要求。

### 问题处理流程

如果某个问题是需要紧急修复的错误，将其标记为下一个补丁版本。然后要么修复它，要么标记为“请帮助（please - help）”。

对于其他问题：鼓励友好讨论、适度辩论、发表你的想法。

### 自身代码变更流程

所有代码变更，无论由谁进行，都需要由其他人审核和合并。此规则适用于所有核心提交者。

例外情况：
- 对他人提交的拉取请求进行的小更正和修复。
- 在正式发布时，发布经理可以进行必要的、适当的更改。
- 强化现有主题的小型文档更改，最常见的包括拼写和语法更正。

### 职责

- 确保每一项被接受的更改都具有跨平台兼容性，包括 Windows、Mac 和 Linux。
- 确保核心代码中没有引入恶意代码。
- 对于任何重大更改和增强功能，创建相关问题。透明地讨论，并获取社区反馈。
- 保持功能版本尽可能小，最好每个版本只包含一个新功能。
- 欢迎新手，鼓励来自不同背景的新贡献者。参见 Python 社区行为准则（https://www.python.org/psf/codeofconduct/）。

### 成为提交者

贡献者可能会获得提交权限。优先考虑以下人员：
1. 对 Freqtrade 和其他相关开源项目有过贡献的人。对 Freqtrade 的贡献包括代码（已接受和待处理的）以及在问题跟踪和拉取请求审核中的友好参与。会同时考虑数量和质量。
2. 编码风格被其他核心提交者认为简洁、精炼且清晰的人。
3. 拥有跨平台开发和测试资源的人。
4. 能定期投入时间到项目中的人。

出于安全原因（用户信任 Freqtrade 处理其交易所 API 密钥），成为提交者并不意味着获得 `develop` 或 `stable` 分支的写入权限。

在成为提交者一段时间后，提交者可能会被任命为核心提交者，并获得完整的仓库访问权限。