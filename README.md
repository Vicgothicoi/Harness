# Multi Coding Agent

基于 Profile 驱动的多角色 Coding Agent：复现 Anthropic [Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps) 中的长任务编排思路

核心循环是通用的 **Plan → Build → Evaluate → Iterate**；具体场景（提示词、工具、评分、钩子）由 Profile 注入

## 特性

- **多角色编排**：Planner / Builder / Evaluator，任务需要多步拆分时可选 Contract 协商
- **可插拔 Profile**：`app-builder`、`terminal`、`swe-bench`、`reasoning`
- **四层记忆**：工作记忆、状态记忆、项目记忆、长期记忆
- **四层上下文压缩**：观察压缩、轨迹压缩、状态压缩、全文重置（handoff）
- **工具与子 Agent**：读写文件、持久 Shell、并行委派、联网检索、偏好记忆
- **钩子**：循环检测、退出前强制验证、时间预算、错误引导、失败恢复
- **Skills 渐进披露**：启动时只注入目录，由 Agent 按需读取 `SKILL.md`
- **浏览器评测 MCP**：`app-builder` 的 Evaluator 可通过 Playwright 做 UI 测试
- **Harbor / Terminal-Bench**：可作为 Installed Agent 跑基准

## 架构

```
用户任务
    │
    ▼
┌──────────────────────────────────────────────────┐                
│  Plan → [Contract] → Build → Evaluate → Iterate  |
└──────────────────────────────────────────────────┘
    │
    ├── Planner          写出 spec.md
    ├── Contract*        协商本轮 contract.md（可选）
    ├── Builder          写代码 / 执行命令
    └── Evaluator        打分并写 feedback.md
              │
              ▼
┌────────────────────────────────────────────┐
│  LLM ↔ Memory ↔ Tool(MCP) ↔ Hook ↔ Skill   │  
└────────────────────────────────────────────┘
```


## 快速开始

安装依赖

```bash
pip install -r requirements.txt
python -m playwright install chromium   # app-builder 浏览器评测需要

cp .env.template .env                   # 填入 API Key
```

命令行运行

```bash
# 默认 profile：app-builder
python harness.py "Build a DAW in the browser"

# or
python harness.py --profile terminal "Fix the broken symlinks in /tmp"

# or
$env:HARNESS_FLAT_WORKSPACE = "1"  
$env:HARNESS_WORKSPACE = "./workspace/logic_error"  
python harness.py --profile swe-bench "Fix the logical error in topKFrequent()"

# or
python harness.py --profile reasoning "What is the escape velocity of Mars?"
```

每次运行会在 `workspace/` 下创建带时间戳的子目录，并 `git init`。产物默认在该目录中。

## Profile

| 名称 | 适用场景 | 说明 |
|------|----------|------|
| `app-builder` | 从一句话搭完整 Web 应用 | 含 Contract 协商；Evaluator 挂载浏览器 MCP |
| `terminal` | Terminal-Bench 风格 CLI 任务 | 时间预算、循环检测、退出验证、恢复策略 |
| `swe-bench` | 在真实仓库里修 GitHub issue | 定位 → 最小补丁 → 跑测试 |
| `reasoning` | 知识密集型问答 | 分步计算/推理，答案写入 `answer.md` |

自定义场景：继承 `profiles/base.py` 中的 `BaseProfile`，实现后再注册到 `profiles/__init__.py`。

## 记忆

Context Builder（`memory/working_memory.py`）按固定顺序把各层投影进对话窗口，原地替换、不重复追加：

| 层 | 作用 | 落盘 |
|----|------|------|
| **工作记忆** | 下面三层的窗口投影 | 不单独落盘 |
| **状态记忆** | TaskBoard：目标、步骤、阻塞、下一步 | 工作区 `progress.md`，每轮自动压缩更新 |
| **项目记忆** | 技术栈、关键文件、决策、已知问题 | 工作区 `project_memory.json`；规划后 seed，每轮 build 后 refresh |
| **长期记忆** | 跨任务用户偏好；任务结束写入 | 默认 `~/.harness/memory/long_term_memory.json` |

## 上下文压缩

压缩按四层递进：

| 层 | 触发 | 行为 |
|----|------|------|
| **观察压缩** | 每次工具返回 | 规则截断；统一上限 `OBSERVATION_MAX_CHARS` |
| **轨迹压缩** | token 超过 `COMPRESS_THRESHOLD` | 旧 ActionRecord 进 buffer，生成 `[TRACE SUMMARY]`，保留最近若干轮 |
| **状态压缩** | 每轮迭代 | LLM 更新 TaskBoard，写入 `progress.md` / `[STATE MEMORY]` |
| **全文压缩** | 超过 `RESET_THRESHOLD`或上下文焦虑 | 写出 `handoff.md`，清空对话并重建窗口；随后由 Context Builder 重新投影记忆 |

## 工具

Agent 在隔离的 `config.WORKSPACE` 内操作，路径不允许逃出工作区。

| 工具 | 用途 |
|------|------|
| `read_file` / `write_file` / `list_files` | 工作区内文件 |
| `read_skill_file` | 读取 `skills/` 下的技能文档 |
| `run_bash` | 持久 Shell 会话（保留 cwd / env） |
| `delegate_task` / `delegate_tasks` | 隔离上下文的子 Agent  |
| `web_search` / `web_fetch` | 查文档、拉网页 |
| `remember_preference` | 写入全局长期偏好 |

工具层会做参数预校验与常见错误自动纠正（空路径、误用绝对路径等）。执行失败由工具返回错误信息给 Agent，不交给 Hook。

`app-builder` 的 Evaluator 额外通过 MCP 使用浏览器工具（导航、截图、交互等）。

## 钩子

Hook 在不改核心循环的前提下塑造行为，由 Profile 按需挂载：

| 钩子 | 时机 | 典型用途 |
|------|------|----------|
| `before_tool` | 工具执行前 | 硬拦截（如恢复模式下禁止某些写入） |
| `post_tool` | 工具执行后 | 循环检测、错误引导 |
| `per_iteration` | 每轮开始 | 时间预算告警 |
| `pre_exit` | Agent 准备结束时 | 强制再验证一遍 |

内置实现：`LoopDetectionHook`、`PreExitVerificationHook`、`TimeBudgetHook`、`ErrorGuidanceHook`、`RecoveryStrategyHook`。

## 技能

遵循 Anthropic 三级渐进披露：

1. 启动扫描 `skills/*/SKILL.md` 的 YAML frontmatter，只把 **名称 + 描述** 注入系统提示
2. Agent 认为相关时，自己 `read_skill_file` 读取完整 `SKILL.md`
3. 文中引用的子文件再按需读取

新增技能：在 `skills/<name>/` 下放带 frontmatter 的 `SKILL.md` 即可。


## 测试

```bash
python -m unittest discover -s tests -v
```

覆盖工作记忆、状态/项目/长期记忆、观察与轨迹压缩、Shell 会话、恢复策略、terminal 流程以及 Harbor 适配器。

## Terminal-Bench

通过 Harbor 以 Installed Agent 方式在容器内跑 `harness.py --profile terminal`：

```bash
pip install harbor

harbor run -d "terminal-bench@2.0" \
  --agent-import-path benchmarks.harbor_agent:HarnessAgent \
  --task-names hello-world
```

也可用仓库内启动脚本 `benchmarks/run_terminal_bench.py`。批量结果分析见 `scripts/analyze_results.py`。
