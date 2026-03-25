# CLI 命令参考

Tachyon 提供四个命令，统一通过 `tachyon` 入口调用。
`tachyon --help` 查看总览，`tachyon <command> --help` 查看单个命令的用法。

```
tachyon --version      显示版本号
tachyon --help         显示帮助信息
```

---

## tachyon chat

AI 增强的交互式 CUDA 性能分析。打开一个类 REPL 的会话，你可以针对 kernel
性能自由提问。Agent 内置九个专用工具，用于查询指标、查看源码和分析 SASS 指令。

```
tachyon chat REPORT_FILE [OPTIONS]
```

### 参数

| 参数 | 说明 |
|------|------|
| `REPORT_FILE` | `.ncu-rep` 文件路径（必填）。 |

### 选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model`, `-m` | TEXT | 从配置读取 | LLM 模型（如 `gpt-4o`、`claude-sonnet-4-20250514`）。 |
| `--provider`, `-p` | TEXT | 从配置读取 | LLM 供应商（`openai` / `anthropic`）。 |
| `--no-ai` | flag | `false` | 强制纯规则模式（不使用 LLM）。 |
| `--lang` | TEXT | 从配置读取 | 输出语言（`en` / `zh`）。 |
| `--verbose`, `-v` | flag | `false` | 显示工具调用和调试信息。 |

### 会话内命令

在 chat 会话中可以使用以下斜杠命令：

| 命令 | 说明 |
|------|------|
| `/help` | 显示可用命令和使用提示。 |
| `/kernels` | 列出当前报告中的所有 kernel。 |
| `/tree` | 显示当前 kernel 的优化决策树。 |
| `/export` | 将对话导出到文件。 |
| `/quit` | 退出 chat 会话。 |

### 示例

```bash
# 使用默认模型进入 chat
tachyon chat report.ncu-rep

# 指定模型和供应商
tachyon chat report.ncu-rep -m claude-sonnet-4-20250514 -p anthropic

# 纯规则模式（不需要 LLM）
tachyon chat report.ncu-rep --no-ai

# 详细模式（查看工具调用）
tachyon chat report.ncu-rep -v
```

### 自动降级

当 LLM API Key 未配置或供应商不可达时，chat 命令会自动降级为纯规则模式。
在该模式下，分析命令仍可正常使用（依赖内置规则引擎），
只是无法进行自由形式的 AI 对话。

---

## tachyon profile

端到端 profiling 流水线：通过 NVIDIA Nsight Compute 的两阶段智能 profiling
运行 CUDA 程序，然后将结果送入完整的分析流水线。

```
tachyon profile EXECUTABLE [EXE_ARGS...] [OPTIONS]
```

### 参数

| 参数 | 说明 |
|------|------|
| `EXECUTABLE` | 待 profiling 的 CUDA 程序路径（必填）。 |
| `EXE_ARGS` | 传递给程序的参数（可选，可指定多个）。 |

### 选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--strategy` | `conservative` / `radical` | 从配置读取 | Profiling 策略。conservative 采集基础+详细指标；radical 额外采集全量指标和源码级计数器。 |
| `--kernel` | TEXT（可重复） | 全部 | 只 profile 指定的 kernel，可多次使用。 |
| `--top-k` | INT | `5` | Stage 2 深入分析的 top-K kernel 数量。 |
| `--ncu-args` | TEXT | 无 | 传给 `ncu` 的额外参数（带引号的字符串）。 |
| `--output`, `-o` | PATH | 临时目录 | 将 `.ncu-rep` 文件保存到指定目录。 |
| `--no-ai` | flag | `false` | 跳过 AI 增强分析（纯规则模式）。 |
| `--format` | `terminal` / `markdown` / `json` | `terminal` | 输出格式。 |
| `--model` | TEXT | 从配置读取 | 覆盖默认 LLM 模型。 |
| `--verbose`, `-v` | flag | `false` | 显示详细输出，包括 ncu 调用命令。 |

### 示例

```bash
# profile 一个简单程序
tachyon profile ./matmul

# 给程序传参
tachyon profile ./app --batch 32

# 使用 radical 策略，过滤指定 kernel
tachyon profile --strategy radical --kernel "matmul_*" ./app

# 保存报告，减少 top-K 数量
tachyon profile --top-k 3 -o ./reports ./app

# 给 ncu 传额外参数
tachyon profile --ncu-args "--replay-mode application" ./app
```

### 两阶段 Profiling

1. **Stage 1（快速扫描）：** 采集所有 kernel 的基础指标，按耗时排序找出 top-K kernel。
2. **Stage 2（深入分析）：** 只对 top-K kernel 做针对性的详细 profiling，采集全量指标，可选源码级计数器。

---

## tachyon diff

对比两份 NCU 报告，逐指标展示差异。
通过颜色标记回退和改进。

```
tachyon diff BEFORE_PATH AFTER_PATH [OPTIONS]
```

### 参数

| 参数 | 说明 |
|------|------|
| `BEFORE_PATH` | 基准 `.ncu-rep` 文件路径（必填）。 |
| `AFTER_PATH` | 对比 `.ncu-rep` 文件路径（必填）。 |

### 选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--kernel`, `-k` | TEXT | 全部 | 按名称过滤 kernel（glob 通配符）。 |
| `--threshold` | FLOAT | `5.0` | 显著性阈值（百分比）。变化低于该阈值的指标默认隐藏，除非加了 `--verbose`。 |
| `--verbose`, `-v` | flag | `false` | 显示所有指标，不只是显著变化的。 |

### 示例

```bash
# 对比两份报告
tachyon diff before.ncu-rep after.ncu-rep

# 聚焦某个 kernel
tachyon diff before.ncu-rep after.ncu-rep --kernel "matmul*"

# 降低显著性阈值
tachyon diff before.ncu-rep after.ncu-rep --threshold 2.0

# 显示所有指标变化
tachyon diff before.ncu-rep after.ncu-rep -v
```

### 回退检测

Tachyon 用简单的启发式规则判断是否发生回退：

- **越高越好的指标**（名称含 `throughput`、`pct`、`utilization`、`hit_rate`）：下降超过阈值视为回退。
- **越低越好的指标**（duration、stall 计数等）：上升超过阈值视为回退。

输出带颜色区分：红色表示回退，绿色表示改进。

---

## tachyon serve

将 Tachyon 作为工具服务器启动，供外部 AI Agent 调用。

```
tachyon serve [OPTIONS]
```

### 选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--mcp` | flag | `false` | 启动 MCP 服务器（stdio 传输），必选。 |
| `--report` | PATH | 无 | 预加载指定的 `.ncu-rep` 文件。 |

### 示例

```bash
# 启动 MCP 服务器，预加载报告
tachyon serve --mcp --report report.ncu-rep

# 启动 MCP 服务器，不预加载报告
tachyon serve --mcp
```

### MCP 协议

服务器通过 stdio 使用 Model Context Protocol 通信，
对外暴露 Tachyon 的全部九个工具：

**数据查询工具：**
- `list_kernels` -- 列出已加载报告中的所有 kernel。
- `get_kernel_metrics` -- 获取指定 kernel 的标量指标。
- `get_kernel_summary` -- 获取 kernel 特征的简要概述。
- `get_ncu_rule_results` -- 获取 NCU 内置规则的分析结果。

**源码关联工具：**
- `get_source_hotspots` -- 获取 Top-N 源码热点。
- `get_sass_for_source_line` -- 获取指定源码行对应的 SASS 指令。
- `get_stall_analysis_for_line` -- 获取指定源码行的 warp stall 分解。

**分析工具：**
- `run_analysis` -- 对 kernel 运行基于规则的分析器。
- `get_optimization_tree` -- 获取优化决策树及其活跃路径。

---

## 常用工作流

### 工作流 1：快速定位瓶颈

```bash
tachyon chat report.ncu-rep --no-ai
```

使用纯规则模式快速分析，了解最严重的问题。

### 工作流 2：深入分析特定 Kernel

```bash
tachyon chat report.ncu-rep
```

进入 chat 会话后：
```
> 这个 kernel 的主要瓶颈在哪？
> 显示源码热点。
> 第 42 行对应的 SASS 指令是什么？
```

### 工作流 3：优化前后对比

```bash
# profile 原始版本
tachyon profile -o ./baseline ./app

# （修改 CUDA 代码）

# profile 优化后的版本
tachyon profile -o ./optimized ./app

# 对比结果
tachyon diff ./baseline/*.ncu-rep ./optimized/*.ncu-rep
```

### 工作流 4：CI/CD 回退检测

```bash
tachyon diff baseline.ncu-rep current.ncu-rep --threshold 3.0
echo $?  # 检测到回退时返回非零退出码
```
