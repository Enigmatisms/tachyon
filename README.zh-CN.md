# Tachyon

[中文文档 (当前)](./README.zh-CN.md) | English Doc 

**AI empowered CUDA kernel profiler**

Tachyon `/ˈtakēˌän/` (迅子，物理上一种只存在于理论上的速度快于光速的粒子) 是一个 CUDA kernel 性能分析与自动优化工具。它结合了大模型 Agent 与传统规则分析，将 NCU 指标、CUDA 源码和底层指令（PTX/SASS）打通，使性能分析不止停留于聚合指标，更能通过指令级分析回溯到源码行。除了分析能力，**evolve** 模式实现了自动优化功能：Agent 读取 profiling 数据，自动编辑源码，编译、重新 profiling，迭代直到收敛——无需手动调参。

本工具完全集成在 Python 中，支持端到端 profiling（类似 ncu，直接运行可执行文件）、AI 交互式分析，以及全自动迭代优化。支持多种 LLM 后端。

## 特性

- **三路串联**: NCU 指标 <-> 源码行 <-> SASS 指令，精确定位每个瓶颈对应的代码
- **两阶段智能 Profiling**: 快速扫描找出最热的 Top-K kernel，再对它们做深度采集
- **规则引擎 + AI Agent**: 7 个内置分析器 (roofline, memory, occupancy, warp stall, ...) 输出结构化结果；LLM Agent 带 9 个专用工具，支持交互式追问
- **Profile Diff**: 对比两份 `.ncu-rep` 报告，高亮性能回退
- **Evolve 模式**: 自动迭代优化 — LLM Agent 读取 NCU profiling 数据，编辑 CUDA 源码，编译、重新 profiling，逐轮接受或回退。不再需要手动调参循环
- **MCP Server**: 通过 MCP 协议 (stdio) 与 Claude Code / Ducc 无缝集成

## 快速开始

### 安装

```bash
# 仅规则引擎（不需要 LLM）
pip install -e .

# 带 AI Agent（OpenAI / Anthropic / LiteLLM）
pip install -e ".[ai]"

# 带 MCP Server
pip install -e ".[mcp]"

# 完整开发环境
pip install -e ".[dev,ai,mcp]"
```

### 基本用法

```bash
tachyon analyze report.ncu-rep
tachyon chat report.ncu-rep --model claude-sonnet-4-20250514
tachyon profile ./my_app --strategy radical
tachyon diff before.ncu-rep after.ncu-rep
tachyon evolve ./my_app --build "make -j8" --max-iterations 10
tachyon serve --mcp --report report.ncu-rep
```

## CLI 命令

| 命令 | 说明 |
|------|------|
| `tachyon analyze` | 解析 `.ncu-rep`，跑规则分析器，输出 findings |
| `tachyon chat` | AI 交互式分析，9 个专用工具 |
| `tachyon profile` | 端到端: profile CUDA 程序 → 自动分析 |
| `tachyon diff` | 对比两份报告，标记性能变化 |
| `tachyon evolve` | LLM Agent 自动迭代优化 kernel 性能 |
| `tachyon serve` | 启动 MCP server，给外部 agent 调用 |

`tachyon <command> --help` 看完整选项。详见 [CLI 命令参考](docs/cli-reference.md)。

## 多 LLM Vendor 切换

```toml
# ~/.tachyon/config.toml
[llm]
provider = "anthropic"          # openai / anthropic / litellm
model = "claude-sonnet-4-20250514"
api_key_env = "ANTHROPIC_API_KEY"
# base_url = "http://localhost:8000/v1"  # 本地模型
```

```bash
# 环境变量切换
export TACHYON_LLM_PROVIDER=litellm
export TACHYON_MODEL=qianfan/ERNIE-Bot-4

# CLI 单次切换
tachyon chat report.ncu-rep --provider openai --model gpt-4o
```

优先级: CLI > 环境变量 > config.toml > 默认值。无 LLM 时自动降级到 Rule-Only 模式。

## Ducc (度厂-达克) 集成

```json
// .claude/settings.json
{
  "mcpServers": {
    "tachyon": {
      "command": "tachyon",
      "args": ["serve", "--mcp", "--report", "./report.ncu-rep"]
    }
  }
}
```

启动 Ducc 后 9 个工具自动注册: list_kernels, get_kernel_metrics, get_source_hotspots, run_analysis 等。

## SDK 用法

Tachyon 也可以当 Python 库用:

```python
from tachyon import NcuProfiler, ToolPathResolver, TachyonConfig

config = TachyonConfig.load()
resolver = ToolPathResolver(config)
profiler = NcuProfiler(config, resolver)
result = profiler.profile_basic("./my_cuda_app")
```

详见 [SDK 使用指南](docs/sdk-guide.md)。

## 配置

`~/.tachyon/config.toml`，支持环境变量和 CLI 参数覆盖。

```toml
[llm]
provider = "anthropic"
model = "claude-sonnet-4-20250514"
temperature = 0.1

[profiling]
strategy = "conservative"

[output]
lang = "zh"
format = "terminal"

[tools]
ncu_path = "/usr/local/cuda/bin/ncu"
```

详见 [配置说明](docs/configuration.md)。

## 架构 (8 层)

```
CLI/Chat > Agent Loop > LLM Backends > Tools(9) > Analyzers(7) > Correlator > Reader > Models
```

数据自底向上流动: `.ncu-rep` → Reader 解析为 KernelReport → Correlator 建立三路映射 → Analyzers 输出 Findings → CLI 渲染 / Agent 交互分析。

详见 [架构文档](docs/architecture.md)。

## 开发

### 环境要求

- Python 3.10+
- NVIDIA Nsight Compute（profiling 和报告解析需要）

### 测试

```bash
pip install -e ".[dev,ai]"
pytest
pytest --cov=tachyon --cov-report=term-missing
mypy src/tachyon/
ruff check src/
```

### 项目结构

```
src/tachyon/
├── cli/           Click 命令 (analyze, chat, profile, diff, serve)
├── agent/         多轮 Agent Loop, persona, 上下文管理
├── llm/           LLM 后端抽象 (OpenAI, Anthropic, LiteLLM)
├── tools/         9 个 Agent 工具 (数据查询, 源码关联, 分析)
├── analyzers/     7 个规则分析器 (roofline, memory, occupancy, ...)
├── correlator/    三路映射引擎 (指标 <-> 源码 <-> SASS)
├── reader/        NCU .ncu-rep 报告解析
├── profiler/      两阶段智能 profiling, 工具路径发现
├── evolve/        迭代优化编排器, 工具, persona
├── models/        核心数据模型 (KernelReport, Finding, OptTree, ...)
├── config/        TOML 配置 + 分层覆盖
├── report/        输出渲染 (terminal, markdown)
├── diff/          Profile 对比引擎
├── server/        MCP server (stdio)
├── tree/          优化探索树
├── errors/        结构化错误处理 (ToolResult, ErrorCode)
└── i18n/          国际化
```

## 项目指标

733 tests | 88% coverage | 0 lint errors | 7 analyzers | 9 tools | 3 LLM backends

## 文档

| 文档 | 说明 |
|------|------|
| [架构](docs/architecture.md) | 8 层架构、数据流、模块职责、设计决策 |
| [CLI 命令参考](docs/cli-reference.md) | 5 个命令的完整选项和使用示例 |
| [配置说明](docs/configuration.md) | config.toml 各项配置、环境变量、优先级 |
| [SDK 使用指南](docs/sdk-guide.md) | Python SDK 接口、常见用法、API 文档 |
| [多 LLM 后端与 Ducc 集成](docs/multi-vendor-integration.md) | LLM 切换、Ducc MCP 集成、自定义 Provider |
| [Evolve 使用指南](docs/evolve-guide.md) | 自动迭代优化: 用法、配置、实际案例 |


## 实际效果 [WIP]

本部分见: [效果展示](assets/README.md)

## License

MIT
