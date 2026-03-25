# Tachyon 架构

Tachyon 采用八层栈式架构，每一层只依赖它下面的层，上层对下层没有反向依赖。这种设计让各层可以独立测试、独立演进，也方便按阶段交付功能。

---

## 项目目录

```
src/tachyon/
├── cli/            # 命令行入口（chat / profile / diff / evolve / serve）
├── agent/          # Agent 循环、系统提示词、上下文压缩
├── llm/            # LLM 后端适配（OpenAI / Anthropic）
├── tools/          # 9 个 Agent 可调用工具 + ToolRegistry
├── server/         # MCP stdio 服务
├── analyzers/      # 7 个规则分析器 + 插件式注册
├── correlator/     # 三向映射引擎（指标 ↔ 源码 ↔ SASS）
├── reader/         # NCU 报告解析（.ncu-rep → KernelReport）
├── profiler/       # 两阶段智能 Profiling + 工具路径自动发现
├── models/         # 核心数据模型（KernelReport / Finding / OptTree 等）
├── config/         # 分层配置（TOML → 环境变量 → CLI 参数）
├── report/         # 输出渲染（Rich 终端 / Markdown）
├── diff/           # 两次 profile 对比 + 回归检测
├── tree/           # 五分支优化树
├── errors/         # ToolResult 封装 + ErrorCode
└── i18n/           # 语言
```

---

## 分层架构

```mermaid
graph TD
    L8["<b>Layer 8 · CLI / Chat UI</b><br/>Click 命令 · Rich 终端 · 流式输出"]
    L7["<b>Layer 7 · Agent Loop</b><br/>多轮 LLM 编排（≤10 轮）· 工具调度<br/>Token 追踪 · Auto-Fallback"]
    L6["<b>Layer 6 · LLM Backends</b><br/>OpenAI / Anthropic<br/>流式响应 · tool-call 解析"]
    L5["<b>Layer 5 · Tools × 9</b><br/>数据查询 4 · 源码 3 · 分析 2<br/>JSON Schema 定义"]
    L4["<b>Layer 4 · Analyzers × 7</b><br/>Roofline · Memory · Occupancy · Instruction<br/>Launch · Warp Stall · NCU Rules<br/>插件注册 AnalyzerRegistry"]
    L3["<b>Layer 3 · Correlator</b><br/>NCU 指标 ↔ CUDA 源码 ↔ SASS 汇编<br/>双阈值热点检测"]
    L2["<b>Layer 2 · Reader / Profiler</b><br/>NcuReportReader · NcuProfiler<br/>ToolPathResolver 自动发现"]
    L1["<b>Layer 1 · Models / Config</b><br/>KernelReport · Finding · OptTree<br/>TachyonConfig（TOML + env + CLI）"]

    L8 --> L7
    L7 --> L6
    L7 --> L5
    L6 --> L5
    L5 --> L4
    L5 --> L3
    L4 --> L3
    L4 --> L2
    L3 --> L2
    L2 --> L1
    L4 --> L1
    L5 --> L1

    style L8 fill:#4a6fa5,stroke:#2d4a7a,color:#fff
    style L7 fill:#5b8c5a,stroke:#3d6b3c,color:#fff
    style L6 fill:#6b8e6b,stroke:#4a6d4a,color:#fff
    style L5 fill:#8a7b5e,stroke:#6b5d42,color:#fff
    style L4 fill:#9b7653,stroke:#7a5a3d,color:#fff
    style L3 fill:#a06b4a,stroke:#7f5238,color:#fff
    style L2 fill:#8b6e8e,stroke:#6b4f6e,color:#fff
    style L1 fill:#7a7a7a,stroke:#5a5a5a,color:#fff
```

---

## 各模块职责

**Layer 8 — 用户接口**

- `cli/` — 四个 Click 命令：`chat`、`profile`、`diff`、`serve`（加上 `evolve` 子命令）
- `server/` — MCP stdio 服务端，把 9 个工具暴露为 MCP Tool，处理调用分发

**Layer 7 — Agent 编排**

- `agent/` — 多轮对话循环，最多 10 轮，管理会话历史，调度工具调用，末轮强制综合输出
- `agent/persona.py` — 从 `persona.md` 模板构造系统提示词，注入工具目录和 kernel 上下文
- `agent/context.py` — 上下文窗口管理，token 超预算时自动压缩历史

**Layer 6 — LLM 后端**

- `llm/` — 抽象基类 `LLMBackend`（`chat_completion()` + 流式），OpenAI / Anthropic 两种实现，工厂函数 `create_backend(config)` 按配置创建

**Layer 5 — 工具层**

- `tools/registry.py` — `ToolDefinition`（名称、描述、JSON Schema、handler）和 `ToolRegistry`（注册 / 查找 / 执行），支持导出 OpenAI、Anthropic、MCP 三种格式
- `tools/data_query.py` — 4 个数据查询工具：`list_kernels`、`get_kernel_metrics`、`get_kernel_summary`、`get_ncu_rule_results`
- `tools/source.py` — 3 个源码工具：`get_source_hotspots`、`get_sass_for_source_line`、`get_stall_analysis_for_line`
- `tools/analysis.py` — 2 个分析工具：`run_analysis`、`get_optimization_tree`
- `tools/context.py` — `SessionContext`，持有已加载的 kernel、correlator、analyzer registry

**Layer 4 — 分析引擎**

- `analyzers/` — 抽象基类 `Analyzer` + 7 个具体分析器，`AnalyzerRegistry` 自动发现子类，可选注入 `SourceCorrelator` 做源码级归因
- `report/` — 输出渲染：`TerminalReporter`（Rich）和 `MarkdownReporter`
- `diff/` — `ProfileDiffer`，按名称匹配 kernel，计算逐指标 delta，检测回归
- `tree/` — `OptimizationTree`，从 Finding 构建五分支优化树，按检测到的瓶颈裁剪

**Layer 3 — 关联映射**

- `correlator/` — `SourceCorrelator`，把 PC 地址映射到源码行和 SASS 指令，按源码行聚合 stall 和执行指标，产出 `SourceHotspot`

**Layer 2 — 数据采集**

- `reader/` — `NcuReportReader`，封装 NVIDIA `ncu_report.py`，把 `.ncu-rep` 解析为 `KernelReport`，自动发现 NCU Python 包
- `profiler/` — `NcuProfiler` 两阶段采集（快扫全量 → 深钻 Top-K），`ToolPathResolver` 自动发现 `ncu` / `nvdisasm` / `cuobjdump`，`pipeline.py` 端到端编排

**Layer 1 — 基础设施**

- `models/` — 核心类型：`KernelReport`、`LaunchParams`、`DeviceInfo`、`MetricValue`、`Finding`、`Severity`、`SourceLocation`、`OptimizationNode`、`SourceHotspot`
- `config/` — `TachyonConfig` 四段配置（`llm` / `profiling` / `output` / `tools`），加载优先级：默认值 → TOML → 环境变量 → CLI 参数
- `errors/` — `ToolResult`（成功/失败信封）、`ErrorCode` 枚举
- `i18n/` — 输出字符串国际化

---

## 数据流

### 规则分析（profile --no-ai 模式）

```mermaid
graph LR
    A[".ncu-rep 文件"] --> B["NcuReportReader.load()"]
    B --> C["list&lt;KernelReport&gt;"]
    C --> D["AnalyzerRegistry.run_all()"]
    D --> E["list&lt;Finding&gt;"]
    E --> F["TerminalReporter.render()"]
    F --> G["终端 / Markdown / JSON"]

    style A fill:#e8e8e8,stroke:#999
    style G fill:#4a6fa5,stroke:#2d4a7a,color:#fff
```

### AI 对话（chat 命令）

```mermaid
sequenceDiagram
    participant U as 用户
    participant CLI as CLI
    participant R as NcuReportReader
    participant SC as SourceCorrelator
    participant AL as AgentLoop
    participant LLM as LLM Backend
    participant T as Tools (×9)

    U->>CLI: chat report.ncu-rep
    CLI->>R: load()
    R-->>CLI: list[KernelReport]
    CLI->>SC: build()
    SC-->>CLI: 三向映射就绪
    CLI->>AL: 启动（SessionContext + ToolRegistry）
    loop 最多 10 轮
        AL->>LLM: chat_completion()
        LLM-->>AL: 文本 或 tool_call
        opt LLM 请求调用工具
            AL->>T: execute(tool_name, args)
            T-->>AL: ToolResult
        end
    end
    AL-->>CLI: 综合分析结果
    CLI-->>U: Rich 流式输出
```

### 端到端采集（profile 命令）

```mermaid
graph LR
    A["CUDA 可执行文件"] --> B["ToolPathResolver.resolve('ncu')"]
    B --> C["ncu 路径"]
    C --> D["NcuProfiler.profile_basic()"]
    D -->|"Stage 1: 快扫全量"| E["NcuProfiler.profile_detailed()"]
    E -->|"Stage 2: 深钻 Top-K"| F["AnalyzerRegistry.run_all()"]
    F --> G["list&lt;Finding&gt;"]
    G --> H["TerminalReporter.render()"]

    style A fill:#e8e8e8,stroke:#999
    style H fill:#4a6fa5,stroke:#2d4a7a,color:#fff
```

### 两次对比（diff 命令）

```mermaid
graph LR
    A["before.ncu-rep"] --> C["NcuReportReader.load()"]
    B["after.ncu-rep"] --> C
    C --> D["两组 list&lt;KernelReport&gt;"]
    D --> E["ProfileDiffer.diff()"]
    E --> F["list&lt;KernelDiff&gt; + MetricDelta"]
    F --> G["回归检测 → 彩色终端输出"]

    style A fill:#e8e8e8,stroke:#999
    style B fill:#e8e8e8,stroke:#999
    style G fill:#4a6fa5,stroke:#2d4a7a,color:#fff
```

---

## 关键设计决策

**1. 工具路径不硬编码**

NVIDIA 工具在不同机器上的安装位置千差万别。`ToolPathResolver` 按优先级搜索：配置文件 → PATH → `$CUDA_HOME` → 常见安装目录 → glob 模式，找到后缓存到会话结束。

**2. 分析器插件化**

每个分析器是一个独立的 `Analyzer` 子类。`AnalyzerRegistry.auto_register()` 自动发现所有子类并注册，加一条新规则只需写一个新类，不用改注册代码。

**3. LLM 后端可替换**

`LLMBackend` 抽象基类隔离了所有供应商相关的逻辑。从 OpenAI 切到 Anthropic 只需要改配置，代码一行不动。

**4. 工具定义多格式导出**

每个 `ToolDefinition` 可以序列化为 OpenAI function-calling、Anthropic tool-use、MCP Tool 三种格式（`to_openai()` / `to_anthropic()` / `to_mcp()`），一套定义服务所有场景。

**5. 无 API Key 自动降级**

没有配置 LLM API Key 时，`chat` 命令自动退化为 Rule-Only 模式——只跑分析器，不走 Agent 循环，确保核心分析功能始终可用。

**6. ToolResult 统一封装**

所有操作返回带类型的 `ToolResult`（成功/失败），从 Reader 到 CLI 全链路统一错误传播，不会出现裸异常在中间层丢失。
