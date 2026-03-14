# SDK 使用指南

除了命令行工具之外，Tachyon 也可以作为 Python 库集成到你的脚本、Notebook
或自动化流水线中。本文档介绍 SDK 的常用模式和完整 API。

---

## 安装

```bash
pip install -e ".[ai]"
```

---

## 公开接口

`tachyon` 顶层包导出了以下核心类：

```python
from tachyon import NcuProfiler, ProfilingStrategy, ToolPathResolver, TachyonConfig
```

| 类名 | 说明 |
|------|------|
| `TachyonConfig` | 配置加载器，支持 TOML 文件、环境变量、运行时覆盖三级优先级。 |
| `ToolPathResolver` | 自动发现 NVIDIA 工具路径（`ncu`、`nvdisasm`、`cuobjdump`）。 |
| `NcuProfiler` | 两阶段智能性能采集引擎。 |
| `ProfilingStrategy` | 枚举类型：`CONSERVATIVE`（保守模式）或 `RADICAL`（激进模式）。 |

---

## 常见用法

### 1. 加载并分析一份报告

```python
from tachyon.reader.ncu_reader import NcuReportReader
from tachyon.analyzers.base import AnalyzerRegistry

# 加载已有的 .ncu-rep 文件
reader = NcuReportReader()
result = reader.load("report.ncu-rep")

if not result.success:
    print(f"加载失败: {result.error.message}")
    raise SystemExit(1)

kernels = result.data

# 执行全部分析器
registry = AnalyzerRegistry()
registry.auto_register()

for kernel in kernels:
    findings = registry.run_all(kernel)
    for f in findings:
        print(f"[{f.severity.value}] {f.title}")
        print(f"  {f.description}")
        if f.action:
            print(f"  建议操作: {f.action}")
```

### 2. 对可执行文件做性能采集

```python
import asyncio
from tachyon import NcuProfiler, ToolPathResolver, TachyonConfig
from tachyon.profiler.ncu_profiler import ProfilingStrategy

config = TachyonConfig.load()
resolver = ToolPathResolver(config)
profiler = NcuProfiler(config, resolver)

# 第一阶段：快速扫描（基础指标，覆盖所有 kernel）
result = asyncio.run(
    profiler.profile_basic("./my_cuda_app", args=["--size", "1024"])
)

if result.success:
    report_path = result.data
    print(f"报告已保存至: {report_path}")
```

### 3. 采集 + 分析一体化流水线

```python
import asyncio
from tachyon import TachyonConfig
from tachyon.profiler.pipeline import run_e2e_pipeline

config = TachyonConfig.load()

result = asyncio.run(
    run_e2e_pipeline(
        executable="./matmul",
        exe_args=["--size", "2048"],
        config=config,
        top_k=3,
        verbose=True,
    )
)

if result.success:
    print(f"性能报告路径: {result.data}")
```

### 4. 对比两份报告

```python
from tachyon.reader.ncu_reader import NcuReportReader
from tachyon.diff.differ import ProfileDiffer

reader = NcuReportReader()
before = reader.load("before.ncu-rep").data
after = NcuReportReader().load("after.ncu-rep").data

differ = ProfileDiffer()
diffs = differ.diff(before, after)

for d in diffs:
    print(f"\n{d.demangled_name}:")
    for m in d.significant_changes:
        print(f"  {m.name}: {m.before:.2f} -> {m.after:.2f} ({m.delta_pct:+.1f}%)")
    for r in d.regressions:
        print(f"  性能退化: {r.name} ({r.delta_pct:+.1f}%)")

print(differ.summary(diffs))
```

### 5. 在自定义 Agent 中使用工具注册表

```python
from tachyon.reader.ncu_reader import NcuReportReader
from tachyon.analyzers.base import AnalyzerRegistry
from tachyon.tools.registry import ToolRegistry
from tachyon.tools.context import SessionContext
from tachyon.tools.data_query import register_data_query_tools
from tachyon.tools.source import register_source_tools
from tachyon.tools.analysis import register_analysis_tools

# 加载报告
reader = NcuReportReader()
kernels = reader.load("report.ncu-rep").data

# 初始化分析栈
analyzer_registry = AnalyzerRegistry()
analyzer_registry.auto_register()

session = SessionContext(
    kernels=kernels,
    registry=analyzer_registry,
)

# 注册全部 9 个工具
tool_registry = ToolRegistry()
register_data_query_tools(tool_registry, session)
register_source_tools(tool_registry, session)
register_analysis_tools(tool_registry, session)

# 导出工具定义，供你的 LLM 使用
openai_tools = [t.to_openai() for t in tool_registry.all_definitions()]
anthropic_tools = [t.to_anthropic() for t in tool_registry.all_definitions()]
mcp_tools = [t.to_mcp() for t in tool_registry.all_definitions()]

# 执行 LLM 返回的工具调用
import asyncio
result = asyncio.run(tool_registry.execute("list_kernels", {}))
print(result.data)
```

### 6. 查找 NVIDIA 工具路径

```python
from tachyon import ToolPathResolver, TachyonConfig

config = TachyonConfig.load()
resolver = ToolPathResolver(config)

# 查找 ncu 可执行文件
result = resolver.resolve("ncu")
if result.success:
    print(f"ncu 路径: {result.data}")

# 查找 nvdisasm
result = resolver.resolve("nvdisasm")
if result.success:
    print(f"nvdisasm 路径: {result.data}")
```

---

## API 参考

### TachyonConfig

```python
@dataclass
class TachyonConfig:
    llm: LLMConfig           # LLM 服务商和模型配置
    profiling: ProfilingConfig  # 性能采集策略
    output: OutputConfig      # 输出语言和格式
    tools: ToolsConfig        # NVIDIA 工具路径覆盖

    @classmethod
    def load(cls, config_path: Path | None = None) -> TachyonConfig:
        """从 ~/.tachyon/config.toml 加载配置，并叠加环境变量。"""

    def apply_cli_overrides(self, **kwargs) -> None:
        """应用命令行参数覆盖（优先级最高）。"""
```

### NcuReportReader

```python
class NcuReportReader:
    def load(self, path: Path | str) -> ToolResult[list[KernelReport]]:
        """解析 .ncu-rep 文件，返回 KernelReport 列表。"""
```

### AnalyzerRegistry

```python
class AnalyzerRegistry:
    def auto_register(self) -> None:
        """自动发现并注册所有 Analyzer 子类。"""

    def run_all(self, report: KernelReport) -> list[Finding]:
        """对一个 kernel 报告运行全部已注册的分析器。"""
```

### ToolRegistry

```python
class ToolRegistry:
    def register(self, tool: ToolDefinition) -> None:
        """注册一个工具。名称重复时抛出 ValueError。"""

    def all_definitions(self) -> list[ToolDefinition]:
        """返回所有已注册的工具定义。"""

    async def execute(self, name: str, arguments: dict) -> ToolResult:
        """按名称执行工具。"""
```

### ToolDefinition

```python
@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    handler: Callable[..., Awaitable[ToolResult]] | None

    def to_openai(self) -> dict:    # 转换为 OpenAI function calling 格式
    def to_anthropic(self) -> dict:  # 转换为 Anthropic tool_use 格式
    def to_mcp(self) -> dict:        # 转换为 MCP Tool 格式
```

### ProfileDiffer

```python
class ProfileDiffer:
    def diff(self, before: list[KernelReport], after: list[KernelReport]) -> list[KernelDiff]:
        """按 kernel_name 匹配并对比两组 kernel 报告。"""

    def summary(self, diffs: list[KernelDiff]) -> str:
        """生成简洁的文本摘要。"""
```

### ToolPathResolver

```python
@dataclass
class ToolPathResolver:
    config: TachyonConfig
    _cache: dict[str, Path]  # 会话级缓存

    def resolve(self, tool_name: str) -> ToolResult[Path]:
        """查找工具路径。优先级：配置文件 > PATH > $CUDA_HOME > 常见安装路径 > glob 搜索。"""
```

### NcuProfiler

```python
class NcuProfiler:
    def __init__(self, config: TachyonConfig, resolver: ToolPathResolver): ...

    async def profile_basic(self, executable: str, args: list[str] | None = None) -> ToolResult[Path]:
        """第一阶段：快速扫描，采集基础指标。"""

    async def profile_detailed(self, executable: str, kernels: list[str], ...) -> ToolResult[Path]:
        """第二阶段：针对指定 kernel 做深度采集，使用完整指标集。"""
```

### 核心数据模型

```python
@dataclass(frozen=True)
class KernelReport:
    kernel_name: str
    demangled_name: str
    launch: LaunchParams
    device: DeviceInfo
    metrics: dict[str, MetricValue]
    rules: list[RuleResult]

@dataclass(frozen=True)
class Finding:
    title: str
    severity: Severity          # CRITICAL, WARNING, INFO
    category: str               # compute, memory, latency, ...
    description: str
    action: str | None          # 建议的修复方案
    source: str                 # 分析器名称
    source_location: SourceLocation | None

class Severity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
```
