# Tachyon

**AI empowered CUDA kernel profiler**

Tachyon `/ˈtakēˌän/` (tachyon — a theoretical particle that travels faster than light) is a CUDA kernel performance analyzer that combines LLM-powered agents with traditional rule-based analysis. Unlike existing open-source tools, Tachyon bridges the full path from NCU metrics to CUDA source code to low-level instructions (PTX/SASS), so performance analysis doesn't stop at aggregate counters — it traces back through the instruction level all the way to your source lines. Like a tachyon traveling faster than the speed of light and reversing through time, we aim to optimize faster than your current speed-of-light, with the ability to trace back to root causes.

Fully integrated in Python. Supports end-to-end profiling (run an executable directly after `tachyon`, like ncu) as well as analysis of manually exported NCU reports. Multiple LLM agent vendors are supported.

## Features

- **Three-way mapping**: NCU metrics <-> source lines <-> SASS instructions, pinpointing the exact code behind each bottleneck.
- **Smart two-stage profiling**: quick scan finds the hottest top-K kernels, then deep dive collects detailed metrics only where it matters.
- **Rule engine + AI agent**: 7 built-in analyzers (roofline, memory, occupancy, warp stall, ...) produce structured findings; an LLM agent with 9 specialized tools supports interactive follow-up.
- **Profile diff**: compare two `.ncu-rep` files side by side, highlight regressions.
- **MCP server**: expose all analysis tools over MCP (stdio transport) for seamless integration with Claude Code, Ducc, Cursor, and custom agents.

## Quick Start

### Installation

```bash
# Rule-based analysis only (no LLM needed)
pip install -e .

# With AI agent (OpenAI / Anthropic / LiteLLM)
pip install -e ".[ai]"

# With MCP server
pip install -e ".[mcp]"

# Full development environment
pip install -e ".[dev,ai,mcp]"
```

### Basic Usage

```bash
tachyon analyze report.ncu-rep
tachyon chat report.ncu-rep --model claude-sonnet-4-20250514
tachyon profile ./my_app --strategy radical
tachyon diff before.ncu-rep after.ncu-rep
tachyon serve --mcp --report report.ncu-rep
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `tachyon analyze` | Parse `.ncu-rep`, run rule-based analyzers, output findings. |
| `tachyon chat` | Interactive AI analysis with 9 specialized tools. |
| `tachyon profile` | End-to-end: profile a CUDA executable, then analyze. |
| `tachyon diff` | Compare two reports, flag performance changes. |
| `tachyon serve` | Start MCP server for external agents. |

`tachyon <command> --help` for full options. See [CLI Reference](docs/cli-reference.md).

## Multi-Vendor LLM Switching

```toml
# ~/.tachyon/config.toml
[llm]
provider = "anthropic"          # openai / anthropic / litellm
model = "claude-sonnet-4-20250514"
api_key_env = "ANTHROPIC_API_KEY"
# base_url = "http://localhost:8000/v1"  # local models
```

```bash
# Switch via environment variables
export TACHYON_LLM_PROVIDER=litellm
export TACHYON_MODEL=qianfan/ERNIE-Bot-4

# Single-session CLI override
tachyon chat report.ncu-rep --provider openai --model gpt-4o
```

Priority: CLI > env vars > config.toml > defaults. Graceful degradation to Rule-Only mode when no LLM is available.

## Ducc Integration

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

Once Ducc starts, all 9 tools are auto-registered: list_kernels, get_kernel_metrics, get_source_hotspots, run_analysis, etc.

## SDK Usage

Tachyon also works as a Python library:

```python
from tachyon import NcuProfiler, ToolPathResolver, TachyonConfig

config = TachyonConfig.load()
resolver = ToolPathResolver(config)
profiler = NcuProfiler(config, resolver)
result = profiler.profile_basic("./my_cuda_app")
```

See [SDK Guide](docs/sdk-guide.md) for detailed examples.

## Configuration

`~/.tachyon/config.toml`, with environment variable and CLI overrides.

```toml
[llm]
provider = "anthropic"
model = "claude-sonnet-4-20250514"
temperature = 0.1

[profiling]
strategy = "conservative"

[output]
lang = "en"
format = "terminal"

[tools]
ncu_path = "/usr/local/cuda/bin/ncu"
```

See [Configuration](docs/configuration.md) for all options.

## Architecture (8 Layers)

```
CLI/Chat > Agent Loop > LLM Backends > Tools(9) > Analyzers(7) > Correlator > Reader > Models
```

Data flows bottom-up: `.ncu-rep` → Reader parses into KernelReport → Correlator builds three-way mapping → Analyzers produce Findings → CLI renders / Agent explores interactively.

See [Architecture](docs/architecture.md) for the full overview.

## Development

### Prerequisites

- Python 3.10+
- NVIDIA Nsight Compute (for profiling and report parsing)

### Tests

```bash
pip install -e ".[dev,ai]"
pytest
pytest --cov=tachyon --cov-report=term-missing
mypy src/tachyon/
ruff check src/
```

### Project Structure

```
src/tachyon/
├── cli/           Click commands (analyze, chat, profile, diff, serve)
├── agent/         Multi-turn agent loop, persona, context manager
├── llm/           LLM backend abstraction (OpenAI, Anthropic, LiteLLM)
├── tools/         9 agent-callable tools (data query, source, analysis)
├── analyzers/     7 rule-based analyzers (roofline, memory, occupancy, ...)
├── correlator/    Three-way mapping engine (metrics <-> source <-> SASS)
├── reader/        NCU .ncu-rep binary report parser
├── profiler/      Two-stage smart profiling, tool path resolver
├── models/        Core data models (KernelReport, Finding, OptTree, ...)
├── config/        TOML configuration with layered overrides
├── report/        Output renderers (terminal, markdown)
├── diff/          Profile comparison engine
├── server/        MCP server (stdio transport)
├── tree/          Optimization tree builder and pruning
├── errors/        Structured error handling (ToolResult, ErrorCode)
└── i18n/          Internationalization support
```

## Metrics

733 tests | 88% coverage | 0 lint errors | 7 analyzers | 9 tools | 3 LLM backends

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | 8-layer architecture, data flow, module responsibilities, design decisions |
| [CLI Reference](docs/cli-reference.md) | Full options and examples for all 5 commands |
| [Configuration](docs/configuration.md) | config.toml settings, env vars, priority chain |
| [SDK Guide](docs/sdk-guide.md) | Python SDK patterns, common usage, API reference |
| [Multi-Vendor LLM & Ducc Integration](docs/multi-vendor-integration.md) | LLM switching, Ducc MCP integration, custom providers |

## License

MIT
