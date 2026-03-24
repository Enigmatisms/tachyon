# Tachyon 日志体系使用指南

Tachyon 使用 **structlog + Python stdlib logging** 的双层架构，支持结构化 JSON 输出和传统文本输出。

**两套系统：**

| 系统 | 用途 | 输出位置 | 控制变量 |
|------|------|---------|---------|
| Console Logging | 常规应用日志（启动、警告、错误） | stderr | `TACHYON_LOG_LEVEL` |
| AgentLogger | Agent-LLM 交互调试（prompt/tool 记录） | `~/.tachyon/debug_logs/` | `TACHYON_DEBUG_RECORD` |

## 环境变量速查

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TACHYON_LOG_LEVEL` | `WARNING` | 控制台日志级别：`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `TACHYON_LOG_FORMAT` | `text` | 控制台输出格式：`text`（人类可读）、`json`（结构化 JSON） |
| `TACHYON_DEBUG_RECORD` | (未设置=关闭) | Agent 调试日志级别：`none`, `prompt`, `all` |

## 使用场景

### Case 1: 正常使用（零配置）

```bash
tachyon chat report.ncu-rep
```

默认行为：
- 控制台仅显示 `WARNING` 及以上级别
- 无调试日志文件
- **零额外开销**（一个 int 比较即返回）

### Case 2: 调试 Agent 行为（查看发给 LLM 的 prompt）

```bash
TACHYON_DEBUG_RECORD=prompt tachyon chat report.ncu-rep
```

行为：
- 在 `~/.tachyon/debug_logs/agent_YYYYMMDD_HHMMSS.log` 写入 prompt 记录
- 每条 prompt 包含：turn 数、message 数、tools 数
- structlog 可用时输出 JSONL 格式，否则降级为文本格式

JSONL 输出示例：
```json
{"event":"PROMPT","turn":1,"messages":3,"tools":12,"timestamp":"2026-03-23T10:30:00"}
{"event":"MESSAGE","role":"SYSTEM","content_chars":1234}
{"event":"MESSAGE","role":"USER","content_chars":56}
```

### Case 3: 完整调试（prompt + tool 调用记录）

```bash
TACHYON_DEBUG_RECORD=all tachyon chat report.ncu-rep
```

额外记录所有 tool 调用：
```json
{"event":"TOOL_CALL","name":"run_analysis","elapsed":0.532,"success":true,"args_keys":["kernel_id"],"result_chars":2341}
```

### Case 4: 开发调试（详细控制台输出）

```bash
TACHYON_LOG_LEVEL=DEBUG tachyon chat report.ncu-rep
```

控制台显示所有日志，包括：
- httpx 请求（被抑制到 WARNING 级别，需额外设置）
- Agent turn 计时
- Context compression 事件
- 工具注册详情

### Case 5: JSON 结构化输出（日志采集）

```bash
TACHYON_LOG_LEVEL=INFO TACHYON_LOG_FORMAT=json tachyon profile ./app
```

所有控制台日志输出为 JSON，每行一个 JSON object：
```json
{"event":"Starting profiling pipeline","logger":"tachyon.profiler.pipeline","level":"info","timestamp":"2026-03-23T10:30:00"}
```

适用场景：ELK/Grafana Loki 等日志采集系统。

### Case 6: 完整调试组合

```bash
TACHYON_LOG_LEVEL=DEBUG \
TACHYON_LOG_FORMAT=json \
TACHYON_DEBUG_RECORD=all \
tachyon chat report.ncu-rep
```

同时启用详细控制台 + JSON 格式 + 完整 Agent 交互记录。

## 开发者指南

### 在模块中添加日志

**方式 1: stdlib logging（推荐，现有 28 个模块均用此方式）**

```python
import logging

_log = logging.getLogger(__name__)

# 使用
_log.info("Loading %d kernels", len(kernels))
_log.warning("SourceCorrelator NOT created: %s", reason)
_log.debug("Agent at turn %d/%d", turn, max_turns)
```

无需任何改动即可享受 structlog 增强（颜色输出、JSON 格式等）。

**方式 2: AgentLogger（仅用于 Agent-LLM 交互记录）**

```python
from tachyon.utils.log import agent_logger

# 在 CLI 入口调用一次
agent_logger.init()

# 在 agent loop 中
agent_logger.record_prompt(messages, tools=tool_defs, turn=1)
agent_logger.record_tool("run_analysis", args, result_str, elapsed, success)
```

**不推荐的方式：** 直接 `print()`（无法被日志级别控制、无法采集）。

### 模块日志级别控制

```python
import logging
logging.getLogger("tachyon.profiler.ncu_profiler").setLevel(logging.DEBUG)
```

已内置的第三方抑制：
```python
logging.getLogger("httpx").setLevel(logging.WARNING)   # 始终抑制
logging.getLogger("httpcore").setLevel(logging.WARNING)  # 始终抑制
```

### 关闭时的零开销保证

`TACHYON_DEBUG_RECORD` 未设置时，`AgentLogger` 的关键路径：

```python
def record_prompt(self, messages, tools=None, turn=0):
    if self._level < _L.PROMPT or self._fh is None:  # ← 一个 int 比较 + None 检查
        return  # ← 立即返回，零分配
```

`TACHYON_LOG_LEVEL=WARNING` 时，`logging.getLogger().debug()` 内部同样快速返回（stdlib 的 level check）。

## 文件结构

```
src/tachyon/utils/
├── log.py              # 统一日志模块（configure_logging + AgentLogger）
├── debug_record.py     # 向后兼容 shim（15 行，全部委托给 log.py）
└── __init__.py
```

## 向后兼容

旧的 import 方式仍然可用：

```python
from tachyon.utils.debug_record import init, record_prompt, record_tool, close
# ↑ 这些函数现在委托给 log.py 的 agent_logger 单例
```

旧的 `record_response()` 函数已从新实现中移除（原本就是死代码，从未被调用）。
