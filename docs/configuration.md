# 配置说明

Tachyon 采用分层配置机制，多个来源的配置按以下优先级合并（越靠前优先级越高）：

1. **命令行参数**（`--model`、`--format`、`--strategy` 等）
2. **环境变量**（`TACHYON_MODEL`、`TACHYON_LLM_PROVIDER` 等）
3. **配置文件**（`~/.tachyon/config.toml`）
4. **内置默认值**

---

## 配置文件

默认路径为 `~/.tachyon/config.toml`，可以手动创建，也可以不建——Tachyon 会使用内置默认值。

### 完整示例

```toml
[llm]
provider = "anthropic"
model = "claude-sonnet-4-20250514"
api_key_env = "ANTHROPIC_API_KEY"
base_url = ""
max_tokens = 4096
temperature = 0.1

[profiling]
strategy = "conservative"

[output]
lang = "en"
format = "terminal"

[tools]
ncu_path = "/usr/local/cuda/bin/ncu"
nvdisasm_path = "/usr/local/cuda/bin/nvdisasm"
cuobjdump_path = "/usr/local/cuda/bin/cuobjdump"
ncu_report_path = "/opt/nvidia/nsight-compute/2024.3.2/extras/python"
```

---

## 各配置段说明

### [llm]

控制 `tachyon chat` 以及 AI 增强分析所使用的大模型后端。

| 键 | 类型 | 默认值 | 说明 |
|-----|------|---------|------|
| `provider` | string | `"openai"` | 模型提供方：`openai`、`anthropic` 或 `litellm`。 |
| `model` | string | `"gpt-4o"` | 模型名称，如 `gpt-4o`、`claude-sonnet-4-20250514`，或任意 LiteLLM 支持的模型标识。 |
| `api_key_env` | string | `"OPENAI_API_KEY"` | 存放 API Key 的环境变量名。 |
| `base_url` | string | `null` | 自定义 API 地址，适用于代理、Azure 或本地端点。 |
| `max_tokens` | int | `4096` | 单次调用返回的最大 token 数。 |
| `temperature` | float | `0.1` | 采样温度，值越低输出越确定。 |

### [profiling]

控制 `tachyon profile` 命令的采集行为。

| 键 | 类型 | 默认值 | 说明 |
|-----|------|---------|------|
| `strategy` | string | `"conservative"` | 采集力度。`conservative` = 基础 + 详细指标；`radical` = 全量指标 + 源码级计数器。 |

**两种策略的具体区别：**

| 策略 | 第一阶段 | 第二阶段 | 适用场景 |
|------|----------|----------|----------|
| `conservative` | 基础指标，超时 600s | 详细指标，超时 1200s | 日常分析，快速出结果。 |
| `radical` | 基础指标，超时 600s | 全量指标 + 源码计数器，超时 1800s | 深度排查，需要最详尽的数据。 |

### [output]

控制输出的格式和语言。

| 键 | 类型 | 默认值 | 说明 |
|-----|------|---------|------|
| `lang` | string | `"en"` | 输出语言：`en`（英文）或 `zh`（中文）。 |
| `format` | string | `"terminal"` | 默认输出格式：`terminal`、`markdown` 或 `json`。 |

### [tools]

手动指定 NVIDIA 工具路径。当自动发现失败或需要使用特定版本时，可在此配置。

| 键 | 类型 | 默认值 | 说明 |
|-----|------|---------|------|
| `ncu_path` | string | `null`（自动） | `ncu` 可执行文件的绝对路径。 |
| `nvdisasm_path` | string | `null`（自动） | `nvdisasm` 可执行文件的绝对路径。 |
| `cuobjdump_path` | string | `null`（自动） | `cuobjdump` 可执行文件的绝对路径。 |
| `ncu_report_path` | string | `null`（自动） | 包含 `ncu_report.py`（NCU Python 绑定）的目录路径。 |

路径留空或省略时，Tachyon 通过 `ToolPathResolver` 按以下顺序自动查找：

1. 配置文件 `[tools]` 段（即本表）
2. `shutil.which()` —— 已在 `PATH` 中的工具
3. `$CUDA_HOME/bin/`
4. `/usr/local/cuda/bin/`
5. 通配符匹配多版本安装（如 `/opt/nvidia/nsight-compute/*/ncu`）

---

## 环境变量

环境变量的优先级介于配置文件和命令行参数之间。

| 变量 | 对应配置项 | 说明 |
|------|-----------|------|
| `TACHYON_LLM_PROVIDER` | `[llm].provider` | 模型提供方。 |
| `TACHYON_MODEL` | `[llm].model` | 模型名称。 |
| `TACHYON_LANG` | `[output].lang` | 输出语言。 |
| `TACHYON_STRATEGY` | `[profiling].strategy` | 采集策略。 |
| `TACHYON_NCU_REPORT_PATH` | `[tools].ncu_report_path` | NCU Python 绑定目录路径。 |

此外，LLM 的 API Key 从 `[llm].api_key_env` 指定的环境变量中读取（默认为 `OPENAI_API_KEY`）。

### 各提供方的 API Key 变量

| 提供方 | 默认 `api_key_env` | 需要设置的变量 |
|--------|---------------------|----------------|
| OpenAI | `OPENAI_API_KEY` | `export OPENAI_API_KEY=sk-...` |
| Anthropic | `ANTHROPIC_API_KEY` | `export ANTHROPIC_API_KEY=sk-ant-...` |
| LiteLLM | 取决于底层模型 | 设置对应提供方所需的 Key 即可。 |

---

## 命令行参数覆盖

命令行参数拥有最高优先级，会覆盖配置文件和环境变量中的同名配置：

| 命令行参数 | 覆盖的配置项 |
|-----------|-------------|
| `--model` / `-m` | `[llm].model` / `TACHYON_MODEL` |
| `--provider` / `-p` | `[llm].provider` / `TACHYON_LLM_PROVIDER` |
| `--format` | `[output].format` |
| `--strategy` | `[profiling].strategy` / `TACHYON_STRATEGY` |
| `--lang` | `[output].lang` / `TACHYON_LANG` |

---

## 快速上手示例

### OpenAI（默认配置）

```bash
# 只需设置 API Key，无需配置文件，默认使用 gpt-4o
export OPENAI_API_KEY=sk-...
tachyon chat report.ncu-rep
```

### Anthropic

```toml
# ~/.tachyon/config.toml
[llm]
provider = "anthropic"
model = "claude-sonnet-4-20250514"
api_key_env = "ANTHROPIC_API_KEY"
```

```bash
# 设置 Key 后直接使用
export ANTHROPIC_API_KEY=sk-ant-...
tachyon chat report.ncu-rep
```

### 纯规则分析（不调用大模型）

```bash
# 不需要 API Key，也不需要配置文件
tachyon analyze report.ncu-rep --no-ai
```

### 自定义 CUDA 安装路径

```toml
# ~/.tachyon/config.toml
# 当 CUDA 安装在非标准位置时，手动指定工具路径
[tools]
ncu_path = "/opt/cuda-12.4/bin/ncu"
nvdisasm_path = "/opt/cuda-12.4/bin/nvdisasm"
cuobjdump_path = "/opt/cuda-12.4/bin/cuobjdump"
ncu_report_path = "/opt/nvidia/nsight-compute/2024.3.2/extras/python"
```
