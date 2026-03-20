"""Configuration management with layered overrides.

Priority (highest to lowest):
1. CLI arguments (--model, --lang, etc.)
2. Environment variables (TACHYON_LLM_PROVIDER, etc.)
3. Config file (~/.tachyon/config.toml)
4. Built-in defaults
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# tomllib is stdlib in 3.11+, fallback to tomli for 3.10
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


DEFAULT_CONFIG_PATH = Path.home() / ".tachyon" / "config.toml"


@dataclass
class LLMConfig:
    """LLM backend configuration."""
    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: str | None = None        # direct key (highest priority)
    api_key_env: str = "OPENAI_API_KEY"  # env var name to read key from
    base_url: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.1
    timeout: int = 600                # agent total timeout in seconds (10 min)
    move_timeout: int = 120           # per-move timeout (LLM call + tool exec)


@dataclass
class ProfilingConfig:
    """Profiling strategy configuration."""
    strategy: str = "conservative"


@dataclass
class OutputConfig:
    """Output format and language configuration."""
    lang: str = ""  # empty = auto-detect from locale; "en" or "zh" to force
    format: str = "terminal"


@dataclass
class ToolsConfig:
    """External tool path configuration. None = auto-discover."""
    ncu_path: str | None = None
    nvdisasm_path: str | None = None
    cuobjdump_path: str | None = None
    ncu_report_path: str | None = None


@dataclass
class TachyonConfig:
    """Top-level configuration with layered overrides."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)

    @classmethod
    def load(cls, config_path: Path | None = None) -> TachyonConfig:
        """Load config from TOML file, then overlay environment variables."""
        cfg = cls()
        path = config_path or DEFAULT_CONFIG_PATH
        toml_loaded = False
        if path.exists():
            with open(path, "rb") as f:
                data = tomllib.load(f)
            cfg._apply_toml(data)
            toml_loaded = "llm" in data  # TOML has explicit LLM config
        cfg._apply_env(toml_has_llm=toml_loaded)
        return cfg

    def _apply_toml(self, data: dict[str, Any]) -> None:
        """Apply TOML config values to dataclass fields."""
        section_map = {
            "llm": self.llm,
            "profiling": self.profiling,
            "output": self.output,
            "tools": self.tools,
        }
        for section_name, section_obj in section_map.items():
            if section_name in data:
                for k, v in data[section_name].items():
                    if hasattr(section_obj, k):
                        setattr(section_obj, k, v)

    def _apply_env(self, *, toml_has_llm: bool = False) -> None:
        """Apply environment variable overrides.

        Auto-detection order for LLM backend (only when TOML doesn't specify [llm]):
        1. Explicit TACHYON_* env vars (highest priority)
        2. OPENAI_API_KEY → OpenAI provider
        3. ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL → Anthropic/ducc provider
        4. ANTHROPIC_API_KEY → Anthropic provider
        5. Built-in defaults (lowest priority)
        """
        env_map = {
            "TACHYON_LLM_PROVIDER": ("llm", "provider"),
            "TACHYON_MODEL": ("llm", "model"),
            "TACHYON_API_KEY": ("llm", "api_key"),
            "TACHYON_API_KEY_ENV": ("llm", "api_key_env"),
            "TACHYON_BASE_URL": ("llm", "base_url"),
            "TACHYON_LANG": ("output", "lang"),
            "TACHYON_STRATEGY": ("profiling", "strategy"),
            "TACHYON_NCU_REPORT_PATH": ("tools", "ncu_report_path"),
        }
        for env_var, (section, attr) in env_map.items():
            if v := os.environ.get(env_var):
                val: Any = v
                # int coercion for numeric fields
                if attr == "timeout":
                    val = int(v)
                setattr(getattr(self, section), attr, val)

        # TACHYON_TIMEOUT needs special int handling
        if timeout_str := os.environ.get("TACHYON_TIMEOUT"):
            try:
                self.llm.timeout = int(timeout_str)
            except ValueError:
                pass

        # Auto-detect LLM backend if no explicit config overrides it
        has_explicit = (
            os.environ.get("TACHYON_LLM_PROVIDER")
            or os.environ.get("TACHYON_API_KEY")
            or os.environ.get("TACHYON_API_KEY_ENV")
            or os.environ.get("TACHYON_BASE_URL")
            or toml_has_llm
        )
        if not has_explicit:
            self._auto_detect_llm()

    def apply_cli_overrides(self, **kwargs: Any) -> None:
        """Apply CLI argument overrides (highest priority)."""
        cli_map = {
            "model": ("llm", "model"),
            "lang": ("output", "lang"),
            "format": ("output", "format"),
            "strategy": ("profiling", "strategy"),
        }
        for key, (section, attr) in cli_map.items():
            if val := kwargs.get(key):
                setattr(getattr(self, section), attr, val)

    def _auto_detect_llm(self) -> None:
        """Auto-detect LLM backend from environment.

        Detection priority:
        1. OPENAI_API_KEY set → use OpenAI (default)
        2. ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL set → ducc environment
           (Baidu's Claude Code fork using OneAPI gateway)
        3. ANTHROPIC_API_KEY set → standard Anthropic
        4. No key found → keep defaults (will fail gracefully at runtime)
        """
        # 1. OpenAI key present → use defaults (already openai)
        if os.environ.get("OPENAI_API_KEY"):
            return

        # 2. Ducc environment (ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL)
        auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        if auth_token and base_url:
            self.llm.provider = "anthropic"
            self.llm.api_key_env = "ANTHROPIC_AUTH_TOKEN"
            self.llm.base_url = base_url
            # Use model from env if available, otherwise default to a good model
            model = os.environ.get("ANTHROPIC_MODEL")
            if model:
                self.llm.model = model
            else:
                self.llm.model = "claude-sonnet-4-20250514"
            return

        # 3. Standard Anthropic API key
        if os.environ.get("ANTHROPIC_API_KEY"):
            self.llm.provider = "anthropic"
            self.llm.api_key_env = "ANTHROPIC_API_KEY"
            self.llm.model = "claude-sonnet-4-20250514"
            return
