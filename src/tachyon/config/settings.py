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
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.1


@dataclass
class ProfilingConfig:
    """Profiling strategy configuration."""
    strategy: str = "conservative"


@dataclass
class OutputConfig:
    """Output format and language configuration."""
    lang: str = "en"
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
        if path.exists():
            with open(path, "rb") as f:
                data = tomllib.load(f)
            cfg._apply_toml(data)
        cfg._apply_env()
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

    def _apply_env(self) -> None:
        """Apply environment variable overrides."""
        env_map = {
            "TACHYON_LLM_PROVIDER": ("llm", "provider"),
            "TACHYON_MODEL": ("llm", "model"),
            "TACHYON_LANG": ("output", "lang"),
            "TACHYON_STRATEGY": ("profiling", "strategy"),
            "TACHYON_NCU_REPORT_PATH": ("tools", "ncu_report_path"),
        }
        for env_var, (section, attr) in env_map.items():
            if v := os.environ.get(env_var):
                setattr(getattr(self, section), attr, v)

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
