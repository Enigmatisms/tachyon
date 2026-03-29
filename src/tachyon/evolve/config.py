"""Evolve configuration — build/run settings and edit path whitelist.

Priority: CLI args > config file > defaults.
Config file: project root `.tachyon/evolve.toml` or `--config` parameter.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields as _dataclass_fields
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

_TACHYON_DIR = ".tachyon"
_CONFIG_FILE = "evolve.toml"


@dataclass
class EvolveConfig:
    """Build/run configuration for evolve iterations."""

    build_cmd: str | None = None
    run_cmd: str | None = None
    build_timeout: int = 300
    run_timeout: int = 120
    max_iterations: int = 10
    target_kernel: str | None = None
    git_auto_commit: bool = True
    git_auto_rollback: bool = True
    allowed_edit_paths: list[str] = field(default_factory=list)
    reprofile_ncu_set: str = "basic"
    deep: bool = False

    @classmethod
    def load(
        cls,
        config_path: Path | None = None,
        **overrides: Any,
    ) -> EvolveConfig:
        """Load from evolve.toml, then apply CLI overrides.

        Searches for config in this order:
          1. Explicit ``config_path`` (``--config``)
          2. ``.tachyon/evolve.toml`` in cwd
          3. ``.tachyon/evolve.toml`` in git root (if in a git repo)
        """
        cfg = cls()

        resolved = _find_config(config_path)
        if resolved is not None:
            cfg = cls._from_toml(resolved)

        # Apply overrides (CLI args take priority)
        _valid_fields = {f.name for f in _dataclass_fields(cls)}
        for key, value in overrides.items():
            if key in _valid_fields and value is not None:
                setattr(cfg, key, value)

        return cfg

    @classmethod
    def _from_toml(cls, path: Path) -> EvolveConfig:
        """Parse a TOML config file."""
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]

        text = path.read_text(encoding="utf-8")
        data = tomllib.loads(text)

        # First try top-level, then fall back to common sections
        _sections = [data, data.get("build", {}), data.get("run", {}),
                      data.get("reprofile", {})]

        def _find(key_path: tuple[str, ...]) -> Any:
            for section in _sections:
                val: Any = section
                found = True
                for k in key_path:
                    if isinstance(val, dict):
                        val = val.get(k)
                    else:
                        found = False
                        break
                if found and val is not None:
                    return val
            return None

        kwargs: dict[str, Any] = {}
        field_map = {
            "build_cmd": ("build", "cmd"),
            "run_cmd": ("run", "cmd"),
            "build_timeout": ("build", "timeout"),
            "run_timeout": ("run", "timeout"),
            "max_iterations": ("max_iterations",),
            "target_kernel": ("target_kernel",),
            "git_auto_commit": ("git", "auto_commit"),
            "git_auto_rollback": ("git", "auto_rollback"),
            "allowed_edit_paths": ("allowed_edit_paths",),
            "reprofile_ncu_set": ("ncu_set",),
            "deep": ("deep",),
        }

        for attr, keys in field_map.items():
            val = _find(keys)
            if val is not None:
                kwargs[attr] = val

        _log.debug("Loaded evolve config from %s: %s", path, kwargs)
        return cls(**kwargs)


def _find_config(explicit_path: Path | None = None) -> Path | None:
    """Locate the evolve.toml config file."""
    if explicit_path is not None:
        if explicit_path.is_file():
            return explicit_path
        _log.warning("Config path %s does not exist, using defaults.", explicit_path)
        return None

    # Check cwd
    candidate = Path.cwd() / _TACHYON_DIR / _CONFIG_FILE
    if candidate.is_file():
        return candidate

    # Check git root
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_root = Path(result.stdout.strip())
            candidate = git_root / _TACHYON_DIR / _CONFIG_FILE
            if candidate.is_file():
                return candidate
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None
