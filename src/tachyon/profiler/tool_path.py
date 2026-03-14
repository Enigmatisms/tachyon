"""Universal NVIDIA tool path detection.

Resolves paths for ncu, nvdisasm, cuobjdump with configurable priority.
Detects once per session, caches results. Never hardcodes paths.

Priority:
  1. Config file explicit path (highest — user overrides everything)
  2. shutil.which() — already in PATH
  3. $CUDA_HOME/bin/
  4. /usr/local/cuda/bin/
  5. Glob patterns for multi-version installs (lowest)

References:
  - Architecture: section 10.2 (ToolPathResolver design)
  - Implementation: section 4.1
"""
from __future__ import annotations

import glob as globmod
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from tachyon.config.settings import TachyonConfig
from tachyon.errors.handler import ErrorCode, ToolResult

logger = logging.getLogger(__name__)


# Known NVIDIA tools and their default binary names.
TOOL_NAMES: dict[str, str] = {
    "ncu": "ncu",
    "nvdisasm": "nvdisasm",
    "cuobjdump": "cuobjdump",
}

# Glob patterns for multi-version installations (Linux).
GLOB_SEARCH_PATTERNS: dict[str, list[str]] = {
    "ncu": ["/opt/nvidia/nsight-compute/*/ncu"],
    "nvdisasm": ["/usr/local/cuda-*/bin/nvdisasm"],
    "cuobjdump": ["/usr/local/cuda-*/bin/cuobjdump"],
}


@dataclass
class ToolPathResolver:
    """Resolves NVIDIA tool paths with multi-source detection and caching.

    Usage::

        resolver = ToolPathResolver(config)
        ncu = resolver.resolve("ncu")          # "/usr/local/cuda/bin/ncu"
        nvdisasm = resolver.resolve("nvdisasm")
    """

    config: TachyonConfig
    _cache: dict[str, str | None] = field(default_factory=dict, init=False, repr=False)

    def resolve(self, tool_name: str) -> str:
        """Resolve tool path, raising on failure.

        Args:
            tool_name: One of ``"ncu"``, ``"nvdisasm"``, ``"cuobjdump"``.

        Returns:
            Absolute path to the tool binary.

        Raises:
            FileNotFoundError: If tool cannot be found after all search methods.
        """
        if tool_name in self._cache:
            cached = self._cache[tool_name]
            if cached is not None:
                return cached
            raise FileNotFoundError(
                f"{tool_name} not found. Install CUDA Toolkit or set "
                f"[tools] {tool_name}_path in ~/.tachyon/config.toml"
            )

        path = self._detect(tool_name)
        self._cache[tool_name] = path

        if path is None:
            raise FileNotFoundError(
                f"{tool_name} not found. Searched: config.toml [tools], PATH, "
                f"$CUDA_HOME/bin, /usr/local/cuda/bin, /opt/nvidia/*/. "
                f"Install CUDA Toolkit or set [tools] {tool_name}_path in config."
            )

        logger.info("Resolved %s -> %s", tool_name, path)
        return path

    def resolve_safe(self, tool_name: str) -> ToolResult[str]:
        """Non-throwing variant that returns ``ToolResult``."""
        try:
            path = self.resolve(tool_name)
            return ToolResult.ok(path)
        except FileNotFoundError as e:
            return ToolResult.fail(
                ErrorCode.TOOL_NOT_FOUND,
                str(e),
                suggestion=(
                    f"Install CUDA Toolkit or configure [tools] "
                    f"{tool_name}_path in ~/.tachyon/config.toml"
                ),
            )

    def _detect(self, tool_name: str) -> str | None:
        """Multi-source detection with priority ordering."""
        binary = TOOL_NAMES.get(tool_name, tool_name)

        # --- Priority 1: User explicit config (highest) ---
        config_path = self._get_config_path(tool_name)
        if config_path:
            p = Path(config_path)
            if p.is_file() and os.access(p, os.X_OK):
                return str(p.resolve())
            logger.warning(
                "Config [tools] %s_path = %s is not a valid executable, "
                "falling through to auto-detection",
                tool_name,
                config_path,
            )

        # --- Priority 2: shutil.which (system PATH) ---
        which_path = shutil.which(binary)
        if which_path:
            return str(Path(which_path).resolve())

        # --- Priority 3: $CUDA_HOME/bin ---
        cuda_home = os.environ.get("CUDA_HOME")
        if cuda_home:
            candidate = Path(cuda_home) / "bin" / binary
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate.resolve())

        # --- Priority 4: /usr/local/cuda/bin (default install) ---
        default_cuda = Path("/usr/local/cuda/bin") / binary
        if default_cuda.is_file() and os.access(default_cuda, os.X_OK):
            return str(default_cuda.resolve())

        # --- Priority 5: Glob multi-version search ---
        patterns = GLOB_SEARCH_PATTERNS.get(tool_name, [])
        for pattern in patterns:
            matches = sorted(globmod.glob(pattern), reverse=True)  # newest first
            for match in matches:
                p = Path(match)
                if p.is_file() and os.access(p, os.X_OK):
                    return str(p.resolve())

        return None

    def _get_config_path(self, tool_name: str) -> str | None:
        """Read user-configured path from TachyonConfig.tools."""
        attr = f"{tool_name}_path"
        return getattr(self.config.tools, attr, None)
