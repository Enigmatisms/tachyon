"""Evolve context — wraps SessionContext with evolve-specific state."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..tools.context import SessionContext

if TYPE_CHECKING:
    from ..profiler.ncu_profiler import NcuProfiler
    from ..reader.ncu_reader import NcuReportReader

from .config import EvolveConfig
from .git import GitRollback
from .session import EvolveSession

_log = logging.getLogger(__name__)


class EvolveContext:
    """Unified context for evolve tool execution."""

    def __init__(
        self,
        base: SessionContext,
        evolve: EvolveSession,
        git: GitRollback,
        config: EvolveConfig,
        *,
        executable: str | None = None,
        exe_args: list[str] | None = None,
        profiler: NcuProfiler | None = None,
        reader: NcuReportReader | None = None,
    ) -> None:
        self.base = base
        self.evolve = evolve
        self.git = git
        self.config = config
        self.executable = executable
        self.exe_args = exe_args or []
        self.profiler = profiler
        self.reader = reader
        self.edit_locked = False  # True once benchmark/reprofile runs
        self.benchmark_fix_allowed = 1  # Allow 1 edit fix after benchmark failure
        self.turn_count = 0      # Incremented by orchestrator per tool call
        self.max_turns = 15      # Set by orchestrator
        self.compile_fail_count = 0  # Consecutive compile failures in iteration
        self.run_fail_count = 0      # Consecutive run failures in iteration
        self.edit_fail_count = 0     # Consecutive edit match failures in iteration
        self.iteration_doomed = False  # Set True when iteration is unrecoverable

    @property
    def allowed_source_paths(self) -> set[str]:
        """Union of analysis whitelist and evolve edit paths."""
        paths = set(self.base.allowed_source_paths)
        if self.config.allowed_edit_paths:
            repo_root = str(self.git.repo_root.resolve())
            for p in self.config.allowed_edit_paths:
                if ".." in Path(p).parts:
                    _log.warning(
                        "Skipping allowed_edit_path with '..' component: %s",
                        p,
                    )
                    continue
                resolved = str(Path(p).resolve())
                if not resolved.startswith(repo_root):
                    _log.warning(
                        "Skipping allowed_edit_path outside git repo: %s",
                        p,
                    )
                    continue
                paths.add(resolved)
        return paths

    @property
    def kernels(self) -> list:
        return self.base.kernels

    @property
    def kernel_count(self) -> int:
        return self.base.kernel_count
