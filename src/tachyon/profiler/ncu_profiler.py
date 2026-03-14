"""Two-stage Smart Profiling engine.

Stage 1 (Quick Scan): Basic metrics for all kernels -> identify top-K by duration.
Stage 2 (Deep Dive): Targeted detailed/full metrics for top-K kernels only.

Security: subprocess with list args (no shell), timeout, stderr capture.

References:
  - Architecture: section 10.1 (profiling strategy modes)
  - Architecture: section 10.2 (NcuProfiler design + two-stage flow)
  - Implementation: section 4.2
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from tachyon.config.settings import TachyonConfig
from tachyon.errors.handler import ErrorCode, ToolResult
from tachyon.profiler.tool_path import ToolPathResolver

logger = logging.getLogger(__name__)


class ProfilingStrategy(str, Enum):
    """Profiling aggressiveness level."""

    CONSERVATIVE = "conservative"
    RADICAL = "radical"


@dataclass
class StageConfig:
    """Configuration for one profiling stage."""

    metric_set: str  # "basic", "detailed", "full"
    timeout_sec: int = 600
    extra_sections: list[str] = field(default_factory=list)
    source_counters: bool = False


# Strategy -> (Stage1, Stage2) configurations.
STRATEGY_CONFIGS: dict[ProfilingStrategy, tuple[StageConfig, StageConfig]] = {
    ProfilingStrategy.CONSERVATIVE: (
        StageConfig(metric_set="basic", timeout_sec=600),
        StageConfig(metric_set="detailed", timeout_sec=1800),
    ),
    ProfilingStrategy.RADICAL: (
        StageConfig(metric_set="detailed", timeout_sec=1200),
        StageConfig(
            metric_set="full",
            timeout_sec=3600,
            source_counters=True,
        ),
    ),
}


@dataclass
class ProfilingResult:
    """Result of a profiling stage."""

    ncu_rep_path: Path
    stage: int  # 1 or 2
    returncode: int
    stderr: str
    elapsed_sec: float


class NcuProfiler:
    """Two-stage smart profiling engine.

    Usage::

        profiler = NcuProfiler(config, resolver)
        stage1 = profiler.profile_basic("./app", ["--size", "1024"])
        # ... parse stage1 report, identify top-K kernels ...
        stage2 = profiler.profile_targeted(
            "./app", ["--size", "1024"],
            kernels=["matmul_kernel", "reduce_kernel"],
        )
    """

    def __init__(self, config: TachyonConfig, resolver: ToolPathResolver) -> None:
        self._config = config
        self._resolver = resolver
        self._ncu_path: str | None = None

    @property
    def ncu_path(self) -> str:
        """Lazy-resolved ncu path via ToolPathResolver."""
        if self._ncu_path is None:
            self._ncu_path = self._resolver.resolve("ncu")
        return self._ncu_path

    def profile_basic(
        self,
        executable: str,
        args: list[str] | None = None,
        *,
        strategy: ProfilingStrategy | None = None,
        output_dir: Path | None = None,
        extra_ncu_args: list[str] | None = None,
    ) -> ToolResult[ProfilingResult]:
        """Stage 1: Quick Scan -- profile all kernels with basic/detailed metrics.

        Returns .ncu-rep path for downstream parsing by NcuReportReader.
        """
        strat = strategy or ProfilingStrategy(self._config.profiling.strategy)
        stage1_cfg, _ = STRATEGY_CONFIGS[strat]

        out_dir = output_dir or Path(tempfile.mkdtemp(prefix="tachyon_"))
        out_file = out_dir / "stage1.ncu-rep"

        cmd = self._build_command(
            metric_set=stage1_cfg.metric_set,
            output_path=out_file,
            executable=executable,
            exe_args=args or [],
            source_counters=stage1_cfg.source_counters,
            extra_ncu_args=extra_ncu_args,
        )

        return self._run_ncu(cmd, stage=1, output_path=out_file, timeout=stage1_cfg.timeout_sec)

    def profile_targeted(
        self,
        executable: str,
        args: list[str] | None = None,
        *,
        kernels: list[str],
        strategy: ProfilingStrategy | None = None,
        output_dir: Path | None = None,
        extra_ncu_args: list[str] | None = None,
    ) -> ToolResult[ProfilingResult]:
        """Stage 2: Deep Dive -- targeted metrics for top-K kernels only.

        Args:
            kernels: Kernel name regex patterns to profile (``--kernel-name``).
            strategy: Overrides config default.
        """
        strat = strategy or ProfilingStrategy(self._config.profiling.strategy)
        _, stage2_cfg = STRATEGY_CONFIGS[strat]

        out_dir = output_dir or Path(tempfile.mkdtemp(prefix="tachyon_"))
        out_file = out_dir / "stage2.ncu-rep"

        kernel_filter_args: list[str] = []
        for k in kernels:
            kernel_filter_args.extend(["--kernel-name", k])

        cmd = self._build_command(
            metric_set=stage2_cfg.metric_set,
            output_path=out_file,
            executable=executable,
            exe_args=args or [],
            source_counters=stage2_cfg.source_counters,
            kernel_filter_args=kernel_filter_args,
            extra_ncu_args=extra_ncu_args,
        )

        return self._run_ncu(cmd, stage=2, output_path=out_file, timeout=stage2_cfg.timeout_sec)

    def _build_command(
        self,
        *,
        metric_set: str,
        output_path: Path,
        executable: str,
        exe_args: list[str],
        source_counters: bool = False,
        kernel_filter_args: list[str] | None = None,
        extra_ncu_args: list[str] | None = None,
    ) -> list[str]:
        """Build ncu command as a list (never shell=True)."""
        cmd = [
            self.ncu_path,
            "--set",
            metric_set,
            "--output",
            str(output_path),
            "--force-overwrite",
            "--target-processes",
            "all",
        ]

        if source_counters:
            cmd.extend(["--section", "SourceCounters"])

        if kernel_filter_args:
            cmd.extend(kernel_filter_args)

        if extra_ncu_args:
            cmd.extend(extra_ncu_args)

        # Executable and its arguments come last.
        cmd.append(executable)
        cmd.extend(exe_args)
        return cmd

    def _run_ncu(
        self,
        cmd: list[str],
        stage: int,
        output_path: Path,
        timeout: int,
    ) -> ToolResult[ProfilingResult]:
        """Execute ncu subprocess with safety constraints."""
        logger.info("Stage %d: running %s", stage, " ".join(cmd))
        start = time.monotonic()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult.fail(
                ErrorCode.UNKNOWN,
                f"ncu Stage {stage} timed out after {timeout}s",
                suggestion=(
                    f"Increase timeout or reduce kernel count. "
                    f"Command: {' '.join(cmd[:6])}..."
                ),
            )
        except FileNotFoundError:
            return ToolResult.fail(
                ErrorCode.TOOL_NOT_FOUND,
                f"ncu binary not found at: {cmd[0]}",
                suggestion="Install CUDA Toolkit or set [tools] ncu_path in config.",
            )

        elapsed = time.monotonic() - start

        if result.returncode != 0:
            return ToolResult.fail(
                ErrorCode.UNKNOWN,
                (
                    f"ncu Stage {stage} failed (exit {result.returncode}): "
                    f"{result.stderr[:500]}"
                ),
                suggestion="Check ncu stderr output for details.",
                context={"stderr": result.stderr, "cmd": cmd},
            )

        if not output_path.exists():
            return ToolResult.fail(
                ErrorCode.UNKNOWN,
                f"ncu completed but output file not found: {output_path}",
                suggestion="Check ncu output path and permissions.",
            )

        return ToolResult.ok(
            ProfilingResult(
                ncu_rep_path=output_path,
                stage=stage,
                returncode=result.returncode,
                stderr=result.stderr,
                elapsed_sec=elapsed,
            )
        )
