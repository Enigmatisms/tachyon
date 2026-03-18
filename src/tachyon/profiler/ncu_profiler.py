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


# ---------------------------------------------------------------------------
# Error classification — distinguish user program vs ncu vs tachyon errors
# ---------------------------------------------------------------------------

class _ErrorSource(str, Enum):
    """Where the error originated."""

    USER_PROGRAM = "user_program"  # The profiled executable crashed / errored
    NCU_TOOL = "ncu_tool"  # NCU itself had an internal error
    UNKNOWN = "unknown"  # Can't determine


# Patterns in stderr that indicate the *user's profiled program* failed.
# These come from the OS or libc, forwarded through NCU's stderr.
_USER_PROGRAM_PATTERNS: list[tuple[str, str]] = [
    # File/path errors
    ("no such file or directory", "The profiled program could not find a required file or path."),
    ("cannot open", "The profiled program failed to open a file."),
    ("failed to load", "The profiled program could not load a resource."),
    ("permission denied", "The profiled program lacks permission to access a resource."),
    # Crash signals
    ("segmentation fault", "The profiled program crashed with a segmentation fault (SIGSEGV)."),
    ("segfault", "The profiled program crashed with a segmentation fault."),
    ("bus error", "The profiled program crashed with a bus error (SIGBUS)."),
    ("aborted", "The profiled program was aborted (SIGABRT)."),
    ("core dumped", "The profiled program crashed and dumped core."),
    ("killed", "The profiled program was killed (SIGKILL), possibly OOM."),
    ("out of memory", "The profiled program ran out of memory."),
    ("std::bad_alloc", "The profiled program failed to allocate memory (C++ std::bad_alloc)."),
    # CUDA runtime errors from user code
    ("cudaerrornodevice", "No CUDA device available for the profiled program."),
    ("cuda error", "The profiled program encountered a CUDA runtime error."),
    ("cuda_error", "The profiled program encountered a CUDA runtime error."),
    ("an illegal memory access", "The profiled program made an illegal GPU memory access."),
    ("unspecified launch failure", "A GPU kernel in the profiled program failed to launch."),
    # User program argument errors
    ("unrecognized option", "The profiled program received an unrecognized command-line option."),
    ("invalid argument", "The profiled program received an invalid argument."),
    ("usage:", "The profiled program printed a usage message — check its arguments."),
    # Dynamic linker
    ("error while loading shared libraries", "The profiled program is missing a shared library."),
    ("cannot open shared object", "The profiled program cannot find a required .so library."),
]

# Patterns that indicate NCU itself had a problem (not the user's program).
_NCU_TOOL_PATTERNS: list[tuple[str, str]] = [
    ("==error==", "NVIDIA Nsight Compute reported an internal error."),
    ("ncu error", "NCU encountered an error."),
    ("failed to attach", "NCU failed to attach to the target process."),
    ("incompatible", "NCU version may be incompatible with the GPU or driver."),
    ("license", "NCU license issue detected."),
    ("driver version", "NCU/driver version mismatch."),
]


def _classify_error(stderr_text: str, returncode: int) -> tuple[_ErrorSource, str, str]:
    """Classify an NCU failure by parsing stderr.

    Returns (source, title, detail) where:
      - source: who caused the error
      - title: short error panel title
      - detail: human-readable explanation
    """
    lower = stderr_text.lower()

    # Check user program patterns first — these are more actionable
    for pattern, explanation in _USER_PROGRAM_PATTERNS:
        if pattern in lower:
            # Find the actual line containing the pattern for context
            for line in stderr_text.splitlines():
                if pattern in line.lower():
                    return (
                        _ErrorSource.USER_PROGRAM,
                        "Profiled Program Error",
                        f"{explanation}\n\nRelevant output:\n  {line.strip()}",
                    )
            return (
                _ErrorSource.USER_PROGRAM,
                "Profiled Program Error",
                explanation,
            )

    # Check NCU tool patterns
    for pattern, explanation in _NCU_TOOL_PATTERNS:
        if pattern in lower:
            return (
                _ErrorSource.NCU_TOOL,
                "NCU Profiler Error",
                explanation,
            )

    # Heuristic: certain exit codes hint at user program issues
    # Signals: 128+signal (e.g. 139=SIGSEGV, 134=SIGABRT, 137=SIGKILL)
    signal_map = {
        139: ("SIGSEGV — Segmentation fault", "The profiled program crashed."),
        134: ("SIGABRT — Aborted", "The profiled program was aborted."),
        137: ("SIGKILL — Killed", "The profiled program was killed (possibly OOM)."),
        136: ("SIGFPE — Floating point exception", "The profiled program had a math error."),
        135: ("SIGBUS — Bus error", "The profiled program had a bus error."),
    }
    if returncode in signal_map:
        sig_name, sig_detail = signal_map[returncode]
        return (
            _ErrorSource.USER_PROGRAM,
            f"Profiled Program Crashed ({sig_name})",
            sig_detail,
        )

    # Non-zero but unrecognized
    return (
        _ErrorSource.UNKNOWN,
        "Profiling Failed",
        f"ncu exited with code {returncode}. Unable to determine error source.",
    )


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
# Conservative: Stage1 basic (quick scan) → Stage2 detailed + extra sections
# Radical: Stage1 detailed → Stage2 full + SourceCounters
STRATEGY_CONFIGS: dict[ProfilingStrategy, tuple[StageConfig, StageConfig]] = {
    ProfilingStrategy.CONSERVATIVE: (
        StageConfig(metric_set="basic", timeout_sec=600),
        StageConfig(
            metric_set="detailed",
            timeout_sec=1800,
            extra_sections=["InstructionStats"],
        ),
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
        metric_set_override: str | None = None,
        metrics_override: str | None = None,
        verbose: bool = False,
    ) -> ToolResult[ProfilingResult]:
        """Stage 1: Quick Scan -- profile all kernels with basic/detailed metrics.

        Returns .ncu-rep path for downstream parsing by NcuReportReader.
        """
        strat = strategy or ProfilingStrategy(self._config.profiling.strategy)
        stage1_cfg, _ = STRATEGY_CONFIGS[strat]

        out_dir = output_dir or Path(tempfile.mkdtemp(prefix="tachyon_"))
        out_file = out_dir / "stage1.ncu-rep"

        cmd = self._build_command(
            metric_set=metric_set_override or stage1_cfg.metric_set,
            output_path=out_file,
            executable=executable,
            exe_args=args or [],
            source_counters=stage1_cfg.source_counters,
            extra_sections=stage1_cfg.extra_sections or None,
            extra_ncu_args=extra_ncu_args,
            metrics_override=metrics_override,
        )

        return self._run_ncu(
            cmd, stage=1, output_path=out_file,
            timeout=stage1_cfg.timeout_sec, verbose=verbose,
        )

    def profile_targeted(
        self,
        executable: str,
        args: list[str] | None = None,
        *,
        kernels: list[str],
        strategy: ProfilingStrategy | None = None,
        output_dir: Path | None = None,
        extra_ncu_args: list[str] | None = None,
        metric_set_override: str | None = None,
        metrics_override: str | None = None,
        verbose: bool = False,
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
            metric_set=metric_set_override or stage2_cfg.metric_set,
            output_path=out_file,
            executable=executable,
            exe_args=args or [],
            source_counters=stage2_cfg.source_counters,
            extra_sections=stage2_cfg.extra_sections or None,
            kernel_filter_args=kernel_filter_args,
            extra_ncu_args=extra_ncu_args,
            metrics_override=metrics_override,
        )

        return self._run_ncu(
            cmd, stage=2, output_path=out_file,
            timeout=stage2_cfg.timeout_sec, verbose=verbose,
        )

    def _build_command(
        self,
        *,
        metric_set: str,
        output_path: Path,
        executable: str,
        exe_args: list[str],
        source_counters: bool = False,
        extra_sections: list[str] | None = None,
        kernel_filter_args: list[str] | None = None,
        extra_ncu_args: list[str] | None = None,
        metrics_override: str | None = None,
    ) -> list[str]:
        """Build ncu command as a list (never shell=True)."""
        cmd = [self.ncu_path]

        # --metrics takes priority over --set (mutually exclusive in ncu)
        if metrics_override:
            cmd.extend(["--metrics", metrics_override])
        else:
            cmd.extend(["--set", metric_set])

        cmd.extend([
            "--export",
            str(output_path),
            "--force-overwrite",
            "--target-processes",
            "all",
            "--import-source",
            "yes",
        ])

        if source_counters:
            cmd.extend(["--section", "SourceCounters"])

        if extra_sections:
            for section in extra_sections:
                cmd.extend(["--section", section])

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
        verbose: bool = False,
    ) -> ToolResult[ProfilingResult]:
        """Execute ncu subprocess with safety constraints.

        Args:
            verbose: If True, stream full NCU stderr to terminal.
                     If False, show a live spinner animation instead.
        """
        from tachyon.utils.progress import (
            NcuSpinner,
            print_error_panel,
            print_ncu_line,
            print_stage_header,
            print_stage_result,
        )

        cmd_summary = f"{cmd[0]} ... {cmd[-1]}" if len(cmd) > 2 else " ".join(cmd)
        logger.info("Stage %d: running %s", stage, " ".join(cmd))
        print_stage_header(stage, cmd_summary)
        start = time.monotonic()

        # Non-verbose mode: spinner animation
        spinner: NcuSpinner | None = None
        if not verbose:
            spinner = NcuSpinner(stage)
            spinner.start()

        stderr_lines: list[str] = []
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # merge stderr into stdout — NCU writes progress to stdout
                text=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                for seg in line.split("\r"):
                    seg = seg.rstrip()
                    if not seg:
                        continue
                    stderr_lines.append(seg)
                    if verbose:
                        print_ncu_line(seg)
                    elif spinner:
                        spinner.update(seg)
            proc.wait(timeout=timeout)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            elapsed = time.monotonic() - start
            print_stage_result(stage, elapsed, success=False)
            print_error_panel(
                f"Stage {stage} Timeout",
                f"ncu timed out after {timeout}s",
                suggestion=f"Increase timeout or reduce kernel count. Command: {' '.join(cmd[:6])}...",
            )
            return ToolResult.fail(
                ErrorCode.UNKNOWN,
                f"ncu Stage {stage} timed out after {timeout}s",
                suggestion=(
                    f"Increase timeout or reduce kernel count. "
                    f"Command: {' '.join(cmd[:6])}..."
                ),
            )
        except FileNotFoundError:
            print_error_panel(
                "NCU Not Found",
                f"ncu binary not found at: {cmd[0]}",
                suggestion="Install CUDA Toolkit or set [tools] ncu_path in config.",
            )
            return ToolResult.fail(
                ErrorCode.TOOL_NOT_FOUND,
                f"ncu binary not found at: {cmd[0]}",
                suggestion="Install CUDA Toolkit or set [tools] ncu_path in config.",
            )
        finally:
            # ALWAYS stop spinner — prevents orphaned Live displays
            if spinner:
                spinner.stop()

        elapsed = time.monotonic() - start
        stderr_text = "\n".join(stderr_lines)

        if returncode != 0:
            print_stage_result(stage, elapsed, success=False)

            # Classify error source for clear attribution
            source, title, detail = _classify_error(stderr_text, returncode)

            if source == _ErrorSource.USER_PROGRAM:
                print_error_panel(
                    title,
                    f"{detail}\n\nExit code: {returncode}",
                    suggestion=(
                        "This is NOT a Tachyon error — your profiled program failed. "
                        "Fix the program and re-run. "
                        "Use -v for full NCU output."
                    ),
                )
            elif source == _ErrorSource.NCU_TOOL:
                print_error_panel(
                    title,
                    f"{detail}\n\nncu exit code: {returncode}\n{stderr_text[:200]}",
                    suggestion=(
                        "This is an NCU profiler error. Check your NCU installation, "
                        "GPU driver, and CUDA toolkit versions."
                    ),
                )
            else:
                # Unknown — show raw stderr for debugging
                print_error_panel(
                    f"Stage {stage} Failed",
                    f"ncu exited with code {returncode}:\n{stderr_text[:300]}",
                    suggestion="Check ncu stderr output above for details."
                    if verbose
                    else "Re-run with -v to see full NCU output.",
                )

            return ToolResult.fail(
                ErrorCode.UNKNOWN,
                (
                    f"[{source.value}] ncu Stage {stage} failed (exit {returncode}): "
                    f"{detail}"
                ),
                suggestion="Check ncu stderr output for details.",
                context={"stderr": stderr_text, "cmd": cmd, "error_source": source.value},
            )

        if not output_path.exists():
            print_stage_result(stage, elapsed, success=False)
            return ToolResult.fail(
                ErrorCode.UNKNOWN,
                f"ncu completed but output file not found: {output_path}",
                suggestion="Check ncu output path and permissions.",
            )

        print_stage_result(stage, elapsed, success=True)

        return ToolResult.ok(
            ProfilingResult(
                ncu_rep_path=output_path,
                stage=stage,
                returncode=returncode,
                stderr=stderr_text,
                elapsed_sec=elapsed,
            )
        )
