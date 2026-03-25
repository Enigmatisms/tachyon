"""Evolve tools — 6 tools for source editing, build, run, reprofile, compare, status.

Registered via ``register_evolve_tools(registry, ctx)`` following the same
closure-over-ctx pattern as the existing 12 analysis tools.
"""
from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from ..errors.handler import ErrorCode, ToolResult
from ..tools.registry import ToolDefinition, ToolRegistry
from .context import EvolveContext

_log = logging.getLogger(__name__)

# C1: Command allowlist for subprocess execution.
# Only these build system / benchmark executables are allowed as first token.
_ALLOWED_COMMANDS = frozenset({
    "make", "cmake", "ninja", "ninja-build",
    "nvcc", "g++", "gcc", "clang++", "clang", "c++", "cc",
    "cargo", "bazel", "xmake", "meson",
    "python", "python3",
    "echo",  # for testing/debugging
})

# H2: build_success field name for experiment tracking
_BUILD_SUCCESS_FIELD = "build_success"

# Error keywords for filtering compile output
_ERROR_KEYWORDS = frozenset({"error", "undefined", "undeclared", "expected", "fatal"})

# --- Diff Safety Scanner ---

# Synchronization primitives whose removal almost always causes race conditions
_SYNC_PRIMITIVES = (
    "__syncthreads", "__syncwarp", "__threadfence",
    "cooperative_groups::sync", "cg::sync",
)

# Signal names for human-readable crash descriptions
_SIGNAL_NAMES: dict[int, str] = {
    139: "SIGSEGV (segfault)",
    134: "SIGABRT (abort)",
    136: "SIGFPE (arithmetic error)",
    137: "SIGKILL (killed)",
}


def _scan_diff_safety(old_content: str, new_content: str) -> list[str]:
    """Analyze code changes for common CUDA safety issues.

    Returns a list of human-readable warnings. Empty list = no issues detected.
    Runs on raw text (no AST), so it's fast and zero-dependency.
    """
    warnings: list[str] = []
    old_lines = old_content.splitlines()
    new_lines = new_content.splitlines()
    removed = {ln.strip() for ln in old_lines} - {ln.strip() for ln in new_lines}
    added = {ln.strip() for ln in new_lines} - {ln.strip() for ln in old_lines}

    # 1. Removed synchronization primitives
    for line in removed:
        for prim in _SYNC_PRIMITIVES:
            if prim in line and not any(prim in a for a in added):
                warnings.append(
                    f"Removed '{prim}' without replacement — likely race condition."
                )
                break  # one warning per removed line

    # 2. Shared memory size changed
    import re as _re
    old_shared = _re.findall(r'__shared__\s+\w+\s+(\w+)\s*\[([^\]]+)\]', old_content)
    new_shared = _re.findall(r'__shared__\s+\w+\s+(\w+)\s*\[([^\]]+)\]', new_content)
    old_map = {name: size for name, size in old_shared}
    new_map = {name: size for name, size in new_shared}
    for name in old_map:
        if name in new_map and old_map[name] != new_map[name]:
            warnings.append(
                f"Shared memory '{name}' size changed "
                f"({old_map[name]} → {new_map[name]}) — verify all indexing matches."
            )

    # 3. Removed bounds guards (if idx < N pattern)
    guard_pat = _re.compile(r'if\s*\(.*(?:threadIdx|blockIdx|idx|tid)\b.*[<>]=?\s*\w+')
    removed_guards = [ln for ln in removed if guard_pat.search(ln)]
    added_guards = [ln for ln in added if guard_pat.search(ln)]
    if removed_guards and not added_guards:
        warnings.append(
            f"Removed {len(removed_guards)} bounds guard(s) — risk of out-of-bounds access."
        )

    return warnings


def _classify_crash(exit_code: int, output: str) -> str:
    """Classify a runtime crash into a concise pattern string."""
    sig = _SIGNAL_NAMES.get(exit_code, "")
    out_lower = output.lower()

    if exit_code == 139 or "segfault" in out_lower or "sigsegv" in out_lower:
        return "SIGSEGV — out-of-bounds memory access"
    if "illegal memory access" in out_lower or "misaligned address" in out_lower:
        return "CUDA illegal memory access — check array indexing and shared memory size"
    if "an illegal instruction" in out_lower:
        return "CUDA illegal instruction — possible type mismatch or uninitialized memory"
    if "launch failed" in out_lower or "launch timed out" in out_lower:
        return "CUDA kernel launch failure — check grid/block dimensions and resource usage"
    if "assert" in out_lower:
        return "Assertion failure — correctness check violated"
    if exit_code == 134:
        return "SIGABRT — assertion or CUDA error handler triggered"
    if sig:
        return sig
    return f"exit code {exit_code}"


def _filter_compile_errors(output: str, max_lines: int = 8) -> str:
    """Extract the most relevant compile error lines.

    Shows the first 5 and last 3 error lines, with a count in between.
    This reduces token waste from cascading errors.
    """
    lines = output.splitlines()
    error_lines = [
        ln for ln in lines
        if any(kw in ln.lower() for kw in _ERROR_KEYWORDS)
        and len(ln.strip()) > 5
    ]
    if not error_lines:
        # No recognized error lines — return tail of output
        return "\n".join(lines[-max_lines:])
    if len(error_lines) <= max_lines:
        return "\n".join(error_lines)
    head = error_lines[:5]
    tail = error_lines[-3:]
    remaining = len(error_lines) - 8
    label = "error" if remaining == 1 else "errors"
    return "\n".join(
        head + [f"... ({remaining} more {label})"] + tail
    )


def _validate_command(cmd_str: str) -> str | None:
    """Validate that the first token of the command is in the allowlist.

    Returns None if valid, or an error message string if not.
    """
    try:
        cmd_args = shlex.split(cmd_str)
    except ValueError as e:
        return f"Invalid command syntax: {e}"

    if not cmd_args:
        return "Empty command."

    first = os.path.basename(cmd_args[0])
    if first not in _ALLOWED_COMMANDS:
        return (
            f"Command '{first}' is not in the allowed list. "
            f"Allowed: {', '.join(sorted(_ALLOWED_COMMANDS))}. "
            f"Use only build/benchmark commands."
        )
    return None


def _run_cmd(
    cmd_str: str,
    *,
    is_default: bool,
    timeout: int,
    cwd: str,
) -> subprocess.CompletedProcess:
    """Run a command, using shell mode for trusted defaults."""
    if is_default:
        return subprocess.run(
            cmd_str, shell=True,
            capture_output=True, text=True,
            timeout=timeout, cwd=cwd,
        )
    return subprocess.run(
        shlex.split(cmd_str),
        capture_output=True, text=True,
        timeout=timeout, cwd=cwd,
    )


def register_evolve_tools(
    registry: ToolRegistry,
    ctx: EvolveContext,
) -> None:
    """Register 6 evolve tools."""

    # Reuse _resolve_file from source_view
    from ..tools.source_view import _resolve_file
    from .models import ExperimentStatus  # L8: top-level import

    # --- 1. edit_source_file ---
    async def edit_source_file(
        file: str,
        old_content: str,
        new_content: str,
        description: str = "",
    ) -> ToolResult:
        """Edit a source file by replacing exact content match.

        The old_content must exactly match a contiguous block in the file.
        Before editing, a git snapshot and file backup are created.
        """
        try:
            if ctx.iteration_doomed:
                return ToolResult.fail(
                    ErrorCode.INVALID_ARGUMENT,
                    "Iteration terminated due to repeated failures. "
                    "Output [SUMMARY] and stop.",
                )

            if ctx.edit_locked and ctx.benchmark_fix_allowed <= 0:
                return ToolResult.fail(
                    ErrorCode.INVALID_ARGUMENT,
                    "No edits allowed — compile already succeeded, "
                    "this iteration is past the editing phase.",
                    "Proceed to: run_benchmark → reprofile → compare_metrics. "
                    "New ideas go to the NEXT iteration.",
                )

            # If this is a benchmark-fix edit, consume the allowance
            if ctx.edit_locked and ctx.benchmark_fix_allowed > 0:
                ctx.benchmark_fix_allowed -= 1
                _log.info(
                    "Benchmark-fix edit allowed (%d remaining)",
                    ctx.benchmark_fix_allowed,
                )

            allowed = ctx.allowed_source_paths
            if not allowed:
                return ToolResult.fail(
                    ErrorCode.INVALID_ARGUMENT,
                    "No source paths are available for editing.",
                    "Ensure NCU report contains source file references.",
                )

            resolved = _resolve_file(file, allowed)
            if resolved is None:
                return ToolResult.fail(
                    ErrorCode.INVALID_ARGUMENT,
                    f"File '{file}' is not in the allowed source paths.",
                    "Use list_source_files to see available files.",
                )

            filepath = Path(resolved)
            if not filepath.is_file():
                return ToolResult.fail(
                    ErrorCode.NO_DEBUG_INFO,
                    f"File '{resolved}' does not exist on disk.",
                )

            current_content = filepath.read_text(encoding="utf-8")

            if old_content not in current_content:
                # Fallback: try fuzzy match by first/last line
                fuzzy_replacement = _fuzzy_match(old_content, new_content, current_content)
                if fuzzy_replacement is not None:
                    new_file_content = fuzzy_replacement
                else:
                    ctx.edit_fail_count += 1
                    # Return first 10 lines as hint so LLM can fix the match
                    hint_lines = current_content.splitlines()[:10]
                    hint = "\n".join(hint_lines)
                    suggestion = (
                        "IMPORTANT: Use the 'raw_text' field from read_source_file as "
                        "old_content. Do NOT reconstruct text from the 'lines' array — "
                        "this causes whitespace mismatches.\n\n"
                        f"Current file starts with:\n{hint}"
                    )
                    if ctx.edit_fail_count >= 3:
                        suggestion += (
                            f"\n\nFATAL: Edit failed {ctx.edit_fail_count} times. "
                            "STOP editing — call compile_kernel if you have pending "
                            "successful edits, or STOP this iteration."
                        )
                    return ToolResult.fail(
                        ErrorCode.INVALID_ARGUMENT,
                        "old_content does not match any section in the file. "
                        "The file may have been modified since you last read it.",
                        suggestion,
                    )
            else:
                # Exact match — standard replace
                new_file_content = current_content.replace(old_content, new_content, 1)

            ctx.edit_fail_count = 0  # Reset on successful match (exact or fuzzy)

            # Record change in current experiment
            from .models import CodeChange

            # Write the edited file
            filepath.write_text(new_file_content, encoding="utf-8")

            diff = CodeChange.compute_diff(current_content, new_file_content)
            lines = CodeChange.compute_lines_changed(current_content, new_file_content)

            # Record change in current experiment
            record = ctx.evolve.experiments[-1] if ctx.evolve.experiments else None
            if record is not None:
                record.code_changes.append(CodeChange(
                    file=resolved,
                    diff=diff[:500],
                    lines_changed=lines,
                ))

            # Save snapshot for debugging: full file after each edit
            try:
                snap_dir = (
                    ctx.git.repo_root / ".tachyon" / "evolve" / "snapshots"
                    / f"iter_{record.iteration}" if record else "latest"
                )
                rel = Path(resolved).relative_to(ctx.git.repo_root)
                (snap_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                (snap_dir / rel).write_text(new_file_content, encoding="utf-8")
            except Exception:
                _log.warning("Failed to save snapshot for %s", resolved)

            result_data: dict[str, Any] = {
                "file": resolved,
                "lines_changed": lines,
                "diff": diff[:2000],
            }

            # Safety scan: detect dangerous CUDA patterns in the diff
            safety_warnings = _scan_diff_safety(old_content, new_content)
            if safety_warnings:
                result_data["safety_warnings"] = safety_warnings
                result_data["next_step"] = (
                    "SAFETY WARNINGS detected — review them before calling "
                    "compile_kernel. Fix issues with another edit if needed."
                )

            return ToolResult.ok(result_data)

        except Exception as e:
            return ToolResult.fail(ErrorCode.UNKNOWN, str(e))

    registry.register(ToolDefinition(
        name="edit_source_file",
        description=(
            "Edit a source file by replacing an exact content match. "
            "The old_content parameter must exactly match the current file content. "
            "A git snapshot is automatically created before editing. "
            "After editing, you MUST call compile_kernel to verify the change compiles."
        ),
        parameters={
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "File path or basename to edit.",
                },
                "old_content": {
                    "type": "string",
                    "description": (
                        "Exact content block to replace. Must match current file content. "
                        "IMPORTANT: Always copy old_content from the 'raw_text' field "
                        "returned by read_source_file — do NOT reconstruct text from "
                        "the 'lines' array, as this causes whitespace mismatches."
                    ),
                },
                "new_content": {
                    "type": "string",
                    "description": "New content to insert in place of old_content.",
                },
                "description": {
                    "type": "string",
                    "description": "Brief description of the edit purpose.",
                },
            },
            "required": ["file", "old_content", "new_content"],
        },
        handler=edit_source_file,
        category="evolve",
    ))

    # --- 2. compile_kernel ---
    async def compile_kernel(
        build_cmd: str | None = None,
        timeout: int | None = None,
    ) -> ToolResult:
        """Compile the project using the configured build command."""
        try:
            if ctx.iteration_doomed:
                return ToolResult.fail(
                    ErrorCode.INVALID_ARGUMENT,
                    "Iteration terminated due to repeated failures. "
                    "Output [SUMMARY] and stop.",
                )

            is_default = build_cmd is None
            cmd_str = build_cmd or ctx.config.build_cmd
            if not cmd_str:
                return ToolResult.fail(
                    ErrorCode.INVALID_ARGUMENT,
                    "No build command configured.",
                    "Set build_cmd in evolve.toml or use --build CLI flag.",
                )

            # C1: Validate LLM-provided commands against allowlist.
            # Skip validation for user-configured defaults (from --build flag).
            if not is_default:
                cmd_error = _validate_command(cmd_str)
                if cmd_error:
                    return ToolResult.fail(
                        ErrorCode.INVALID_ARGUMENT,
                        cmd_error,
                        "Use build_cmd in evolve.toml to configure a trusted command.",
                    )

            timeout_sec = timeout or ctx.config.build_timeout
            record = ctx.evolve.experiments[-1] if ctx.evolve.experiments else None

            if record is not None:
                record.status = ExperimentStatus.COMPILING

            _log.info("Running build: %s", cmd_str)
            t0 = time.monotonic()

            # For default (shell) commands, strip leading "cd <path> &&" prefix
            # because cwd is already set to repo_root. This handles the common
            # case where build_cmd was written for a different working directory.
            effective_cmd = cmd_str
            if effective_cmd.lstrip().lower().startswith("cd "):
                stripped = effective_cmd.strip()
                m = re.match(r'(?i)cd\s+\S+\s*&&\s*', stripped)
                if m:
                    effective_cmd = stripped[m.end():]
                    _log.info(
                        "Stripped leading 'cd ... &&' from build_cmd (cwd=%s): %s",
                        ctx.git.repo_root, effective_cmd,
                    )

            result = _run_cmd(
                effective_cmd, is_default=is_default,
                timeout=timeout_sec, cwd=str(ctx.git.repo_root),
            )
            elapsed = time.monotonic() - t0

            success = result.returncode == 0
            error_output = ""

            if success:
                error_output = result.stderr.strip()
                ctx.compile_fail_count = 0
                ctx.edit_locked = True  # Compile succeeded — editing phase is over
            else:
                # Combine stderr + last 50 lines of stdout for context
                error_output = result.stderr.strip()
                stdout_lines = result.stdout.strip().splitlines()
                if stdout_lines:
                    error_output += "\n" + "\n".join(stdout_lines[-50:])
                ctx.compile_fail_count += 1

            build_log = result.stdout.strip()
            if result.stderr.strip():
                build_log += "\n--- stderr ---\n" + result.stderr.strip()

            # H2: Track build success in experiment record
            if record is not None:
                record.build_log = build_log[-2000:]
                record.comparison = record.comparison or {}
                record.comparison[_BUILD_SUCCESS_FIELD] = success

            # After 2 consecutive failures, hard-refuse further attempts
            if ctx.compile_fail_count >= 2:
                ctx.iteration_doomed = True
                return ToolResult.fail(
                    ErrorCode.INVALID_ARGUMENT,
                    f"BLOCKED: Build failed {ctx.compile_fail_count} consecutive "
                    "times. This iteration cannot be saved. "
                    "Do NOT call compile_kernel or edit_source_file again.",
                    "Output [SUMMARY] describing what you tried and why it failed, "
                    "then STOP.",
                )

            result_data: dict[str, Any] = {
                "success": success,
                "exit_code": result.returncode,
                "elapsed_sec": round(elapsed, 1),
                "error_output": (
                    _filter_compile_errors(error_output) if not success
                    else error_output[:1000]
                ),
                "build_cmd_used": cmd_str,
                "cwd": str(ctx.git.repo_root),
                "next_step": (
                    "run_benchmark → reprofile → compare_metrics"
                    if success else
                    "Read the error carefully. Fix ONLY the specific error "
                    "with edit_source_file, then call compile_kernel() "
                    "with no arguments. Do NOT rewrite large sections."
                ),
            }
            # Warn LLM if custom build_cmd was used — the output binary
            # may not match the configured executable for benchmark/reprofile
            if success and not is_default and ctx.executable:
                result_data["WARNING"] = (
                    "You used a custom build command. Ensure it produces the "
                    "same binary that run_benchmark and reprofile will measure "
                    f"({ctx.executable}). If the output binary differs, "
                    "reprofile will measure the WRONG binary. "
                    "Prefer compile_kernel() with NO arguments."
                )

            return ToolResult.ok(result_data)

        except subprocess.TimeoutExpired:
            return ToolResult.fail(
                ErrorCode.UNKNOWN,
                f"Build timed out after {timeout_sec}s.",
                "Consider increasing build_timeout or simplifying the build.",
            )
        except Exception as e:
            return ToolResult.fail(ErrorCode.UNKNOWN, str(e))

    registry.register(ToolDefinition(
        name="compile_kernel",
        description=(
            "Compile the project using the configured build command. "
            "Returns success/failure, exit code, and any error output. "
            "Always call this after edit_source_file to verify changes compile."
        ),
        parameters={
            "type": "object",
            "properties": {
                "build_cmd": {
                    "type": "string",
                    "description": "Override build command (default: config build_cmd).",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Build timeout in seconds (default: config build_timeout).",
                },
            },
            "required": [],
        },
        handler=compile_kernel,
        category="evolve",
    ))

    # --- 3. run_benchmark ---
    async def run_benchmark(
        run_cmd: str | None = None,
        timeout: int | None = None,
        metric_filter: list[str] | None = None,
    ) -> ToolResult:
        """Run the benchmark and parse performance metrics from output."""
        try:
            if ctx.iteration_doomed:
                return ToolResult.fail(
                    ErrorCode.INVALID_ARGUMENT,
                    "Iteration terminated due to repeated failures. "
                    "Output [SUMMARY] and stop.",
                )

            ctx.edit_locked = True
            is_default = run_cmd is None
            cmd_str = run_cmd or ctx.config.run_cmd
            if not cmd_str:
                return ToolResult.fail(
                    ErrorCode.INVALID_ARGUMENT,
                    "No run command configured.",
                    "Set run_cmd in evolve.toml or use --run CLI flag.",
                )

            # C1: Validate LLM-provided commands against allowlist.
            # Skip validation for user-configured defaults (executable from CLI).
            if not is_default:
                cmd_error = _validate_command(cmd_str)
                if cmd_error:
                    ctx.run_fail_count += 1
                    if ctx.run_fail_count >= 2:
                        return ToolResult.fail(
                            ErrorCode.INVALID_ARGUMENT,
                            cmd_error + (
                                f"\nFATAL: Run command failed {ctx.run_fail_count} times. "
                                "STOP — call run_benchmark() with NO arguments to use "
                                "the pre-configured command."
                            ),
                        )
                    return ToolResult.fail(
                        ErrorCode.INVALID_ARGUMENT,
                        cmd_error,
                        "Call run_benchmark() with NO arguments to use the pre-configured command.",
                    )

            timeout_sec = timeout or ctx.config.run_timeout
            record = ctx.evolve.experiments[-1] if ctx.evolve.experiments else None

            if record is not None:
                record.status = ExperimentStatus.RUNNING

            _log.info("Running benchmark: %s", cmd_str)
            t0 = time.monotonic()

            # Strip leading "cd <path> &&" for same reason as compile_kernel
            effective_run_cmd = cmd_str
            if effective_run_cmd.lstrip().lower().startswith("cd "):
                stripped = effective_run_cmd.strip()
                m = re.match(r'(?i)cd\s+\S+\s*&&\s*', stripped)
                if m:
                    effective_run_cmd = stripped[m.end():]

            result = _run_cmd(
                effective_run_cmd, is_default=is_default,
                timeout=timeout_sec, cwd=str(ctx.git.repo_root),
            )
            elapsed = time.monotonic() - t0

            stdout = result.stdout
            stderr = result.stderr
            parsed = _parse_benchmark_output(stdout, metric_filter)

            if record is not None:
                record.run_output = stdout[-2000:]
                record.run_exit_code = result.returncode

            if result.returncode == 0:
                ctx.run_fail_count = 0  # Reset on success
            elif ctx.benchmark_fix_allowed > 0:
                # Allow one fix edit after benchmark correctness failure
                ctx.edit_locked = False
                ctx.benchmark_fix_allowed -= 1
                _log.info(
                    "Benchmark failed (exit %d) — unlocked edit for fix "
                    "(%d remaining)", result.returncode,
                    ctx.benchmark_fix_allowed,
                )

            return ToolResult.ok({
                "exit_code": result.returncode,
                "elapsed_sec": round(elapsed, 1),
                "stdout": stdout[-3000:],
                "stderr": stderr[:1000],
                "parsed_metrics": parsed,
                "run_cmd_used": cmd_str,
                "next_step": (
                    "reprofile → compare_metrics"
                    if result.returncode == 0 else
                    "Correctness broken (exit {}). You have ONE chance to fix it: "
                    "call edit_source_file to fix the correctness bug only, then "
                    "compile_kernel → run_benchmark → reprofile → compare_metrics.".format(
                        result.returncode,
                    )
                ),
            })

        except subprocess.TimeoutExpired:
            return ToolResult.fail(
                ErrorCode.UNKNOWN,
                f"Benchmark timed out after {timeout_sec}s.",
                "Consider increasing run_timeout.",
            )
        except Exception as e:
            return ToolResult.fail(ErrorCode.UNKNOWN, str(e))

    registry.register(ToolDefinition(
        name="run_benchmark",
        description=(
            "Run the project benchmark and extract performance metrics. "
            "Parses common output formats (time, throughput, bandwidth). "
            "Returns parsed_metrics dict and raw output."
        ),
        parameters={
            "type": "object",
            "properties": {
                "run_cmd": {
                    "type": "string",
                    "description": "Override run command (default: config run_cmd).",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Run timeout in seconds (default: config run_timeout).",
                },
                "metric_filter": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Regex patterns to filter parsed metrics.",
                },
            },
            "required": [],
        },
        handler=run_benchmark,
        category="evolve",
    ))

    # --- 4. reprofile ---
    async def reprofile(
        kernel: str | None = None,
        ncu_set: str = "full",
        ncu_metrics: str | None = None,
    ) -> ToolResult:
        """Re-run NCU profiling on the modified binary.

        Uses the executable from evolve context to auto-profile.
        Loads the new .ncu-rep and refreshes the session context so that
        subsequent analysis tools see the updated metrics.
        """
        try:
            if ctx.iteration_doomed:
                return ToolResult.fail(
                    ErrorCode.INVALID_ARGUMENT,
                    "Iteration terminated due to repeated failures. "
                    "Output [SUMMARY] and stop.",
                )

            ctx.edit_locked = True
            from .models import MetricSnapshot

            # Need executable for re-profiling
            if not ctx.executable:
                return ToolResult.fail(
                    ErrorCode.TOOL_NOT_FOUND,
                    "No executable configured for re-profiling.",
                    "Provide an executable when starting evolve: "
                    "'tachyon evolve ./my_app [ARGS]'",
                )

            # Create profiler on demand if not provided at init
            profiler = ctx.profiler
            if profiler is None:
                from ..config.settings import TachyonConfig
                from ..profiler.ncu_profiler import NcuProfiler
                from ..profiler.tool_path import ToolPathResolver

                # Build a minimal config for the profiler
                tachyon_cfg = TachyonConfig.load()
                resolver = ToolPathResolver(tachyon_cfg)
                profiler = NcuProfiler(tachyon_cfg, resolver)
                ctx.profiler = profiler

            reader = ctx.reader
            if reader is None:
                from ..config.settings import TachyonConfig
                from ..reader.ncu_reader import NcuReportReader

                tachyon_cfg = TachyonConfig.load()
                reader = NcuReportReader(tachyon_cfg)
                ctx.reader = reader

            # Profile the modified binary (suppress spinner — evolve display is active)
            result = profiler.profile_basic(
                ctx.executable,
                ctx.exe_args,
                metric_set_override=ncu_set,
                metrics_override=ncu_metrics,
                no_spinner=True,
            )
            if not result.success or result.data is None:
                return ToolResult.fail(
                    ErrorCode.ANALYZER_FAILED,
                    "NCU profiling failed.",
                    result.error.suggestion if result.error else "",
                )

            ncu_rep_path = result.data.ncu_rep_path
            _log.info("New NCU report: %s", ncu_rep_path)

            # Load the new report
            load_result = reader.load(str(ncu_rep_path))
            if not load_result.success or load_result.data is None:
                return ToolResult.fail(
                    ErrorCode.ANALYZER_FAILED,
                    f"Failed to load new NCU report: {ncu_rep_path}",
                )

            new_kernels = load_result.data

            # Apply kernel filter if specified
            if kernel is not None:
                from tachyon.utils.kernel_filter import match_kernel_name
                new_kernels = [
                    k for k in new_kernels
                    if match_kernel_name(k.kernel_name, k.demangled_name, kernel)
                ]
                if not new_kernels:
                    return ToolResult.fail(
                        ErrorCode.METRIC_NOT_FOUND,
                        f"No kernels matching '{kernel}' found in new report.",
                    )

            # Update the base context with new kernel data
            ctx.base.kernels = new_kernels

            # Rebuild mapper from new report — PC addresses change after recompile
            try:
                from ..correlator.source_mapper import NCUMappingSystem
                ctx.base.mapper = NCUMappingSystem(str(ncu_rep_path))
            except Exception:
                _log.warning("Failed to rebuild mapper from new report.")

            ctx.base.allowed_source_paths = (
                ctx.base.build_allowed_source_paths(new_kernels, ctx.base.mapper)
            )
            ctx.base.embedded_sources = (
                ctx.base.collect_embedded_sources(new_kernels)
            )

            # Update experiment record
            record = ctx.evolve.experiments[-1] if ctx.evolve.experiments else None
            if record is not None:
                record.ncu_rep_path = ncu_rep_path
                record.status = ExperimentStatus.PROFILING
                # Extract metrics for the target kernel
                if new_kernels:
                    metrics = {
                        name: mv.value
                        for name, mv in new_kernels[0].metrics.items()
                        if mv is not None
                    }
                    record.optimized_metrics = MetricSnapshot.from_kernel_metrics(
                        metrics,
                    )

            # Summary of key metrics
            key_metrics: dict[str, Any] = {}
            for k in (new_kernels[:3] if kernel is None else new_kernels):
                name = k.demangled_name or k.kernel_name
                duration = k.metric_value("gpu__time_duration.sum")
                sm = k.metric_value("sm__throughput.avg.pct_of_peak_sustained_elapsed")
                dram = k.metric_value("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed")
                key_metrics[name] = {
                    "duration_ms": round((duration or 0) / 1e6, 2),
                    "sm_throughput_pct": round(sm or 0, 1),
                    "dram_throughput_pct": round(dram or 0, 1),
                }

            result_data: dict[str, Any] = {
                "ncu_rep_path": str(ncu_rep_path),
                "kernel_count": len(new_kernels),
                "key_metrics_summary": key_metrics,
            }

            # Compare with baseline so LLM sees the effect immediately
            baseline = ctx.evolve.baseline_metrics
            if baseline and baseline.duration_ms and record and record.optimized_metrics:
                opt_dur = record.optimized_metrics.duration_ms
                if opt_dur is not None and baseline.duration_ms > 0:
                    pct = (baseline.duration_ms - opt_dur) / baseline.duration_ms * 100.0
                    result_data["comparison_with_baseline"] = {
                        "baseline_ms": round(baseline.duration_ms, 2),
                        "optimized_ms": round(opt_dur, 2),
                        "change_pct": round(pct, 2),
                        "verdict": "IMPROVED" if pct > 3.0 else ("UNCHANGED" if pct >= 0 else "REGRESSED"),
                    }

            return ToolResult.ok(result_data)

        except Exception as e:
            return ToolResult.fail(ErrorCode.UNKNOWN, str(e))

    registry.register(ToolDefinition(
        name="reprofile",
        description=(
            "Re-run NCU profiling on the modified binary using the evolve executable. "
            "Automatically profiles the current build, loads the new .ncu-rep, and "
            "refreshes kernel data so subsequent analysis tools see updated metrics. "
            "After editing and compiling, use this to measure the effect of changes."
        ),
        parameters={
            "type": "object",
            "properties": {
                "kernel": {
                    "type": "string",
                    "description": "Target kernel name filter (optional).",
                },
                "ncu_set": {
                    "type": "string",
                    "description": "NCU metric set name (e.g., 'full', 'detailed', 'basic').",
                },
                "ncu_metrics": {
                    "type": "string",
                    "description": "Comma-separated NCU metric names.",
                },
            },
            "required": [],
        },
        handler=reprofile,
        category="evolve",
    ))

    # --- 5. compare_metrics ---
    async def compare_metrics(
        iteration: int | None = None,
        metric_names: list[str] | None = None,
    ) -> ToolResult:
        """Compare baseline and optimized metrics for an iteration."""
        try:
            experiments = ctx.evolve.experiments
            if not experiments:
                return ToolResult.fail(
                    ErrorCode.INVALID_ARGUMENT,
                    "No experiments recorded yet.",
                )

            # Find the experiment to compare
            if iteration is not None:
                record = next(
                    (e for e in experiments if e.iteration == iteration),
                    None,
                )
                if record is None:
                    return ToolResult.fail(
                        ErrorCode.INVALID_ARGUMENT,
                        f"No experiment found for iteration {iteration}.",
                    )
            else:
                record = experiments[-1]

            # Use session-level baseline if record baseline not yet set
            if record.baseline_metrics is None and ctx.evolve.baseline_metrics is not None:
                record.baseline_metrics = ctx.evolve.baseline_metrics

            if record.optimized_metrics is None:
                return ToolResult.fail(
                    ErrorCode.METRIC_NOT_FOUND,
                    "No optimized metrics — call reprofile first.",
                )
            if record.baseline_metrics is None:
                return ToolResult.ok({
                    "comparison": "(first iteration — baseline established, no comparison yet)",
                    "summary": f"Iter {record.iteration}: baseline established",
                    "avg_improvement_pct": 0.0,
                    "regressions": [],
                    "note": "Optimized metrics recorded. This will serve as baseline for subsequent iterations.",
                    "optimized_duration_ms": record.optimized_metrics.duration_ms,
                })

            comparison = record.compare(metric_names)

            # Format compact comparison
            rows = comparison["rows"]
            lines = []
            for r in rows:
                base = f"{r['baseline']:.2f}" if r["baseline"] is not None else "?"
                opt = f"{r['optimized']:.2f}" if r["optimized"] is not None else "?"
                change = f"{r['change_pct']:+.1f}%" if r["change_pct"] is not None else "?"
                # Use short metric name (strip common prefixes/suffixes)
                name = r["metric"].split(".")
                short = name[0] if len(name) <= 2 else ".".join(name[:2])
                lines.append(f"{short}: {base} → {opt} ({change}) [{r['status']}]")

            summary_line = (
                f"Iter {record.iteration}: "
                f"{comparison['avg_improvement_pct']:+.1f}% avg, "
                f"{comparison['improved_count']} improved, "
                f"{comparison['regressed_count']} regressed"
            )

            return ToolResult.ok({
                "comparison": "\n".join(lines),
                "summary": summary_line,
                "avg_improvement_pct": comparison["avg_improvement_pct"],
                "regressions": comparison["regressions"],
            })

        except Exception as e:
            return ToolResult.fail(ErrorCode.UNKNOWN, str(e))

    registry.register(ToolDefinition(
        name="compare_metrics",
        description=(
            "Compare baseline vs optimized metrics for an experiment iteration. "
            "Shows a formatted comparison table with per-metric change analysis. "
            "Metrics with < 2% change are marked UNCHANGED."
        ),
        parameters={
            "type": "object",
            "properties": {
                "iteration": {
                    "type": "integer",
                    "description": "Iteration number (default: latest).",
                },
                "metric_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Metric names to compare (default: standard set).",
                },
            },
            "required": [],
        },
        handler=compare_metrics,
        category="evolve",
    ))

    # --- 6. get_evolve_status ---
    async def get_evolve_status() -> ToolResult:
        """Get current evolve session status and experiment history."""
        try:
            data = ctx.evolve.get_status_summary()

            # Phase-aware next_action based on current iteration state
            record = ctx.evolve.experiments[-1] if ctx.evolve.experiments else None
            if record is not None:
                has_compiled = bool(record.build_log)
                build_ok = (
                    record.comparison.get(_BUILD_SUCCESS_FIELD, True)
                    if record.comparison else True
                )
                has_run = record.run_exit_code is not None
                has_profile = record.optimized_metrics is not None
                remaining = ctx.max_turns - ctx.turn_count

                if not has_compiled:
                    if remaining <= 5:
                        data["URGENT"] = (
                            f"{remaining} turns left — call compile_kernel NOW."
                        )
                elif not build_ok:
                    data["next_action"] = (
                        "Fix the compile error, then call compile_kernel()."
                    )
                elif not has_run:
                    data["next_action"] = (
                        "Call run_benchmark() with no arguments."
                    )
                elif not has_profile:
                    data["next_action"] = "Call reprofile() with no arguments."
                else:
                    data["next_action"] = (
                        "Call compare_metrics() then output [SUMMARY]."
                    )

            # Turn budget warning
            half = ctx.max_turns // 2
            if ctx.turn_count >= half and "URGENT" not in data:
                remaining = ctx.max_turns - ctx.turn_count
                data["WARNING"] = (
                    f"Turn {ctx.turn_count}/{ctx.max_turns} — "
                    f"only {remaining} left. "
                    "Complete: run_benchmark → reprofile → compare_metrics."
                )
            return ToolResult.ok(data)
        except Exception as e:
            return ToolResult.fail(ErrorCode.UNKNOWN, str(e))

    registry.register(ToolDefinition(
        name="get_evolve_status",
        description=(
            "Get the current evolve session status, iteration count, "
            "convergence state, and experiment history summary."
        ),
        parameters={
            "type": "object",
            "properties": {},
        },
        handler=get_evolve_status,
        category="evolve",
    ))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fuzzy_match(
    old_content: str,
    new_content: str,
    current_content: str,
    tolerance: int = 5,
) -> str | None:
    """Try to locate old_content in the file using first/last line anchoring.

    When exact match fails (e.g., whitespace differences from LLM
    reconstructing text from a lines array), this fallback:
      1. Extracts the first and last non-empty line of old_content.
      2. Finds the first line's position in the current file.
      3. Verifies the last line appears within ±tolerance lines.
      4. If matched, replaces the located region with new_content.

    Returns the full new file content on success, or None if no match found.
    """
    old_lines = old_content.splitlines()
    # Filter empty lines for anchors
    non_empty = [l for l in old_lines if l.strip()]
    if len(non_empty) < 2:
        return None

    first_anchor = non_empty[0].strip()
    last_anchor = non_empty[-1].strip()

    file_lines = current_content.splitlines()
    # Find first anchor
    start_idx = None
    for i, line in enumerate(file_lines):
        if line.strip() == first_anchor:
            start_idx = i
            break
    if start_idx is None:
        return None

    # Find last anchor within tolerance
    expected_range = len(old_lines) + tolerance
    end_idx = None
    for i in range(start_idx, min(start_idx + expected_range, len(file_lines))):
        if file_lines[i].strip() == last_anchor:
            end_idx = i
            break
    if end_idx is None:
        return None

    # Replace the located region
    before = "\n".join(file_lines[:start_idx])
    after = "\n".join(file_lines[end_idx + 1:])
    parts = [before]
    if before and not before.endswith("\n"):
        parts.append("\n")
    parts.append(new_content)
    if after and not new_content.endswith("\n"):
        parts.append("\n")
    parts.append(after)
    result = "".join(parts)

    _log.info(
        "Fuzzy match succeeded: lines %d-%d matched by first/last anchor",
        start_idx + 1, end_idx + 1,
    )
    return result


# Common benchmark output patterns
_BENCHMARK_PATTERNS = [
    # Time: "elapsed: 1.234 ms" / "Time: 1.234 ms"
    (r"(?:elapsed|time|duration)[:\s]+([\d.]+)\s*(ms|us|s|ns)", "time"),
    # Throughput: "1234.5 MB/s" / "throughput: 1234.5"
    (r"(?:throughput|bandwidth)[:\s]+([\d.]+)\s*(GB/s|MB/s|KB/s|TB/s)", "throughput"),
    # "X ms" standalone (Google Benchmark style)
    (r"([\d.]+)\s*ms", "time_ms"),
    # "ops/s: 1234"
    (r"(?:ops/s|ops_per_sec|operations/s)[:\s]+([\d.]+)", "ops_per_sec"),
    # "Mean: 1.234 ms"
    (r"Mean[:\s]+([\d.]+)\s*(ms|us|s)", "mean_time"),
]


def _parse_benchmark_output(
    stdout: str,
    metric_filter: list[str] | None = None,
) -> dict[str, float]:
    """Parse common benchmark output formats from stdout."""
    metrics: dict[str, float] = {}
    seen_keys: set[str] = set()

    for pattern, key in _BENCHMARK_PATTERNS:
        for m in re.finditer(pattern, stdout, re.IGNORECASE):
            value = float(m.group(1))
            unit = m.group(2) if len(m.groups()) > 1 else ""

            # Normalize to ms
            normalized_key = key
            if key == "time" and unit:
                normalized_key = f"time_{unit}"
                if unit == "us":
                    value = value / 1000.0
                elif unit == "s":
                    value = value * 1000.0
                elif unit == "ns":
                    value = value / 1_000_000.0

            # Deduplicate
            if normalized_key not in seen_keys:
                seen_keys.add(normalized_key)
                metrics[normalized_key] = round(value, 3)

    # Apply filter
    if metric_filter:
        filtered: dict[str, float] = {}
        for k, v in metrics.items():
            for pat in metric_filter:
                if re.search(pat, k, re.IGNORECASE):
                    filtered[k] = v
                    break
        metrics = filtered

    return metrics
