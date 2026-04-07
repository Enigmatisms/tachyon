"""Evolve orchestrator — multi-iteration outer loop."""
from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from ..agent.loop import AgentEvent, run_agent_loop
from ..evolve.display import _strip_markdown
from ..llm.backend import LLMBackend
from ..tools.registry import ToolRegistry
from .context import EvolveContext
from .models import ExperimentRecord, ExperimentStatus
from .persona import (
    build_evolve_iteration_prompt,
    build_evolve_lean_prompt,
    build_evolve_system_prompt,
)
from .tools import _BUILD_SUCCESS_FIELD

_log = logging.getLogger(__name__)

IterationCallback = Callable[[ExperimentRecord], None]

# Acceptance threshold: GPU time improvement > 3% → accept
_SIGNIFICANT_THRESHOLD = 3.0

# Adaptive temperature: ramps up within a direction, resets on strategy switch
_TEMP_BASE = 0.1
_TEMP_STEP = 0.05
_TEMP_MAX = 0.4

# Tools available during evolve iterations (analysis tools excluded to prevent waste)
_EVOLVE_TOOLS = {
    "read_source_file",
    "edit_source_file",
    "compile_kernel",
    "run_benchmark",
    "reprofile",
    "compare_metrics",
    "get_evolve_status",
}

# Analysis tools unlocked in deep mode (Phase 2) for SASS/stall investigation
_DEEP_ANALYSIS_TOOLS = {
    "get_stall_analysis_for_line",
    "get_sass_for_source_line",
    "get_performance_hotspots",
}

# Map tool names → display status
_TOOL_STATUS = {
    "read_source_file": "READING",
    "edit_source_file": "EDITING",
    "compile_kernel": "COMPILING",
    "run_benchmark": "BENCHMARKING",
    "reprofile": "PROFILING",
    "compare_metrics": "COMPARING",
    "get_evolve_status": "READING",
}

# Sentence boundary regex: CN/EN punctuation
_SENTENCE_RE = re.compile(r'(?<=[.!?。！？])\s+|(?<=[。！？])(?=\S)')

# Patterns that indicate LLM reasoning/planning, not a summary
_NON_SUMMARY_RE = re.compile(
    r'(?i)^('
    r'now |let me |I\'ll |I will |I need |I should |'
    r'let\'s |we need |we should |next |first |'
    r'my (approach|plan|implementation|fix) |'
    r'looking at |based on |since |'
    r'however[,. ]|unfortunately[,. ]|instead[,. ]'
    r')',
)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, handling both CN and EN punctuation."""
    parts = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip() and len(s.strip()) > 5]


def _is_summary_text(text: str) -> bool:
    """Check if text looks like an optimization summary (not LLM reasoning)."""
    return not bool(_NON_SUMMARY_RE.match(text.strip()))


class EvolveOrchestrator:
    """Multi-iteration optimization loop."""

    def __init__(
        self,
        backend: LLMBackend,
        registry: ToolRegistry,
        ctx: EvolveContext,
        *,
        max_iterations: int = 10,
        max_agent_turns: int = 15,
        total_timeout: int = 3600,
        context_budget: int = 120_000,
        interactive: bool = False,
        quiet: bool = False,
        on_iteration: IterationCallback | None = None,
        display=None,
        skill_registry=None,
    ) -> None:
        self._backend = backend
        self._registry = registry
        self._ctx = ctx
        self._max_iterations = max_iterations
        self._max_agent_turns = max_agent_turns
        self._total_timeout = total_timeout
        self._context_budget = context_budget
        self._interactive = interactive
        self._quiet = quiet
        self._on_iteration = on_iteration
        self._display = display
        self._skill_registry = skill_registry
        self._system_prompt: str | None = None

    async def run(self) -> list[ExperimentRecord]:
        """Run the full evolve loop until convergence or max iterations."""
        session = self._ctx.evolve
        results: list[ExperimentRecord] = []
        t_start = time.monotonic()
        user_feedback: str = ""
        strategy_reset: bool = False
        direction_iter: int = 0  # iterations since last direction reset
        original_branch = self._ctx.git.get_current_branch()

        try:
            while not session.is_finished:
                elapsed = time.monotonic() - t_start
                if elapsed > self._total_timeout:
                    _log.warning(
                        "Total timeout (%ds) reached after %d iterations.",
                        self._total_timeout, session.current_iteration,
                    )
                    if self._display:
                        self._display.notify(
                            f"[yellow]Timeout ({self._total_timeout}s) reached "
                            f"after {session.current_iteration} iterations. "
                            f"Use --timeout to increase.[/yellow]"
                        )
                    break

                record, fatal = await self._run_iteration(
                    session.current_iteration, user_feedback,
                    strategy_reset=strategy_reset,
                    direction_iter=direction_iter,
                )
                strategy_reset = False
                direction_iter += 1
                results.append(record)
                session.complete_iteration()

                if self._on_iteration:
                    self._on_iteration(record)

                # Deep mode: activate analysis after 2 consecutive stagnations
                if (self._ctx.config.deep
                        and not self._ctx.deep_active
                        and session.convergence_count >= 2):
                    self._ctx.deep_active = True
                    _log.info(
                        "Deep analysis activated after %d stagnation events",
                        session.convergence_count,
                    )
                    if self._display:
                        self._display.notify(
                            "[cyan]Deep Analysis Mode activated — "
                            "NCU bottleneck data now enabled.[/cyan]"
                        )

                # Only real crashes (agent loop exceptions) are fatal.
                # Transient issues (API glitch, no tool calls) are just
                # FAILED iterations — the loop continues naturally.
                if fatal:
                    break

                # Back off after LLM errors to give the API time to recover.
                if (record.status == ExperimentStatus.FAILED
                        and record.decision
                        and "LLM error" in record.decision):
                    _log.info(
                        "LLM error in iteration %d, waiting 10s before retry.",
                        record.iteration,
                    )
                    await asyncio.sleep(10)

                if session.has_converged:
                    _log.info(
                        "Converged after %d iterations — saving and switching strategy.",
                        session.current_iteration,
                    )
                    try:
                        # Clean unstaged changes from last iteration
                        self._ctx.git._run(["checkout", "."])
                        self._ctx.git._run(["clean", "-fd"])

                        # Commit staged (accepted) changes to a new branch
                        exhausted = f"tachyon-optimized-{int(time.time())}"
                        self._ctx.git.create_branch(exhausted)
                        self._ctx.git.checkout(exhausted)
                        self._ctx.git.snapshot(
                            f"[tachyon] optimized kernel "
                            f"(best improvement from iter {session.best_iteration})"
                        )

                        # Switch back to original branch (clean state)
                        self._ctx.git.checkout(original_branch)
                        _log.info(
                            "Saved optimized state to branch %s, "
                            "back on %s for new direction.",
                            exhausted, original_branch,
                        )

                        if self._display:
                            self._display.notify(
                                f"[cyan]Strategy converged — saved to branch "
                                f"{exhausted}. Trying new direction.[/cyan]"
                            )

                        # Record this direction's best as global candidate
                        session.record_direction_best(exhausted)
                    except Exception as e:
                        _log.warning("Failed to save optimized state: %s", e)

                    # Reset convergence and best metrics for fresh direction
                    session.convergence_count = 0
                    session.best_metrics = None
                    session.best_iteration = -1
                    self._ctx.deep_active = False
                    strategy_reset = True
                    direction_iter = 0

                if self._interactive and not session.is_finished:
                    user_feedback = await self._prompt_user_feedback(
                        session.current_iteration, record,
                    )

            # --- Finalize: save current direction + apply global best ---
            self._finalize_global_best(session, original_branch)

        finally:
            current = self._ctx.git.get_current_branch()
            if current != original_branch:
                self._ctx.git.checkout(original_branch)

        return results

    async def _prompt_user_feedback(
        self, iteration: int, record: ExperimentRecord,
    ) -> str:
        loop = asyncio.get_running_loop()

        print(f"\n{'─' * 60}")
        print(f"  Iteration {iteration} complete: {record.status.value}")
        if record.decision:
            print(f"  {record.decision}")
        print(f"  Next: iteration {iteration + 1}")
        print(f"{'─' * 60}")

        try:
            feedback = await loop.run_in_executor(
                None,
                lambda: input("  Enter guidance (or press Enter to continue): "),
            )
            return feedback.strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Continuing autonomously...")
            return ""

    async def run_single(
        self,
        user_message: str,
    ) -> AsyncIterator[AgentEvent]:
        """Run a single analysis-edit-verify cycle (for interactive chat mode)."""
        session = self._ctx.evolve
        session.start_new_experiment()

        _toolset = _EVOLVE_TOOLS | (_DEEP_ANALYSIS_TOOLS if self._ctx.deep_active else set())
        evolve_registry = self._registry.filter(_toolset)
        kernel_summary = self._format_kernel_summary()
        source_files_str = self._format_source_files()
        skill_knowledge, skill_brief = self._query_skills("evolve")

        self._system_prompt = build_evolve_system_prompt(
            evolve_registry,
            baseline_metrics=self._format_baseline_metrics(),
            current_iteration=session.current_iteration,
            max_iterations=self._max_iterations,
            best_iteration=session.best_iteration,
            best_improvement=self._ctx.evolve.get_best_improvement(),
            experiment_history=self._format_experiment_history(),
            max_turns=self._max_agent_turns,
            kernel_summary=kernel_summary,
            source_files=source_files_str,
            skill_knowledge=skill_knowledge,
            deep=self._ctx.deep_active,
        )

        lean_prompt = build_evolve_lean_prompt(
            evolve_registry,
            baseline_metrics=self._format_baseline_metrics(),
            current_iteration=session.current_iteration,
            max_iterations=self._max_iterations,
            best_iteration=session.best_iteration,
            best_improvement=self._ctx.evolve.get_best_improvement(),
            experiment_history=self._format_experiment_history(),
            max_turns=self._max_agent_turns,
            kernel_summary=kernel_summary,
            source_files=source_files_str,
            skill_brief=skill_brief,
            deep=self._ctx.deep_active,
        )

        async for event in run_agent_loop(
            backend=self._backend,
            registry=evolve_registry,
            user_message=user_message,
            system_prompt=self._system_prompt,
            context_budget=self._context_budget,
            max_turns=self._max_agent_turns,
            move_timeout=150,
            max_retries=2,
            skip_synthesis=True,
            lean_system_prompt=lean_prompt,
            urgent_compile_tool="compile_kernel",
            force_stop_check=self._check_force_stop,
        ):
            yield event

    async def _run_iteration(
        self, iteration: int, user_feedback: str = "",
        *, strategy_reset: bool = False, direction_iter: int = 0,
    ) -> tuple[ExperimentRecord, bool]:
        """Execute one complete evolve iteration.

        Returns (record, fatal). If fatal is True, the caller should stop
        the loop — the LLM is unusable (auth error, no tool calls, etc.).
        """
        session = self._ctx.evolve
        record = session.start_new_experiment()
        record.status = ExperimentStatus.HYPOTHESIS
        self._ctx.edit_locked = False  # Allow edits for this iteration
        self._ctx.benchmark_fix_allowed = 1  # Reset benchmark-fix allowance
        self._ctx.turn_count = 0       # Reset turn counter
        self._ctx.max_turns = self._max_agent_turns
        self._ctx.compile_fail_count = 0  # Reset compile failure counter
        self._ctx.run_fail_count = 0      # Reset run failure counter
        self._ctx.edit_fail_count = 0     # Reset edit match failure counter
        self._ctx.iteration_doomed = False  # Reset doomed flag
        t0 = time.monotonic()

        _log.info("=== Evolve Iteration %d ===", iteration)

        # Update live display
        if self._display:
            self._display.set_status("THINKING")

        # Filter registry to evolve-only tools (prevents analysis tool waste)
        _toolset = _EVOLVE_TOOLS | (_DEEP_ANALYSIS_TOOLS if self._ctx.deep_active else set())
        evolve_registry = self._registry.filter(_toolset)

        timer = self._ctx.timer

        # Pre-load analysis data to embed in system prompt
        with timer.phase("prompt_build"):
            kernel_summary = self._format_kernel_summary()
            source_files_str = self._format_source_files()
            skill_knowledge, skill_brief = self._query_skills("evolve")

            self._system_prompt = build_evolve_system_prompt(
                evolve_registry,
                baseline_metrics=self._format_baseline_metrics(),
                current_iteration=iteration,
                max_iterations=self._max_iterations,
                best_iteration=session.best_iteration,
                best_improvement=self._ctx.evolve.get_best_improvement(),
                experiment_history=self._format_experiment_history(),
                max_turns=self._max_agent_turns,
                kernel_summary=kernel_summary,
                source_files=source_files_str,
                skill_knowledge=skill_knowledge,
                deep=self._ctx.deep_active,
            )

            # Lean system prompt: used after turn 0 to save tokens
            lean_prompt = build_evolve_lean_prompt(
                evolve_registry,
                baseline_metrics=self._format_baseline_metrics(),
                current_iteration=iteration,
                max_iterations=self._max_iterations,
                best_iteration=session.best_iteration,
                best_improvement=self._ctx.evolve.get_best_improvement(),
                experiment_history=self._format_experiment_history(),
                max_turns=self._max_agent_turns,
                kernel_summary=kernel_summary,
                source_files=source_files_str,
                skill_brief=skill_brief,
                deep=self._ctx.deep_active,
            )

            user_message = build_evolve_iteration_prompt(
                iteration=iteration,
                best_metrics_summary=self._format_best_metrics(),
                max_turns=self._max_agent_turns,
                strategy_reset=strategy_reset,
            )

            if user_feedback:
                user_message = (
                    f"<user_guidance>\n{user_feedback}\n</user_guidance>\n\n"
                    + user_message
                )

            # Pre-read primary source file so LLM can edit immediately
            source_preview = self._read_primary_source()
            if source_preview:
                user_message += source_preview

        # Save working tree state for rollback (no commit)
        with timer.phase("git_ops"):
            state_patch = self._ctx.git.save_working_state()
            record.git_commit_hash = self._ctx.git.get_current_hash()

        # Run agent loop and track tool calls
        temperature = min(_TEMP_BASE + direction_iter * _TEMP_STEP, _TEMP_MAX)
        _log.info("  Temperature: %.2f (direction_iter=%d)", temperature, direction_iter)
        fatal = False
        _llm_t0: float = 0.0  # Track LLM API wall time
        _agent_t0 = time.monotonic()  # Track agent loop wall time
        _tool_before = timer.sum_matching("tool:") if timer.enabled else 0.0
        _llm_before = timer._totals.get("llm_api", 0.0) if timer.enabled else 0.0
        try:
            tool_call_count = 0
            hypothesis_parts: list[str] = []
            thinking_parts: list[str] = []
            llm_error_msg: str | None = None

            async for event in run_agent_loop(
                backend=self._backend,
                registry=evolve_registry,
                user_message=user_message,
                system_prompt=self._system_prompt,
                context_budget=self._context_budget,
                max_turns=self._max_agent_turns,
                move_timeout=150,
                max_retries=2,
                skip_synthesis=True,
                lean_system_prompt=lean_prompt,
                urgent_compile_tool="compile_kernel",
                force_stop_check=self._check_force_stop,
                temperature=temperature,
            ):
                if event.type == "text" and event.content:
                    hypothesis_parts.append(event.content)
                elif event.type == "thinking" and event.content:
                    thinking_parts.append(event.content)
                elif event.type == "tool_call":
                    # Record LLM API time (from llm_start to first tool_call)
                    if timer.enabled and _llm_t0 > 0:
                        timer.record("llm_api", time.monotonic() - _llm_t0)
                        _llm_t0 = 0.0
                    tool_call_count += 1
                    self._ctx.turn_count = tool_call_count
                    _log.info("  Tool call: %s", event.content)
                    if self._display and event.data:
                        tool_name = event.data.get("name", "")
                        status = _TOOL_STATUS.get(tool_name, "RUNNING")
                        self._display.set_status(status, tool_name)
                elif event.type == "tool_result":
                    # Record tool execution time from event data
                    if timer.enabled and event.data:
                        tn = event.data.get("name", "tool")
                        te = event.data.get("elapsed", 0)
                        if te > 0:
                            timer.record(f"tool:{tn}", te)
                elif event.type == "system":
                    if event.content:
                        _log.info("  System: %s", event.content)
                    # LLM call starting → track time and show THINKING
                    if event.data and event.data.get("llm_start"):
                        if timer.enabled:
                            if _llm_t0 > 0:
                                timer.record("llm_api", time.monotonic() - _llm_t0)
                            _llm_t0 = time.monotonic()
                        if self._display and tool_call_count == 0:
                            self._display.set_status("THINKING")
                    if "LLM error" in (event.content or ""):
                        llm_error_msg = event.content

            record.tool_call_count = tool_call_count

            # Capture trailing LLM time (final response after last tool)
            if timer.enabled and _llm_t0 > 0:
                timer.record("llm_api", time.monotonic() - _llm_t0)
                _llm_t0 = 0.0

            # Record agent overhead: agent loop wall minus LLM API minus tool execution
            if timer.enabled:
                agent_wall = time.monotonic() - _agent_t0
                tool_delta = timer.sum_matching("tool:") - _tool_before
                llm_delta = timer._totals.get("llm_api", 0.0) - _llm_before
                overhead = agent_wall - tool_delta - llm_delta
                if overhead > 0.05:
                    timer.record("agent_overhead", overhead)

            # Surface LLM errors to user (not just logs)
            if llm_error_msg:
                if self._display:
                    self._display.notify(
                        f"[red]Iteration {iteration} LLM error:[/red] {llm_error_msg}"
                    )
                record.status = ExperimentStatus.FAILED
                record.decision = llm_error_msg

            elif tool_call_count == 0:
                if self._display:
                    self._display.notify(
                        f"[yellow]Iteration {iteration}: LLM did not call any tools. "
                        f"The model may not support function calling.[/yellow]"
                    )
                record.status = ExperimentStatus.FAILED
                record.decision = (
                    "LLM did not call any tools. "
                    "The model may not support function calling."
                )

        except Exception as e:
            msg = f"Agent loop error in iteration {iteration}: {e}"
            if self._display:
                self._display.notify(f"[red]{msg}[/red]")
            _log.error(msg)
            record.status = ExperimentStatus.FAILED
            record.decision = str(e)
            fatal = True

        # Build hypothesis from thinking + text events
        # thinking (mid-loop) first, text (final) last — [SUMMARY] is at the end
        full_text = "\n".join(thinking_parts + hypothesis_parts)
        record.hypothesis = full_text[:2000]  # truncated for display/history

        elapsed = time.monotonic() - t0
        record.elapsed_sec = elapsed

        # Extract summary from full text (before truncation)
        if not fatal and full_text:
            self._extract_summary(record, full_text)

        if not fatal:
            with timer.phase("evaluate"):
                self._evaluate_iteration(record, state_patch)
        else:
            with timer.phase("rollback"):
                self._rollback(record, state_patch)

        return record, fatal

    @staticmethod
    def _extract_summary(
        record: ExperimentRecord,
        text: str,
        marker: str = "[SUMMARY]",
        max_len: int = 300,
    ) -> None:
        """Extract summary from LLM output text using the [SUMMARY] marker.

        Three strategies, tried in order:
        1. Find the last ``[SUMMARY]`` marker and take 1-3 sentences after it.
        2. Take the last substantive paragraph (>30 chars) as a summary.
        3. Take the last 3 sentences of the full text.

        Result is capped at *max_len* characters.
        """
        if not text:
            return
        clean = _strip_markdown(text)

        def _cap(s: str) -> str:
            if len(s) <= max_len:
                return s
            cut = s.rfind(" ", 0, max_len)
            return s[:cut if cut > max_len // 2 else max_len] + "..."

        # Strategy 1: [SUMMARY] marker (last occurrence)
        idx = clean.rfind(marker)
        if idx >= 0:
            after = clean[idx + len(marker):].strip()
            # Stop at the next bracket marker if any
            end = after.find("[")
            if end > 0:
                after = after[:end].strip()
            sentences = _split_sentences(after)
            if sentences:
                record.summary = _cap(" ".join(sentences[:3]))
                return

        # Strategy 2: Last substantive paragraph that looks like a summary
        paragraphs = [
            p.strip() for p in clean.split("\n\n")
            if len(p.strip()) > 30
        ]
        for para in reversed(paragraphs):
            flat_para = para.replace("\n", " ")
            if _is_summary_text(flat_para):
                sentences = _split_sentences(flat_para)
                if sentences:
                    record.summary = _cap(" ".join(sentences[-3:]))
                    return

        # Strategy 3: Last 3 sentences that look like a summary
        flat = clean.replace("\n", " ")
        sentences = _split_sentences(flat)
        summary_sentences = [s for s in sentences if _is_summary_text(s)]
        if summary_sentences:
            record.summary = _cap(" ".join(summary_sentences[-3:]))
        elif sentences:
            record.summary = _cap(" ".join(sentences[-3:]))

    def _evaluate_iteration(
        self,
        record: ExperimentRecord,
        state_patch: str | None,
    ) -> None:
        """Evaluate the result of an iteration and decide next steps."""
        # Don't overwrite a pre-existing failure (e.g. LLM error set
        # before this point).  Just rollback and move on.
        if record.status == ExperimentStatus.FAILED and record.decision:
            self._rollback(record, state_patch)
            return

        build_ok = True
        if record.comparison and _BUILD_SUCCESS_FIELD in record.comparison:
            build_ok = record.comparison[_BUILD_SUCCESS_FIELD]

        has_compiled = bool(record.build_log) or record.status not in (
            ExperimentStatus.PENDING,
            ExperimentStatus.HYPOTHESIS,
            ExperimentStatus.EDITING,
        )

        if not has_compiled:
            tool_count = record.tool_call_count
            if tool_count == 0:
                reason = (
                    "LLM did not call any tools. The model may not support "
                    f"function calling. Elapsed: {record.elapsed_sec:.1f}s."
                )
            else:
                reason = (
                    f"LLM called {tool_count} tool(s) but compile_kernel "
                    "was not invoked."
                )
            _log.warning("Iteration %d: %s", record.iteration, reason)
            record.status = ExperimentStatus.FAILED
            record.decision = reason
            self._rollback(record, state_patch)
            return

        if has_compiled and not build_ok:
            record.status = ExperimentStatus.FAILED
            record.decision = "Compilation failed — fix build errors before retrying."
            self._rollback(record, state_patch)
            return

        if record.run_exit_code is not None and record.run_exit_code != 0:
            record.status = ExperimentStatus.FAILED
            record.decision = (
                f"Binary crashed (exit code {record.run_exit_code}). "
                "Optimization may have broken correctness."
            )
            self._rollback(record, state_patch)
            return

        if record.optimized_metrics is None:
            record.status = ExperimentStatus.FAILED
            calls = record.tool_call_count
            record.decision = (
                f"No optimized metrics after {calls} tool call(s) — "
                "reprofile was not called. "
                "Always run: compile_kernel → run_benchmark → reprofile."
            )
            self._rollback(record, state_patch)
            return

        if record.optimized_metrics is not None and self._ctx.evolve.baseline_metrics is not None:
            record.baseline_metrics = self._ctx.evolve.baseline_metrics
            record.compare()  # Keep for LLM context / experiment history

            dur_opt = record.optimized_metrics.duration_ms
            # Comparison target: best accepted (if any), else original baseline
            best = self._ctx.evolve.best_metrics or self._ctx.evolve.baseline_metrics
            dur_ref = best.duration_ms if best else None

            if dur_opt is not None and dur_ref is not None and dur_ref > 0:
                pct = (dur_ref - dur_opt) / dur_ref * 100.0  # positive = faster

                if pct > _SIGNIFICANT_THRESHOLD:
                    record.status = ExperimentStatus.SUCCESS
                    # Show improvement vs reference, and vs original baseline
                    pct_vs_base = (
                        (self._ctx.evolve.baseline_metrics.duration_ms - dur_opt)
                        / self._ctx.evolve.baseline_metrics.duration_ms * 100.0
                        if self._ctx.evolve.baseline_metrics.duration_ms
                        else None
                    )
                    if best is self._ctx.evolve.baseline_metrics:
                        record.decision = f"Accepted: GPU time -{pct:.1f}%"
                    else:
                        record.decision = (
                            f"Accepted: GPU time -{pct:.1f}% "
                            f"(total: {pct_vs_base:.1f}%)"
                        )
                    self._ctx.evolve.record_improvement(record)
                    # Stage accepted files so rollbacks preserve them
                    if record.code_changes:
                        changed = list({c.file for c in record.code_changes})
                        self._ctx.git.stage_files(changed)
                elif pct >= 0:
                    record.status = ExperimentStatus.REGRESSION
                    record.decision = (
                        f"Rejected: minor {pct:.1f}% "
                        f"(<{_SIGNIFICANT_THRESHOLD:.0f}% threshold)"
                    )
                    self._ctx.evolve.convergence_count += 1
                    self._rollback(record, state_patch)
                else:
                    record.status = ExperimentStatus.REGRESSION
                    record.decision = f"Rejected: GPU time +{-pct:.1f}%"
                    self._ctx.evolve.convergence_count += 1
                    self._rollback(record, state_patch)
            else:
                record.status = ExperimentStatus.FAILED
                record.decision = "No GPU time data collected."
                self._rollback(record, state_patch)
        else:
            if record.optimized_metrics is not None and self._ctx.evolve.baseline_metrics is None:
                self._ctx.evolve.baseline_metrics = record.optimized_metrics
                record.status = ExperimentStatus.SUCCESS
                record.decision = "Baseline metrics established."
                self._ctx.evolve.record_improvement(record)
                if record.code_changes:
                    changed = list({c.file for c in record.code_changes})
                    self._ctx.git.stage_files(changed)

    def _rollback(
        self,
        record: ExperimentRecord,
        state_patch: str,
    ) -> None:
        if not self._ctx.config.git_auto_rollback:
            return
        if not state_patch:
            return

        success = self._ctx.git.restore_working_state(state_patch)
        if success:
            # Preserve FAILED status for diagnostics; only override others
            if record.status != ExperimentStatus.FAILED:
                record.status = ExperimentStatus.ROLLED_BACK
            _log.info("Rolled back iteration %d.", record.iteration)

    def _finalize_global_best(
        self,
        session,
        original_branch: str,
    ) -> None:
        """Save current direction + apply global best to original branch.

        After the loop ends:
        1. If current direction has accepted changes, save to a branch.
        2. Compare all directions' best results, pick the global winner.
        3. Apply the winner's file changes to the original branch (unstaged).
        4. Delete all tachyon-optimized-* branches.
        """
        git = self._ctx.git

        # Save current direction if it has unsaved accepted changes
        if session.best_metrics is not None:
            try:
                git._run(["checkout", "."])
                git._run(["clean", "-fd"])
                # Check if there are staged changes to save
                staged = git._run(["diff", "--cached", "--stat"])
                if staged.stdout.strip():
                    final_branch = f"tachyon-optimized-{int(time.time())}"
                    git.create_branch(final_branch)
                    git.checkout(final_branch)
                    git.snapshot(
                        f"[tachyon] optimized kernel "
                        f"(best from iter {session.best_iteration})"
                    )
                    session.record_direction_best(final_branch)
                    git.checkout(original_branch)
            except Exception as e:
                _log.warning("Failed to save final direction: %s", e)

        global_branch = session.global_best_branch
        if not global_branch:
            return

        try:
            # Ensure we're on the original branch
            current = git.get_current_branch()
            if current != original_branch:
                git.checkout(original_branch)

            # Apply global best: pull file state from best branch
            git._run(["checkout", global_branch, "--", "."])
            # Unstage so changes appear as working tree modifications
            git._run(["reset", "HEAD"])

            if self._display:
                dur = session.global_best_metrics.duration_ms if session.global_best_metrics else None
                dur_str = f"{dur:.2f}ms" if dur else "?"
                self._display.notify(
                    f"[green]Applied global best (iter {session.global_best_iteration}, "
                    f"GPU time {dur_str}) to working tree.[/green]"
                )

            # Clean up all tachyon-optimized-* branches
            branch_result = git._run(
                ["for-each-ref", "--format=%(refname:short)",
                 "refs/heads/tachyon-optimized-*"],
                check=False,
            )
            if branch_result.returncode == 0 and branch_result.stdout.strip():
                for b in branch_result.stdout.strip().splitlines():
                    b = b.strip()
                    if b:
                        git._run(["branch", "-D", b], check=False)
                _log.info("Cleaned up optimization branches.")
        except Exception as e:
            _log.warning("Failed to apply global best: %s", e)

    def _check_force_stop(self, messages: list, turn: int) -> str | None:
        """Return a stop reason if the iteration should be force-terminated."""
        ctx = self._ctx
        if ctx.edit_fail_count >= 4:
            ctx.iteration_doomed = True
            return (
                f"{ctx.edit_fail_count} consecutive edit failures — "
                "iteration terminated."
            )
        if ctx.compile_fail_count >= 2:
            ctx.iteration_doomed = True
            return (
                f"{ctx.compile_fail_count} consecutive compile failures — "
                "iteration terminated."
            )
        if ctx.run_fail_count >= 2:
            ctx.iteration_doomed = True
            return (
                f"{ctx.run_fail_count} consecutive run failures — "
                "iteration terminated."
            )
        return None

    # --- Skill knowledge ---

    def _get_skill_tags(self) -> set[str]:
        """Derive skill query tags from kernel bottleneck classification."""
        if not self._ctx.kernels:
            return set()
        k = self._ctx.kernels[0]
        sm = (
            k.metric_value("sm__throughput.avg.pct_of_peak_sustained_elapsed")
            or 0
        )
        dram = (
            k.metric_value(
                "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
            )
            or 0
        )
        if sm > 60 and dram < 60:
            return {"compute-bound"}
        if dram > 60 and sm < 60:
            return {"memory-bound"}
        if sm < 40 and dram < 40:
            return {"latency-bound"}
        return {"balanced"}

    def _query_skills(self, mode: str = "evolve") -> tuple[str, str]:
        """Return (skill_knowledge, skill_brief) for the given mode."""
        reg = self._skill_registry
        if not reg or reg.count == 0:
            return "", ""
        tags = self._get_skill_tags()
        knowledge = reg.query(tags=tags, mode=mode)
        brief = reg.query_brief(tags=tags, mode=mode)
        return knowledge, brief

    # --- Formatting helpers ---

    @staticmethod
    def _format_snapshot(metrics) -> str:
        """Format duration/SM/DRAM from a MetricSnapshot."""
        parts: list[str] = []
        if metrics.duration_ms is not None:
            parts.append(f"duration={metrics.duration_ms:.2f}ms")
        if metrics.sm_throughput is not None:
            parts.append(f"SM={metrics.sm_throughput:.1f}%")
        if metrics.dram_throughput is not None:
            parts.append(f"DRAM={metrics.dram_throughput:.1f}%")
        return ", ".join(parts)

    def _format_baseline_metrics(self) -> str:
        baseline = self._ctx.evolve.baseline_metrics
        if baseline is not None:
            return self._format_snapshot(baseline)
        if self._ctx.kernels:
            k = self._ctx.kernels[0]
            duration = k.metric_value("gpu__time_duration.sum")
            sm = k.metric_value("sm__throughput.avg.pct_of_peak_sustained_elapsed")
            dram = k.metric_value("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed")
            parts = []
            if duration is not None:
                parts.append(f"duration={duration / 1e6:.2f}ms")
            if sm is not None:
                parts.append(f"SM={sm:.1f}%")
            if dram is not None:
                parts.append(f"DRAM={dram:.1f}%")
            return ", ".join(parts) if parts else "(no metrics)"
        return "(no baseline)"

    def _format_best_metrics(self) -> str:
        best = self._ctx.evolve.best_metrics
        if best is None:
            return self._format_baseline_metrics()
        return self._format_snapshot(best)

    def _format_experiment_history(self) -> str:
        experiments = self._ctx.evolve.experiments
        if not experiments:
            return ""

        lines = []
        for e in experiments:
            improvement = "N/A"
            if e.comparison:
                improvement = f"{e.comparison.get('avg_improvement_pct', 0):+.1f}%"
            decision = e.decision or "no decision"
            line = (
                f"- Iteration {e.iteration}: {e.status.value} "
                f"({improvement}) — {decision[:120]}"
            )
            # Show what optimization was tried (so LLM avoids repeats)
            method = e.summary or ""
            if not method and e.hypothesis:
                # Extract first substantive sentence from hypothesis
                for sent in _split_sentences(e.hypothesis):
                    if _is_summary_text(sent):
                        method = sent
                        break
            if method:
                line += f"\n  Tried: {method[:150]}"
            # For failed builds, include error hint so LLM learns from mistakes
            if e.status == ExperimentStatus.FAILED and e.build_log:
                log_lower = e.build_log.lower()
                has_error = any(
                    kw in log_lower
                    for kw in ("error", "fail", "no such file", "not found")
                )
                if has_error:
                    err_lines = [
                        ln.strip() for ln in e.build_log.splitlines()
                        if any(
                            kw in ln.lower()
                            for kw in ("error", "fail", "no such file", "not found")
                        )
                        and len(ln.strip()) > 10
                    ]
                    if err_lines:
                        line += f"\n  Build error: {err_lines[-1][:150]}"
            # Runtime crash: classify and inject pattern so LLM avoids same mistake
            if (
                e.run_exit_code is not None
                and e.run_exit_code != 0
                and e.status in (ExperimentStatus.FAILED, ExperimentStatus.ROLLED_BACK)
            ):
                from .tools import _classify_crash
                crash_type = _classify_crash(e.run_exit_code, e.run_output)
                line += f"\n  CRASH: {crash_type}"
                # Show what code change caused it (so LLM avoids the pattern)
                if e.code_changes:
                    change_files = [c.file.rsplit("/", 1)[-1] for c in e.code_changes]
                    line += f"\n  Changed: {', '.join(change_files[:3])}"
                    # Extract first meaningful diff hunk as context
                    for c in e.code_changes:
                        if c.diff:
                            line += f"\n  AVOID: the edit pattern that caused this crash"
                            break
            # Show failure phase
            if e.status == ExperimentStatus.FAILED:
                reached = "compile" if e.build_log else "edit"
                if e.run_exit_code is not None:
                    reached = "benchmark"
                if e.optimized_metrics is not None:
                    reached = "reprofile"
                line += f"\n  Reached: {reached} phase"
            lines.append(line)

        return "\n".join(lines)

    def _format_kernel_summary(self) -> str:
        """Format kernel summary for embedding in system prompt."""
        if not self._ctx.kernels:
            return ""
        k = self._ctx.kernels[0]
        name = getattr(k, "demangled_name", None) or getattr(k, "kernel_name", "unknown")
        duration = k.metric_value("gpu__time_duration.sum") if hasattr(k, "metric_value") else None
        sm = k.metric_value("sm__throughput.avg.pct_of_peak_sustained_elapsed") if hasattr(k, "metric_value") else None
        dram = k.metric_value("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed") if hasattr(k, "metric_value") else None
        occ = k.metric_value("sm__warps_active.avg.pct_of_peak_sustained_active") if hasattr(k, "metric_value") else None
        parts = [f"Name: {name}"]
        if duration is not None:
            parts.append(f"Duration: {duration / 1e6:.2f} ms")
        if sm is not None:
            parts.append(f"SM throughput: {sm:.1f}%")
        if dram is not None:
            parts.append(f"DRAM throughput: {dram:.1f}%")
        if occ is not None:
            parts.append(f"Occupancy: {occ:.1f}%")
        # Bottleneck classification
        sm_val = sm or 0
        dram_val = dram or 0
        if sm_val > 60 and dram_val < 60:
            parts.append("Classification: COMPUTE-BOUND")
        elif dram_val > 60 and sm_val < 60:
            parts.append("Classification: MEMORY-BOUND")
        elif sm_val < 40 and dram_val < 40:
            parts.append("Classification: LATENCY-BOUND")
        else:
            parts.append("Classification: BALANCED")
        # Deep mode: inject source-level hotspot data for data-driven optimization
        if self._ctx.deep_active:
            hotspot_text = self._format_deep_hotspots()
            if hotspot_text:
                parts.append(hotspot_text)
        return "\n".join(parts)

    def _format_deep_hotspots(self) -> str:
        """Format top-3 source hotspots for deep mode kernel summary."""
        mapper = self._ctx.base.mapper
        if mapper is None:
            return ""

        try:
            report = mapper.get_bottleneck_report(top_n=3)
        except Exception:
            return ""

        if not report:
            return ""

        lines = ["\nHotspots (by severity):"]
        for item in report[:3]:
            file_short = Path(item["file"]).name
            lines.append(
                f"  L{item['line']} ({file_short}): "
                f"{item['severity']}% severity, "
                f"SPI={item['spi']}, "
                f"{item['dominant_stall']} — {item['focus_hint']}"
            )
        return "\n".join(lines)

    def _format_source_files(self) -> str:
        """Format source file list for embedding in system prompt."""
        paths = sorted(self._ctx.allowed_source_paths)
        if not paths:
            return "(no source files available)"
        return "\n".join(f"- {p}" for p in paths)

    def _read_primary_source(self) -> str:
        """Read primary source file for pre-loading into user prompt.

        Returns formatted source text or empty string.
        Limits to 300 lines to avoid token overflow.
        """
        from pathlib import Path

        paths = sorted(self._ctx.allowed_source_paths)
        if not paths:
            return ""
        # Pick the first .cu or .cuh file, or first file
        primary = paths[0]
        for p in paths:
            if p.endswith((".cu", ".cuh")):
                primary = p
                break
        try:
            content = Path(primary).read_text(encoding="utf-8")
            lines = content.splitlines()
            if len(lines) > 300:
                content = "\n".join(lines[:300])
                content += f"\n... ({len(lines) - 300} more lines truncated)"
            return (
                f"\n\n## Source Code (pre-loaded — use this as old_content for edits)\n"
                f"File: {primary}\n"
                f"```\n{content}\n```"
            )
        except Exception:
            return ""
