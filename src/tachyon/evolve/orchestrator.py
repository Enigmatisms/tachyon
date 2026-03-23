"""Evolve orchestrator — multi-iteration outer loop."""
from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator, Callable

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

# Map tool names → display status
_TOOL_STATUS = {
    "reprofile": "PROFILING",
    "compile_kernel": "COMPILING",
    "run_benchmark": "BENCHMARKING",
}

# Sentence boundary regex: CN/EN punctuation
_SENTENCE_RE = re.compile(r'(?<=[.!?。！？])\s+|(?<=[。！？])(?=\S)')

# Patterns that indicate LLM reasoning/planning, not a summary
_NON_SUMMARY_RE = re.compile(
    r'(?i)^('
    r'now |let me |I\'ll |I will |I need |I should |'
    r'let\'s |we need |we should |next |first |'
    r'the .{0,30}(approach|optimization|error|bug|issue|problem) |'
    r'my (approach|implementation|code|fix) |'
    r'this (approach|means|is|was|requires|doesn\'t) |'
    r'looking at |based on |after |before |since |'
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
        self._system_prompt: str | None = None

    async def run(self) -> list[ExperimentRecord]:
        """Run the full evolve loop until convergence or max iterations."""
        session = self._ctx.evolve
        results: list[ExperimentRecord] = []
        t_start = time.monotonic()
        user_feedback: str = ""
        strategy_reset: bool = False
        original_branch = self._ctx.git.get_current_branch()

        try:
            while not session.is_finished:
                elapsed = time.monotonic() - t_start
                if elapsed > self._total_timeout:
                    _log.warning(
                        "Total timeout (%ds) reached after %d iterations.",
                        self._total_timeout, session.current_iteration,
                    )
                    break

                record, fatal = await self._run_iteration(
                    session.current_iteration, user_feedback,
                    strategy_reset=strategy_reset,
                )
                strategy_reset = False
                results.append(record)
                session.complete_iteration()

                if self._on_iteration:
                    self._on_iteration(record)

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
                    except Exception as e:
                        _log.warning("Failed to save optimized state: %s", e)

                    # Reset convergence and best metrics for fresh direction
                    session.convergence_count = 0
                    session.best_metrics = None
                    session.best_iteration = -1
                    strategy_reset = True

                if self._interactive and not session.is_finished:
                    user_feedback = await self._prompt_user_feedback(
                        session.current_iteration, record,
                    )
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

        self._system_prompt = build_evolve_system_prompt(
            self._registry,
            baseline_metrics=self._format_baseline_metrics(),
            current_iteration=session.current_iteration,
            max_iterations=self._max_iterations,
            best_iteration=session.best_iteration,
            best_improvement=self._ctx.evolve.get_best_improvement(),
            experiment_history=self._format_experiment_history(),
            max_turns=self._max_agent_turns,
        )

        lean_prompt = build_evolve_lean_prompt(
            self._registry,
            baseline_metrics=self._format_baseline_metrics(),
            current_iteration=session.current_iteration,
            max_iterations=self._max_iterations,
            best_iteration=session.best_iteration,
            best_improvement=self._ctx.evolve.get_best_improvement(),
            experiment_history=self._format_experiment_history(),
            max_turns=self._max_agent_turns,
        )

        async for event in run_agent_loop(
            backend=self._backend,
            registry=self._registry,
            user_message=user_message,
            system_prompt=self._system_prompt,
            context_budget=self._context_budget,
            max_turns=self._max_agent_turns,
            skip_synthesis=True,
            lean_system_prompt=lean_prompt,
        ):
            yield event

    async def _run_iteration(
        self, iteration: int, user_feedback: str = "",
        *, strategy_reset: bool = False,
    ) -> tuple[ExperimentRecord, bool]:
        """Execute one complete evolve iteration.

        Returns (record, fatal). If fatal is True, the caller should stop
        the loop — the LLM is unusable (auth error, no tool calls, etc.).
        """
        session = self._ctx.evolve
        record = session.start_new_experiment()
        record.status = ExperimentStatus.HYPOTHESIS
        self._ctx.edit_locked = False  # Allow edits for this iteration
        self._ctx.turn_count = 0       # Reset turn counter
        self._ctx.max_turns = self._max_agent_turns
        self._ctx.compile_fail_count = 0  # Reset compile failure counter
        self._ctx.run_fail_count = 0      # Reset run failure counter
        self._ctx.edit_fail_count = 0     # Reset edit match failure counter
        t0 = time.monotonic()

        _log.info("=== Evolve Iteration %d ===", iteration)

        # Update live display
        if self._display:
            self._display.set_status("HYPOTHESIS")

        self._system_prompt = build_evolve_system_prompt(
            self._registry,
            baseline_metrics=self._format_baseline_metrics(),
            current_iteration=iteration,
            max_iterations=self._max_iterations,
            best_iteration=session.best_iteration,
            best_improvement=self._ctx.evolve.get_best_improvement(),
            experiment_history=self._format_experiment_history(),
            max_turns=self._max_agent_turns,
        )

        # Lean system prompt: used after turn 0 to save tokens
        lean_prompt = build_evolve_lean_prompt(
            self._registry,
            baseline_metrics=self._format_baseline_metrics(),
            current_iteration=iteration,
            max_iterations=self._max_iterations,
            best_iteration=session.best_iteration,
            best_improvement=self._ctx.evolve.get_best_improvement(),
            experiment_history=self._format_experiment_history(),
            max_turns=self._max_agent_turns,
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

        # Save working tree state for rollback (no commit)
        state_patch = self._ctx.git.save_working_state()
        record.git_commit_hash = self._ctx.git.get_current_hash()

        # Run agent loop and track tool calls
        fatal = False
        try:
            tool_call_count = 0
            hypothesis_parts: list[str] = []
            thinking_parts: list[str] = []
            llm_error_msg: str | None = None

            async for event in run_agent_loop(
                backend=self._backend,
                registry=self._registry,
                user_message=user_message,
                system_prompt=self._system_prompt,
                context_budget=self._context_budget,
                max_turns=self._max_agent_turns,
                skip_synthesis=True,
                lean_system_prompt=lean_prompt,
            ):
                if event.type == "text" and event.content:
                    hypothesis_parts.append(event.content)
                elif event.type == "thinking" and event.content:
                    thinking_parts.append(event.content)
                elif event.type == "tool_call":
                    tool_call_count += 1
                    self._ctx.turn_count = tool_call_count
                    _log.info("  Tool call: %s", event.content)
                    if self._display and event.data:
                        tool_name = event.data.get("name", "")
                        if tool_name:
                            self._display.set_status(
                                _TOOL_STATUS.get(tool_name, "EDITING"),
                                tool_name,
                            )
                elif event.type == "tool_result":
                    if self._display and event.data:
                        tool_name = event.data.get("name", "")
                        if tool_name:
                            self._display.set_status(
                                _TOOL_STATUS.get(tool_name, "EDITING"),
                                tool_name,
                            )
                elif event.type == "system":
                    if event.content:
                        _log.info("  System: %s", event.content)
                    if "LLM error" in (event.content or ""):
                        llm_error_msg = event.content

            record.tool_call_count = tool_call_count

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
            self._evaluate_iteration(record, state_patch)
        else:
            self._rollback(record, state_patch)

        return record, fatal

    @staticmethod
    def _extract_summary(
        record: ExperimentRecord,
        text: str,
        marker: str = "[SUMMARY]",
    ) -> None:
        """Extract summary from LLM output text using the [SUMMARY] marker.

        Three strategies, tried in order:
        1. Find the last ``[SUMMARY]`` marker and take 1-3 sentences after it.
        2. Take the last substantive paragraph (>30 chars) as a summary.
        3. Take the last 3 sentences of the full text.
        """
        if not text:
            return
        clean = _strip_markdown(text)

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
                record.summary = " ".join(sentences[:3])
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
                    record.summary = " ".join(sentences[-3:])
                    return

        # Strategy 3: Last 3 sentences that look like a summary
        flat = clean.replace("\n", " ")
        sentences = _split_sentences(flat)
        summary_sentences = [s for s in sentences if _is_summary_text(s)]
        if summary_sentences:
            record.summary = " ".join(summary_sentences[-3:])
        elif sentences:
            record.summary = " ".join(sentences[-3:])

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
            lines.append(line)

        return "\n".join(lines)
