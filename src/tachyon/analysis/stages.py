"""Multi-stage deep analysis — split AI analysis into focused phases.

Instead of a single-shot LLM call, deep analysis breaks the work into
sequential stages, each with a focused prompt and independent tool budget.
This gives the LLM more "thinking room" per phase and produces deeper,
more actionable output.

Usage::

    from tachyon.analysis.stages import build_stage_prompts, run_staged_analysis

    specs = build_stage_prompts(mode="profile")
    results = await run_staged_analysis(
        backend, registry, user_prompt, system_prompt, specs,
        render_fn=render_callback,
    )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# English fallbacks (used when i18n is not initialized)
# ---------------------------------------------------------------------------

_FALLBACK_STAGE_1 = """## Stage 1: Metric Analysis

Above are pre-computed kernel metrics and rule engine findings. Perform deep analysis on this basis:

1. Call `list_kernels` and `get_kernel_summary` to review key metrics for all kernels
2. Call `run_analysis` for each kernel to get structured rule-engine analysis
3. Call `get_optimization_tree` to see the full optimization landscape
4. Focus on:
   - Roofline metrics: SM throughput / DRAM throughput / occupancy
   - CRITICAL/WARNING findings from the rule engine with specific values
   - Primary bottleneck type for each kernel (compute-bound / memory-bound / latency-bound)

**Do NOT call source-related tools (get_performance_hotspots, get_sass_for_source_line, get_stall_analysis_for_line, read_source_file).**

Output format: Group by kernel, one analysis section per kernel.
At the end, list the top-3 hotspots you consider most noteworthy (kernel_id + brief reason)."""

_FALLBACK_STAGE_2 = """## Stage 2: Source Attribution

Based on the metric analysis from the previous stage, now trace back to source code:

1. For the top hotspots identified, call `get_performance_hotspots` to get source-level hotspots
2. For each important hotspot, call `get_stall_analysis_for_line` to get stall breakdown and SPI
3. If specific SASS instructions are needed, call `get_sass_for_source_line`
4. To view source context, call `read_source_file`

**Do NOT analyze from memory alone. You MUST call tools to gather data.**

Output requirements:
- Per hotspot: file:line + severity% + SPI + dominant_stall + dominant_sass + include_chain
- Explain the root cause of stalls (instruction dependency? memory latency? barrier?)
- Distinguish "surface symptoms" from "root causes"
- At the end, list the top-3 most critical optimization opportunities, sorted by impact"""

_FALLBACK_STAGE_3 = """## Stage 3: Optimization Plan

Based on the analysis results from the first two stages, provide a comprehensive optimization plan.

1. For each optimization opportunity, provide:
   - Problem description (specific to file:line and metric values)
   - Root cause analysis (citing SPI, stall breakdown, SASS evidence)
   - Specific optimization suggestion (with code examples or pseudocode)
   - Expected effect (quantitative estimate of improvement)
   - Priority (HIGH / MEDIUM / LOW)
2. If there are multiple optimization directions, explain dependencies between them
3. Provide a verification plan: which metrics to monitor to validate optimization effect

Output format: Numbered list sorted by priority, each containing [Problem] [Root Cause] [Solution] [Expected Effect] [Priority]."""


# ---------------------------------------------------------------------------
# StageSpec
# ---------------------------------------------------------------------------

@dataclass
class StageSpec:
    """One stage of the multi-stage analysis."""

    name: str  # "Stage 1: Metric Analysis"
    prompt: str  # user prompt for this stage
    max_turns: int = 10


# ---------------------------------------------------------------------------
# build_stage_prompts
# ---------------------------------------------------------------------------

def build_stage_prompts(
    mode: str = "profile",
) -> list[StageSpec]:
    """Build stage specifications based on analysis mode.

    Uses i18n for stage names and prompts. Falls back to English constants
    when i18n is not initialized.

    Args:
        mode: "profile" for 3 stages, "chat" for 2 stages.

    Returns:
        List of StageSpec objects.
    """
    from tachyon.i18n import t

    if mode == "profile":
        return [
            StageSpec(
                name=t("stage.name.one", fallback="Stage 1: Metric Analysis"),
                prompt=t("stage.1.metrics.prompt", fallback=_FALLBACK_STAGE_1),
            ),
            StageSpec(
                name=t("stage.name.two", fallback="Stage 2: Source Attribution"),
                prompt=t("stage.2.source.prompt", fallback=_FALLBACK_STAGE_2),
            ),
            StageSpec(
                name=t("stage.name.three", fallback="Stage 3: Optimization Plan"),
                prompt=t("stage.3.recommend.prompt", fallback=_FALLBACK_STAGE_3),
            ),
        ]
    else:
        # Chat mode: 2 stages (metrics → source/recommend combined)
        combined_prompt = t("stage.2.source.prompt", fallback=_FALLBACK_STAGE_2) + "\n\n" + t("stage.3.recommend.prompt", fallback=_FALLBACK_STAGE_3)
        return [
            StageSpec(
                name=t("stage.name.one", fallback="Stage 1: Metric Analysis"),
                prompt=t("stage.1.metrics.prompt", fallback=_FALLBACK_STAGE_1),
            ),
            StageSpec(
                name=t("stage.name.combined", fallback="Stage 2: Deep Analysis"),
                prompt=combined_prompt,
            ),
        ]


# ---------------------------------------------------------------------------
# run_staged_analysis
# ---------------------------------------------------------------------------

async def run_staged_analysis(
    backend: Any,
    registry: Any,
    user_prompt: str,
    system_prompt: str,
    stage_specs: list[StageSpec],
    *,
    timeout_per_stage: int = 300,
    move_timeout: int = 120,
    render_fn: Callable[[int, str, str], None] | None = None,
) -> list[str]:
    """Run multi-stage analysis, returning text from each stage.

    Each stage reuses the same backend and registry, but runs its own
    agent loop with a fresh user prompt. The full text output of each
    stage is injected into the next stage's user message so the LLM
    has complete context without needing to re-call all tools.

    Args:
        backend: LLMBackend instance.
        registry: ToolRegistry instance.
        user_prompt: Pre-computed data prompt (data only, no instructions;
                    injected into stage 1 only).
        system_prompt: System prompt from persona.
        stage_specs: List of StageSpec objects.
        timeout_per_stage: Timeout per stage in seconds.
        move_timeout: Per-move timeout for LLM calls.
        render_fn: Optional callback(stage_idx, stage_name, text) for
                   real-time display.

    Returns:
        List of markdown strings, one per stage.
    """
    from tachyon.agent.loop import run_agent_loop
    from tachyon.i18n import t

    results: list[str] = []
    context_header = t("stage.context.previous", fallback="Below are the analysis results from previous stages (complete content). Continue building on this:")

    for idx, spec in enumerate(stage_specs):
        # Build user message for this stage
        if idx == 0:
            # Stage 1: pre-computed data + stage-specific instructions
            stage_user_msg = user_prompt + "\n\n" + spec.prompt
        else:
            # Subsequent stages: inject ALL previous stage outputs as context.
            # This gives the LLM the complete analysis so far, avoiding
            # the need to re-call tools just to recover data from prior stages.
            prev_outputs = "\n\n---\n\n".join(
                f"### {stage_specs[i].name}\n{results[i]}"
                for i in range(len(results))
            )
            stage_user_msg = (
                f"{context_header}\n\n"
                f"<previous_stage_output>\n{prev_outputs}\n</previous_stage_output>\n\n"
                + spec.prompt
            )

        # Collect output from this stage's agent loop
        text_parts: list[str] = []
        tool_calls_made: list[str] = []

        try:
            from tachyon.agent.context import DEEP_TOKEN_BUDGET

            async for event in run_agent_loop(
                backend=backend,
                registry=registry,
                user_message=stage_user_msg,
                system_prompt=system_prompt,
                history=None,
                stream=False,
                timeout=timeout_per_stage,
                move_timeout=move_timeout,
                context_budget=DEEP_TOKEN_BUDGET,
            ):
                if event.type == "text" and event.content:
                    text_parts.append(event.content)
                elif event.type == "tool_call":
                    name = event.data["name"] if event.data else "?"
                    tool_calls_made.append(name)
                    if render_fn:
                        render_fn(idx, spec.name, f"[tool] {name}")
                elif event.type == "system":
                    if render_fn and event.content:
                        render_fn(idx, spec.name, event.content)
        except Exception as e:
            logger.warning("Stage %d (%s) failed: %s", idx, spec.name, e)
            text_parts.append(f"*(Stage {idx + 1} failed: {e})*")

        stage_text = "".join(text_parts) or f"*(No output for {spec.name})*"
        results.append(stage_text)

        logger.info(
            "Stage %d/%d complete: %s (%d tool calls, %d chars)",
            idx + 1, len(stage_specs), spec.name,
            len(tool_calls_made), len(stage_text),
        )

    return results
