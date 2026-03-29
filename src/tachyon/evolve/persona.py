"""Evolve persona builder — constructs the optimization mode system prompt.

Loads ``persona_evolve.md`` template and fills in evolve-specific variables.
"""
from __future__ import annotations

from pathlib import Path

from ..agent.persona import _lang_prefix
from ..i18n import t as _t
from ..tools.registry import ToolRegistry

_EVOLVE_PERSONA_MD = Path(__file__).parent.parent / "agent" / "persona_evolve.md"

# Marker that the LLM must emit before the summary line.
_SUMMARY_MARKER = "[SUMMARY]"


def build_evolve_system_prompt(
    registry: ToolRegistry,
    baseline_metrics: str = "",
    current_iteration: int = 0,
    max_iterations: int = 10,
    best_iteration: int = -1,
    best_improvement: float = 0.0,
    experiment_history: str = "",
    max_turns: int = 15,
    kernel_summary: str = "",
    source_files: str = "",
    skill_knowledge: str = "",
    deep: bool = False,
) -> str:
    """Build the system prompt for evolve (optimization) mode."""
    template = _EVOLVE_PERSONA_MD.read_text(encoding="utf-8")

    # Build tool catalog from the (already filtered) registry
    tool_catalog = "\n".join(
        f"  - `{t.name}` — {t.description.split('.')[0]}"
        for t in registry.all_definitions()
    )

    # Build pre-loaded context section
    preloaded = ""
    if kernel_summary or source_files:
        preloaded_parts = [
            "\n## Pre-loaded Context (DO NOT call analysis tools — this data is already available)\n"
        ]
        if kernel_summary:
            preloaded_parts.append(f"### Kernel\n{kernel_summary}\n")
        if source_files:
            preloaded_parts.append(f"### Source Files\n{source_files}\n")
        if deep:
            preloaded_parts.append(
                "### Deep Analysis Mode\n"
                "Hotspot data (severity %, SPI, stall type, focus_hint) is in the Kernel section "
                "above and in every reprofile result. Your workflow:\n"
                "1. **Read** the top hotspot line (from Hotspots list above, or reprofile bottleneck_analysis)\n"
                "2. **Make ONE targeted edit** at that exact line based on dominant_stall + focus_hint\n"
                "3. **Immediately compile** — do NOT read more lines or make additional edits first\n"
                "4. After reprofile, check if top hotspot severity decreased. If it shifted, target the new line.\n"
                "Do NOT explore the file or make speculative edits. Data-driven means: read hotspot → edit → measure.\n"
            )
        preloaded = "\n".join(preloaded_parts)

    body = template.format(
        tool_catalog=tool_catalog,
        current_iteration=current_iteration,
        max_iterations=max_iterations,
        best_iteration=best_iteration,
        best_improvement=f"{best_improvement:.1f}%",
        baseline_metrics=baseline_metrics or "(no baseline)",
        experiment_history=experiment_history or "(no previous experiments)",
        max_turns=max_turns,
        half_budget=max_turns // 2,
    )

    if skill_knowledge:
        body += "\n\n## Optimization Knowledge\n\n" + skill_knowledge

    return _lang_prefix() + body + preloaded


def build_evolve_lean_prompt(
    registry: ToolRegistry,
    baseline_metrics: str = "",
    current_iteration: int = 0,
    max_iterations: int = 10,
    best_iteration: int = -1,
    best_improvement: float = 0.0,
    experiment_history: str = "",
    max_turns: int = 15,
    kernel_summary: str = "",
    source_files: str = "",
    skill_brief: str = "",
    deep: bool = False,
) -> str:
    """Build a minimal system prompt used after turn 0 to save tokens.

    Keeps: tool list, cycle order, critical rules, session state, pre-loaded
    context, and experiment history.
    """
    tool_catalog = "\n".join(
        f"  - `{t.name}` — {t.description.split('.')[0]}"
        for t in registry.all_definitions()
    )

    parts = [
        _lang_prefix(),
        "You are Tachyon Optimization Mode — a CUDA kernel optimizer.\n",
        "## Tools\n" + tool_catalog + "\n",
        "## Cycle: read_source_file → edit_source_file → compile_kernel → "
        "run_benchmark → reprofile → compare_metrics → STOP\n",
        "## Critical Rules (always apply)\n"
        "1. Use raw_text from read_source_file (or pre-loaded source) as "
        "old_content for edit_source_file. NEVER reconstruct from memory.\n"
        "2. Call compile_kernel() with NO arguments after editing.\n"
        "3. If compile fails 2 times, STOP editing — iteration is terminated.\n"
        "4. Must complete: compile → run_benchmark → reprofile → compare_metrics.\n"
        "5. After reprofile, iteration is OVER. New ideas → next iteration.\n"
        "6. Mentally verify your edit before calling edit_source_file: "
        "check variable names, brace matching, types, and includes.\n",
        f"**Turn budget**: {max_turns} turns. Reach compile_kernel by turn "
        f"{max_turns // 2}.\n",
        "## Session\n",
        f"Iteration: {current_iteration} / {max_iterations}\n",
        f"Best result: iteration {best_iteration} ({best_improvement:.1f}% improvement)\n",
        f"Baseline metrics: {baseline_metrics or '(no baseline)'}\n",
    ]

    # Include pre-loaded context in lean prompt too (critical for avoiding re-calls)
    if skill_brief:
        parts.append(f"\n## Optimization Skills\n{skill_brief}\n")
    if kernel_summary:
        parts.append(f"\n## Kernel\n{kernel_summary}\n")
    if source_files:
        parts.append(f"\n## Source Files\n{source_files}\n")

    if experiment_history:
        parts.append(f"\n## History\n{experiment_history}\n")

    if deep:
        parts.append(
            "\n## Deep Analysis\n"
            "Read top hotspot line → ONE edit at that line → compile immediately.\n"
            "Do NOT read multiple lines or make extra edits before compiling.\n"
        )

    return "\n".join(parts)


def build_evolve_iteration_prompt(
    iteration: int,
    best_metrics_summary: str,
    summary_marker: str = _SUMMARY_MARKER,
    max_turns: int = 15,
    strategy_reset: bool = False,
) -> str:
    """Build the user prompt for a single evolve iteration."""
    parts = [f"## Iteration {iteration}"]

    if strategy_reset:
        parts.append(
            "\n**STRATEGY RESET**: Previous optimization direction converged. "
            "Try a fundamentally different approach — do NOT repeat previous techniques."
        )

    if best_metrics_summary:
        parts.append(f"\nCurrent best: {best_metrics_summary}")

    hint = _t(
        "evolve.iteration.prompt",
        marker=summary_marker,
        max_turns=max_turns,
        half=max_turns // 2,
    )
    parts.append(f"\n{hint}")

    return "\n".join(parts)
