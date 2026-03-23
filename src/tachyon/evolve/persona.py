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
) -> str:
    """Build the system prompt for evolve (optimization) mode."""
    template = _EVOLVE_PERSONA_MD.read_text(encoding="utf-8")

    # Build tool catalog for evolve tools only
    evolve_tools = [
        t for t in registry.all_definitions()
        if t.name in (
            "edit_source_file", "compile_kernel", "run_benchmark",
            "reprofile", "compare_metrics", "get_evolve_status",
        )
    ]
    tool_catalog = "\n".join(
        f"  - `{t.name}` — {t.description.split('.')[0]}"
        for t in evolve_tools
    )

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

    return _lang_prefix() + body


def build_evolve_lean_prompt(
    registry: ToolRegistry,
    baseline_metrics: str = "",
    current_iteration: int = 0,
    max_iterations: int = 10,
    best_iteration: int = -1,
    best_improvement: float = 0.0,
    experiment_history: str = "",
    max_turns: int = 15,
) -> str:
    """Build a minimal system prompt used after turn 0 to save tokens.

    Keeps only: tool list, cycle order, session state, and experiment history.
    Strips rules, strategy, anti-patterns (LLM already internalized them).
    """
    evolve_tools = [
        t for t in registry.all_definitions()
        if t.name in (
            "edit_source_file", "compile_kernel", "run_benchmark",
            "reprofile", "compare_metrics", "get_evolve_status",
        )
    ]
    tool_catalog = "\n".join(
        f"  - `{t.name}` — {t.description.split('.')[0]}"
        for t in evolve_tools
    )

    parts = [
        _lang_prefix(),
        "You are Tachyon Optimization Mode — a CUDA kernel optimizer.\n",
        "## Tools\n" + tool_catalog + "\n",
        "## Cycle: analyze(1-2) → read+edit(1-3) → compile_kernel → "
        "run_benchmark → reprofile → compare_metrics → STOP\n",
        "**CRITICAL**: Must call compile_kernel AND reprofile every iteration.\n",
        f"**Progress check**: If ≥{max_turns // 2} turns used and not compiled, "
        "compile NOW with current edits.\n",
        "## Session\n",
        f"Iteration: {current_iteration} / {max_iterations}\n",
        f"Best result: iteration {best_iteration} ({best_improvement:.1f}% improvement)\n",
        f"Baseline metrics: {baseline_metrics or '(no baseline)'}\n",
    ]

    if experiment_history:
        parts.append(f"\n## History\n{experiment_history}\n")

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
