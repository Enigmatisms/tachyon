"""Shared analysis pipeline — kernel merge → Rule Engine → output → AI.

This module is the single source of truth for Tachyon's analysis logic.
Both ``tachyon profile`` and ``tachyon analyze`` delegate here.
"""
from __future__ import annotations

import logging
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from tachyon.config.settings import TachyonConfig
from tachyon.models.finding import Finding, Severity
from tachyon.models.kernel import KernelReport, MetricValue

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kernel merge
# ---------------------------------------------------------------------------

def merge_duplicate_kernels(reports: list[KernelReport]) -> list[KernelReport]:
    """Merge kernel launches with identical (name, grid, block) into averages.

    When users run the same kernel N times for statistical stability, we:
    - Average all scalar metrics across the N runs
    - Keep launch_params / device_info from the first run
    - Attach ``run_count`` (int) and ``metric_ranges`` (dict) as extra attrs

    Returns a new list with deduplicated kernels (order preserved).
    """
    groups: dict[tuple, list[KernelReport]] = defaultdict(list)
    order: list[tuple] = []  # preserve first-seen order
    for r in reports:
        key = (r.demangled_name, r.launch_params.grid, r.launch_params.block)
        if key not in groups:
            order.append(key)
        groups[key].append(r)

    merged: list[KernelReport] = []
    for key in order:
        group = groups[key]
        if len(group) == 1:
            group[0].run_count = 1  # type: ignore[attr-defined]
            merged.append(group[0])
            continue

        base = group[0]
        all_names = set()
        for r in group:
            all_names.update(r.metrics.keys())

        avg_metrics: dict[str, MetricValue] = {}
        ranges: dict[str, dict[str, float]] = {}
        for mname in all_names:
            vals = [r.metrics[mname].value for r in group if mname in r.metrics]
            unit = next(
                (r.metrics[mname].unit for r in group if mname in r.metrics), ""
            )
            if vals:
                avg = statistics.mean(vals)
                avg_metrics[mname] = MetricValue(name=mname, value=avg, unit=unit)
                if len(vals) > 1:
                    ranges[mname] = {
                        "min": min(vals),
                        "max": max(vals),
                        "avg": avg,
                        "stddev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                        "runs": len(vals),
                    }

        mr = KernelReport(
            kernel_name=base.kernel_name,
            demangled_name=base.demangled_name,
            launch_params=base.launch_params,
            device_info=base.device_info,
            metrics=avg_metrics,
            instanced_metrics=base.instanced_metrics,
            source_files=base.source_files,
            rule_results=base.rule_results,
        )
        mr.run_count = len(group)  # type: ignore[attr-defined]
        mr.metric_ranges = ranges  # type: ignore[attr-defined]
        merged.append(mr)

    return merged


# ---------------------------------------------------------------------------
# Rule Engine
# ---------------------------------------------------------------------------

def run_rule_engine(
    reports: list[KernelReport],
    *,
    verbose: bool = False,
    quiet: bool = False,
) -> dict[str, list[Finding]]:
    """Run all built-in analyzers on each kernel.

    Args:
        verbose: Include INFO-level findings.
        quiet: Only include CRITICAL-level findings.

    Returns:
        {demangled_name: [Finding, ...]}, sorted CRITICAL > WARNING > INFO.
    """
    from tachyon.analyzers.base import AnalyzerRegistry

    registry = AnalyzerRegistry()
    registry.auto_register()

    findings_map: dict[str, list[Finding]] = {}
    for report in reports:
        findings = registry.run_all(report)
        if quiet:
            findings = [f for f in findings if f.severity == Severity.CRITICAL]
        elif not verbose:
            findings = [f for f in findings if f.severity != Severity.INFO]
        findings_map[report.demangled_name] = findings

    return findings_map


# ---------------------------------------------------------------------------
# AI-enhanced analysis
# ---------------------------------------------------------------------------

def build_ai_prompt(
    reports: list[KernelReport],
    findings_map: dict[str, list[Finding]],
) -> str:
    """Build a focused AI prompt from kernel metrics + rule-engine findings.

    The prompt provides all pre-computed data AND explicitly instructs the
    agent to call tools for deeper investigation (source, SASS, stalls).
    """
    parts: list[str] = []
    category_counts: dict[str, int] = defaultdict(int)

    parts.append("# Pre-computed Analysis Data\n")
    parts.append("The following metrics and findings have been pre-computed. "
                 "Use them as starting context, then call tools to investigate deeper.\n")

    # Raw kernel metrics overview
    for r in reports:
        parts.append(f"\n## Kernel: {r.demangled_name}")
        lp = r.launch_params
        parts.append(f"- Grid: {lp.grid}, Block: {lp.block}")
        parts.append(f"- Registers/thread: {lp.registers_per_thread}, "
                      f"Shared mem: {lp.shared_mem_bytes}B")
        run_count = getattr(r, "run_count", 1)
        if run_count > 1:
            parts.append(f"- Averaged over {run_count} runs")

        # Source files if available
        if r.source_files:
            parts.append(f"- Source files embedded: {', '.join(r.source_files[:5])}")

        # Top metrics
        key_metrics = sorted(
            r.metrics.items(),
            key=lambda kv: abs(kv[1].value),
            reverse=True,
        )[:25]
        if key_metrics:
            parts.append("\n### Key Metrics")
            for name, mv in key_metrics:
                unit = mv.unit or ""
                parts.append(f"  {name} = {mv.value:.4g} {unit}")

    # Append rule-engine findings
    for kernel_name, findings in findings_map.items():
        if not findings:
            continue
        parts.append(f"\n## Rule-Engine Findings: {kernel_name}")
        for f in findings:
            parts.append(
                f"- **[{f.severity.value.upper()}]** {f.title}\n"
                f"  {f.detail}"
            )
            if f.metrics:
                metric_str = ", ".join(
                    f"{k}={v:.2f}" for k, v in list(f.metrics.items())[:5]
                )
                parts.append(f"  Metrics: {metric_str}")
            if f.source_location:
                parts.append(f"  Source: {f.source_location.file}:{f.source_location.line}")
            if f.category:
                category_counts[f.category] += 1

    findings_text = "\n".join(parts)

    # Determine dominant bottleneck for prompt specialization
    total = sum(category_counts.values()) or 1
    dominant = max(category_counts, key=category_counts.get) if category_counts else "general"  # type: ignore[arg-type]
    ratio = category_counts.get(dominant, 0) / total

    if ratio > 0.5 and dominant in ("compute",):
        focus = (
            "The dominant bottleneck is **COMPUTE**. Focus on:\n"
            "- Instruction-level inefficiencies (FP32 vs FP16/TF32, divergent branches)\n"
            "- Tensor Core / WMMA utilization opportunities\n"
            "- Algorithmic complexity reduction"
        )
    elif ratio > 0.5 and dominant in ("memory",):
        focus = (
            "The dominant bottleneck is **MEMORY**. Focus on:\n"
            "- Global memory coalescing patterns (SoA vs AoS, alignment)\n"
            "- Shared memory bank conflicts and padding\n"
            "- L2 cache utilization and memory traffic reduction"
        )
    elif ratio > 0.5 and dominant in ("latency",):
        focus = (
            "The dominant bottleneck is **LATENCY**. Focus on:\n"
            "- Occupancy limiters (registers, shared memory, block size)\n"
            "- Warp stall reasons (long scoreboard, barrier, dependency)\n"
            "- Synchronization overhead and instruction-level overlap"
        )
    else:
        focus = (
            "Multiple bottleneck types detected. Provide a comprehensive analysis:\n"
            "- Identify the #1 priority optimization\n"
            "- For each finding, explain WHY and HOW to fix"
        )

    # Build the instruction section with explicit tool-calling directives
    kernel_ids = list(range(len(reports)))
    tool_instructions = (
        f"# Your Task\n\n"
        f"{focus}\n\n"
        f"# Required Steps (follow in order)\n\n"
        f"1. **Call `run_analysis`** for each kernel ({kernel_ids}) to get "
        f"structured rule-engine findings with severity and recommendations.\n"
        f"2. **Call `get_source_hotspots`** for each kernel to find hot source lines "
        f"(requires -lineinfo; if unavailable, skip to step 4).\n"
        f"3. For each hotspot found, **call `get_sass_for_source_line`** and "
        f"**`get_stall_analysis_for_line`** to get instruction-level root cause.\n"
        f"4. **Call `get_optimization_tree`** for the optimization landscape.\n"
        f"5. Synthesize all evidence into a diagnosis with:\n"
        f"   - Bottleneck classification (confirmed by tool data)\n"
        f"   - Root cause with evidence citations [metric=value] or [file:line→SASS]\n"
        f"   - Prioritized, concrete recommendations\n"
        f"   - A 'Data Sources' section listing tools called and key data points\n\n"
        f"**IMPORTANT**: Do NOT skip tool calls. The user can see which tools you "
        f"called — analysis without tool evidence will appear ungrounded."
    )

    return f"{findings_text}\n\n{tool_instructions}"


def _show_ai_context(
    reports: list[KernelReport],
    findings_map: dict[str, list[Finding]],
    verbose: bool = False,
) -> None:
    """Show the user what data is being sent to the AI agent."""
    from tachyon.utils.progress import console

    lines = []
    for r in reports:
        name = r.demangled_name or r.kernel_name
        sm = r.metric_value("sm__throughput.avg.pct_of_peak_sustained_elapsed")
        dram = r.metric_value("dram__throughput.avg.pct_of_peak_sustained_elapsed")
        dur = r.metric_value("gpu__time_duration.sum")
        lines.append(
            f"  [bold]{name}[/bold]: "
            f"SM={sm:.1f}%" if sm is not None else "SM=-"
        )
        parts = []
        if dram is not None:
            parts.append(f"DRAM={dram:.1f}%")
        if dur is not None:
            parts.append(f"dur={dur / 1e6:.2f}ms")
        parts.append(f"regs={r.launch_params.registers_per_thread}")
        if parts:
            lines[-1] += f", {', '.join(parts)}"

    total_findings = sum(len(fs) for fs in findings_map.values())
    n_critical = sum(
        1 for fs in findings_map.values()
        for f in fs if f.severity == Severity.CRITICAL
    )
    n_warning = sum(
        1 for fs in findings_map.values()
        for f in fs if f.severity == Severity.WARNING
    )

    console.print("[dim]Data sent to AI agent:[/dim]")
    for line in lines:
        console.print(f"[dim]{line}[/dim]")
    console.print(
        f"[dim]  Rule-engine findings: {total_findings} total "
        f"({n_critical} CRITICAL, {n_warning} WARNING)[/dim]"
    )
    console.print(
        f"[dim]  Tools available: list_kernels, get_kernel_metrics, "
        f"run_analysis, get_source_hotspots, get_sass_for_source_line, "
        f"get_stall_analysis_for_line, get_optimization_tree[/dim]"
    )


def try_ai_analysis(
    reports: list[KernelReport],
    findings_map: dict[str, list[Finding]],
    config: TachyonConfig,
    verbose: bool = False,
) -> str | None:
    """Run AI-enhanced analysis. Returns markdown text or None if unavailable.

    Graceful: never raises, returns None on any failure.
    """
    from tachyon.utils.progress import console

    user_prompt = build_ai_prompt(reports, findings_map)
    if not user_prompt:
        return None

    # Transparency: show excerpt of what's being sent to AI
    _show_ai_context(reports, findings_map, verbose)

    # Try to create LLM backend
    try:
        from tachyon.llm.backend import create_backend

        api_key = config.llm.api_key or os.environ.get(config.llm.api_key_env)
        backend = create_backend(
            provider=config.llm.provider,
            model=config.llm.model,
            api_key=api_key,
            base_url=config.llm.base_url,
        )
    except Exception as e:
        if verbose:
            console.print(f"  [dim]AI skipped: {e}[/dim]")
        return None

    # Set up tools + session context
    from tachyon.analyzers.base import AnalyzerRegistry
    from tachyon.tools.analysis import register_analysis_tools
    from tachyon.tools.context import SessionContext
    from tachyon.tools.data_query import register_data_query_tools
    from tachyon.tools.registry import ToolRegistry
    from tachyon.tools.source import register_source_tools

    analyzer_registry = AnalyzerRegistry()
    analyzer_registry.auto_register()

    session = SessionContext(
        kernels=reports,
        action=None,
        correlator=None,
        registry=analyzer_registry,
    )
    tool_registry = ToolRegistry()
    register_data_query_tools(tool_registry, session)
    register_source_tools(tool_registry, session)
    register_analysis_tools(tool_registry, session)

    # Build system prompt
    from tachyon.agent.persona import build_kernel_context, build_system_prompt

    system_prompt = build_system_prompt(
        tool_registry, build_kernel_context(reports)
    )

    # Run agent loop (single-shot)
    import asyncio

    from tachyon.agent.loop import run_agent_loop

    async def _run() -> str:
        text_parts: list[str] = []
        tool_calls_made: list[str] = []
        async for event in run_agent_loop(
            backend=backend,
            registry=tool_registry,
            user_message=user_prompt,
            system_prompt=system_prompt,
            stream=False,
            timeout=config.llm.timeout,
        ):
            if event.type == "text" and event.content:
                text_parts.append(event.content)
            elif event.type == "tool_call":
                name = event.data["name"] if event.data else "?"
                tool_calls_made.append(name)
                if verbose:
                    console.print(f"  [dim]→ {event.content}[/dim]")
            elif event.type == "tool_result" and verbose:
                console.print(f"  [dim]← {event.content}[/dim]")
        if tool_calls_made:
            console.print(
                f"  [dim]Agent tool calls: "
                f"{' → '.join(tool_calls_made)}[/dim]"
            )
        return "".join(text_parts)

    try:
        return asyncio.run(_run()) or None
    except Exception as e:
        logger.warning("AI analysis failed: %s", e)
        if verbose:
            console.print(f"  [yellow]AI analysis failed: {e}[/yellow]")
        return None


# ---------------------------------------------------------------------------
# Full analysis pipeline (the shared entry point)
# ---------------------------------------------------------------------------

def run_analysis(
    report_path: Path,
    config: TachyonConfig,
    *,
    kernel_filter: str | None = None,
    verbose: bool = False,
    quiet: bool = False,
    no_ai: bool = False,
    output_file: Path | None = None,
) -> None:
    """Complete analysis pipeline: load → merge → rules → render → AI.

    This is the **single shared entry point** used by both
    ``tachyon profile`` and ``tachyon analyze``.

    Args:
        report_path: Path to .ncu-rep file.
        config: TachyonConfig instance.
        kernel_filter: Optional glob pattern to filter kernels.
        verbose: Show all findings including INFO.
        quiet: Show only CRITICAL findings.
        no_ai: Skip AI-enhanced analysis.
        output_file: Write output to file instead of stdout.
    """
    import sys

    import click
    from rich.markdown import Markdown

    from tachyon.reader.ncu_reader import NcuReportReader
    from tachyon.utils.progress import console, print_error_panel

    # ── Step 1: Load report ──
    reader = NcuReportReader(config)
    load_result = reader.load(report_path)
    if not load_result.success:
        assert load_result.error is not None
        print_error_panel(
            "Report Load Error",
            load_result.error.message,
            suggestion=load_result.error.suggestion,
        )
        sys.exit(1)

    assert load_result.data is not None
    raw_reports = load_result.data

    # ── Step 2: Filter kernels ──
    if kernel_filter:
        from tachyon.utils.kernel_filter import filter_kernels
        raw_reports = filter_kernels(raw_reports, kernel_filter)
        if not raw_reports:
            console.print(f"[yellow]No kernels matching '{kernel_filter}'.[/yellow]")
            sys.exit(0)

    # ── Step 3: Merge duplicate kernel runs ──
    reports = merge_duplicate_kernels(raw_reports)
    if len(reports) < len(raw_reports):
        console.print(
            f"  [dim]Merged {len(raw_reports)} launches → "
            f"{len(reports)} unique kernel(s) (averaged)[/dim]"
        )

    # ── Step 4: Rule Engine ──
    findings_map = run_rule_engine(reports, verbose=verbose, quiet=quiet)

    # ── Step 5: Render terminal output ──
    from tachyon.report.terminal import TerminalReporter

    reporter = TerminalReporter()
    output_text = reporter.render(reports, findings_map)

    if output_file:
        output_file.write_text(output_text)
        click.echo(f"Analysis written to {output_file}")
    else:
        click.echo(output_text)

    # ── Step 6: AI-enhanced analysis (optional) ──
    if not no_ai:
        console.print()
        console.rule("[bold cyan]AI-Enhanced Analysis[/bold cyan]")
        ai_text = try_ai_analysis(reports, findings_map, config, verbose=verbose)
        if ai_text:
            console.print(Markdown(ai_text))
        else:
            key_env = config.llm.api_key_env
            provider = config.llm.provider
            # Detect ducc environment for better guidance
            import os
            if os.environ.get("ANTHROPIC_AUTH_TOKEN") and os.environ.get("ANTHROPIC_BASE_URL"):
                console.print(
                    f"  [yellow]AI analysis unavailable.[/yellow]\n"
                    f"  Ducc environment detected (ANTHROPIC_AUTH_TOKEN set).\n"
                    f"  Provider: {provider}, Model: {config.llm.model}\n"
                    f"  Base URL: {config.llm.base_url}\n"
                    f"  If authentication fails, the token may need re-encryption.\n"
                    f"  Or use [bold]--no-ai[/bold] to suppress this message.\n"
                    f"  For interactive AI analysis: [bold]tachyon chat {report_path}[/bold]"
                )
            else:
                console.print(
                    f"  [yellow]AI analysis unavailable.[/yellow]\n"
                    f"  To enable, set: [bold]export {key_env}=<your-key>[/bold]\n"
                    f"  Provider: {provider}, Model: {config.llm.model}\n"
                    f"  Or use [bold]--no-ai[/bold] to suppress this message.\n"
                    f"  For interactive AI analysis: [bold]tachyon chat {report_path}[/bold]"
                )

    click.secho(f"\nReport: {report_path}", fg="green")
