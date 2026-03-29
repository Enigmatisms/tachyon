"""CLI ``profile`` command — the E2E "one command to rule them all".

Usage::

    tachyon profile ./app --size 1024
    tachyon profile --deep ./app --size 1024
    tachyon profile --radical ./app --size 1024
    tachyon profile --kernel matmul_kernel ./app
    tachyon profile --no-ai ./app          # Rule-Only, no LLM

Flow: Smart Profiling (Stage 1→2) → Shared Analysis Pipeline (merge → rules → render → AI).
This is the only command a user needs for the full experience.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click

from tachyon.cli.main import app
from tachyon.config.settings import TachyonConfig


@app.command()
@click.argument("executable")
@click.argument("exe_args", nargs=-1)
@click.option(
    "--deep", "depth", flag_value="deep", default=None,
    help="2-layer analysis: metrics + source attribution.",
)
@click.option(
    "--radical", "depth", flag_value="radical", default=None,
    help="3-layer analysis with aggressive NCU profiling.",
)
@click.option(
    "--kernel", "-k",
    multiple=True,
    help="Only profile specified kernel(s). Repeatable. Supports globs.",
)
@click.option(
    "--top-k",
    type=int,
    default=5,
    help="Number of top kernels for Stage 2 (default: 5).",
)
@click.option(
    "--ncu-args",
    type=str,
    default=None,
    help="Extra arguments to pass to ncu (quoted string).",
)
@click.option(
    "--ncu-set",
    type=click.Choice(["basic", "detailed", "full"]),
    default=None,
    help="Override NCU metric set.",
)
@click.option(
    "--ncu-metrics",
    type=str,
    default=None,
    help="Comma-separated NCU metrics (replaces --set).",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Save outputs to this directory.",
)
@click.option("--no-ai", is_flag=True, help="Skip AI analysis (Rule-Only).")
@click.option("--lang", default=None, help="Language (en/zh).")
@click.option(
    "--model", type=str, default=None, help="LLM model override.",
)
@click.option(
    "--verbose", "-v", is_flag=True, help="Show verbose output.",
)
@click.option(
    "--export", type=click.Path(path_type=Path), default=None,
    help="Export AI analysis to markdown file.",
)
def profile(
    executable: str,
    exe_args: tuple[str, ...],
    depth: str | None,
    kernel: tuple[str, ...],
    top_k: int,
    ncu_args: str | None,
    ncu_set: str | None,
    ncu_metrics: str | None,
    output: Path | None,
    no_ai: bool,
    lang: str | None,
    model: str | None,
    verbose: bool,
    export: Path | None,
) -> None:
    """End-to-end: profile → analyze → report.

    Runs two-stage smart profiling, then feeds results through the
    full analysis pipeline (Rule Engine → Terminal Report → AI Analysis).

    \b
    Examples:
        tachyon profile ./matmul
        tachyon profile --deep ./app --batch 32
        tachyon profile --radical ./app --batch 32
        tachyon profile --kernel "matmul_*" --top-k 3 ./app
        tachyon profile --no-ai ./app
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(log_level)

    config = TachyonConfig.load()
    config.apply_cli_overrides(model=model, depth=depth)
    if lang:
        config.output.lang = lang

    from tachyon.profiler.ncu_profiler import AnalysisDepth, DEPTH_CONFIGS
    from tachyon.profiler.pipeline import run_profiling_pipeline
    from tachyon.utils.progress import console, print_error_panel, print_profile_summary

    analysis_depth = AnalysisDepth(depth) if depth else AnalysisDepth.BASIC
    ncu_strategy, ai_layers = DEPTH_CONFIGS[analysis_depth]

    extra_ncu = ncu_args.split() if ncu_args else None

    # Convert glob patterns to NCU regexes
    kernel_list: list[str] | None = None
    if kernel:
        from tachyon.utils.kernel_filter import to_ncu_regex
        kernel_list = [to_ncu_regex(k) for k in kernel]

    # Display profiling configuration
    print_profile_summary(
        executable=executable,
        depth=analysis_depth.value,
        kernels=list(kernel) if kernel else None,
        ncu_set=ncu_set,
        ncu_metrics=ncu_metrics,
        top_k=top_k,
    )

    # ── Phase 1: Profiling ──
    result = asyncio.run(
        run_profiling_pipeline(
            executable=executable,
            exe_args=list(exe_args),
            config=config,
            strategy=ncu_strategy,
            top_k=top_k,
            kernel_filter=kernel_list,
            output_dir=output,
            extra_ncu_args=extra_ncu,
            metric_set_override=ncu_set,
            metrics_override=ncu_metrics,
            verbose=verbose,
        )
    )

    if not result.success:
        assert result.error is not None
        print_error_panel(
            "Profiling Error",
            result.error.message,
            suggestion=result.error.suggestion,
        )
        sys.exit(1)

    assert result.data is not None
    report_path = result.data

    console.print()
    console.rule("[bold green]Analysis[/bold green]")

    # ── Phase 2: Analysis (shared pipeline) ──
    from tachyon.analysis.pipeline import run_analysis

    run_analysis(
        report_path,
        config,
        kernel_filter=kernel[0] if len(kernel) == 1 else None,
        verbose=verbose,
        no_ai=no_ai,
        output_file=(output / "analysis.txt") if output else None,
        ai_layers=ai_layers,
        ai_output_file=export,
    )

    # ── Depth hint ──
    from tachyon.utils.progress import print_depth_hint
    print_depth_hint(analysis_depth)
