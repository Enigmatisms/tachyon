"""CLI profile command -- end-to-end profiling pipeline.

Usage::

    tachyon profile ./app --size 1024
    tachyon profile --strategy radical ./app --size 1024
    tachyon profile --kernel matmul_kernel ./app
    tachyon profile --ncu-args "--replay-mode application" ./app
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
    "--strategy",
    type=click.Choice(["conservative", "radical"]),
    default=None,
    help="Profiling strategy (default: from config).",
)
@click.option(
    "--kernel",
    multiple=True,
    help="Only profile specified kernel(s). Repeatable.",
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
    help="Override NCU metric set (overrides --strategy).",
)
@click.option(
    "--ncu-metrics",
    type=str,
    default=None,
    help="Comma-separated NCU metrics (replaces --set, e.g. 'sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed').",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Save .ncu-rep files to this directory.",
)
@click.option(
    "--no-ai",
    is_flag=True,
    help="Skip AI-enhanced analysis (Rule-Only).",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["terminal", "markdown", "json"]),
    default="terminal",
    help="Output format.",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="LLM model override.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show verbose output.",
)
def profile(
    executable: str,
    exe_args: tuple[str, ...],
    strategy: str | None,
    kernel: tuple[str, ...],
    top_k: int,
    ncu_args: str | None,
    ncu_set: str | None,
    ncu_metrics: str | None,
    output: Path | None,
    no_ai: bool,
    fmt: str,
    model: str | None,
    verbose: bool,
) -> None:
    """End-to-end: profile executable -> analyze -> report.

    Runs two-stage smart profiling, then feeds results through the
    full analysis pipeline (Analyzers -> OptTree -> Report).

    \b
    Examples:
        tachyon profile ./matmul
        tachyon profile --strategy radical ./app --batch 32
        tachyon profile --kernel "matmul_*" --top-k 3 ./app
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    config = TachyonConfig.load()
    config.apply_cli_overrides(model=model, format=fmt, strategy=strategy)

    from tachyon.profiler.ncu_profiler import ProfilingStrategy
    from tachyon.profiler.pipeline import run_e2e_pipeline
    from tachyon.utils.progress import console, print_error_panel, print_profile_summary

    extra_ncu = ncu_args.split() if ncu_args else None
    strat = ProfilingStrategy(strategy) if strategy else None

    # Convert glob patterns to NCU regexes for --kernel-name
    kernel_list: list[str] | None = None
    if kernel:
        from tachyon.utils.kernel_filter import to_ncu_regex
        kernel_list = [to_ncu_regex(k) for k in kernel]

    # Display profiling configuration summary
    print_profile_summary(
        executable=executable,
        strategy=strategy or config.profiling.strategy,
        kernels=list(kernel) if kernel else None,
        ncu_set=ncu_set,
        ncu_metrics=ncu_metrics,
        top_k=top_k,
    )

    result = asyncio.run(
        run_e2e_pipeline(
            executable=executable,
            exe_args=list(exe_args),
            config=config,
            strategy=strat,
            top_k=top_k,
            kernel_filter=kernel_list,
            use_ai=not no_ai,
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
            "Pipeline Error",
            result.error.message,
            suggestion=result.error.suggestion,
        )
        sys.exit(1)

    assert result.data is not None
    report_path = result.data

    console.print()
    console.rule("[bold green]Analysis[/bold green]")

    # --- Run analysis on the profiled report ---
    _analyze_report(report_path, config, fmt, verbose, no_ai, output)


def _analyze_report(
    report_path: Path,
    config: TachyonConfig,
    fmt: str,
    verbose: bool,
    no_ai: bool,
    output_dir: Path | None,
) -> None:
    """Run the analysis pipeline on a .ncu-rep file."""
    from tachyon.reader.ncu_reader import NcuReportReader

    reader = NcuReportReader()
    load_result = reader.load(report_path)
    if not load_result.success:
        assert load_result.error is not None
        click.secho(f"Error loading report: {load_result.error.message}", fg="red", err=True)
        sys.exit(1)

    assert load_result.data is not None
    reports = load_result.data

    from tachyon.analyzers.base import AnalyzerRegistry
    from tachyon.models.finding import Severity

    registry = AnalyzerRegistry()
    registry.auto_register()

    all_findings: dict[str, list] = {}
    for report in reports:
        findings = registry.run_all(report)
        if not verbose:
            findings = [f for f in findings if f.severity != Severity.INFO]
        all_findings[report.demangled_name] = findings

    from tachyon.report.terminal import TerminalReporter

    reporter = TerminalReporter()
    output_text = reporter.render(reports, all_findings)

    if output_dir:
        out_file = output_dir / "analysis.txt"
        out_file.write_text(output_text)
        click.echo(f"Analysis written to {out_file}")
    else:
        click.echo(output_text)

    click.secho(f"\nProfile report: {report_path}", fg="green")
