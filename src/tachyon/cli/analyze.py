"""CLI ``analyze`` command — offline analysis of existing .ncu-rep files.

Usage::

    tachyon analyze report.ncu-rep
    tachyon analyze report.ncu-rep --kernel "softmax*" -v
    tachyon analyze report.ncu-rep --no-ai -o results.txt
    tachyon analyze report.ncu-rep -q        # CRITICAL only

Flow: Load .ncu-rep → Shared Analysis Pipeline (merge → rules → render → AI).
Same analysis quality as ``tachyon profile``, but without the profiling step.
"""
from __future__ import annotations

import logging
from pathlib import Path

import click

from tachyon.cli.main import app
from tachyon.config.settings import TachyonConfig


@app.command()
@click.argument("report_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Write output to file instead of stdout.",
)
@click.option("--no-ai", is_flag=True, help="Skip AI analysis (Rule-Only).")
@click.option(
    "--kernel", "-k",
    default=None,
    help="Filter kernels by name (glob pattern, e.g. 'softmax*').",
)
@click.option(
    "--model", type=str, default=None, help="LLM model override.",
)
@click.option(
    "--verbose", "-v", is_flag=True, help="Show all findings including INFO.",
)
@click.option(
    "--quiet", "-q", is_flag=True, help="Show only CRITICAL findings.",
)
def analyze(
    report_path: Path,
    output: Path | None,
    no_ai: bool,
    kernel: str | None,
    model: str | None,
    verbose: bool,
    quiet: bool,
) -> None:
    """Analyze an existing NCU report file.

    Loads a .ncu-rep file and runs the full analysis pipeline:
    Rule Engine → Terminal Report → AI Analysis (optional).

    \b
    Examples:
        tachyon analyze report.ncu-rep
        tachyon analyze report.ncu-rep --kernel "matmul*"
        tachyon analyze report.ncu-rep --no-ai -v
    """
    log_level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    config = TachyonConfig.load()
    if model:
        config.apply_cli_overrides(model=model)

    from tachyon.analysis.pipeline import run_analysis

    run_analysis(
        report_path,
        config,
        kernel_filter=kernel,
        verbose=verbose,
        quiet=quiet,
        no_ai=no_ai,
        output_file=output,
    )
