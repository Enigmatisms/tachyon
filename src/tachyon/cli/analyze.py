"""The ``analyze`` subcommand — core M1 deliverable.

Usage examples::

    tachyon analyze report.ncu-rep --no-ai
    tachyon analyze report.ncu-rep --format terminal --kernel "softmax*"
    tachyon analyze report.ncu-rep -o results.md --format markdown
    tachyon analyze report.ncu-rep -v   # verbose: show all findings including INFO
    tachyon analyze report.ncu-rep -q   # quiet: show only CRITICAL findings

Pipeline: load report -> filter kernels -> run analyzers -> severity filter -> render
"""
from __future__ import annotations

import fnmatch
import logging
import sys
from pathlib import Path

import click

from tachyon.cli.main import app
from tachyon.config.settings import TachyonConfig


@app.command()
@click.argument("report_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--format", "fmt",
    type=click.Choice(["terminal", "markdown", "json"]),
    default="terminal",
    help="Output format.",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Write output to file instead of stdout.",
)
@click.option(
    "--no-ai",
    is_flag=True,
    default=False,
    help="Force Rule-Only mode (no LLM). Always true in M1.",
)
@click.option(
    "--kernel", "-k",
    default=None,
    help="Filter kernels by name (supports glob patterns, e.g. 'softmax*').",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show all findings including INFO severity.",
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    help="Show only CRITICAL findings.",
)
def analyze(
    report_path: Path,
    fmt: str,
    output: Path | None,
    no_ai: bool,
    kernel: str | None,
    verbose: bool,
    quiet: bool,
) -> None:
    """Analyze an NCU report file and display performance findings."""
    # Configure logging based on verbosity
    log_level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    # Load config with CLI overrides
    config = TachyonConfig.load()
    config.apply_cli_overrides(format=fmt)

    # ── Step 1: Load report ──
    from tachyon.reader.ncu_reader import NcuReportReader

    reader = NcuReportReader()
    result = reader.load(report_path)
    if not result.success:
        click.secho(f"Error: {result.error.message}", fg="red", err=True)
        click.echo(f"Suggestion: {result.error.suggestion}", err=True)
        sys.exit(1)

    reports = result.data
    assert reports is not None

    # ── Step 2: Filter kernels if --kernel specified ──
    if kernel:
        reports = [
            r for r in reports
            if fnmatch.fnmatch(r.demangled_name, kernel)
            or fnmatch.fnmatch(r.kernel_name, kernel)
        ]
        if not reports:
            click.secho(
                f"No kernels matching '{kernel}' found.", fg="yellow", err=True
            )
            sys.exit(0)

    # ── Step 3: Run analyzers ──
    from tachyon.analyzers.base import AnalyzerRegistry
    from tachyon.models.finding import Severity

    registry = AnalyzerRegistry()
    registry.auto_register()

    all_findings: dict[str, list] = {}
    for report in reports:
        findings = registry.run_all(report)

        # Apply severity filter based on verbosity flags
        if quiet:
            findings = [f for f in findings if f.severity == Severity.CRITICAL]
        elif not verbose:
            findings = [f for f in findings if f.severity != Severity.INFO]

        all_findings[report.demangled_name] = findings

    # ── Step 4: Render output ──
    from tachyon.report.terminal import TerminalReporter

    reporter = TerminalReporter()
    output_text = reporter.render(reports, all_findings)

    if output:
        output.write_text(output_text)
        click.echo(f"Report written to {output}")
    else:
        click.echo(output_text)
