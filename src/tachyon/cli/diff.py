"""CLI diff command — compare two NCU reports.

Usage examples::

    tachyon diff before.ncu-rep after.ncu-rep
    tachyon diff before.ncu-rep after.ncu-rep --kernel "matmul*"
    tachyon diff before.ncu-rep after.ncu-rep --threshold 10
    tachyon diff before.ncu-rep after.ncu-rep -v
"""
from __future__ import annotations

import fnmatch
import sys
from pathlib import Path

import click

from tachyon.cli.main import app


@app.command()
@click.argument("before_path", type=click.Path(exists=True, path_type=Path))
@click.argument("after_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--kernel", "-k",
    default=None,
    help="Filter kernels by name (glob pattern).",
)
@click.option(
    "--threshold",
    type=float,
    default=5.0,
    help="Significance threshold in percent (default: 5.0).",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show all metrics, not just significant changes.",
)
def diff(
    before_path: Path,
    after_path: Path,
    kernel: str | None,
    threshold: float,
    verbose: bool,
) -> None:
    """Compare two NCU report files and show metric differences."""
    from tachyon.diff.differ import ProfileDiffer
    from tachyon.reader.ncu_reader import NcuReportReader

    reader = NcuReportReader()

    # Load both reports
    before_result = reader.load(before_path)
    if not before_result.success:
        click.secho(
            f"Error loading {before_path}: {before_result.error.message}",
            fg="red",
            err=True,
        )
        sys.exit(1)

    # Need a fresh reader instance for second file
    reader2 = NcuReportReader()
    after_result = reader2.load(after_path)
    if not after_result.success:
        click.secho(
            f"Error loading {after_path}: {after_result.error.message}",
            fg="red",
            err=True,
        )
        sys.exit(1)

    before_kernels = before_result.data
    after_kernels = after_result.data

    # Optional kernel filter
    if kernel:
        before_kernels = [
            k for k in before_kernels
            if fnmatch.fnmatch(k.demangled_name, kernel)
            or fnmatch.fnmatch(k.kernel_name, kernel)
        ]
        after_kernels = [
            k for k in after_kernels
            if fnmatch.fnmatch(k.demangled_name, kernel)
            or fnmatch.fnmatch(k.kernel_name, kernel)
        ]

    # Diff
    differ = ProfileDiffer()
    diffs = differ.diff(before_kernels, after_kernels)

    if not diffs:
        click.echo("No common kernels found between the two reports.")
        sys.exit(0)

    # Output
    click.secho(
        f"Comparing: {before_path.name} \u2192 {after_path.name}",
        fg="cyan",
        bold=True,
    )
    click.echo(f"Common kernels: {len(diffs)}\n")

    for d in diffs:
        changes = (
            d.metric_deltas
            if verbose
            else [m for m in d.metric_deltas if abs(m.delta_pct) > threshold]
        )
        regs = d.regressions

        color = "red" if regs else ("yellow" if changes else "green")
        click.secho(f"  {d.demangled_name}", fg=color, bold=True)

        if not changes:
            click.echo("    No significant changes")
            continue

        for m in changes:
            arrow = "\u2191" if m.delta > 0 else "\u2193"
            fg = None
            # Color regressions red
            higher_is_better = any(
                kw in m.name
                for kw in ["throughput", "pct", "utilization", "hit_rate"]
            )
            if higher_is_better and m.delta < 0 and abs(m.delta_pct) > threshold:
                fg = "red"
            elif not higher_is_better and m.delta > 0 and abs(m.delta_pct) > threshold:
                fg = "red"
            elif abs(m.delta_pct) > threshold:
                fg = "green"

            click.secho(
                f"    {m.name}: {m.before:.2f} \u2192 {m.after:.2f} "
                f"({arrow}{abs(m.delta_pct):.1f}%)",
                fg=fg,
            )

    # Summary
    total_regs = sum(len(d.regressions) for d in diffs)
    if total_regs:
        click.secho(
            f"\n\u26a0 {total_regs} regressions detected across {len(diffs)} kernels",
            fg="red",
        )
    else:
        click.secho(
            f"\n\u2713 No regressions detected across {len(diffs)} kernels",
            fg="green",
        )
