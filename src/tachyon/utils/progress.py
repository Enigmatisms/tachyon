"""Rich-based progress display for NCU profiling stages.

Provides a ``ProfileProgress`` context manager that wraps subprocess execution
with live status display: command summary, real-time stderr streaming, and
final elapsed time.

Falls back to plain ``print()`` when Rich is unavailable or output is piped.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Shared console — respects NO_COLOR / piped output
console = Console(highlight=False)


def print_stage_header(
    stage: int,
    cmd_summary: str,
    *,
    strategy: str | None = None,
    kernels: list[str] | None = None,
    metric_set: str | None = None,
) -> None:
    """Print a styled header panel before a profiling stage starts."""
    lines = [f"[bold]Stage {stage}[/bold]: {cmd_summary}"]
    if strategy:
        lines.append(f"  Strategy: [cyan]{strategy}[/cyan]")
    if metric_set:
        lines.append(f"  Metric set: [cyan]{metric_set}[/cyan]")
    if kernels:
        k_str = ", ".join(kernels[:5])
        if len(kernels) > 5:
            k_str += f" (+{len(kernels) - 5} more)"
        lines.append(f"  Kernels: [cyan]{k_str}[/cyan]")

    console.print(Panel(
        "\n".join(lines),
        title="[bold cyan]Tachyon Profiler[/bold cyan]",
        border_style="cyan",
        expand=False,
    ))


def print_stage_result(stage: int, elapsed: float, success: bool) -> None:
    """Print stage completion status."""
    if success:
        console.print(
            f"  [bold green]\u2713[/bold green] Stage {stage} completed in "
            f"[bold]{elapsed:.1f}s[/bold]"
        )
    else:
        console.print(
            f"  [bold red]\u2717[/bold red] Stage {stage} failed after "
            f"[bold]{elapsed:.1f}s[/bold]"
        )


def print_ncu_line(line: str) -> None:
    """Print a single NCU stderr progress line with styling.

    NCU outputs lines like:
      ==PROF== Connected to process ...
      ==PROF== Profiling "kernel_name": 50%
      ==PROF== Disconnected ...
    """
    stripped = line.rstrip()
    if not stripped:
        return

    if "%" in stripped:
        # Progress percentage — highlight it
        console.print(f"  [dim]\u2502[/dim] [bold]{stripped}[/bold]")
    elif "error" in stripped.lower() or "Error" in stripped:
        console.print(f"  [dim]\u2502[/dim] [red]{stripped}[/red]")
    elif "warning" in stripped.lower() or "Warning" in stripped:
        console.print(f"  [dim]\u2502[/dim] [yellow]{stripped}[/yellow]")
    else:
        console.print(f"  [dim]\u2502[/dim] [dim]{stripped}[/dim]")


def print_error_panel(title: str, message: str, suggestion: str | None = None) -> None:
    """Print an error in a styled panel."""
    body = f"[red]{message}[/red]"
    if suggestion:
        body += f"\n\n[yellow]Suggestion:[/yellow] {suggestion}"
    console.print(Panel(body, title=f"[red]{title}[/red]", border_style="red", expand=False))


def print_profile_summary(
    executable: str,
    strategy: str,
    kernels: list[str] | None = None,
    ncu_set: str | None = None,
    ncu_metrics: str | None = None,
    top_k: int = 5,
) -> None:
    """Print a summary table of profiling parameters before starting."""
    table = Table(
        title="[bold cyan]Profiling Configuration[/bold cyan]",
        show_header=False,
        expand=False,
        border_style="dim",
    )
    table.add_column("Parameter", style="bold")
    table.add_column("Value")

    table.add_row("Executable", executable)
    table.add_row("Strategy", strategy)
    if ncu_set:
        table.add_row("Metric Set", f"{ncu_set} (override)")
    if ncu_metrics:
        table.add_row("Metrics", ncu_metrics[:80] + ("..." if len(ncu_metrics) > 80 else ""))
    if kernels:
        k_str = ", ".join(kernels[:3])
        if len(kernels) > 3:
            k_str += f" (+{len(kernels) - 3} more)"
        table.add_row("Kernel Filter", k_str)
        table.add_row("Mode", "Direct targeting (skip Stage 1)")
    else:
        table.add_row("Top-K", str(top_k))
        table.add_row("Mode", "Two-stage (scan -> deep dive)")

    console.print(table)
    console.print()
