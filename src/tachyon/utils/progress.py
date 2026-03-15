"""Rich-based progress display for NCU profiling stages.

Two modes controlled by ``verbose``:
  - verbose=True:  Print all NCU stderr lines in real time (styled).
  - verbose=False: Show a live spinner with elapsed time; stderr is hidden
                   but captured and available in the result.

Falls back to plain text when Rich is unavailable or output is piped.
"""
from __future__ import annotations

import re
import time
from contextlib import contextmanager
from typing import Any, Generator

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

# Shared console — respects NO_COLOR / piped output
console = Console(highlight=False)


# ── Stage header / result ──────────────────────────────────────────────


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


# ── NCU line printing (verbose mode) ──────────────────────────────────


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
        console.print(f"  [dim]\u2502[/dim] [bold]{stripped}[/bold]")
    elif "error" in stripped.lower():
        console.print(f"  [dim]\u2502[/dim] [red]{stripped}[/red]")
    elif "warning" in stripped.lower():
        console.print(f"  [dim]\u2502[/dim] [yellow]{stripped}[/yellow]")
    else:
        console.print(f"  [dim]\u2502[/dim] [dim]{stripped}[/dim]")


# ── Spinner context (non-verbose mode) ─────────────────────────────────


_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


class NcuSpinner:
    """Live spinner that updates from NCU stderr lines.

    Shows a spinner + last meaningful status line + elapsed time.
    Implements __rich_console__ so Rich Live re-renders every refresh cycle,
    keeping the elapsed timer ticking even when no new NCU output arrives.
    """

    def __init__(self, stage: int) -> None:
        self._stage = stage
        self._start = time.monotonic()
        self._status = "Starting NCU..."
        self._spinner = Spinner("dots", style="cyan")
        self._live: Live | None = None

    def start(self) -> None:
        self._live = Live(
            self,  # pass self — Live calls __rich_console__ on each refresh
            console=console,
            refresh_per_second=8,
            transient=True,
        )
        self._live.start()

    def stop(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None

    def update(self, line: str) -> None:
        """Feed an NCU stderr line to update the spinner status."""
        stripped = line.rstrip()
        if not stripped:
            return

        # Extract meaningful status from NCU output
        pct = _PERCENT_RE.search(stripped)
        if pct:
            self._status = f"Profiling... {pct.group(0)}"
        elif "Connected" in stripped:
            self._status = "Connected to target process"
        elif "Profiling" in stripped:
            # ==PROF== Profiling "kernel_name" ...
            self._status = stripped.replace("==PROF==", "").strip()
            if len(self._status) > 60:
                self._status = self._status[:57] + "..."
        elif "Disconnected" in stripped:
            self._status = "Disconnected, finalizing..."
        elif "Saving" in stripped or "report" in stripped.lower():
            self._status = "Saving report..."

    def __rich_console__(self, console: Console, options: Any) -> Any:
        """Called by Rich Live on every refresh — elapsed time stays fresh."""
        elapsed = time.monotonic() - self._start
        grid = Table.grid(padding=0)
        grid.add_row(
            "  ",
            self._spinner,
            Text(f" Stage {self._stage}: ", style="bold"),
            Text(self._status),
            Text(f"  [{elapsed:.0f}s]", style="dim"),
        )
        yield grid


# ── Error panel ────────────────────────────────────────────────────────


def print_error_panel(title: str, message: str, suggestion: str | None = None) -> None:
    """Print an error in a styled panel."""
    body = f"[red]{message}[/red]"
    if suggestion:
        body += f"\n\n[yellow]Suggestion:[/yellow] {suggestion}"
    console.print(Panel(body, title=f"[red]{title}[/red]", border_style="red", expand=False))


# ── Profile summary ───────────────────────────────────────────────────


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
        table.add_row("Mode", "Two-stage (scan \u2192 deep dive)")

    console.print(table)
    console.print()
