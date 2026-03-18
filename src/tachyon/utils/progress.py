"""Rich-based progress display for NCU profiling stages.

Two modes controlled by ``verbose``:
  - verbose=True:  Print all NCU stderr lines in real time (styled).
  - verbose=False: Show a live spinner with elapsed time; stderr is hidden
                   but captured and available in the result.

Falls back to plain text when Rich is unavailable or output is piped.

Robustness: NcuSpinner wraps all Rich Live interactions in try/except so
that terminal state changes, piped output, or rendering errors never crash
the profiling run. Worst case: the spinner silently degrades to no-op.
"""
from __future__ import annotations

import logging
import re
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Generator

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

_log = logging.getLogger(__name__)

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
    """Live profiling status panel with spinner, progress bar, and timing.

    Visual design (non-verbose mode):
    ┌─ Tachyon Profiler ── Stage 1 ───────────────────┐
    │ ⠹ Connected to target process           [12s]   │
    │   ━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━ 50%   │
    │   Kernel: matmul_kernel  (3 profiled)            │
    └──────────────────────────────────────────────────┘

    Robustness:
      - All Rich rendering wrapped in try/except
      - stop() is idempotent
      - Falls back to plain-text when Rich Live fails
      - try/finally in _run_ncu guarantees cleanup
    """

    def __init__(self, stage: int) -> None:
        self._stage = stage
        self._start = time.monotonic()
        self._status = "Starting NCU..."
        self._phase = "init"  # init → connected → profiling → saving → done
        self._percent: float | None = None  # per-launch 0-100
        self._kernel_name: str | None = None
        self._launch: int = 0  # how many launches started
        self._unique_kernels: set[str] = set()  # distinct kernel names seen
        self._spinner = Spinner("dots", style="bold cyan")
        self._live: Live | None = None
        self._stopped = False
        self._fallback_thread: threading.Thread | None = None
        self._fallback_stop = threading.Event()

    def start(self) -> None:
        """Start the live display. Falls back to plain text on failure."""
        if self._stopped:
            return
        try:
            # Pass self — Live calls __rich_console__ on every refresh,
            # keeping the elapsed timer ticking even without new data.
            self._live = Live(
                self,
                console=console,
                refresh_per_second=8,
                transient=False,
            )
            self._live.start()
        except Exception:
            self._live = None
            _log.debug("Rich Live unavailable, using plain text fallback")
            self._start_fallback()

    def _start_fallback(self) -> None:
        """Background thread: print status every 5 seconds for non-TTY."""
        def _ticker() -> None:
            while not self._fallback_stop.wait(5.0):
                elapsed = time.monotonic() - self._start
                pct = f" {self._percent:.0f}%" if self._percent is not None else ""
                try:
                    print(
                        f"  [Stage {self._stage}] {self._status}{pct}  [{elapsed:.0f}s]",
                        file=sys.stderr, flush=True,
                    )
                except Exception:
                    pass

        self._fallback_thread = threading.Thread(target=_ticker, daemon=True)
        self._fallback_thread.start()

    def stop(self) -> None:
        """Stop and clear the live display. Idempotent."""
        if self._stopped:
            return
        self._stopped = True

        if self._live is not None:
            try:
                # Clear last frame so print_stage_result can follow cleanly
                self._live.update(Text(""))
                self._live.stop()
            except Exception:
                _log.debug("Error stopping Rich Live", exc_info=True)
            self._live = None

        if self._fallback_thread is not None:
            self._fallback_stop.set()
            self._fallback_thread = None

    def update(self, line: str) -> None:
        """Feed an NCU stderr line to update status. State only — no rendering."""
        if self._stopped:
            return
        stripped = line.rstrip()
        if not stripped:
            return
        try:
            self._parse_ncu_line(stripped)
        except Exception:
            pass

    def _parse_ncu_line(self, stripped: str) -> None:
        """Parse NCU stderr and update internal state."""
        # New launch: ==PROF== Profiling "kernel_name": 0%...
        # Each launch resets per-launch progress to 0%.
        if "Profiling" in stripped and '"' in stripped:
            parts = stripped.split('"')
            if len(parts) >= 2 and parts[1]:
                self._kernel_name = parts[1]
                if len(self._kernel_name) > 45:
                    self._kernel_name = self._kernel_name[:42] + "..."
                self._unique_kernels.add(self._kernel_name)
            self._launch += 1
            self._percent = 0.0
            self._phase = "profiling"
            # Don't return — still extract % from this line if present

        # Extract percentage (take the LAST match — NCU puts "0%..50%..100%" on one line)
        all_pcts = _PERCENT_RE.findall(stripped)
        if all_pcts:
            new_pct = float(all_pcts[-1])
            if self._percent is None or new_pct >= self._percent:
                self._percent = new_pct
            self._status = f"Launch {self._launch}: {self._percent:.0f}%"
            self._phase = "profiling"
            return

        if "Connected" in stripped:
            self._status = "Connected to target process"
            self._phase = "connected"
        elif "Profiling" in stripped:
            self._status = f"Launch {self._launch}: profiling..."
            self._phase = "profiling"
        elif "Disconnected" in stripped:
            self._status = "Disconnected, finalizing..."
            self._percent = None
            self._phase = "saving"
        elif "Saving" in stripped or "report" in stripped.lower():
            self._status = "Saving report..."
            self._percent = None
            self._phase = "saving"

    def _build_panel(self) -> Panel:
        """Build a bordered panel with spinner, progress, and kernel info."""
        elapsed = time.monotonic() - self._start

        # Phase indicator dots:  ● ○ ○ ○  or  ● ● ◉ ○
        phases = ["init", "connected", "profiling", "saving"]
        phase_idx = phases.index(self._phase) if self._phase in phases else 0
        dots = []
        for i, p in enumerate(phases):
            if i < phase_idx:
                dots.append("[bold green]\u25cf[/bold green]")  # ● done
            elif i == phase_idx:
                dots.append("[bold cyan]\u25c9[/bold cyan]")  # ◉ current
            else:
                dots.append("[dim]\u25cb[/dim]")  # ○ pending
        phase_bar = " ".join(dots)

        # Row 1: spinner + status + elapsed
        inner = Table.grid(padding=(0, 1))
        inner.add_column(width=2)  # spinner
        inner.add_column(ratio=1)  # status
        inner.add_column(justify="right", width=8)  # timer

        inner.add_row(
            self._spinner,
            Text(self._status),
            Text(f"[{elapsed:.0f}s]", style="dim"),
        )

        # Row 2: progress bar (when percentage available)
        if self._percent is not None:
            bar_grid = Table.grid(padding=(0, 1))
            bar_grid.add_column(width=2)
            bar_grid.add_column(ratio=1)
            bar_grid.add_column(justify="right", width=8)
            pbar = ProgressBar(total=100, completed=self._percent, width=40)
            bar_grid.add_row(
                "",
                pbar,
                Text(f"{self._percent:.0f}%", style="bold green"),
            )
            inner.add_row("", bar_grid, "")

        # Row 3: kernel info + phase dots
        info_parts = []
        if self._kernel_name:
            info_parts.append(f"Kernel: {self._kernel_name}")
        if self._launch > 0:
            nk = len(self._unique_kernels)
            if nk > 1:
                info_parts.append(f"(launch {self._launch}, {nk} kernels)")
            else:
                info_parts.append(f"(launch {self._launch})")
        info_str = "  ".join(info_parts) if info_parts else ""

        info_grid = Table.grid(padding=(0, 1))
        info_grid.add_column(ratio=1)
        info_grid.add_column(justify="right")
        info_grid.add_row(
            Text(info_str, style="dim"),
            Text.from_markup(phase_bar),
        )
        inner.add_row("", info_grid, "")

        return Panel(
            inner,
            title=f"[bold cyan]Tachyon Profiler[/bold cyan] [dim]—[/dim] [bold]Stage {self._stage}[/bold]",
            border_style="cyan",
            expand=False,
            width=min(console.width, 64),
        )

    def __rich_console__(self, rconsole: Console, options: Any) -> Any:
        """Fallback for direct Live(self, ...) usage."""
        try:
            yield self._build_panel()
        except Exception:
            elapsed = time.monotonic() - self._start
            yield Text(f"  Stage {self._stage}: {self._status}  [{elapsed:.0f}s]")

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass


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
