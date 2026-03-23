"""Evolve progress display — Rich Live panel for iteration tracking."""
from __future__ import annotations

import re as _re
import time
from typing import TYPE_CHECKING

from rich.console import Console
from rich.live import Live
from rich.text import Text

from ..utils.progress import console as _shared_console

if TYPE_CHECKING:
    from .models import ExperimentRecord


_ICONS = {
    "PENDING": "\u25cb",
    "HYPOTHESIS": "\u25cf",
    "EDITING": "\u270e",
    "COMPILING": "\u2699",
    "BENCHMARKING": "\u25b6",
    "RUNNING": "\u25b6",
    "PROFILING": "\u25b6",
    "SUCCESS": "\u2713",
    "REGRESSION": "\u2717",
    "FAILED": "\u2717",
    "ROLLED_BACK": "\u21a9",
}

_COLORS = {
    "PENDING": "dim",
    "HYPOTHESIS": "cyan",
    "EDITING": "yellow",
    "COMPILING": "yellow",
    "BENCHMARKING": "green",
    "RUNNING": "cyan",
    "PROFILING": "cyan",
    "SUCCESS": "green",
    "REGRESSION": "red",
    "FAILED": "red",
    "ROLLED_BACK": "yellow",
}


def _strip_markdown(text: str) -> str:
    """Remove markdown bold/italic/backtick markers from text."""
    return _re.sub(r'[*`]+', '', text)


def _get_method(record: ExperimentRecord) -> str:
    """Get optimization summary: LLM summary first, else hypothesis, else code_changes."""
    if record.summary:
        return _strip_markdown(record.summary)
    if record.hypothesis:
        lines = [l.strip() for l in record.hypothesis.split("\n")
                   if l.strip() and not l.strip().startswith("#")]
        if lines:
            return _strip_markdown("\n".join(lines))
    if record.code_changes:
        names = sorted({str(c.file).split("/")[-1] for c in record.code_changes})
        if len(names) == 1:
            return f"Edited {names[0]}"
        return f"Edited {', '.join(names[:3])}"
    return ""


def _format_gpu(record: ExperimentRecord) -> str:
    """Format GPU time with baseline comparison.

    Shows: current_ms (base: original_ms, +X%)
    """
    if record.optimized_metrics is None:
        return ""
    gpu = record.optimized_metrics.duration_ms
    if gpu is None:
        return ""

    base = record.baseline_metrics.duration_ms if record.baseline_metrics else None
    parts = [f"{gpu:.2f}ms"]
    if base is not None:
        pct = (base - gpu) / base * 100.0 if base > 0 else 0.0
        parts.append(f"(base: {base:.2f}ms, {pct:+.1f}%)")
    return " - " + " ".join(parts)


class EvolveProgressDisplay:
    """Rich Live progress display for the evolve orchestrator.

    Renders as simple multi-line text (no Panel/Table) to avoid visual
    conflicts with other Rich Live instances.
    """

    def __init__(
        self,
        max_iterations: int = 10,
        console: Console | None = None,
    ) -> None:
        self._console = console or _shared_console
        self._max_iterations = max_iterations
        self._current_iteration = 0
        self._current_status = "PENDING"
        self._current_tool = ""
        self._timer_start = time.monotonic()
        self._results: list[dict] = []
        self._live: Live | None = None

    def start(self) -> None:
        self._live = Live(
            self,
            console=self._console,
            refresh_per_second=4,
            transient=True,
        )
        self._live.start()

    def stop(self) -> None:
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None

    def set_status(self, status: str, tool: str = "") -> None:
        self._current_status = status
        self._current_tool = tool
        if self._live:
            self._live.refresh()

    def add_result(self, record) -> None:
        self._results.append({"record": record})
        self._current_iteration = record.iteration + 1
        self._timer_start = time.monotonic()

    def notify(self, text: str) -> None:
        """Print through the console (stops Live first to prevent duplicate render)."""
        if self._live:
            self._live.stop()
            self._live = None
        self._console.print(text)

    def render_summary(self, experiments: list) -> None:
        """Print final results as multi-line summary."""
        self._console.print()
        self._console.print("[bold cyan]Evolve Results[/bold cyan]")
        for e in experiments:
            for line in _render_result(e):
                if line.spans:
                    self._console.print(line)
            self._console.print()

    def __rich_console__(self, rconsole: Console, options):
        """Render results + status footer for Rich Live."""
        # Result lines first
        for r in self._results[-8:]:
            for line in _render_result(r["record"], compact=True):
                yield line

        # Status footer (always last line)
        footer = Text()
        footer.append("Tachyon Evolve", style="bold cyan")
        footer.append(f"  |  Iter {self._current_iteration}/{self._max_iterations}")
        elapsed = time.monotonic() - self._timer_start
        footer.append(f"  |  {elapsed:.0f}s", style="dim")
        if self._current_tool:
            icon = _ICONS.get(self._current_status, "\u25cb")
            color = _COLORS.get(self._current_status, "cyan")
            footer.append("  |  ")
            footer.append(f"{icon} {self._current_status}", style=color)
            footer.append(f": {self._current_tool}", style="dim")
        yield footer


def _render_result(record, *, compact: bool = False) -> list[Text]:
    """Render one iteration result as multi-line Text objects.

    compact=True truncates the optimization summary for Live display.
    """
    icon = _ICONS.get(record.status.value, "\u25cb")
    color = _COLORS.get(record.status.value, "dim")

    # Line 1: #N icon STATUS  Decision - GPU Time
    main = Text()
    main.append(f"  #{record.iteration} ", style="dim")
    main.append(f"{icon} {record.status.value}", style=color)
    if record.decision:
        main.append(f"  {record.decision}")
    gpu_str = _format_gpu(record)
    if gpu_str:
        main.append(gpu_str)

    lines: list[Text] = [main]

    # Line 2+: Optimization summary
    method = _get_method(record)
    if method:
        if compact:
            method = _truncate(method, 500)
        detail = Text()
        detail.append("    Optimization: ", style="dim")
        detail.append(method, style="dim")
        lines.append(detail)

    return lines


def _truncate(text: str, limit: int) -> str:
    """Truncate text at a word boundary."""
    if len(text) <= limit:
        return text
    # Find the last space before the limit
    cut = text.rfind(" ", 0, limit)
    if cut <= limit // 2:
        cut = limit  # no good word boundary, hard cut
    return text[:cut] + "..."
