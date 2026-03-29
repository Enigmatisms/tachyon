"""Lightweight phase timer for evolve profiling.

Accumulates wall-clock time per named phase.  Completely no-op when
disabled — zero overhead in production.

Usage::

    timer = DebugTimer(enabled=True)

    with timer.phase("compile"):
        subprocess.run(...)

    timer.record("llm_api", 12.3)

    print(timer.format_report(wall_time=120.0))
"""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager


class DebugTimer:
    """Accumulates wall-clock time per named phase."""

    __slots__ = ("enabled", "_totals", "_counts")

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)

    @contextmanager
    def phase(self, name: str):
        """Context manager to time a named phase."""
        if not self.enabled:
            yield
            return
        t0 = time.monotonic()
        try:
            yield
        finally:
            self._totals[name] += time.monotonic() - t0
            self._counts[name] += 1

    def record(self, name: str, duration: float) -> None:
        """Record a duration directly."""
        if not self.enabled:
            return
        self._totals[name] += duration
        self._counts[name] += 1

    def format_report(self, wall_time: float | None = None) -> str:
        """Format sorted phase breakdown.

        Returns a multi-line string ready for console output.
        """
        if not self._totals:
            return ""

        lines: list[str] = []
        tracked = sum(self._totals.values())

        for name, dur in sorted(self._totals.items(), key=lambda x: -x[1]):
            count = self._counts[name]
            avg = dur / count if count > 0 else 0
            pct_str = ""
            if wall_time and wall_time > 0:
                pct_str = f"({dur / wall_time * 100:>5.1f}%)"
            lines.append(
                f"  {name:<30s} {dur:>8.1f}s  {pct_str:>8s}  "
                f"x{count:<3d} avg {avg:.1f}s"
            )

        if wall_time is not None:
            untracked = wall_time - tracked
            if untracked > 0.5:
                pct = untracked / wall_time * 100 if wall_time > 0 else 0
                lines.append(
                    f"  {'(untracked)':<30s} {untracked:>8.1f}s  "
                    f"({pct:>5.1f}%)"
                )
            lines.append(f"  {'TOTAL':<30s} {wall_time:>8.1f}s")

        return "\n".join(lines)
