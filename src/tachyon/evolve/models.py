"""Experiment data models for evolve iterations.

Core types:
  ExperimentStatus — lifecycle states
  CodeChange — single file edit record
  MetricSnapshot — performance metrics at a point in time
  ExperimentRecord — full record of one evolve iteration
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ExperimentStatus(str, Enum):
    """Lifecycle states of a single evolve iteration."""

    PENDING = "PENDING"
    HYPOTHESIS = "HYPOTHESIS"
    EDITING = "EDITING"
    COMPILING = "COMPILING"
    RUNNING = "RUNNING"
    PROFILING = "PROFILING"
    SUCCESS = "SUCCESS"
    REGRESSION = "REGRESSION"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"


# --- Default comparison metrics ---
_DEFAULT_METRICS = [
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
]

# Improvement threshold: changes < 2% are considered UNCHANGED
_IMPROVEMENT_THRESHOLD = 2.0


@dataclass
class CodeChange:
    """Record of a single file edit."""

    file: str
    diff: str
    lines_changed: tuple[int, int]  # (start_line, end_line)

    @staticmethod
    def compute_diff(old: str, new: str) -> str:
        """Compute unified diff between old and new content."""
        import difflib

        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)
        diff = difflib.unified_diff(
            old_lines, new_lines,
            fromfile="original", tofile="modified", lineterm="",
        )
        return "".join(diff)

    @staticmethod
    def compute_lines_changed(old: str, new: str) -> tuple[int, int]:
        """Estimate the range of lines changed."""
        import difflib

        old_lines = old.splitlines()
        new_lines = new.splitlines()
        sm = difflib.SequenceMatcher(None, old_lines, new_lines)
        changes = sm.get_opcodes()

        start = len(old_lines)
        end = 0
        for tag, i1, i2, j1, j2 in changes:
            if tag in ("replace", "insert", "delete"):
                start = min(start, i1)
                end = max(end, i2)

        if start >= end and start == len(old_lines):
            return (0, 0)
        return (start + 1, end)  # 1-based


@dataclass
class MetricSnapshot:
    """Performance metrics at a point in time."""

    timestamp: str
    duration_ms: float | None = None
    sm_throughput: float | None = None
    dram_throughput: float | None = None
    occupancy: float | None = None
    custom_metrics: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_kernel_metrics(
        cls,
        metrics: dict[str, float],
        label: str = "",
    ) -> MetricSnapshot:
        """Build a MetricSnapshot from a flat metric dict."""
        now = datetime.now().isoformat(timespec="seconds")
        raw: dict[str, float] = {}

        for k, v in metrics.items():
            name = k.replace(".pct_of_peak_sustained_elapsed", "")
            raw[name] = v

        # Extract known fields (gpu__time_duration.sum is in nanoseconds)
        duration = raw.get("gpu__time_duration.sum")
        sm = raw.get("sm__throughput.avg")
        dram = raw.get("gpu__dram_throughput.avg")
        occ = raw.get("sm__warps_active.avg")

        return cls(
            timestamp=now,
            duration_ms=duration / 1e6 if duration is not None else None,
            sm_throughput=sm,
            dram_throughput=dram,
            occupancy=occ,
            raw=raw,
        )

    def get(self, metric_name: str) -> float | None:
        """Get a metric by name (with and without suffix)."""
        if metric_name in self.raw:
            return self.raw[metric_name]
        # Try with suffix stripped
        stripped = metric_name.replace(".pct_of_peak_sustained_elapsed", "")
        if stripped in self.raw:
            return self.raw[stripped]
        return None


@dataclass
class ExperimentRecord:
    """Full record of one evolve iteration."""

    iteration: int
    status: ExperimentStatus = ExperimentStatus.PENDING
    hypothesis: str = ""
    summary: str = ""
    code_changes: list[CodeChange] = field(default_factory=list)
    build_log: str = ""
    run_output: str = ""
    run_exit_code: int | None = None
    baseline_metrics: MetricSnapshot | None = None
    optimized_metrics: MetricSnapshot | None = None
    ncu_rep_path: Path | None = None
    git_commit_hash: str | None = None
    comparison: dict[str, Any] | None = None
    decision: str = ""
    elapsed_sec: float = 0.0
    tool_call_count: int = 0

    def compare(self, metrics: list[str] | None = None) -> dict[str, Any]:
        """Compare baseline vs optimized metrics.

        Returns a comparison dict with per-metric analysis and summary.
        """
        if self.baseline_metrics is None or self.optimized_metrics is None:
            return {"error": "Missing baseline or optimized metrics"}

        if metrics is None:
            metrics = _DEFAULT_METRICS

        rows: list[dict[str, Any]] = []
        improvements: list[float] = []
        regressions: list[str] = []

        for m in metrics:
            base_val = self.baseline_metrics.get(m)
            opt_val = self.optimized_metrics.get(m)

            if base_val is None or opt_val is None:
                rows.append({
                    "metric": m,
                    "baseline": base_val,
                    "optimized": opt_val,
                    "change_pct": None,
                    "status": "N/A",
                })
                continue

            if base_val == 0:
                change_pct = 100.0 if opt_val > 0 else 0.0
            else:
                # For throughput metrics, higher is better (positive improvement)
                # For duration metrics, lower is better (negative = improvement)
                change_pct = ((opt_val - base_val) / base_val) * 100.0

            is_duration = "duration" in m.lower() or "time" in m.lower()

            if abs(change_pct) < _IMPROVEMENT_THRESHOLD:
                status = "UNCHANGED"
            elif is_duration:
                # Lower duration = improvement
                status = "IMPROVED" if change_pct < 0 else "REGRESSED"
            else:
                # Higher throughput = improvement
                status = "IMPROVED" if change_pct > 0 else "REGRESSED"

            if status == "IMPROVED":
                improvements.append(abs(change_pct))
            elif status == "REGRESSED":
                regressions.append(m)

            rows.append({
                "metric": m,
                "baseline": base_val,
                "optimized": opt_val,
                "change_pct": round(change_pct, 2),
                "status": status,
            })

        avg_improvement = (
            sum(improvements) / len(improvements) if improvements else 0.0
        )

        self.comparison = {
            **(self.comparison or {}),
            "rows": rows,
            "avg_improvement_pct": round(avg_improvement, 2),
            "regressions": regressions,
            "improved_count": len(improvements),
            "regressed_count": len(regressions),
        }

        return self.comparison
