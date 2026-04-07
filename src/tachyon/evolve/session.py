"""Evolve session — iterative experiment state management.

Tracks experiments across iterations, baseline/best metrics, and
convergence detection. Session is in-memory only — no persistence.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .config import EvolveConfig
from .models import ExperimentRecord, ExperimentStatus, MetricSnapshot

_log = logging.getLogger(__name__)

_CONVERGENCE_THRESHOLD = 5.0  # percent


@dataclass
class EvolveSession:
    """Manages state across multiple evolve iterations."""

    config: EvolveConfig
    experiments: list[ExperimentRecord] = field(default_factory=list)
    current_iteration: int = 0
    baseline_metrics: MetricSnapshot | None = None
    best_metrics: MetricSnapshot | None = None
    best_iteration: int = -1
    convergence_count: int = 0
    # Global best across all directions (survives convergence resets)
    global_best_metrics: MetricSnapshot | None = None
    global_best_iteration: int = -1
    global_best_branch: str | None = None

    @property
    def has_converged(self) -> bool:
        return self.convergence_count >= self.config.convergence_threshold

    @property
    def is_finished(self) -> bool:
        return self.current_iteration >= self.config.max_iterations

    def start_new_experiment(self) -> ExperimentRecord:
        record = ExperimentRecord(
            iteration=self.current_iteration,
            status=ExperimentStatus.PENDING,
        )
        self.experiments.append(record)
        return record

    def record_improvement(self, record: ExperimentRecord) -> None:
        if record.optimized_metrics is None:
            return

        if self.baseline_metrics is None:
            self.baseline_metrics = record.optimized_metrics
            self.best_metrics = record.optimized_metrics
            self.best_iteration = record.iteration
            return

        comparison = record.compare()
        improvement = comparison.get("avg_improvement_pct", 0.0)

        if improvement > 0:
            if self.best_metrics is not None:
                best_comparison = _quick_compare(
                    self.best_metrics, record.optimized_metrics,
                )
                if best_comparison > 0:
                    self.best_metrics = record.optimized_metrics
                    self.best_iteration = record.iteration
            else:
                self.best_metrics = record.optimized_metrics
                self.best_iteration = record.iteration

            if improvement < _CONVERGENCE_THRESHOLD:
                self.convergence_count += 1
            else:
                self.convergence_count = 0
        else:
            self.convergence_count += 1

    def complete_iteration(self) -> None:
        self.current_iteration += 1

    def record_direction_best(self, branch: str) -> None:
        """Record current direction's best as a candidate for global best."""
        if self.best_metrics is None:
            return
        if self.global_best_metrics is None or (
            self.best_metrics.duration_ms is not None
            and self.global_best_metrics.duration_ms is not None
            and self.best_metrics.duration_ms < self.global_best_metrics.duration_ms
        ):
            self.global_best_metrics = self.best_metrics
            self.global_best_iteration = self.best_iteration
            self.global_best_branch = branch

    def get_best_improvement(self) -> float:
        """Improvement of best over baseline, in percent."""
        if self.baseline_metrics is None or self.best_metrics is None:
            return 0.0
        return _quick_compare(self.baseline_metrics, self.best_metrics)

    def get_global_best_improvement(self) -> float:
        """Global best improvement over baseline, across all directions."""
        target = self.global_best_metrics or self.best_metrics
        if self.baseline_metrics is None or target is None:
            return 0.0
        return _quick_compare(self.baseline_metrics, target)

    def get_status_summary(self) -> dict[str, Any]:
        best_improvement = self.get_best_improvement()
        global_improvement = self.get_global_best_improvement()

        return {
            "current_iteration": self.current_iteration,
            "max_iterations": self.config.max_iterations,
            "has_converged": self.has_converged,
            "is_finished": self.is_finished,
            "experiments_count": len(self.experiments),
            "experiments_summary": [
                {
                    "iteration": e.iteration,
                    "status": e.status.value,
                    "hypothesis": e.hypothesis[:80] if e.hypothesis else "",
                    "decision": e.decision[:80] if e.decision else "",
                }
                for e in self.experiments
            ],
            "best_iteration": self.best_iteration,
            "best_improvement_pct": round(best_improvement, 2),
            "global_best_iteration": self.global_best_iteration,
            "global_best_improvement_pct": round(global_improvement, 2),
        }


def _quick_compare(
    baseline: MetricSnapshot,
    optimized: MetricSnapshot,
) -> float:
    if baseline.duration_ms and optimized.duration_ms:
        if baseline.duration_ms > 0:
            return ((baseline.duration_ms - optimized.duration_ms)
                    / baseline.duration_ms * 100.0)
    if baseline.sm_throughput and optimized.sm_throughput:
        if baseline.sm_throughput > 0:
            return ((optimized.sm_throughput - baseline.sm_throughput)
                    / baseline.sm_throughput * 100.0)
    return 0.0
