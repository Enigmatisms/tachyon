"""Unit tests for EvolveSession."""
from tachyon.evolve.config import EvolveConfig
from tachyon.evolve.models import (
    ExperimentRecord,
    ExperimentStatus,
    MetricSnapshot,
)
from tachyon.evolve.session import EvolveSession


class TestEvolveSession:

    def test_start_new_experiment(self) -> None:
        cfg = EvolveConfig(max_iterations=5)
        session = EvolveSession(config=cfg)
        record = session.start_new_experiment()

        assert record.iteration == 0
        assert record.status == ExperimentStatus.PENDING
        assert len(session.experiments) == 1

    def test_complete_iteration(self) -> None:
        cfg = EvolveConfig(max_iterations=5)
        session = EvolveSession(config=cfg)
        session.start_new_experiment()
        session.complete_iteration()

        assert session.current_iteration == 1

    def test_is_finished_max_iterations(self) -> None:
        cfg = EvolveConfig(max_iterations=3)
        session = EvolveSession(config=cfg)
        session.current_iteration = 3

        assert session.is_finished is True

    def test_convergence_does_not_finish_session(self) -> None:
        """has_converged should NOT make is_finished True.

        Convergence triggers strategy switch, not loop exit.
        Only max_iterations stops the loop.
        """
        cfg = EvolveConfig(max_iterations=10)
        session = EvolveSession(config=cfg)
        session.convergence_count = 3  # has_converged is True

        assert session.has_converged is True
        assert session.is_finished is False  # loop continues

    def test_convergence_detection(self) -> None:
        cfg = EvolveConfig(max_iterations=10)
        session = EvolveSession(config=cfg)

        baseline = MetricSnapshot.from_kernel_metrics({
            "gpu__time_duration.sum": 10000000.0,
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 50.0,
        })

        # First iteration: sets baseline
        record0 = ExperimentRecord(
            iteration=0,
            optimized_metrics=MetricSnapshot.from_kernel_metrics({
                "gpu__time_duration.sum": 10000000.0,
                "sm__throughput.avg.pct_of_peak_sustained_elapsed": 50.0,
            }),
        )
        session.record_improvement(record0)
        session.complete_iteration()

        # Next 3 iterations: small improvements (<5%) should trigger convergence
        for i in range(1, 4):
            record = ExperimentRecord(
                iteration=i,
                baseline_metrics=baseline,
                optimized_metrics=MetricSnapshot.from_kernel_metrics({
                    "gpu__time_duration.sum": 9900000.0,
                    "sm__throughput.avg.pct_of_peak_sustained_elapsed": 50.5,
                }),
            )
            session.record_improvement(record)
            session.complete_iteration()

        assert session.convergence_count >= 3
        assert session.has_converged is True

    def test_record_improvement(self) -> None:
        cfg = EvolveConfig(max_iterations=10)
        session = EvolveSession(config=cfg)

        baseline = MetricSnapshot.from_kernel_metrics({
            "gpu__time_duration.sum": 10000000.0,
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 50.0,
        })
        optimized = MetricSnapshot.from_kernel_metrics({
            "gpu__time_duration.sum": 5000000.0,
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 75.0,
        })

        record = ExperimentRecord(
            iteration=0,
            baseline_metrics=baseline,
            optimized_metrics=optimized,
        )
        session.record_improvement(record)

        assert session.best_iteration == 0
        assert session.baseline_metrics is not None

    def test_get_status_summary(self) -> None:
        cfg = EvolveConfig(max_iterations=5)
        session = EvolveSession(config=cfg)
        session.start_new_experiment()
        session.complete_iteration()

        summary = session.get_status_summary()
        assert summary["current_iteration"] == 1
        assert summary["max_iterations"] == 5
        assert summary["experiments_count"] == 1

    def test_failures_do_not_trigger_convergence(self) -> None:
        """Failed iterations (compile error, crash) should NOT count as convergence.

        Convergence means: code works but performance plateaued.
        Failures are LLM mistakes, not evidence of plateau.
        """
        cfg = EvolveConfig(max_iterations=10)
        session = EvolveSession(config=cfg)

        # Simulate 3 failed iterations — convergence_count stays 0
        for _i in range(3):
            session.start_new_experiment()
            # orchestrator no longer increments convergence_count for FAILED
            session.complete_iteration()

        assert session.convergence_count == 0
        assert session.has_converged is False

    def test_regressions_trigger_convergence(self) -> None:
        """3 consecutive regressions (code works, perf bad) = converged."""
        cfg = EvolveConfig(max_iterations=10)
        session = EvolveSession(config=cfg)

        # Set baseline first
        record0 = ExperimentRecord(
            iteration=0,
            optimized_metrics=MetricSnapshot.from_kernel_metrics({
                "gpu__time_duration.sum": 10000000.0,
            }),
        )
        session.record_improvement(record0)
        session.complete_iteration()

        # 3 regressions (improvement < threshold)
        for i in range(1, 4):
            record = ExperimentRecord(
                iteration=i,
                baseline_metrics=session.baseline_metrics,
                optimized_metrics=MetricSnapshot.from_kernel_metrics({
                    "gpu__time_duration.sum": 10100000.0,  # slightly worse
                }),
            )
            session.record_improvement(record)
            session.complete_iteration()

        assert session.convergence_count >= 3
        assert session.has_converged is True

    def test_success_resets_convergence_count(self) -> None:
        """A significant improvement should reset convergence_count."""
        cfg = EvolveConfig(max_iterations=10)
        session = EvolveSession(config=cfg)

        # Simulate 2 regressions (convergence_count = 2)
        record0 = ExperimentRecord(
            iteration=0,
            optimized_metrics=MetricSnapshot.from_kernel_metrics({
                "gpu__time_duration.sum": 10000000.0,
            }),
        )
        session.record_improvement(record0)  # sets baseline
        session.complete_iteration()

        # Two minor regressions
        for i in range(1, 3):
            r = ExperimentRecord(
                iteration=i,
                baseline_metrics=session.baseline_metrics,
                optimized_metrics=MetricSnapshot.from_kernel_metrics({
                    "gpu__time_duration.sum": 10100000.0,
                }),
            )
            session.record_improvement(r)
            session.complete_iteration()

        assert session.convergence_count == 2

        # Significant improvement resets count
        record = ExperimentRecord(
            iteration=0,
            optimized_metrics=MetricSnapshot.from_kernel_metrics({
                "gpu__time_duration.sum": 10000000.0,
            }),
        )
        session.record_improvement(record)  # sets baseline
        session.complete_iteration()

        record2 = ExperimentRecord(
            iteration=1,
            baseline_metrics=session.baseline_metrics,
            optimized_metrics=MetricSnapshot.from_kernel_metrics({
                "gpu__time_duration.sum": 8000000.0,  # 20% improvement
            }),
        )
        session.record_improvement(record2)

        assert session.convergence_count == 0


class TestGlobalBest:
    """Test global best tracking across direction resets."""

    def test_record_direction_best_tracks_global(self) -> None:
        cfg = EvolveConfig(max_iterations=20)
        session = EvolveSession(config=cfg)

        # Simulate direction A: baseline 10ms → best 5ms
        session.baseline_metrics = MetricSnapshot.from_kernel_metrics({
            "gpu__time_duration.sum": 10_000_000.0,
        })
        session.best_metrics = MetricSnapshot.from_kernel_metrics({
            "gpu__time_duration.sum": 5_000_000.0,
        })
        session.best_iteration = 3

        session.record_direction_best("branch-A")

        assert session.global_best_metrics is not None
        assert session.global_best_metrics.duration_ms == 5.0
        assert session.global_best_iteration == 3
        assert session.global_best_branch == "branch-A"

    def test_global_best_keeps_better_direction(self) -> None:
        """Direction B worse than A → global best stays A."""
        cfg = EvolveConfig(max_iterations=20)
        session = EvolveSession(config=cfg)

        session.baseline_metrics = MetricSnapshot.from_kernel_metrics({
            "gpu__time_duration.sum": 10_000_000.0,
        })

        # Direction A: 5ms
        session.best_metrics = MetricSnapshot.from_kernel_metrics({
            "gpu__time_duration.sum": 5_000_000.0,
        })
        session.best_iteration = 3
        session.record_direction_best("branch-A")

        # Direction B: 8ms (worse)
        session.best_metrics = MetricSnapshot.from_kernel_metrics({
            "gpu__time_duration.sum": 8_000_000.0,
        })
        session.best_iteration = 7
        session.record_direction_best("branch-B")

        # Global best should still be direction A
        assert session.global_best_metrics.duration_ms == 5.0
        assert session.global_best_iteration == 3
        assert session.global_best_branch == "branch-A"

    def test_global_best_updates_for_better_direction(self) -> None:
        """Direction C better than A → global best updates to C."""
        cfg = EvolveConfig(max_iterations=20)
        session = EvolveSession(config=cfg)

        session.baseline_metrics = MetricSnapshot.from_kernel_metrics({
            "gpu__time_duration.sum": 10_000_000.0,
        })

        # Direction A: 5ms
        session.best_metrics = MetricSnapshot.from_kernel_metrics({
            "gpu__time_duration.sum": 5_000_000.0,
        })
        session.best_iteration = 3
        session.record_direction_best("branch-A")

        # Direction C: 3ms (better)
        session.best_metrics = MetricSnapshot.from_kernel_metrics({
            "gpu__time_duration.sum": 3_000_000.0,
        })
        session.best_iteration = 9
        session.record_direction_best("branch-C")

        assert session.global_best_metrics.duration_ms == 3.0
        assert session.global_best_iteration == 9
        assert session.global_best_branch == "branch-C"

    def test_global_best_survives_session_reset(self) -> None:
        """Resetting best_metrics (direction switch) doesn't affect global."""
        cfg = EvolveConfig(max_iterations=20)
        session = EvolveSession(config=cfg)

        session.baseline_metrics = MetricSnapshot.from_kernel_metrics({
            "gpu__time_duration.sum": 10_000_000.0,
        })
        session.best_metrics = MetricSnapshot.from_kernel_metrics({
            "gpu__time_duration.sum": 5_000_000.0,
        })
        session.best_iteration = 3
        session.record_direction_best("branch-A")

        # Simulate convergence reset
        session.convergence_count = 0
        session.best_metrics = None
        session.best_iteration = -1

        # Global best untouched
        assert session.global_best_metrics is not None
        assert session.global_best_metrics.duration_ms == 5.0
        assert session.global_best_branch == "branch-A"

    def test_get_global_best_improvement(self) -> None:
        cfg = EvolveConfig(max_iterations=20)
        session = EvolveSession(config=cfg)

        session.baseline_metrics = MetricSnapshot.from_kernel_metrics({
            "gpu__time_duration.sum": 10_000_000.0,
        })
        session.global_best_metrics = MetricSnapshot.from_kernel_metrics({
            "gpu__time_duration.sum": 5_000_000.0,
        })

        improvement = session.get_global_best_improvement()
        assert improvement == 50.0  # 50% faster

    def test_get_global_best_improvement_falls_back_to_current(self) -> None:
        """No global best → falls back to current direction best."""
        cfg = EvolveConfig(max_iterations=20)
        session = EvolveSession(config=cfg)

        session.baseline_metrics = MetricSnapshot.from_kernel_metrics({
            "gpu__time_duration.sum": 10_000_000.0,
        })
        session.best_metrics = MetricSnapshot.from_kernel_metrics({
            "gpu__time_duration.sum": 7_000_000.0,
        })

        improvement = session.get_global_best_improvement()
        assert improvement == 30.0  # 30% faster
