"""Unit tests for evolve data models."""
import pytest

from tachyon.evolve.config import EvolveConfig
from tachyon.evolve.models import (
    CodeChange,
    ExperimentRecord,
    ExperimentStatus,
    MetricSnapshot,
)


class TestEvolveConfig:
    """Tests for EvolveConfig."""

    def test_defaults(self) -> None:
        cfg = EvolveConfig()
        assert cfg.build_cmd is None
        assert cfg.run_cmd is None
        assert cfg.max_iterations == 10
        assert cfg.git_auto_commit is True
        assert cfg.allowed_edit_paths == []

    def test_load_without_file(self, tmp_path: pytest.TempPathFactory) -> None:
        cfg = EvolveConfig.load(config_path=tmp_path / "nonexistent.toml")
        assert cfg.max_iterations == 10

    def test_load_from_toml(self, tmp_path: pytest.TempPathFactory) -> None:
        toml = tmp_path / "evolve.toml"
        toml.write_text(
            '[build]\n'
            'cmd = "make -j8"\n'
            'timeout = 600\n'
            '[run]\n'
            'cmd = "./app --size 1024"\n'
            'max_iterations = 5\n'
            'target_kernel = "matmul"\n'
        )
        cfg = EvolveConfig.load(config_path=toml)
        assert cfg.build_cmd == "make -j8"
        assert cfg.build_timeout == 600
        assert cfg.run_cmd == "./app --size 1024"
        # max_iterations and target_kernel are under [run] section in this TOML
        # but should be found via fallback section search
        assert cfg.max_iterations == 5
        assert cfg.target_kernel == "matmul"

    def test_load_overrides(self, tmp_path: pytest.TempPathFactory) -> None:
        toml = tmp_path / "evolve.toml"
        toml.write_text('max_iterations = 5\n')
        cfg = EvolveConfig.load(config_path=toml, max_iterations=20)
        assert cfg.max_iterations == 20


class TestExperimentStatus:
    """Tests for ExperimentStatus enum."""

    def test_values(self) -> None:
        assert ExperimentStatus.PENDING.value == "PENDING"
        assert ExperimentStatus.SUCCESS.value == "SUCCESS"
        assert ExperimentStatus.ROLLED_BACK.value == "ROLLED_BACK"


class TestCodeChange:
    """Tests for CodeChange."""

    def test_compute_diff(self) -> None:
        old = "line1\nline2\nline3\n"
        new = "line1\nmodified\nline3\n"
        diff = CodeChange.compute_diff(old, new)
        assert "modified" in diff
        assert "-line2" in diff

    def test_compute_lines_changed(self) -> None:
        old = "line1\nline2\nline3\nline4\n"
        new = "line1\nmodified\nline3\nline4\n"
        start, end = CodeChange.compute_lines_changed(old, new)
        assert start == 2
        assert end == 2

    def test_compute_lines_unchanged(self) -> None:
        content = "line1\nline2\n"
        start, end = CodeChange.compute_lines_changed(content, content)
        assert (start, end) == (0, 0)


class TestMetricSnapshot:
    """Tests for MetricSnapshot."""

    def test_from_kernel_metrics(self) -> None:
        metrics = {
            "gpu__time_duration.sum": 5000000.0,  # 5ms (nanoseconds)
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 75.0,
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": 40.0,
        }
        snap = MetricSnapshot.from_kernel_metrics(metrics)
        assert snap.duration_ms == pytest.approx(5.0)
        assert snap.sm_throughput == 75.0
        assert snap.dram_throughput == 40.0

    def test_get_metric(self) -> None:
        metrics = {
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 80.0,
        }
        snap = MetricSnapshot.from_kernel_metrics(metrics)
        assert snap.get("sm__throughput.avg.pct_of_peak_sustained_elapsed") == 80.0
        assert snap.get("sm__throughput.avg") == 80.0
        assert snap.get("nonexistent") is None

    def test_empty_metrics(self) -> None:
        snap = MetricSnapshot.from_kernel_metrics({})
        assert snap.duration_ms is None
        assert snap.sm_throughput is None


class TestExperimentRecord:
    """Tests for ExperimentRecord."""

    def test_compare_success(self) -> None:
        baseline = MetricSnapshot.from_kernel_metrics({
            "gpu__time_duration.sum": 10000000.0,  # 10ms
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 50.0,
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": 30.0,
        })
        optimized = MetricSnapshot.from_kernel_metrics({
            "gpu__time_duration.sum": 8000000.0,  # 8ms (20% faster)
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 60.0,
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": 35.0,
        })

        record = ExperimentRecord(
            iteration=0,
            baseline_metrics=baseline,
            optimized_metrics=optimized,
        )
        comparison = record.compare()

        assert comparison["regressed_count"] == 0
        assert comparison["improved_count"] > 0
        assert comparison["avg_improvement_pct"] > 0

    def test_compare_regression(self) -> None:
        baseline = MetricSnapshot.from_kernel_metrics({
            "gpu__time_duration.sum": 5000000.0,  # 5ms
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 80.0,
        })
        optimized = MetricSnapshot.from_kernel_metrics({
            "gpu__time_duration.sum": 7000000.0,  # 7ms (worse)
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 40.0,  # worse
        })

        record = ExperimentRecord(
            iteration=1,
            baseline_metrics=baseline,
            optimized_metrics=optimized,
        )
        comparison = record.compare()
        assert len(comparison["regressions"]) > 0

    def test_compare_missing_metrics(self) -> None:
        record = ExperimentRecord(iteration=0)
        comparison = record.compare()
        assert "error" in comparison

    def test_default_status(self) -> None:
        record = ExperimentRecord(iteration=0)
        assert record.status == ExperimentStatus.PENDING
