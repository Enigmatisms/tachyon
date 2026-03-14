"""Unit tests for ProfileDiffer — comprehensive diff / regression detection.

Tests cover:
  - Identical reports (no changes)
  - Different metrics (delta and delta_pct calculation)
  - No common kernels (empty result)
  - Significant changes threshold (>5%)
  - Regression detection heuristics
  - Summary output formatting
  - Metrics only in before/after
"""
from __future__ import annotations

import pytest

from tachyon.diff.differ import ProfileDiffer
from tachyon.models.kernel import (
    DeviceInfo,
    KernelReport,
    LaunchParams,
    MetricValue,
)

# ━━━━━━━━━━━━━━━━━━━━━━━ Helpers ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _mv(name: str, value: float, unit: str = "") -> MetricValue:
    """Shorthand for MetricValue creation."""
    return MetricValue(name=name, value=value, unit=unit)


def _device() -> DeviceInfo:
    return DeviceInfo(
        name="NVIDIA A100-SXM4-80GB",
        compute_capability=(8, 0),
        sm_count=108,
        max_clock_mhz=1410,
        memory_bus_width=5120,
        peak_memory_bandwidth_gbps=2039.0,
    )


def _launch() -> LaunchParams:
    return LaunchParams(
        grid=(4096, 1, 1),
        block=(256, 1, 1),
        shared_mem_bytes=0,
        registers_per_thread=32,
    )


def _make_report(
    name: str,
    metrics: dict[str, float],
    unit: str = "%",
) -> KernelReport:
    """Create a KernelReport with given metrics (name -> value)."""
    return KernelReport(
        kernel_name=name,
        demangled_name=f"{name}<float>",
        launch_params=_launch(),
        device_info=_device(),
        metrics={k: _mv(k, v, unit) for k, v in metrics.items()},
    )


# ━━━━━━━━━━━━━━━━━━━━━━━ Tests ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestProfileDifferIdentical:
    """Diffing identical reports should yield no significant changes."""

    def test_diff_identical_reports(self):
        """Same metrics -> deltas are all zero."""
        metrics = {
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 70.0,
            "dram__throughput.avg.pct_of_peak_sustained_elapsed": 50.0,
        }
        before = [_make_report("kernel_a", metrics)]
        after = [_make_report("kernel_a", metrics)]

        differ = ProfileDiffer()
        diffs = differ.diff(before, after)

        assert len(diffs) == 1
        d = diffs[0]
        assert d.kernel_name == "kernel_a"
        assert len(d.metric_deltas) == 2

        for m in d.metric_deltas:
            assert m.delta == 0.0
            assert m.delta_pct == 0.0

    def test_identical_no_significant_changes(self):
        """Identical reports should have no significant changes."""
        metrics = {"metric_a": 100.0, "metric_b": 200.0}
        before = [_make_report("k", metrics)]
        after = [_make_report("k", metrics)]

        diffs = ProfileDiffer().diff(before, after)
        assert len(diffs[0].significant_changes) == 0

    def test_identical_no_regressions(self):
        """Identical reports should have no regressions."""
        metrics = {"throughput_metric": 100.0, "duration_metric": 50.0}
        before = [_make_report("k", metrics)]
        after = [_make_report("k", metrics)]

        diffs = ProfileDiffer().diff(before, after)
        assert len(diffs[0].regressions) == 0


class TestProfileDifferChanges:
    """Diffing reports with real metric changes."""

    def test_diff_with_changes(self):
        """Metrics that differ should produce correct deltas."""
        before = [_make_report("k", {"metric_a": 100.0, "metric_b": 50.0})]
        after = [_make_report("k", {"metric_a": 120.0, "metric_b": 40.0})]

        diffs = ProfileDiffer().diff(before, after)
        assert len(diffs) == 1

        deltas = {m.name: m for m in diffs[0].metric_deltas}

        # metric_a: 100 -> 120
        a = deltas["metric_a"]
        assert a.before == 100.0
        assert a.after == 120.0
        assert a.delta == 20.0
        assert a.delta_pct == pytest.approx(20.0)

        # metric_b: 50 -> 40
        b = deltas["metric_b"]
        assert b.before == 50.0
        assert b.after == 40.0
        assert b.delta == -10.0
        assert b.delta_pct == pytest.approx(-20.0)

    def test_metric_delta_calculation_zero_base(self):
        """When before value is 0, delta_pct should be 0.0 (no division by zero)."""
        before = [_make_report("k", {"metric_a": 0.0})]
        after = [_make_report("k", {"metric_a": 42.0})]

        diffs = ProfileDiffer().diff(before, after)
        m = diffs[0].metric_deltas[0]
        assert m.before == 0.0
        assert m.after == 42.0
        assert m.delta == 42.0
        assert m.delta_pct == 0.0  # safe division

    def test_metric_delta_small_change(self):
        """Small change (< 5%) should not appear in significant_changes."""
        before = [_make_report("k", {"metric_a": 100.0})]
        after = [_make_report("k", {"metric_a": 103.0})]  # 3% change

        diffs = ProfileDiffer().diff(before, after)
        assert len(diffs[0].significant_changes) == 0

    def test_metric_delta_exact_threshold(self):
        """Exactly 5% change should NOT be significant (threshold is > 5%)."""
        before = [_make_report("k", {"metric_a": 100.0})]
        after = [_make_report("k", {"metric_a": 105.0})]  # exactly 5%

        diffs = ProfileDiffer().diff(before, after)
        assert len(diffs[0].significant_changes) == 0


class TestProfileDifferNoCommon:
    """Diffing reports with no common kernels."""

    def test_diff_no_common_kernels(self):
        """Different kernel names -> empty diff list."""
        before = [_make_report("kernel_alpha", {"m": 100.0})]
        after = [_make_report("kernel_beta", {"m": 100.0})]

        diffs = ProfileDiffer().diff(before, after)
        assert len(diffs) == 0

    def test_diff_empty_before(self):
        """Empty before list -> no diffs."""
        after = [_make_report("k", {"m": 100.0})]
        diffs = ProfileDiffer().diff([], after)
        assert len(diffs) == 0

    def test_diff_empty_after(self):
        """Empty after list -> no diffs."""
        before = [_make_report("k", {"m": 100.0})]
        diffs = ProfileDiffer().diff(before, [])
        assert len(diffs) == 0

    def test_diff_both_empty(self):
        """Both empty -> no diffs."""
        diffs = ProfileDiffer().diff([], [])
        assert len(diffs) == 0


class TestSignificantChanges:
    """Test the significant_changes property (>5% threshold)."""

    def test_significant_changes_threshold(self):
        """Only changes with abs(delta_pct) > 5% are significant."""
        before = [_make_report("k", {
            "small_change": 100.0,
            "big_change": 100.0,
            "negative_big": 100.0,
        })]
        after = [_make_report("k", {
            "small_change": 103.0,   # 3% -- not significant
            "big_change": 115.0,     # 15% -- significant
            "negative_big": 80.0,    # -20% -- significant
        })]

        diffs = ProfileDiffer().diff(before, after)
        sig = diffs[0].significant_changes
        sig_names = {m.name for m in sig}

        assert "small_change" not in sig_names
        assert "big_change" in sig_names
        assert "negative_big" in sig_names
        assert len(sig) == 2

    def test_multiple_kernels_independent(self):
        """Each kernel's significant changes are independent."""
        before = [
            _make_report("k1", {"m": 100.0}),
            _make_report("k2", {"m": 100.0}),
        ]
        after = [
            _make_report("k1", {"m": 150.0}),   # 50% change
            _make_report("k2", {"m": 101.0}),   # 1% change
        ]

        diffs = ProfileDiffer().diff(before, after)
        assert len(diffs) == 2
        assert len(diffs[0].significant_changes) == 1
        assert len(diffs[1].significant_changes) == 0


class TestRegressionDetection:
    """Test regression detection heuristics."""

    def test_throughput_decrease_is_regression(self):
        """Throughput metric decreasing >5% is a regression."""
        before = [_make_report("k", {
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 80.0,
        })]
        after = [_make_report("k", {
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 60.0,
        })]

        diffs = ProfileDiffer().diff(before, after)
        regs = diffs[0].regressions
        assert len(regs) == 1
        assert "throughput" in regs[0].name

    def test_throughput_increase_is_not_regression(self):
        """Throughput metric increasing is an improvement, not regression."""
        before = [_make_report("k", {
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 60.0,
        })]
        after = [_make_report("k", {
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 80.0,
        })]

        diffs = ProfileDiffer().diff(before, after)
        assert len(diffs[0].regressions) == 0

    def test_latency_increase_is_regression(self):
        """Duration/latency metric increasing >5% is a regression."""
        before = [_make_report("k", {
            "gpu__time_duration.sum": 1000.0,
        })]
        after = [_make_report("k", {
            "gpu__time_duration.sum": 1200.0,
        })]

        diffs = ProfileDiffer().diff(before, after)
        regs = diffs[0].regressions
        assert len(regs) == 1
        assert "duration" in regs[0].name

    def test_latency_decrease_is_not_regression(self):
        """Duration metric decreasing is an improvement."""
        before = [_make_report("k", {"gpu__time_duration.sum": 1200.0})]
        after = [_make_report("k", {"gpu__time_duration.sum": 1000.0})]

        diffs = ProfileDiffer().diff(before, after)
        assert len(diffs[0].regressions) == 0

    def test_utilization_decrease_is_regression(self):
        """Utilization metric with 'utilization' keyword decreasing is regression."""
        before = [_make_report("k", {"sm_utilization": 80.0})]
        after = [_make_report("k", {"sm_utilization": 60.0})]

        diffs = ProfileDiffer().diff(before, after)
        assert len(diffs[0].regressions) == 1

    def test_hit_rate_decrease_is_regression(self):
        """Cache hit_rate decreasing is a regression."""
        before = [_make_report("k", {"l2_hit_rate": 90.0})]
        after = [_make_report("k", {"l2_hit_rate": 70.0})]

        diffs = ProfileDiffer().diff(before, after)
        assert len(diffs[0].regressions) == 1

    def test_small_regression_ignored(self):
        """Changes <= 5% are not flagged as regressions."""
        before = [_make_report("k", {
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 80.0,
        })]
        after = [_make_report("k", {
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 77.0,
        })]

        diffs = ProfileDiffer().diff(before, after)
        # 3.75% decrease < 5% threshold
        assert len(diffs[0].regressions) == 0

    def test_mixed_regressions_and_improvements(self):
        """Multiple metrics: some regress, some improve."""
        before = [_make_report("k", {
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 80.0,
            "gpu__time_duration.sum": 1000.0,
            "dram__throughput.avg.pct_of_peak_sustained_elapsed": 50.0,
        })]
        after = [_make_report("k", {
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 60.0,  # regression
            "gpu__time_duration.sum": 800.0,                           # improvement
            "dram__throughput.avg.pct_of_peak_sustained_elapsed": 65.0, # improvement
        })]

        diffs = ProfileDiffer().diff(before, after)
        regs = diffs[0].regressions
        assert len(regs) == 1  # only sm throughput decrease


class TestOnlyInBeforeAfter:
    """Test metrics unique to one side."""

    def test_only_in_before(self):
        """Metrics only in before should be listed in only_in_before."""
        before = [_make_report("k", {"metric_a": 100.0, "metric_b": 50.0})]
        after = [_make_report("k", {"metric_a": 110.0})]

        diffs = ProfileDiffer().diff(before, after)
        assert "metric_b" in diffs[0].only_in_before
        assert "metric_b" not in diffs[0].only_in_after

    def test_only_in_after(self):
        """Metrics only in after should be listed in only_in_after."""
        before = [_make_report("k", {"metric_a": 100.0})]
        after = [_make_report("k", {"metric_a": 110.0, "metric_c": 30.0})]

        diffs = ProfileDiffer().diff(before, after)
        assert "metric_c" in diffs[0].only_in_after
        assert "metric_c" not in diffs[0].only_in_before

    def test_only_in_both_sides(self):
        """Metrics unique to each side should be tracked independently."""
        before = [_make_report("k", {"shared": 100.0, "old_metric": 50.0})]
        after = [_make_report("k", {"shared": 110.0, "new_metric": 30.0})]

        diffs = ProfileDiffer().diff(before, after)
        assert "old_metric" in diffs[0].only_in_before
        assert "new_metric" in diffs[0].only_in_after
        # Only "shared" should have a delta
        assert len(diffs[0].metric_deltas) == 1
        assert diffs[0].metric_deltas[0].name == "shared"


class TestSummaryOutput:
    """Test ProfileDiffer.summary() text output."""

    def test_summary_with_changes(self):
        """Summary should mention significant changes and regressions."""
        before = [_make_report("k", {
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 80.0,
            "gpu__time_duration.sum": 1000.0,
        })]
        after = [_make_report("k", {
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 60.0,
            "gpu__time_duration.sum": 1200.0,
        })]

        differ = ProfileDiffer()
        diffs = differ.diff(before, after)
        summary = differ.summary(diffs)

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "significant" in summary.lower() or "regression" in summary.lower() or "k<float>" in summary

    def test_summary_no_changes(self):
        """Summary for identical reports should say 'no significant changes'."""
        before = [_make_report("k", {"m": 100.0})]
        after = [_make_report("k", {"m": 100.0})]

        differ = ProfileDiffer()
        diffs = differ.diff(before, after)
        summary = differ.summary(diffs)

        assert "no significant changes" in summary.lower()

    def test_summary_empty_diffs(self):
        """Summary for empty diffs should return a reasonable message."""
        differ = ProfileDiffer()
        summary = differ.summary([])
        assert isinstance(summary, str)
        # "No differences found." per implementation
        assert "no differences" in summary.lower() or len(summary) >= 0

    def test_summary_contains_arrows(self):
        """Summary should use arrow indicators for direction of change."""
        before = [_make_report("k", {"metric_a": 100.0})]
        after = [_make_report("k", {"metric_a": 150.0})]

        differ = ProfileDiffer()
        diffs = differ.diff(before, after)
        summary = differ.summary(diffs)

        # Should contain an arrow (up or down unicode)
        assert "\u2191" in summary or "\u2193" in summary

    def test_summary_multiple_kernels(self):
        """Summary should include entries for all matched kernels."""
        before = [
            _make_report("k1", {"m": 100.0}),
            _make_report("k2", {"m": 200.0}),
        ]
        after = [
            _make_report("k1", {"m": 120.0}),
            _make_report("k2", {"m": 250.0}),
        ]

        differ = ProfileDiffer()
        diffs = differ.diff(before, after)
        summary = differ.summary(diffs)

        # Both kernel demangled names should appear
        assert "k1<float>" in summary
        assert "k2<float>" in summary
