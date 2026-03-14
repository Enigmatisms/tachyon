"""Unit tests for data models."""
from tachyon.models.finding import Finding, Severity, SourceLocation
from tachyon.models.kernel import (
    DeviceInfo,
    InstancedMetricValue,
    KernelReport,
    LaunchParams,
    MetricValue,
    RuleResult,
)

# ━━━━━━━━━━━━━━━━━━━━━━━ LaunchParams ━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLaunchParams:
    def test_total_threads(self):
        lp = LaunchParams(grid=(4, 2, 1), block=(256, 1, 1),
                          shared_mem_bytes=0, registers_per_thread=32)
        assert lp.total_threads == 4 * 2 * 1 * 256 * 1 * 1

    def test_block_size(self):
        lp = LaunchParams(grid=(1, 1, 1), block=(16, 16, 1),
                          shared_mem_bytes=0, registers_per_thread=32)
        assert lp.block_size == 256

    def test_grid_size(self):
        lp = LaunchParams(grid=(10, 20, 3), block=(128, 1, 1),
                          shared_mem_bytes=0, registers_per_thread=32)
        assert lp.grid_size == 600

    def test_frozen(self):
        lp = LaunchParams(grid=(1, 1, 1), block=(1, 1, 1),
                          shared_mem_bytes=0, registers_per_thread=32)
        import dataclasses
        assert dataclasses.is_dataclass(lp)
        # frozen=True should raise FrozenInstanceError
        try:
            lp.shared_mem_bytes = 100  # type: ignore[misc]
            assert False, "Should raise FrozenInstanceError"
        except (AttributeError, dataclasses.FrozenInstanceError):
            pass

    def test_static_shared_default(self):
        lp = LaunchParams(grid=(1, 1, 1), block=(1, 1, 1),
                          shared_mem_bytes=0, registers_per_thread=32)
        assert lp.static_shared_mem_bytes == 0

    def test_3d_launch(self):
        lp = LaunchParams(grid=(4, 4, 4), block=(8, 8, 4),
                          shared_mem_bytes=1024, registers_per_thread=64)
        assert lp.grid_size == 64
        assert lp.block_size == 256
        assert lp.total_threads == 64 * 256


# ━━━━━━━━━━━━━━━━━━━━━━━ DeviceInfo ━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDeviceInfo:
    def test_frozen(self):
        d = DeviceInfo(name="H100", compute_capability=(9, 0),
                       sm_count=132, max_clock_mhz=1980,
                       memory_bus_width=5120, peak_memory_bandwidth_gbps=3352.0)
        try:
            d.name = "changed"  # type: ignore[misc]
            assert False, "Should be frozen"
        except (AttributeError, Exception):
            pass

    def test_creation(self):
        d = DeviceInfo(name="A100", compute_capability=(8, 0),
                       sm_count=108, max_clock_mhz=1410,
                       memory_bus_width=5120, peak_memory_bandwidth_gbps=2039.0)
        assert d.compute_capability == (8, 0)
        assert d.sm_count == 108


# ━━━━━━━━━━━━━━━━━━━━━━━ MetricValue / InstancedMetricValue ━━━━


class TestMetricValue:
    def test_creation(self):
        mv = MetricValue(name="test_metric", value=42.5, unit="%")
        assert mv.name == "test_metric"
        assert mv.value == 42.5
        assert mv.unit == "%"

    def test_frozen(self):
        mv = MetricValue(name="m", value=1.0, unit="")
        try:
            mv.value = 2.0  # type: ignore[misc]
            assert False
        except (AttributeError, Exception):
            pass


class TestInstancedMetricValue:
    def test_defaults(self):
        iv = InstancedMetricValue(pc=0xDEAD, value=3.14)
        assert iv.source_file is None
        assert iv.source_line is None

    def test_with_source(self):
        iv = InstancedMetricValue(pc=0x1234, value=99.9,
                                  source_file="kernel.cu", source_line=42)
        assert iv.source_file == "kernel.cu"
        assert iv.source_line == 42


# ━━━━━━━━━━━━━━━━━━━━━━━ RuleResult ━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRuleResult:
    def test_creation(self):
        rr = RuleResult(rule_name="SpeedOfLight", severity="HIGH",
                        message="Utilization is low.")
        assert rr.rule_name == "SpeedOfLight"
        assert rr.severity == "HIGH"


# ━━━━━━━━━━━━━━━━━━━━━━━ KernelReport ━━━━━━━━━━━━━━━━━━━━━━━━━


class TestKernelReport:
    def test_metric_value_found(self, report_compute_bound: KernelReport):
        val = report_compute_bound.metric_value(
            "sm__throughput.avg.pct_of_peak_sustained_elapsed"
        )
        assert val == 85.0

    def test_metric_value_not_found(self, report_compute_bound: KernelReport):
        val = report_compute_bound.metric_value("nonexistent_metric")
        assert val is None

    def test_has_metrics_true(self, report_compute_bound: KernelReport):
        assert report_compute_bound.has_metrics([
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        ])

    def test_has_metrics_false(self, report_compute_bound: KernelReport):
        assert not report_compute_bound.has_metrics(["missing_metric"])

    def test_has_metrics_empty(self, report_minimal: KernelReport):
        assert report_minimal.has_metrics([]) is True

    def test_mutable(self, report_minimal: KernelReport):
        """KernelReport is mutable for incremental population."""
        report_minimal.metrics["new"] = MetricValue("new", 1.0, "")
        assert "new" in report_minimal.metrics

    def test_default_fields(self):
        lp = LaunchParams(grid=(1, 1, 1), block=(1, 1, 1),
                          shared_mem_bytes=0, registers_per_thread=16)
        di = DeviceInfo(name="test", compute_capability=(7, 0),
                        sm_count=1, max_clock_mhz=1000,
                        memory_bus_width=256, peak_memory_bandwidth_gbps=100.0)
        kr = KernelReport(kernel_name="k", demangled_name="k",
                          launch_params=lp, device_info=di)
        assert kr.metrics == {}
        assert kr.instanced_metrics == {}
        assert kr.source_files == {}
        assert kr.rule_results == []

    def test_instanced_metrics_as_tuples(self):
        """M2: instanced_metrics_as_tuples converts InstancedMetricValue to (pc, value)."""
        lp = LaunchParams(grid=(1, 1, 1), block=(1, 1, 1),
                          shared_mem_bytes=0, registers_per_thread=16)
        di = DeviceInfo(name="test", compute_capability=(7, 0),
                        sm_count=1, max_clock_mhz=1000,
                        memory_bus_width=256, peak_memory_bandwidth_gbps=100.0)
        kr = KernelReport(kernel_name="k", demangled_name="k",
                          launch_params=lp, device_info=di)
        kr.instanced_metrics["stall_metric"] = [
            InstancedMetricValue(pc=0x1000, value=100.0),
            InstancedMetricValue(pc=0x2000, value=200.0),
        ]
        result = kr.instanced_metrics_as_tuples()
        assert "stall_metric" in result
        assert result["stall_metric"] == [(0x1000, 100.0), (0x2000, 200.0)]

    def test_instanced_metrics_as_tuples_empty(self, report_minimal: KernelReport):
        """Empty instanced_metrics returns empty dict."""
        result = report_minimal.instanced_metrics_as_tuples()
        assert result == {}


# ━━━━━━━━━━━━━━━━━━━━━━━ Severity ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSeverity:
    def test_ordering(self):
        assert Severity.INFO < Severity.WARNING
        assert Severity.WARNING < Severity.CRITICAL
        assert not Severity.CRITICAL < Severity.INFO

    def test_le(self):
        assert Severity.INFO <= Severity.INFO
        assert Severity.INFO <= Severity.WARNING

    def test_gt(self):
        assert Severity.CRITICAL > Severity.WARNING
        assert Severity.WARNING > Severity.INFO

    def test_ge(self):
        assert Severity.CRITICAL >= Severity.CRITICAL
        assert Severity.CRITICAL >= Severity.INFO

    def test_sort(self):
        items = [Severity.INFO, Severity.CRITICAL, Severity.WARNING]
        sorted_items = sorted(items)
        assert sorted_items == [Severity.INFO, Severity.WARNING, Severity.CRITICAL]

    def test_values(self):
        assert Severity.CRITICAL.value == "critical"
        assert Severity.WARNING.value == "warning"
        assert Severity.INFO.value == "info"


# ━━━━━━━━━━━━━━━━━━━━━━━ SourceLocation ━━━━━━━━━━━━━━━━━━━━━━━


class TestSourceLocation:
    def test_creation(self):
        sl = SourceLocation(file="kernel.cu", line=42, function="myKernel")
        assert sl.file == "kernel.cu"
        assert sl.line == 42
        assert sl.function == "myKernel"

    def test_no_function(self):
        sl = SourceLocation(file="a.cu", line=1)
        assert sl.function is None


# ━━━━━━━━━━━━━━━━━━━━━━━ Finding ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestFinding:
    def test_creation(self):
        f = Finding(
            severity=Severity.WARNING,
            title="Test finding",
            detail="Detail text",
            action="Do something",
            source="TestAnalyzer",
        )
        assert f.title == "Test finding"
        assert f.category == ""
        assert f.metrics == {}

    def test_with_source_location(self):
        f = Finding(
            severity=Severity.CRITICAL,
            title="Hot",
            detail="d",
            action="a",
            source="s",
            source_location=SourceLocation(file="k.cu", line=10),
        )
        assert f.source_location is not None
        assert f.source_location.line == 10

    def test_sort_by_severity(self):
        findings = [
            Finding(severity=Severity.INFO, title="low", detail="", action="", source=""),
            Finding(severity=Severity.CRITICAL, title="high", detail="", action="", source=""),
            Finding(severity=Severity.WARNING, title="mid", detail="", action="", source=""),
        ]
        findings.sort(key=lambda f: f.severity, reverse=True)
        assert findings[0].title == "high"
        assert findings[1].title == "mid"
        assert findings[2].title == "low"
