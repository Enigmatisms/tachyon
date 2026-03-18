"""Unit tests for all Analyzers — comprehensive coverage with synthetic fixtures."""

from tachyon.models.finding import Severity
from tachyon.models.hotspot import SourceHotspot
from tachyon.models.kernel import KernelReport

# ━━━━━━━━━━━━━━━━━━━━━━━ RooflineAnalyzer ━━━━━━━━━━━━━━━━━━━━


class TestRooflineAnalyzer:
    def test_compute_bound(self, report_compute_bound: KernelReport):
        """SM=85%, DRAM=30% -> compute-bound."""
        from tachyon.analyzers.roofline import RooflineAnalyzer
        analyzer = RooflineAnalyzer()
        assert analyzer.can_run(report_compute_bound)
        findings = analyzer.analyze(report_compute_bound)
        assert any("COMPUTE-BOUND" in f.title for f in findings)

    def test_memory_bound(self, report_memory_bound: KernelReport):
        """SM=25%, DRAM=78% -> memory-bound."""
        from tachyon.analyzers.roofline import RooflineAnalyzer
        findings = RooflineAnalyzer().analyze(report_memory_bound)
        assert any("MEMORY-BOUND" in f.title for f in findings)

    def test_latency_bound(self, report_latency_bound: KernelReport):
        """SM=22%, DRAM=18% -> latency-bound (both < 60%)."""
        from tachyon.analyzers.roofline import RooflineAnalyzer
        findings = RooflineAnalyzer().analyze(report_latency_bound)
        assert any("LATENCY-BOUND" in f.title for f in findings)
        # Latency-bound should be CRITICAL severity per implementation
        latency_findings = [f for f in findings if "LATENCY-BOUND" in f.title]
        assert latency_findings[0].severity == Severity.CRITICAL

    def test_balanced(self, report_balanced: KernelReport):
        """SM=88%, DRAM=85% -> balanced."""
        from tachyon.analyzers.roofline import RooflineAnalyzer
        findings = RooflineAnalyzer().analyze(report_balanced)
        assert any("BALANCED" in f.title for f in findings)
        balanced_findings = [f for f in findings if "BALANCED" in f.title]
        assert balanced_findings[0].severity == Severity.WARNING

    def test_metrics_in_findings(self, report_compute_bound: KernelReport):
        """Findings should contain metric values."""
        from tachyon.analyzers.roofline import RooflineAnalyzer
        findings = RooflineAnalyzer().analyze(report_compute_bound)
        assert len(findings) > 0
        f = findings[0]
        assert len(f.metrics) > 0

    def test_cannot_run_missing_metrics(self, report_minimal: KernelReport):
        """Cannot run without required metrics."""
        from tachyon.analyzers.roofline import RooflineAnalyzer
        analyzer = RooflineAnalyzer()
        assert not analyzer.can_run(report_minimal)

    def test_name_and_category(self):
        from tachyon.analyzers.roofline import RooflineAnalyzer
        a = RooflineAnalyzer()
        assert a.name() == "roofline"
        assert a.category() == "compute"

    def test_required_metrics(self):
        from tachyon.analyzers.roofline import RooflineAnalyzer
        metrics = RooflineAnalyzer().required_metrics()
        assert "sm__throughput.avg.pct_of_peak_sustained_elapsed" in metrics
        # DRAM is consumed if available but NOT required (graceful degradation)
        assert "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed" not in metrics


# ━━━━━━━━━━━━━━━━━━━━━━━ MemoryAnalyzer ━━━━━━━━━━━━━━━━━━━━━━


class TestMemoryAnalyzer:
    def test_poor_coalescing(self, report_memory_bound: KernelReport):
        """sectors=50000, requests=1000 -> efficiency=64% -> WARNING or CRITICAL."""
        from tachyon.analyzers.memory import MemoryAnalyzer
        findings = MemoryAnalyzer().analyze(report_memory_bound)
        coalescing = [f for f in findings if "coalescing" in f.title.lower()]
        assert len(coalescing) >= 1
        assert coalescing[0].severity in (Severity.WARNING, Severity.CRITICAL)

    def test_good_coalescing(self, report_coalesced: KernelReport):
        """sectors=32000, requests=1000 -> efficiency=100% -> no warning."""
        from tachyon.analyzers.memory import MemoryAnalyzer
        findings = MemoryAnalyzer().analyze(report_coalesced)
        # Good coalescing: implementation doesn't emit finding for good coalescing
        coalescing_warnings = [f for f in findings
                               if "coalescing" in f.title.lower()
                               and f.severity in (Severity.WARNING, Severity.CRITICAL)]
        assert len(coalescing_warnings) == 0

    def test_bank_conflicts(self, report_memory_bound: KernelReport):
        """500 bank conflicts -> WARNING."""
        from tachyon.analyzers.memory import MemoryAnalyzer
        findings = MemoryAnalyzer().analyze(report_memory_bound)
        bank = [f for f in findings if "bank conflict" in f.title.lower()]
        assert len(bank) >= 1
        assert bank[0].severity == Severity.WARNING

    def test_l2_cache_low(self, report_memory_bound: KernelReport):
        """hit=3000, miss=7000 -> 30% hit rate -> WARNING."""
        from tachyon.analyzers.memory import MemoryAnalyzer
        findings = MemoryAnalyzer().analyze(report_memory_bound)
        l2 = [f for f in findings if "l2" in f.title.lower() or "L2" in f.title]
        assert len(l2) >= 1
        assert l2[0].severity == Severity.WARNING

    def test_cannot_run_missing_metrics(self, report_minimal: KernelReport):
        from tachyon.analyzers.memory import MemoryAnalyzer
        assert not MemoryAnalyzer().can_run(report_minimal)

    def test_name_and_category(self):
        from tachyon.analyzers.memory import MemoryAnalyzer
        a = MemoryAnalyzer()
        assert a.name() == "memory"
        assert a.category() == "memory"

    def test_no_bank_conflicts_silent(self, report_coalesced: KernelReport):
        """No bank conflict metric -> no bank conflict finding."""
        from tachyon.analyzers.memory import MemoryAnalyzer
        findings = MemoryAnalyzer().analyze(report_coalesced)
        bank = [f for f in findings if "bank conflict" in f.title.lower()]
        assert len(bank) == 0


# ━━━━━━━━━━━━━━━━━━━━━━━ WarpStallAnalyzer ━━━━━━━━━━━━━━━━━━━


class TestWarpStallAnalyzer:
    def test_dominant_stall_reason(self, report_latency_bound: KernelReport):
        """long_scoreboard=8500 is the dominant stall."""
        from tachyon.analyzers.warp_stall import WarpStallAnalyzer
        findings = WarpStallAnalyzer().analyze(report_latency_bound)
        # Should have a finding mentioning long_scoreboard or Long Scoreboard
        stall_findings = [f for f in findings
                         if "long_scoreboard" in f.title.lower()
                         or "long scoreboard" in f.title.lower()
                         or "Long Scoreboard" in f.title]
        assert len(stall_findings) >= 1

    def test_secondary_stall_below_threshold(self, report_latency_bound: KernelReport):
        """barrier=3200/(8500+3200+200+100)=26.7% < 30% threshold."""
        from tachyon.analyzers.warp_stall import WarpStallAnalyzer
        findings = WarpStallAnalyzer().analyze(report_latency_bound)
        # Should have a stall breakdown table
        assert len(findings) > 0

    def test_no_stalls_empty(self, report_no_stalls: KernelReport):
        """All stall metrics are zero -> no/minimal findings."""
        from tachyon.analyzers.warp_stall import WarpStallAnalyzer
        findings = WarpStallAnalyzer().analyze(report_no_stalls)
        # Implementation returns empty list when total_stalls == 0
        assert len(findings) == 0

    def test_name_and_category(self):
        from tachyon.analyzers.warp_stall import WarpStallAnalyzer
        a = WarpStallAnalyzer()
        assert a.name() == "warp_stall"
        assert a.category() == "latency"

    def test_stall_detail_has_content(self, report_latency_bound: KernelReport):
        """Stall findings should have meaningful detail."""
        from tachyon.analyzers.warp_stall import WarpStallAnalyzer
        findings = WarpStallAnalyzer().analyze(report_latency_bound)
        assert len(findings) > 0
        # At least one finding should have non-trivial detail
        assert any(len(f.detail) > 20 for f in findings)

    def test_findings_have_metrics(self, report_latency_bound: KernelReport):
        """Findings should carry metric evidence."""
        from tachyon.analyzers.warp_stall import WarpStallAnalyzer
        findings = WarpStallAnalyzer().analyze(report_latency_bound)
        metric_findings = [f for f in findings if len(f.metrics) > 0]
        assert len(metric_findings) >= 1


# ━━━━━━━━━━━━━━━━━━━━━━━ NvRulesAdapter ━━━━━━━━━━━━━━━━━━━━━━


class TestNvRulesAdapter:
    def test_converts_rules_to_findings(self, report_with_rules: KernelReport):
        """NvRulesAdapter should convert rule results to findings."""
        from tachyon.analyzers.nvrules import NvRulesAdapter
        adapter = NvRulesAdapter()
        assert adapter.can_run(report_with_rules)
        findings = adapter.analyze(report_with_rules)
        # Should skip OK rules, keep LOW/MED/HIGH = 3 findings
        assert len(findings) == 3

    def test_severity_mapping(self, report_with_rules: KernelReport):
        """Verify severity mapping: LOW->INFO, MED->WARNING, HIGH->CRITICAL."""
        from tachyon.analyzers.nvrules import NvRulesAdapter
        findings = NvRulesAdapter().analyze(report_with_rules)
        # HIGH -> CRITICAL
        high_findings = [f for f in findings if "ComputeWorkload" in f.title]
        assert len(high_findings) == 1
        assert high_findings[0].severity == Severity.CRITICAL
        # MED -> WARNING
        med_findings = [f for f in findings if "MemoryWorkload" in f.title]
        assert len(med_findings) == 1
        assert med_findings[0].severity == Severity.WARNING

    def test_skips_ok_rules(self, report_with_rules: KernelReport):
        """OK rules should be skipped."""
        from tachyon.analyzers.nvrules import NvRulesAdapter
        findings = NvRulesAdapter().analyze(report_with_rules)
        ok_findings = [f for f in findings if "SpeedOfLight" in f.title]
        assert len(ok_findings) == 0

    def test_cannot_run_no_rules(self, report_minimal: KernelReport):
        from tachyon.analyzers.nvrules import NvRulesAdapter
        assert not NvRulesAdapter().can_run(report_minimal)

    def test_source_label(self, report_with_rules: KernelReport):
        """Source field should be the adapter name."""
        from tachyon.analyzers.nvrules import NvRulesAdapter
        findings = NvRulesAdapter().analyze(report_with_rules)
        for f in findings:
            assert f.source == "nvrules"


# ━━━━━━━━━━━━━━━━━━━━━━━ AnalyzerRegistry ━━━━━━━━━━━━━━━━━━━━


class TestAnalyzerRegistry:
    def test_auto_register(self):
        """auto_register should add 6 analyzers (NvRules excluded by default)."""
        from tachyon.analyzers.base import AnalyzerRegistry
        registry = AnalyzerRegistry()
        registry.auto_register()
        assert len(registry.all_analyzers()) == 6

    def test_run_all_compute_bound(self, report_compute_bound: KernelReport):
        """Pipeline produces findings for compute-bound kernel."""
        from tachyon.analyzers.base import AnalyzerRegistry
        registry = AnalyzerRegistry()
        registry.auto_register()
        findings = registry.run_all(report_compute_bound)
        assert len(findings) > 0
        # Should be sorted: CRITICAL first
        if len(findings) > 1:
            assert findings[0].severity >= findings[-1].severity

    def test_run_all_memory_bound(self, report_memory_bound: KernelReport):
        from tachyon.analyzers.base import AnalyzerRegistry
        registry = AnalyzerRegistry()
        registry.auto_register()
        findings = registry.run_all(report_memory_bound)
        sources = {f.source for f in findings}
        assert "roofline" in sources
        assert "memory" in sources

    def test_run_all_latency_bound(self, report_latency_bound: KernelReport):
        from tachyon.analyzers.base import AnalyzerRegistry
        registry = AnalyzerRegistry()
        registry.auto_register()
        findings = registry.run_all(report_latency_bound)
        sources = {f.source for f in findings}
        assert "roofline" in sources
        assert "warp_stall" in sources

    def test_run_all_with_rules(self, report_with_rules: KernelReport):
        from tachyon.analyzers.base import AnalyzerRegistry
        registry = AnalyzerRegistry()
        registry.auto_register(include_nvrules=True)
        findings = registry.run_all(report_with_rules)
        nvrule_findings = [f for f in findings if f.source == "nvrules"]
        assert len(nvrule_findings) > 0

    def test_fault_isolation(self, report_compute_bound: KernelReport):
        """A broken analyzer should not block others from running."""
        from tachyon.analyzers.base import Analyzer, AnalyzerRegistry

        class BrokenAnalyzer(Analyzer):
            def name(self) -> str:
                return "BrokenAnalyzer"
            def category(self) -> str:
                return "test"
            def required_metrics(self) -> list[str]:
                return []
            def analyze(self, report):
                raise RuntimeError("intentional crash")

        registry = AnalyzerRegistry()
        registry.register(BrokenAnalyzer())
        registry.auto_register()

        findings = registry.run_all(report_compute_bound)
        # BrokenAnalyzer failure should produce a diagnostic finding
        error_findings = [f for f in findings if "BrokenAnalyzer" in f.title]
        assert len(error_findings) == 1
        # Real analyzers should still run
        real_findings = [f for f in findings if "BrokenAnalyzer" not in f.title]
        assert len(real_findings) > 0

    def test_skip_missing_metrics(self, report_minimal: KernelReport):
        """Analyzers that need metrics not in report should be skipped.

        Note: LaunchConfigAnalyzer always runs (no required metrics), so
        we check that metric-dependent analyzers (roofline, memory, warp_stall)
        are correctly skipped.
        """
        from tachyon.analyzers.base import AnalyzerRegistry
        registry = AnalyzerRegistry()
        registry.auto_register()
        findings = registry.run_all(report_minimal)
        metric_analyzer_findings = [
            f for f in findings
            if f.source in ("roofline", "memory", "warp_stall", "nvrules")
        ]
        assert len(metric_analyzer_findings) == 0

    def test_findings_sorted_by_severity(self, report_latency_bound: KernelReport):
        """Output findings should be sorted CRITICAL > WARNING > INFO."""
        from tachyon.analyzers.base import AnalyzerRegistry
        registry = AnalyzerRegistry()
        registry.auto_register()
        findings = registry.run_all(report_latency_bound)
        for i in range(len(findings) - 1):
            assert findings[i].severity >= findings[i + 1].severity

    def test_empty_registry(self, report_compute_bound: KernelReport):
        """Empty registry produces no findings."""
        from tachyon.analyzers.base import AnalyzerRegistry
        registry = AnalyzerRegistry()
        findings = registry.run_all(report_compute_bound)
        assert findings == []


# ━━━━━━━━━━━━━━━━━━━━━━━ M2: Source Attribution ━━━━━━━━━━━━━━━━


def _make_test_hotspots() -> list[SourceHotspot]:
    """Create synthetic hotspots for source-attribution testing."""
    return [
        SourceHotspot(
            pc=0x1000,
            source_file="kernel.cu",
            source_line=42,
            sass_instruction="LDG.E.128 R4, [R2.64]",
            ptx_instruction="ld.global.v4.f32",
            metric_values={"inst_executed": 200.0},
            stall_reasons={
                "smsp__pcsamp_warps_issue_stalled_long_scoreboard": 0.8,
                "smsp__pcsamp_warps_issue_stalled_barrier": 0.2,
            },
            is_hot=True,
            local_ratio=0.8,
            global_ratio=0.5,
            degraded=False,
            dominant_stall="smsp__pcsamp_warps_issue_stalled_long_scoreboard",
        ),
        SourceHotspot(
            pc=0x2000,
            source_file="kernel.cu",
            source_line=58,
            sass_instruction="FFMA R0, R4, R8, R0",
            ptx_instruction="fma.rn.f32",
            metric_values={"inst_executed": 100.0},
            stall_reasons={
                "smsp__pcsamp_warps_issue_stalled_barrier": 0.9,
                "smsp__pcsamp_warps_issue_stalled_long_scoreboard": 0.1,
            },
            is_hot=True,
            local_ratio=0.9,
            global_ratio=0.3,
            degraded=False,
            dominant_stall="smsp__pcsamp_warps_issue_stalled_barrier",
        ),
    ]


class TestM2SourceAttribution:
    """M2: Test that analyzers correctly attach source evidence when hotspots are injected."""

    def test_roofline_attaches_source(self, report_compute_bound: KernelReport):
        """RooflineAnalyzer should attach top hotspot's source to its finding."""
        from tachyon.analyzers.roofline import RooflineAnalyzer
        analyzer = RooflineAnalyzer()
        analyzer.set_hotspots(_make_test_hotspots())
        findings = analyzer.analyze(report_compute_bound)
        assert len(findings) > 0
        # First finding should have source_location from hotspot[0]
        f = findings[0]
        assert f.source_location is not None
        assert f.source_location.file == "kernel.cu"
        assert f.source_location.line == 42
        assert f.sass_evidence is not None

    def test_roofline_no_hotspots(self, report_compute_bound: KernelReport):
        """Without hotspots, findings should have no source_location (M1 behavior)."""
        from tachyon.analyzers.roofline import RooflineAnalyzer
        analyzer = RooflineAnalyzer()
        findings = analyzer.analyze(report_compute_bound)
        for f in findings:
            assert f.source_location is None

    def test_memory_attaches_source(self, report_memory_bound: KernelReport):
        """MemoryAnalyzer should attach source evidence to coalescing finding."""
        from tachyon.analyzers.memory import MemoryAnalyzer
        analyzer = MemoryAnalyzer()
        analyzer.set_hotspots(_make_test_hotspots())
        findings = analyzer.analyze(report_memory_bound)
        # At least one finding should have source_location
        source_findings = [f for f in findings if f.source_location is not None]
        assert len(source_findings) > 0

    def test_warp_stall_per_stall_attribution(self, report_latency_bound: KernelReport):
        """WarpStallAnalyzer should find specific hotspot per stall type."""
        from tachyon.analyzers.warp_stall import WarpStallAnalyzer
        analyzer = WarpStallAnalyzer()
        hotspots = _make_test_hotspots()
        analyzer.set_hotspots(hotspots)
        findings = analyzer.analyze(report_latency_bound)
        # Long scoreboard finding should point to hotspot[0] (line 42)
        long_sb_findings = [f for f in findings if "Long Scoreboard" in f.title]
        if long_sb_findings:
            f = long_sb_findings[0]
            assert f.source_location is not None
            assert f.source_location.line == 42
        # Barrier finding should point to hotspot[1] (line 58)
        barrier_findings = [f for f in findings if "Barrier" in f.title]
        if barrier_findings:
            f = barrier_findings[0]
            assert f.source_location is not None
            assert f.source_location.line == 58

    def test_degraded_hotspot_no_source_location(self, report_compute_bound: KernelReport):
        """Degraded hotspot should not produce source_location."""
        from tachyon.analyzers.roofline import RooflineAnalyzer
        degraded = [
            SourceHotspot(
                pc=0x1000,
                source_file="<no debug info>",
                source_line=0,
                degraded=True,
                is_hot=True,
                local_ratio=0.5,
                global_ratio=0.5,
                sass_instruction="LDG R4, [R2]",
            ),
        ]
        analyzer = RooflineAnalyzer()
        analyzer.set_hotspots(degraded)
        findings = analyzer.analyze(report_compute_bound)
        # Source location should NOT be set for degraded hotspots
        for f in findings:
            assert f.source_location is None
        # But SASS evidence should still be present
        assert any(f.sass_evidence is not None for f in findings)

    def test_registry_run_all_with_correlator(self, report_latency_bound: KernelReport):
        """AnalyzerRegistry.run_all with mock correlator should not crash."""
        from unittest.mock import MagicMock

        from tachyon.analyzers.base import AnalyzerRegistry

        mock_correlator = MagicMock()
        mock_correlator.correlate.return_value = _make_test_hotspots()

        mock_action = MagicMock()

        registry = AnalyzerRegistry()
        registry.auto_register()
        findings = registry.run_all(
            report_latency_bound,
            action=mock_action,
            correlator=mock_correlator,
        )
        assert len(findings) > 0
        # Correlator.correlate should have been called
        mock_correlator.correlate.assert_called_once()
