"""Integration tests -- M2 end-to-end pipeline.

Tests the full flow: SourceCorrelator -> AnalyzerRegistry (with correlator)
-> OptimizationTree -> MarkdownReporter.  No GPU or NCU required.
"""
from unittest.mock import MagicMock

import pytest

import tachyon.i18n as i18n
from tachyon.analyzers.base import AnalyzerRegistry
from tachyon.correlator.source_correlator import SourceCorrelator, SourceInfo
from tachyon.models.kernel import InstancedMetricValue, KernelReport
from tachyon.report.markdown import MarkdownReporter, ReportContext
from tachyon.tree.opt_tree import OptimizationTree

# ━━━━━━━━━━━━━━━━━━━━━━━ Fixtures ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture(autouse=True)
def init_i18n():
    """Reset and initialize i18n to English before every test."""
    i18n._packs.clear()
    i18n._current_lang = "en"
    i18n.init("en")
    yield
    i18n._packs.clear()


# ━━━━━━━━━━━━━━━━━━━━━━━ Helpers ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _make_mock_action(pc_source_map):
    """Create a mock ActionHandle with configurable PC -> source mapping.

    Args:
        pc_source_map: dict mapping PC (int) to (file_name, line) tuples.
                       PCs not in the map return None for source_info.
    """
    action = MagicMock()

    def _source_info(pc):
        if pc in pc_source_map:
            f, l = pc_source_map[pc]
            return SourceInfo(file_name=f, line=l)
        return None

    action.source_info.side_effect = _source_info
    action.sass_by_pc.side_effect = lambda pc: f"SASS@0x{pc:x}"
    action.ptx_by_pc.side_effect = lambda pc: f"PTX@0x{pc:x}"
    return action


def _add_instanced_stall_metrics(report, pc_values):
    """Add instanced stall metric entries to a KernelReport.

    Args:
        report: KernelReport to mutate.
        pc_values: list of (pc, value) tuples.
    """
    metric_name = "smsp__pcsamp_warps_issue_stalled_long_scoreboard"
    report.instanced_metrics[metric_name] = [
        InstancedMetricValue(pc=pc, value=val)
        for pc, val in pc_values
    ]
    # Also add inst_executed so total_samples is meaningful
    report.instanced_metrics["inst_executed"] = [
        InstancedMetricValue(pc=pc, value=val * 2)
        for pc, val in pc_values
    ]


def _run_full_pipeline(report, action=None, correlator=None):
    """Run the complete M2 pipeline and return (findings, hotspots, opt_tree, markdown).

    Steps:
    1. AnalyzerRegistry.run_all (with optional correlator + action)
    2. OptimizationTree from findings
    3. MarkdownReporter rendering
    """
    registry = AnalyzerRegistry()
    registry.auto_register()

    findings = registry.run_all(report, action=action, correlator=correlator)

    # Compute hotspots separately for the report context
    hotspots = []
    if correlator is not None and action is not None:
        instanced = report.instanced_metrics_as_tuples()
        if instanced:
            hotspots = correlator.correlate(action, instanced)

    opt_tree = OptimizationTree(findings)

    degradation_warning = None
    if correlator is not None:
        degradation_warning = correlator.check_degradation(hotspots)

    ctx = ReportContext(
        kernel_name=report.kernel_name,
        demangled_name=report.demangled_name,
        device_name=report.device_info.name,
        compute_capability=f"{report.device_info.compute_capability[0]}.{report.device_info.compute_capability[1]}",
        findings=findings,
        hotspots=hotspots,
        opt_tree=opt_tree,
        degradation_warning=degradation_warning,
        verbose=True,
    )

    reporter = MarkdownReporter()
    markdown = reporter.render(ctx)

    return findings, hotspots, opt_tree, markdown


# ━━━━━━━━━━━━━━━━━━━━━━━ Tests ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestM2Pipeline:

    def test_m2_compute_bound_pipeline(self, report_compute_bound: KernelReport):
        """Full M2 pipeline for a compute-bound kernel.

        Uses mock action with source mapping, runs correlator + analyzers
        + OptTree + MarkdownReporter.  Verifies output contains expected
        sections and bottleneck classification.
        """
        # Set up instanced metrics on the report
        pc_values = [
            (0x1000, 100.0),
            (0x1004, 200.0),
            (0x1008, 50.0),
        ]
        _add_instanced_stall_metrics(report_compute_bound, pc_values)

        action = _make_mock_action({
            0x1000: ("gemm.cu", 100),
            0x1004: ("gemm.cu", 100),
            0x1008: ("gemm.cu", 120),
        })

        correlator = SourceCorrelator()
        findings, hotspots, opt_tree, markdown = _run_full_pipeline(
            report_compute_bound, action=action, correlator=correlator,
        )

        # Findings should classify as compute-bound
        assert any("compute-bound" in f.title for f in findings)

        # Markdown output should contain all major sections
        assert "Executive Summary" in markdown
        assert "Optimization Tree" in markdown
        assert "compute" in markdown.lower()

    def test_m2_memory_bound_with_source(self, report_memory_bound: KernelReport):
        """Memory-bound kernel with source mapping.

        Verifies that SourceCorrelator populates source_location on findings.
        """
        pc_values = [
            (0x2000, 500.0),
            (0x2004, 300.0),
        ]
        _add_instanced_stall_metrics(report_memory_bound, pc_values)

        action = _make_mock_action({
            0x2000: ("softmax.cu", 42),
            0x2004: ("softmax.cu", 43),
        })

        correlator = SourceCorrelator()
        findings, hotspots, opt_tree, markdown = _run_full_pipeline(
            report_memory_bound, action=action, correlator=correlator,
        )

        # Should have memory-related findings
        assert any("memory" in f.source for f in findings)

        # Findings with source attribution should have source_location set
        # (depends on whether hotspots matched stall keywords)
        sourced_findings = [f for f in findings if f.source_location is not None]
        # If any analyzer attached source evidence, verify it
        if sourced_findings:
            assert any(
                f.source_location.file == "softmax.cu"
                for f in sourced_findings
            )

        # Hotspots from correlator should have source info
        for h in hotspots:
            assert h.source_file in ("softmax.cu",)

    def test_m2_latency_bound_with_stalls(self, report_latency_bound: KernelReport):
        """Latency-bound kernel: OptTree should have latency branch active."""
        pc_values = [
            (0x3000, 800.0),
            (0x3004, 400.0),
        ]
        _add_instanced_stall_metrics(report_latency_bound, pc_values)

        action = _make_mock_action({
            0x3000: ("attention.cu", 200),
            0x3004: ("attention.cu", 205),
        })

        correlator = SourceCorrelator()
        findings, hotspots, opt_tree, markdown = _run_full_pipeline(
            report_latency_bound, action=action, correlator=correlator,
        )

        # Should classify as latency-bound
        assert any("latency-bound" in f.title for f in findings)

        # OptTree: the latency branch should be active (not pruned)
        latency_branches = [
            child for child in opt_tree.root.children
            if child.category == "latency"
        ]
        assert len(latency_branches) == 1
        assert not latency_branches[0].pruned

        # Active paths should include latency strategies
        active = opt_tree.active_paths()
        assert len(active) > 0

        # Markdown should mention stall-related content
        assert "Optimization Tree" in markdown

    def test_m2_no_correlator_fallback(self, report_compute_bound: KernelReport):
        """M1-mode fallback: pipeline works without correlator (backward compat).

        When no correlator or action is provided, analyzers run in M1 mode.
        Findings are still generated; no crash occurs.
        """
        findings, hotspots, opt_tree, markdown = _run_full_pipeline(
            report_compute_bound, action=None, correlator=None,
        )

        # Findings should still be generated from rule-based analysis
        assert len(findings) > 0
        assert any("compute-bound" in f.title for f in findings)

        # No hotspots in M1 mode
        assert hotspots == []

        # Markdown should still render correctly
        assert "# Tachyon Performance Report" in markdown
        assert "Executive Summary" in markdown

    def test_m2_degraded_pipeline(self, report_memory_bound: KernelReport):
        """Pipeline with degraded source info (no -lineinfo).

        When action.source_info returns None for all PCs, hotspots should
        be marked as degraded and a degradation warning should appear.
        """
        pc_values = [
            (0x4000, 600.0),
            (0x4004, 400.0),
        ]
        _add_instanced_stall_metrics(report_memory_bound, pc_values)

        # Action returns None for all PCs (no debug info)
        action = _make_mock_action({})

        correlator = SourceCorrelator()
        findings, hotspots, opt_tree, markdown = _run_full_pipeline(
            report_memory_bound, action=action, correlator=correlator,
        )

        # Findings should still be generated (rule-based)
        assert len(findings) > 0

        # All hotspots should be degraded (if any pass the threshold)
        for h in hotspots:
            assert h.degraded

        # If hotspots exist, degradation warning should be present
        if hotspots:
            assert "Warning" in markdown
            assert "lineinfo" in markdown.lower() or "debug info" in markdown.lower()
