"""Integration tests — end-to-end analysis pipeline without real NCU."""

from tachyon.analyzers.base import AnalyzerRegistry
from tachyon.models.kernel import KernelReport
from tachyon.report.terminal import TerminalReporter


class TestFullPipeline:
    def test_compute_bound_pipeline(self, report_compute_bound: KernelReport):
        """End-to-end: compute-bound KernelReport -> Analyzers -> TerminalReport."""
        registry = AnalyzerRegistry()
        registry.auto_register()
        findings = registry.run_all(report_compute_bound)

        reporter = TerminalReporter()
        output = reporter.render_single_kernel(report_compute_bound, findings)
        assert "COMPUTE-BOUND" in output
        assert len(output) > 100

    def test_memory_bound_pipeline(self, report_memory_bound: KernelReport):
        """Memory-bound kernel produces memory-related findings."""
        registry = AnalyzerRegistry()
        registry.auto_register()
        findings = registry.run_all(report_memory_bound)

        sources = {f.source for f in findings}
        assert "memory" in sources

        reporter = TerminalReporter()
        output = reporter.render_single_kernel(report_memory_bound, findings)
        assert "MEMORY-BOUND" in output or "memory" in output.lower()
        assert len(output) > 100

    def test_latency_bound_pipeline(self, report_latency_bound: KernelReport):
        """Latency-bound kernel produces stall findings."""
        registry = AnalyzerRegistry()
        registry.auto_register()
        findings = registry.run_all(report_latency_bound)

        assert len(findings) > 0
        sources = {f.source for f in findings}
        assert "warp_stall" in sources

        reporter = TerminalReporter()
        output = reporter.render_single_kernel(report_latency_bound, findings)
        assert len(output) > 100

    def test_multi_kernel_pipeline(
        self,
        report_compute_bound: KernelReport,
        report_memory_bound: KernelReport,
        report_latency_bound: KernelReport,
    ):
        """Multiple kernels analyzed and rendered."""
        reports = [report_compute_bound, report_memory_bound, report_latency_bound]

        registry = AnalyzerRegistry()
        registry.auto_register()

        findings_map = {}
        for report in reports:
            findings = registry.run_all(report)
            findings_map[report.demangled_name] = findings

        reporter = TerminalReporter()
        output = reporter.render(reports, findings_map)

        assert "Kernels analyzed" in output
        assert "compute_kernel" in output
        assert "memory_kernel" in output
        assert "latency_kernel" in output

    def test_minimal_kernel_graceful(self, report_minimal: KernelReport):
        """Minimal kernel with no metrics produces no crash."""
        registry = AnalyzerRegistry()
        registry.auto_register()
        findings = registry.run_all(report_minimal)

        reporter = TerminalReporter()
        output = reporter.render_single_kernel(report_minimal, findings)
        # Should not crash, should show "no significant findings"
        assert "No significant findings" in output or len(output) > 10

    def test_findings_severity_order(self, report_latency_bound: KernelReport):
        """Pipeline output is sorted CRITICAL > WARNING > INFO."""
        registry = AnalyzerRegistry()
        registry.auto_register()
        findings = registry.run_all(report_latency_bound)
        for i in range(len(findings) - 1):
            assert findings[i].severity >= findings[i + 1].severity

    def test_with_nvrules(self, report_with_rules: KernelReport):
        """Pipeline includes NvRules findings."""
        registry = AnalyzerRegistry()
        registry.auto_register(include_nvrules=True)
        findings = registry.run_all(report_with_rules)

        nvrule_findings = [f for f in findings if f.source == "nvrules"]
        assert len(nvrule_findings) > 0

        reporter = TerminalReporter()
        output = reporter.render_single_kernel(report_with_rules, findings)
        assert "NCU" in output
