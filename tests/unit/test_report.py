"""Unit tests for TerminalReporter."""
from tachyon.models.finding import Finding, Severity
from tachyon.models.kernel import KernelReport
from tachyon.report.terminal import TerminalReporter


class TestTerminalReporter:
    def test_render_single_kernel(self, report_compute_bound: KernelReport):
        """Render a single kernel with findings."""
        findings = [
            Finding(
                severity=Severity.WARNING,
                title="Kernel is compute-bound",
                detail="SM=85%, DRAM=30%",
                action="Reduce arithmetic",
                source="roofline",
            ),
        ]
        reporter = TerminalReporter()
        output = reporter.render_single_kernel(report_compute_bound, findings)
        assert "compute_kernel" in output
        assert "compute-bound" in output

    def test_render_no_findings(self, report_minimal: KernelReport):
        """Render a kernel with no findings shows 'no significant findings'."""
        reporter = TerminalReporter()
        output = reporter.render_single_kernel(report_minimal, [])
        assert "No significant findings" in output or "no significant" in output.lower()

    def test_render_multiple_kernels(
        self,
        report_compute_bound: KernelReport,
        report_memory_bound: KernelReport,
    ):
        """Render multiple kernels."""
        reporter = TerminalReporter()
        findings_map = {
            "compute_kernel<float>": [
                Finding(severity=Severity.INFO, title="Compute info",
                        detail="d", action="a", source="s"),
            ],
            "memory_kernel<float>": [
                Finding(severity=Severity.WARNING, title="Memory warning",
                        detail="d", action="a", source="s"),
            ],
        }
        output = reporter.render(
            [report_compute_bound, report_memory_bound], findings_map
        )
        assert "Tachyon Performance Analysis" in output
        assert "Kernels analyzed" in output

    def test_render_conclusion_first(self, report_latency_bound: KernelReport):
        """Critical/warning findings appear in Key Findings tree."""
        findings = [
            Finding(severity=Severity.CRITICAL, title="Critical issue",
                    detail="d", action="a", source="s"),
            Finding(severity=Severity.WARNING, title="Warning issue",
                    detail="d", action="a", source="s"),
            Finding(severity=Severity.INFO, title="Info issue",
                    detail="d", action="a", source="s"),
        ]
        reporter = TerminalReporter()
        output = reporter.render_single_kernel(report_latency_bound, findings)
        assert "Key Findings" in output
        assert "CRITICAL" in output

    def test_render_has_table(self, report_compute_bound: KernelReport):
        """Output should contain finding details in table format."""
        findings = [
            Finding(severity=Severity.WARNING, title="Test finding",
                    detail="Test detail", action="Test action", source="test_src"),
        ]
        reporter = TerminalReporter()
        output = reporter.render_single_kernel(report_compute_bound, findings)
        assert "Test finding" in output
        assert "test_src" in output

    def test_render_kernel_header(self, report_compute_bound: KernelReport):
        """Kernel header should show grid, block, registers."""
        reporter = TerminalReporter()
        output = reporter.render_single_kernel(report_compute_bound, [
            Finding(severity=Severity.INFO, title="t", detail="d", action="a", source="s"),
        ])
        assert "4096" in output  # grid size
        assert "256" in output   # block size
        assert "32" in output    # registers

    def test_long_action_rendered(self, report_compute_bound: KernelReport):
        """Very long action text should still be included in output."""
        findings = [
            Finding(severity=Severity.WARNING, title="Test",
                    detail="d", action="A" * 200, source="s"),
        ]
        reporter = TerminalReporter()
        output = reporter.render_single_kernel(report_compute_bound, findings)
        # Action text should appear in the output
        assert "AAAA" in output
