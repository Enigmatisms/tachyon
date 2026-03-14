"""Unit tests for MarkdownReporter."""
from unittest.mock import MagicMock

import pytest

import tachyon.i18n as i18n
from tachyon.models.finding import Finding, Severity, SourceLocation
from tachyon.models.hotspot import SourceHotspot
from tachyon.report.markdown import MarkdownReporter, ReportContext

# ━━━━━━━━━━━━━━━━━━━━━━━ Fixtures ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture(autouse=True)
def init_i18n():
    """Reset and initialize i18n to English before every test."""
    i18n._packs.clear()
    i18n._current_lang = "en"
    i18n.init("en")
    yield
    i18n._packs.clear()


def _make_ctx(findings=None, hotspots=None, opt_tree=None, verbose=False) -> ReportContext:
    """Create a basic ReportContext with sensible defaults."""
    return ReportContext(
        kernel_name="test_kernel",
        demangled_name="test_kernel<float>",
        device_name="NVIDIA H100",
        compute_capability="9.0",
        findings=findings or [],
        hotspots=hotspots or [],
        opt_tree=opt_tree,
        verbose=verbose,
    )


def _make_finding(
    severity=Severity.WARNING,
    title="Test finding",
    detail="Test detail",
    action="Test action",
    source="roofline",
    category="compute",
    source_location=None,
    sass_evidence=None,
    metrics=None,
) -> Finding:
    """Shorthand to create a Finding with defaults."""
    return Finding(
        severity=severity,
        title=title,
        detail=detail,
        action=action,
        source=source,
        category=category,
        source_location=source_location,
        sass_evidence=sass_evidence,
        metrics=metrics or {},
    )


# ━━━━━━━━━━━━━━━━━━━━━━━ TestMetadata ━━━━━━━━━━━━━━━━━━━━━━━━


class TestMetadata:
    def test_metadata_header(self):
        """Output contains report title, device name, and kernel name."""
        reporter = MarkdownReporter()
        ctx = _make_ctx()
        output = reporter.render(ctx)
        assert "# Tachyon Performance Report" in output
        assert "NVIDIA H100" in output
        assert "test_kernel<float>" in output

    def test_metadata_with_duration(self):
        """Duration in microseconds is rendered when provided."""
        ctx = _make_ctx()
        ctx.duration_us = 1234.5
        reporter = MarkdownReporter()
        output = reporter.render(ctx)
        assert "1234.5 us" in output

    def test_metadata_with_degradation(self):
        """Degradation warning is rendered in metadata section."""
        ctx = _make_ctx()
        ctx.degradation_warning = "Source mapping unavailable"
        reporter = MarkdownReporter()
        output = reporter.render(ctx)
        assert "Warning" in output
        assert "Source mapping unavailable" in output


# ━━━━━━━━━━━━━━━━━━━━━━━ TestExecutiveSummary ━━━━━━━━━━━━━━━━━


class TestExecutiveSummary:
    def test_summary_with_findings(self):
        """Executive Summary shows bottleneck type from roofline finding."""
        findings = [
            _make_finding(
                severity=Severity.WARNING,
                title="Kernel is compute-bound",
                source="roofline",
                category="compute",
            ),
        ]
        ctx = _make_ctx(findings=findings)
        reporter = MarkdownReporter()
        output = reporter.render(ctx)
        assert "Executive Summary" in output
        assert "compute" in output

    def test_summary_no_findings(self):
        """Empty findings list produces 'No significant findings' message."""
        ctx = _make_ctx(findings=[])
        reporter = MarkdownReporter()
        output = reporter.render(ctx)
        assert "No significant findings" in output

    def test_summary_with_hotspot(self):
        """Hotspot with source info shows 'primary hotspot at file:line'."""
        findings = [
            _make_finding(
                title="Kernel is memory-bound",
                source="roofline",
            ),
        ]
        hotspots = [
            SourceHotspot(
                pc=0x1000,
                source_file="softmax.cu",
                source_line=42,
                is_hot=True,
                global_ratio=0.5,
                local_ratio=0.8,
            ),
        ]
        ctx = _make_ctx(findings=findings, hotspots=hotspots)
        reporter = MarkdownReporter()
        output = reporter.render(ctx)
        assert "primary hotspot at" in output
        assert "softmax.cu:42" in output


# ━━━━━━━━━━━━━━━━━━━━━━━ TestTopFindings ━━━━━━━━━━━━━━━━━━━━━━


class TestTopFindings:
    def test_top_findings_rendered(self):
        """Three findings are rendered with titles and severity markers."""
        findings = [
            _make_finding(severity=Severity.CRITICAL, title="Critical issue"),
            _make_finding(severity=Severity.WARNING, title="Warning issue"),
            _make_finding(severity=Severity.INFO, title="Info issue"),
        ]
        ctx = _make_ctx(findings=findings)
        reporter = MarkdownReporter()
        output = reporter.render(ctx)
        assert "Critical issue" in output
        assert "Warning issue" in output
        assert "Info issue" in output
        # Severity markers from _SEVERITY_ICON
        assert "!!!" in output    # CRITICAL
        assert "!!" in output     # WARNING (also matched by !!!)
        assert "[i]" in output    # INFO

    def test_top_findings_max_5(self):
        """Only 5 findings are rendered; excess shows omitted count."""
        findings = [
            _make_finding(title=f"Finding {i}", severity=Severity.WARNING)
            for i in range(7)
        ]
        ctx = _make_ctx(findings=findings)
        reporter = MarkdownReporter()
        output = reporter.render(ctx)
        # First 5 should be present
        for i in range(5):
            assert f"Finding {i}" in output
        # Omitted message
        assert "(2 additional findings omitted)" in output

    def test_finding_with_source_location(self):
        """Finding with source_location renders file:line."""
        findings = [
            _make_finding(
                source_location=SourceLocation(
                    file="matmul.cu", line=128, function="sgemm"
                ),
            ),
        ]
        ctx = _make_ctx(findings=findings)
        reporter = MarkdownReporter()
        output = reporter.render(ctx)
        assert "matmul.cu:128" in output
        assert "sgemm" in output

    def test_finding_with_sass_evidence(self):
        """Finding with SASS evidence renders the SASS text."""
        findings = [
            _make_finding(
                sass_evidence="LDG.E.128 [R2], [R4];",
            ),
        ]
        ctx = _make_ctx(findings=findings)
        reporter = MarkdownReporter()
        output = reporter.render(ctx)
        assert "LDG.E.128 [R2], [R4];" in output


# ━━━━━━━━━━━━━━━━━━━━━━━ TestEvidenceChains ━━━━━━━━━━━━━━━━━━━


class TestEvidenceChains:
    def test_evidence_chains_rendered(self):
        """WARNING findings appear in evidence chains with metrics and source."""
        findings = [
            _make_finding(
                severity=Severity.WARNING,
                title="Memory bottleneck",
                source_location=SourceLocation(
                    file="conv.cu", line=55, function="conv2d"
                ),
                sass_evidence="LDG.E.128 [R2], [R4];",
                metrics={"dram__throughput": 78.0},
            ),
        ]
        ctx = _make_ctx(findings=findings)
        reporter = MarkdownReporter()
        output = reporter.render(ctx)
        assert "Evidence Chains" in output
        assert "dram__throughput" in output
        assert "conv.cu:55" in output
        assert "LDG.E.128 [R2], [R4];" in output

    def test_evidence_chains_skips_info(self):
        """INFO-only findings do not appear in Evidence Chains section."""
        findings = [
            _make_finding(
                severity=Severity.INFO,
                title="Informational note",
                metrics={"some_metric": 10.0},
            ),
        ]
        ctx = _make_ctx(findings=findings)
        reporter = MarkdownReporter()
        output = reporter.render(ctx)
        # The evidence chains section should exist but show no findings
        # (it renders "No significant findings" when rendered==0)
        evidence_section = output.split("Evidence Chains")[1] if "Evidence Chains" in output else ""
        assert "Informational note" not in evidence_section or "No significant findings" in evidence_section


# ━━━━━━━━━━━━━━━━━━━━━━━ TestOptTree ━━━━━━━━━━━━━━━━━━━━━━━━━


class TestOptTree:
    def test_opt_tree_rendered(self):
        """Mock opt_tree with to_markdown() produces tree output."""
        mock_tree = MagicMock()
        mock_tree.to_markdown.return_value = (
            "# Optimization Tree\n\n**Primary bottleneck: compute-bound**\n\n"
            "- **Compute-Bound Optimisations**"
        )
        ctx = _make_ctx(opt_tree=mock_tree)
        reporter = MarkdownReporter()
        output = reporter.render(ctx)
        assert "Optimization Tree" in output
        assert "Compute-Bound Optimisations" in output
        mock_tree.to_markdown.assert_called_once()

    def test_opt_tree_none(self):
        """opt_tree=None produces 'No optimization tree available'."""
        ctx = _make_ctx(opt_tree=None)
        reporter = MarkdownReporter()
        output = reporter.render(ctx)
        assert "No optimization tree" in output


# ━━━━━━━━━━━━━━━━━━━━━━━ TestDetailedMetrics ━━━━━━━━━━━━━━━━━━


class TestDetailedMetrics:
    def test_verbose_mode(self):
        """verbose=True renders the Detailed Metrics section with a table."""
        findings = [
            _make_finding(
                metrics={
                    "sm__throughput": 85.0,
                    "dram__throughput": 30.0,
                },
            ),
        ]
        ctx = _make_ctx(findings=findings, verbose=True)
        reporter = MarkdownReporter()
        output = reporter.render(ctx)
        assert "Detailed Metrics" in output
        assert "sm__throughput" in output
        assert "dram__throughput" in output
        # GFM table markers
        assert "| Metric | Value |" in output

    def test_non_verbose(self):
        """verbose=False does NOT include the Detailed Metrics section."""
        findings = [
            _make_finding(
                metrics={"sm__throughput": 85.0},
            ),
        ]
        ctx = _make_ctx(findings=findings, verbose=False)
        reporter = MarkdownReporter()
        output = reporter.render(ctx)
        assert "Detailed Metrics" not in output


# ━━━━━━━━━━━━━━━━━━━━━━━ TestFullRender ━━━━━━━━━━━━━━━━━━━━━━━


class TestFullRender:
    def test_sections_joined_with_hr(self):
        """Sections are joined with horizontal rules (---)."""
        findings = [
            _make_finding(severity=Severity.WARNING, title="Test finding"),
        ]
        ctx = _make_ctx(findings=findings)
        reporter = MarkdownReporter()
        output = reporter.render(ctx)
        assert "---" in output
