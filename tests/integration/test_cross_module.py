"""Cross-module integration tests — verify Tachyon modules compose correctly.

Tests the data flow across module boundaries without requiring real .ncu-rep
files or GPU hardware.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tachyon.analyzers.base import AnalyzerRegistry
from tachyon.config.settings import TachyonConfig
from tachyon.diff.differ import ProfileDiffer
from tachyon.errors.handler import ErrorCode, ToolResult
from tachyon.models.finding import Finding, Severity
from tachyon.models.kernel import (
    DeviceInfo,
    KernelReport,
    LaunchParams,
    MetricValue,
)
from tachyon.report.terminal import TerminalReporter
from tachyon.tools.context import SessionContext
from tachyon.tools.registry import ToolRegistry
from tachyon.tree.opt_tree import OptimizationTree


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def full_kernel() -> KernelReport:
    """Kernel with enough metrics for most analyzers."""
    return KernelReport(
        kernel_name="integration_kernel",
        demangled_name="integration_kernel<float>",
        launch_params=LaunchParams(
            grid=(4096, 1, 1), block=(256, 1, 1),
            shared_mem_bytes=0, registers_per_thread=32,
        ),
        device_info=DeviceInfo(
            name="H100", compute_capability=(9, 0),
            sm_count=132, max_clock_mhz=1980,
            memory_bus_width=5120, peak_memory_bandwidth_gbps=3352.0,
        ),
        metrics={
            "sm__throughput.avg.pct_of_peak_sustained_elapsed":
                MetricValue(name="sm__throughput.avg.pct_of_peak_sustained_elapsed",
                            value=85.0, unit="%"),
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed":
                MetricValue(name="gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
                            value=30.0, unit="%"),
        },
    )


# ─── Tests ───────────────────────────────────────────────────────────


class TestCrossModule:

    def test_config_to_profiler_integration(self):
        """TachyonConfig creates valid config for profiler construction."""
        config = TachyonConfig()
        from tachyon.profiler.tool_path import ToolPathResolver

        resolver = ToolPathResolver(config)
        assert resolver is not None
        # Should not raise during construction

    def test_reader_to_analyzer_pipeline(self, full_kernel):
        """AnalyzerRegistry.run_all() produces findings from KernelReport."""
        registry = AnalyzerRegistry()
        registry.auto_register()

        findings = registry.run_all(full_kernel)
        assert isinstance(findings, list)
        assert len(findings) > 0
        assert all(isinstance(f, Finding) for f in findings)

    def test_analyzer_findings_to_report(self, full_kernel):
        """TerminalReporter renders findings from AnalyzerRegistry."""
        registry = AnalyzerRegistry()
        registry.auto_register()
        findings = registry.run_all(full_kernel)

        renderer = TerminalReporter()
        output = renderer.render(
            [full_kernel],
            {full_kernel.demangled_name: findings},
        )
        assert "integration_kernel" in output
        assert len(output) > 100  # Non-trivial output

    def test_kernel_to_opt_tree(self, full_kernel):
        """Findings build an OptimizationTree with active paths."""
        registry = AnalyzerRegistry()
        registry.auto_register()
        findings = registry.run_all(full_kernel)

        tree = OptimizationTree(findings)
        assert tree.root is not None
        md = tree.to_markdown()
        assert "Optimization Tree" in md

        paths = tree.active_paths()
        assert isinstance(paths, list)

    def test_tool_registry_all_tools(self, full_kernel):
        """All 9 tools register and have valid MCP/OpenAI/Anthropic schemas."""
        from tachyon.tools.analysis import register_analysis_tools
        from tachyon.tools.data_query import register_data_query_tools
        from tachyon.tools.source import register_source_tools

        analyzer_registry = AnalyzerRegistry()
        analyzer_registry.auto_register()
        session = SessionContext(
            kernels=[full_kernel],
            registry=analyzer_registry,
        )

        tool_registry = ToolRegistry()
        register_data_query_tools(tool_registry, session)
        register_source_tools(tool_registry, session)
        register_analysis_tools(tool_registry, session)

        names = tool_registry.tool_names()
        assert len(names) == 9

        for td in tool_registry.all_definitions():
            mcp = td.to_mcp()
            assert "name" in mcp
            assert "description" in mcp
            assert "inputSchema" in mcp

            oai = td.to_openai()
            assert oai["type"] == "function"

            anth = td.to_anthropic()
            assert "input_schema" in anth

    def test_diff_with_real_kernel_data(self, full_kernel):
        """ProfileDiffer works with full KernelReport objects."""
        after_kernel = KernelReport(
            kernel_name=full_kernel.kernel_name,
            demangled_name=full_kernel.demangled_name,
            launch_params=full_kernel.launch_params,
            device_info=full_kernel.device_info,
            metrics={
                "sm__throughput.avg.pct_of_peak_sustained_elapsed":
                    MetricValue(
                        name="sm__throughput.avg.pct_of_peak_sustained_elapsed",
                        value=92.0, unit="%",
                    ),
                "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed":
                    MetricValue(
                        name="gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
                        value=25.0, unit="%",
                    ),
            },
        )

        differ = ProfileDiffer()
        diffs = differ.diff([full_kernel], [after_kernel])
        assert len(diffs) == 1
        assert diffs[0].kernel_name == "integration_kernel"
        assert len(diffs[0].metric_deltas) == 2

        summary = differ.summary(diffs)
        assert "integration_kernel" in summary

    def test_session_context_kernel_lookup(self, full_kernel):
        """SessionContext provides kernel lookup by index."""
        session = SessionContext(kernels=[full_kernel])
        assert session.kernel_count == 1
        assert session.get_kernel(0) is full_kernel

        with pytest.raises(IndexError):
            session.get_kernel(1)

    def test_i18n_initialization(self):
        """i18n module initializes and returns strings."""
        from tachyon.i18n import init, t, current_lang

        init("en")
        assert current_lang() == "en"
        # t() with missing key returns the key itself
        result = t("nonexistent.key")
        assert result == "nonexistent.key"

    def test_error_propagation_chain(self):
        """ToolResult errors maintain structure across module boundaries."""
        result = ToolResult.fail(
            ErrorCode.TOOL_NOT_FOUND,
            "ncu not found",
            suggestion="Install CUDA Toolkit",
        )
        assert not result.success
        assert result.error is not None
        assert result.error.code == ErrorCode.TOOL_NOT_FOUND
        assert result.error.message == "ncu not found"
        assert result.error.suggestion == "Install CUDA Toolkit"

        # Chain with ok result
        ok = ToolResult.ok({"test": "data"})
        assert ok.success
        assert ok.data == {"test": "data"}

    def test_sdk_imports(self):
        """All public SDK imports work correctly."""
        from tachyon import (
            NcuProfiler,
            ProfilingStrategy,
            TachyonConfig,
            ToolPathResolver,
        )

        assert NcuProfiler is not None
        assert ProfilingStrategy is not None
        assert TachyonConfig is not None
        assert ToolPathResolver is not None
