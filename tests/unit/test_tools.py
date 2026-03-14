"""Comprehensive unit tests for the Tachyon Tool layer.

Covers:
  - ToolDefinition: OpenAI / Anthropic / MCP format conversion
  - ToolRegistry: registration, lookup, execution lifecycle
  - SessionContext: kernel lookup, bounds checking
  - Data Query Tools: list_kernels, get_kernel_metrics, get_kernel_summary, get_ncu_rule_results
  - Source Tools: error paths when correlator/instanced metrics missing
  - Analysis Tools: run_analysis (all + single + unknown), get_optimization_tree
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tachyon.analyzers.base import AnalyzerRegistry
from tachyon.errors.handler import ErrorCode, ToolResult
from tachyon.models.kernel import DeviceInfo, KernelReport, LaunchParams
from tachyon.tools.analysis import register_analysis_tools
from tachyon.tools.context import SessionContext
from tachyon.tools.data_query import register_data_query_tools
from tachyon.tools.registry import ToolDefinition, ToolRegistry
from tachyon.tools.source import register_source_tools

# ━━━━━━━━━━━━━━━━━━━━━━━ Helpers ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _make_tool(
    name: str = "test_tool",
    description: str = "A test tool.",
    parameters: dict[str, Any] | None = None,
    handler: Any = None,
) -> ToolDefinition:
    """Convenience factory for ToolDefinition."""
    return ToolDefinition(
        name=name,
        description=description,
        parameters=parameters or {"type": "object", "properties": {}, "required": []},
        handler=handler,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━ TestToolDefinition ━━━━━━━━━━━━━━━━━━━━


class TestToolDefinition:
    """Tests for ToolDefinition schema conversion methods."""

    def test_to_openai_format(self):
        """to_openai() must wrap in {type: 'function', function: ...}."""
        td = _make_tool(
            name="my_tool",
            description="Desc",
            parameters={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
        )
        result = td.to_openai()
        assert result["type"] == "function"
        fn = result["function"]
        assert fn["name"] == "my_tool"
        assert fn["description"] == "Desc"
        assert fn["parameters"]["properties"]["x"]["type"] == "integer"
        assert "required" in fn["parameters"]

    def test_to_openai_has_no_extra_keys(self):
        """OpenAI format should contain exactly type + function."""
        td = _make_tool()
        result = td.to_openai()
        assert set(result.keys()) == {"type", "function"}
        assert set(result["function"].keys()) == {"name", "description", "parameters"}

    def test_to_anthropic_format(self):
        """to_anthropic() must produce {name, description, input_schema}."""
        td = _make_tool(
            name="anthr_tool",
            description="Anthropic desc",
            parameters={"type": "object", "properties": {"y": {"type": "string"}}},
        )
        result = td.to_anthropic()
        assert result["name"] == "anthr_tool"
        assert result["description"] == "Anthropic desc"
        assert result["input_schema"]["properties"]["y"]["type"] == "string"

    def test_to_anthropic_has_correct_keys(self):
        """Anthropic format should contain exactly name, description, input_schema."""
        td = _make_tool()
        result = td.to_anthropic()
        assert set(result.keys()) == {"name", "description", "input_schema"}

    def test_to_mcp_format(self):
        """to_mcp() must produce {name, description, inputSchema}."""
        td = _make_tool(
            name="mcp_tool",
            description="MCP desc",
            parameters={"type": "object", "properties": {"z": {"type": "boolean"}}},
        )
        result = td.to_mcp()
        assert result["name"] == "mcp_tool"
        assert result["description"] == "MCP desc"
        assert result["inputSchema"]["properties"]["z"]["type"] == "boolean"

    def test_to_mcp_has_correct_keys(self):
        """MCP format should contain exactly name, description, inputSchema."""
        td = _make_tool()
        result = td.to_mcp()
        assert set(result.keys()) == {"name", "description", "inputSchema"}

    def test_all_formats_share_same_parameters(self):
        """All three formats should reference the exact same parameter schema."""
        params = {"type": "object", "properties": {"a": {"type": "number"}}, "required": ["a"]}
        td = _make_tool(parameters=params)
        assert td.to_openai()["function"]["parameters"] is params
        assert td.to_anthropic()["input_schema"] is params
        assert td.to_mcp()["inputSchema"] is params


# ━━━━━━━━━━━━━━━━━━━━━━━ TestToolRegistry ━━━━━━━━━━━━━━━━━━━━━━


class TestToolRegistry:
    """Tests for ToolRegistry lifecycle: register, get, execute."""

    def test_register_and_get(self):
        """register() then get() should return the same ToolDefinition."""
        reg = ToolRegistry()
        tool = _make_tool(name="alpha")
        reg.register(tool)
        assert reg.get("alpha") is tool

    def test_get_unknown_returns_none(self):
        """get() on an unregistered name should return None."""
        reg = ToolRegistry()
        assert reg.get("nonexistent") is None

    def test_register_duplicate_raises(self):
        """Registering the same name twice should raise ValueError."""
        reg = ToolRegistry()
        reg.register(_make_tool(name="dup"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(_make_tool(name="dup"))

    def test_all_definitions_returns_list(self):
        """all_definitions() should return all registered tools as a list."""
        reg = ToolRegistry()
        reg.register(_make_tool(name="a"))
        reg.register(_make_tool(name="b"))
        reg.register(_make_tool(name="c"))
        defs = reg.all_definitions()
        assert isinstance(defs, list)
        assert len(defs) == 3

    def test_all_definitions_is_copy(self):
        """all_definitions() should return a copy, not the internal dict values view."""
        reg = ToolRegistry()
        reg.register(_make_tool(name="x"))
        defs1 = reg.all_definitions()
        defs1.clear()
        # Internal state should not be affected
        assert len(reg.all_definitions()) == 1

    def test_tool_names(self):
        """tool_names() should return list of registered names."""
        reg = ToolRegistry()
        reg.register(_make_tool(name="first"))
        reg.register(_make_tool(name="second"))
        names = reg.tool_names()
        assert "first" in names
        assert "second" in names
        assert len(names) == 2

    @pytest.mark.asyncio
    async def test_execute_valid_tool(self):
        """execute() with a registered tool should invoke its handler."""
        handler = AsyncMock(return_value=ToolResult.ok({"answer": 42}))
        reg = ToolRegistry()
        reg.register(_make_tool(name="calc", handler=handler))
        result = await reg.execute("calc", {"x": 1})
        assert result.success is True
        assert result.data == {"answer": 42}
        handler.assert_awaited_once_with(x=1)

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_returns_error(self):
        """execute() with unknown name should return TOOL_NOT_FOUND error."""
        reg = ToolRegistry()
        result = await reg.execute("ghost", {})
        assert result.success is False
        assert result.error.code == ErrorCode.TOOL_NOT_FOUND
        assert "ghost" in result.error.message

    @pytest.mark.asyncio
    async def test_execute_no_handler_returns_error(self):
        """execute() when handler is None should return UNKNOWN error."""
        reg = ToolRegistry()
        reg.register(_make_tool(name="noop", handler=None))
        result = await reg.execute("noop", {})
        assert result.success is False
        assert "no handler" in result.error.message

    @pytest.mark.asyncio
    async def test_execute_handler_raises_returns_error(self):
        """execute() when handler raises an exception should return error."""
        async def bad_handler(**kwargs):
            raise RuntimeError("boom")

        reg = ToolRegistry()
        reg.register(_make_tool(name="bad", handler=bad_handler))
        result = await reg.execute("bad", {})
        assert result.success is False
        assert "boom" in result.error.message

    @pytest.mark.asyncio
    async def test_execute_passes_kwargs(self):
        """execute() should unpack arguments dict as **kwargs to handler."""
        captured = {}

        async def capture_handler(**kwargs):
            captured.update(kwargs)
            return ToolResult.ok("ok")

        reg = ToolRegistry()
        reg.register(_make_tool(name="cap", handler=capture_handler))
        await reg.execute("cap", {"kernel_id": 0, "metric_names": ["a", "b"]})
        assert captured == {"kernel_id": 0, "metric_names": ["a", "b"]}


# ━━━━━━━━━━━━━━━━━━━━━━━ TestSessionContext ━━━━━━━━━━━━━━━━━━━━


class TestSessionContext:
    """Tests for SessionContext kernel lookup."""

    def test_get_kernel_valid_index(self, report_compute_bound: KernelReport):
        """get_kernel(0) should return the first kernel."""
        ctx = SessionContext(kernels=[report_compute_bound])
        k = ctx.get_kernel(0)
        assert k is report_compute_bound

    def test_get_kernel_multiple(
        self,
        report_compute_bound: KernelReport,
        report_memory_bound: KernelReport,
    ):
        """get_kernel() should work for all valid indices."""
        ctx = SessionContext(kernels=[report_compute_bound, report_memory_bound])
        assert ctx.get_kernel(0) is report_compute_bound
        assert ctx.get_kernel(1) is report_memory_bound

    def test_get_kernel_negative_index_raises(self, report_compute_bound: KernelReport):
        """get_kernel(-1) should raise IndexError."""
        ctx = SessionContext(kernels=[report_compute_bound])
        with pytest.raises(IndexError, match="out of range"):
            ctx.get_kernel(-1)

    def test_get_kernel_out_of_range_raises(self, report_compute_bound: KernelReport):
        """get_kernel(N) for N >= len should raise IndexError."""
        ctx = SessionContext(kernels=[report_compute_bound])
        with pytest.raises(IndexError, match="out of range"):
            ctx.get_kernel(1)

    def test_kernel_count_property(
        self,
        report_compute_bound: KernelReport,
        report_memory_bound: KernelReport,
        report_latency_bound: KernelReport,
    ):
        """kernel_count should reflect the number of loaded kernels."""
        ctx = SessionContext(kernels=[
            report_compute_bound, report_memory_bound, report_latency_bound,
        ])
        assert ctx.kernel_count == 3

    def test_kernel_count_empty(self):
        """kernel_count == 0 for empty kernel list."""
        ctx = SessionContext(kernels=[])
        assert ctx.kernel_count == 0


# ━━━━━━━━━━━━━━━━━━━━━━━ TestDataQueryTools ━━━━━━━━━━━━━━━━━━━━


class TestDataQueryTools:
    """Tests for the 4 data query tools: list, metrics, summary, rules."""

    @pytest.fixture()
    def data_registry(
        self,
        report_compute_bound: KernelReport,
        report_memory_bound: KernelReport,
    ) -> ToolRegistry:
        """Build a ToolRegistry with data query tools for two kernels."""
        ctx = SessionContext(kernels=[report_compute_bound, report_memory_bound])
        reg = ToolRegistry()
        register_data_query_tools(reg, ctx)
        return reg

    @pytest.fixture()
    def rules_registry(self, report_with_rules: KernelReport) -> ToolRegistry:
        """Build a ToolRegistry with data query tools for the rules fixture."""
        ctx = SessionContext(kernels=[report_with_rules])
        reg = ToolRegistry()
        register_data_query_tools(reg, ctx)
        return reg

    # --- list_kernels ---

    @pytest.mark.asyncio
    async def test_list_kernels_count(self, data_registry: ToolRegistry):
        """list_kernels should return 2 kernels."""
        result = await data_registry.execute("list_kernels", {})
        assert result.success is True
        assert len(result.data) == 2

    @pytest.mark.asyncio
    async def test_list_kernels_fields(self, data_registry: ToolRegistry):
        """Each kernel entry should contain kernel_id, name, grid, block, registers."""
        result = await data_registry.execute("list_kernels", {})
        entry = result.data[0]
        assert "kernel_id" in entry
        assert "name" in entry
        assert "grid" in entry
        assert "block" in entry
        assert "registers" in entry
        assert entry["kernel_id"] == 0

    @pytest.mark.asyncio
    async def test_list_kernels_uses_demangled_name(self, data_registry: ToolRegistry):
        """list_kernels should prefer demangled_name over kernel_name."""
        result = await data_registry.execute("list_kernels", {})
        assert result.data[0]["name"] == "compute_kernel<float>"

    # --- get_kernel_metrics ---

    @pytest.mark.asyncio
    async def test_get_kernel_metrics_all(self, data_registry: ToolRegistry):
        """get_kernel_metrics without filter should return all metrics."""
        result = await data_registry.execute("get_kernel_metrics", {"kernel_id": 0})
        assert result.success is True
        # compute_bound fixture has 2 metrics
        assert len(result.data) == 2
        for _name, entry in result.data.items():
            assert "value" in entry
            assert "unit" in entry

    @pytest.mark.asyncio
    async def test_get_kernel_metrics_filtered(self, data_registry: ToolRegistry):
        """get_kernel_metrics with metric_names should filter output."""
        result = await data_registry.execute(
            "get_kernel_metrics",
            {
                "kernel_id": 0,
                "metric_names": ["sm__throughput.avg.pct_of_peak_sustained_elapsed"],
            },
        )
        assert result.success is True
        assert len(result.data) == 1
        val = result.data["sm__throughput.avg.pct_of_peak_sustained_elapsed"]
        assert val["value"] == 85.0
        assert val["unit"] == "%"

    @pytest.mark.asyncio
    async def test_get_kernel_metrics_missing_metric_returns_none(self, data_registry: ToolRegistry):
        """Requesting a non-existent metric name should return None for that key."""
        result = await data_registry.execute(
            "get_kernel_metrics",
            {"kernel_id": 0, "metric_names": ["nonexistent_metric"]},
        )
        assert result.success is True
        assert result.data["nonexistent_metric"] is None

    @pytest.mark.asyncio
    async def test_get_kernel_metrics_invalid_id(self, data_registry: ToolRegistry):
        """get_kernel_metrics with out-of-range kernel_id should return error."""
        result = await data_registry.execute("get_kernel_metrics", {"kernel_id": 99})
        assert result.success is False
        assert result.error.code == ErrorCode.METRIC_NOT_FOUND

    @pytest.mark.asyncio
    async def test_get_kernel_metrics_second_kernel(self, data_registry: ToolRegistry):
        """get_kernel_metrics for kernel_id=1 should return memory-bound metrics."""
        result = await data_registry.execute("get_kernel_metrics", {"kernel_id": 1})
        assert result.success is True
        # memory_bound fixture has 7 metrics
        assert len(result.data) >= 5

    # --- get_kernel_summary ---

    @pytest.mark.asyncio
    async def test_get_kernel_summary(self, data_registry: ToolRegistry):
        """get_kernel_summary should return structured summary dict."""
        result = await data_registry.execute("get_kernel_summary", {"kernel_id": 0})
        assert result.success is True
        d = result.data
        assert d["name"] == "compute_kernel<float>"
        assert "launch" in d
        assert "device" in d
        assert "key_metrics" in d

    @pytest.mark.asyncio
    async def test_get_kernel_summary_launch_params(self, data_registry: ToolRegistry):
        """Summary launch block should include grid, block, shared_mem, registers, total_threads."""
        result = await data_registry.execute("get_kernel_summary", {"kernel_id": 0})
        launch = result.data["launch"]
        assert launch["grid"] == [4096, 1, 1]
        assert launch["block"] == [256, 1, 1]
        assert "shared_mem" in launch
        assert "registers" in launch
        assert launch["total_threads"] == 4096 * 256

    @pytest.mark.asyncio
    async def test_get_kernel_summary_device_info(self, data_registry: ToolRegistry):
        """Summary device block should include name, cc, sm_count."""
        result = await data_registry.execute("get_kernel_summary", {"kernel_id": 0})
        device = result.data["device"]
        assert "H100" in device["name"]
        assert device["cc"] == "9.0"
        assert device["sm_count"] == 132

    @pytest.mark.asyncio
    async def test_get_kernel_summary_key_metrics(self, data_registry: ToolRegistry):
        """Summary key_metrics should contain sm_pct and dram_pct for compute_bound fixture."""
        result = await data_registry.execute("get_kernel_summary", {"kernel_id": 0})
        km = result.data["key_metrics"]
        assert km["sm_pct"] == 85.0
        assert km["dram_pct"] == 30.0

    @pytest.mark.asyncio
    async def test_get_kernel_summary_invalid_id(self, data_registry: ToolRegistry):
        """get_kernel_summary with bad kernel_id should return error."""
        result = await data_registry.execute("get_kernel_summary", {"kernel_id": -1})
        assert result.success is False

    # --- get_ncu_rule_results ---

    @pytest.mark.asyncio
    async def test_get_ncu_rule_results_no_rules(self, data_registry: ToolRegistry):
        """Kernel with no rules should return empty list."""
        result = await data_registry.execute("get_ncu_rule_results", {"kernel_id": 0})
        assert result.success is True
        assert result.data == []

    @pytest.mark.asyncio
    async def test_get_ncu_rule_results_with_rules(self, rules_registry: ToolRegistry):
        """Kernel with NCU rules should return all rule results."""
        result = await rules_registry.execute("get_ncu_rule_results", {"kernel_id": 0})
        assert result.success is True
        assert len(result.data) == 4
        rule_names = {r["rule"] for r in result.data}
        assert "SpeedOfLight" in rule_names
        assert "MemoryWorkloadAnalysis" in rule_names
        assert "ComputeWorkloadAnalysis" in rule_names
        assert "Occupancy" in rule_names

    @pytest.mark.asyncio
    async def test_get_ncu_rule_results_fields(self, rules_registry: ToolRegistry):
        """Each rule result should have rule, severity, message fields."""
        result = await rules_registry.execute("get_ncu_rule_results", {"kernel_id": 0})
        for r in result.data:
            assert "rule" in r
            assert "severity" in r
            assert "message" in r

    @pytest.mark.asyncio
    async def test_get_ncu_rule_results_invalid_id(self, rules_registry: ToolRegistry):
        """get_ncu_rule_results with bad kernel_id should return error."""
        result = await rules_registry.execute("get_ncu_rule_results", {"kernel_id": 10})
        assert result.success is False
        assert result.error.code == ErrorCode.METRIC_NOT_FOUND


# ━━━━━━━━━━━━━━━━━━━━━━━ TestSourceTools ━━━━━━━━━━━━━━━━━━━━━━━


class TestSourceTools:
    """Tests for source correlation tools — error paths when no correlator."""

    @pytest.fixture()
    def source_registry_no_correlator(
        self, report_compute_bound: KernelReport,
    ) -> ToolRegistry:
        """Registry with source tools but NO correlator in context."""
        ctx = SessionContext(kernels=[report_compute_bound], correlator=None, action=None)
        reg = ToolRegistry()
        register_source_tools(reg, ctx)
        return reg

    @pytest.fixture()
    def source_registry_no_instanced(
        self, report_compute_bound: KernelReport,
    ) -> ToolRegistry:
        """Registry with source tools, correlator stub but no instanced metrics."""
        # Provide a mock correlator+action, but the kernel has no instanced metrics
        from unittest.mock import MagicMock
        mock_correlator = MagicMock()
        mock_action = MagicMock()
        ctx = SessionContext(
            kernels=[report_compute_bound],
            correlator=mock_correlator,
            action=mock_action,
        )
        reg = ToolRegistry()
        register_source_tools(reg, ctx)
        return reg

    # --- get_source_hotspots ---

    @pytest.mark.asyncio
    async def test_get_source_hotspots_no_correlator(
        self, source_registry_no_correlator: ToolRegistry,
    ):
        """get_source_hotspots without correlator should return NO_DEBUG_INFO error."""
        result = await source_registry_no_correlator.execute(
            "get_source_hotspots", {"kernel_id": 0},
        )
        assert result.success is False
        assert result.error.code == ErrorCode.NO_DEBUG_INFO
        assert "correlator" in result.error.message.lower() or "not available" in result.error.message.lower()

    @pytest.mark.asyncio
    async def test_get_source_hotspots_no_instanced_metrics(
        self, source_registry_no_instanced: ToolRegistry,
    ):
        """get_source_hotspots with correlator but empty instanced metrics should fail."""
        result = await source_registry_no_instanced.execute(
            "get_source_hotspots", {"kernel_id": 0},
        )
        assert result.success is False
        assert result.error.code == ErrorCode.NO_DEBUG_INFO
        assert "instanced" in result.error.message.lower() or "no instanced" in result.error.message.lower()

    @pytest.mark.asyncio
    async def test_get_source_hotspots_invalid_kernel(
        self, source_registry_no_correlator: ToolRegistry,
    ):
        """get_source_hotspots with out-of-range kernel_id still hits correlator check first."""
        result = await source_registry_no_correlator.execute(
            "get_source_hotspots", {"kernel_id": 99},
        )
        # Correlator check comes before kernel lookup, so this should be NO_DEBUG_INFO
        assert result.success is False
        assert result.error.code == ErrorCode.NO_DEBUG_INFO

    # --- get_sass_for_source_line ---

    @pytest.mark.asyncio
    async def test_get_sass_no_correlator(
        self, source_registry_no_correlator: ToolRegistry,
    ):
        """get_sass_for_source_line without correlator should return error."""
        result = await source_registry_no_correlator.execute(
            "get_sass_for_source_line",
            {"kernel_id": 0, "file": "test.cu", "line": 10},
        )
        assert result.success is False
        assert result.error.code == ErrorCode.NO_DEBUG_INFO

    # --- get_stall_analysis_for_line ---

    @pytest.mark.asyncio
    async def test_get_stall_analysis_no_correlator(
        self, source_registry_no_correlator: ToolRegistry,
    ):
        """get_stall_analysis_for_line without correlator should return error."""
        result = await source_registry_no_correlator.execute(
            "get_stall_analysis_for_line",
            {"kernel_id": 0, "file": "test.cu", "line": 10},
        )
        assert result.success is False
        assert result.error.code == ErrorCode.NO_DEBUG_INFO

    # --- All 3 source tools registered ---

    def test_source_tools_registered_count(
        self, source_registry_no_correlator: ToolRegistry,
    ):
        """register_source_tools should register exactly 3 tools."""
        assert len(source_registry_no_correlator.tool_names()) == 3
        expected = {"get_source_hotspots", "get_sass_for_source_line", "get_stall_analysis_for_line"}
        assert set(source_registry_no_correlator.tool_names()) == expected


# ━━━━━━━━━━━━━━━━━━━━━━━ TestAnalysisTools ━━━━━━━━━━━━━━━━━━━━━


class TestAnalysisTools:
    """Tests for the 2 analysis tools: run_analysis and get_optimization_tree."""

    @pytest.fixture()
    def analysis_registry(
        self, report_compute_bound: KernelReport,
    ) -> ToolRegistry:
        """Build a ToolRegistry with analysis tools + auto-registered analyzers."""
        analyzer_reg = AnalyzerRegistry()
        analyzer_reg.auto_register()
        ctx = SessionContext(
            kernels=[report_compute_bound],
            registry=analyzer_reg,
        )
        reg = ToolRegistry()
        register_analysis_tools(reg, ctx)
        return reg

    @pytest.fixture()
    def analysis_registry_memory(
        self, report_memory_bound: KernelReport,
    ) -> ToolRegistry:
        """Analysis tools with a memory-bound kernel."""
        analyzer_reg = AnalyzerRegistry()
        analyzer_reg.auto_register()
        ctx = SessionContext(
            kernels=[report_memory_bound],
            registry=analyzer_reg,
        )
        reg = ToolRegistry()
        register_analysis_tools(reg, ctx)
        return reg

    @pytest.fixture()
    def analysis_registry_no_analyzer(
        self, report_compute_bound: KernelReport,
    ) -> ToolRegistry:
        """Analysis tools with NO AnalyzerRegistry (None)."""
        ctx = SessionContext(kernels=[report_compute_bound], registry=None)
        reg = ToolRegistry()
        register_analysis_tools(reg, ctx)
        return reg

    # --- run_analysis (all analyzers) ---

    @pytest.mark.asyncio
    async def test_run_analysis_all(self, analysis_registry: ToolRegistry):
        """run_analysis without analyzer_name should return findings from all analyzers."""
        result = await analysis_registry.execute("run_analysis", {"kernel_id": 0})
        assert result.success is True
        assert isinstance(result.data, list)
        assert len(result.data) >= 1
        # Each finding should have expected fields
        for f in result.data:
            assert "severity" in f
            assert "title" in f
            assert "detail" in f
            assert "action" in f
            assert "source" in f
            assert "category" in f

    @pytest.mark.asyncio
    async def test_run_analysis_all_has_roofline(self, analysis_registry: ToolRegistry):
        """run_analysis (all) on compute_bound should include roofline findings."""
        result = await analysis_registry.execute("run_analysis", {"kernel_id": 0})
        sources = {f["source"] for f in result.data}
        assert "roofline" in sources

    @pytest.mark.asyncio
    async def test_run_analysis_all_memory_kernel(
        self, analysis_registry_memory: ToolRegistry,
    ):
        """run_analysis on memory_bound kernel should find memory-related findings."""
        result = await analysis_registry_memory.execute("run_analysis", {"kernel_id": 0})
        assert result.success is True
        sources = {f["source"] for f in result.data}
        # Memory-bound kernel should have memory analyzer findings
        assert "memory" in sources or "roofline" in sources

    # --- run_analysis (specific analyzer) ---

    @pytest.mark.asyncio
    async def test_run_analysis_specific_roofline(self, analysis_registry: ToolRegistry):
        """run_analysis with analyzer_name='roofline' should only return roofline findings."""
        result = await analysis_registry.execute(
            "run_analysis", {"kernel_id": 0, "analyzer_name": "roofline"},
        )
        assert result.success is True
        assert len(result.data) >= 1
        for f in result.data:
            assert f["source"] == "roofline"

    @pytest.mark.asyncio
    async def test_run_analysis_specific_memory(
        self, analysis_registry_memory: ToolRegistry,
    ):
        """run_analysis with analyzer_name='memory' on memory_bound kernel."""
        result = await analysis_registry_memory.execute(
            "run_analysis", {"kernel_id": 0, "analyzer_name": "memory"},
        )
        assert result.success is True
        for f in result.data:
            assert f["source"] == "memory"

    # --- run_analysis (unknown analyzer) ---

    @pytest.mark.asyncio
    async def test_run_analysis_unknown_analyzer(self, analysis_registry: ToolRegistry):
        """run_analysis with non-existent analyzer_name should return error."""
        result = await analysis_registry.execute(
            "run_analysis", {"kernel_id": 0, "analyzer_name": "nonexistent"},
        )
        assert result.success is False
        assert result.error.code == ErrorCode.ANALYZER_FAILED
        assert "nonexistent" in result.error.message.lower()
        # Suggestion should list available analyzers
        assert "available" in result.error.suggestion.lower()

    # --- run_analysis (no registry) ---

    @pytest.mark.asyncio
    async def test_run_analysis_no_registry(
        self, analysis_registry_no_analyzer: ToolRegistry,
    ):
        """run_analysis when AnalyzerRegistry is None should return error."""
        result = await analysis_registry_no_analyzer.execute(
            "run_analysis", {"kernel_id": 0},
        )
        assert result.success is False
        assert result.error.code == ErrorCode.ANALYZER_FAILED

    # --- run_analysis (invalid kernel_id) ---

    @pytest.mark.asyncio
    async def test_run_analysis_invalid_kernel(self, analysis_registry: ToolRegistry):
        """run_analysis with out-of-range kernel_id should return error."""
        result = await analysis_registry.execute(
            "run_analysis", {"kernel_id": 999},
        )
        assert result.success is False
        assert result.error.code == ErrorCode.METRIC_NOT_FOUND

    # --- get_optimization_tree ---

    @pytest.mark.asyncio
    async def test_get_optimization_tree(self, analysis_registry: ToolRegistry):
        """get_optimization_tree should return tree structure."""
        result = await analysis_registry.execute(
            "get_optimization_tree", {"kernel_id": 0},
        )
        assert result.success is True
        d = result.data
        assert "bottleneck" in d
        assert "active_paths" in d
        assert "pruned_branches" in d
        assert "tree_markdown" in d

    @pytest.mark.asyncio
    async def test_get_optimization_tree_has_active_paths(
        self, analysis_registry: ToolRegistry,
    ):
        """get_optimization_tree should have at least 1 active path."""
        result = await analysis_registry.execute(
            "get_optimization_tree", {"kernel_id": 0},
        )
        assert len(result.data["active_paths"]) >= 1
        # Each active path should have path, category, description
        for p in result.data["active_paths"]:
            assert "path" in p
            assert "category" in p

    @pytest.mark.asyncio
    async def test_get_optimization_tree_pruned_count(
        self, analysis_registry: ToolRegistry,
    ):
        """get_optimization_tree should report pruned branch count (0..5)."""
        result = await analysis_registry.execute(
            "get_optimization_tree", {"kernel_id": 0},
        )
        assert 0 <= result.data["pruned_branches"] <= 5

    @pytest.mark.asyncio
    async def test_get_optimization_tree_markdown_truncated(
        self, analysis_registry: ToolRegistry,
    ):
        """tree_markdown should be truncated to at most 500 chars."""
        result = await analysis_registry.execute(
            "get_optimization_tree", {"kernel_id": 0},
        )
        assert len(result.data["tree_markdown"]) <= 500

    @pytest.mark.asyncio
    async def test_get_optimization_tree_no_registry(
        self, analysis_registry_no_analyzer: ToolRegistry,
    ):
        """get_optimization_tree without AnalyzerRegistry should return error."""
        result = await analysis_registry_no_analyzer.execute(
            "get_optimization_tree", {"kernel_id": 0},
        )
        assert result.success is False
        assert result.error.code == ErrorCode.ANALYZER_FAILED

    @pytest.mark.asyncio
    async def test_get_optimization_tree_invalid_kernel(
        self, analysis_registry: ToolRegistry,
    ):
        """get_optimization_tree with bad kernel_id should return error."""
        result = await analysis_registry.execute(
            "get_optimization_tree", {"kernel_id": -1},
        )
        assert result.success is False

    # --- tool registration ---

    def test_analysis_tools_registered_count(self, analysis_registry: ToolRegistry):
        """register_analysis_tools should register exactly 2 tools."""
        names = analysis_registry.tool_names()
        assert len(names) == 2
        assert set(names) == {"run_analysis", "get_optimization_tree"}


# ━━━━━━━━━━━━━━━━━━━━━━━ TestAllToolsIntegration ━━━━━━━━━━━━━━━


class TestAllToolsIntegration:
    """Integration test: register all 9 tools in one registry and verify coexistence."""

    @pytest.fixture()
    def full_registry(self, report_compute_bound: KernelReport) -> ToolRegistry:
        """Registry with all 4 data + 3 source + 2 analysis tools."""
        analyzer_reg = AnalyzerRegistry()
        analyzer_reg.auto_register()
        ctx = SessionContext(
            kernels=[report_compute_bound],
            registry=analyzer_reg,
            correlator=None,
            action=None,
        )
        reg = ToolRegistry()
        register_data_query_tools(reg, ctx)
        register_source_tools(reg, ctx)
        register_analysis_tools(reg, ctx)
        return reg

    def test_all_9_tools_registered(self, full_registry: ToolRegistry):
        """All 9 tools should be registered without name conflicts."""
        names = full_registry.tool_names()
        assert len(names) == 9
        expected = {
            "list_kernels", "get_kernel_metrics", "get_kernel_summary", "get_ncu_rule_results",
            "get_source_hotspots", "get_sass_for_source_line", "get_stall_analysis_for_line",
            "run_analysis", "get_optimization_tree",
        }
        assert set(names) == expected

    def test_all_definitions_have_openai_format(self, full_registry: ToolRegistry):
        """Every registered tool should produce valid OpenAI format."""
        for td in full_registry.all_definitions():
            oai = td.to_openai()
            assert oai["type"] == "function"
            assert "name" in oai["function"]
            assert "description" in oai["function"]
            assert "parameters" in oai["function"]

    def test_all_definitions_have_anthropic_format(self, full_registry: ToolRegistry):
        """Every registered tool should produce valid Anthropic format."""
        for td in full_registry.all_definitions():
            anth = td.to_anthropic()
            assert "name" in anth
            assert "description" in anth
            assert "input_schema" in anth

    def test_all_definitions_have_mcp_format(self, full_registry: ToolRegistry):
        """Every registered tool should produce valid MCP format."""
        for td in full_registry.all_definitions():
            mcp = td.to_mcp()
            assert "name" in mcp
            assert "description" in mcp
            assert "inputSchema" in mcp

    @pytest.mark.asyncio
    async def test_cross_tool_workflow(self, full_registry: ToolRegistry):
        """Simulate a typical agent workflow: list -> summary -> metrics -> analysis."""
        # Step 1: list kernels
        r1 = await full_registry.execute("list_kernels", {})
        assert r1.success is True
        kid = r1.data[0]["kernel_id"]

        # Step 2: get summary
        r2 = await full_registry.execute("get_kernel_summary", {"kernel_id": kid})
        assert r2.success is True
        assert "name" in r2.data

        # Step 3: get metrics
        r3 = await full_registry.execute("get_kernel_metrics", {"kernel_id": kid})
        assert r3.success is True

        # Step 4: run analysis
        r4 = await full_registry.execute("run_analysis", {"kernel_id": kid})
        assert r4.success is True
        assert len(r4.data) >= 1

        # Step 5: get optimization tree
        r5 = await full_registry.execute("get_optimization_tree", {"kernel_id": kid})
        assert r5.success is True
        assert "bottleneck" in r5.data


# ━━━━━━━━━━━━━━━━━━━━━━━ TestSourceToolsWithCorrelator ━━━━━━━━━━━━━━━
#
# Integration tests for the 3 source tools with a REAL SourceCorrelator
# and mock ActionHandle. These cover the happy-path branches that were
# previously untested (correlator IS available, instanced metrics present).
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSourceToolsWithCorrelator:
    """Tests for source tools when a real SourceCorrelator + mock action are available."""

    # ── Shared helpers / fixtures ──

    @staticmethod
    def _build_kernel_with_instanced_metrics(
        device_info: DeviceInfo, launch: LaunchParams,
    ) -> KernelReport:
        """Build a KernelReport with instanced metrics that produce hot hotspots.

        Layout (2 PCs, one source line each):
          PC 0x1000 → kernel.cu:42  — heavy barrier stall (dominant), some inst_executed
          PC 0x2000 → kernel.cu:99  — lighter long_scoreboard stall, some inst_executed

        The metric values are chosen so that:
          PC 0x1000: local_ratio = 800/1000 = 0.8 (> 0.30), global_ratio = 1000/1500 = 0.67 (> 0.10) → is_hot
          PC 0x2000: local_ratio = 300/500 = 0.6 (> 0.30), global_ratio = 500/1500 = 0.33 (> 0.10) → is_hot
        """
        from tachyon.models.kernel import InstancedMetricValue

        instanced: dict[str, list[InstancedMetricValue]] = {
            "smsp__pcsamp_warps_issue_stalled_barrier": [
                InstancedMetricValue(pc=0x1000, value=800.0),
                InstancedMetricValue(pc=0x2000, value=100.0),
            ],
            "smsp__pcsamp_warps_issue_stalled_long_scoreboard": [
                InstancedMetricValue(pc=0x1000, value=200.0),
                InstancedMetricValue(pc=0x2000, value=300.0),
            ],
            "inst_executed": [
                InstancedMetricValue(pc=0x1000, value=1000.0),
                InstancedMetricValue(pc=0x2000, value=500.0),
            ],
        }
        return KernelReport(
            kernel_name="hotspot_kernel",
            demangled_name="hotspot_kernel<float>",
            launch_params=launch,
            device_info=device_info,
            metrics={},
            instanced_metrics=instanced,
        )

    @staticmethod
    def _build_kernel_no_instanced(
        device_info: DeviceInfo, launch: LaunchParams,
    ) -> KernelReport:
        """Build a KernelReport with NO instanced metrics (empty dict)."""
        return KernelReport(
            kernel_name="empty_kernel",
            demangled_name="empty_kernel<float>",
            launch_params=launch,
            device_info=device_info,
            metrics={},
            instanced_metrics={},
        )

    @staticmethod
    def _make_mock_action() -> MagicMock:
        """Create a mock ActionHandle with source_info, sass_by_pc, ptx_by_pc."""
        from unittest.mock import MagicMock

        from tachyon.correlator.source_correlator import SourceInfo

        pc_map: dict[int, tuple[str, int]] = {
            0x1000: ("kernel.cu", 42),
            0x2000: ("kernel.cu", 99),
        }

        action = MagicMock()

        def _source_info(pc: int):
            if pc in pc_map:
                f, l = pc_map[pc]
                return SourceInfo(file_name=f, line=l)
            return None

        action.source_info.side_effect = _source_info
        action.sass_by_pc.side_effect = lambda pc: f"SASS@0x{pc:x}"
        action.ptx_by_pc.side_effect = lambda pc: f"PTX@0x{pc:x}"
        return action

    @pytest.fixture()
    def source_registry_with_correlator(
        self,
        device_h100: DeviceInfo,
        launch_256x4096: LaunchParams,
    ) -> ToolRegistry:
        """Registry with source tools, REAL SourceCorrelator, and mock action."""
        from tachyon.correlator.source_correlator import SourceCorrelator

        kernel = self._build_kernel_with_instanced_metrics(device_h100, launch_256x4096)
        action = self._make_mock_action()
        correlator = SourceCorrelator()  # default thresholds
        ctx = SessionContext(
            kernels=[kernel],
            action=action,
            correlator=correlator,
        )
        reg = ToolRegistry()
        register_source_tools(reg, ctx)
        return reg

    @pytest.fixture()
    def source_registry_no_instanced_real(
        self,
        device_h100: DeviceInfo,
        launch_256x4096: LaunchParams,
    ) -> ToolRegistry:
        """Registry with correlator present but kernel has NO instanced metrics."""
        from tachyon.correlator.source_correlator import SourceCorrelator

        kernel = self._build_kernel_no_instanced(device_h100, launch_256x4096)
        action = self._make_mock_action()
        correlator = SourceCorrelator()
        ctx = SessionContext(
            kernels=[kernel],
            action=action,
            correlator=correlator,
        )
        reg = ToolRegistry()
        register_source_tools(reg, ctx)
        return reg

    # ── get_source_hotspots: happy path ──

    @pytest.mark.asyncio
    async def test_hotspots_returns_success(
        self, source_registry_with_correlator: ToolRegistry,
    ):
        """get_source_hotspots with valid data should succeed."""
        result = await source_registry_with_correlator.execute(
            "get_source_hotspots", {"kernel_id": 0},
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_hotspots_returns_list(
        self, source_registry_with_correlator: ToolRegistry,
    ):
        """get_source_hotspots data should be a list of hotspot dicts."""
        result = await source_registry_with_correlator.execute(
            "get_source_hotspots", {"kernel_id": 0},
        )
        assert isinstance(result.data, list)
        assert len(result.data) >= 1

    @pytest.mark.asyncio
    async def test_hotspots_fields_present(
        self, source_registry_with_correlator: ToolRegistry,
    ):
        """Each hotspot should have rank, file, line, global_pct, local_pct, dominant_stall, is_hot, degraded."""
        result = await source_registry_with_correlator.execute(
            "get_source_hotspots", {"kernel_id": 0},
        )
        for h in result.data:
            assert "rank" in h
            assert "file" in h
            assert "line" in h
            assert "global_pct" in h
            assert "local_pct" in h
            assert "dominant_stall" in h
            assert "is_hot" in h
            assert "degraded" in h

    @pytest.mark.asyncio
    async def test_hotspots_sorted_by_global_pct_desc(
        self, source_registry_with_correlator: ToolRegistry,
    ):
        """Hotspots should be sorted by global_pct descending (rank 1 has highest)."""
        result = await source_registry_with_correlator.execute(
            "get_source_hotspots", {"kernel_id": 0},
        )
        pcts = [h["global_pct"] for h in result.data]
        assert pcts == sorted(pcts, reverse=True)

    @pytest.mark.asyncio
    async def test_hotspots_source_file_line_correct(
        self, source_registry_with_correlator: ToolRegistry,
    ):
        """Hotspots should reference kernel.cu lines 42 and 99."""
        result = await source_registry_with_correlator.execute(
            "get_source_hotspots", {"kernel_id": 0},
        )
        files_lines = {(h["file"], h["line"]) for h in result.data}
        assert ("kernel.cu", 42) in files_lines
        assert ("kernel.cu", 99) in files_lines

    @pytest.mark.asyncio
    async def test_hotspots_not_degraded(
        self, source_registry_with_correlator: ToolRegistry,
    ):
        """Hotspots from mapped source should NOT be degraded."""
        result = await source_registry_with_correlator.execute(
            "get_source_hotspots", {"kernel_id": 0},
        )
        for h in result.data:
            assert h["degraded"] is False

    @pytest.mark.asyncio
    async def test_hotspots_are_hot(
        self, source_registry_with_correlator: ToolRegistry,
    ):
        """Both PCs were designed to pass dual-threshold → is_hot=True."""
        result = await source_registry_with_correlator.execute(
            "get_source_hotspots", {"kernel_id": 0},
        )
        for h in result.data:
            assert h["is_hot"] is True

    @pytest.mark.asyncio
    async def test_hotspots_top_n_limits_output(
        self, source_registry_with_correlator: ToolRegistry,
    ):
        """get_source_hotspots with top_n=1 should return at most 1 entry."""
        result = await source_registry_with_correlator.execute(
            "get_source_hotspots", {"kernel_id": 0, "top_n": 1},
        )
        assert result.success is True
        assert len(result.data) == 1
        assert result.data[0]["rank"] == 1

    @pytest.mark.asyncio
    async def test_hotspots_dominant_stall_is_barrier(
        self, source_registry_with_correlator: ToolRegistry,
    ):
        """Line 42 (PC 0x1000) dominant stall should be barrier (800/1000)."""
        result = await source_registry_with_correlator.execute(
            "get_source_hotspots", {"kernel_id": 0},
        )
        # Find line 42 hotspot
        h42 = next(h for h in result.data if h["line"] == 42)
        assert "barrier" in h42["dominant_stall"]

    # ── get_source_hotspots: no instanced metrics ──

    @pytest.mark.asyncio
    async def test_hotspots_no_instanced_returns_error(
        self, source_registry_no_instanced_real: ToolRegistry,
    ):
        """get_source_hotspots with correlator but empty instanced → NO_DEBUG_INFO."""
        result = await source_registry_no_instanced_real.execute(
            "get_source_hotspots", {"kernel_id": 0},
        )
        assert result.success is False
        assert result.error.code == ErrorCode.NO_DEBUG_INFO
        assert "instanced" in result.error.message.lower()

    # ── get_source_hotspots: invalid kernel_id with correlator ──

    @pytest.mark.asyncio
    async def test_hotspots_invalid_kernel_with_correlator(
        self, source_registry_with_correlator: ToolRegistry,
    ):
        """get_source_hotspots with out-of-range kernel_id should return METRIC_NOT_FOUND."""
        result = await source_registry_with_correlator.execute(
            "get_source_hotspots", {"kernel_id": 99},
        )
        assert result.success is False
        assert result.error.code == ErrorCode.METRIC_NOT_FOUND

    # ── get_sass_for_source_line: happy path ──

    @pytest.mark.asyncio
    async def test_sass_matching_line_returns_success(
        self, source_registry_with_correlator: ToolRegistry,
    ):
        """get_sass_for_source_line for kernel.cu:42 should succeed."""
        result = await source_registry_with_correlator.execute(
            "get_sass_for_source_line",
            {"kernel_id": 0, "file": "kernel.cu", "line": 42},
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_sass_returns_correct_fields(
        self, source_registry_with_correlator: ToolRegistry,
    ):
        """SASS result should have file, line, sass, ptx, pc fields."""
        result = await source_registry_with_correlator.execute(
            "get_sass_for_source_line",
            {"kernel_id": 0, "file": "kernel.cu", "line": 42},
        )
        d = result.data
        assert d["file"] == "kernel.cu"
        assert d["line"] == 42
        assert "sass" in d
        assert "ptx" in d
        assert "pc" in d

    @pytest.mark.asyncio
    async def test_sass_has_sass_and_ptx_text(
        self, source_registry_with_correlator: ToolRegistry,
    ):
        """SASS and PTX text should be derived from mock action (SASS@0x1000 etc.)."""
        result = await source_registry_with_correlator.execute(
            "get_sass_for_source_line",
            {"kernel_id": 0, "file": "kernel.cu", "line": 42},
        )
        d = result.data
        # Our mock returns "SASS@0x1000" for PC 0x1000
        assert "SASS@" in d["sass"]
        assert "PTX@" in d["ptx"]

    @pytest.mark.asyncio
    async def test_sass_second_line(
        self, source_registry_with_correlator: ToolRegistry,
    ):
        """get_sass_for_source_line for kernel.cu:99 should also succeed."""
        result = await source_registry_with_correlator.execute(
            "get_sass_for_source_line",
            {"kernel_id": 0, "file": "kernel.cu", "line": 99},
        )
        assert result.success is True
        assert result.data["line"] == 99

    # ── get_sass_for_source_line: non-matching line ──

    @pytest.mark.asyncio
    async def test_sass_non_matching_line_returns_error(
        self, source_registry_with_correlator: ToolRegistry,
    ):
        """get_sass_for_source_line for a line with no hotspot → NO_DEBUG_INFO."""
        result = await source_registry_with_correlator.execute(
            "get_sass_for_source_line",
            {"kernel_id": 0, "file": "kernel.cu", "line": 999},
        )
        assert result.success is False
        assert result.error.code == ErrorCode.NO_DEBUG_INFO
        assert "kernel.cu:999" in result.error.message

    @pytest.mark.asyncio
    async def test_sass_non_matching_file_returns_error(
        self, source_registry_with_correlator: ToolRegistry,
    ):
        """get_sass_for_source_line for a wrong file → NO_DEBUG_INFO."""
        result = await source_registry_with_correlator.execute(
            "get_sass_for_source_line",
            {"kernel_id": 0, "file": "other.cu", "line": 42},
        )
        assert result.success is False
        assert result.error.code == ErrorCode.NO_DEBUG_INFO

    # ── get_stall_analysis_for_line: happy path ──

    @pytest.mark.asyncio
    async def test_stall_analysis_matching_line_returns_success(
        self, source_registry_with_correlator: ToolRegistry,
    ):
        """get_stall_analysis_for_line for kernel.cu:42 should succeed."""
        result = await source_registry_with_correlator.execute(
            "get_stall_analysis_for_line",
            {"kernel_id": 0, "file": "kernel.cu", "line": 42},
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_stall_analysis_returns_correct_fields(
        self, source_registry_with_correlator: ToolRegistry,
    ):
        """Stall analysis should return file, line, dominant_stall, global_pct, breakdown."""
        result = await source_registry_with_correlator.execute(
            "get_stall_analysis_for_line",
            {"kernel_id": 0, "file": "kernel.cu", "line": 42},
        )
        d = result.data
        assert d["file"] == "kernel.cu"
        assert d["line"] == 42
        assert "dominant_stall" in d
        assert "global_pct" in d
        assert "breakdown" in d

    @pytest.mark.asyncio
    async def test_stall_analysis_breakdown_has_stall_reasons(
        self, source_registry_with_correlator: ToolRegistry,
    ):
        """Breakdown should contain stall reason keys with ratio and pct."""
        result = await source_registry_with_correlator.execute(
            "get_stall_analysis_for_line",
            {"kernel_id": 0, "file": "kernel.cu", "line": 42},
        )
        breakdown = result.data["breakdown"]
        assert len(breakdown) >= 1
        for short_name, info in breakdown.items():
            assert "ratio" in info
            assert "pct" in info
            assert isinstance(info["ratio"], float)
            assert isinstance(info["pct"], float)

    @pytest.mark.asyncio
    async def test_stall_analysis_dominant_stall_is_barrier(
        self, source_registry_with_correlator: ToolRegistry,
    ):
        """Line 42 dominant stall should be barrier (prefix stripped)."""
        result = await source_registry_with_correlator.execute(
            "get_stall_analysis_for_line",
            {"kernel_id": 0, "file": "kernel.cu", "line": 42},
        )
        assert "barrier" in result.data["dominant_stall"]

    @pytest.mark.asyncio
    async def test_stall_analysis_global_pct_positive(
        self, source_registry_with_correlator: ToolRegistry,
    ):
        """global_pct should be > 0 for an active line."""
        result = await source_registry_with_correlator.execute(
            "get_stall_analysis_for_line",
            {"kernel_id": 0, "file": "kernel.cu", "line": 42},
        )
        assert result.data["global_pct"] > 0

    @pytest.mark.asyncio
    async def test_stall_analysis_second_line(
        self, source_registry_with_correlator: ToolRegistry,
    ):
        """get_stall_analysis_for_line for kernel.cu:99 should also succeed."""
        result = await source_registry_with_correlator.execute(
            "get_stall_analysis_for_line",
            {"kernel_id": 0, "file": "kernel.cu", "line": 99},
        )
        assert result.success is True
        assert result.data["line"] == 99
        # Line 99 dominant stall should be long_scoreboard (300 vs 100 barrier)
        assert "long_scoreboard" in result.data["dominant_stall"]

    # ── get_stall_analysis_for_line: non-matching line ──

    @pytest.mark.asyncio
    async def test_stall_analysis_non_matching_line_returns_error(
        self, source_registry_with_correlator: ToolRegistry,
    ):
        """get_stall_analysis_for_line for a non-existent line → NO_DEBUG_INFO."""
        result = await source_registry_with_correlator.execute(
            "get_stall_analysis_for_line",
            {"kernel_id": 0, "file": "kernel.cu", "line": 1},
        )
        assert result.success is False
        assert result.error.code == ErrorCode.NO_DEBUG_INFO
        assert "kernel.cu:1" in result.error.message

    # ── get_sass_for_source_line: invalid kernel_id ──

    @pytest.mark.asyncio
    async def test_sass_invalid_kernel_with_correlator(
        self, source_registry_with_correlator: ToolRegistry,
    ):
        """get_sass_for_source_line with out-of-range kernel_id → METRIC_NOT_FOUND."""
        result = await source_registry_with_correlator.execute(
            "get_sass_for_source_line",
            {"kernel_id": 99, "file": "kernel.cu", "line": 42},
        )
        assert result.success is False
        assert result.error.code == ErrorCode.METRIC_NOT_FOUND

    # ── get_stall_analysis_for_line: invalid kernel_id ──

    @pytest.mark.asyncio
    async def test_stall_analysis_invalid_kernel_with_correlator(
        self, source_registry_with_correlator: ToolRegistry,
    ):
        """get_stall_analysis_for_line with out-of-range kernel_id → METRIC_NOT_FOUND."""
        result = await source_registry_with_correlator.execute(
            "get_stall_analysis_for_line",
            {"kernel_id": 99, "file": "kernel.cu", "line": 42},
        )
        assert result.success is False
        assert result.error.code == ErrorCode.METRIC_NOT_FOUND
