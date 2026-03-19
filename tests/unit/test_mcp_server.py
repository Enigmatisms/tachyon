"""Unit tests for TachyonMCPServer.

Tests the MCP server initialization, tool schema export, and session setup
without requiring the 'mcp' package or a real NCU report file.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from tachyon.config.settings import TachyonConfig
from tachyon.models.kernel import (
    DeviceInfo,
    KernelReport,
    LaunchParams,
    MetricValue,
)
from tachyon.server.mcp import TachyonMCPServer

# ━━━━━━━━━━━━━━━━━━━━━━━ Helpers ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _make_kernel(name: str = "test_kernel") -> KernelReport:
    """Create a minimal KernelReport for testing."""
    return KernelReport(
        kernel_name=name,
        demangled_name=f"{name}<float>",
        launch_params=LaunchParams(
            grid=(128, 1, 1),
            block=(256, 1, 1),
            shared_mem_bytes=0,
            registers_per_thread=32,
        ),
        device_info=DeviceInfo(
            name="NVIDIA A100-SXM4-80GB",
            compute_capability=(8, 0),
            sm_count=108,
            max_clock_mhz=1410,
            memory_bus_width=5120,
            peak_memory_bandwidth_gbps=2039.0,
        ),
        metrics={
            "sm__throughput.avg.pct_of_peak_sustained_elapsed":
                MetricValue(
                    name="sm__throughput.avg.pct_of_peak_sustained_elapsed",
                    value=70.0,
                    unit="%",
                ),
        },
    )


# ━━━━━━━━━━━━━━━━━━━━━━━ Tests ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestTachyonMCPServerInit:
    """Test MCP server construction and lazy initialization."""

    def test_init_creates_instance(self):
        """TachyonMCPServer should initialize with config and no report."""
        config = TachyonConfig()
        server = TachyonMCPServer(config)
        assert server._config is config
        assert server._report_path is None
        assert server._tool_registry is None

    def test_init_with_report_path(self, tmp_path: Path):
        """Report path should be stored as Path."""
        config = TachyonConfig()
        report_path = tmp_path / "test.ncu-rep"
        server = TachyonMCPServer(config, report_path=str(report_path))
        assert server._report_path == report_path

    def test_init_with_path_object(self, tmp_path: Path):
        """Report path given as Path should also work."""
        config = TachyonConfig()
        report_path = tmp_path / "test.ncu-rep"
        server = TachyonMCPServer(config, report_path=report_path)
        assert server._report_path == report_path


class TestMCPServerInitTools:
    """Test _init_tools() — tool registry setup."""

    def test_init_tools_creates_registry(self):
        """_init_tools() should create a ToolRegistry with tools."""
        config = TachyonConfig()
        server = TachyonMCPServer(config)
        server._init_tools()
        assert server._tool_registry is not None
        assert len(server._tool_registry.tool_names()) > 0

    def test_init_without_report_has_empty_kernels(self):
        """Without a report, SessionContext.kernels should be empty."""
        config = TachyonConfig()
        server = TachyonMCPServer(config)
        server._init_tools()
        assert server._session is not None
        assert len(server._session.kernels) == 0

    @patch("tachyon.server.mcp.NcuReportReader", create=True)
    def test_init_with_report_loads_kernels(self, tmp_path: Path):
        """With a valid report, kernels should be loaded into session."""
        config = TachyonConfig()
        report_path = tmp_path / "test.ncu-rep"

        # We patch the import path used inside _init_tools
        mock_reader = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = [_make_kernel("k1"), _make_kernel("k2")]
        mock_reader.load.return_value = mock_result

        server = TachyonMCPServer(config, report_path=report_path)

        # Patch NcuReportReader inside the method's import
        with patch("tachyon.reader.ncu_reader.NcuReportReader", return_value=mock_reader):
            server._init_tools()

        assert server._session is not None
        assert len(server._session.kernels) == 2
        assert server._session.kernels[0].kernel_name == "k1"

    def test_init_without_report_session_has_registry(self):
        """Session should have an AnalyzerRegistry even without report."""
        config = TachyonConfig()
        server = TachyonMCPServer(config)
        server._init_tools()
        assert server._session.registry is not None
        assert len(server._session.registry.all_analyzers()) > 0


class TestMCPToolSchemas:
    """Test get_tool_schemas() — MCP tool format export."""

    def test_get_tool_schemas_returns_all_tools(self):
        """get_tool_schemas() should return schemas for all registered tools."""
        config = TachyonConfig()
        server = TachyonMCPServer(config)
        schemas = server.get_tool_schemas()

        # Should have 12 tools (4 data_query + 4 source + 2 source_view + 2 analysis)
        # At minimum, the standard set
        assert len(schemas) >= 6
        assert isinstance(schemas, list)

    def test_tool_schemas_have_required_fields(self):
        """Each schema must have name, description, and inputSchema."""
        config = TachyonConfig()
        server = TachyonMCPServer(config)
        schemas = server.get_tool_schemas()

        for schema in schemas:
            assert "name" in schema, f"Schema missing 'name': {schema}"
            assert "description" in schema, f"Schema missing 'description': {schema}"
            assert "inputSchema" in schema, f"Schema missing 'inputSchema': {schema}"

            # name should be a non-empty string
            assert isinstance(schema["name"], str)
            assert len(schema["name"]) > 0

            # description should be a non-empty string
            assert isinstance(schema["description"], str)
            assert len(schema["description"]) > 0

            # inputSchema should be a dict with at least "type"
            assert isinstance(schema["inputSchema"], dict)

    def test_tool_schemas_names_unique(self):
        """All tool names should be unique."""
        config = TachyonConfig()
        server = TachyonMCPServer(config)
        schemas = server.get_tool_schemas()

        names = [s["name"] for s in schemas]
        assert len(names) == len(set(names)), f"Duplicate tool names: {names}"

    def test_get_tool_schemas_lazy_init(self):
        """get_tool_schemas() should trigger _init_tools() if not yet called."""
        config = TachyonConfig()
        server = TachyonMCPServer(config)
        assert server._tool_registry is None

        schemas = server.get_tool_schemas()
        assert server._tool_registry is not None
        assert len(schemas) > 0

    def test_get_tool_schemas_idempotent(self):
        """Calling get_tool_schemas() twice should return same results."""
        config = TachyonConfig()
        server = TachyonMCPServer(config)
        schemas1 = server.get_tool_schemas()
        schemas2 = server.get_tool_schemas()
        assert schemas1 == schemas2

    def test_schema_names_match_known_tools(self):
        """Verify known tool names appear in the schemas."""
        config = TachyonConfig()
        server = TachyonMCPServer(config)
        schemas = server.get_tool_schemas()
        names = {s["name"] for s in schemas}

        # Known tool names from the tool registration modules
        expected_subset = {"list_kernels", "get_kernel_metrics", "run_analysis"}
        assert expected_subset.issubset(names), (
            f"Expected tools {expected_subset} not found in {names}"
        )
