"""MCP Server — expose Tachyon Tools to external agents.

Implements Model Context Protocol (stdio transport) for zero-config integration
with Claude Code, Cursor, and custom agents.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from tachyon.config.settings import TachyonConfig

logger = logging.getLogger(__name__)


class TachyonMCPServer:
    """MCP stdio server exposing all Tachyon analysis tools.

    Usage:
        server = TachyonMCPServer(config, report_path="report.ncu-rep")
        await server.run()  # Blocks, serving on stdio
    """

    def __init__(self, config: TachyonConfig, report_path: str | Path | None = None):
        self._config = config
        self._report_path = Path(report_path) if report_path else None
        self._tool_registry = None  # Lazy init
        self._session = None

    def _init_tools(self) -> None:
        """Initialize Tachyon analysis stack and register tools.

        Creates: NcuReportReader -> SessionContext -> ToolRegistry with all 12 tools.
        """
        from tachyon.analyzers.base import AnalyzerRegistry
        from tachyon.tools.analysis import register_analysis_tools
        from tachyon.tools.context import SessionContext
        from tachyon.tools.data_query import register_data_query_tools
        from tachyon.tools.registry import ToolRegistry
        from tachyon.tools.source import register_source_tools
        from tachyon.tools.source_view import register_source_view_tools

        kernels = []
        reader = None  # Track reader for action handle
        if self._report_path:
            from tachyon.reader.ncu_reader import NcuReportReader

            reader = NcuReportReader(self._config)
            result = reader.load(self._report_path)
            if result.success and result.data:
                kernels = result.data

        analyzer_registry = AnalyzerRegistry()
        analyzer_registry.auto_register()

        # Create correlator if we have instanced metrics and reader
        correlator = None
        if reader is not None and any(k.instanced_metrics for k in kernels):
            from tachyon.correlator.source_correlator import SourceCorrelator
            correlator = SourceCorrelator()
            logger.info(
                "SourceCorrelator initialized: %d kernel(s) with instanced metrics",
                sum(1 for k in kernels if k.instanced_metrics),
            )
        else:
            if reader is not None:
                logger.warning(
                    "SourceCorrelator NOT created: no instanced metrics. "
                    "Source correlation requires '--set detailed' or higher.",
                )

        # Initialize NCUMappingSystem independently (does NOT depend on instanced_metrics)
        mapper = None
        if reader is not None:
            try:
                from tachyon.correlator.source_mapper import NCUMappingSystem
                mapper = NCUMappingSystem(str(self._report_path))
                logger.info(
                    "NCUMappingSystem initialized: %d mapped instructions",
                    len(mapper._s2as_flat),
                )
            except Exception as e:
                logger.warning("NCUMappingSystem NOT created: %s", e)

        self._session = SessionContext(
            kernels=kernels,
            action=reader,  # NcuReportReader implements ActionHandle protocol
            correlator=correlator,
            registry=analyzer_registry,
            mapper=mapper,
            allowed_source_paths=SessionContext.build_allowed_source_paths(kernels, mapper),
        )

        self._tool_registry = ToolRegistry()
        register_data_query_tools(self._tool_registry, self._session)
        register_source_tools(self._tool_registry, self._session)
        register_source_view_tools(self._tool_registry, self._session)
        register_analysis_tools(self._tool_registry, self._session)

    async def run(self) -> None:
        """Start MCP server on stdio transport. Blocks until client disconnects."""
        try:
            from mcp.server import Server
            from mcp.server.stdio import stdio_server
            from mcp.types import Tool as MCPTool
        except ImportError as exc:
            raise ImportError(
                "MCP server requires the 'mcp' package. "
                "Install with: pip install tachyon-cuda[mcp]"
            ) from exc

        self._init_tools()
        assert self._tool_registry is not None

        server = Server("tachyon")
        registry = self._tool_registry

        @server.list_tools()
        async def list_tools() -> list[MCPTool]:
            tools = []
            for tool_def in registry.all_definitions():
                mcp_schema = tool_def.to_mcp()
                tools.append(MCPTool(
                    name=mcp_schema["name"],
                    description=mcp_schema["description"],
                    inputSchema=mcp_schema["inputSchema"],
                ))
            return tools

        @server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[dict]:
            result = await registry.execute(name, arguments)

            if result.success:
                content = json.dumps(result.data, default=str, ensure_ascii=False)
            else:
                assert result.error is not None
                content = json.dumps({
                    "error": result.error.message,
                    "code": result.error.code.value,
                    "suggestion": result.error.suggestion,
                }, ensure_ascii=False)

            return [{"type": "text", "text": content}]

        logger.info("Starting Tachyon MCP server (stdio transport)")
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    def get_tool_schemas(self) -> list[dict]:
        """Return MCP tool schemas without starting the server (for testing)."""
        if self._tool_registry is None:
            self._init_tools()
        assert self._tool_registry is not None
        return [td.to_mcp() for td in self._tool_registry.all_definitions()]
