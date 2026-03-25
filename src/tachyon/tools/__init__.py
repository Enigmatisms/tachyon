"""Tool definitions and execution layer.

Public API:
  - ToolDefinition — tool schema + handler + category
  - ToolRegistry — register, lookup, execute, query by category
  - SessionContext — shared session state for tools
  - register_all_tools — one-call registration of all standard tools
"""
from __future__ import annotations

from .context import SessionContext
from .registry import ToolDefinition, ToolRegistry


def register_all_tools(
    registry: ToolRegistry,
    ctx: SessionContext,
) -> None:
    """Register all standard analysis tools into the given registry."""
    from .analysis import register_analysis_tools
    from .data_query import register_data_query_tools
    from .source import register_source_tools
    from .source_view import register_source_view_tools

    register_data_query_tools(registry, ctx)
    register_source_tools(registry, ctx)
    register_source_view_tools(registry, ctx)
    register_analysis_tools(registry, ctx)


__all__ = [
    "ToolDefinition",
    "ToolRegistry",
    "SessionContext",
    "register_all_tools",
]
