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
    *,
    include_evolve: bool = False,
    evolve_ctx: object | None = None,
) -> None:
    """Register all standard analysis tools into the given registry.

    Replaces the repeated 4-call pattern with a single call.
    Individual register functions remain available for direct use.
    """
    from .analysis import register_analysis_tools
    from .data_query import register_data_query_tools
    from .source import register_source_tools
    from .source_view import register_source_view_tools

    register_data_query_tools(registry, ctx)
    register_source_tools(registry, ctx)
    register_source_view_tools(registry, ctx)
    register_analysis_tools(registry, ctx)

    if include_evolve:
        if evolve_ctx is None:
            raise ValueError("evolve_ctx is required when include_evolve=True")
        from ..evolve.tools import register_evolve_tools
        register_evolve_tools(registry, evolve_ctx)


__all__ = [
    "ToolDefinition",
    "ToolRegistry",
    "SessionContext",
    "register_all_tools",
]
