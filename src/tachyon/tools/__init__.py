"""Tool definitions and execution layer (M3).

Public API:
  - ToolDefinition — tool schema + handler
  - ToolRegistry — register, lookup, execute tools
  - SessionContext — shared session state for tools
"""
from .context import SessionContext
from .registry import ToolDefinition, ToolRegistry

__all__ = [
    "ToolDefinition",
    "ToolRegistry",
    "SessionContext",
]
