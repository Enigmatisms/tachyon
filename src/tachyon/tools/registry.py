"""Tool definition, registration, and execution framework.

Provides:
  - ToolDefinition: schema + handler, with to_openai/to_anthropic/to_mcp converters
  - ToolRegistry: register, lookup, execute tools by name

Token-efficiency: Tool descriptions are concise but informative. JSON Schema
parameters use minimal required fields. Tool results are serialized compactly.
"""
from __future__ import annotations

import json
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ..errors.handler import ErrorCode, ToolResult

_log = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """A single tool's metadata + execution entry point.

    parameters follows JSON Schema format, enabling automatic conversion
    to OpenAI / Anthropic / MCP tool spec formats.

    category is a Tachyon-internal label for grouping (not sent to LLM APIs).
    """
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    handler: Callable[..., Awaitable[ToolResult]] | None = None
    category: str = "general"

    def to_openai(self) -> dict:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic(self) -> dict:
        """Convert to Anthropic tool_use format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def to_mcp(self) -> dict:
        """Convert to MCP Tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.parameters,
        }


class ToolRegistry:
    """Central registry for all Tachyon tools.

    Manages tool lifecycle:
      1. Registration (at init or lazily)
      2. Schema export (for LLM tool definitions)
      3. Execution (dispatch tool calls from Agent Loop)
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool. Raises ValueError on duplicate name."""
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def all_definitions(self) -> list[ToolDefinition]:
        """Return all registered tools (for LLM tool specs)."""
        return list(self._tools.values())

    def tool_names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    def filter(self, names: set[str]) -> ToolRegistry:
        """Return a new registry containing only the named tools.

        Tools not in *names* are silently skipped.
        """
        filtered = ToolRegistry()
        for name in names:
            tool = self._tools.get(name)
            if tool is not None:
                filtered._tools[name] = tool
        return filtered

    async def execute(self, name: str, arguments: dict[str, Any] | str) -> ToolResult:
        """Execute a tool by name with given arguments.

        Arguments can be a dict or a JSON string (from LLM responses).

        Returns:
            ToolResult with success/data or failure/error.
        """
        tool = self._tools.get(name)
        if not tool:
            return ToolResult.fail(
                ErrorCode.TOOL_NOT_FOUND,
                f"Unknown tool: {name}",
                f"Available tools: {', '.join(self._tools.keys())}",
            )
        if not tool.handler:
            return ToolResult.fail(
                ErrorCode.UNKNOWN,
                f"Tool '{name}' has no handler",
            )
        # Normalize arguments: accept JSON string or dict
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                return ToolResult.fail(
                    ErrorCode.INVALID_ARGUMENT,
                    f"Invalid JSON arguments for tool '{name}': {arguments[:200]}",
                )
        if not isinstance(arguments, dict):
            arguments = {}
        # Filter arguments to match handler signature (LLM may pass extras)
        try:
            sig = inspect.signature(tool.handler)
            has_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
            if not has_var_kw:
                valid = {
                    name for name, p in sig.parameters.items()
                    if p.kind in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                }
                arguments = {k: v for k, v in arguments.items() if k in valid}
        except (TypeError, ValueError):
            pass
        try:
            return await tool.handler(**arguments)
        except Exception as e:
            _log.exception("Tool '%s' execution failed", name)
            return ToolResult.fail(
                ErrorCode.UNKNOWN,
                f"Tool '{name}' execution failed: {e}",
            )
