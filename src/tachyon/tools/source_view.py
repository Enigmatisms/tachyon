"""Source file viewing tools — read source code at hotspot locations.

2 tools:
  - list_source_files: List source files available for reading
  - read_source_file: Read source file content with optional line window

These tools complement the SASS mapping tools in source.py by providing
actual source code context. This enables the agent workflow:
  hotspot → source code understanding → SASS instruction analysis → diagnosis.
"""
from __future__ import annotations

import os

from ..errors.handler import ErrorCode, ToolResult
from ..tools.registry import ToolDefinition, ToolRegistry
from .context import SessionContext


def register_source_view_tools(
    registry: ToolRegistry, ctx: SessionContext,
) -> None:
    """Register 2 source viewing tools."""

    async def list_source_files() -> ToolResult:
        """List source files available for reading.

        Returns files from the session's allowed source path whitelist,
        which is built from kernel metadata and NCU mapper data.
        """
        try:
            files = sorted(ctx.allowed_source_paths)
            return ToolResult.ok({"files": files})
        except Exception as e:
            return ToolResult.fail(ErrorCode.UNKNOWN, str(e))

    async def read_source_file(
        file: str,
        line: int | None = None,
        context_lines: int = 40,
        max_lines: int = 200,
    ) -> ToolResult:
        """Read source file content at a specific location.

        Args:
            file: File path or basename for fuzzy matching.
            line: Center line number. If provided, returns lines around it.
            context_lines: Number of lines before/after center (default 40).
            max_lines: Maximum lines to return (default 200, cap 500).
        """
        try:
            # Cap max_lines
            max_lines = min(max(max_lines, 1), 500)
            context_lines = min(max(context_lines, 0), 200)

            # Resolve file path: exact match first, then basename match
            resolved = _resolve_file(file, ctx.allowed_source_paths)
            if resolved is None:
                return ToolResult.fail(
                    ErrorCode.INVALID_ARGUMENT,
                    f"File '{file}' is not in the allowed source paths.",
                    "Use list_source_files to see available files.",
                )

            # Read file (disk first, fall back to embedded source)
            all_lines: list[str] = []
            if os.path.isfile(resolved):
                try:
                    with open(resolved, encoding="utf-8", errors="replace") as f:
                        all_lines = f.readlines()
                except OSError as e:
                    return ToolResult.fail(
                        ErrorCode.NO_DEBUG_INFO,
                        f"Cannot read file '{resolved}': {e}",
                    )
            else:
                embedded = ctx.embedded_sources.get(resolved)
                if not embedded:
                    return ToolResult.fail(
                        ErrorCode.NO_DEBUG_INFO,
                        f"File '{resolved}' does not exist on disk and "
                        "no embedded source available.",
                        "The file may have been moved since profiling, "
                        "or the report was captured without --import-source yes.",
                    )
                all_lines = embedded.splitlines()

            total_lines = len(all_lines)

            # Determine window
            if line is not None:
                line = max(1, min(line, total_lines))
                start = max(0, line - 1 - context_lines)
                end = min(total_lines, line + context_lines)
            else:
                start = 0
                end = min(total_lines, max_lines)

            # Enforce max_lines
            if end - start > max_lines:
                end = start + max_lines

            lines = []
            for i in range(start, end):
                lines.append({
                    "line_num": i + 1,
                    "content": all_lines[i].rstrip("\n\r"),
                })

            return ToolResult.ok({
                "file": resolved,
                "total_lines": total_lines,
                "start_line": start + 1,
                "end_line": end,
                "lines": lines,
            })

        except Exception as e:
            return ToolResult.fail(ErrorCode.UNKNOWN, str(e))

    registry.register(ToolDefinition(
        name="list_source_files",
        description=(
            "List all source files available for reading. "
            "These are files referenced by profiled kernels "
            "(from kernel metadata and NCU source mapping)."
        ),
        parameters={
            "type": "object",
            "properties": {},
        },
        handler=list_source_files,
    ))

    registry.register(ToolDefinition(
        name="read_source_file",
        description=(
            "Read source file content, optionally centered on a specific line. "
            "Use this to understand the algorithm at a hotspot location before "
            "analyzing SASS instructions. Supports fuzzy matching by basename."
        ),
        parameters={
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": (
                        "File path or basename. "
                        "Use list_source_files to discover available files."
                    ),
                },
                "line": {
                    "type": "integer",
                    "description": "Center line number (1-based). "
                                   "Returns lines around this location.",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Lines before/after center (default 40).",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Max lines to return (default 200, max 500).",
                },
            },
            "required": ["file"],
        },
        handler=read_source_file,
    ))


def _resolve_file(
    file: str, allowed: set[str],
) -> str | None:
    """Resolve file path against the whitelist.

    Strategy:
      1. Exact match (file == path in allowed)
      2. Suffix match (file is suffix of some allowed path)
      3. Basename match (os.path.basename(file) matches basename of allowed path)
    """
    # 1. Exact match
    if file in allowed:
        return file

    # 2. Suffix match: file ends with the query
    matches = [p for p in allowed if p.endswith(file)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Prefer shortest (most specific)
        return min(matches, key=len)

    # 3. Basename match
    query_basename = os.path.basename(file)
    if query_basename:
        matches = [
            p for p in allowed
            if os.path.basename(p) == query_basename
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return min(matches, key=len)

    return None
