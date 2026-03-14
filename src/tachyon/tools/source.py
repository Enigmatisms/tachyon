"""Source correlation tools — Tachyon's unique three-way mapping capabilities.

3 tools:
  - get_source_hotspots: Top-N source code hotspots
  - get_sass_for_source_line: SASS instructions for a specific source line
  - get_stall_analysis_for_line: Warp stall breakdown for a source line

These tools expose the SourceCorrelator engine built in M2 as Agent-callable
APIs. They require instanced metrics (from --set detailed or higher) and
ideally -lineinfo compilation for source attribution.
"""
from __future__ import annotations

from ..errors.handler import ErrorCode, ToolResult
from .context import SessionContext
from .registry import ToolDefinition, ToolRegistry


def register_source_tools(registry: ToolRegistry, ctx: SessionContext) -> None:
    """Register 3 source correlation tools."""

    # --- get_source_hotspots ---
    async def get_source_hotspots(
        kernel_id: int, top_n: int = 5,
    ) -> ToolResult:
        """Get the top-N source code hotspots for a kernel."""
        if ctx.correlator is None or ctx.action is None:
            return ToolResult.fail(
                ErrorCode.NO_DEBUG_INFO,
                "Source correlation not available (no correlator or action handle).",
                "Ensure the report was loaded with full instanced metrics.",
            )
        try:
            kernel = ctx.get_kernel(kernel_id)
            instanced = kernel.instanced_metrics_as_tuples()
            if not instanced:
                return ToolResult.fail(
                    ErrorCode.NO_DEBUG_INFO,
                    "No instanced metrics in this kernel.",
                    "Re-profile with --set detailed or higher.",
                )
            all_hotspots = ctx.correlator.correlate(ctx.action, instanced)
            # Filter to hot ones and limit
            hot = [h for h in all_hotspots if h.is_hot][:top_n]
            if not hot:
                hot = all_hotspots[:top_n]  # fallback: return top by global_ratio
            return ToolResult.ok([
                {
                    "rank": i + 1,
                    "file": h.source_file,
                    "line": h.source_line,
                    "function": h.function,
                    "global_pct": round(h.global_ratio * 100, 1),
                    "local_pct": round(h.local_ratio * 100, 1),
                    "dominant_stall": h.dominant_stall,
                    "is_hot": h.is_hot,
                    "degraded": h.degraded,
                }
                for i, h in enumerate(hot)
            ])
        except IndexError as e:
            return ToolResult.fail(ErrorCode.METRIC_NOT_FOUND, str(e))
        except Exception as e:
            return ToolResult.fail(ErrorCode.UNKNOWN, str(e))

    registry.register(ToolDefinition(
        name="get_source_hotspots",
        description=(
            "Get top-N source code hotspots. Each shows file:line, "
            "global/local percentages, dominant stall reason. "
            "Requires -lineinfo compilation. Tachyon's core differentiator."
        ),
        parameters={
            "type": "object",
            "properties": {
                "kernel_id": {
                    "type": "integer",
                    "description": "Kernel ID from list_kernels.",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of hotspots (default: 5).",
                    "default": 5,
                },
            },
            "required": ["kernel_id"],
        },
        handler=get_source_hotspots,
    ))

    # --- get_sass_for_source_line ---
    async def get_sass_for_source_line(
        kernel_id: int, file: str, line: int,
    ) -> ToolResult:
        """Get SASS instructions for a specific source line."""
        if ctx.correlator is None or ctx.action is None:
            return ToolResult.fail(
                ErrorCode.NO_DEBUG_INFO,
                "Source correlation not available.",
            )
        try:
            kernel = ctx.get_kernel(kernel_id)
            instanced = kernel.instanced_metrics_as_tuples()
            all_hotspots = ctx.correlator.correlate(ctx.action, instanced)
            # Find the hotspot matching file:line
            match = None
            for h in all_hotspots:
                if h.source_file == file and h.source_line == line:
                    match = h
                    break
            if match is None:
                return ToolResult.fail(
                    ErrorCode.NO_DEBUG_INFO,
                    f"No hotspot data for {file}:{line}.",
                    "Check file/line with get_source_hotspots first.",
                )
            result = {
                "file": match.source_file,
                "line": match.source_line,
                "sass": match.sass_instruction or "(no SASS available)",
                "ptx": match.ptx_instruction or "(no PTX available)",
                "pc": hex(match.pc),
            }
            return ToolResult.ok(result)
        except IndexError as e:
            return ToolResult.fail(ErrorCode.METRIC_NOT_FOUND, str(e))
        except Exception as e:
            return ToolResult.fail(ErrorCode.UNKNOWN, str(e))

    registry.register(ToolDefinition(
        name="get_sass_for_source_line",
        description=(
            "Get SASS/PTX instructions for a source line. "
            "Use after get_source_hotspots to drill into instruction-level detail."
        ),
        parameters={
            "type": "object",
            "properties": {
                "kernel_id": {
                    "type": "integer",
                    "description": "Kernel ID from list_kernels.",
                },
                "file": {
                    "type": "string",
                    "description": "Source file path (from get_source_hotspots).",
                },
                "line": {
                    "type": "integer",
                    "description": "Source line number.",
                },
            },
            "required": ["kernel_id", "file", "line"],
        },
        handler=get_sass_for_source_line,
    ))

    # --- get_stall_analysis_for_line ---
    async def get_stall_analysis_for_line(
        kernel_id: int, file: str, line: int,
    ) -> ToolResult:
        """Get warp stall breakdown for a specific source line."""
        if ctx.correlator is None or ctx.action is None:
            return ToolResult.fail(
                ErrorCode.NO_DEBUG_INFO,
                "Source correlation not available.",
            )
        try:
            kernel = ctx.get_kernel(kernel_id)
            instanced = kernel.instanced_metrics_as_tuples()
            all_hotspots = ctx.correlator.correlate(ctx.action, instanced)
            match = None
            for h in all_hotspots:
                if h.source_file == file and h.source_line == line:
                    match = h
                    break
            if match is None:
                return ToolResult.fail(
                    ErrorCode.NO_DEBUG_INFO,
                    f"No stall data for {file}:{line}.",
                )
            # Build stall breakdown from stall_reasons dict
            total = sum(match.stall_reasons.values()) or 1.0
            breakdown = {}
            for reason, ratio in sorted(
                match.stall_reasons.items(), key=lambda x: x[1], reverse=True,
            ):
                if ratio > 0:
                    # Extract short name from full metric name
                    short = reason.replace("smsp__pcsamp_warps_issue_stalled_", "")
                    breakdown[short] = {
                        "ratio": round(ratio, 3),
                        "pct": round(ratio / total * 100, 1),
                    }
            return ToolResult.ok({
                "file": file,
                "line": line,
                "dominant_stall": match.dominant_stall.replace(
                    "smsp__pcsamp_warps_issue_stalled_", ""
                ) if match.dominant_stall else "unknown",
                "global_pct": round(match.global_ratio * 100, 1),
                "breakdown": breakdown,
            })
        except IndexError as e:
            return ToolResult.fail(ErrorCode.METRIC_NOT_FOUND, str(e))
        except Exception as e:
            return ToolResult.fail(ErrorCode.UNKNOWN, str(e))

    registry.register(ToolDefinition(
        name="get_stall_analysis_for_line",
        description=(
            "Get warp stall breakdown for a source line. "
            "Shows each stall reason with percentage. "
            "Reveals micro-architectural bottleneck at instruction level."
        ),
        parameters={
            "type": "object",
            "properties": {
                "kernel_id": {
                    "type": "integer",
                    "description": "Kernel ID from list_kernels.",
                },
                "file": {
                    "type": "string",
                    "description": "Source file path.",
                },
                "line": {
                    "type": "integer",
                    "description": "Source line number.",
                },
            },
            "required": ["kernel_id", "file", "line"],
        },
        handler=get_stall_analysis_for_line,
    ))
