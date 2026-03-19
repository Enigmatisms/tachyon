"""Source correlation tools — Tachyon's unique three-way mapping capabilities.

4 tools:
  - get_source_hotspots: Top-N source code hotspots
  - get_sass_for_source_line: SASS instructions for a specific source line
  - get_stall_analysis_for_line: Warp stall breakdown for a source line
  - get_performance_hotspots: Categorized hotspot report with SASS mix

The first 3 tools use SourceCorrelator (instanced metrics from NcuReportReader).
The 4th tool (get_performance_hotspots) uses NCUMappingSystem when available,
which provides richer analysis: categorized stall profiles, SASS instruction
classification, and bidirectional source<->SASS mapping.

All tools degrade gracefully when source mapping is unavailable.
"""
from __future__ import annotations

from ..errors.handler import ErrorCode, ToolResult
from .context import SessionContext
from .registry import ToolDefinition, ToolRegistry


def register_source_tools(registry: ToolRegistry, ctx: SessionContext) -> None:
    """Register 4 source correlation tools."""

    # Helper to generate diagnostic message
    def _diagnose_missing_correlation() -> str:
        """Generate diagnostic message for why correlation is unavailable."""
        if ctx.mapper is not None:
            return (
                "Correlator is None but mapper is available. "
                "Using mapper-based analysis (limited stall breakdown)."
            )
        if ctx.action is None:
            return (
                "Action handle is None: no .ncu-rep file loaded. "
                "Use 'tachyon serve --mcp --report <file.ncu-rep>' "
                "or 'tachyon chat <file.ncu-rep>' to load a report."
            )
        elif ctx.correlator is None:
            has_instanced = any(
                bool(k.instanced_metrics) for k in ctx.kernels
            ) if ctx.kernels else False
            if not has_instanced:
                return (
                    "Correlator is None: no instanced metrics in report. "
                    "Source correlation requires PC-sampling metrics from "
                    "'--set detailed' or '--set full'. "
                    "Did you use '--set basic'? That won't work."
                )
            else:
                return (
                    "Correlator is None: failed to initialize. "
                    "Check that the .ncu-rep file is valid."
                )
        return "Unknown reason."

    # --- get_source_hotspots ---
    async def get_source_hotspots(
        kernel_id: int, top_n: int = 5,
    ) -> ToolResult:
        """Get the top-N source code hotspots for a kernel."""
        try:
            kernel = ctx.get_kernel(kernel_id)
        except IndexError as e:
            return ToolResult.fail(ErrorCode.METRIC_NOT_FOUND, str(e))

        # Primary path: SourceCorrelator (instanced metrics)
        if ctx.correlator is not None and ctx.action is not None:
            try:
                instanced = kernel.instanced_metrics_as_tuples()
                if not instanced:
                    return ToolResult.fail(
                        ErrorCode.NO_DEBUG_INFO,
                        "No instanced metrics in this kernel.",
                        "Re-profile with --set detailed or higher.",
                    )
                all_hotspots = ctx.correlator.correlate(
                    ctx.action, instanced, kernel_name=kernel.kernel_name,
                )
                if all_hotspots:
                    hot = [h for h in all_hotspots if h.is_hot][:top_n]
                    if not hot:
                        hot = all_hotspots[:top_n]
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
                # Correlator returned empty — fall through to mapper
            except Exception as e:
                return ToolResult.fail(ErrorCode.UNKNOWN, str(e))

        # Fallback path: NCUMappingSystem (pre-built source<->SASS mapping)
        if ctx.mapper is not None:
            try:
                report = ctx.mapper.get_bottleneck_report(top_n=top_n)
                kernel_report = [
                    e for e in report
                    if e["kernel"] == kernel.kernel_name
                ]
                if not kernel_report:
                    return ToolResult.fail(
                        ErrorCode.NO_DEBUG_INFO,
                        "No source-mapped instructions for this kernel.",
                        "The kernel may have been compiled without -lineinfo.",
                    )
                return ToolResult.ok([
                    {
                        "rank": i + 1,
                        "file": entry["file"],
                        "line": entry["line"],
                        "function": None,
                        "global_pct": entry["severity"],
                        "local_pct": None,
                        "dominant_stall": entry["dominant_stall"],
                        "is_hot": entry["severity"] >= 5.0,
                        "degraded": None,
                    }
                    for i, entry in enumerate(kernel_report[:top_n])
                ])
            except Exception as e:
                return ToolResult.fail(ErrorCode.UNKNOWN, str(e))

        # No source mapping available at all
        return ToolResult.fail(
            ErrorCode.NO_DEBUG_INFO,
            "Source correlation not available (no correlator or mapper).",
            _diagnose_missing_correlation(),
        )

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
        try:
            kernel = ctx.get_kernel(kernel_id)

            # Fast path: use mapper for full SASS list with categories
            if ctx.mapper is not None:
                insts = ctx.mapper.get_sass_by_line(
                    kernel.kernel_name, file, line,
                )
                if insts:
                    return ToolResult.ok({
                        "file": file,
                        "line": line,
                        "total_instructions": len(insts),
                        "sass_instructions": [
                            {
                                "pc": e["pc"],
                                "sass": e["sass"],
                                "category": NCUMappingSystem.classify_sass(
                                    e["sass"]
                                ),
                            }
                            for e in insts
                        ],
                    })

            # Fallback: use correlator (may return partial data)
            if ctx.correlator is None or ctx.action is None:
                return ToolResult.fail(
                    ErrorCode.NO_DEBUG_INFO,
                    "Source correlation not available.",
                    _diagnose_missing_correlation(),
                )
            instanced = kernel.instanced_metrics_as_tuples()
            all_hotspots = ctx.correlator.correlate(
                ctx.action, instanced, kernel_name=kernel.kernel_name,
            )
            match = None
            for h in all_hotspots:
                if h.source_file == file and h.source_line == line:
                    match = h
                    break
            if match is None:
                return ToolResult.fail(
                    ErrorCode.NO_DEBUG_INFO,
                    f"No data for {file}:{line}.",
                    "Check file/line with get_source_hotspots first.",
                )
            return ToolResult.ok({
                "file": match.source_file,
                "line": match.source_line,
                "sass": match.sass_instruction or "(no SASS available)",
                "ptx": match.ptx_instruction or "(no PTX available)",
                "pc": hex(match.pc),
                "total_instructions": 1,
                "sass_instructions": [
                    {
                        "pc": hex(match.pc),
                        "sass": match.sass_instruction or "(no SASS)",
                        "category": None,
                    }
                ],
            })
        except IndexError as e:
            return ToolResult.fail(ErrorCode.METRIC_NOT_FOUND, str(e))
        except Exception as e:
            return ToolResult.fail(ErrorCode.UNKNOWN, str(e))

    registry.register(ToolDefinition(
        name="get_sass_for_source_line",
        description=(
            "Get SASS instructions for a source line. "
            "Returns full instruction list with categories (Memory Load, "
            "Float Compute, Tensor/Matrix, etc.) when source mapper is available. "
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
        try:
            kernel = ctx.get_kernel(kernel_id)
        except IndexError as e:
            return ToolResult.fail(ErrorCode.METRIC_NOT_FOUND, str(e))

        # Primary path: SourceCorrelator (instanced metrics)
        if ctx.correlator is not None and ctx.action is not None:
            try:
                instanced = kernel.instanced_metrics_as_tuples()
                all_hotspots = ctx.correlator.correlate(
                    ctx.action, instanced, kernel_name=kernel.kernel_name,
                )
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
                total = sum(match.stall_reasons.values()) or 1.0
                breakdown = {}
                for reason, ratio in sorted(
                    match.stall_reasons.items(), key=lambda x: x[1], reverse=True,
                ):
                    if ratio > 0:
                        short = reason.replace(
                            "smsp__pcsamp_warps_issue_stalled_", ""
                        )
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
            except Exception as e:
                return ToolResult.fail(ErrorCode.UNKNOWN, str(e))

        # Fallback path: NCUMappingSystem
        if ctx.mapper is not None:
            try:
                report = ctx.mapper.get_bottleneck_report(top_n=500)
                match = None
                for entry in report:
                    if (
                        entry["kernel"] == kernel.kernel_name
                        and entry["file"] == file
                        and entry["line"] == line
                    ):
                        match = entry
                        break
                if match is None:
                    return ToolResult.fail(
                        ErrorCode.NO_DEBUG_INFO,
                        f"No stall data for {file}:{line}.",
                    )
                return ToolResult.ok({
                    "file": file,
                    "line": line,
                    "dominant_stall": match["dominant_stall"],
                    "global_pct": match["severity"],
                    "breakdown": {
                        k: {"ratio": None, "pct": v}
                        for k, v in match.get("stall_profile", {}).items()
                    },
                })
            except Exception as e:
                return ToolResult.fail(ErrorCode.UNKNOWN, str(e))

        # No source mapping available at all
        return ToolResult.fail(
            ErrorCode.NO_DEBUG_INFO,
            "Source correlation not available (no correlator or mapper).",
            _diagnose_missing_correlation(),
        )

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

    # --- get_performance_hotspots (uses NCUMappingSystem) ---
    async def get_performance_hotspots(
        kernel_id: int, top_n: int = 10,
    ) -> ToolResult:
        """Get categorized performance hotspot report with SASS mix.

        Uses NCUMappingSystem (pre-built source<->SASS mapping) when available.
        Provides richer analysis than get_source_hotspots:
        - Categorized stall profile (Memory/Compute/Sync/Scheduling)
        - SASS instruction mix (what types of instructions on each line)
        - Representative SASS preview
        - Severity ranking by PC-sampling or execution frequency
        """
        if ctx.mapper is None:
            return ToolResult.fail(
                ErrorCode.NO_DEBUG_INFO,
                "Source mapper not available.",
                "Source mapper requires NCU and a valid .ncu-rep file. "
                "Ensure NCU is installed and the report was profiled "
                "with --set detailed or higher.",
            )
        try:
            kernel = ctx.get_kernel(kernel_id)
            report = ctx.mapper.get_bottleneck_report(top_n=top_n)

            # Filter to the requested kernel
            kernel_report = [
                entry for entry in report
                if entry["kernel"] == kernel.kernel_name
            ]

            if not kernel_report:
                return ToolResult.ok({
                    "kernel": kernel.demangled_name or kernel.kernel_name,
                    "hotspots": [],
                    "message": (
                        "No source-mapped instructions found for this kernel. "
                        "The kernel may have been compiled without -lineinfo."
                    ),
                })

            return ToolResult.ok({
                "kernel": kernel.demangled_name or kernel.kernel_name,
                "severity_metric": kernel_report[0]["severity_metric"],
                "total_samples": ctx.mapper._total_samples_per_kernel.get(
                    kernel.kernel_name, 0,
                ),
                "total_exec": ctx.mapper._total_exec_per_kernel.get(
                    kernel.kernel_name, 0,
                ),
                "hotspots": [
                    {
                        "rank": i + 1,
                        "file": entry["file"],
                        "line": entry["line"],
                        "severity": entry["severity"],
                        "severity_metric": entry["severity_metric"],
                        "num_instructions": entry["num_insts"],
                        "dominant_stall": entry["dominant_stall"],
                        "dominant_sass": entry["dominant_sass"],
                        "sass_mix": entry["sass_mix"],
                        "stall_profile": entry["stall_profile"],
                        "sass_preview": entry["sass_preview"],
                    }
                    for i, entry in enumerate(kernel_report)
                ],
            })
        except IndexError as e:
            return ToolResult.fail(ErrorCode.METRIC_NOT_FOUND, str(e))
        except Exception as e:
            return ToolResult.fail(ErrorCode.UNKNOWN, str(e))

    registry.register(ToolDefinition(
        name="get_performance_hotspots",
        description=(
            "Get categorized source-level hotspot report. Shows severity%, "
            "SASS instruction mix (Memory/Compute/Tensor/Control), stall "
            "profile (Memory/Compute/Sync/Scheduling), and representative "
            "SASS instructions for each hotspot line. Richer than "
            "get_source_hotspots. Use for in-depth performance root cause."
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
                    "description": "Number of hotspots (default: 10).",
                    "default": 10,
                },
            },
            "required": ["kernel_id"],
        },
        handler=get_performance_hotspots,
    ))


# Bottom import to avoid circular dependency
from ..correlator.source_mapper import NCUMappingSystem
