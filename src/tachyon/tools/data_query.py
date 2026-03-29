"""Data query tools — expose kernel metrics and NCU rule results to the Agent.

4 tools:
  - list_kernels: List all kernels in the loaded report (with optional name filter)
  - get_kernel_metrics: Get scalar metrics for a specific kernel
  - get_kernel_summary: Get a concise summary of kernel characteristics
  - get_ncu_rule_results: Get NCU built-in rule analysis results

Token-efficiency: Output is compact JSON — no redundant whitespace or keys.
Large metric dicts are optionally filtered by caller-specified metric_names.
"""
from __future__ import annotations

from typing import Any

from ..errors.handler import ErrorCode, ToolResult
from .context import SessionContext
from .registry import ToolDefinition, ToolRegistry


def register_data_query_tools(registry: ToolRegistry, ctx: SessionContext) -> None:
    """Register 4 data query tools against a ToolRegistry."""

    # --- list_kernels ---
    async def list_kernels(name_pattern: str | None = None) -> ToolResult:
        """List all kernels in the loaded report, optionally filtered by name pattern."""
        try:
            from tachyon.utils.kernel_filter import match_kernel_name

            items = []
            for i, k in enumerate(ctx.kernels):
                name = k.demangled_name or k.kernel_name
                if name_pattern and not match_kernel_name(
                    k.kernel_name, k.demangled_name, name_pattern
                ):
                    continue
                items.append({
                    "kernel_id": i,
                    "name": name,
                    "grid": list(k.launch_params.grid),
                    "block": list(k.launch_params.block),
                    "registers": k.launch_params.registers_per_thread,
                    "duration_ms": round(
                        (k.metric_value("gpu__time_duration.sum") or 0) / 1e6, 2
                    ),
                })
            return ToolResult.ok(items)
        except Exception as e:
            return ToolResult.fail(ErrorCode.UNKNOWN, str(e))

    registry.register(ToolDefinition(
        name="list_kernels",
        description=(
            "List CUDA kernels in the profiling report. "
            "Returns kernel_id, name, grid/block dimensions. "
            "Use name_pattern to filter by glob (e.g. 'matmul*') or substring match. "
            "Use this first to find the kernel(s) you want to analyze."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name_pattern": {
                    "type": "string",
                    "description": "Optional glob pattern or substring to filter kernel names (e.g. 'matmul*', 'reduce').",
                },
            },
            "required": [],
        },
        handler=list_kernels,
        category="data_query",
    ))

    # --- get_kernel_metrics ---
    async def get_kernel_metrics(
        kernel_id: int, metric_names: list[str] | None = None,
    ) -> ToolResult:
        """Get scalar metrics for a specific kernel."""
        try:
            kernel = ctx.get_kernel(kernel_id)
            result: dict[str, Any] = {}
            if metric_names:
                for name in metric_names:
                    mv = kernel.metrics.get(name)
                    if mv is not None:
                        result[name] = {"value": mv.value, "unit": mv.unit}
                    else:
                        result[name] = None
            else:
                all_metrics = [
                    (name, mv.value, mv.unit)
                    for name, mv in kernel.metrics.items()
                    if mv is not None
                ]
                all_metrics.sort(key=lambda x: abs(x[1]), reverse=True)
                result["_metric_count"] = len(all_metrics)
                if all_metrics:
                    result["_top_metrics"] = [
                        {"name": n, "value": round(v, 2), "unit": u}
                        for n, v, u in all_metrics[:15]
                    ]
            return ToolResult.ok(result)
        except IndexError as e:
            return ToolResult.fail(ErrorCode.METRIC_NOT_FOUND, str(e))
        except Exception as e:
            return ToolResult.fail(ErrorCode.UNKNOWN, str(e))

    registry.register(ToolDefinition(
        name="get_kernel_metrics",
        description=(
            "Get performance metrics for a kernel by kernel_id. "
            "If metric_names is omitted, returns all metrics. "
            "Returns {name: {value, unit}} dict."
        ),
        parameters={
            "type": "object",
            "properties": {
                "kernel_id": {
                    "type": "integer",
                    "description": "Kernel ID from list_kernels.",
                },
                "metric_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of metric names to filter.",
                },
            },
            "required": ["kernel_id"],
        },
        handler=get_kernel_metrics,
        category="data_query",
    ))

    # --- get_kernel_summary ---
    async def get_kernel_summary(kernel_id: int) -> ToolResult:
        """Get a concise summary of a kernel's characteristics."""
        try:
            k = ctx.get_kernel(kernel_id)
            duration_ms = round(
                (k.metric_value("gpu__time_duration.sum") or 0) / 1e6, 2
            )
            summary = {
                "kernel_id": kernel_id,
                "name": k.demangled_name or k.kernel_name,
                "duration_ms": duration_ms,
                "launch": {
                    "grid": list(k.launch_params.grid),
                    "block": list(k.launch_params.block),
                    "shared_mem": k.launch_params.shared_mem_bytes,
                    "registers": k.launch_params.registers_per_thread,
                    "total_threads": k.launch_params.total_threads,
                },
                "device": {
                    "name": k.device_info.name,
                    "cc": (
                        f"{k.device_info.compute_capability[0]}"
                        f".{k.device_info.compute_capability[1]}"
                    ),
                    "sm_count": k.device_info.sm_count,
                },
                "key_metrics": {},
            }
            # Include key performance metrics if available
            key_names = [
                "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
                "sm__warps_active.avg.pct_of_peak_sustained_active",
            ]
            short = {"sm__throughput": "sm_pct", "gpu__dram_throughput": "dram_pct",
                      "sm__warps_active": "occupancy_pct"}
            for name in key_names:
                mv = k.metrics.get(name)
                if mv is not None:
                    prefix = name.split(".")[0]
                    label = short.get(prefix, prefix)
                    summary["key_metrics"][label] = round(mv.value, 1)
            return ToolResult.ok(summary)
        except IndexError as e:
            return ToolResult.fail(ErrorCode.METRIC_NOT_FOUND, str(e))
        except Exception as e:
            return ToolResult.fail(ErrorCode.UNKNOWN, str(e))

    registry.register(ToolDefinition(
        name="get_kernel_summary",
        description=(
            "Get concise kernel summary: launch params, device info, key metrics "
            "(SM throughput, DRAM throughput, occupancy). Good starting point."
        ),
        parameters={
            "type": "object",
            "properties": {
                "kernel_id": {
                    "type": "integer",
                    "description": "Kernel ID from list_kernels.",
                },
            },
            "required": ["kernel_id"],
        },
        handler=get_kernel_summary,
        category="data_query",
    ))

    # --- get_ncu_rule_results ---
    async def get_ncu_rule_results(kernel_id: int) -> ToolResult:
        """Get NCU built-in rule analysis results for a kernel."""
        try:
            k = ctx.get_kernel(kernel_id)
            return ToolResult.ok([
                {
                    "rule": r.rule_name,
                    "severity": r.severity,
                    "message": r.message[:200],  # truncate for token efficiency
                }
                for r in k.rule_results
            ])
        except IndexError as e:
            return ToolResult.fail(ErrorCode.METRIC_NOT_FOUND, str(e))
        except Exception as e:
            return ToolResult.fail(ErrorCode.UNKNOWN, str(e))

    registry.register(ToolDefinition(
        name="get_ncu_rule_results",
        description=(
            "Get NCU built-in rule analysis results (SpeedOfLight, "
            "MemoryWorkloadAnalysis, etc.). Complements Tachyon's analyzers."
        ),
        parameters={
            "type": "object",
            "properties": {
                "kernel_id": {
                    "type": "integer",
                    "description": "Kernel ID from list_kernels.",
                },
            },
            "required": ["kernel_id"],
        },
        handler=get_ncu_rule_results,
        category="data_query",
    ))
