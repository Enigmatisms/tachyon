"""Rule engine tools — expose analyzers and OptTree to the Agent.

2 tools:
  - run_analysis: Run rule-based analyzers on a kernel
  - get_optimization_tree: Get OptTree with active paths

Token-efficiency: Finding output is concise (title, severity, action, source_location).
OptTree output is just active paths + pruned count.
"""
from __future__ import annotations

from ..errors.handler import ErrorCode, ToolResult
from .context import SessionContext
from .registry import ToolDefinition, ToolRegistry


def register_analysis_tools(registry: ToolRegistry, ctx: SessionContext) -> None:
    """Register 2 rule engine tools."""

    # --- run_analysis ---
    async def run_analysis(
        kernel_id: int, analyzer_name: str | None = None,
    ) -> ToolResult:
        """Run rule-based analysis on a kernel."""
        if ctx.registry is None:
            return ToolResult.fail(
                ErrorCode.ANALYZER_FAILED,
                "AnalyzerRegistry not available in session context.",
            )
        try:
            kernel = ctx.get_kernel(kernel_id)
            if analyzer_name:
                # Run a single analyzer by name
                found = None
                for a in ctx.registry.all_analyzers():
                    if a.name() == analyzer_name:
                        found = a
                        break
                if found is None:
                    available = [a.name() for a in ctx.registry.all_analyzers()]
                    return ToolResult.fail(
                        ErrorCode.ANALYZER_FAILED,
                        f"Unknown analyzer '{analyzer_name}'. Available: {', '.join(available)}",
                    )
                if not found.can_run(kernel):
                    return ToolResult.fail(
                        ErrorCode.METRIC_NOT_FOUND,
                        f"Analyzer '{analyzer_name}' cannot run: missing metrics.",
                    )
                findings = found.analyze(kernel)
            else:
                # Run all analyzers
                findings = ctx.registry.run_all(
                    kernel, action=ctx.action, correlator=ctx.correlator,
                )
            return ToolResult.ok([
                {
                    "severity": f.severity.value,
                    "title": f.title,
                    "detail": f.detail[:300],  # truncate for token efficiency
                    "action": f.action[:200],
                    "source": f.source,
                    "category": f.category,
                    "source_location": (
                        f"{f.source_location.file}:{f.source_location.line}"
                        if f.source_location else None
                    ),
                }
                for f in findings
            ])
        except IndexError as e:
            return ToolResult.fail(ErrorCode.METRIC_NOT_FOUND, str(e))
        except Exception as e:
            return ToolResult.fail(ErrorCode.ANALYZER_FAILED, str(e))

    registry.register(ToolDefinition(
        name="run_analysis",
        description=(
            "Run Tachyon's rule-based analyzers on a kernel. "
            "Without analyzer_name, runs all (Roofline, Memory, WarpStall, NvRules). "
            "Returns structured Findings with severity and recommendations."
        ),
        parameters={
            "type": "object",
            "properties": {
                "kernel_id": {
                    "type": "integer",
                    "description": "Kernel ID from list_kernels.",
                },
                "analyzer_name": {
                    "type": "string",
                    "description": "Optional: roofline, memory, warp_stall, nvrules.",
                },
            },
            "required": ["kernel_id"],
        },
        handler=run_analysis,
        category="analysis",
    ))

    # --- get_optimization_tree ---
    async def get_optimization_tree(kernel_id: int) -> ToolResult:
        """Get the optimization exploration tree for a kernel."""
        if ctx.registry is None:
            return ToolResult.fail(
                ErrorCode.ANALYZER_FAILED,
                "AnalyzerRegistry not available.",
            )
        try:
            kernel = ctx.get_kernel(kernel_id)
            findings = ctx.registry.run_all(
                kernel, action=ctx.action, correlator=ctx.correlator,
            )
            from ..tree.opt_tree import OptimizationTree
            tree = OptimizationTree(findings)
            active = tree.active_paths()
            # Count pruned branches
            pruned_count = sum(
                1 for c in tree.root.children if c.pruned
            )
            return ToolResult.ok({
                "bottleneck": tree.root.strategy_name,
                "active_paths": [
                    {
                        "path": " → ".join(
                            n.strategy_name for n in path
                        ),
                        "category": path[-1].category if path else "",
                        "description": path[-1].description if path else "",
                    }
                    for path in active
                ],
                "pruned_branches": pruned_count,
                "tree_markdown": tree.to_markdown()[:500],  # truncated
            })
        except IndexError as e:
            return ToolResult.fail(ErrorCode.METRIC_NOT_FOUND, str(e))
        except Exception as e:
            return ToolResult.fail(ErrorCode.ANALYZER_FAILED, str(e))

    registry.register(ToolDefinition(
        name="get_optimization_tree",
        description=(
            "Get Optimization Tree: 5 branches (compute/memory/latency/"
            "load-balancing/balanced) with evidence-based pruning. "
            "Shows active optimization paths and recommended strategies."
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
        handler=get_optimization_tree,
        category="analysis",
    ))
