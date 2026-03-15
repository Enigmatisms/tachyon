"""End-to-end profiling pipeline.

Orchestrates: NcuProfiler -> NcuReportReader -> Analyzers -> OptTree -> Report.
Optionally invokes Agent for AI-enhanced analysis.

References:
  - Implementation: section 4.2 (E2E Pipeline)
  - Architecture: section 10.3 (E2E flow)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from tachyon.config.settings import TachyonConfig
from tachyon.errors.handler import ToolResult
from tachyon.profiler.ncu_profiler import NcuProfiler, ProfilingStrategy
from tachyon.profiler.tool_path import ToolPathResolver

logger = logging.getLogger(__name__)


async def run_e2e_pipeline(
    executable: str,
    exe_args: list[str],
    *,
    config: TachyonConfig,
    strategy: ProfilingStrategy | None = None,
    top_k: int = 5,
    kernel_filter: list[str] | None = None,
    use_ai: bool = True,
    output_dir: Path | None = None,
    extra_ncu_args: list[str] | None = None,
    metric_set_override: str | None = None,
    metrics_override: str | None = None,
    verbose: bool = False,
) -> ToolResult[Path]:
    """Full profile -> analyze -> report pipeline.

    Steps:
      1. Stage 1: Quick Scan (NcuProfiler.profile_basic)
      2. Parse Stage 1 report (NcuReportReader) -> identify top-K kernels
      3. Stage 2: Deep Dive on top-K (NcuProfiler.profile_targeted)
      4. Parse Stage 2 report -> full analysis pipeline
      5. (Optional) Agent-enhanced analysis
      6. Generate report

    Returns:
        ToolResult wrapping the final report path (the .ncu-rep used for analysis).
    """
    resolver = ToolPathResolver(config)
    profiler = NcuProfiler(config, resolver)

    # --- Stage 1: Quick Scan ---
    # If user explicitly specified kernel_filter, skip Stage 1 entirely —
    # no need to scan all kernels just to discover top-K.
    if kernel_filter:
        logger.info("Kernel filter specified, skipping Stage 1 quick scan.")
        kernel_names = kernel_filter
        stage1_data = None
    else:
        logger.info("Stage 1: Quick Scan starting for %s", executable)
        stage1 = profiler.profile_basic(
            executable,
            exe_args,
            strategy=strategy,
            output_dir=output_dir,
            extra_ncu_args=extra_ncu_args,
            metric_set_override=metric_set_override,
            metrics_override=metrics_override,
        )
        if not stage1.success:
            return stage1  # type: ignore[return-value]

        assert stage1.data is not None
        stage1_data = stage1.data
        logger.info(
            "Stage 1 complete: %s (%.1fs)",
            stage1.data.ncu_rep_path,
            stage1.data.elapsed_sec,
        )

        # --- Parse Stage 1, find top-K kernels ---
        from tachyon.reader.ncu_reader import NcuReportReader

        reader = NcuReportReader()
        load_result = reader.load(stage1.data.ncu_rep_path)
        if not load_result.success:
            return load_result  # type: ignore[return-value]

        assert load_result.data is not None
        kernels = load_result.data

        # Sort by duration (if available) or SM throughput, pick top-K.
        def _sort_key(k: Any) -> float:
            """Sort kernels by duration, falling back to SM throughput."""
            dur = k.metrics.get("gpu__time_duration.sum")
            if dur:
                return dur.value
            sm = k.metrics.get("sm__throughput.avg.pct_of_peak_sustained_elapsed")
            return sm.value if sm else 0.0

        sorted_kernels = sorted(kernels, key=_sort_key, reverse=True)
        top_kernels = sorted_kernels[:top_k]
        kernel_names = list({k.kernel_name for k in top_kernels})

    logger.info("Top-%d kernels for Stage 2: %s", top_k, kernel_names)

    # --- Stage 2: Deep Dive on top-K ---
    stage2 = profiler.profile_targeted(
        executable,
        exe_args,
        kernels=kernel_names,
        strategy=strategy,
        output_dir=output_dir,
        extra_ncu_args=extra_ncu_args,
        metric_set_override=metric_set_override,
        metrics_override=metrics_override,
    )

    if not stage2.success:
        assert stage2.error is not None
        if stage1_data is not None:
            # Fallback: use Stage 1 data for analysis (degraded but functional).
            logger.warning("Stage 2 failed, falling back to Stage 1 data: %s", stage2.error.message)
            report_path = stage1_data.ncu_rep_path
        else:
            # No Stage 1 data (kernel_filter mode skipped it), propagate error.
            return stage2  # type: ignore[return-value]
    else:
        assert stage2.data is not None
        report_path = stage2.data.ncu_rep_path
        logger.info(
            "Stage 2 complete: %s (%.1fs)",
            stage2.data.ncu_rep_path,
            stage2.data.elapsed_sec,
        )

    return ToolResult.ok(report_path)
