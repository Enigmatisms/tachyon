"""Two-stage NCU profiling pipeline.

Orchestrates: NcuProfiler Stage 1 (Quick Scan) → top-K selection → Stage 2 (Deep Dive).

This module is **profiling only** — it returns the .ncu-rep path for downstream
analysis. Analysis (Rule Engine + AI) is handled by ``tachyon.analysis.pipeline``.

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


async def run_profiling_pipeline(
    executable: str,
    exe_args: list[str],
    *,
    config: TachyonConfig,
    strategy: ProfilingStrategy | None = None,
    top_k: int = 5,
    kernel_filter: list[str] | None = None,
    output_dir: Path | None = None,
    extra_ncu_args: list[str] | None = None,
    metric_set_override: str | None = None,
    metrics_override: str | None = None,
    verbose: bool = False,
) -> ToolResult[Path]:
    """Two-stage smart profiling: Quick Scan → identify top-K → Deep Dive.

    Steps:
      1. Stage 1: Quick Scan (all kernels, basic/detailed metrics)
      2. Parse Stage 1 → rank kernels by duration → select top-K
      3. Stage 2: Deep Dive (top-K kernels, detailed/full metrics)

    If ``kernel_filter`` is specified, Stage 1 is skipped entirely.

    Returns:
        ToolResult wrapping the final .ncu-rep path for downstream analysis.
    """
    resolver = ToolPathResolver(config)
    profiler = NcuProfiler(config, resolver)

    # --- Stage 1: Quick Scan ---
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
            verbose=verbose,
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

        # Parse Stage 1, find top-K kernels
        from tachyon.reader.ncu_reader import NcuReportReader

        reader = NcuReportReader(config)
        load_result = reader.load(stage1.data.ncu_rep_path)
        if not load_result.success:
            return load_result  # type: ignore[return-value]

        assert load_result.data is not None
        kernels = load_result.data

        def _sort_key(k: Any) -> float:
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
        verbose=verbose,
    )

    if not stage2.success:
        assert stage2.error is not None
        if stage1_data is not None:
            logger.warning(
                "Stage 2 failed, falling back to Stage 1 data: %s",
                stage2.error.message,
            )
            report_path = stage1_data.ncu_rep_path
        else:
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


# Backward compatibility alias
run_e2e_pipeline = run_profiling_pipeline
