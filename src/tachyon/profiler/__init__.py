"""Profiling layer -- NCU profiler integration.

Public API:
  - ToolPathResolver: NVIDIA tool path auto-detection
  - NcuProfiler: Two-stage smart profiling engine
  - ProfilingStrategy: conservative / radical strategy enum (internal)
  - AnalysisDepth: basic / deep / radical analysis depth (user-facing)
  - run_profiling_pipeline: Two-stage profiling pipeline (profiling only)
  - run_e2e_pipeline: Backward-compatible alias for run_profiling_pipeline
"""
from tachyon.profiler.ncu_profiler import AnalysisDepth, NcuProfiler, ProfilingStrategy
from tachyon.profiler.pipeline import run_e2e_pipeline, run_profiling_pipeline
from tachyon.profiler.tool_path import ToolPathResolver

__all__ = [
    "AnalysisDepth",
    "NcuProfiler",
    "ProfilingStrategy",
    "ToolPathResolver",
    "run_profiling_pipeline",
    "run_e2e_pipeline",
]
