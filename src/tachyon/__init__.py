"""Tachyon — AI-Powered CUDA Performance Analyzer.

SDK / library mode (Mode C) public API::

    from tachyon import NcuProfiler, ToolPathResolver, TachyonConfig

    config = TachyonConfig.load()
    profiler = NcuProfiler(config, ToolPathResolver(config))
    stage1 = profiler.profile_basic("./my_app")
"""

__version__ = "0.1.0"

# Profiler (M4)
# Config
from tachyon.config.settings import TachyonConfig
from tachyon.profiler.ncu_profiler import NcuProfiler, ProfilingStrategy
from tachyon.profiler.tool_path import ToolPathResolver

__all__ = [
    "NcuProfiler",
    "ProfilingStrategy",
    "ToolPathResolver",
    "TachyonConfig",
]
