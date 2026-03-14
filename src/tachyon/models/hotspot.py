"""SourceHotspot model -- core output of SourceCorrelator (M2).

Represents a source-level hotspot with three-way mapping:
  metrics (stall counts, inst_executed) <-> source (file:line) <-> SASS/PTX.

Populated by SourceCorrelator.correlate(). Mutable dataclass to allow
post-construction attribute updates (local_ratio, global_ratio, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SourceHotspot:
    """A source location with aggregated performance data.

    Three-way mapping: metrics <-> source code <-> SASS/PTX instructions.

    Fields are populated in two phases:
    1. Constructor: pc, source_file/line, sass/ptx, metric_values, stall_reasons
    2. Post-construction: local_ratio, global_ratio, degraded, dominant_stall,
       is_hot (set by SourceCorrelator after dual-threshold test)
    """

    # -- Representative PC address --
    pc: int = 0

    # -- Source location --
    source_file: str = ""         # file path, or "<no debug info>"
    source_line: int = 0          # line number (0 if unknown)

    # -- Machine instructions --
    sass_instruction: str = ""    # SASS disassembly (concatenated if multiple PCs)
    ptx_instruction: str = ""     # PTX intermediate representation

    # -- Metric data --
    metric_values: dict[str, float] = field(default_factory=dict)
    # Aggregated instanced metric values for this source line.
    # Keys: full NCU metric names (e.g. "smsp__pcsamp_warps_issue_stalled_barrier")

    stall_reasons: dict[str, float] = field(default_factory=dict)
    # Stall metric ratios (0.0-1.0), normalized within this line.
    # Sum of all values == 1.0 (when stall data is present).

    # -- Hotspot detection --
    is_hot: bool = False          # passed dual-threshold test
    local_ratio: float = 0.0     # dominant_stall_value / line_total_samples
    global_ratio: float = 0.0    # line_total_samples / kernel_total_samples

    # -- Degradation --
    degraded: bool = False        # True when no source debug info available

    # -- Stall attribution --
    dominant_stall: str = ""      # name of top stall reason metric

    # -- Function context --
    function: str | None = None  # enclosing function name (if available)
