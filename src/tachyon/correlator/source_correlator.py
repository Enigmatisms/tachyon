"""SourceCorrelator -- Three-way mapping engine (metrics <-> source <-> SASS/PTX).

Core innovation of Tachyon. Takes instanced metrics from NcuReportReader,
correlates each PC address to source location and SASS instruction,
aggregates by source line, and applies dual-threshold hotspot detection.

Design constraints:
- ActionHandle is a Protocol (no import from ncu_reader to avoid circular deps)
- SourceInfo is defined here as a lightweight dataclass
- All stall metric names use the NCU canonical prefix
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol

from tachyon.models.hotspot import SourceHotspot

# ---------------------------------------------------------------------------
# Stall & execution metric name constants
# ---------------------------------------------------------------------------

# 18 warp stall reason metrics from NCU SourceCounters section.
STALL_METRICS: list[str] = [
    "smsp__pcsamp_warps_issue_stalled_barrier",
    "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_lg_throttle",
    "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle",
    "smsp__pcsamp_warps_issue_stalled_short_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_mio_throttle",
    "smsp__pcsamp_warps_issue_stalled_wait",
    "smsp__pcsamp_warps_issue_stalled_drain",
    "smsp__pcsamp_warps_issue_stalled_imc_miss",
    "smsp__pcsamp_warps_issue_stalled_membar",
    "smsp__pcsamp_warps_issue_stalled_misc",
    "smsp__pcsamp_warps_issue_stalled_no_instructions",
    "smsp__pcsamp_warps_issue_stalled_not_selected",
    "smsp__pcsamp_warps_issue_stalled_sleeping",
    "smsp__pcsamp_warps_issue_stalled_tex_throttle",
    "smsp__pcsamp_warps_issue_stalled_branch_resolving",
    "smsp__pcsamp_warps_issue_stalled_dispatch_stall",
    "smsp__pcsamp_warps_issue_stalled_selected",
]

# Per-PC instruction execution metrics.
EXEC_METRICS: list[str] = [
    "inst_executed",
    "thread_inst_executed_true",
]

# Complete list of instanced metrics to collect.
INSTANCED_METRICS: list[str] = STALL_METRICS + EXEC_METRICS

# Substring used to identify stall metrics in arbitrary metric names.
_STALL_SUBSTR = "pcsamp_warps_issue_stalled"

# Substring used to identify execution count metrics.
_EXEC_SUBSTR = "inst_executed"


# ---------------------------------------------------------------------------
# SourceInfo -- lightweight source location from NCU
# ---------------------------------------------------------------------------

@dataclass
class SourceInfo:
    """Simplified source information returned by ActionHandle.source_info()."""
    file_name: str
    line: int


# ---------------------------------------------------------------------------
# ActionHandle -- Protocol for NCU action handle
# ---------------------------------------------------------------------------

class ActionHandle(Protocol):
    """Protocol for NCU action handle -- implemented by NcuReportReader.

    Defines the three correlation APIs that SourceCorrelator depends on.
    Using a Protocol avoids importing ncu_reader (circular dependency).
    """

    def source_info(self, pc: int, kernel_name: str | None = None) -> SourceInfo | None:
        """Map PC -> source file + line number.

        Returns None when -lineinfo was not used during compilation.
        """
        ...

    def sass_by_pc(self, pc: int, kernel_name: str | None = None) -> str | None:
        """Map PC -> SASS disassembly text.

        Always available (does not require -lineinfo).
        """
        ...

    def ptx_by_pc(self, pc: int, kernel_name: str | None = None) -> str | None:
        """Map PC -> PTX intermediate representation text."""
        ...


# ---------------------------------------------------------------------------
# Internal accumulators (not part of public API)
# ---------------------------------------------------------------------------

@dataclass
class PcAccumulator:
    """Accumulates metric samples for a single PC address."""

    pc: int = 0
    samples: dict[str, float] = field(default_factory=dict)
    # metric_name -> aggregated value (SUM for stall/exec counts, MAX for others)

    source_file: str | None = None
    source_line: int | None = None
    sass_text: str | None = None
    ptx_text: str | None = None

    @property
    def total_stall_samples(self) -> float:
        """Sum of all stall-related metric samples at this PC."""
        return sum(
            v for k, v in self.samples.items()
            if _STALL_SUBSTR in k
        )

    @property
    def total_samples(self) -> float:
        """Total instruction execution samples at this PC.

        Uses inst_executed as the canonical sample count; falls back to
        the sum of all stall metrics if inst_executed is not available.
        """
        if "inst_executed" in self.samples:
            return self.samples["inst_executed"]
        return self.total_stall_samples

    def add_sample(self, metric_name: str, value: float) -> None:
        """Accumulate a metric sample.

        Aggregation strategy:
        - Stall metrics (pcsamp_warps_issue_stalled_*): SUM
        - Execution counts (inst_executed, thread_inst_executed_true): SUM
        - All other metrics (throughput, ratios): MAX
        """
        if _STALL_SUBSTR in metric_name or _EXEC_SUBSTR in metric_name:
            self.samples[metric_name] = self.samples.get(metric_name, 0.0) + value
        else:
            # Throughput / ratio metrics: keep MAX across correlation entries
            self.samples[metric_name] = max(
                self.samples.get(metric_name, 0.0), value
            )


@dataclass
class LineAccumulator:
    """Accumulates metrics from multiple PCs belonging to the same source line."""

    pcs: list[PcAccumulator] = field(default_factory=list)
    aggregated: dict[str, float] = field(default_factory=dict)
    sass_list: list[str] = field(default_factory=list)

    def merge_pc(self, pc_acc: PcAccumulator) -> None:
        """Merge a PC accumulator into this line accumulator.

        - Appends the PC to the internal list
        - Deduplicates SASS text
        - Aggregates metrics: SUM for stall/exec, MAX for others
        """
        self.pcs.append(pc_acc)
        if pc_acc.sass_text and pc_acc.sass_text not in self.sass_list:
            self.sass_list.append(pc_acc.sass_text)
        for metric_name, value in pc_acc.samples.items():
            if _STALL_SUBSTR in metric_name or _EXEC_SUBSTR in metric_name:
                self.aggregated[metric_name] = (
                    self.aggregated.get(metric_name, 0.0) + value
                )
            else:
                self.aggregated[metric_name] = max(
                    self.aggregated.get(metric_name, 0.0), value
                )

    @property
    def total_stall_samples(self) -> float:
        """Sum of all stall metric values aggregated at this source line."""
        return sum(
            v for k, v in self.aggregated.items()
            if _STALL_SUBSTR in k
        )

    @property
    def total_samples(self) -> float:
        """Total samples at this source line.

        Uses inst_executed if available, otherwise falls back to stall sum.
        """
        if "inst_executed" in self.aggregated:
            return self.aggregated["inst_executed"]
        return self.total_stall_samples

    @property
    def dominant_stall_reason(self) -> tuple[str, float]:
        """Return (stall_metric_name, value) for the highest stall contributor.

        Returns ("none", 0.0) when no stall metrics are present.
        """
        stalls = {
            k: v for k, v in self.aggregated.items()
            if _STALL_SUBSTR in k
        }
        if not stalls:
            return ("none", 0.0)
        top = max(stalls, key=stalls.get)  # type: ignore[arg-type]
        return (top, stalls[top])


# ---------------------------------------------------------------------------
# CorrelatorConfig
# ---------------------------------------------------------------------------

@dataclass
class CorrelatorConfig:
    """Configurable thresholds for hotspot detection."""

    local_ratio_threshold: float = 0.30
    """Minimum local_ratio (dominant_stall / line_total) to qualify as hot."""

    global_ratio_threshold: float = 0.10
    """Minimum global_ratio (line_total / kernel_total) to qualify as hot."""

    max_hotspots: int = 20
    """Maximum number of hotspots to return."""


# ---------------------------------------------------------------------------
# SourceCorrelator
# ---------------------------------------------------------------------------

class SourceCorrelator:
    """Three-way mapping engine: metrics <-> source <-> SASS/PTX.

    Algorithm (5 steps):
    1. Extract PC addresses from correlation_ids (instanced metric entries)
    2. For each PC: get source_info + sass_by_pc + ptx_by_pc from action handle
    3. Aggregate by source line (SUM stalls/exec, MAX throughput, keep all SASS)
    4. Dual-threshold hotspot: local_ratio > 0.30 AND global_ratio > 0.10
    5. Sort by global_ratio desc, return top max_hotspots

    Degradation: when -lineinfo is absent, source_info returns None.
    Those PCs fall back to PC-level pseudo-lines with degraded=True.
    """

    def __init__(self, config: CorrelatorConfig | None = None) -> None:
        self.config = config or CorrelatorConfig()

    def correlate(
        self,
        action: ActionHandle,
        instanced_metrics: dict[str, list[tuple[int, float]]],
        kernel_name: str | None = None,
    ) -> list[SourceHotspot]:
        """Execute the 5-step correlation algorithm.

        Args:
            action: Object implementing the ActionHandle protocol (exposes
                    source_info, sass_by_pc, ptx_by_pc).
            instanced_metrics: Dict mapping metric name to list of
                    (pc_address, value) tuples from NcuReportReader.
            kernel_name: Optional kernel name for multi-kernel reports.
                    Passed through to action methods to select the correct
                    NCU action handle.

        Returns:
            List of SourceHotspot sorted by global_ratio descending,
            limited to config.max_hotspots entries.
        """
        # -- Step 1: Extract PC addresses from correlation_ids ---------------
        pc_stats: dict[int, PcAccumulator] = defaultdict(
            lambda: PcAccumulator()
        )
        for metric_name, entries in instanced_metrics.items():
            for pc, value in entries:
                acc = pc_stats[pc]
                acc.pc = pc
                acc.add_sample(metric_name, value)

        if not pc_stats:
            return []

        # -- Step 2: For each PC, get source location + SASS/PTX ------------
        for pc, acc in pc_stats.items():
            src_info = action.source_info(pc, kernel_name=kernel_name)
            if src_info is not None:
                acc.source_file = src_info.file_name
                acc.source_line = src_info.line
            acc.sass_text = action.sass_by_pc(pc, kernel_name=kernel_name)
            acc.ptx_text = action.ptx_by_pc(pc, kernel_name=kernel_name)

        # -- Step 3: Aggregate by source line --------------------------------
        # Location key: (file, line) for mapped PCs, (None, pc) for unmapped
        loc_key_t = tuple[str | None, int | None]
        line_stats: dict[loc_key_t, LineAccumulator] = defaultdict(
            LineAccumulator
        )

        for pc, acc in pc_stats.items():
            if acc.source_file is not None and acc.source_line is not None:
                key: loc_key_t = (acc.source_file, acc.source_line)
            else:
                # No debug info: use PC address as pseudo source-line key
                key = (None, pc)
            line_stats[key].merge_pc(acc)

        # -- Step 4: Dual-threshold hotspot detection ------------------------
        kernel_total = sum(la.total_samples for la in line_stats.values())
        if kernel_total <= 0:
            return []

        hotspots: list[SourceHotspot] = []
        for loc_key, line_acc in line_stats.items():
            line_total = line_acc.total_samples
            if line_total <= 0:
                continue

            dominant_name, dominant_val = line_acc.dominant_stall_reason
            local_ratio = dominant_val / line_total if line_total > 0 else 0.0
            global_ratio = line_total / kernel_total

            # Strict > for both thresholds (exactly-at-threshold is NOT hot)
            is_hot = (
                local_ratio > self.config.local_ratio_threshold
                and global_ratio > self.config.global_ratio_threshold
            )

            src_file, src_line = loc_key
            degraded = src_file is None

            hotspot = SourceHotspot(
                pc=line_acc.pcs[0].pc if line_acc.pcs else 0,
                source_file=src_file or "<no debug info>",
                source_line=src_line or 0,
                sass_instruction="; ".join(line_acc.sass_list[:5]),
                ptx_instruction=(
                    line_acc.pcs[0].ptx_text or "" if line_acc.pcs else ""
                ),
                metric_values=dict(line_acc.aggregated),
                stall_reasons=self._extract_stall_ratios(line_acc),
                is_hot=is_hot,
                local_ratio=local_ratio,
                global_ratio=global_ratio,
                degraded=degraded,
                dominant_stall=dominant_name,
            )

            if is_hot:
                hotspots.append(hotspot)

        # -- Step 5: Sort by global_ratio descending -------------------------
        hotspots.sort(key=lambda h: h.global_ratio, reverse=True)
        return hotspots[: self.config.max_hotspots]

    def check_degradation(
        self, hotspots: list[SourceHotspot]
    ) -> str | None:
        """Check if source correlation is degraded and return warning message.

        Returns:
            None if no degradation detected (all hotspots have source info).
            Warning string describing the degradation otherwise.
        """
        if not hotspots:
            return None

        degraded_count = sum(1 for h in hotspots if h.degraded)
        if degraded_count == 0:
            return None

        if degraded_count == len(hotspots):
            return (
                "Source mapping unavailable: kernel was compiled without "
                "debug info. Current output is SASS PC-level hotspot "
                "analysis. Suggestion: add -lineinfo to compiler flags for "
                "full source attribution. (nvcc -lineinfo has <1% runtime "
                "impact, safe for profiling builds)"
            )

        return (
            f"{degraded_count}/{len(hotspots)} hotspots lack source mapping. "
            "Partial debug info detected. Recompile all translation units "
            "with -lineinfo."
        )

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _extract_stall_ratios(
        line_acc: LineAccumulator,
    ) -> dict[str, float]:
        """Convert absolute stall counts to ratios (0.0-1.0) for a source line.

        Only includes stall metrics with non-zero values. Ratios are
        relative to the total stall samples at this line, so they sum to 1.0.
        """
        total = line_acc.total_stall_samples
        if total <= 0:
            return {}
        return {
            k: v / total
            for k, v in line_acc.aggregated.items()
            if _STALL_SUBSTR in k and v > 0
        }
