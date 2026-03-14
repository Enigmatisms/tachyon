"""Warp-Stall Analyzer — identify dominant warp stall reasons.

Uses PC-sampling stall-reason metrics to build a breakdown table and flag
the most impactful stall reasons with actionable guidance.

Dual-threshold approach:
  - LOCAL  = 30 % — a stall reason is flagged if it accounts for ≥ 30 % of
    *total stall samples* (dominates relative to other stalls).
  - GLOBAL = 10 % — a stall reason is flagged if it accounts for ≥ 10 % of
    *total warp-cycles* (significant in absolute terms).
"""
from __future__ import annotations

from tachyon.analyzers.base import Analyzer
from tachyon.models.finding import Finding, Severity
from tachyon.models.kernel import KernelReport

# ---------------------------------------------------------------------------
# Thresholds (percent)
# ---------------------------------------------------------------------------
LOCAL_THRESHOLD = 30.0   # % of total stalls
GLOBAL_THRESHOLD = 10.0  # % of total warp-cycles

# ---------------------------------------------------------------------------
# Metric prefix
# ---------------------------------------------------------------------------
_PREFIX = "smsp__pcsamp_warps_issue_stalled_"
_SUFFIX = ".sum"

# ---------------------------------------------------------------------------
# 18 stall reasons — (metric_key, human_name, explanation, action)
# ---------------------------------------------------------------------------
STALL_REASONS: list[tuple[str, str, str, str]] = [
    (
        "long_scoreboard",
        "Long Scoreboard",
        "Waiting for an L1TEX (global/local/texture) memory operation to complete.",
        "Improve memory access patterns, increase cache hit rate, or prefetch data.",
    ),
    (
        "barrier",
        "Barrier",
        "Waiting at a __syncthreads() or cooperative-group barrier.",
        "Reduce barrier frequency, balance work across warps, or remove unnecessary syncs.",
    ),
    (
        "math_pipe_throttle",
        "Math Pipe Throttle",
        "Math execution pipeline is fully occupied — back-pressure from ALU.",
        "Reduce arithmetic intensity, use faster intrinsics, or move work to tensor cores.",
    ),
    (
        "short_scoreboard",
        "Short Scoreboard",
        "Waiting for a short-latency MIO (shared memory, constant, or special func) op.",
        "Reduce shared-memory bank conflicts or constant-cache misses.",
    ),
    (
        "not_selected",
        "Not Selected",
        "Warp was eligible but scheduler chose another warp (resource contention).",
        "This is usually benign; high values may indicate sub-optimal occupancy balance.",
    ),
    (
        "wait",
        "Wait",
        "Warp stalled on a fixed-latency execution dependency.",
        "Interleave independent instructions to hide the dependency latency.",
    ),
    (
        "mio_throttle",
        "MIO Throttle",
        "MIO instruction queue is full.",
        "Reduce shared-memory or special-function pressure.",
    ),
    (
        "lg_throttle",
        "LG Throttle",
        "Local/global memory pipeline is congested.",
        "Reduce global memory traffic or improve L1 cache reuse.",
    ),
    (
        "tex_throttle",
        "Texture Throttle",
        "Texture pipeline is congested.",
        "Reduce texture fetch rate or improve texture cache locality.",
    ),
    (
        "dispatch_stall",
        "Dispatch Stall",
        "Warp could not dispatch (scheduler busy or port conflict).",
        "Reduce instruction mix diversity to alleviate port contention.",
    ),
    (
        "membar",
        "Memory Barrier",
        "Waiting for a __threadfence() or memory-order operation.",
        "Minimize fence scope (block vs. device vs. system) and frequency.",
    ),
    (
        "imc_miss",
        "IMC Miss",
        "Instruction cache miss — fetching instructions from L2/DRAM.",
        "Reduce code footprint; avoid excessive loop-unrolling.",
    ),
    (
        "drain",
        "Drain",
        "Waiting for outstanding memory writes to complete at exit.",
        "Reduce outstanding stores before kernel exit; coalesce writes.",
    ),
    (
        "no_instruction",
        "No Instruction",
        "No valid instruction to issue (e.g. branch target not yet fetched).",
        "Reduce branch divergence; keep hot loops compact.",
    ),
    (
        "sleeping",
        "Sleeping",
        "Warp is explicitly sleeping (nanosleep or similar).",
        "Only expected with explicit sleep APIs — verify intentional usage.",
    ),
    (
        "selected",
        "Selected (issuing)",
        "Warp was selected and issued an instruction (not really a stall).",
        "No action needed — this represents productive work.",
    ),
    (
        "branch_resolving",
        "Branch Resolving",
        "Waiting for a branch target to be resolved.",
        "Reduce branch divergence and use predication where possible.",
    ),
    (
        "misc",
        "Miscellaneous",
        "Other / uncategorised stall reason.",
        "Investigate with source-level PC sampling for more detail.",
    ),
]


def _metric_name(reason_key: str) -> str:
    """Build the full NCU metric name for a stall reason."""
    return f"{_PREFIX}{reason_key}{_SUFFIX}"


class WarpStallAnalyzer(Analyzer):
    """Identify the dominant warp-stall reasons from PC-sampling data.

    M2 enhancement: per-stall-type source-level attribution via hotspots.
    """

    def name(self) -> str:
        return "warp_stall"

    def category(self) -> str:
        return "latency"

    def required_metrics(self) -> list[str]:
        # Minimum viable set — long_scoreboard + barrier are the two most
        # common stall reasons collected by default.
        return [
            _metric_name("long_scoreboard"),
            _metric_name("barrier"),
        ]

    def analyze(self, report: KernelReport) -> list[Finding]:
        # Gather all available stall-reason values.
        stall_values: list[tuple[str, str, str, str, float]] = []
        for reason_key, human_name, explanation, action in STALL_REASONS:
            val = report.metric_value(_metric_name(reason_key))
            if val is not None:
                stall_values.append((reason_key, human_name, explanation, action, val))

        if not stall_values:
            return []

        total_stalls = sum(v[4] for v in stall_values)
        if total_stalls == 0:
            return []

        # Sort descending by stall count for the table.
        stall_values.sort(key=lambda x: x[4], reverse=True)

        findings: list[Finding] = []

        # --- Build summary text table ---
        lines: list[str] = []
        header = f"{'Reason':<25s} {'Samples':>12s} {'% of Total':>10s}"
        lines.append(header)
        lines.append("-" * len(header))
        for _reason_key, human_name, _expl, _act, val in stall_values:
            pct = (val / total_stalls) * 100.0
            lines.append(f"{human_name:<25s} {val:>12.0f} {pct:>9.1f}%")
        table_text = "\n".join(lines)

        findings.append(
            Finding(
                severity=Severity.INFO,
                title="Warp stall breakdown",
                detail=table_text,
                action="See individual stall-reason findings below for guidance.",
                source=self.name(),
                category=self.category(),
                metrics={_metric_name(r[0]): r[4] for r in stall_values},
            )
        )

        # --- Per-reason findings (dual threshold) ---
        # We use total_stalls as proxy for "warp-cycles" for the GLOBAL
        # threshold; in a full integration the denominator would be
        # smsp__warps_active.sum but total stall samples is a reasonable
        # proxy when only stall metrics are available.
        for reason_key, human_name, explanation, action, val in stall_values:
            local_pct = (val / total_stalls) * 100.0
            if local_pct >= LOCAL_THRESHOLD or local_pct >= GLOBAL_THRESHOLD:
                severity = Severity.WARNING if local_pct >= LOCAL_THRESHOLD else Severity.INFO
                finding = Finding(
                    severity=severity,
                    title=f"High warp stall: {human_name} ({local_pct:.1f}%)",
                    detail=(
                        f"{explanation}\n"
                        f"This reason accounts for {val:.0f} samples "
                        f"({local_pct:.1f}% of total stalls)."
                    ),
                    action=action,
                    source=self.name(),
                    category=self.category(),
                    metrics={_metric_name(reason_key): val},
                )

                # M2: source-level attribution per stall type
                hotspot = self._find_hotspot_for_stall(reason_key)
                self._attach_source_evidence(finding, hotspot)

                findings.append(finding)

        return findings
