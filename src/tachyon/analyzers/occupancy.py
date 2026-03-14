"""Occupancy Analyzer — identify occupancy-limiting factors.

Examines achieved occupancy vs. theoretical limits and pinpoints the
dominant limiter (registers, shared memory, or block count).

Thresholds:
  - LOW   = 50 %  achieved occupancy → WARNING
  - HIGH  = 75 %  achieved occupancy → healthy (INFO)
  - REGISTER_SPILL = 64  registers/thread → WARNING (likely spilling)
"""
from __future__ import annotations

from tachyon.analyzers.base import Analyzer
from tachyon.models.finding import Finding, Severity
from tachyon.models.kernel import KernelReport

# ---------------------------------------------------------------------------
# Metric names
# ---------------------------------------------------------------------------
_ACHIEVED_OCCUPANCY = "sm__warps_active.avg.pct_of_peak_sustained_active"
_LIMIT_WARPS = "launch__occupancy_limit_warps"
_LIMIT_REGISTERS = "launch__occupancy_limit_registers"
_LIMIT_SHARED_MEM = "launch__occupancy_limit_shared_mem"
_LIMIT_BLOCKS = "launch__occupancy_limit_blocks"

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
LOW_OCCUPANCY = 50.0      # below → WARNING
HIGH_OCCUPANCY = 75.0     # above → healthy INFO
REGISTER_SPILL = 64       # registers/thread → WARNING


class OccupancyAnalyzer(Analyzer):
    """Analyze occupancy-limiting factors and achieved occupancy.

    M2 enhancement: annotates findings with top contributing source line.
    """

    def name(self) -> str:
        return "occupancy"

    def category(self) -> str:
        return "occupancy"

    def required_metrics(self) -> list[str]:
        return [
            _ACHIEVED_OCCUPANCY,
            _LIMIT_WARPS,
            _LIMIT_REGISTERS,
            _LIMIT_SHARED_MEM,
            _LIMIT_BLOCKS,
        ]

    def analyze(self, report: KernelReport) -> list[Finding]:
        achieved = report.metric_value(_ACHIEVED_OCCUPANCY)
        limit_warps = report.metric_value(_LIMIT_WARPS)
        limit_regs = report.metric_value(_LIMIT_REGISTERS)
        limit_smem = report.metric_value(_LIMIT_SHARED_MEM)
        limit_blocks = report.metric_value(_LIMIT_BLOCKS)

        # Defensive: required_metrics guarantees presence, but guard anyway.
        if any(
            v is None
            for v in (
                achieved, limit_warps, limit_regs, limit_smem, limit_blocks
            )
        ):
            return []

        findings: list[Finding] = []
        metrics = {
            _ACHIEVED_OCCUPANCY: achieved,
            _LIMIT_WARPS: limit_warps,
            _LIMIT_REGISTERS: limit_regs,
            _LIMIT_SHARED_MEM: limit_smem,
            _LIMIT_BLOCKS: limit_blocks,
        }

        # M2: get top contributing source line for annotation
        top_hotspot = self._hotspots[0] if self._hotspots else None

        # --- Identify the dominant occupancy limiter ---
        limiters = {
            "registers": limit_regs,
            "shared_mem": limit_smem,
            "blocks": limit_blocks,
        }
        dominant_limiter = min(limiters, key=limiters.get)  # type: ignore[arg-type]
        dominant_value = limiters[dominant_limiter]

        limiter_descriptions = {
            "registers": "register usage per thread",
            "shared_mem": "shared memory usage per block",
            "blocks": "maximum blocks per SM",
        }
        limiter_actions = {
            "registers": (
                "Reduce register usage: simplify kernel logic, use "
                "__launch_bounds__, or trade registers for shared memory."
            ),
            "shared_mem": (
                "Reduce shared memory per block: shrink shared arrays, "
                "use dynamic allocation, or split across multiple kernels."
            ),
            "blocks": (
                "Use smaller block sizes to allow more blocks per SM, "
                "or restructure the kernel to reduce per-block resource usage."
            ),
        }

        # --- Low occupancy check ---
        if achieved < LOW_OCCUPANCY:
            findings.append(
                Finding(
                    severity=Severity.WARNING,
                    title=f"Low occupancy ({achieved:.1f}%)",
                    detail=(
                        f"Achieved occupancy is {achieved:.1f}%, below the "
                        f"{LOW_OCCUPANCY:.0f}% threshold. Theoretical max warps: "
                        f"{limit_warps:.0f}. The dominant limiter is "
                        f"{limiter_descriptions[dominant_limiter]} "
                        f"(limit: {dominant_value:.0f} warps)."
                    ),
                    action=limiter_actions[dominant_limiter],
                    source=self.name(),
                    category=self.category(),
                    metrics=metrics,
                )
            )

        # --- Register pressure check ---
        regs_per_thread = report.launch_params.registers_per_thread
        if regs_per_thread > REGISTER_SPILL:
            findings.append(
                Finding(
                    severity=Severity.WARNING,
                    title=(
                        f"High register pressure "
                        f"({regs_per_thread} regs/thread, possible spill)"
                    ),
                    detail=(
                        f"The kernel uses {regs_per_thread} registers per thread, "
                        f"exceeding the {REGISTER_SPILL}-register threshold. "
                        "High register usage reduces occupancy and may cause "
                        "register spilling to local memory, increasing latency."
                    ),
                    action=(
                        "Use __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor) "
                        "to hint the compiler, simplify per-thread state, or "
                        "manually spill to shared memory for better control."
                    ),
                    source=self.name(),
                    category=self.category(),
                    metrics={
                        "registers_per_thread": float(regs_per_thread),
                        _LIMIT_REGISTERS: limit_regs,
                    },
                )
            )

        # --- Healthy occupancy ---
        if achieved > HIGH_OCCUPANCY:
            findings.append(
                Finding(
                    severity=Severity.INFO,
                    title=f"Occupancy is healthy ({achieved:.1f}%)",
                    detail=(
                        f"Achieved occupancy is {achieved:.1f}%, above the "
                        f"{HIGH_OCCUPANCY:.0f}% healthy threshold. "
                        f"Theoretical max warps: {limit_warps:.0f}."
                    ),
                    action=(
                        "Occupancy is not the bottleneck. Focus optimization "
                        "efforts on other areas (memory throughput, instruction "
                        "mix, warp stalls)."
                    ),
                    source=self.name(),
                    category=self.category(),
                    metrics=metrics,
                )
            )

        # M2: annotate all findings with top contributing source line
        for finding in findings:
            self._attach_source_evidence(finding, top_hotspot)

        return findings
