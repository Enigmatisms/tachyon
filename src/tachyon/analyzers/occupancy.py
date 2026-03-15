"""Occupancy Analyzer — identify occupancy-limiting factors.

Examines achieved occupancy vs. theoretical limits and pinpoints the
dominant limiter (registers, shared memory, or block count).

Design: Graceful degradation — runs with just achieved occupancy and
launch_params.registers_per_thread.  Full limiter analysis requires
the occupancy limit metrics (warps/regs/smem/blocks), but partial
analysis is still valuable.

Thresholds:
  - LOW   = 50 %  achieved occupancy → WARNING
  - HIGH  = 75 %  achieved occupancy → healthy (INFO)
  - REGISTER_SPILL = 64  registers/thread → WARNING (likely spilling)
  - LOCAL_MEM_THRESHOLD = 0  bytes → any local memory = WARNING (spill indicator)
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
# Local memory (register spill indicator)
_LOCAL_STORE = "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum"
_LOCAL_LOAD = "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum"

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
LOW_OCCUPANCY = 50.0      # below → WARNING
HIGH_OCCUPANCY = 75.0     # above → healthy INFO
REGISTER_SPILL = 64       # registers/thread → WARNING
LOCAL_MEM_WARN = 0        # any local memory traffic → WARNING


class OccupancyAnalyzer(Analyzer):
    """Analyze occupancy-limiting factors and achieved occupancy.

    Graceful degradation:
      - Full mode: achieved occupancy + all 4 limit metrics → detailed limiter analysis
      - Partial mode: just launch_params → register pressure + local memory detection
      - Always checks: registers_per_thread, shared_mem_bytes, local memory traffic
    """

    def name(self) -> str:
        return "occupancy"

    def category(self) -> str:
        return "occupancy"

    def required_metrics(self) -> list[str]:
        # No metrics strictly required — launch_params always available
        return []

    def can_run(self, report: KernelReport) -> bool:
        """Always runnable — launch_params are always present."""
        return True

    def analyze(self, report: KernelReport) -> list[Finding]:
        findings: list[Finding] = []

        achieved = report.metric_value(_ACHIEVED_OCCUPANCY)
        limit_warps = report.metric_value(_LIMIT_WARPS)
        limit_regs = report.metric_value(_LIMIT_REGISTERS)
        limit_smem = report.metric_value(_LIMIT_SHARED_MEM)
        limit_blocks = report.metric_value(_LIMIT_BLOCKS)

        # M2: get top contributing source line for annotation
        top_hotspot = self._hotspots[0] if self._hotspots else None

        has_full_limits = all(
            v is not None
            for v in (limit_warps, limit_regs, limit_smem, limit_blocks)
        )

        # ── Register pressure check (always available from launch_params) ──
        regs_per_thread = report.launch_params.registers_per_thread
        if regs_per_thread > REGISTER_SPILL:
            metrics = {"registers_per_thread": float(regs_per_thread)}
            if limit_regs is not None:
                metrics[_LIMIT_REGISTERS] = limit_regs
            findings.append(Finding(
                severity=Severity.WARNING,
                title=(
                    f"High register pressure "
                    f"({regs_per_thread} regs/thread, likely spilling to local memory)"
                ),
                detail=(
                    f"The kernel uses {regs_per_thread} registers per thread, "
                    f"exceeding the {REGISTER_SPILL}-register threshold. "
                    "High register usage reduces occupancy and may cause "
                    "register spilling to local memory (LMEM), significantly "
                    "increasing memory latency (L1 cache access per spilled variable)."
                ),
                action=(
                    "Use __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor) "
                    "to hint the compiler. Simplify per-thread state, reduce live "
                    "variables across loops, or manually spill to shared memory "
                    "for better control."
                ),
                source=self.name(),
                category=self.category(),
                metrics=metrics,
            ))

        # ── Local memory traffic check (spill indicator) ──
        local_st = report.metric_value(_LOCAL_STORE)
        local_ld = report.metric_value(_LOCAL_LOAD)
        if local_st is not None and local_ld is not None:
            local_total = local_st + local_ld
            if local_total > LOCAL_MEM_WARN:
                findings.append(Finding(
                    severity=Severity.WARNING,
                    title=f"Register spilling detected ({local_total:.0f} local memory sectors)",
                    detail=(
                        f"Local memory traffic: {local_ld:.0f} load sectors + "
                        f"{local_st:.0f} store sectors = {local_total:.0f} total. "
                        "Local memory is off-chip DRAM accessed through L1 cache — "
                        "each spilled register access costs ~100x more latency than "
                        "a register access. This is a strong indicator of register "
                        "pressure causing the compiler to spill to local memory."
                    ),
                    action=(
                        "Reduce register pressure: use __launch_bounds__(), simplify "
                        "kernel logic, reduce the number of live variables, or split "
                        "the kernel into smaller sub-kernels."
                    ),
                    source=self.name(),
                    category=self.category(),
                    metrics={
                        _LOCAL_STORE: local_st,
                        _LOCAL_LOAD: local_ld,
                        "registers_per_thread": float(regs_per_thread),
                    },
                ))

        # ── Shared memory usage check ──
        shared_bytes = report.launch_params.shared_mem_bytes
        if shared_bytes > 0 and report.device_info is not None:
            # 48 KB is the default shared memory per SM on most architectures
            max_shared_per_sm = 49152  # 48 KB default
            if shared_bytes > max_shared_per_sm * 0.8:
                findings.append(Finding(
                    severity=Severity.WARNING,
                    title=f"High shared memory usage ({shared_bytes} bytes/block)",
                    detail=(
                        f"The kernel uses {shared_bytes} bytes of shared memory "
                        f"per block ({shared_bytes / 1024:.1f} KB), which is "
                        f">80% of the default {max_shared_per_sm / 1024:.0f} KB limit. "
                        "High shared memory usage can limit the number of concurrent "
                        "blocks per SM, reducing occupancy."
                    ),
                    action=(
                        "Consider reducing shared memory usage, or use "
                        "cudaFuncSetAttribute() to increase the shared memory "
                        "limit if your GPU supports it (up to 164 KB on H100)."
                    ),
                    source=self.name(),
                    category=self.category(),
                    metrics={
                        "shared_mem_bytes": float(shared_bytes),
                    },
                ))

        # ── Achieved occupancy analysis (requires metric) ──
        if achieved is not None:
            metrics_occ: dict[str, float] = {_ACHIEVED_OCCUPANCY: achieved}

            # Full limiter analysis
            if has_full_limits:
                assert limit_warps is not None
                assert limit_regs is not None
                assert limit_smem is not None
                assert limit_blocks is not None

                metrics_occ.update({
                    _LIMIT_WARPS: limit_warps,
                    _LIMIT_REGISTERS: limit_regs,
                    _LIMIT_SHARED_MEM: limit_smem,
                    _LIMIT_BLOCKS: limit_blocks,
                })

                limiters = {
                    "registers": limit_regs,
                    "shared_mem": limit_smem,
                    "blocks": limit_blocks,
                }
                dominant = min(limiters, key=limiters.get)  # type: ignore[arg-type]
                dominant_value = limiters[dominant]

                limiter_desc = {
                    "registers": "register usage per thread",
                    "shared_mem": "shared memory usage per block",
                    "blocks": "maximum blocks per SM",
                }
                limiter_action = {
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

                if achieved < LOW_OCCUPANCY:
                    findings.append(Finding(
                        severity=Severity.WARNING,
                        title=f"Low occupancy ({achieved:.1f}%)",
                        detail=(
                            f"Achieved occupancy is {achieved:.1f}%, below the "
                            f"{LOW_OCCUPANCY:.0f}% threshold. Theoretical max warps: "
                            f"{limit_warps:.0f}. The dominant limiter is "
                            f"{limiter_desc[dominant]} "
                            f"(limit: {dominant_value:.0f} warps)."
                        ),
                        action=limiter_action[dominant],
                        source=self.name(),
                        category=self.category(),
                        metrics=metrics_occ,
                    ))
                elif achieved > HIGH_OCCUPANCY:
                    findings.append(Finding(
                        severity=Severity.INFO,
                        title=f"Occupancy is healthy ({achieved:.1f}%)",
                        detail=(
                            f"Achieved occupancy is {achieved:.1f}%, above the "
                            f"{HIGH_OCCUPANCY:.0f}% healthy threshold. "
                            f"Theoretical max warps: {limit_warps:.0f}."
                        ),
                        action=(
                            "Occupancy is not the bottleneck. Focus optimization "
                            "efforts on other areas."
                        ),
                        source=self.name(),
                        category=self.category(),
                        metrics=metrics_occ,
                    ))
            else:
                # Partial: have achieved but not full limiters
                if achieved < LOW_OCCUPANCY:
                    findings.append(Finding(
                        severity=Severity.WARNING,
                        title=f"Low occupancy ({achieved:.1f}%)",
                        detail=(
                            f"Achieved occupancy is {achieved:.1f}%, below the "
                            f"{LOW_OCCUPANCY:.0f}% threshold. "
                            f"Registers per thread: {regs_per_thread}. "
                            "Detailed occupancy limiter data not available — "
                            "re-profile with `--ncu-set detailed` to identify "
                            "the dominant limiter."
                        ),
                        action=(
                            "Common occupancy limiters: high register count "
                            "(use __launch_bounds__), excessive shared memory, "
                            "or small block size."
                        ),
                        source=self.name(),
                        category=self.category(),
                        metrics=metrics_occ,
                    ))

        # M2: annotate all findings with top contributing source line
        for finding in findings:
            self._attach_source_evidence(finding, top_hotspot)

        return findings
