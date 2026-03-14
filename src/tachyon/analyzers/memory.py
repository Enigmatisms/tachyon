"""Memory Analyzer — coalescing efficiency, bank conflicts, L2 hit rate.

Checks three independent memory-subsystem health indicators and emits
findings when thresholds are violated.
"""
from __future__ import annotations

from tachyon.analyzers.base import Analyzer
from tachyon.models.finding import Finding, Severity
from tachyon.models.kernel import KernelReport

# ---------------------------------------------------------------------------
# Metric names
# ---------------------------------------------------------------------------

# Global load coalescing
_LD_SECTORS = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
_LD_REQUESTS = "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"

# Shared-memory bank conflicts
_BANK_CONFLICTS = "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"

# L2 cache hit / miss
_L2_HIT = "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum"
_L2_MISS = "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum"

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Coalescing efficiency (ideal = 100 %)
_COALESCING_WARN = 75.0   # below → WARNING
_COALESCING_CRIT = 50.0   # below → CRITICAL

# Bank-conflict count
_BANK_CONFLICT_WARN = 100  # above → WARNING

# L2 hit-rate (ideal = 100 %)
_L2_HIT_WARN = 50.0  # below → WARNING


class MemoryAnalyzer(Analyzer):
    """Detect common memory-access inefficiencies.

    M2 enhancement: locates uncoalesced access / bank conflict to source lines.
    """

    def name(self) -> str:
        return "memory"

    def category(self) -> str:
        return "memory"

    def required_metrics(self) -> list[str]:
        # We only require the coalescing pair as the absolute minimum —
        # bank-conflict and L2 checks are optional (guarded internally).
        return [_LD_SECTORS, _LD_REQUESTS]

    def analyze(self, report: KernelReport) -> list[Finding]:
        findings: list[Finding] = []

        self._check_coalescing(report, findings)
        self._check_bank_conflicts(report, findings)
        self._check_l2_cache(report, findings)

        # M2: annotate findings with source evidence
        hotspot = self._find_hotspot_for_stall("long_scoreboard")
        for finding in findings:
            self._attach_source_evidence(finding, hotspot)

        return findings

    # ---- individual checks ------------------------------------------------

    def _check_coalescing(
        self, report: KernelReport, findings: list[Finding]
    ) -> None:
        sectors = report.metric_value(_LD_SECTORS)
        requests = report.metric_value(_LD_REQUESTS)
        if sectors is None or requests is None or requests == 0:
            return

        # Ideal: each warp request produces exactly 1 cache-line sector.
        # Perfectly coalesced: sectors == requests → efficiency = 100%.
        # Uncoalesced: sectors > requests (multiple cache lines per request).
        efficiency = (requests / sectors) * 100.0

        # Clamp to 100% (should not exceed, but be safe).
        efficiency = min(efficiency, 100.0)

        metrics = {
            _LD_SECTORS: sectors,
            _LD_REQUESTS: requests,
            "coalescing_efficiency_pct": efficiency,
        }

        if efficiency < _COALESCING_CRIT:
            findings.append(
                Finding(
                    severity=Severity.CRITICAL,
                    title=f"Very poor global-load coalescing ({efficiency:.1f}%)",
                    detail=(
                        f"Global loads generated {sectors:.0f} sectors for "
                        f"{requests:.0f} requests (ideal ratio = 1:1). "
                        f"Coalescing efficiency is {efficiency:.1f}%, well below "
                        f"the {_COALESCING_CRIT:.0f}% critical threshold."
                    ),
                    action=(
                        "Ensure threads in a warp access contiguous, aligned "
                        "addresses. Consider using vectorized loads (float4) or "
                        "restructuring data layout to SoA."
                    ),
                    source=self.name(),
                    category=self.category(),
                    metrics=metrics,
                )
            )
        elif efficiency < _COALESCING_WARN:
            findings.append(
                Finding(
                    severity=Severity.WARNING,
                    title=f"Sub-optimal global-load coalescing ({efficiency:.1f}%)",
                    detail=(
                        f"Global loads generated {sectors:.0f} sectors for "
                        f"{requests:.0f} requests. Coalescing efficiency "
                        f"({efficiency:.1f}%) is below {_COALESCING_WARN:.0f}%."
                    ),
                    action=(
                        "Review global memory access patterns. Prefer "
                        "structure-of-arrays (SoA) over array-of-structures (AoS) "
                        "and align base addresses to 128-byte boundaries."
                    ),
                    source=self.name(),
                    category=self.category(),
                    metrics=metrics,
                )
            )

    def _check_bank_conflicts(
        self, report: KernelReport, findings: list[Finding]
    ) -> None:
        conflicts = report.metric_value(_BANK_CONFLICTS)
        if conflicts is None:
            return

        if conflicts > _BANK_CONFLICT_WARN:
            findings.append(
                Finding(
                    severity=Severity.WARNING,
                    title=f"Shared-memory bank conflicts detected ({conflicts:.0f})",
                    detail=(
                        f"There were {conflicts:.0f} shared-memory bank conflicts "
                        f"on load operations (threshold: {_BANK_CONFLICT_WARN})."
                    ),
                    action=(
                        "Pad shared-memory arrays (e.g. __shared__ float s[32][33]) "
                        "to avoid stride-induced conflicts, or reorganise access "
                        "patterns so that threads in a warp hit distinct banks."
                    ),
                    source=self.name(),
                    category=self.category(),
                    metrics={_BANK_CONFLICTS: conflicts},
                )
            )

    def _check_l2_cache(
        self, report: KernelReport, findings: list[Finding]
    ) -> None:
        hit = report.metric_value(_L2_HIT)
        miss = report.metric_value(_L2_MISS)
        if hit is None or miss is None:
            return

        total = hit + miss
        if total == 0:
            return

        hit_rate = (hit / total) * 100.0
        metrics = {
            _L2_HIT: hit,
            _L2_MISS: miss,
            "l2_hit_rate_pct": hit_rate,
        }

        if hit_rate < _L2_HIT_WARN:
            findings.append(
                Finding(
                    severity=Severity.WARNING,
                    title=f"Low L2 cache hit rate ({hit_rate:.1f}%)",
                    detail=(
                        f"L2 read hit rate is {hit_rate:.1f}% "
                        f"({hit:.0f} hits / {total:.0f} total sectors). "
                        f"Below {_L2_HIT_WARN:.0f}% threshold."
                    ),
                    action=(
                        "Improve spatial/temporal locality: tile data to fit "
                        "in L2, reorder iterations, or use CUDA L2 persistence "
                        "hints (cudaAccessPolicyWindow) on Ampere+."
                    ),
                    source=self.name(),
                    category=self.category(),
                    metrics=metrics,
                )
            )
