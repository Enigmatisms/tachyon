"""Roofline Analyzer — classifies kernels as compute/memory/latency/balanced-bound.

Threshold constants mirror NVIDIA's SpeedOfLight.py rule:
  - HIGH  = 80 %  (throughput is near peak → healthy)
  - LOW   = 60 %  (throughput is worryingly low)
  - IMBALANCE = 20 pp  (gap between SM and DRAM throughput)
"""
from __future__ import annotations

from tachyon.analyzers.base import Analyzer
from tachyon.models.finding import Finding, Severity
from tachyon.models.kernel import KernelReport

# Metric names
_SM_THROUGHPUT = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
_DRAM_THROUGHPUT = "dram__throughput.avg.pct_of_peak_sustained_elapsed"

# Thresholds (percent)
HIGH = 80.0
LOW = 60.0
IMBALANCE = 20.0  # percentage-points


class RooflineAnalyzer(Analyzer):
    """Classify kernel bottleneck via roofline-model heuristics.

    M2 enhancement: annotates findings with top contributing source line.
    """

    def name(self) -> str:
        return "roofline"

    def category(self) -> str:
        return "compute"

    def required_metrics(self) -> list[str]:
        return [_SM_THROUGHPUT, _DRAM_THROUGHPUT]

    def analyze(self, report: KernelReport) -> list[Finding]:
        sm_pct = report.metric_value(_SM_THROUGHPUT)
        mem_pct = report.metric_value(_DRAM_THROUGHPUT)

        # Defensive: required_metrics guarantees presence, but guard anyway.
        if sm_pct is None or mem_pct is None:
            return []

        findings: list[Finding] = []
        metrics = {_SM_THROUGHPUT: sm_pct, _DRAM_THROUGHPUT: mem_pct}

        # M2: get top contributing source line for annotation
        top_hotspot = self._hotspots[0] if self._hotspots else None

        # --- Classification ---
        if sm_pct > mem_pct + IMBALANCE:
            classification = "compute-bound"
            findings.append(
                Finding(
                    severity=Severity.INFO,
                    title="Kernel is compute-bound",
                    detail=(
                        f"SM throughput ({sm_pct:.1f}%) exceeds DRAM throughput "
                        f"({mem_pct:.1f}%) by more than {IMBALANCE:.0f} pp."
                    ),
                    action=(
                        "Focus on reducing arithmetic complexity, improving "
                        "instruction-level parallelism, or using faster math "
                        "intrinsics (e.g. __fmaf_rn, tensor cores)."
                    ),
                    source=self.name(),
                    category=self.category(),
                    metrics=metrics,
                )
            )
        elif mem_pct > sm_pct + IMBALANCE:
            classification = "memory-bound"
            findings.append(
                Finding(
                    severity=Severity.INFO,
                    title="Kernel is memory-bound",
                    detail=(
                        f"DRAM throughput ({mem_pct:.1f}%) exceeds SM throughput "
                        f"({sm_pct:.1f}%) by more than {IMBALANCE:.0f} pp."
                    ),
                    action=(
                        "Focus on improving data reuse (tiling, shared memory), "
                        "coalescing global accesses, and reducing memory traffic."
                    ),
                    source=self.name(),
                    category="memory",
                    metrics=metrics,
                )
            )
        elif sm_pct < LOW and mem_pct < LOW:
            classification = "latency-bound"
            findings.append(
                Finding(
                    severity=Severity.WARNING,
                    title="Kernel is latency-bound",
                    detail=(
                        f"Both SM ({sm_pct:.1f}%) and DRAM ({mem_pct:.1f}%) "
                        f"throughput are below {LOW:.0f}%. The kernel is not "
                        "saturating either resource — likely stalling on "
                        "dependencies or synchronization."
                    ),
                    action=(
                        "Increase occupancy (more warps), overlap computation "
                        "with memory access, or reduce synchronization barriers."
                    ),
                    source=self.name(),
                    category="latency",
                    metrics=metrics,
                )
            )
        else:
            classification = "balanced"
            findings.append(
                Finding(
                    severity=Severity.INFO,
                    title="Kernel is balanced (compute ≈ memory)",
                    detail=(
                        f"SM throughput ({sm_pct:.1f}%) and DRAM throughput "
                        f"({mem_pct:.1f}%) are within {IMBALANCE:.0f} pp of "
                        "each other. The kernel utilises both subsystems."
                    ),
                    action=(
                        "Both pipelines are busy — look for algorithmic "
                        "improvements or consider whether the workload can be "
                        "reduced."
                    ),
                    source=self.name(),
                    category=self.category(),
                    metrics=metrics,
                )
            )

        # --- Extra: low-utilization warning ---
        if sm_pct < LOW and mem_pct < LOW and classification != "latency-bound":
            findings.append(
                Finding(
                    severity=Severity.WARNING,
                    title="Low utilization on both SM and DRAM",
                    detail=(
                        f"SM throughput ({sm_pct:.1f}%) and DRAM throughput "
                        f"({mem_pct:.1f}%) are both below {LOW:.0f}%."
                    ),
                    action=(
                        "The kernel is underutilising the GPU. Investigate "
                        "occupancy limiters, warp stalls, and launch configuration."
                    ),
                    source=self.name(),
                    category="latency",
                    metrics=metrics,
                )
            )

        # M2: annotate all findings with top contributing source line
        for finding in findings:
            self._attach_source_evidence(finding, top_hotspot)

        return findings
