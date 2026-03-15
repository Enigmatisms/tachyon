"""Roofline Analyzer — classifies kernels as compute/memory/latency/balanced-bound.

Threshold constants mirror NVIDIA's SpeedOfLight.py rule:
  - HIGH  = 80 %  (throughput is near peak → healthy)
  - LOW   = 60 %  (throughput is worryingly low)
  - IMBALANCE = 20 pp  (gap between SM and DRAM throughput)

Design: Graceful degradation — if only SM throughput is available (DRAM
missing), produces a partial roofline finding with what we have.  This is
far more useful than silently skipping the most fundamental analyzer.
"""
from __future__ import annotations

from tachyon.analyzers.base import Analyzer
from tachyon.models.finding import Finding, Severity
from tachyon.models.kernel import KernelReport

# Metric names
_SM_THROUGHPUT = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
_DRAM_THROUGHPUT = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
_DURATION = "gpu__time_duration.sum"

# Thresholds (percent)
HIGH = 80.0
LOW = 60.0
IMBALANCE = 20.0  # percentage-points


class RooflineAnalyzer(Analyzer):
    """Classify kernel bottleneck via roofline-model heuristics.

    Graceful degradation:
      - SM + DRAM present → full roofline classification
      - SM only (DRAM missing) → partial analysis with SM throughput only
      - Neither → cannot run (returns empty)
    """

    def name(self) -> str:
        return "roofline"

    def category(self) -> str:
        return "compute"

    def required_metrics(self) -> list[str]:
        # Only SM throughput is truly required; DRAM is consumed if available
        return [_SM_THROUGHPUT]

    def analyze(self, report: KernelReport) -> list[Finding]:
        sm_pct = report.metric_value(_SM_THROUGHPUT)
        mem_pct = report.metric_value(_DRAM_THROUGHPUT)
        duration = report.metric_value(_DURATION)

        if sm_pct is None:
            return []

        findings: list[Finding] = []
        metrics: dict[str, float] = {_SM_THROUGHPUT: sm_pct}
        if mem_pct is not None:
            metrics[_DRAM_THROUGHPUT] = mem_pct
        if duration is not None:
            metrics[_DURATION] = duration

        # M2: get top contributing source line for annotation
        top_hotspot = self._hotspots[0] if self._hotspots else None

        # --- Full roofline (both SM + DRAM available) ---
        if mem_pct is not None:
            findings.extend(self._classify_full(sm_pct, mem_pct, metrics))
        else:
            # --- Partial roofline (SM only, DRAM missing) ---
            findings.extend(self._classify_partial(sm_pct, duration, metrics))

        # M2: annotate all findings with top contributing source line
        for finding in findings:
            self._attach_source_evidence(finding, top_hotspot)

        return findings

    def _classify_full(
        self, sm_pct: float, mem_pct: float, metrics: dict[str, float]
    ) -> list[Finding]:
        """Full roofline with both SM and DRAM throughput."""
        if sm_pct > mem_pct + IMBALANCE:
            return [Finding(
                severity=Severity.WARNING,
                title=f"Kernel is COMPUTE-BOUND (SM={sm_pct:.1f}%, DRAM={mem_pct:.1f}%)",
                detail=(
                    f"SM throughput ({sm_pct:.1f}%) exceeds DRAM throughput "
                    f"({mem_pct:.1f}%) by {sm_pct - mem_pct:.0f} pp (threshold: {IMBALANCE:.0f} pp). "
                    f"Performance is limited by compute pipeline capacity."
                ),
                action=(
                    "Focus on reducing arithmetic complexity, improving "
                    "instruction-level parallelism, or using faster math "
                    "intrinsics (e.g. __fmaf_rn, tensor cores)."
                ),
                source=self.name(),
                category=self.category(),
                metrics=metrics,
            )]
        elif mem_pct > sm_pct + IMBALANCE:
            return [Finding(
                severity=Severity.WARNING,
                title=f"Kernel is MEMORY-BOUND (DRAM={mem_pct:.1f}%, SM={sm_pct:.1f}%)",
                detail=(
                    f"DRAM throughput ({mem_pct:.1f}%) exceeds SM throughput "
                    f"({sm_pct:.1f}%) by {mem_pct - sm_pct:.0f} pp (threshold: {IMBALANCE:.0f} pp). "
                    f"Performance is limited by memory bandwidth."
                ),
                action=(
                    "Focus on improving data reuse (tiling, shared memory), "
                    "coalescing global accesses, and reducing memory traffic."
                ),
                source=self.name(),
                category="memory",
                metrics=metrics,
            )]
        elif sm_pct < LOW and mem_pct < LOW:
            return [Finding(
                severity=Severity.CRITICAL,
                title=f"Kernel is LATENCY-BOUND (SM={sm_pct:.1f}%, DRAM={mem_pct:.1f}%)",
                detail=(
                    f"Both SM ({sm_pct:.1f}%) and DRAM ({mem_pct:.1f}%) "
                    f"throughput are below {LOW:.0f}%. The kernel is not "
                    "saturating either resource — likely stalling on "
                    "dependencies, synchronization, or low occupancy. "
                    "This is the worst scenario: the GPU is mostly idle."
                ),
                action=(
                    "Increase occupancy (more warps), overlap computation "
                    "with memory access, reduce synchronization barriers, "
                    "or increase block size."
                ),
                source=self.name(),
                category="latency",
                metrics=metrics,
            )]
        else:
            return [Finding(
                severity=Severity.WARNING,
                title=f"Kernel is BALANCED (SM={sm_pct:.1f}%, DRAM={mem_pct:.1f}%)",
                detail=(
                    f"SM throughput ({sm_pct:.1f}%) and DRAM throughput "
                    f"({mem_pct:.1f}%) are within {IMBALANCE:.0f} pp — "
                    "both pipelines are active. This is generally healthy."
                ),
                action=(
                    "Both pipelines are busy — look for algorithmic "
                    "improvements or consider whether the workload can be "
                    "reduced."
                ),
                source=self.name(),
                category=self.category(),
                metrics=metrics,
            )]

    def _classify_partial(
        self, sm_pct: float, duration: float | None, metrics: dict[str, float]
    ) -> list[Finding]:
        """Partial roofline — only SM throughput available (DRAM missing).

        Still produces actionable findings: low SM → likely latency-bound,
        high SM → likely compute-bound.  Tells user to profile with more
        metrics for full classification.
        """
        findings: list[Finding] = []

        if sm_pct < LOW:
            findings.append(Finding(
                severity=Severity.WARNING,
                title=f"Low SM throughput ({sm_pct:.1f}%) — likely LATENCY or MEMORY-BOUND",
                detail=(
                    f"SM throughput is only {sm_pct:.1f}%, well below {LOW:.0f}%. "
                    "The compute pipeline is far from saturated. "
                    "Without DRAM throughput data, the exact bottleneck cannot be "
                    "determined, but the kernel is clearly underutilizing the GPU."
                    + (f" Kernel duration: {duration / 1e6:.2f} ms." if duration else "")
                ),
                action=(
                    "Re-profile with `--ncu-set detailed` or `--ncu-set full` for "
                    "full roofline classification (DRAM throughput needed). "
                    "Common causes: low occupancy, warp stalls, or memory latency."
                ),
                source=self.name(),
                category="latency",
                metrics=metrics,
            ))
        elif sm_pct > HIGH:
            findings.append(Finding(
                severity=Severity.WARNING,
                title=f"High SM throughput ({sm_pct:.1f}%) — likely COMPUTE-BOUND",
                detail=(
                    f"SM throughput is {sm_pct:.1f}%, above {HIGH:.0f}%. "
                    "The compute pipeline is heavily utilized. "
                    "Without DRAM throughput, full roofline classification is unavailable."
                    + (f" Kernel duration: {duration / 1e6:.2f} ms." if duration else "")
                ),
                action=(
                    "Focus on instruction-level optimizations: faster math intrinsics, "
                    "Tensor Core utilization, FP16/BF16 mixed precision. "
                    "Re-profile with `--ncu-set detailed` for full roofline analysis."
                ),
                source=self.name(),
                category=self.category(),
                metrics=metrics,
            ))
        else:
            findings.append(Finding(
                severity=Severity.WARNING,
                title=f"SM throughput is moderate ({sm_pct:.1f}%) — incomplete roofline data",
                detail=(
                    f"SM throughput is {sm_pct:.1f}% (between {LOW:.0f}% and {HIGH:.0f}%). "
                    "DRAM throughput metric is missing — cannot determine if the kernel "
                    "is memory-bound, compute-bound, or balanced."
                    + (f" Kernel duration: {duration / 1e6:.2f} ms." if duration else "")
                ),
                action=(
                    "Re-profile with `--ncu-set detailed` or `--ncu-set full` to "
                    "collect DRAM throughput for full roofline classification."
                ),
                source=self.name(),
                category="general",
                metrics=metrics,
            ))

        return findings
