"""Instruction Analyzer — analyze instruction mix and compute efficiency.

Examines the distribution of floating-point precision levels (FP16/FP32/FP64)
and Tensor Core utilization to identify optimization opportunities.

Key checks:
  - FP precision distribution reporting
  - High FP64 usage warning (> 30 % of total FP)
  - Low Tensor Core utilization (< 10 % of total instructions)
  - Missed Tensor Core opportunity (FP16 present but no TC usage)
"""
from __future__ import annotations

from tachyon.analyzers.base import Analyzer
from tachyon.models.finding import Finding, Severity
from tachyon.models.kernel import KernelReport

# ---------------------------------------------------------------------------
# Metric names
# ---------------------------------------------------------------------------
_FP16_INST = "sm__sass_thread_inst_executed_op_fp16_pred_on.sum"
_FP32_INST = "sm__sass_thread_inst_executed_op_fp32_pred_on.sum"
_FP64_INST = "sm__sass_thread_inst_executed_op_fp64_pred_on.sum"
_TENSOR_INST = "sm__inst_executed_pipe_tensor.sum"
_TOTAL_INST = "sm__inst_executed.sum"

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
FP64_WARN_RATIO = 30.0        # FP64 > 30 % of total FP → WARNING
TENSOR_LOW_RATIO = 10.0       # Tensor Core < 10 % of total → INFO


class InstructionAnalyzer(Analyzer):
    """Analyze instruction mix and compute precision efficiency.

    M2 enhancement: annotates findings with top contributing source line.
    """

    def name(self) -> str:
        return "instruction"

    def category(self) -> str:
        return "compute"

    def required_metrics(self) -> list[str]:
        return [
            _FP16_INST,
            _FP32_INST,
            _FP64_INST,
            _TENSOR_INST,
            _TOTAL_INST,
        ]

    def analyze(self, report: KernelReport) -> list[Finding]:
        fp16 = report.metric_value(_FP16_INST)
        fp32 = report.metric_value(_FP32_INST)
        fp64 = report.metric_value(_FP64_INST)
        tensor = report.metric_value(_TENSOR_INST)
        total = report.metric_value(_TOTAL_INST)

        # Defensive: required_metrics guarantees presence, but guard anyway.
        if any(v is None for v in (fp16, fp32, fp64, tensor, total)):
            return []

        # If no instruction metrics available (all zero), produce no findings.
        if fp16 == 0 and fp32 == 0 and fp64 == 0 and tensor == 0 and total == 0:
            return []

        findings: list[Finding] = []
        metrics = {
            _FP16_INST: fp16,
            _FP32_INST: fp32,
            _FP64_INST: fp64,
            _TENSOR_INST: tensor,
            _TOTAL_INST: total,
        }

        # M2: get top contributing source line for annotation
        top_hotspot = self._hotspots[0] if self._hotspots else None

        # --- FP precision distribution ---
        total_fp = fp16 + fp32 + fp64
        if total_fp > 0:
            fp16_pct = (fp16 / total_fp) * 100.0
            fp32_pct = (fp32 / total_fp) * 100.0
            fp64_pct = (fp64 / total_fp) * 100.0

            findings.append(
                Finding(
                    severity=Severity.INFO,
                    title="FP precision distribution",
                    detail=(
                        f"FP16: {fp16_pct:.1f}% ({fp16:.0f} inst), "
                        f"FP32: {fp32_pct:.1f}% ({fp32:.0f} inst), "
                        f"FP64: {fp64_pct:.1f}% ({fp64:.0f} inst). "
                        f"Total FP instructions: {total_fp:.0f}."
                    ),
                    action=(
                        "Review whether the precision level matches algorithmic "
                        "requirements. FP16/BF16 can deliver 2x throughput on "
                        "modern GPUs with acceptable accuracy for many workloads."
                    ),
                    source=self.name(),
                    category=self.category(),
                    metrics=metrics,
                )
            )

            # --- High FP64 usage ---
            if fp64_pct > FP64_WARN_RATIO:
                findings.append(
                    Finding(
                        severity=Severity.WARNING,
                        title=f"High FP64 usage ({fp64_pct:.1f}% of FP instructions)",
                        detail=(
                            f"FP64 instructions account for {fp64_pct:.1f}% of all "
                            f"floating-point operations ({fp64:.0f} out of {total_fp:.0f}). "
                            f"FP64 throughput is typically 1/32 or 1/64 of FP32 on "
                            "consumer GPUs and 1/2 on data-center GPUs."
                        ),
                        action=(
                            "Consider using FP32 or FP16 where full FP64 precision "
                            "is not required. Use __float2half() or explicit casts, "
                            "and verify numerical accuracy is acceptable."
                        ),
                        source=self.name(),
                        category=self.category(),
                        metrics=metrics,
                    )
                )

        # --- Tensor Core utilization ---
        if total > 0 and tensor > 0:
            tensor_pct = (tensor / total) * 100.0
            if tensor_pct < TENSOR_LOW_RATIO:
                findings.append(
                    Finding(
                        severity=Severity.INFO,
                        title=(
                            f"Low Tensor Core utilization "
                            f"({tensor_pct:.1f}% of total instructions)"
                        ),
                        detail=(
                            f"Tensor Core instructions account for only {tensor_pct:.1f}% "
                            f"of total instructions ({tensor:.0f} out of {total:.0f}). "
                            "Tensor Cores can accelerate matrix operations by 4-16x."
                        ),
                        action=(
                            "Ensure matrix dimensions are aligned to Tensor Core "
                            "tile sizes (typically multiples of 8 or 16). Use "
                            "WMMA/MMA PTX intrinsics or cuBLAS/cuDNN for automatic "
                            "Tensor Core dispatch."
                        ),
                        source=self.name(),
                        category=self.category(),
                        metrics=metrics,
                    )
                )

        # --- FP16 present but no Tensor Core usage ---
        if tensor == 0 and fp16 > 0:
            findings.append(
                Finding(
                    severity=Severity.INFO,
                    title="FP16 operations detected but no Tensor Core usage",
                    detail=(
                        f"The kernel executes {fp16:.0f} FP16 instructions but "
                        "does not use Tensor Cores. Tensor Cores can accelerate "
                        "FP16 matrix multiply-accumulate operations significantly."
                    ),
                    action=(
                        "Consider using WMMA/MMA intrinsics (nvcuda::wmma) or "
                        "higher-level libraries (cuBLAS, cuDNN) to leverage "
                        "Tensor Cores for FP16 matrix operations."
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
