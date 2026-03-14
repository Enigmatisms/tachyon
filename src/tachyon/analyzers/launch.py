"""Launch Configuration Analyzer — detect sub-optimal launch parameters.

Examines the CUDA launch configuration (grid size, block size) against
GPU hardware properties to identify common misconfigurations.

Key checks:
  - Block size not warp-aligned (not multiple of 32)
  - Small block size (< 128 threads)
  - Block size exceeding GPU limit (> 1024 threads)
  - Grid size smaller than SM count (GPU underutilized)
  - Grid size not multiple of SM count (tail effect)
  - All-clear healthy status
"""
from __future__ import annotations

from tachyon.analyzers.base import Analyzer
from tachyon.models.finding import Finding, Severity
from tachyon.models.kernel import KernelReport

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
WARP_SIZE = 32
MIN_BLOCK_SIZE = 128
MAX_BLOCK_SIZE = 1024


class LaunchConfigAnalyzer(Analyzer):
    """Analyze CUDA launch configuration for common issues.

    This analyzer does not require any NCU metrics — it operates solely
    on LaunchParams and DeviceInfo from the KernelReport.

    M2 enhancement: annotates findings with top contributing source line.
    """

    def name(self) -> str:
        return "launch_config"

    def category(self) -> str:
        return "launch"

    def required_metrics(self) -> list[str]:
        return []

    def can_run(self, report: KernelReport) -> bool:
        """Always runnable — uses LaunchParams, not metrics."""
        return True

    def analyze(self, report: KernelReport) -> list[Finding]:
        lp = report.launch_params
        di = report.device_info

        block_size = lp.block[0] * lp.block[1] * lp.block[2]
        grid_size = lp.grid[0] * lp.grid[1] * lp.grid[2]

        findings: list[Finding] = []
        metrics = {
            "block_size": float(block_size),
            "grid_size": float(grid_size),
            "sm_count": float(di.sm_count),
        }

        # M2: get top contributing source line for annotation
        top_hotspot = self._hotspots[0] if self._hotspots else None

        has_issue = False

        # --- Block size not warp-aligned ---
        if block_size % WARP_SIZE != 0:
            has_issue = True
            findings.append(
                Finding(
                    severity=Severity.WARNING,
                    title=f"Block size not warp-aligned ({block_size} threads)",
                    detail=(
                        f"Block size is {block_size} threads "
                        f"(block dims: {lp.block[0]}x{lp.block[1]}x{lp.block[2]}), "
                        f"which is not a multiple of the warp size ({WARP_SIZE}). "
                        "The last partial warp wastes execution slots."
                    ),
                    action=(
                        "Round block size up to the nearest multiple of 32. "
                        "Common efficient sizes: 128, 256, 512."
                    ),
                    source=self.name(),
                    category=self.category(),
                    metrics=metrics,
                )
            )

        # --- Small block size ---
        if block_size < MIN_BLOCK_SIZE:
            has_issue = True
            findings.append(
                Finding(
                    severity=Severity.WARNING,
                    title=f"Small block size may limit occupancy ({block_size} threads)",
                    detail=(
                        f"Block size is {block_size} threads, below the "
                        f"recommended minimum of {MIN_BLOCK_SIZE}. Small blocks "
                        "reduce the number of warps available for latency hiding "
                        "and may prevent the SM from reaching full occupancy."
                    ),
                    action=(
                        "Increase block size to at least 128 threads (4 warps). "
                        "256 threads (8 warps) is often a good default."
                    ),
                    source=self.name(),
                    category=self.category(),
                    metrics=metrics,
                )
            )

        # --- Block size exceeds GPU limit ---
        if block_size > MAX_BLOCK_SIZE:
            has_issue = True
            findings.append(
                Finding(
                    severity=Severity.WARNING,
                    title=f"Block size exceeds GPU limit ({block_size} > {MAX_BLOCK_SIZE})",
                    detail=(
                        f"Block size is {block_size} threads, which exceeds the "
                        f"maximum of {MAX_BLOCK_SIZE} threads per block. "
                        "This launch should fail at the CUDA driver level."
                    ),
                    action=(
                        "Reduce block size to at most 1024 threads. "
                        "Split work across more blocks in the grid."
                    ),
                    source=self.name(),
                    category=self.category(),
                    metrics=metrics,
                )
            )

        # --- Grid size smaller than SM count ---
        if grid_size < di.sm_count:
            has_issue = True
            findings.append(
                Finding(
                    severity=Severity.WARNING,
                    title=f"Grid size smaller than SM count ({grid_size} < {di.sm_count} SMs)",
                    detail=(
                        f"Grid has only {grid_size} blocks but the GPU has "
                        f"{di.sm_count} SMs. At least {di.sm_count - grid_size} SMs "
                        "will be idle, leaving GPU resources underutilized."
                    ),
                    action=(
                        "Increase parallelism: launch more blocks, or use "
                        "persistent-kernel patterns to keep all SMs busy."
                    ),
                    source=self.name(),
                    category=self.category(),
                    metrics=metrics,
                )
            )

        # --- Grid size not multiple of SM count (tail effect) ---
        if grid_size > 0 and grid_size % di.sm_count != 0:
            has_issue = True
            findings.append(
                Finding(
                    severity=Severity.INFO,
                    title=(
                        f"Grid size not multiple of SM count "
                        f"({grid_size} blocks / {di.sm_count} SMs)"
                    ),
                    detail=(
                        f"Grid has {grid_size} blocks across {di.sm_count} SMs. "
                        f"The remainder ({grid_size % di.sm_count} blocks) forms a "
                        "'tail wave' where most SMs are idle, reducing efficiency "
                        "for the final scheduling wave."
                    ),
                    action=(
                        "If possible, pad the grid to a multiple of the SM count "
                        f"({di.sm_count}), or use work-stealing / persistent-thread "
                        "patterns to mitigate tail-effect overhead."
                    ),
                    source=self.name(),
                    category=self.category(),
                    metrics=metrics,
                )
            )

        # --- All checks pass ---
        if not has_issue:
            findings.append(
                Finding(
                    severity=Severity.INFO,
                    title="Launch configuration looks reasonable",
                    detail=(
                        f"Block size: {block_size} threads "
                        f"({lp.block[0]}x{lp.block[1]}x{lp.block[2]}), "
                        f"grid size: {grid_size} blocks "
                        f"({lp.grid[0]}x{lp.grid[1]}x{lp.grid[2]}), "
                        f"GPU SMs: {di.sm_count}. "
                        "No obvious launch configuration issues detected."
                    ),
                    action=(
                        "Launch configuration is not the bottleneck. Focus "
                        "optimization efforts on other areas."
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
