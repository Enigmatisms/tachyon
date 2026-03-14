"""Example: Write a custom analyzer and register it.

Demonstrates how to:
  1. Subclass the Analyzer base class
  2. Implement the 4 required abstract methods (name, category, required_metrics, analyze)
  3. Register the custom analyzer with AnalyzerRegistry
  4. Run it against a synthetic KernelReport

Usage:
    python examples/custom_analyzer.py

No GPU or CUDA installation required.
"""
from __future__ import annotations

from tachyon.analyzers.base import Analyzer, AnalyzerRegistry
from tachyon.models.finding import Finding, Severity
from tachyon.models.kernel import (
    DeviceInfo,
    KernelReport,
    LaunchParams,
    MetricValue,
)


# ---------------------------------------------------------------------------
# Step 1: Define a custom analyzer by subclassing Analyzer
# ---------------------------------------------------------------------------


class SharedMemEfficiencyAnalyzer(Analyzer):
    """Custom analyzer: detect inefficient shared memory usage.

    Checks:
      - High static shared memory relative to block size (> 128 bytes/thread)
      - Dynamic shared memory allocated but potentially unused
      - Total shared memory close to hardware limit (48 KB on most GPUs)

    This analyzer operates on LaunchParams only (no NCU metrics required),
    similar to LaunchConfigAnalyzer.
    """

    # -- Required method 1: unique short name
    def name(self) -> str:
        return "shared_mem_efficiency"

    # -- Required method 2: category tag for grouping
    def category(self) -> str:
        return "memory"

    # -- Required method 3: required NCU metric names
    def required_metrics(self) -> list[str]:
        # This analyzer works without NCU metrics (uses LaunchParams).
        # Return an empty list to indicate it always can run.
        return []

    # -- Override can_run if you need custom logic
    def can_run(self, report: KernelReport) -> bool:
        """Only run if shared memory is actually used."""
        lp = report.launch_params
        return lp.shared_mem_bytes > 0 or lp.static_shared_mem_bytes > 0

    # -- Required method 4: the actual analysis logic
    def analyze(self, report: KernelReport) -> list[Finding]:
        """Produce Findings based on shared memory usage patterns."""
        lp = report.launch_params
        block_size = lp.block[0] * lp.block[1] * lp.block[2]
        total_shared = lp.shared_mem_bytes + lp.static_shared_mem_bytes

        findings: list[Finding] = []

        # Check 1: shared memory per thread
        if block_size > 0:
            bytes_per_thread = total_shared / block_size
            if bytes_per_thread > 128:
                findings.append(Finding(
                    severity=Severity.WARNING,
                    title=f"High shared memory per thread ({bytes_per_thread:.0f} B/thread)",
                    detail=(
                        f"The kernel uses {total_shared} bytes of shared memory "
                        f"across {block_size} threads ({bytes_per_thread:.0f} B/thread). "
                        f"High shared memory usage per thread can limit occupancy "
                        f"by reducing the number of blocks that can run per SM."
                    ),
                    action=(
                        "Consider reducing shared memory usage by: (1) using smaller "
                        "data types (e.g., float16 instead of float32), (2) processing "
                        "data in smaller tiles, (3) using registers for frequently "
                        "accessed values."
                    ),
                    source=self.name(),
                    category=self.category(),
                    metrics={
                        "total_shared_mem_bytes": float(total_shared),
                        "bytes_per_thread": bytes_per_thread,
                    },
                ))

        # Check 2: close to hardware limit (48 KB = 49152 bytes)
        hw_limit = 49152  # 48 KB, typical for compute capability 7.x+
        if total_shared > hw_limit * 0.9:
            findings.append(Finding(
                severity=Severity.WARNING,
                title=f"Shared memory near hardware limit ({total_shared}/{hw_limit} bytes)",
                detail=(
                    f"Total shared memory ({total_shared} bytes) is "
                    f"{total_shared / hw_limit * 100:.0f}% of the typical 48 KB limit. "
                    f"This severely limits the number of blocks per SM, reducing "
                    f"occupancy."
                ),
                action=(
                    "Consider using opt-in extended shared memory (up to 164 KB "
                    "on Hopper) via cudaFuncSetAttribute, or reduce shared memory "
                    "usage by tiling the computation."
                ),
                source=self.name(),
                category=self.category(),
                metrics={
                    "total_shared_mem_bytes": float(total_shared),
                    "hw_limit_bytes": float(hw_limit),
                },
            ))

        # Check 3: healthy usage
        if not findings:
            findings.append(Finding(
                severity=Severity.INFO,
                title=f"Shared memory usage is reasonable ({total_shared} bytes)",
                detail=(
                    f"Total shared memory: {total_shared} bytes "
                    f"({total_shared / hw_limit * 100:.0f}% of 48 KB limit). "
                    f"This should not be the occupancy bottleneck."
                ),
                action="Shared memory is not the bottleneck. Look elsewhere.",
                source=self.name(),
                category=self.category(),
            ))

        # Optionally attach source evidence (M2 feature)
        top_hotspot = self._hotspots[0] if self._hotspots else None
        for finding in findings:
            self._attach_source_evidence(finding, top_hotspot)

        return findings


# ---------------------------------------------------------------------------
# Step 2: Build a synthetic report to test our analyzer
# ---------------------------------------------------------------------------


def _make_report(shared_bytes: int, static_shared: int = 0) -> KernelReport:
    """Create a synthetic report with configurable shared memory."""
    return KernelReport(
        kernel_name="_Z12shared_demoPfS_",
        demangled_name="shared_demo(float*, float*)",
        launch_params=LaunchParams(
            grid=(512, 1, 1),
            block=(256, 1, 1),
            shared_mem_bytes=shared_bytes,
            registers_per_thread=32,
            static_shared_mem_bytes=static_shared,
        ),
        device_info=DeviceInfo(
            name="NVIDIA A100-SXM4-80GB",
            compute_capability=(8, 0),
            sm_count=108,
            max_clock_mhz=1410,
            memory_bus_width=5120,
            peak_memory_bandwidth_gbps=2039.0,
        ),
        metrics={},
    )


def main() -> None:
    print("Custom Analyzer Example: SharedMemEfficiencyAnalyzer")
    print("=" * 60)
    print()

    # --- Standalone usage ---
    analyzer = SharedMemEfficiencyAnalyzer()
    print(f"Analyzer name:     {analyzer.name()}")
    print(f"Analyzer category: {analyzer.category()}")
    print(f"Required metrics:  {analyzer.required_metrics()}")
    print()

    # Test case 1: healthy usage
    report_ok = _make_report(shared_bytes=8192, static_shared=4096)
    print(f"--- Case 1: Healthy ({report_ok.launch_params.shared_mem_bytes} + "
          f"{report_ok.launch_params.static_shared_mem_bytes} bytes shared) ---")
    findings = analyzer.analyze(report_ok)
    for f in findings:
        print(f"  [{f.severity.value.upper()}] {f.title}")
    print()

    # Test case 2: high per-thread usage
    report_high = _make_report(shared_bytes=40000, static_shared=2000)
    print(f"--- Case 2: High per-thread ({report_high.launch_params.shared_mem_bytes} + "
          f"{report_high.launch_params.static_shared_mem_bytes} bytes, "
          f"block=256) ---")
    findings = analyzer.analyze(report_high)
    for f in findings:
        print(f"  [{f.severity.value.upper()}] {f.title}")
        print(f"    Detail: {f.detail[:100]}...")
    print()

    # Test case 3: near hardware limit
    report_limit = _make_report(shared_bytes=46000, static_shared=2000)
    print(f"--- Case 3: Near HW limit ({report_limit.launch_params.shared_mem_bytes} + "
          f"{report_limit.launch_params.static_shared_mem_bytes} = "
          f"{46000+2000} bytes) ---")
    findings = analyzer.analyze(report_limit)
    for f in findings:
        print(f"  [{f.severity.value.upper()}] {f.title}")
    print()

    # --- Register with AnalyzerRegistry alongside built-in analyzers ---
    print("=" * 60)
    print("Registering with AnalyzerRegistry:")
    print("=" * 60)

    registry = AnalyzerRegistry()
    registry.auto_register()           # Add all 7 built-in analyzers
    registry.register(analyzer)        # Add our custom analyzer

    print(f"  Total analyzers: {len(registry.all_analyzers())}")
    print(f"  Names: {[a.name() for a in registry.all_analyzers()]}")
    print()

    # Run the full pipeline (our analyzer participates alongside built-ins)
    findings = registry.run_all(report_high)
    custom_findings = [f for f in findings if f.source == "shared_mem_efficiency"]
    print(f"  Total findings from pipeline: {len(findings)}")
    print(f"  Findings from our custom analyzer: {len(custom_findings)}")
    for f in custom_findings:
        print(f"    [{f.severity.value.upper()}] {f.title}")

    print()
    print("Done. Custom analyzer integrated successfully.")


if __name__ == "__main__":
    main()
