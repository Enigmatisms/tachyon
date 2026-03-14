"""Example: Compare two kernel reports using ProfileDiffer.

Demonstrates how to:
  1. Create two KernelReport objects with different metric values
  2. Run ProfileDiffer to compute per-metric deltas
  3. Inspect significant changes and regressions
  4. Print a summary report

Usage:
    python examples/diff_reports.py

No GPU or .ncu-rep files required -- everything runs in-memory.
"""
from __future__ import annotations

from tachyon.diff.differ import ProfileDiffer
from tachyon.models.kernel import (
    DeviceInfo,
    KernelReport,
    LaunchParams,
    MetricValue,
)


def _mv(name: str, value: float, unit: str = "") -> MetricValue:
    """Shorthand for MetricValue creation."""
    return MetricValue(name=name, value=value, unit=unit)


def _make_device() -> DeviceInfo:
    return DeviceInfo(
        name="NVIDIA A100-SXM4-80GB",
        compute_capability=(8, 0),
        sm_count=108,
        max_clock_mhz=1410,
        memory_bus_width=5120,
        peak_memory_bandwidth_gbps=2039.0,
    )


def _make_launch() -> LaunchParams:
    return LaunchParams(
        grid=(4096, 1, 1),
        block=(256, 1, 1),
        shared_mem_bytes=0,
        registers_per_thread=32,
    )


def create_before_reports() -> list[KernelReport]:
    """Simulate profiling results BEFORE optimization."""
    device = _make_device()
    launch = _make_launch()

    matmul = KernelReport(
        kernel_name="_Z6matmulPfS_S_i",
        demangled_name="matmul(float*, float*, float*, int)",
        launch_params=launch,
        device_info=device,
        metrics={
            "sm__throughput.avg.pct_of_peak_sustained_elapsed":
                _mv("sm__throughput.avg.pct_of_peak_sustained_elapsed", 45.0, "%"),
            "dram__throughput.avg.pct_of_peak_sustained_elapsed":
                _mv("dram__throughput.avg.pct_of_peak_sustained_elapsed", 70.0, "%"),
            "gpu__time_duration.sum":
                _mv("gpu__time_duration.sum", 2500.0, "us"),
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum":
                _mv("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", 50000.0, "sector"),
        },
    )

    reduce = KernelReport(
        kernel_name="_Z6reducePfS_i",
        demangled_name="reduce(float*, float*, int)",
        launch_params=launch,
        device_info=device,
        metrics={
            "sm__throughput.avg.pct_of_peak_sustained_elapsed":
                _mv("sm__throughput.avg.pct_of_peak_sustained_elapsed", 30.0, "%"),
            "dram__throughput.avg.pct_of_peak_sustained_elapsed":
                _mv("dram__throughput.avg.pct_of_peak_sustained_elapsed", 55.0, "%"),
            "gpu__time_duration.sum":
                _mv("gpu__time_duration.sum", 800.0, "us"),
        },
    )

    return [matmul, reduce]


def create_after_reports() -> list[KernelReport]:
    """Simulate profiling results AFTER optimization.

    Changes:
      - matmul: improved SM throughput and coalescing, but duration slightly up
      - reduce: significant improvement in all metrics
    """
    device = _make_device()
    launch = _make_launch()

    matmul = KernelReport(
        kernel_name="_Z6matmulPfS_S_i",
        demangled_name="matmul(float*, float*, float*, int)",
        launch_params=launch,
        device_info=device,
        metrics={
            "sm__throughput.avg.pct_of_peak_sustained_elapsed":
                _mv("sm__throughput.avg.pct_of_peak_sustained_elapsed", 62.0, "%"),
            "dram__throughput.avg.pct_of_peak_sustained_elapsed":
                _mv("dram__throughput.avg.pct_of_peak_sustained_elapsed", 65.0, "%"),
            "gpu__time_duration.sum":
                _mv("gpu__time_duration.sum", 2600.0, "us"),  # slightly worse
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum":
                _mv("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", 35000.0, "sector"),
            # New metric only in "after"
            "sm__warps_active.avg.pct_of_peak_sustained_active":
                _mv("sm__warps_active.avg.pct_of_peak_sustained_active", 78.0, "%"),
        },
    )

    reduce = KernelReport(
        kernel_name="_Z6reducePfS_i",
        demangled_name="reduce(float*, float*, int)",
        launch_params=launch,
        device_info=device,
        metrics={
            "sm__throughput.avg.pct_of_peak_sustained_elapsed":
                _mv("sm__throughput.avg.pct_of_peak_sustained_elapsed", 50.0, "%"),
            "dram__throughput.avg.pct_of_peak_sustained_elapsed":
                _mv("dram__throughput.avg.pct_of_peak_sustained_elapsed", 45.0, "%"),
            "gpu__time_duration.sum":
                _mv("gpu__time_duration.sum", 450.0, "us"),  # much better
        },
    )

    return [matmul, reduce]


def main() -> None:
    print("ProfileDiffer Example")
    print("=" * 60)
    print()

    # --- Step 1: Create before/after reports ---
    before = create_before_reports()
    after = create_after_reports()

    print(f"Before: {len(before)} kernels -> "
          f"{[k.demangled_name for k in before]}")
    print(f"After:  {len(after)} kernels -> "
          f"{[k.demangled_name for k in after]}")
    print()

    # --- Step 2: Run diff ---
    differ = ProfileDiffer()
    diffs = differ.diff(before, after)

    print(f"Common kernels matched: {len(diffs)}")
    print()

    # --- Step 3: Inspect each kernel diff ---
    for d in diffs:
        print(f"{'=' * 60}")
        print(f"Kernel: {d.demangled_name}")
        print(f"  All metric deltas: {len(d.metric_deltas)}")
        print(f"  Significant changes (>5%): {len(d.significant_changes)}")
        print(f"  Regressions: {len(d.regressions)}")

        if d.only_in_before:
            print(f"  Metrics only in BEFORE: {d.only_in_before}")
        if d.only_in_after:
            print(f"  Metrics only in AFTER:  {d.only_in_after}")

        print()

        # Show all deltas
        for m in d.metric_deltas:
            arrow = "\u2191" if m.delta > 0 else "\u2193"
            marker = ""
            if abs(m.delta_pct) > 5.0:
                marker = " ***"
            if m in d.regressions:
                marker = " [REGRESSION]"
            print(f"    {m.name}:")
            print(f"      {m.before:.2f} -> {m.after:.2f} "
                  f"({arrow}{abs(m.delta_pct):.1f}%){marker}")

        print()

    # --- Step 4: Print built-in summary ---
    print("=" * 60)
    print("Built-in Summary:")
    print(differ.summary(diffs))
    print()

    # --- Step 5: Programmatic regression check ---
    total_regressions = sum(len(d.regressions) for d in diffs)
    if total_regressions > 0:
        print(f"\n[WARNING] {total_regressions} regression(s) detected!")
        for d in diffs:
            for r in d.regressions:
                print(f"  - {d.kernel_name}: {r.name} "
                      f"({r.before:.2f} -> {r.after:.2f}, {r.delta_pct:+.1f}%)")
    else:
        print("\n[OK] No regressions detected.")


if __name__ == "__main__":
    main()
