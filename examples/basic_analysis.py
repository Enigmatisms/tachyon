"""Example: Analyze an existing NCU report file.

Usage:
    python examples/basic_analysis.py [report.ncu-rep]

If no report file is provided, a synthetic in-memory KernelReport is created
to demonstrate the analyzer pipeline without requiring a real NCU installation.

This example shows how to:
  1. Create a KernelReport (or load one from an .ncu-rep file)
  2. Register and run all built-in analyzers
  3. Inspect the resulting Findings (severity, title, detail, action)
"""
from __future__ import annotations

import sys

from tachyon.analyzers.base import AnalyzerRegistry
from tachyon.models.finding import Severity
from tachyon.models.kernel import (
    DeviceInfo,
    KernelReport,
    LaunchParams,
    MetricValue,
)


def _make_sample_report() -> KernelReport:
    """Build a synthetic KernelReport that exercises multiple analyzers.

    This simulates a memory-bound kernel with poor coalescing, low L2 hit
    rate, high FP64 usage, and a sub-optimal block size -- plenty of issues
    for the rule engine to flag.
    """
    device = DeviceInfo(
        name="NVIDIA A100-SXM4-80GB",
        compute_capability=(8, 0),
        sm_count=108,
        max_clock_mhz=1410,
        memory_bus_width=5120,
        peak_memory_bandwidth_gbps=2039.0,
    )
    launch = LaunchParams(
        grid=(200, 1, 1),
        block=(96, 1, 1),          # Not warp-aligned (96 % 32 == 0, but < 128)
        shared_mem_bytes=4096,
        registers_per_thread=72,   # > 64 -> register pressure warning
    )

    def mv(name: str, value: float, unit: str = "") -> MetricValue:
        return MetricValue(name=name, value=value, unit=unit)

    metrics = {
        # Roofline
        "sm__throughput.avg.pct_of_peak_sustained_elapsed":
            mv("sm__throughput.avg.pct_of_peak_sustained_elapsed", 28.0, "%"),
        "dram__throughput.avg.pct_of_peak_sustained_elapsed":
            mv("dram__throughput.avg.pct_of_peak_sustained_elapsed", 75.0, "%"),
        # Memory coalescing
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum":
            mv("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", 40000.0, "sector"),
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum":
            mv("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", 1000.0, "request"),
        # L2 cache
        "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum":
            mv("lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum", 2000.0, "sector"),
        "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum":
            mv("lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum", 8000.0, "sector"),
        # Warp stalls
        "smsp__pcsamp_warps_issue_stalled_long_scoreboard.sum":
            mv("smsp__pcsamp_warps_issue_stalled_long_scoreboard.sum", 9000.0, ""),
        "smsp__pcsamp_warps_issue_stalled_barrier.sum":
            mv("smsp__pcsamp_warps_issue_stalled_barrier.sum", 1000.0, ""),
        # Occupancy
        "sm__warps_active.avg.pct_of_peak_sustained_active":
            mv("sm__warps_active.avg.pct_of_peak_sustained_active", 42.0, "%"),
        "launch__occupancy_limit_warps":
            mv("launch__occupancy_limit_warps", 64.0, "warps"),
        "launch__occupancy_limit_registers":
            mv("launch__occupancy_limit_registers", 24.0, "warps"),
        "launch__occupancy_limit_shared_mem":
            mv("launch__occupancy_limit_shared_mem", 48.0, "warps"),
        "launch__occupancy_limit_blocks":
            mv("launch__occupancy_limit_blocks", 32.0, "warps"),
        # Instruction mix
        "sm__sass_thread_inst_executed_op_fp16_pred_on.sum":
            mv("sm__sass_thread_inst_executed_op_fp16_pred_on.sum", 500.0, "inst"),
        "sm__sass_thread_inst_executed_op_fp32_pred_on.sum":
            mv("sm__sass_thread_inst_executed_op_fp32_pred_on.sum", 2000.0, "inst"),
        "sm__sass_thread_inst_executed_op_fp64_pred_on.sum":
            mv("sm__sass_thread_inst_executed_op_fp64_pred_on.sum", 5000.0, "inst"),
        "sm__inst_executed_pipe_tensor.sum":
            mv("sm__inst_executed_pipe_tensor.sum", 0.0, "inst"),
        "sm__inst_executed.sum":
            mv("sm__inst_executed.sum", 50000.0, "inst"),
    }

    return KernelReport(
        kernel_name="_Z12memory_boundPfS_i",
        demangled_name="memory_bound(float*, float*, int)",
        launch_params=launch,
        device_info=device,
        metrics=metrics,
    )


def _severity_color(severity: Severity) -> str:
    """ANSI color code for severity display."""
    return {
        Severity.CRITICAL: "\033[91m",  # red
        Severity.WARNING:  "\033[93m",  # yellow
        Severity.INFO:     "\033[94m",  # blue
    }.get(severity, "")


def main() -> None:
    # --- Step 1: Get a KernelReport ---
    if len(sys.argv) > 1:
        # Real .ncu-rep file
        from pathlib import Path
        from tachyon.reader.ncu_reader import NcuReportReader

        path = Path(sys.argv[1])
        reader = NcuReportReader()
        result = reader.load(path)
        if not result.success:
            print(f"Error loading report: {result.error.message}")
            sys.exit(1)
        reports = result.data
        print(f"Loaded {len(reports)} kernel(s) from {path.name}")
    else:
        # In-memory synthetic report
        reports = [_make_sample_report()]
        print("Using synthetic in-memory KernelReport (no .ncu-rep file provided)")

    # --- Step 2: Register analyzers ---
    registry = AnalyzerRegistry()
    registry.auto_register()
    print(f"Registered {len(registry.all_analyzers())} analyzers: "
          f"{[a.name() for a in registry.all_analyzers()]}\n")

    # --- Step 3: Run analysis on each kernel ---
    for i, report in enumerate(reports):
        print(f"{'=' * 60}")
        print(f"Kernel {i}: {report.demangled_name}")
        print(f"  Launch: grid={report.launch_params.grid} "
              f"block={report.launch_params.block} "
              f"regs={report.launch_params.registers_per_thread}")
        print(f"  Device: {report.device_info.name} ({report.device_info.sm_count} SMs)")
        print()

        findings = registry.run_all(report)

        if not findings:
            print("  No findings produced (metrics may be missing).")
            continue

        # --- Step 4: Print findings grouped by severity ---
        for finding in findings:
            color = _severity_color(finding.severity)
            reset = "\033[0m"
            print(f"  {color}[{finding.severity.value.upper():8s}]{reset} {finding.title}")
            print(f"             Source: {finding.source}")
            # Truncate detail for readability
            detail_preview = finding.detail[:120]
            if len(finding.detail) > 120:
                detail_preview += "..."
            print(f"             Detail: {detail_preview}")
            print(f"             Action: {finding.action[:100]}")
            if finding.source_location:
                loc = finding.source_location
                print(f"             Location: {loc.file}:{loc.line}")
            print()

        # Summary
        crit = sum(1 for f in findings if f.severity == Severity.CRITICAL)
        warn = sum(1 for f in findings if f.severity == Severity.WARNING)
        info = sum(1 for f in findings if f.severity == Severity.INFO)
        print(f"  Summary: {crit} critical, {warn} warnings, {info} info")
        print()


if __name__ == "__main__":
    main()
