"""Shared test fixtures — synthetic KernelReport instances for unit tests.

No GPU or NCU installation required. All fixtures use realistic metric values
derived from actual NVIDIA GPU profiling scenarios.
"""
import pytest

from tachyon.models.kernel import (
    DeviceInfo,
    KernelReport,
    LaunchParams,
    MetricValue,
    RuleResult,
)

# ━━━━━━━━━━━━━━━━━━━━━━━ Device fixtures ━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def device_h100() -> DeviceInfo:
    """NVIDIA H100 80GB HBM3."""
    return DeviceInfo(
        name="NVIDIA H100 80GB HBM3",
        compute_capability=(9, 0),
        sm_count=132,
        max_clock_mhz=1980,
        memory_bus_width=5120,
        peak_memory_bandwidth_gbps=3352.0,
    )


@pytest.fixture
def device_a100() -> DeviceInfo:
    """NVIDIA A100 80GB."""
    return DeviceInfo(
        name="NVIDIA A100-SXM4-80GB",
        compute_capability=(8, 0),
        sm_count=108,
        max_clock_mhz=1410,
        memory_bus_width=5120,
        peak_memory_bandwidth_gbps=2039.0,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━ Launch config fixtures ━━━━━━━━━━━━━━━━━


@pytest.fixture
def launch_256x4096() -> LaunchParams:
    """Common launch config: 256 threads/block, 4096 blocks."""
    return LaunchParams(
        grid=(4096, 1, 1),
        block=(256, 1, 1),
        shared_mem_bytes=0,
        registers_per_thread=32,
    )


@pytest.fixture
def launch_128x1024_shared() -> LaunchParams:
    """Launch config with shared memory."""
    return LaunchParams(
        grid=(1024, 1, 1),
        block=(128, 1, 1),
        shared_mem_bytes=16384,
        registers_per_thread=48,
        static_shared_mem_bytes=4096,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━ Helper ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _mv(name: str, value: float, unit: str = "") -> MetricValue:
    """Shorthand for MetricValue creation."""
    return MetricValue(name=name, value=value, unit=unit)


# ━━━━━━━━━━━━━━━━━━━━━━━ KernelReport fixtures ━━━━━━━━━━━━━━━━━


@pytest.fixture
def report_compute_bound(launch_256x4096: LaunchParams, device_h100: DeviceInfo) -> KernelReport:
    """Kernel with high SM throughput, low DRAM -> compute-bound."""
    return KernelReport(
        kernel_name="compute_kernel",
        demangled_name="compute_kernel<float>",
        launch_params=launch_256x4096,
        device_info=device_h100,
        metrics={
            "sm__throughput.avg.pct_of_peak_sustained_elapsed":
                _mv("sm__throughput.avg.pct_of_peak_sustained_elapsed", 85.0, "%"),
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed":
                _mv("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", 30.0, "%"),
        },
    )


@pytest.fixture
def report_memory_bound(launch_256x4096: LaunchParams, device_h100: DeviceInfo) -> KernelReport:
    """Kernel with low SM, high DRAM -> memory-bound."""
    return KernelReport(
        kernel_name="memory_kernel",
        demangled_name="memory_kernel<float>",
        launch_params=launch_256x4096,
        device_info=device_h100,
        metrics={
            "sm__throughput.avg.pct_of_peak_sustained_elapsed":
                _mv("sm__throughput.avg.pct_of_peak_sustained_elapsed", 25.0, "%"),
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed":
                _mv("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", 78.0, "%"),
            # Memory analyzer metrics
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum":
                _mv("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", 50000.0, "sector"),
            "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum":
                _mv("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", 1000.0, "request"),
            # Bank conflicts
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum":
                _mv("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", 500.0, ""),
            # L2 cache
            "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum":
                _mv("lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum", 3000.0, "sector"),
            "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum":
                _mv("lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum", 7000.0, "sector"),
        },
    )


@pytest.fixture
def report_latency_bound(launch_256x4096: LaunchParams, device_h100: DeviceInfo) -> KernelReport:
    """Kernel with both SM and DRAM below 60% -> latency-bound."""
    return KernelReport(
        kernel_name="latency_kernel",
        demangled_name="latency_kernel<float>",
        launch_params=launch_256x4096,
        device_info=device_h100,
        metrics={
            "sm__throughput.avg.pct_of_peak_sustained_elapsed":
                _mv("sm__throughput.avg.pct_of_peak_sustained_elapsed", 22.0, "%"),
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed":
                _mv("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", 18.0, "%"),
            # Warp stall metrics
            "smsp__pcsamp_warps_issue_stalled_long_scoreboard.sum":
                _mv("smsp__pcsamp_warps_issue_stalled_long_scoreboard.sum", 8500.0, ""),
            "smsp__pcsamp_warps_issue_stalled_barrier.sum":
                _mv("smsp__pcsamp_warps_issue_stalled_barrier.sum", 3200.0, ""),
            "smsp__pcsamp_warps_issue_stalled_not_selected.sum":
                _mv("smsp__pcsamp_warps_issue_stalled_not_selected.sum", 200.0, ""),
            "smsp__pcsamp_warps_issue_stalled_short_scoreboard.sum":
                _mv("smsp__pcsamp_warps_issue_stalled_short_scoreboard.sum", 100.0, ""),
        },
    )


@pytest.fixture
def report_balanced(launch_256x4096: LaunchParams, device_h100: DeviceInfo) -> KernelReport:
    """Kernel with both SM and DRAM above 80% -> balanced (near-optimal)."""
    return KernelReport(
        kernel_name="balanced_kernel",
        demangled_name="balanced_kernel<float>",
        launch_params=launch_256x4096,
        device_info=device_h100,
        metrics={
            "sm__throughput.avg.pct_of_peak_sustained_elapsed":
                _mv("sm__throughput.avg.pct_of_peak_sustained_elapsed", 88.0, "%"),
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed":
                _mv("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", 85.0, "%"),
        },
    )


@pytest.fixture
def report_with_rules(launch_256x4096: LaunchParams, device_h100: DeviceInfo) -> KernelReport:
    """Kernel with NCU rule results."""
    return KernelReport(
        kernel_name="rules_kernel",
        demangled_name="rules_kernel<float>",
        launch_params=launch_256x4096,
        device_info=device_h100,
        metrics={
            "sm__throughput.avg.pct_of_peak_sustained_elapsed":
                _mv("sm__throughput.avg.pct_of_peak_sustained_elapsed", 50.0, "%"),
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed":
                _mv("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", 50.0, "%"),
        },
        rule_results=[
            RuleResult(rule_name="SpeedOfLight", severity="OK", message="Utilization is fine."),
            RuleResult(rule_name="MemoryWorkloadAnalysis", severity="MED",
                       message="Memory is moderately utilized."),
            RuleResult(rule_name="ComputeWorkloadAnalysis", severity="HIGH",
                       message="Compute pipeline is oversubscribed."),
            RuleResult(rule_name="Occupancy", severity="LOW",
                       message="Low occupancy: 25%"),
        ],
    )


@pytest.fixture
def report_coalesced(launch_256x4096: LaunchParams, device_h100: DeviceInfo) -> KernelReport:
    """Kernel with perfectly coalesced memory access."""
    return KernelReport(
        kernel_name="coalesced_kernel",
        demangled_name="coalesced_kernel<float>",
        launch_params=launch_256x4096,
        device_info=device_h100,
        metrics={
            "sm__throughput.avg.pct_of_peak_sustained_elapsed":
                _mv("sm__throughput.avg.pct_of_peak_sustained_elapsed", 40.0, "%"),
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed":
                _mv("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", 70.0, "%"),
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum":
                _mv("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", 1100.0, "sector"),
            "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum":
                _mv("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", 1000.0, "request"),
        },
    )


@pytest.fixture
def report_no_stalls(launch_256x4096: LaunchParams, device_h100: DeviceInfo) -> KernelReport:
    """Kernel with stall metrics all at zero."""
    return KernelReport(
        kernel_name="no_stall_kernel",
        demangled_name="no_stall_kernel<float>",
        launch_params=launch_256x4096,
        device_info=device_h100,
        metrics={
            "sm__throughput.avg.pct_of_peak_sustained_elapsed":
                _mv("sm__throughput.avg.pct_of_peak_sustained_elapsed", 90.0, "%"),
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed":
                _mv("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", 20.0, "%"),
            "smsp__pcsamp_warps_issue_stalled_long_scoreboard.sum":
                _mv("smsp__pcsamp_warps_issue_stalled_long_scoreboard.sum", 0.0, ""),
            "smsp__pcsamp_warps_issue_stalled_barrier.sum":
                _mv("smsp__pcsamp_warps_issue_stalled_barrier.sum", 0.0, ""),
        },
    )


@pytest.fixture
def report_minimal(device_h100: DeviceInfo) -> KernelReport:
    """Minimal kernel report with no metrics (tests graceful degradation)."""
    return KernelReport(
        kernel_name="minimal_kernel",
        demangled_name="minimal_kernel",
        launch_params=LaunchParams(
            grid=(1, 1, 1), block=(1, 1, 1),
            shared_mem_bytes=0, registers_per_thread=16,
        ),
        device_info=device_h100,
        metrics={},
    )
