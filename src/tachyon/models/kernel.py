"""Core data models for kernel analysis reports.

These models form the backbone of the entire analysis pipeline. Every layer
from Reader to Report consumes or produces these types.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LaunchParams:
    """CUDA kernel launch configuration."""
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    shared_mem_bytes: int
    registers_per_thread: int
    static_shared_mem_bytes: int = 0

    @property
    def total_threads(self) -> int:
        """Total number of threads across all blocks."""
        return (self.grid[0] * self.grid[1] * self.grid[2]
                * self.block[0] * self.block[1] * self.block[2])

    @property
    def block_size(self) -> int:
        """Number of threads per block."""
        return self.block[0] * self.block[1] * self.block[2]

    @property
    def grid_size(self) -> int:
        """Number of blocks in the grid."""
        return self.grid[0] * self.grid[1] * self.grid[2]


@dataclass(frozen=True)
class DeviceInfo:
    """GPU device properties."""
    name: str
    compute_capability: tuple[int, int]
    sm_count: int
    max_clock_mhz: int
    memory_bus_width: int
    peak_memory_bandwidth_gbps: float


@dataclass(frozen=True)
class MetricValue:
    """Scalar metric value (kernel-level aggregate)."""
    name: str
    value: float
    unit: str


@dataclass(frozen=True)
class InstancedMetricValue:
    """Per-PC instanced metric value."""
    pc: int
    value: float
    source_file: str | None = None
    source_line: int | None = None


@dataclass(frozen=True)
class RuleResult:
    """NCU built-in rule analysis result."""
    rule_name: str
    severity: str       # "OK" / "LOW" / "MED" / "HIGH"
    message: str


@dataclass
class KernelReport:
    """Complete analysis snapshot for a single kernel launch.

    This is the primary data carrier through the entire pipeline:
    Reader -> Analyzers -> Report.
    """
    kernel_name: str
    demangled_name: str
    launch_params: LaunchParams
    device_info: DeviceInfo
    metrics: dict[str, MetricValue] = field(default_factory=dict)
    instanced_metrics: dict[str, list[InstancedMetricValue]] = field(default_factory=dict)
    source_files: dict[str, str] = field(default_factory=dict)
    rule_results: list[RuleResult] = field(default_factory=list)

    def metric_value(self, name: str) -> float | None:
        """Convenience accessor: get scalar metric value by name, or None."""
        mv = self.metrics.get(name)
        return mv.value if mv is not None else None

    def has_metrics(self, names: list[str]) -> bool:
        """Check whether all named metrics are present."""
        return all(name in self.metrics for name in names)

    def instanced_metrics_as_tuples(
        self,
    ) -> dict[str, list[tuple[int, float]]]:
        """Convert instanced metrics to (pc, value) tuples for SourceCorrelator.

        Returns:
            Dict mapping metric_name to list of (pc_address, value) tuples.
        """
        result: dict[str, list[tuple[int, float]]] = {}
        for name, values in self.instanced_metrics.items():
            tuples = [(imv.pc, imv.value) for imv in values]
            if tuples:
                result[name] = tuples
        return result
