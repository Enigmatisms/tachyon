"""Unit tests for the 3 new analyzers: Occupancy, Instruction, LaunchConfig.

Uses synthetic KernelReport fixtures with controlled metric values to trigger
specific code paths in each analyzer.
"""
from __future__ import annotations

from tachyon.models.finding import Severity
from tachyon.models.kernel import (
    DeviceInfo,
    KernelReport,
    LaunchParams,
    MetricValue,
)

# ━━━━━━━━━━━━━━━━━━━━━━━ Helpers ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _mv(name: str, value: float, unit: str = "") -> MetricValue:
    return MetricValue(name=name, value=value, unit=unit)


def _device(sm_count: int = 108) -> DeviceInfo:
    return DeviceInfo(
        name="NVIDIA A100-SXM4-80GB",
        compute_capability=(8, 0),
        sm_count=sm_count,
        max_clock_mhz=1410,
        memory_bus_width=5120,
        peak_memory_bandwidth_gbps=2039.0,
    )


def _occupancy_metrics(
    achieved: float = 80.0,
    limit_warps: float = 64.0,
    limit_regs: float = 48.0,
    limit_smem: float = 48.0,
    limit_blocks: float = 32.0,
) -> dict[str, MetricValue]:
    """Build a full set of occupancy metrics."""
    return {
        "sm__warps_active.avg.pct_of_peak_sustained_active":
            _mv("sm__warps_active.avg.pct_of_peak_sustained_active", achieved, "%"),
        "launch__occupancy_limit_warps":
            _mv("launch__occupancy_limit_warps", limit_warps, "warps"),
        "launch__occupancy_limit_registers":
            _mv("launch__occupancy_limit_registers", limit_regs, "warps"),
        "launch__occupancy_limit_shared_mem":
            _mv("launch__occupancy_limit_shared_mem", limit_smem, "warps"),
        "launch__occupancy_limit_blocks":
            _mv("launch__occupancy_limit_blocks", limit_blocks, "warps"),
    }


def _instruction_metrics(
    fp16: float = 1000.0,
    fp32: float = 5000.0,
    fp64: float = 500.0,
    tensor: float = 200.0,
    total: float = 50000.0,
) -> dict[str, MetricValue]:
    """Build a full set of instruction mix metrics."""
    return {
        "sm__sass_thread_inst_executed_op_fp16_pred_on.sum":
            _mv("sm__sass_thread_inst_executed_op_fp16_pred_on.sum", fp16, "inst"),
        "sm__sass_thread_inst_executed_op_fp32_pred_on.sum":
            _mv("sm__sass_thread_inst_executed_op_fp32_pred_on.sum", fp32, "inst"),
        "sm__sass_thread_inst_executed_op_fp64_pred_on.sum":
            _mv("sm__sass_thread_inst_executed_op_fp64_pred_on.sum", fp64, "inst"),
        "sm__inst_executed_pipe_tensor.sum":
            _mv("sm__inst_executed_pipe_tensor.sum", tensor, "inst"),
        "sm__inst_executed.sum":
            _mv("sm__inst_executed.sum", total, "inst"),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━ OccupancyAnalyzer ━━━━━━━━━━━━━━━━━━━━━


class TestOccupancyAnalyzer:
    """Test OccupancyAnalyzer for various occupancy scenarios."""

    def _make_report(
        self,
        achieved: float,
        regs_per_thread: int = 32,
        limit_regs: float = 48.0,
        limit_smem: float = 48.0,
        limit_blocks: float = 32.0,
    ) -> KernelReport:
        return KernelReport(
            kernel_name="occ_kernel",
            demangled_name="occ_kernel<float>",
            launch_params=LaunchParams(
                grid=(4096, 1, 1),
                block=(256, 1, 1),
                shared_mem_bytes=0,
                registers_per_thread=regs_per_thread,
            ),
            device_info=_device(),
            metrics=_occupancy_metrics(
                achieved=achieved,
                limit_regs=limit_regs,
                limit_smem=limit_smem,
                limit_blocks=limit_blocks,
            ),
        )

    def test_name_and_category(self):
        from tachyon.analyzers.occupancy import OccupancyAnalyzer
        a = OccupancyAnalyzer()
        assert a.name() == "occupancy"
        assert a.category() == "occupancy"

    def test_required_metrics(self):
        from tachyon.analyzers.occupancy import OccupancyAnalyzer
        metrics = OccupancyAnalyzer().required_metrics()
        # No required metrics — graceful degradation from launch_params
        assert len(metrics) == 0

    def test_always_can_run(self):
        """OccupancyAnalyzer always runs (uses launch_params, not just metrics)."""
        from tachyon.analyzers.occupancy import OccupancyAnalyzer
        from tachyon.models.kernel import DeviceInfo, KernelReport, LaunchParams
        minimal = KernelReport(
            kernel_name="k", demangled_name="k",
            launch_params=LaunchParams(grid=(1,1,1), block=(256,1,1),
                                       shared_mem_bytes=0, registers_per_thread=32),
            device_info=_device(), metrics={},
        )
        assert OccupancyAnalyzer().can_run(minimal)

    def test_low_occupancy_warning(self):
        """Achieved occupancy below 50% should produce a WARNING."""
        from tachyon.analyzers.occupancy import OccupancyAnalyzer
        report = self._make_report(achieved=35.0)
        findings = OccupancyAnalyzer().analyze(report)

        low_findings = [f for f in findings if "Low occupancy" in f.title]
        assert len(low_findings) == 1
        assert low_findings[0].severity == Severity.WARNING
        assert "35.0%" in low_findings[0].title

    def test_low_occupancy_identifies_dominant_limiter(self):
        """Low occupancy should identify register limit as dominant when it's smallest."""
        from tachyon.analyzers.occupancy import OccupancyAnalyzer
        report = self._make_report(
            achieved=40.0,
            limit_regs=16.0,    # smallest -> dominant
            limit_smem=48.0,
            limit_blocks=32.0,
        )
        findings = OccupancyAnalyzer().analyze(report)
        low_findings = [f for f in findings if "Low occupancy" in f.title]
        assert len(low_findings) == 1
        assert "register" in low_findings[0].detail.lower()

    def test_low_occupancy_smem_dominant(self):
        """Shared memory as dominant limiter."""
        from tachyon.analyzers.occupancy import OccupancyAnalyzer
        report = self._make_report(
            achieved=42.0,
            limit_regs=48.0,
            limit_smem=12.0,    # smallest -> dominant
            limit_blocks=32.0,
        )
        findings = OccupancyAnalyzer().analyze(report)
        low_findings = [f for f in findings if "Low occupancy" in f.title]
        assert len(low_findings) == 1
        assert "shared memory" in low_findings[0].detail.lower()

    def test_high_register_pressure_warning(self):
        """Registers per thread > 64 should produce a register pressure WARNING."""
        from tachyon.analyzers.occupancy import OccupancyAnalyzer
        report = self._make_report(achieved=60.0, regs_per_thread=80)
        findings = OccupancyAnalyzer().analyze(report)

        reg_findings = [f for f in findings if "register pressure" in f.title.lower()]
        assert len(reg_findings) == 1
        assert reg_findings[0].severity == Severity.WARNING
        assert "80" in reg_findings[0].title

    def test_no_register_pressure_at_threshold(self):
        """Exactly 64 registers should NOT trigger register pressure warning."""
        from tachyon.analyzers.occupancy import OccupancyAnalyzer
        report = self._make_report(achieved=60.0, regs_per_thread=64)
        findings = OccupancyAnalyzer().analyze(report)

        reg_findings = [f for f in findings if "register pressure" in f.title.lower()]
        assert len(reg_findings) == 0

    def test_healthy_occupancy_info(self):
        """Achieved occupancy above 75% should produce an INFO finding."""
        from tachyon.analyzers.occupancy import OccupancyAnalyzer
        report = self._make_report(achieved=82.0)
        findings = OccupancyAnalyzer().analyze(report)

        healthy = [f for f in findings if "healthy" in f.title.lower()]
        assert len(healthy) == 1
        assert healthy[0].severity == Severity.INFO
        assert "82.0%" in healthy[0].title

    def test_mid_occupancy_no_warning_no_healthy(self):
        """Occupancy between 50% and 75% -> no low warning, no healthy info."""
        from tachyon.analyzers.occupancy import OccupancyAnalyzer
        report = self._make_report(achieved=60.0)
        findings = OccupancyAnalyzer().analyze(report)

        low_findings = [f for f in findings if "Low occupancy" in f.title]
        healthy = [f for f in findings if "healthy" in f.title.lower()]
        assert len(low_findings) == 0
        assert len(healthy) == 0

    def test_runs_with_no_metrics(self):
        """Should run even with no metrics (uses launch_params for register/smem checks)."""
        from tachyon.analyzers.occupancy import OccupancyAnalyzer
        report = KernelReport(
            kernel_name="k",
            demangled_name="k",
            launch_params=LaunchParams(
                grid=(1, 1, 1), block=(256, 1, 1),
                shared_mem_bytes=0, registers_per_thread=32,
            ),
            device_info=_device(),
            metrics={},
        )
        assert OccupancyAnalyzer().can_run(report)
        # Should not crash, may produce no findings with normal regs
        findings = OccupancyAnalyzer().analyze(report)
        assert isinstance(findings, list)

    def test_combined_low_occupancy_and_register_pressure(self):
        """Both low occupancy AND high register pressure should produce 2 findings."""
        from tachyon.analyzers.occupancy import OccupancyAnalyzer
        report = self._make_report(achieved=30.0, regs_per_thread=96)
        findings = OccupancyAnalyzer().analyze(report)

        low_findings = [f for f in findings if "Low occupancy" in f.title]
        reg_findings = [f for f in findings if "register pressure" in f.title.lower()]
        assert len(low_findings) == 1
        assert len(reg_findings) == 1

    def test_findings_have_metrics(self):
        """Findings should carry metric evidence."""
        from tachyon.analyzers.occupancy import OccupancyAnalyzer
        report = self._make_report(achieved=35.0)
        findings = OccupancyAnalyzer().analyze(report)

        for f in findings:
            assert len(f.metrics) > 0


# ━━━━━━━━━━━━━━━━━━━━━━━ InstructionAnalyzer ━━━━━━━━━━━━━━━━━━━━


class TestInstructionAnalyzer:
    """Test InstructionAnalyzer for FP precision and Tensor Core checks."""

    def _make_report(self, **kwargs) -> KernelReport:
        return KernelReport(
            kernel_name="inst_kernel",
            demangled_name="inst_kernel<float>",
            launch_params=LaunchParams(
                grid=(4096, 1, 1),
                block=(256, 1, 1),
                shared_mem_bytes=0,
                registers_per_thread=32,
            ),
            device_info=_device(),
            metrics=_instruction_metrics(**kwargs),
        )

    def test_name_and_category(self):
        from tachyon.analyzers.instruction import InstructionAnalyzer
        a = InstructionAnalyzer()
        assert a.name() == "instruction"
        assert a.category() == "compute"

    def test_required_metrics(self):
        from tachyon.analyzers.instruction import InstructionAnalyzer
        metrics = InstructionAnalyzer().required_metrics()
        assert len(metrics) == 5
        assert "sm__inst_executed.sum" in metrics

    def test_fp64_heavy_warning(self):
        """FP64 > 30% of total FP should produce a WARNING."""
        from tachyon.analyzers.instruction import InstructionAnalyzer
        # fp64=5000, fp32=2000, fp16=500 -> total_fp=7500, fp64_pct=66.7%
        report = self._make_report(fp16=500, fp32=2000, fp64=5000, tensor=100, total=50000)
        findings = InstructionAnalyzer().analyze(report)

        fp64_findings = [f for f in findings if "FP64" in f.title and "High" in f.title]
        assert len(fp64_findings) == 1
        assert fp64_findings[0].severity == Severity.WARNING

    def test_fp64_below_threshold(self):
        """FP64 < 30% should not produce FP64 warning."""
        from tachyon.analyzers.instruction import InstructionAnalyzer
        # fp64=100, fp32=5000, fp16=1000 -> total_fp=6100, fp64_pct=1.6%
        report = self._make_report(fp16=1000, fp32=5000, fp64=100, tensor=200, total=50000)
        findings = InstructionAnalyzer().analyze(report)

        fp64_warnings = [f for f in findings
                         if "FP64" in f.title and f.severity == Severity.WARNING]
        assert len(fp64_warnings) == 0

    def test_tensor_core_unused_with_fp16(self):
        """FP16 present but tensor=0 should produce info about missed opportunity."""
        from tachyon.analyzers.instruction import InstructionAnalyzer
        report = self._make_report(fp16=3000, fp32=5000, fp64=100, tensor=0, total=50000)
        findings = InstructionAnalyzer().analyze(report)

        tc_findings = [f for f in findings if "Tensor Core" in f.title and "FP16" in f.title]
        assert len(tc_findings) == 1
        assert tc_findings[0].severity == Severity.INFO

    def test_tensor_core_low_utilization(self):
        """Tensor < 10% of total should produce low TC utilization finding."""
        from tachyon.analyzers.instruction import InstructionAnalyzer
        # tensor=100, total=50000 -> tensor_pct=0.2%
        report = self._make_report(fp16=1000, fp32=5000, fp64=500, tensor=100, total=50000)
        findings = InstructionAnalyzer().analyze(report)

        tc_low = [f for f in findings if "Low Tensor Core" in f.title]
        assert len(tc_low) == 1
        assert tc_low[0].severity == Severity.INFO

    def test_tensor_core_high_utilization(self):
        """Tensor > 10% of total should NOT produce low TC warning."""
        from tachyon.analyzers.instruction import InstructionAnalyzer
        # tensor=8000, total=50000 -> tensor_pct=16%
        report = self._make_report(fp16=1000, fp32=5000, fp64=500, tensor=8000, total=50000)
        findings = InstructionAnalyzer().analyze(report)

        tc_low = [f for f in findings if "Low Tensor Core" in f.title]
        assert len(tc_low) == 0

    def test_fp_distribution_always_present(self):
        """FP precision distribution finding should always be present when FP > 0."""
        from tachyon.analyzers.instruction import InstructionAnalyzer
        report = self._make_report(fp16=1000, fp32=5000, fp64=500, tensor=200, total=50000)
        findings = InstructionAnalyzer().analyze(report)

        dist_findings = [f for f in findings if "precision distribution" in f.title.lower()]
        assert len(dist_findings) == 1
        assert dist_findings[0].severity == Severity.INFO

    def test_all_zeros_no_findings(self):
        """All instruction counts zero -> no findings."""
        from tachyon.analyzers.instruction import InstructionAnalyzer
        report = self._make_report(fp16=0, fp32=0, fp64=0, tensor=0, total=0)
        findings = InstructionAnalyzer().analyze(report)
        assert len(findings) == 0

    def test_normal_fp_mix(self):
        """Normal FP mix: no warnings, just distribution INFO."""
        from tachyon.analyzers.instruction import InstructionAnalyzer
        # Balanced: fp16=2000, fp32=5000, fp64=500 -> fp64_pct=6.7% (< 30%)
        # tensor=6000, total=50000 -> tensor_pct=12% (> 10%)
        report = self._make_report(fp16=2000, fp32=5000, fp64=500, tensor=6000, total=50000)
        findings = InstructionAnalyzer().analyze(report)

        warnings = [f for f in findings if f.severity == Severity.WARNING]
        assert len(warnings) == 0
        # Should still have the FP distribution INFO
        dist = [f for f in findings if "precision distribution" in f.title.lower()]
        assert len(dist) == 1

    def test_cannot_run_missing_metrics(self):
        """Should not run when instruction metrics are missing."""
        from tachyon.analyzers.instruction import InstructionAnalyzer
        report = KernelReport(
            kernel_name="k",
            demangled_name="k",
            launch_params=LaunchParams(
                grid=(1, 1, 1), block=(256, 1, 1),
                shared_mem_bytes=0, registers_per_thread=32,
            ),
            device_info=_device(),
            metrics={},
        )
        assert not InstructionAnalyzer().can_run(report)

    def test_findings_have_metrics(self):
        """Findings should carry metric evidence."""
        from tachyon.analyzers.instruction import InstructionAnalyzer
        report = self._make_report(fp16=1000, fp32=5000, fp64=500, tensor=200, total=50000)
        findings = InstructionAnalyzer().analyze(report)

        for f in findings:
            assert len(f.metrics) > 0


# ━━━━━━━━━━━━━━━━━━━━━━━ LaunchConfigAnalyzer ━━━━━━━━━━━━━━━━━━━


class TestLaunchConfigAnalyzer:
    """Test LaunchConfigAnalyzer for launch configuration issues."""

    def _make_report(
        self,
        grid: tuple[int, int, int] = (4096, 1, 1),
        block: tuple[int, int, int] = (256, 1, 1),
        sm_count: int = 108,
    ) -> KernelReport:
        return KernelReport(
            kernel_name="launch_kernel",
            demangled_name="launch_kernel<float>",
            launch_params=LaunchParams(
                grid=grid,
                block=block,
                shared_mem_bytes=0,
                registers_per_thread=32,
            ),
            device_info=_device(sm_count=sm_count),
            metrics={},
        )

    def test_name_and_category(self):
        from tachyon.analyzers.launch import LaunchConfigAnalyzer
        a = LaunchConfigAnalyzer()
        assert a.name() == "launch_config"
        assert a.category() == "launch"

    def test_always_can_run(self):
        """LaunchConfigAnalyzer should always be runnable (no metrics required)."""
        from tachyon.analyzers.launch import LaunchConfigAnalyzer
        report = KernelReport(
            kernel_name="k",
            demangled_name="k",
            launch_params=LaunchParams(
                grid=(1, 1, 1), block=(1, 1, 1),
                shared_mem_bytes=0, registers_per_thread=16,
            ),
            device_info=_device(),
            metrics={},
        )
        assert LaunchConfigAnalyzer().can_run(report)

    def test_unaligned_block_size(self):
        """Block size not multiple of 32 -> WARNING."""
        from tachyon.analyzers.launch import LaunchConfigAnalyzer
        # 100 is not multiple of 32
        report = self._make_report(block=(100, 1, 1))
        findings = LaunchConfigAnalyzer().analyze(report)

        unaligned = [f for f in findings if "warp-aligned" in f.title.lower()]
        assert len(unaligned) == 1
        assert unaligned[0].severity == Severity.WARNING
        assert "100" in unaligned[0].title

    def test_aligned_block_no_warning(self):
        """Block size that is multiple of 32 should not trigger unaligned warning."""
        from tachyon.analyzers.launch import LaunchConfigAnalyzer
        report = self._make_report(block=(256, 1, 1))
        findings = LaunchConfigAnalyzer().analyze(report)

        unaligned = [f for f in findings if "warp-aligned" in f.title.lower()]
        assert len(unaligned) == 0

    def test_small_block_size(self):
        """Block size < 128 -> WARNING."""
        from tachyon.analyzers.launch import LaunchConfigAnalyzer
        report = self._make_report(block=(64, 1, 1))
        findings = LaunchConfigAnalyzer().analyze(report)

        small = [f for f in findings if "Small block" in f.title]
        assert len(small) == 1
        assert small[0].severity == Severity.WARNING

    def test_adequate_block_size(self):
        """Block size >= 128 should not trigger small block warning."""
        from tachyon.analyzers.launch import LaunchConfigAnalyzer
        report = self._make_report(block=(128, 1, 1))
        findings = LaunchConfigAnalyzer().analyze(report)

        small = [f for f in findings if "Small block" in f.title]
        assert len(small) == 0

    def test_small_grid_size(self):
        """Grid size < SM count -> WARNING about underutilization."""
        from tachyon.analyzers.launch import LaunchConfigAnalyzer
        # grid=50 blocks < 108 SMs
        report = self._make_report(grid=(50, 1, 1), block=(256, 1, 1), sm_count=108)
        findings = LaunchConfigAnalyzer().analyze(report)

        small_grid = [f for f in findings if "smaller than SM count" in f.title]
        assert len(small_grid) == 1
        assert small_grid[0].severity == Severity.WARNING
        assert "50" in small_grid[0].title
        assert "108" in small_grid[0].title

    def test_adequate_grid_size(self):
        """Grid size >= SM count should not trigger underutilization warning."""
        from tachyon.analyzers.launch import LaunchConfigAnalyzer
        report = self._make_report(grid=(108, 1, 1), sm_count=108)
        findings = LaunchConfigAnalyzer().analyze(report)

        small_grid = [f for f in findings if "smaller than SM count" in f.title]
        assert len(small_grid) == 0

    def test_tail_effect(self):
        """Grid not multiple of SM count -> INFO about tail effect."""
        from tachyon.analyzers.launch import LaunchConfigAnalyzer
        # 200 % 108 != 0
        report = self._make_report(grid=(200, 1, 1), sm_count=108)
        findings = LaunchConfigAnalyzer().analyze(report)

        tail = [f for f in findings if "tail" in f.title.lower() or "not multiple" in f.title.lower()]
        assert len(tail) == 1
        assert tail[0].severity == Severity.INFO

    def test_no_tail_effect_when_aligned(self):
        """Grid multiple of SM count should not trigger tail effect."""
        from tachyon.analyzers.launch import LaunchConfigAnalyzer
        # 216 % 108 == 0
        report = self._make_report(grid=(216, 1, 1), sm_count=108)
        findings = LaunchConfigAnalyzer().analyze(report)

        tail = [f for f in findings if "tail" in f.title.lower() or "not multiple" in f.title.lower()]
        assert len(tail) == 0

    def test_healthy_config(self):
        """Good launch config: warp-aligned, >= 128, grid >= SMs, multiple of SMs."""
        from tachyon.analyzers.launch import LaunchConfigAnalyzer
        # 256 threads, grid=216 (2*108)
        report = self._make_report(grid=(216, 1, 1), block=(256, 1, 1), sm_count=108)
        findings = LaunchConfigAnalyzer().analyze(report)

        healthy = [f for f in findings if "reasonable" in f.title.lower()]
        assert len(healthy) == 1
        assert healthy[0].severity == Severity.INFO

    def test_3d_block_size(self):
        """3D block size should be computed correctly."""
        from tachyon.analyzers.launch import LaunchConfigAnalyzer
        # 8x8x2 = 128 threads, aligned to warp, >= 128
        report = self._make_report(grid=(216, 1, 1), block=(8, 8, 2), sm_count=108)
        findings = LaunchConfigAnalyzer().analyze(report)

        # Should detect healthy (128 threads, grid aligned)
        healthy = [f for f in findings if "reasonable" in f.title.lower()]
        assert len(healthy) == 1

    def test_3d_block_unaligned(self):
        """3D block with total not multiple of 32 -> WARNING."""
        from tachyon.analyzers.launch import LaunchConfigAnalyzer
        # 7x7x1 = 49 threads -- not warp aligned
        report = self._make_report(grid=(4096, 1, 1), block=(7, 7, 1))
        findings = LaunchConfigAnalyzer().analyze(report)

        unaligned = [f for f in findings if "warp-aligned" in f.title.lower()]
        assert len(unaligned) == 1

    def test_multiple_issues_combined(self):
        """A report can trigger multiple launch issues simultaneously."""
        from tachyon.analyzers.launch import LaunchConfigAnalyzer
        # block=50 (not aligned, < 128), grid=10 (< 108 SMs)
        report = self._make_report(grid=(10, 1, 1), block=(50, 1, 1), sm_count=108)
        findings = LaunchConfigAnalyzer().analyze(report)

        # Should have at least 3 issues: unaligned, small block, small grid
        assert len(findings) >= 3

        titles = " ".join(f.title for f in findings).lower()
        assert "warp-aligned" in titles
        assert "small block" in titles
        assert "smaller than sm count" in titles

    def test_healthy_not_emitted_when_issues_exist(self):
        """When issues exist, the 'looks reasonable' finding should NOT appear."""
        from tachyon.analyzers.launch import LaunchConfigAnalyzer
        # Small block = issue present
        report = self._make_report(grid=(216, 1, 1), block=(64, 1, 1), sm_count=108)
        findings = LaunchConfigAnalyzer().analyze(report)

        healthy = [f for f in findings if "reasonable" in f.title.lower()]
        assert len(healthy) == 0

    def test_findings_have_metrics(self):
        """Findings should carry block_size, grid_size, sm_count metrics."""
        from tachyon.analyzers.launch import LaunchConfigAnalyzer
        report = self._make_report(grid=(200, 1, 1), block=(256, 1, 1))
        findings = LaunchConfigAnalyzer().analyze(report)

        for f in findings:
            assert "block_size" in f.metrics
            assert "grid_size" in f.metrics
            assert "sm_count" in f.metrics

    def test_block_exceeding_gpu_limit(self):
        """Block size > 1024 -> WARNING."""
        from tachyon.analyzers.launch import LaunchConfigAnalyzer
        # 2048 threads (impossible on real GPU, but test the check)
        report = self._make_report(grid=(100, 1, 1), block=(2048, 1, 1))
        findings = LaunchConfigAnalyzer().analyze(report)

        exceed = [f for f in findings if "exceeds GPU limit" in f.title]
        assert len(exceed) == 1
        assert exceed[0].severity == Severity.WARNING
