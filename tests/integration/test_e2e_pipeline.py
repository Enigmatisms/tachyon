"""Integration tests — E2E profiling pipeline (two-stage NcuProfiler flow).

Tests the full run_e2e_pipeline orchestration:
  Stage 1 (Quick Scan) -> parse -> top-K -> Stage 2 (Deep Dive) -> report path.

All external dependencies (NcuProfiler, NcuReportReader, ToolPathResolver) are
mocked so no GPU or NCU installation is required.
"""
from pathlib import Path
from unittest.mock import patch

from tachyon.config.settings import TachyonConfig
from tachyon.errors.handler import ErrorCode, ToolResult
from tachyon.models.kernel import (
    DeviceInfo,
    KernelReport,
    LaunchParams,
    MetricValue,
)
from tachyon.profiler.ncu_profiler import ProfilingResult, ProfilingStrategy
from tachyon.profiler.pipeline import run_e2e_pipeline

# ━━━━━━━━━━━━━━━━━━━━━━━ Helpers ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _mv(name: str, value: float, unit: str = "") -> MetricValue:
    return MetricValue(name=name, value=value, unit=unit)


def _make_config() -> TachyonConfig:
    """Minimal TachyonConfig with conservative strategy."""
    cfg = TachyonConfig()
    cfg.profiling.strategy = "conservative"
    return cfg


def _make_profiling_result(stage: int, path: Path) -> ProfilingResult:
    return ProfilingResult(
        ncu_rep_path=path,
        stage=stage,
        returncode=0,
        stderr="",
        elapsed_sec=1.5,
    )


_DEVICE = DeviceInfo(
    name="NVIDIA H100 80GB HBM3",
    compute_capability=(9, 0),
    sm_count=132,
    max_clock_mhz=1980,
    memory_bus_width=5120,
    peak_memory_bandwidth_gbps=3352.0,
)

_LAUNCH = LaunchParams(
    grid=(4096, 1, 1),
    block=(256, 1, 1),
    shared_mem_bytes=0,
    registers_per_thread=32,
)


def _make_kernels(names: list[str], durations: list[float]) -> list[KernelReport]:
    """Create KernelReport list with gpu__time_duration.sum for sorting."""
    reports = []
    for name, dur in zip(names, durations):
        reports.append(
            KernelReport(
                kernel_name=name,
                demangled_name=f"{name}<float>",
                launch_params=_LAUNCH,
                device_info=_DEVICE,
                metrics={
                    "gpu__time_duration.sum": _mv("gpu__time_duration.sum", dur, "ns"),
                    "sm__throughput.avg.pct_of_peak_sustained_elapsed": _mv(
                        "sm__throughput.avg.pct_of_peak_sustained_elapsed", 50.0, "%"
                    ),
                },
            )
        )
    return reports


# ━━━━━━━━━━━━━━━━━━━━━━━ Tests ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestE2EPipeline:

    async def test_stage1_failure_reports_error(self, tmp_path: Path):
        """Stage 1 profiling fails -> pipeline returns ToolResult.fail."""
        config = _make_config()
        stage1_error = ToolResult.fail(
            ErrorCode.TOOL_NOT_FOUND,
            "ncu binary not found at: /fake/ncu",
            suggestion="Install CUDA Toolkit.",
        )

        with patch("tachyon.profiler.pipeline.ToolPathResolver"), \
             patch("tachyon.profiler.pipeline.NcuProfiler") as MockProfiler:

            profiler_inst = MockProfiler.return_value
            profiler_inst.profile_basic.return_value = stage1_error

            result = await run_e2e_pipeline(
                "./my_app", ["--size", "1024"],
                config=config,
                output_dir=tmp_path,
            )

        assert not result.success
        assert result.error is not None
        assert result.error.code == ErrorCode.TOOL_NOT_FOUND
        assert "ncu binary not found" in result.error.message
        # profile_targeted must NOT be called when stage 1 fails
        profiler_inst.profile_targeted.assert_not_called()

    async def test_stage2_fallback_to_stage1(self, tmp_path: Path):
        """Stage 1 succeeds, Stage 2 fails -> pipeline returns Stage 1 path."""
        config = _make_config()
        stage1_path = tmp_path / "stage1.ncu-rep"
        stage2_error = ToolResult.fail(
            ErrorCode.UNKNOWN,
            "ncu Stage 2 timed out after 1800s",
            suggestion="Increase timeout.",
        )

        kernels = _make_kernels(["matmul", "relu"], [5000.0, 1000.0])

        with patch("tachyon.profiler.pipeline.ToolPathResolver"), \
             patch("tachyon.profiler.pipeline.NcuProfiler") as MockProfiler, \
             patch("tachyon.reader.ncu_reader.NcuReportReader") as MockReader:

            profiler_inst = MockProfiler.return_value
            profiler_inst.profile_basic.return_value = ToolResult.ok(
                _make_profiling_result(1, stage1_path),
            )
            profiler_inst.profile_targeted.return_value = stage2_error

            reader_inst = MockReader.return_value
            reader_inst.load.return_value = ToolResult.ok(kernels)

            result = await run_e2e_pipeline(
                "./my_app", ["--size", "1024"],
                config=config,
                output_dir=tmp_path,
            )

        assert result.success
        assert result.data == stage1_path
        # Stage 2 was attempted
        profiler_inst.profile_targeted.assert_called_once()

    async def test_full_pipeline_conservative(self, tmp_path: Path):
        """Full two-stage pipeline with conservative strategy succeeds."""
        config = _make_config()
        stage1_path = tmp_path / "stage1.ncu-rep"
        stage2_path = tmp_path / "stage2.ncu-rep"

        kernels = _make_kernels(
            ["matmul", "softmax", "relu", "dropout", "layernorm", "gelu"],
            [10000.0, 8000.0, 6000.0, 4000.0, 2000.0, 1000.0],
        )

        with patch("tachyon.profiler.pipeline.ToolPathResolver"), \
             patch("tachyon.profiler.pipeline.NcuProfiler") as MockProfiler, \
             patch("tachyon.reader.ncu_reader.NcuReportReader") as MockReader:

            profiler_inst = MockProfiler.return_value
            profiler_inst.profile_basic.return_value = ToolResult.ok(
                _make_profiling_result(1, stage1_path),
            )
            profiler_inst.profile_targeted.return_value = ToolResult.ok(
                _make_profiling_result(2, stage2_path),
            )

            reader_inst = MockReader.return_value
            reader_inst.load.return_value = ToolResult.ok(kernels)

            result = await run_e2e_pipeline(
                "./my_app", ["--size", "1024"],
                config=config,
                strategy=ProfilingStrategy.CONSERVATIVE,
                top_k=5,
                output_dir=tmp_path,
            )

        # Pipeline returns Stage 2 path on full success
        assert result.success
        assert result.data == stage2_path

        # Verify Stage 1 was called correctly
        profiler_inst.profile_basic.assert_called_once()
        call_args = profiler_inst.profile_basic.call_args
        assert call_args[0][0] == "./my_app"
        assert call_args[0][1] == ["--size", "1024"]

        # Verify Stage 2 received top-K kernel names (top 5 of 6)
        profiler_inst.profile_targeted.assert_called_once()
        s2_call = profiler_inst.profile_targeted.call_args
        passed_kernels = s2_call[1]["kernels"] if "kernels" in s2_call[1] else s2_call[0][2]
        # Top-5 by duration: matmul, softmax, relu, dropout, layernorm
        # (gelu at 1000.0 is excluded from top 5)
        assert "gelu" not in passed_kernels
        assert len(passed_kernels) <= 5

        # Verify reader.load received Stage 1 path
        reader_inst.load.assert_called_once_with(stage1_path)

    async def test_kernel_filter_overrides_top_k(self, tmp_path: Path):
        """When kernel_filter is provided, those names override auto-detected top-K."""
        config = _make_config()
        stage1_path = tmp_path / "stage1.ncu-rep"
        stage2_path = tmp_path / "stage2.ncu-rep"

        kernels = _make_kernels(
            ["matmul", "softmax", "relu"],
            [10000.0, 5000.0, 1000.0],
        )
        user_filter = ["my_custom_kernel", "another_kernel"]

        with patch("tachyon.profiler.pipeline.ToolPathResolver"), \
             patch("tachyon.profiler.pipeline.NcuProfiler") as MockProfiler, \
             patch("tachyon.reader.ncu_reader.NcuReportReader") as MockReader:

            profiler_inst = MockProfiler.return_value
            profiler_inst.profile_basic.return_value = ToolResult.ok(
                _make_profiling_result(1, stage1_path),
            )
            profiler_inst.profile_targeted.return_value = ToolResult.ok(
                _make_profiling_result(2, stage2_path),
            )

            reader_inst = MockReader.return_value
            reader_inst.load.return_value = ToolResult.ok(kernels)

            result = await run_e2e_pipeline(
                "./my_app", [],
                config=config,
                kernel_filter=user_filter,
                output_dir=tmp_path,
            )

        assert result.success
        assert result.data == stage2_path

        # Stage 2 should receive user-specified kernel names, NOT auto-detected ones
        s2_call = profiler_inst.profile_targeted.call_args
        passed_kernels = s2_call[1]["kernels"] if "kernels" in s2_call[1] else s2_call[0][2]
        assert passed_kernels == user_filter

    async def test_stage1_load_failure(self, tmp_path: Path):
        """Stage 1 succeeds but NcuReportReader.load fails -> returns error."""
        config = _make_config()
        stage1_path = tmp_path / "stage1.ncu-rep"

        load_error = ToolResult.fail(
            ErrorCode.INVALID_BINARY,
            "Failed to load report: corrupt file",
            suggestion="The file may be corrupt.",
        )

        with patch("tachyon.profiler.pipeline.ToolPathResolver"), \
             patch("tachyon.profiler.pipeline.NcuProfiler") as MockProfiler, \
             patch("tachyon.reader.ncu_reader.NcuReportReader") as MockReader:

            profiler_inst = MockProfiler.return_value
            profiler_inst.profile_basic.return_value = ToolResult.ok(
                _make_profiling_result(1, stage1_path),
            )

            reader_inst = MockReader.return_value
            reader_inst.load.return_value = load_error

            result = await run_e2e_pipeline(
                "./my_app", [],
                config=config,
                output_dir=tmp_path,
            )

        assert not result.success
        assert result.error is not None
        assert result.error.code == ErrorCode.INVALID_BINARY
        assert "corrupt" in result.error.message
        # Stage 2 must NOT be attempted when load fails
        profiler_inst.profile_targeted.assert_not_called()


class TestE2EPipelineEdgeCases:

    async def test_top_k_greater_than_kernel_count(self, tmp_path: Path):
        """top_k=10 but only 3 kernels -> passes all 3 to Stage 2."""
        config = _make_config()
        stage1_path = tmp_path / "stage1.ncu-rep"
        stage2_path = tmp_path / "stage2.ncu-rep"

        kernels = _make_kernels(["k1", "k2", "k3"], [3000.0, 2000.0, 1000.0])

        with patch("tachyon.profiler.pipeline.ToolPathResolver"), \
             patch("tachyon.profiler.pipeline.NcuProfiler") as MockProfiler, \
             patch("tachyon.reader.ncu_reader.NcuReportReader") as MockReader:

            profiler_inst = MockProfiler.return_value
            profiler_inst.profile_basic.return_value = ToolResult.ok(
                _make_profiling_result(1, stage1_path),
            )
            profiler_inst.profile_targeted.return_value = ToolResult.ok(
                _make_profiling_result(2, stage2_path),
            )

            reader_inst = MockReader.return_value
            reader_inst.load.return_value = ToolResult.ok(kernels)

            result = await run_e2e_pipeline(
                "./my_app", [],
                config=config,
                top_k=10,
                output_dir=tmp_path,
            )

        assert result.success
        s2_call = profiler_inst.profile_targeted.call_args
        passed_kernels = s2_call[1]["kernels"] if "kernels" in s2_call[1] else s2_call[0][2]
        assert set(passed_kernels) == {"k1", "k2", "k3"}

    async def test_sort_falls_back_to_sm_throughput(self, tmp_path: Path):
        """Kernels without gpu__time_duration use SM throughput for sorting."""
        config = _make_config()
        stage1_path = tmp_path / "stage1.ncu-rep"
        stage2_path = tmp_path / "stage2.ncu-rep"

        # Build kernels with only SM throughput (no duration metric)
        kernels = []
        for name, sm_pct in [("high_sm", 90.0), ("low_sm", 10.0), ("mid_sm", 50.0)]:
            kernels.append(
                KernelReport(
                    kernel_name=name,
                    demangled_name=f"{name}<float>",
                    launch_params=_LAUNCH,
                    device_info=_DEVICE,
                    metrics={
                        "sm__throughput.avg.pct_of_peak_sustained_elapsed": _mv(
                            "sm__throughput.avg.pct_of_peak_sustained_elapsed", sm_pct, "%"
                        ),
                    },
                )
            )

        with patch("tachyon.profiler.pipeline.ToolPathResolver"), \
             patch("tachyon.profiler.pipeline.NcuProfiler") as MockProfiler, \
             patch("tachyon.reader.ncu_reader.NcuReportReader") as MockReader:

            profiler_inst = MockProfiler.return_value
            profiler_inst.profile_basic.return_value = ToolResult.ok(
                _make_profiling_result(1, stage1_path),
            )
            profiler_inst.profile_targeted.return_value = ToolResult.ok(
                _make_profiling_result(2, stage2_path),
            )

            reader_inst = MockReader.return_value
            reader_inst.load.return_value = ToolResult.ok(kernels)

            result = await run_e2e_pipeline(
                "./my_app", [],
                config=config,
                top_k=2,
                output_dir=tmp_path,
            )

        assert result.success
        s2_call = profiler_inst.profile_targeted.call_args
        passed_kernels = s2_call[1]["kernels"] if "kernels" in s2_call[1] else s2_call[0][2]
        # top-2 by SM throughput: high_sm (90) and mid_sm (50); low_sm excluded
        assert "low_sm" not in passed_kernels
        assert len(passed_kernels) == 2

    async def test_duplicate_kernel_names_deduped(self, tmp_path: Path):
        """Duplicate kernel names in top-K are deduplicated via set()."""
        config = _make_config()
        stage1_path = tmp_path / "stage1.ncu-rep"
        stage2_path = tmp_path / "stage2.ncu-rep"

        # Two launches of the same kernel with different durations
        kernels = _make_kernels(
            ["matmul", "matmul", "relu"],
            [10000.0, 9000.0, 1000.0],
        )

        with patch("tachyon.profiler.pipeline.ToolPathResolver"), \
             patch("tachyon.profiler.pipeline.NcuProfiler") as MockProfiler, \
             patch("tachyon.reader.ncu_reader.NcuReportReader") as MockReader:

            profiler_inst = MockProfiler.return_value
            profiler_inst.profile_basic.return_value = ToolResult.ok(
                _make_profiling_result(1, stage1_path),
            )
            profiler_inst.profile_targeted.return_value = ToolResult.ok(
                _make_profiling_result(2, stage2_path),
            )

            reader_inst = MockReader.return_value
            reader_inst.load.return_value = ToolResult.ok(kernels)

            result = await run_e2e_pipeline(
                "./my_app", [],
                config=config,
                top_k=3,
                output_dir=tmp_path,
            )

        assert result.success
        s2_call = profiler_inst.profile_targeted.call_args
        passed_kernels = s2_call[1]["kernels"] if "kernels" in s2_call[1] else s2_call[0][2]
        # "matmul" appears twice in top-3 but kernel_names uses set() dedup
        assert passed_kernels.count("matmul") == 1
        assert "relu" in passed_kernels

    async def test_extra_ncu_args_forwarded(self, tmp_path: Path):
        """extra_ncu_args are forwarded to both profile_basic and profile_targeted."""
        config = _make_config()
        stage1_path = tmp_path / "stage1.ncu-rep"
        stage2_path = tmp_path / "stage2.ncu-rep"

        kernels = _make_kernels(["matmul"], [5000.0])
        extra = ["--replay-mode", "kernel"]

        with patch("tachyon.profiler.pipeline.ToolPathResolver"), \
             patch("tachyon.profiler.pipeline.NcuProfiler") as MockProfiler, \
             patch("tachyon.reader.ncu_reader.NcuReportReader") as MockReader:

            profiler_inst = MockProfiler.return_value
            profiler_inst.profile_basic.return_value = ToolResult.ok(
                _make_profiling_result(1, stage1_path),
            )
            profiler_inst.profile_targeted.return_value = ToolResult.ok(
                _make_profiling_result(2, stage2_path),
            )

            reader_inst = MockReader.return_value
            reader_inst.load.return_value = ToolResult.ok(kernels)

            result = await run_e2e_pipeline(
                "./my_app", [],
                config=config,
                extra_ncu_args=extra,
                output_dir=tmp_path,
            )

        assert result.success
        # Verify extra_ncu_args forwarded to profile_basic
        s1_kwargs = profiler_inst.profile_basic.call_args[1]
        assert s1_kwargs["extra_ncu_args"] == extra
        # Verify extra_ncu_args forwarded to profile_targeted
        s2_kwargs = profiler_inst.profile_targeted.call_args[1]
        assert s2_kwargs["extra_ncu_args"] == extra

    async def test_strategy_forwarded(self, tmp_path: Path):
        """Explicit strategy parameter is forwarded to both stages."""
        config = _make_config()
        stage1_path = tmp_path / "stage1.ncu-rep"
        stage2_path = tmp_path / "stage2.ncu-rep"

        kernels = _make_kernels(["matmul"], [5000.0])

        with patch("tachyon.profiler.pipeline.ToolPathResolver"), \
             patch("tachyon.profiler.pipeline.NcuProfiler") as MockProfiler, \
             patch("tachyon.reader.ncu_reader.NcuReportReader") as MockReader:

            profiler_inst = MockProfiler.return_value
            profiler_inst.profile_basic.return_value = ToolResult.ok(
                _make_profiling_result(1, stage1_path),
            )
            profiler_inst.profile_targeted.return_value = ToolResult.ok(
                _make_profiling_result(2, stage2_path),
            )

            reader_inst = MockReader.return_value
            reader_inst.load.return_value = ToolResult.ok(kernels)

            result = await run_e2e_pipeline(
                "./my_app", [],
                config=config,
                strategy=ProfilingStrategy.RADICAL,
                output_dir=tmp_path,
            )

        assert result.success
        # Verify strategy forwarded to profile_basic
        s1_kwargs = profiler_inst.profile_basic.call_args[1]
        assert s1_kwargs["strategy"] == ProfilingStrategy.RADICAL
        # Verify strategy forwarded to profile_targeted
        s2_kwargs = profiler_inst.profile_targeted.call_args[1]
        assert s2_kwargs["strategy"] == ProfilingStrategy.RADICAL

    async def test_output_dir_forwarded(self, tmp_path: Path):
        """output_dir is forwarded to both profile_basic and profile_targeted."""
        config = _make_config()
        out = tmp_path / "profiling_output"
        out.mkdir()
        stage1_path = out / "stage1.ncu-rep"
        stage2_path = out / "stage2.ncu-rep"

        kernels = _make_kernels(["matmul"], [5000.0])

        with patch("tachyon.profiler.pipeline.ToolPathResolver"), \
             patch("tachyon.profiler.pipeline.NcuProfiler") as MockProfiler, \
             patch("tachyon.reader.ncu_reader.NcuReportReader") as MockReader:

            profiler_inst = MockProfiler.return_value
            profiler_inst.profile_basic.return_value = ToolResult.ok(
                _make_profiling_result(1, stage1_path),
            )
            profiler_inst.profile_targeted.return_value = ToolResult.ok(
                _make_profiling_result(2, stage2_path),
            )

            reader_inst = MockReader.return_value
            reader_inst.load.return_value = ToolResult.ok(kernels)

            result = await run_e2e_pipeline(
                "./my_app", [],
                config=config,
                output_dir=out,
            )

        assert result.success
        s1_kwargs = profiler_inst.profile_basic.call_args[1]
        assert s1_kwargs["output_dir"] == out
        s2_kwargs = profiler_inst.profile_targeted.call_args[1]
        assert s2_kwargs["output_dir"] == out

    async def test_kernels_with_no_metrics_sort_to_zero(self, tmp_path: Path):
        """Kernels with no duration and no SM throughput get sort key 0.0."""
        config = _make_config()
        stage1_path = tmp_path / "stage1.ncu-rep"
        stage2_path = tmp_path / "stage2.ncu-rep"

        kernels = [
            KernelReport(
                kernel_name="empty_kernel",
                demangled_name="empty_kernel",
                launch_params=_LAUNCH,
                device_info=_DEVICE,
                metrics={},  # no metrics at all
            ),
            *_make_kernels(["matmul"], [5000.0]),
        ]

        with patch("tachyon.profiler.pipeline.ToolPathResolver"), \
             patch("tachyon.profiler.pipeline.NcuProfiler") as MockProfiler, \
             patch("tachyon.reader.ncu_reader.NcuReportReader") as MockReader:

            profiler_inst = MockProfiler.return_value
            profiler_inst.profile_basic.return_value = ToolResult.ok(
                _make_profiling_result(1, stage1_path),
            )
            profiler_inst.profile_targeted.return_value = ToolResult.ok(
                _make_profiling_result(2, stage2_path),
            )

            reader_inst = MockReader.return_value
            reader_inst.load.return_value = ToolResult.ok(kernels)

            result = await run_e2e_pipeline(
                "./my_app", [],
                config=config,
                top_k=1,
                output_dir=tmp_path,
            )

        assert result.success
        # matmul (5000.0) should be top-1, empty_kernel (0.0) excluded
        s2_call = profiler_inst.profile_targeted.call_args
        passed_kernels = s2_call[1]["kernels"] if "kernels" in s2_call[1] else s2_call[0][2]
        assert "matmul" in passed_kernels
        assert "empty_kernel" not in passed_kernels
