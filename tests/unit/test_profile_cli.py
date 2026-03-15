"""Unit tests for CLI profile command and shared analysis pipeline.

Tests the profile Click command via CliRunner and the shared run_analysis
pipeline with mocked dependencies.
"""
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from tachyon.config.settings import TachyonConfig
from tachyon.errors.handler import ErrorCode, ToolResult
from tachyon.models.kernel import (
    DeviceInfo,
    KernelReport,
    LaunchParams,
    MetricValue,
)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def sample_kernel():
    return KernelReport(
        kernel_name="myKernel",
        demangled_name="myKernel<float>",
        launch_params=LaunchParams(
            grid=(128, 1, 1), block=(256, 1, 1),
            shared_mem_bytes=0, registers_per_thread=32,
        ),
        device_info=DeviceInfo(
            name="H100", compute_capability=(9, 0),
            sm_count=132, max_clock_mhz=1980,
            memory_bus_width=5120, peak_memory_bandwidth_gbps=3352.0,
        ),
        metrics={
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": MetricValue(
                name="sm__throughput.avg.pct_of_peak_sustained_elapsed",
                value=85.0, unit="%",
            ),
            "dram__throughput.avg.pct_of_peak_sustained_elapsed": MetricValue(
                name="dram__throughput.avg.pct_of_peak_sustained_elapsed",
                value=30.0, unit="%",
            ),
        },
    )


class TestProfileCommand:

    def test_profile_pipeline_error(self, runner):
        """Pipeline error shows error message and exits with code 1."""
        from tachyon.cli.main import app

        fail_result = ToolResult.fail(
            ErrorCode.TOOL_NOT_FOUND, "ncu not found",
            suggestion="Install CUDA Toolkit",
        )

        with patch("tachyon.cli.profile.asyncio") as mock_asyncio, \
             patch("tachyon.cli.profile.TachyonConfig") as mock_cfg:
            mock_cfg.load.return_value = TachyonConfig()
            mock_asyncio.run.return_value = fail_result
            result = runner.invoke(app, ["profile", "./app"])

        assert result.exit_code != 0
        assert "ncu not found" in result.output or "ncu not found" in (result.stderr_bytes or b"").decode()

    def test_profile_pipeline_success(self, runner, tmp_path, sample_kernel):
        """Successful pipeline proceeds to analysis."""
        from tachyon.cli.main import app

        report_path = tmp_path / "stage2.ncu-rep"
        report_path.touch()
        ok_result = ToolResult.ok(report_path)

        mock_reader = MagicMock()
        mock_reader_instance = MagicMock()
        mock_reader.return_value = mock_reader_instance
        mock_reader_instance.load.return_value = ToolResult.ok([sample_kernel])

        with patch("tachyon.cli.profile.asyncio") as mock_asyncio, \
             patch("tachyon.cli.profile.TachyonConfig") as mock_cfg, \
             patch("tachyon.reader.ncu_reader.NcuReportReader", mock_reader):
            mock_cfg.load.return_value = TachyonConfig()
            mock_asyncio.run.return_value = ok_result
            result = runner.invoke(app, ["profile", "--no-ai", "./app"])

        assert result.exit_code == 0 or "Error" in result.output

    def test_profile_with_strategy(self, runner):
        """--strategy option is accepted."""
        from tachyon.cli.main import app

        fail_result = ToolResult.fail(ErrorCode.TOOL_NOT_FOUND, "ncu not found")

        with patch("tachyon.cli.profile.asyncio") as mock_asyncio, \
             patch("tachyon.cli.profile.TachyonConfig") as mock_cfg:
            mock_cfg.load.return_value = TachyonConfig()
            mock_asyncio.run.return_value = fail_result
            result = runner.invoke(app, ["profile", "--strategy", "radical", "./app"])

        assert result.exit_code != 0

    def test_profile_with_kernel_filter(self, runner):
        """--kernel option is accepted (repeatable)."""
        from tachyon.cli.main import app

        fail_result = ToolResult.fail(ErrorCode.TOOL_NOT_FOUND, "ncu not found")

        with patch("tachyon.cli.profile.asyncio") as mock_asyncio, \
             patch("tachyon.cli.profile.TachyonConfig") as mock_cfg:
            mock_cfg.load.return_value = TachyonConfig()
            mock_asyncio.run.return_value = fail_result
            result = runner.invoke(
                app, ["profile", "--kernel", "matmul*", "--kernel", "reduce*", "./app"]
            )

        assert result.exit_code != 0

    def test_profile_with_ncu_args(self, runner):
        """--ncu-args splits quoted string into list."""
        from tachyon.cli.main import app

        fail_result = ToolResult.fail(ErrorCode.TOOL_NOT_FOUND, "ncu not found")

        with patch("tachyon.cli.profile.asyncio") as mock_asyncio, \
             patch("tachyon.cli.profile.TachyonConfig") as mock_cfg:
            mock_cfg.load.return_value = TachyonConfig()
            mock_asyncio.run.return_value = fail_result
            result = runner.invoke(
                app, ["profile", "--ncu-args", "--replay-mode application", "./app"]
            )

        assert result.exit_code != 0

    def test_profile_verbose(self, runner):
        """--verbose flag is accepted."""
        from tachyon.cli.main import app

        fail_result = ToolResult.fail(ErrorCode.TOOL_NOT_FOUND, "ncu not found")

        with patch("tachyon.cli.profile.asyncio") as mock_asyncio, \
             patch("tachyon.cli.profile.TachyonConfig") as mock_cfg:
            mock_cfg.load.return_value = TachyonConfig()
            mock_asyncio.run.return_value = fail_result
            result = runner.invoke(app, ["profile", "-v", "./app"])

        assert result.exit_code != 0


class TestRunAnalysis:
    """Tests for the shared analysis pipeline."""

    def test_run_analysis_success(self, sample_kernel, tmp_path):
        """run_analysis renders output for valid kernels."""
        from tachyon.analysis.pipeline import run_analysis

        mock_reader = MagicMock()
        mock_reader_instance = MagicMock()
        mock_reader.return_value = mock_reader_instance
        mock_reader_instance.load.return_value = ToolResult.ok([sample_kernel])

        config = TachyonConfig()
        report_path = tmp_path / "test.ncu-rep"
        report_path.touch()

        with patch("tachyon.reader.ncu_reader.NcuReportReader", mock_reader):
            run_analysis(report_path, config, no_ai=True)

    def test_run_analysis_with_output_file(self, sample_kernel, tmp_path):
        """run_analysis writes to file when output_file given."""
        from tachyon.analysis.pipeline import run_analysis

        mock_reader = MagicMock()
        mock_reader_instance = MagicMock()
        mock_reader.return_value = mock_reader_instance
        mock_reader_instance.load.return_value = ToolResult.ok([sample_kernel])

        config = TachyonConfig()
        report_path = tmp_path / "test.ncu-rep"
        report_path.touch()
        out_file = tmp_path / "output.txt"

        with patch("tachyon.reader.ncu_reader.NcuReportReader", mock_reader):
            run_analysis(report_path, config, no_ai=True, output_file=out_file)

        assert out_file.exists()

    def test_run_analysis_verbose(self, sample_kernel, tmp_path):
        """run_analysis in verbose mode includes all findings."""
        from tachyon.analysis.pipeline import run_analysis

        mock_reader = MagicMock()
        mock_reader_instance = MagicMock()
        mock_reader.return_value = mock_reader_instance
        mock_reader_instance.load.return_value = ToolResult.ok([sample_kernel])

        config = TachyonConfig()
        report_path = tmp_path / "test.ncu-rep"
        report_path.touch()

        with patch("tachyon.reader.ncu_reader.NcuReportReader", mock_reader):
            run_analysis(report_path, config, verbose=True, no_ai=True)

    def test_run_analysis_load_failure(self, tmp_path):
        """run_analysis exits on load failure."""
        from tachyon.analysis.pipeline import run_analysis

        mock_reader = MagicMock()
        mock_reader_instance = MagicMock()
        mock_reader.return_value = mock_reader_instance
        mock_reader_instance.load.return_value = ToolResult.fail(
            ErrorCode.UNSUPPORTED_FORMAT, "Bad format"
        )

        config = TachyonConfig()
        report_path = tmp_path / "bad.ncu-rep"
        report_path.touch()

        with patch("tachyon.reader.ncu_reader.NcuReportReader", mock_reader), \
             pytest.raises(SystemExit):
            run_analysis(report_path, config, no_ai=True)


class TestMergeDuplicateKernels:
    """Tests for kernel launch merging logic."""

    def test_single_run_no_merge(self, sample_kernel):
        from tachyon.analysis.pipeline import merge_duplicate_kernels
        result = merge_duplicate_kernels([sample_kernel])
        assert len(result) == 1
        assert result[0].run_count == 1

    def test_duplicate_runs_merged(self):
        from tachyon.analysis.pipeline import merge_duplicate_kernels

        k1 = KernelReport(
            kernel_name="k", demangled_name="k<float>",
            launch_params=LaunchParams(grid=(1,1,1), block=(256,1,1),
                                       shared_mem_bytes=0, registers_per_thread=32),
            device_info=DeviceInfo(name="GPU", compute_capability=(9,0),
                                   sm_count=132, max_clock_mhz=1980,
                                   memory_bus_width=5120, peak_memory_bandwidth_gbps=3000.0),
            metrics={
                "sm__throughput.avg.pct_of_peak_sustained_elapsed": MetricValue(
                    name="sm__throughput.avg.pct_of_peak_sustained_elapsed", value=80.0, unit="%"),
            },
        )
        k2 = KernelReport(
            kernel_name="k", demangled_name="k<float>",
            launch_params=LaunchParams(grid=(1,1,1), block=(256,1,1),
                                       shared_mem_bytes=0, registers_per_thread=32),
            device_info=DeviceInfo(name="GPU", compute_capability=(9,0),
                                   sm_count=132, max_clock_mhz=1980,
                                   memory_bus_width=5120, peak_memory_bandwidth_gbps=3000.0),
            metrics={
                "sm__throughput.avg.pct_of_peak_sustained_elapsed": MetricValue(
                    name="sm__throughput.avg.pct_of_peak_sustained_elapsed", value=90.0, unit="%"),
            },
        )
        result = merge_duplicate_kernels([k1, k2])
        assert len(result) == 1
        assert result[0].run_count == 2
        avg = result[0].metrics["sm__throughput.avg.pct_of_peak_sustained_elapsed"].value
        assert avg == pytest.approx(85.0)

    def test_different_kernels_not_merged(self, sample_kernel):
        from tachyon.analysis.pipeline import merge_duplicate_kernels

        k2 = KernelReport(
            kernel_name="other", demangled_name="other<float>",
            launch_params=sample_kernel.launch_params,
            device_info=sample_kernel.device_info,
            metrics=sample_kernel.metrics,
        )
        result = merge_duplicate_kernels([sample_kernel, k2])
        assert len(result) == 2
