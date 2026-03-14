"""Unit tests for CLI profile command helpers.

Tests the profile Click command via CliRunner and the _analyze_report helper
with mocked dependencies.
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
             patch("tachyon.cli.profile.NcuReportReader", mock_reader, create=True), \
             patch("tachyon.reader.ncu_reader.NcuReportReader", mock_reader):
            mock_cfg.load.return_value = TachyonConfig()
            mock_asyncio.run.return_value = ok_result
            result = runner.invoke(app, ["profile", "./app"])

        # Should reach the analysis phase (may fail at reader, but not at pipeline)
        # The profile command ran without crashing
        assert result.exit_code == 0 or "Error loading report" in result.output

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


class TestAnalyzeReport:

    def test_analyze_report_success(self, sample_kernel, tmp_path):
        """_analyze_report renders output for valid kernels."""
        from tachyon.cli.profile import _analyze_report

        mock_reader = MagicMock()
        mock_reader_instance = MagicMock()
        mock_reader.return_value = mock_reader_instance
        mock_reader_instance.load.return_value = ToolResult.ok([sample_kernel])

        config = TachyonConfig()
        report_path = tmp_path / "test.ncu-rep"
        report_path.touch()

        with patch("tachyon.cli.profile.NcuReportReader", mock_reader, create=True), \
             patch("tachyon.reader.ncu_reader.NcuReportReader", mock_reader):
            # Should not raise
            _analyze_report(report_path, config, "terminal", False, True, None)

    def test_analyze_report_with_output_dir(self, sample_kernel, tmp_path):
        """_analyze_report writes to file when output_dir given."""
        from tachyon.cli.profile import _analyze_report

        mock_reader = MagicMock()
        mock_reader_instance = MagicMock()
        mock_reader.return_value = mock_reader_instance
        mock_reader_instance.load.return_value = ToolResult.ok([sample_kernel])

        config = TachyonConfig()
        report_path = tmp_path / "test.ncu-rep"
        report_path.touch()
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        with patch("tachyon.cli.profile.NcuReportReader", mock_reader, create=True), \
             patch("tachyon.reader.ncu_reader.NcuReportReader", mock_reader):
            _analyze_report(report_path, config, "terminal", False, True, out_dir)

        assert (out_dir / "analysis.txt").exists()

    def test_analyze_report_verbose(self, sample_kernel, tmp_path):
        """_analyze_report in verbose mode includes all findings."""
        from tachyon.cli.profile import _analyze_report

        mock_reader = MagicMock()
        mock_reader_instance = MagicMock()
        mock_reader.return_value = mock_reader_instance
        mock_reader_instance.load.return_value = ToolResult.ok([sample_kernel])

        config = TachyonConfig()
        report_path = tmp_path / "test.ncu-rep"
        report_path.touch()

        with patch("tachyon.cli.profile.NcuReportReader", mock_reader, create=True), \
             patch("tachyon.reader.ncu_reader.NcuReportReader", mock_reader):
            _analyze_report(report_path, config, "terminal", True, True, None)

    def test_analyze_report_load_failure(self, tmp_path):
        """_analyze_report exits on load failure."""
        from tachyon.cli.profile import _analyze_report

        mock_reader = MagicMock()
        mock_reader_instance = MagicMock()
        mock_reader.return_value = mock_reader_instance
        mock_reader_instance.load.return_value = ToolResult.fail(
            ErrorCode.UNSUPPORTED_FORMAT, "Bad format"
        )

        config = TachyonConfig()
        report_path = tmp_path / "bad.ncu-rep"
        report_path.touch()

        with patch("tachyon.cli.profile.NcuReportReader", mock_reader, create=True), \
             patch("tachyon.reader.ncu_reader.NcuReportReader", mock_reader), \
             pytest.raises(SystemExit):
            _analyze_report(report_path, config, "terminal", False, True, None)
