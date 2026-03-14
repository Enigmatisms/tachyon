"""Unit tests for CLI commands using Click's test runner."""
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from tachyon.cli.main import app
from tachyon.errors.handler import ErrorCode, ToolResult
from tachyon.models.kernel import (
    DeviceInfo,
    KernelReport,
    LaunchParams,
    MetricValue,
)


def _make_test_report() -> KernelReport:
    """Create a synthetic KernelReport for CLI testing."""
    return KernelReport(
        kernel_name="test_kernel",
        demangled_name="test_kernel<float>",
        launch_params=LaunchParams(
            grid=(1024, 1, 1), block=(256, 1, 1),
            shared_mem_bytes=0, registers_per_thread=32,
        ),
        device_info=DeviceInfo(
            name="Test GPU", compute_capability=(8, 0),
            sm_count=108, max_clock_mhz=1410,
            memory_bus_width=5120, peak_memory_bandwidth_gbps=2039.0,
        ),
        metrics={
            "sm__throughput.avg.pct_of_peak_sustained_elapsed":
                MetricValue("sm__throughput.avg.pct_of_peak_sustained_elapsed", 75.0, "%"),
            "dram__throughput.avg.pct_of_peak_sustained_elapsed":
                MetricValue("dram__throughput.avg.pct_of_peak_sustained_elapsed", 40.0, "%"),
        },
    )


class TestCliVersion:
    def test_version_flag(self):
        runner = CliRunner()
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestCliHelp:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "AI-Powered CUDA Performance Analyzer" in result.output

    def test_analyze_help(self):
        runner = CliRunner()
        result = runner.invoke(app, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "--format" in result.output
        assert "--no-ai" in result.output
        assert "--kernel" in result.output


class TestAnalyzeCommand:
    def test_file_not_found(self):
        """analyze with nonexistent file should fail."""
        runner = CliRunner()
        result = runner.invoke(app, ["analyze", "/nonexistent/report.ncu-rep"])
        assert result.exit_code != 0

    def test_successful_analysis(self, tmp_path: Path):
        """Mock successful analysis pipeline."""
        fake_rep = tmp_path / "test.ncu-rep"
        fake_rep.write_bytes(b"fake data")

        mock_reader = MagicMock()
        mock_reader.load.return_value = ToolResult.ok([_make_test_report()])

        with patch("tachyon.reader.ncu_reader.NcuReportReader", return_value=mock_reader):
            runner = CliRunner()
            result = runner.invoke(app, ["analyze", str(fake_rep)])
            assert result.exit_code == 0
            assert "test_kernel" in result.output

    def test_load_error(self, tmp_path: Path):
        """Mock load failure shows error and suggestion."""
        fake_rep = tmp_path / "bad.ncu-rep"
        fake_rep.write_bytes(b"corrupt")

        mock_reader = MagicMock()
        mock_reader.load.return_value = ToolResult.fail(
            ErrorCode.INVALID_BINARY,
            "File corrupt",
            suggestion="Re-download the file.",
        )

        with patch("tachyon.reader.ncu_reader.NcuReportReader", return_value=mock_reader):
            runner = CliRunner()
            result = runner.invoke(app, ["analyze", str(fake_rep)])
            assert result.exit_code == 1

    def test_kernel_filter(self, tmp_path: Path):
        """--kernel flag filters by glob pattern."""
        fake_rep = tmp_path / "test.ncu-rep"
        fake_rep.write_bytes(b"data")

        reports = [
            _make_test_report(),
            KernelReport(
                kernel_name="other_kernel",
                demangled_name="other_kernel<int>",
                launch_params=LaunchParams(
                    grid=(1, 1, 1), block=(1, 1, 1),
                    shared_mem_bytes=0, registers_per_thread=16,
                ),
                device_info=DeviceInfo(
                    name="GPU", compute_capability=(8, 0), sm_count=1,
                    max_clock_mhz=1000, memory_bus_width=256,
                    peak_memory_bandwidth_gbps=100.0,
                ),
            ),
        ]

        mock_reader = MagicMock()
        mock_reader.load.return_value = ToolResult.ok(reports)

        with patch("tachyon.reader.ncu_reader.NcuReportReader", return_value=mock_reader):
            runner = CliRunner()
            result = runner.invoke(app, ["analyze", str(fake_rep), "-k", "test*"])
            assert result.exit_code == 0
            assert "test_kernel" in result.output

    def test_output_to_file(self, tmp_path: Path):
        """--output flag writes to file."""
        fake_rep = tmp_path / "test.ncu-rep"
        fake_rep.write_bytes(b"data")
        out_file = tmp_path / "output.txt"

        mock_reader = MagicMock()
        mock_reader.load.return_value = ToolResult.ok([_make_test_report()])

        with patch("tachyon.reader.ncu_reader.NcuReportReader", return_value=mock_reader):
            runner = CliRunner()
            result = runner.invoke(app, [
                "analyze", str(fake_rep), "-o", str(out_file)
            ])
            assert result.exit_code == 0
            assert out_file.exists()
            content = out_file.read_text()
            assert "test_kernel" in content

    def test_quiet_mode(self, tmp_path: Path):
        """--quiet shows only CRITICAL findings."""
        fake_rep = tmp_path / "test.ncu-rep"
        fake_rep.write_bytes(b"data")

        mock_reader = MagicMock()
        mock_reader.load.return_value = ToolResult.ok([_make_test_report()])

        with patch("tachyon.reader.ncu_reader.NcuReportReader", return_value=mock_reader):
            runner = CliRunner()
            result = runner.invoke(app, ["analyze", str(fake_rep), "-q"])
            assert result.exit_code == 0

    def test_verbose_mode(self, tmp_path: Path):
        """--verbose shows all findings including INFO."""
        fake_rep = tmp_path / "test.ncu-rep"
        fake_rep.write_bytes(b"data")

        mock_reader = MagicMock()
        mock_reader.load.return_value = ToolResult.ok([_make_test_report()])

        with patch("tachyon.reader.ncu_reader.NcuReportReader", return_value=mock_reader):
            runner = CliRunner()
            result = runner.invoke(app, ["analyze", str(fake_rep), "-v"])
            assert result.exit_code == 0
