"""Unit tests for tachyon.utils.progress display functions."""
from __future__ import annotations

from io import StringIO
from unittest.mock import patch

from rich.console import Console

from tachyon.utils.progress import (
    NcuSpinner,
    print_error_panel,
    print_ncu_line,
    print_profile_summary,
    print_stage_header,
    print_stage_result,
)


def _capture_output(func, *args, **kwargs) -> str:
    """Capture Rich output by temporarily replacing the module-level console."""
    buf = StringIO()
    test_console = Console(file=buf, force_terminal=True, width=120, highlight=False)
    with patch("tachyon.utils.progress.console", test_console):
        func(*args, **kwargs)
    return buf.getvalue()


class TestPrintStageHeader:

    def test_basic_header(self):
        output = _capture_output(print_stage_header, 1, "ncu ... ./app")
        assert "Stage 1" in output
        assert "ncu ... ./app" in output

    def test_header_with_strategy(self):
        output = _capture_output(
            print_stage_header, 2, "ncu ... ./app", strategy="radical"
        )
        assert "radical" in output

    def test_header_with_kernels(self):
        output = _capture_output(
            print_stage_header, 2, "ncu ...",
            kernels=["matmul_kernel", "reduce_kernel"],
        )
        assert "matmul_kernel" in output
        assert "reduce_kernel" in output


class TestPrintStageResult:

    def test_success(self):
        output = _capture_output(print_stage_result, 1, 12.5, True)
        assert "12.5s" in output

    def test_failure(self):
        output = _capture_output(print_stage_result, 2, 5.3, False)
        assert "5.3s" in output


class TestPrintNcuLine:

    def test_progress_line(self):
        output = _capture_output(print_ncu_line, "==PROF== Profiling kernel: 50%")
        assert "50%" in output

    def test_error_line(self):
        output = _capture_output(print_ncu_line, "Error: some error")
        assert "Error" in output

    def test_empty_line(self):
        output = _capture_output(print_ncu_line, "")
        assert output == ""

    def test_normal_line(self):
        output = _capture_output(print_ncu_line, "==PROF== Connected to process 1234")
        assert "Connected" in output


class TestPrintErrorPanel:

    def test_error_with_suggestion(self):
        output = _capture_output(
            print_error_panel, "NCU Not Found", "ncu not at /usr/bin/ncu",
            suggestion="Install CUDA Toolkit",
        )
        assert "NCU Not Found" in output
        assert "ncu not at" in output
        assert "Install CUDA" in output

    def test_error_without_suggestion(self):
        output = _capture_output(
            print_error_panel, "Error", "something went wrong",
        )
        assert "something went wrong" in output


class TestPrintProfileSummary:

    def test_basic_summary(self):
        output = _capture_output(
            print_profile_summary,
            executable="./app",
            strategy="conservative",
        )
        assert "./app" in output
        assert "conservative" in output
        assert "Two-stage" in output

    def test_summary_with_kernel_filter(self):
        output = _capture_output(
            print_profile_summary,
            executable="./app",
            strategy="radical",
            kernels=["matmul*"],
        )
        assert "matmul" in output
        assert "skip Stage 1" in output.lower() or "Direct" in output

    def test_summary_with_ncu_set(self):
        output = _capture_output(
            print_profile_summary,
            executable="./app",
            strategy="conservative",
            ncu_set="full",
        )
        assert "full" in output
        assert "override" in output.lower()


class TestNcuSpinner:

    def test_spinner_creation(self):
        spinner = NcuSpinner(1)
        assert spinner._stage == 1

    def test_spinner_update_percentage(self):
        spinner = NcuSpinner(1)
        spinner.update('==PROF== Profiling "kern": 50%')
        assert spinner._percent == 50.0
        assert "50%" in spinner._status

    def test_spinner_update_connected(self):
        spinner = NcuSpinner(1)
        spinner.update("==PROF== Connected to process 12345")
        assert "Connected" in spinner._status

    def test_spinner_update_disconnected(self):
        spinner = NcuSpinner(1)
        spinner.update("==PROF== Disconnected from process")
        assert "Disconnected" in spinner._status

    def test_spinner_update_empty_line(self):
        spinner = NcuSpinner(1)
        original = spinner._status
        spinner.update("")
        assert spinner._status == original

    def test_spinner_update_kernel_name(self):
        """Kernel name extraction from ==PROF== Profiling "kernel_name" line."""
        spinner = NcuSpinner(1)
        spinner.update('==PROF== Profiling "matmul_kernel": 0%')
        assert spinner._kernel_name == "matmul_kernel"
        assert spinner._launch == 1

    def test_spinner_launch_count_increments(self):
        """Each Profiling line = new launch, even for same kernel."""
        spinner = NcuSpinner(1)
        spinner.update('==PROF== Profiling "kern": 0%')
        assert spinner._launch == 1
        spinner.update('....50%....100% - 11 passes')
        spinner.update('==PROF== Profiling "kern": 0%')
        assert spinner._launch == 2
        spinner.update('==PROF== Profiling "kern": 0%')
        assert spinner._launch == 3

    def test_spinner_percent_no_backward(self):
        """Percentage never goes backward within a launch."""
        spinner = NcuSpinner(1)
        spinner.update('==PROF== Profiling "kern": 0%')
        spinner.update('....50%....100% - 11 passes')
        assert spinner._percent == 100.0
        # New launch resets to 0
        spinner.update('==PROF== Profiling "kern": 0%')
        assert spinner._percent == 0.0
        spinner.update('....25%')
        assert spinner._percent == 25.0
        # Forward only within a launch
        spinner.update('....50%')
        assert spinner._percent == 50.0

    def test_spinner_stop_idempotent(self):
        """Calling stop() multiple times is safe."""
        spinner = NcuSpinner(1)
        spinner.stop()
        spinner.stop()
        spinner.stop()
        assert spinner._stopped is True

    def test_spinner_update_after_stop(self):
        """Updating after stop is a no-op, not a crash."""
        spinner = NcuSpinner(1)
        spinner.stop()
        spinner.update("==PROF== Profiling: 50%")
        # Should not crash

    def test_spinner_start_stop(self):
        """Spinner starts and stops without errors (transient mode)."""
        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True, width=120, highlight=False)
        with patch("tachyon.utils.progress.console", test_console):
            spinner = NcuSpinner(1)
            spinner.start()
            spinner.update("==PROF== Profiling: 25%")
            spinner.stop()
        # No assertions on output — just ensure no crash

    def test_spinner_del_cleanup(self):
        """__del__ cleans up even if stop() wasn't called."""
        spinner = NcuSpinner(1)
        spinner._stopped = False
        spinner.__del__()
        assert spinner._stopped is True
