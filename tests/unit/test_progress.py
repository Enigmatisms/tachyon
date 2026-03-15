"""Unit tests for tachyon.utils.progress display functions."""
from __future__ import annotations

from io import StringIO
from unittest.mock import patch

from rich.console import Console

from tachyon.utils.progress import (
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
