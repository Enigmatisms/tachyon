"""Unit tests for NCU error classification in ncu_profiler.py."""
from __future__ import annotations

import pytest

from tachyon.profiler.ncu_profiler import _ErrorSource, _classify_error


class TestClassifyErrorUserProgram:
    """Errors originating from the user's profiled program."""

    def test_file_not_found(self):
        stderr = "Error: No such file or directory: ./scene/xml/whiskey.xml"
        source, title, detail = _classify_error(stderr, 1)
        assert source == _ErrorSource.USER_PROGRAM
        assert "Profiled Program" in title
        assert "whiskey.xml" in detail

    def test_segfault(self):
        stderr = "Segmentation fault (core dumped)"
        source, title, detail = _classify_error(stderr, 139)
        assert source == _ErrorSource.USER_PROGRAM

    def test_segfault_exit_code_only(self):
        """Even without text, exit code 139 implies SIGSEGV."""
        source, title, detail = _classify_error("", 139)
        assert source == _ErrorSource.USER_PROGRAM
        assert "SIGSEGV" in title

    def test_sigabrt_exit_code(self):
        source, title, detail = _classify_error("", 134)
        assert source == _ErrorSource.USER_PROGRAM
        assert "SIGABRT" in title

    def test_sigkill_exit_code(self):
        source, title, detail = _classify_error("", 137)
        assert source == _ErrorSource.USER_PROGRAM
        assert "SIGKILL" in title

    def test_permission_denied(self):
        stderr = "/usr/bin/app: Permission denied"
        source, title, detail = _classify_error(stderr, 126)
        assert source == _ErrorSource.USER_PROGRAM
        assert "permission" in detail.lower()

    def test_cuda_error(self):
        stderr = "CUDA error at main.cu:42: an illegal memory access was encountered"
        source, title, detail = _classify_error(stderr, 1)
        assert source == _ErrorSource.USER_PROGRAM

    def test_missing_shared_library(self):
        stderr = "error while loading shared libraries: libcudnn.so.8: cannot open shared object file"
        source, title, detail = _classify_error(stderr, 127)
        assert source == _ErrorSource.USER_PROGRAM
        assert "shared librar" in detail.lower()

    def test_bad_alloc(self):
        stderr = "terminate called after throwing std::bad_alloc"
        source, title, detail = _classify_error(stderr, 134)
        assert source == _ErrorSource.USER_PROGRAM
        assert "bad_alloc" in detail

    def test_unrecognized_option(self):
        stderr = "app: unrecognized option '--badarg'"
        source, title, detail = _classify_error(stderr, 1)
        assert source == _ErrorSource.USER_PROGRAM

    def test_out_of_memory(self):
        stderr = "Out of memory: Killed process 12345"
        source, title, detail = _classify_error(stderr, 137)
        assert source == _ErrorSource.USER_PROGRAM

    def test_usage_message(self):
        stderr = "Usage: ./app [OPTIONS] <input_file>"
        source, title, detail = _classify_error(stderr, 1)
        assert source == _ErrorSource.USER_PROGRAM


class TestClassifyErrorNcuTool:
    """Errors originating from NCU itself."""

    def test_ncu_internal_error(self):
        stderr = "==ERROR== Internal profiler error"
        source, title, detail = _classify_error(stderr, 1)
        assert source == _ErrorSource.NCU_TOOL
        assert "NCU" in title

    def test_ncu_failed_to_attach(self):
        stderr = "==ERROR== Failed to attach to process 12345"
        source, title, detail = _classify_error(stderr, 1)
        assert source == _ErrorSource.NCU_TOOL

    def test_ncu_driver_version(self):
        stderr = "The driver version on the system is not supported"
        source, title, detail = _classify_error(stderr, 1)
        assert source == _ErrorSource.NCU_TOOL

    def test_ncu_license(self):
        stderr = "License check failed for feature NsightCompute"
        source, title, detail = _classify_error(stderr, 1)
        assert source == _ErrorSource.NCU_TOOL


class TestClassifyErrorUnknown:
    """Errors we can't classify."""

    def test_unknown_exit_code(self):
        stderr = "some unknown error output"
        source, title, detail = _classify_error(stderr, 42)
        assert source == _ErrorSource.UNKNOWN
        assert "42" in detail

    def test_empty_stderr(self):
        source, title, detail = _classify_error("", 1)
        assert source == _ErrorSource.UNKNOWN

    def test_generic_error(self):
        """Exit code 2 with unhelpful stderr."""
        source, title, detail = _classify_error("stuff happened\n", 2)
        assert source == _ErrorSource.UNKNOWN


class TestClassifyErrorPriority:
    """User program errors take priority over NCU errors when both match."""

    def test_user_error_priority(self):
        """If stderr has both user and NCU patterns, user wins."""
        stderr = (
            "==ERROR== some ncu issue\n"
            "Segmentation fault (core dumped)"
        )
        source, _, _ = _classify_error(stderr, 139)
        # User program patterns are checked first
        assert source == _ErrorSource.USER_PROGRAM
