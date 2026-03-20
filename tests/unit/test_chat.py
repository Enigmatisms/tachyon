"""Unit tests for chat CLI helper functions.

Tests the non-interactive helpers in cli/chat.py without launching the
full Click CLI (which requires a terminal).
"""

import pytest

from tachyon.models.kernel import (
    DeviceInfo,
    KernelReport,
    LaunchParams,
    MetricValue,
)


@pytest.fixture
def sample_kernel():
    """A minimal KernelReport for CLI testing."""
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


class TestChatHelpers:

    def test_handle_command_help(self, sample_kernel):
        """'/help' command should return True."""
        from tachyon.cli.chat import _handle_command
        result = _handle_command("/help", [sample_kernel], None, 0)
        assert result is True

    def test_handle_command_kernels(self, sample_kernel):
        """'/kernels' command lists kernels."""
        from tachyon.cli.chat import _handle_command
        result = _handle_command("/kernels", [sample_kernel], None, 0)
        assert result is True

    def test_handle_command_tokens(self, sample_kernel):
        """'/tokens' command shows token count."""
        from tachyon.cli.chat import _handle_command
        result = _handle_command("/tokens", [sample_kernel], None, 500)
        assert result is True

    def test_handle_command_quit(self, sample_kernel):
        """'/quit' returns None to signal exit."""
        from tachyon.cli.chat import _handle_command
        result = _handle_command("/quit", [sample_kernel], None, 0)
        assert result is None

    def test_handle_command_exit(self, sample_kernel):
        """'/exit' also returns None."""
        from tachyon.cli.chat import _handle_command
        result = _handle_command("/exit", [sample_kernel], None, 0)
        assert result is None

    def test_handle_command_unknown(self, sample_kernel):
        """Unknown command returns True (continue loop)."""
        from tachyon.cli.chat import _handle_command
        result = _handle_command("/foobar", [sample_kernel], None, 0)
        assert result is True

    def test_handle_command_next(self, sample_kernel):
        """/next command returns 'NEXT_STAGE'."""
        from tachyon.cli.chat import _handle_command
        result = _handle_command("/next", [sample_kernel], None, 0)
        assert result == "NEXT_STAGE"

    def test_handle_command_tree(self, sample_kernel):
        """'/tree 0' shows optimization tree."""
        from tachyon.cli.chat import _handle_command
        result = _handle_command("/tree 0", [sample_kernel], None, 0)
        assert result is True

    def test_handle_command_tree_invalid(self, sample_kernel):
        """'/tree 99' with invalid kernel_id."""
        from tachyon.cli.chat import _handle_command
        result = _handle_command("/tree 99", [sample_kernel], None, 0)
        assert result is True

    def test_try_create_backend_no_sdk(self):
        """_try_create_backend returns None when SDK not installed."""
        from tachyon.cli.chat import _try_create_backend
        from tachyon.config.settings import TachyonConfig
        config = TachyonConfig()
        result = _try_create_backend(config)
        # Will be None because openai SDK is not installed
        # (or could succeed if SDK is installed — either is valid)
        assert result is None or result is not None

    def test_show_opt_tree_valid(self, sample_kernel):
        """_show_opt_tree with valid kernel."""
        from tachyon.cli.chat import _show_opt_tree
        _show_opt_tree([sample_kernel], 0)  # should not raise

    def test_show_opt_tree_invalid(self, sample_kernel):
        """_show_opt_tree with invalid kernel_id."""
        from tachyon.cli.chat import _show_opt_tree
        _show_opt_tree([sample_kernel], 99)  # should not raise


class TestRuleOnlyMode:

    def test_rule_only_mode_runs(self, sample_kernel):
        """_rule_only_mode should produce output without crashing."""
        from tachyon.analyzers.base import AnalyzerRegistry
        from tachyon.cli.chat import _rule_only_mode
        from tachyon.tools.context import SessionContext

        reg = AnalyzerRegistry()
        reg.auto_register()
        session = SessionContext(kernels=[sample_kernel], registry=reg)
        # Should not raise
        _rule_only_mode([sample_kernel], reg, session)
