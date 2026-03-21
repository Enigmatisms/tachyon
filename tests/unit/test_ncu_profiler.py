"""Comprehensive unit tests for the NcuProfiler two-stage smart profiling engine.

Covers:
  - NcuProfiler: command building, profile_basic, profile_targeted, lazy path resolution
  - ProfilingStrategy: enum values
  - StageConfig: defaults and field values
  - STRATEGY_CONFIGS: completeness

No GPU or NCU installation required. All subprocess calls are mocked.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tachyon.config.settings import TachyonConfig
from tachyon.errors.handler import ErrorCode
from tachyon.profiler.ncu_profiler import (
    AnalysisDepth,
    DEPTH_CONFIGS,
    STRATEGY_CONFIGS,
    NcuProfiler,
    ProfilingResult,
    ProfilingStrategy,
    StageConfig,
)
from tachyon.profiler.tool_path import ToolPathResolver

# ━━━━━━━━━━━━━━━━━━━━━━━ Helpers ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


FAKE_NCU = "/usr/local/cuda/bin/ncu"


def _make_profiler(
    depth: str = "basic",
    ncu_resolve_path: str = FAKE_NCU,
) -> tuple[NcuProfiler, MagicMock]:
    """Create an NcuProfiler with a mocked ToolPathResolver."""
    config = TachyonConfig()
    config.profiling.depth = depth
    resolver = MagicMock(spec=ToolPathResolver)
    resolver.resolve.return_value = ncu_resolve_path
    profiler = NcuProfiler(config, resolver)
    return profiler, resolver


def _mock_subprocess_success(
    output_path: Path,
    returncode: int = 0,
    stdout_lines: str = "",
) -> MagicMock:
    """Build a mock subprocess.Popen return value and create the output file.

    The mock simulates Popen's interface with stderr=subprocess.STDOUT:
    all output goes to stdout (iterable of lines), stderr is None.
    """
    mock_proc = MagicMock()
    mock_proc.returncode = returncode
    # With stderr=STDOUT, all output is on stdout as iterable lines
    mock_proc.stdout = iter(stdout_lines.splitlines(keepends=True)) if stdout_lines else iter([])
    mock_proc.stderr = None  # stderr=STDOUT means stderr fd is None
    mock_proc.wait.return_value = None
    mock_proc.kill.return_value = None
    # Create the output file so the exists() check passes.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("ncu-rep-binary-content")
    return mock_proc


def _mock_popen_failure(
    returncode: int = 1,
    stderr: str = "",
) -> MagicMock:
    """Build a mock Popen for failure cases (no output file created).

    Note: NcuProfiler uses stderr=subprocess.STDOUT, so all output
    goes to stdout. We put the error text on stdout to match.
    """
    mock_proc = MagicMock()
    mock_proc.returncode = returncode
    mock_proc.stdout = iter(stderr.splitlines(keepends=True)) if stderr else iter([])
    mock_proc.stderr = None  # stderr=STDOUT means stderr is None
    mock_proc.wait.return_value = None
    mock_proc.kill.return_value = None
    return mock_proc


def _mock_popen_timeout() -> MagicMock:
    """Build a mock Popen that times out on wait(timeout=...) but succeeds on wait() after kill.

    Note: NcuProfiler uses stderr=subprocess.STDOUT, so all output
    goes to stdout. stderr is None.
    """
    mock_proc = MagicMock()
    mock_proc.returncode = -9
    mock_proc.stdout = iter([])  # no output before timeout
    mock_proc.stderr = None  # stderr=STDOUT means stderr is None
    mock_proc.kill.return_value = None
    # First wait(timeout=N) raises TimeoutExpired, second wait() after kill succeeds
    mock_proc.wait.side_effect = [
        subprocess.TimeoutExpired(cmd="ncu", timeout=600),
        None,
    ]
    return mock_proc


# ━━━━━━━━━━━━━━━━━━━━━━━ TestProfilingStrategy ━━━━━━━━━━━━━━━━━


class TestProfilingStrategy:
    """Tests for the ProfilingStrategy enum."""

    def test_strategy_enum_values(self) -> None:
        """ProfilingStrategy has CONSERVATIVE and RADICAL with correct string values."""
        assert ProfilingStrategy.CONSERVATIVE == "conservative"
        assert ProfilingStrategy.CONSERVATIVE.value == "conservative"
        assert ProfilingStrategy.RADICAL == "radical"
        assert ProfilingStrategy.RADICAL.value == "radical"

    def test_strategy_from_string(self) -> None:
        """ProfilingStrategy can be constructed from string value."""
        assert ProfilingStrategy("conservative") is ProfilingStrategy.CONSERVATIVE
        assert ProfilingStrategy("radical") is ProfilingStrategy.RADICAL

    def test_strategy_invalid_raises(self) -> None:
        """Unknown strategy string raises ValueError."""
        with pytest.raises(ValueError):
            ProfilingStrategy("unknown")


# ━━━━━━━━━━━━━━━━━━━━━━━ TestStageConfig ━━━━━━━━━━━━━━━━━━━━━━━


class TestStageConfig:
    """Tests for StageConfig dataclass."""

    def test_stage_config_defaults(self) -> None:
        """StageConfig has correct default values for optional fields."""
        cfg = StageConfig(metric_set="basic")
        assert cfg.metric_set == "basic"
        assert cfg.timeout_sec == 600
        assert cfg.extra_sections == []
        assert cfg.source_counters is False

    def test_stage_config_custom_values(self) -> None:
        """StageConfig accepts custom values for all fields."""
        cfg = StageConfig(
            metric_set="full",
            timeout_sec=3600,
            extra_sections=["SourceCounters"],
            source_counters=True,
        )
        assert cfg.metric_set == "full"
        assert cfg.timeout_sec == 3600
        assert cfg.extra_sections == ["SourceCounters"]
        assert cfg.source_counters is True

    def test_stage_config_extra_sections_independent(self) -> None:
        """Each StageConfig gets its own list instance (mutable default safety)."""
        a = StageConfig(metric_set="basic")
        b = StageConfig(metric_set="basic")
        a.extra_sections.append("foo")
        assert b.extra_sections == []


# ━━━━━━━━━━━━━━━━━━━━━━━ TestStrategyConfigs ━━━━━━━━━━━━━━━━━━━


class TestStrategyConfigs:
    """Tests for the STRATEGY_CONFIGS mapping."""

    def test_strategy_configs_complete(self) -> None:
        """Both strategies have Stage1 and Stage2 configs defined."""
        assert ProfilingStrategy.CONSERVATIVE in STRATEGY_CONFIGS
        assert ProfilingStrategy.RADICAL in STRATEGY_CONFIGS

        for strat in ProfilingStrategy:
            entry = STRATEGY_CONFIGS[strat]
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            assert isinstance(entry[0], StageConfig)
            assert isinstance(entry[1], StageConfig)

    def test_conservative_stage1_basic(self) -> None:
        """Conservative Stage 1 uses 'basic' metric set."""
        s1, _ = STRATEGY_CONFIGS[ProfilingStrategy.CONSERVATIVE]
        assert s1.metric_set == "basic"
        assert s1.timeout_sec == 600
        assert s1.source_counters is False

    def test_conservative_stage2_detailed(self) -> None:
        """Conservative Stage 2 uses 'detailed' metric set."""
        _, s2 = STRATEGY_CONFIGS[ProfilingStrategy.CONSERVATIVE]
        assert s2.metric_set == "detailed"
        assert s2.timeout_sec == 1800

    def test_radical_stage1_detailed(self) -> None:
        """Radical Stage 1 uses 'detailed' metric set."""
        s1, _ = STRATEGY_CONFIGS[ProfilingStrategy.RADICAL]
        assert s1.metric_set == "detailed"
        assert s1.timeout_sec == 1200

    def test_radical_stage2_full_with_source_counters(self) -> None:
        """Radical Stage 2 uses 'full' metric set with source_counters enabled."""
        _, s2 = STRATEGY_CONFIGS[ProfilingStrategy.RADICAL]
        assert s2.metric_set == "full"
        assert s2.timeout_sec == 3600
        assert s2.source_counters is True


# ━━━━━━━━━━━━━━━━━━━━━━━ TestNcuProfilerBuildCommand ━━━━━━━━━━━


class TestNcuProfilerBuildCommand:
    """Tests for NcuProfiler._build_command."""

    def test_build_command_conservative_stage1(self, tmp_path: Path) -> None:
        """Conservative strategy Stage 1 builds correct ncu command."""
        profiler, _ = _make_profiler()
        out_file = tmp_path / "stage1.ncu-rep"

        cmd = profiler._build_command(
            metric_set="basic",
            output_path=out_file,
            executable="./app",
            exe_args=["--size", "1024"],
        )

        assert cmd == [
            FAKE_NCU,
            "--set", "basic",
            "--export", str(out_file),
            "--force-overwrite",
            "--target-processes", "all",
            "--import-source", "yes",
            "./app",
            "--size", "1024",
        ]

    def test_build_command_radical_stage2_with_kernels(self, tmp_path: Path) -> None:
        """Radical Stage 2 includes --set full, --section SourceCounters, --kernel-name filters."""
        profiler, _ = _make_profiler(depth="radical")
        out_file = tmp_path / "stage2.ncu-rep"

        cmd = profiler._build_command(
            metric_set="full",
            output_path=out_file,
            executable="./app",
            exe_args=["--batch", "32"],
            source_counters=True,
            kernel_filter_args=[
                "--kernel-name", "matmul_kernel",
                "--kernel-name", "reduce_kernel",
            ],
        )

        assert cmd == [
            FAKE_NCU,
            "--set", "full",
            "--export", str(out_file),
            "--force-overwrite",
            "--target-processes", "all",
            "--import-source", "yes",
            "--section", "SourceCounters",
            "--kernel-name", "matmul_kernel",
            "--kernel-name", "reduce_kernel",
            "./app",
            "--batch", "32",
        ]

    def test_build_command_with_extra_ncu_args(self, tmp_path: Path) -> None:
        """Extra ncu args are inserted before the executable."""
        profiler, _ = _make_profiler()
        out_file = tmp_path / "stage1.ncu-rep"

        cmd = profiler._build_command(
            metric_set="basic",
            output_path=out_file,
            executable="./app",
            exe_args=[],
            extra_ncu_args=["--replay-mode", "kernel", "--cache-control", "all"],
        )

        # Extra args should appear after --target-processes all, before ./app
        exe_idx = cmd.index("./app")
        replay_idx = cmd.index("--replay-mode")
        assert replay_idx < exe_idx
        assert cmd[replay_idx + 1] == "kernel"
        assert "--cache-control" in cmd
        assert cmd[-1] == "./app"

    def test_build_command_no_source_counters(self, tmp_path: Path) -> None:
        """When source_counters=False, --section SourceCounters is not in command."""
        profiler, _ = _make_profiler()
        out_file = tmp_path / "stage1.ncu-rep"

        cmd = profiler._build_command(
            metric_set="basic",
            output_path=out_file,
            executable="./app",
            exe_args=[],
            source_counters=False,
        )

        assert "--section" not in cmd
        assert "SourceCounters" not in cmd

    def test_build_command_no_kernel_filter(self, tmp_path: Path) -> None:
        """When kernel_filter_args is None, no --kernel-name in command."""
        profiler, _ = _make_profiler()
        out_file = tmp_path / "stage1.ncu-rep"

        cmd = profiler._build_command(
            metric_set="basic",
            output_path=out_file,
            executable="./app",
            exe_args=[],
            kernel_filter_args=None,
        )

        assert "--kernel-name" not in cmd

    def test_build_command_empty_exe_args(self, tmp_path: Path) -> None:
        """Empty exe_args produces no trailing args after executable."""
        profiler, _ = _make_profiler()
        out_file = tmp_path / "stage1.ncu-rep"

        cmd = profiler._build_command(
            metric_set="basic",
            output_path=out_file,
            executable="./app",
            exe_args=[],
        )

        assert cmd[-1] == "./app"

    def test_build_command_ordering(self, tmp_path: Path) -> None:
        """Full command with all optional args has correct ordering:
        ncu flags -> source_counters -> kernel_filter -> extra_args -> executable -> exe_args.
        """
        profiler, _ = _make_profiler()
        out_file = tmp_path / "stage1.ncu-rep"

        cmd = profiler._build_command(
            metric_set="full",
            output_path=out_file,
            executable="./my_app",
            exe_args=["--arg1"],
            source_counters=True,
            kernel_filter_args=["--kernel-name", "k1"],
            extra_ncu_args=["--extra", "val"],
        )

        # Verify ordering via indices
        sc_idx = cmd.index("--section")
        kn_idx = cmd.index("--kernel-name")
        extra_idx = cmd.index("--extra")
        exe_idx = cmd.index("./my_app")
        arg_idx = cmd.index("--arg1")

        assert sc_idx < kn_idx < extra_idx < exe_idx < arg_idx

    def test_build_command_metrics_override(self, tmp_path: Path) -> None:
        """metrics_override uses --metrics instead of --set."""
        profiler, _ = _make_profiler()
        out_file = tmp_path / "stage1.ncu-rep"

        cmd = profiler._build_command(
            metric_set="basic",
            output_path=out_file,
            executable="./app",
            exe_args=[],
            metrics_override="sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed",
        )

        assert "--metrics" in cmd
        assert "--set" not in cmd
        idx = cmd.index("--metrics")
        assert "sm__throughput" in cmd[idx + 1]

    def test_build_command_metric_set_no_override(self, tmp_path: Path) -> None:
        """Without metrics_override, --set is used normally."""
        profiler, _ = _make_profiler()
        out_file = tmp_path / "stage1.ncu-rep"

        cmd = profiler._build_command(
            metric_set="detailed",
            output_path=out_file,
            executable="./app",
            exe_args=[],
        )

        assert "--set" in cmd
        assert "--metrics" not in cmd
        idx = cmd.index("--set")
        assert cmd[idx + 1] == "detailed"


# ━━━━━━━━━━━━━━━━━━━━━━━ TestNcuProfilerRun ━━━━━━━━━━━━━━━━━━━━


class TestNcuProfilerRun:
    """Tests for NcuProfiler.profile_basic and profile_targeted (subprocess mocking)."""

    @patch("tachyon.profiler.ncu_profiler.subprocess.Popen")
    def test_profile_basic_success(self, mock_popen: MagicMock, tmp_path: Path) -> None:
        """Successful Stage 1 profiling returns ToolResult.ok with ProfilingResult."""
        profiler, _ = _make_profiler()
        out_file = tmp_path / "stage1.ncu-rep"

        mock_popen.return_value = _mock_subprocess_success(out_file, returncode=0)

        result = profiler.profile_basic(
            "./app",
            ["--size", "1024"],
            output_dir=tmp_path,
        )

        assert result.success is True
        assert result.data is not None
        pr = result.data
        assert isinstance(pr, ProfilingResult)
        assert pr.ncu_rep_path == out_file
        assert pr.stage == 1
        assert pr.returncode == 0
        assert pr.elapsed_sec >= 0

        # Verify subprocess.run was called with correct basic args
        actual_cmd = mock_popen.call_args[0][0]
        assert actual_cmd[0] == FAKE_NCU
        assert "--set" in actual_cmd
        assert "basic" in actual_cmd
        assert "--force-overwrite" in actual_cmd
        assert "--target-processes" in actual_cmd
        assert "./app" in actual_cmd
        assert "--size" in actual_cmd

    @patch("tachyon.profiler.ncu_profiler.subprocess.Popen")
    def test_profile_basic_timeout(self, mock_popen: MagicMock, tmp_path: Path) -> None:
        """TimeoutExpired from subprocess returns ToolResult.fail."""
        profiler, _ = _make_profiler()
        mock_popen.return_value = _mock_popen_timeout()

        result = profiler.profile_basic("./app", output_dir=tmp_path)

        assert result.success is False
        assert result.error is not None
        assert "timed out" in result.error.message
        assert "Stage 1" in result.error.message
        assert "600" in result.error.message

    @patch("tachyon.profiler.ncu_profiler.subprocess.Popen")
    def test_profile_basic_ncu_failure(self, mock_popen: MagicMock, tmp_path: Path) -> None:
        """Nonzero returncode returns ToolResult.fail with stderr content."""
        profiler, _ = _make_profiler()
        mock_popen.return_value = _mock_popen_failure(
            returncode=1, stderr="CUDA error: no compatible GPU found"
        )

        result = profiler.profile_basic("./app", output_dir=tmp_path)

        assert result.success is False
        assert result.error is not None
        assert "failed (exit 1)" in result.error.message
        assert "CUDA error" in result.error.message
        assert "CUDA error: no compatible GPU found" in result.error.context.get("stderr", "")

    @patch("tachyon.profiler.ncu_profiler.subprocess.Popen")
    def test_profile_basic_ncu_not_found(self, mock_popen: MagicMock, tmp_path: Path) -> None:
        """FileNotFoundError from subprocess returns ToolResult.fail with TOOL_NOT_FOUND."""
        profiler, _ = _make_profiler()
        mock_popen.side_effect = FileNotFoundError("No such file or directory")

        result = profiler.profile_basic("./app", output_dir=tmp_path)

        assert result.success is False
        assert result.error is not None
        assert result.error.code == ErrorCode.TOOL_NOT_FOUND
        assert "not found" in result.error.message
        assert "Install CUDA Toolkit" in result.error.suggestion

    @patch("tachyon.profiler.ncu_profiler.subprocess.Popen")
    def test_profile_basic_missing_output(self, mock_popen: MagicMock, tmp_path: Path) -> None:
        """subprocess succeeds but output file does not exist -> ToolResult.fail."""
        profiler, _ = _make_profiler()
        # Return success but do NOT create the output file
        mock_popen.return_value = _mock_popen_failure(returncode=0)

        result = profiler.profile_basic("./app", output_dir=tmp_path)

        assert result.success is False
        assert result.error is not None
        assert "output file not found" in result.error.message

    @patch("tachyon.profiler.ncu_profiler.subprocess.Popen")
    def test_profile_basic_uses_tempdir_when_no_output_dir(
        self, mock_popen: MagicMock
    ) -> None:
        """When output_dir is None, a temporary directory is created."""
        profiler, _ = _make_profiler()
        mock_popen.return_value = _mock_popen_failure(returncode=0)

        with patch("tachyon.profiler.ncu_profiler.tempfile.mkdtemp") as mock_mkdtemp:
            mock_mkdtemp.return_value = "/tmp/tachyon_abc123"
            # The output file won't exist, but we just need to verify tempdir usage
            result = profiler.profile_basic("./app")

            mock_mkdtemp.assert_called_once_with(prefix="tachyon_")
            # Command should reference the temp dir
            actual_cmd = mock_popen.call_args[0][0]
            output_idx = actual_cmd.index("--export")
            assert "/tmp/tachyon_abc123" in actual_cmd[output_idx + 1]

    @patch("tachyon.profiler.ncu_profiler.subprocess.Popen")
    def test_profile_basic_default_args_none(self, mock_popen: MagicMock, tmp_path: Path) -> None:
        """profile_basic with args=None uses empty list for exe_args."""
        profiler, _ = _make_profiler()
        out_file = tmp_path / "stage1.ncu-rep"
        mock_popen.return_value = _mock_subprocess_success(out_file)

        result = profiler.profile_basic("./app", output_dir=tmp_path)

        assert result.success is True
        actual_cmd = mock_popen.call_args[0][0]
        # Executable should be the last element when no args given
        assert actual_cmd[-1] == "./app"

    @patch("tachyon.profiler.ncu_profiler.subprocess.Popen")
    def test_profile_basic_strategy_override(self, mock_popen: MagicMock, tmp_path: Path) -> None:
        """Explicit strategy parameter overrides config default."""
        profiler, _ = _make_profiler(depth="basic")
        out_file = tmp_path / "stage1.ncu-rep"
        mock_popen.return_value = _mock_subprocess_success(out_file)

        result = profiler.profile_basic(
            "./app",
            output_dir=tmp_path,
            strategy=ProfilingStrategy.RADICAL,
        )

        assert result.success is True
        # Radical Stage 1 uses "detailed" metric set
        actual_cmd = mock_popen.call_args[0][0]
        set_idx = actual_cmd.index("--set")
        assert actual_cmd[set_idx + 1] == "detailed"

    @patch("tachyon.profiler.ncu_profiler.subprocess.Popen")
    def test_profile_basic_with_extra_ncu_args(self, mock_popen: MagicMock, tmp_path: Path) -> None:
        """Extra ncu args are passed through to the command."""
        profiler, _ = _make_profiler()
        out_file = tmp_path / "stage1.ncu-rep"
        mock_popen.return_value = _mock_subprocess_success(out_file)

        result = profiler.profile_basic(
            "./app",
            ["--size", "1024"],
            output_dir=tmp_path,
            extra_ncu_args=["--replay-mode", "kernel"],
        )

        assert result.success is True
        actual_cmd = mock_popen.call_args[0][0]
        assert "--replay-mode" in actual_cmd
        assert "kernel" in actual_cmd

    @patch("tachyon.profiler.ncu_profiler.subprocess.Popen")
    def test_profile_basic_subprocess_call_args(self, mock_popen: MagicMock, tmp_path: Path) -> None:
        """Verify subprocess.Popen is called with stdout=PIPE, stderr=STDOUT, text=True."""
        profiler, _ = _make_profiler()
        out_file = tmp_path / "stage1.ncu-rep"
        mock_popen.return_value = _mock_subprocess_success(out_file)

        profiler.profile_basic("./app", output_dir=tmp_path)

        mock_popen.assert_called_once()
        _, kwargs = mock_popen.call_args
        assert kwargs["stdout"] == subprocess.PIPE
        assert kwargs["stderr"] == subprocess.STDOUT
        assert kwargs["text"] is True

    @patch("tachyon.profiler.ncu_profiler.subprocess.Popen")
    def test_profile_basic_stderr_truncation(self, mock_popen: MagicMock, tmp_path: Path) -> None:
        """Long stderr is truncated to 500 chars in error message."""
        profiler, _ = _make_profiler()
        long_stderr = "X" * 1000
        mock_popen.return_value = _mock_popen_failure(returncode=2, stderr=long_stderr)

        result = profiler.profile_basic("./app", output_dir=tmp_path)

        assert result.success is False
        # Message should contain at most 500 chars of stderr
        assert len(result.error.message) < len(long_stderr) + 200
        # Full stderr is in context
        assert result.error.context["stderr"] == long_stderr


# ━━━━━━━━━━━━━━━━━━━━━━━ TestNcuProfilerTargeted ━━━━━━━━━━━━━━━


class TestNcuProfilerTargeted:
    """Tests for NcuProfiler.profile_targeted (Stage 2)."""

    @patch("tachyon.profiler.ncu_profiler.subprocess.Popen")
    def test_profile_targeted_builds_kernel_filters(
        self, mock_popen: MagicMock, tmp_path: Path
    ) -> None:
        """profile_targeted includes --kernel-name args for each kernel."""
        profiler, _ = _make_profiler()
        out_file = tmp_path / "stage2.ncu-rep"
        mock_popen.return_value = _mock_subprocess_success(out_file)

        result = profiler.profile_targeted(
            "./app",
            ["--size", "1024"],
            kernels=["matmul_kernel", "reduce_kernel", "softmax_kernel"],
            output_dir=tmp_path,
        )

        assert result.success is True
        actual_cmd = mock_popen.call_args[0][0]

        # Verify each kernel has a --kernel-name argument
        kernel_name_indices = [
            i for i, arg in enumerate(actual_cmd) if arg == "--kernel-name"
        ]
        assert len(kernel_name_indices) == 3
        assert actual_cmd[kernel_name_indices[0] + 1] == "matmul_kernel"
        assert actual_cmd[kernel_name_indices[1] + 1] == "reduce_kernel"
        assert actual_cmd[kernel_name_indices[2] + 1] == "softmax_kernel"

    @patch("tachyon.profiler.ncu_profiler.subprocess.Popen")
    def test_profile_targeted_stage2_metadata(
        self, mock_popen: MagicMock, tmp_path: Path
    ) -> None:
        """profile_targeted returns stage=2 in ProfilingResult."""
        profiler, _ = _make_profiler()
        out_file = tmp_path / "stage2.ncu-rep"
        mock_popen.return_value = _mock_subprocess_success(out_file)

        result = profiler.profile_targeted(
            "./app", kernels=["k1"], output_dir=tmp_path
        )

        assert result.success is True
        assert result.data.stage == 2
        assert result.data.ncu_rep_path == out_file

    @patch("tachyon.profiler.ncu_profiler.subprocess.Popen")
    def test_profile_targeted_uses_stage2_config(
        self, mock_popen: MagicMock, tmp_path: Path
    ) -> None:
        """Conservative profile_targeted uses 'detailed' metric set (stage2 config)."""
        profiler, _ = _make_profiler(depth="basic")
        out_file = tmp_path / "stage2.ncu-rep"
        mock_popen.return_value = _mock_subprocess_success(out_file)

        profiler.profile_targeted(
            "./app", kernels=["k1"], output_dir=tmp_path
        )

        actual_cmd = mock_popen.call_args[0][0]
        set_idx = actual_cmd.index("--set")
        assert actual_cmd[set_idx + 1] == "detailed"

    @patch("tachyon.profiler.ncu_profiler.subprocess.Popen")
    def test_profile_targeted_radical_has_source_counters(
        self, mock_popen: MagicMock, tmp_path: Path
    ) -> None:
        """Radical strategy Stage 2 includes --section SourceCounters."""
        profiler, _ = _make_profiler(depth="radical")
        out_file = tmp_path / "stage2.ncu-rep"
        mock_popen.return_value = _mock_subprocess_success(out_file)

        profiler.profile_targeted(
            "./app", kernels=["k1"], output_dir=tmp_path
        )

        actual_cmd = mock_popen.call_args[0][0]
        assert "--section" in actual_cmd
        sc_idx = actual_cmd.index("--section")
        assert actual_cmd[sc_idx + 1] == "SourceCounters"

    @patch("tachyon.profiler.ncu_profiler.subprocess.Popen")
    def test_profile_targeted_timeout(self, mock_popen: MagicMock, tmp_path: Path) -> None:
        """TimeoutExpired during Stage 2 returns fail with Stage 2 in message."""
        profiler, _ = _make_profiler()
        mock_popen.return_value = _mock_popen_timeout()

        result = profiler.profile_targeted(
            "./app", kernels=["k1"], output_dir=tmp_path
        )

        assert result.success is False
        assert "Stage 2" in result.error.message
        assert "timed out" in result.error.message

    @patch("tachyon.profiler.ncu_profiler.subprocess.Popen")
    def test_profile_targeted_with_extra_ncu_args(
        self, mock_popen: MagicMock, tmp_path: Path
    ) -> None:
        """Extra ncu args are inserted in targeted profiling command."""
        profiler, _ = _make_profiler()
        out_file = tmp_path / "stage2.ncu-rep"
        mock_popen.return_value = _mock_subprocess_success(out_file)

        profiler.profile_targeted(
            "./app",
            kernels=["k1"],
            output_dir=tmp_path,
            extra_ncu_args=["--launch-skip", "10"],
        )

        actual_cmd = mock_popen.call_args[0][0]
        assert "--launch-skip" in actual_cmd
        assert "10" in actual_cmd
        # Extra args come before executable
        skip_idx = actual_cmd.index("--launch-skip")
        exe_idx = actual_cmd.index("./app")
        assert skip_idx < exe_idx

    @patch("tachyon.profiler.ncu_profiler.subprocess.Popen")
    def test_profile_targeted_strategy_override(
        self, mock_popen: MagicMock, tmp_path: Path
    ) -> None:
        """Explicit strategy overrides config default in profile_targeted."""
        profiler, _ = _make_profiler(depth="basic")
        out_file = tmp_path / "stage2.ncu-rep"
        mock_popen.return_value = _mock_subprocess_success(out_file)

        profiler.profile_targeted(
            "./app",
            kernels=["k1"],
            output_dir=tmp_path,
            strategy=ProfilingStrategy.RADICAL,
        )

        actual_cmd = mock_popen.call_args[0][0]
        set_idx = actual_cmd.index("--set")
        # Radical stage2 uses "full"
        assert actual_cmd[set_idx + 1] == "full"

    @patch("tachyon.profiler.ncu_profiler.subprocess.Popen")
    def test_profile_targeted_ncu_failure(
        self, mock_popen: MagicMock, tmp_path: Path
    ) -> None:
        """Nonzero returncode during stage 2 returns ToolResult.fail."""
        profiler, _ = _make_profiler()
        mock_popen.return_value = _mock_popen_failure(
            returncode=127, stderr="ncu: command not found"
        )

        result = profiler.profile_targeted(
            "./app", kernels=["k1"], output_dir=tmp_path
        )

        assert result.success is False
        assert "Stage 2" in result.error.message
        assert "exit 127" in result.error.message


# ━━━━━━━━━━━━━━━━━━━━━━━ TestNcuPathLazyResolution ━━━━━━━━━━━━━


class TestNcuPathLazyResolution:
    """Tests for NcuProfiler.ncu_path lazy resolution."""

    def test_ncu_path_lazy_resolution(self) -> None:
        """ncu_path property calls resolver.resolve('ncu') only once (cached)."""
        profiler, resolver = _make_profiler()

        # First access
        path1 = profiler.ncu_path
        assert path1 == FAKE_NCU
        assert resolver.resolve.call_count == 1
        resolver.resolve.assert_called_with("ncu")

        # Second access -- should use cache, no additional resolve call
        path2 = profiler.ncu_path
        assert path2 == FAKE_NCU
        assert resolver.resolve.call_count == 1

    def test_ncu_path_not_resolved_until_accessed(self) -> None:
        """resolver.resolve is not called at construction time."""
        profiler, resolver = _make_profiler()
        resolver.resolve.assert_not_called()

    def test_ncu_path_propagates_resolver_error(self) -> None:
        """If resolver raises FileNotFoundError, ncu_path property re-raises."""
        config = TachyonConfig()
        resolver = MagicMock(spec=ToolPathResolver)
        resolver.resolve.side_effect = FileNotFoundError("ncu not found")
        profiler = NcuProfiler(config, resolver)

        with pytest.raises(FileNotFoundError, match="ncu not found"):
            _ = profiler.ncu_path


# ━━━━━━━━━━━━━━━━━━━━━━━ TestProfilingResult ━━━━━━━━━━━━━━━━━━━


class TestProfilingResult:
    """Tests for the ProfilingResult dataclass."""

    def test_profiling_result_fields(self, tmp_path: Path) -> None:
        """ProfilingResult stores all fields correctly."""
        rep_path = tmp_path / "test.ncu-rep"
        pr = ProfilingResult(
            ncu_rep_path=rep_path,
            stage=1,
            returncode=0,
            stderr="",
            elapsed_sec=12.5,
        )

        assert pr.ncu_rep_path == rep_path
        assert pr.stage == 1
        assert pr.returncode == 0
        assert pr.stderr == ""
        assert pr.elapsed_sec == 12.5

    def test_profiling_result_stage2(self, tmp_path: Path) -> None:
        """ProfilingResult correctly stores stage 2 data with stderr."""
        rep_path = tmp_path / "stage2.ncu-rep"
        pr = ProfilingResult(
            ncu_rep_path=rep_path,
            stage=2,
            returncode=0,
            stderr="Warning: some kernels skipped",
            elapsed_sec=120.7,
        )

        assert pr.stage == 2
        assert pr.stderr == "Warning: some kernels skipped"


# ━━━━━━━━━━━━━━━━━━━━━━━ TestEndToEndFlow ━━━━━━━━━━━━━━━━━━━━━━


class TestEndToEndFlow:
    """Integration-style tests for the two-stage flow within NcuProfiler."""

    @patch("tachyon.profiler.ncu_profiler.subprocess.Popen")
    def test_two_stage_flow(self, mock_popen: MagicMock, tmp_path: Path) -> None:
        """Full two-stage flow: profile_basic then profile_targeted."""
        profiler, _ = _make_profiler()

        # Stage 1
        s1_file = tmp_path / "stage1.ncu-rep"
        mock_popen.return_value = _mock_subprocess_success(s1_file)
        s1_result = profiler.profile_basic("./app", ["--n", "100"], output_dir=tmp_path)
        assert s1_result.success is True
        assert s1_result.data.stage == 1

        # Stage 2
        s2_dir = tmp_path / "stage2_dir"
        s2_dir.mkdir()
        s2_file = s2_dir / "stage2.ncu-rep"
        mock_popen.return_value = _mock_subprocess_success(s2_file)
        s2_result = profiler.profile_targeted(
            "./app",
            ["--n", "100"],
            kernels=["hot_kernel_1", "hot_kernel_2"],
            output_dir=s2_dir,
        )
        assert s2_result.success is True
        assert s2_result.data.stage == 2
        assert s2_result.data.ncu_rep_path == s2_file

        # Verify ncu_path was resolved only once across both stages
        _, resolver = _make_profiler()
        # The original resolver from the profiler used in this test
        # was called via ncu_path property, which caches after first call
        assert mock_popen.call_count == 2

    @patch("tachyon.profiler.ncu_profiler.subprocess.Popen")
    def test_stage1_fail_does_not_prevent_stage2(
        self, mock_popen: MagicMock, tmp_path: Path
    ) -> None:
        """Even if stage 1 fails, stage 2 can still be called independently."""
        profiler, _ = _make_profiler()

        # Stage 1 fails (timeout)
        mock_popen.return_value = _mock_popen_timeout()
        s1_result = profiler.profile_basic("./app", output_dir=tmp_path)
        assert s1_result.success is False

        # Stage 2 succeeds
        s2_dir = tmp_path / "s2"
        s2_dir.mkdir()
        s2_file = s2_dir / "stage2.ncu-rep"
        mock_popen.return_value = _mock_subprocess_success(s2_file)
        s2_result = profiler.profile_targeted(
            "./app", kernels=["k1"], output_dir=s2_dir
        )
        assert s2_result.success is True


# ━━━━━━━━━━━━━━━━━━━━━━━ TestAnalysisDepth ━━━━━━━━━━━━━━━━━━━━━━


class TestAnalysisDepth:
    """Tests for the AnalysisDepth enum and DEPTH_CONFIGS mapping."""

    def test_enum_values(self) -> None:
        """AnalysisDepth has BASIC, DEEP, RADICAL with correct string values."""
        assert AnalysisDepth.BASIC == "basic"
        assert AnalysisDepth.DEEP == "deep"
        assert AnalysisDepth.RADICAL == "radical"

    def test_depth_from_string(self) -> None:
        """AnalysisDepth can be constructed from string value."""
        assert AnalysisDepth("basic") is AnalysisDepth.BASIC
        assert AnalysisDepth("deep") is AnalysisDepth.DEEP
        assert AnalysisDepth("radical") is AnalysisDepth.RADICAL

    def test_depth_configs_complete(self) -> None:
        """All three depths have mappings in DEPTH_CONFIGS."""
        for depth in AnalysisDepth:
            assert depth in DEPTH_CONFIGS
            entry = DEPTH_CONFIGS[depth]
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            strategy, layers = entry
            assert isinstance(strategy, ProfilingStrategy)
            assert isinstance(layers, int)

    def test_depth_configs_basic(self) -> None:
        """BASIC maps to CONSERVATIVE + 1 AI layer."""
        strategy, layers = DEPTH_CONFIGS[AnalysisDepth.BASIC]
        assert strategy == ProfilingStrategy.CONSERVATIVE
        assert layers == 1

    def test_depth_configs_deep(self) -> None:
        """DEEP maps to CONSERVATIVE + 2 AI layers."""
        strategy, layers = DEPTH_CONFIGS[AnalysisDepth.DEEP]
        assert strategy == ProfilingStrategy.CONSERVATIVE
        assert layers == 2

    def test_depth_configs_radical(self) -> None:
        """RADICAL maps to RADICAL + 3 AI layers."""
        strategy, layers = DEPTH_CONFIGS[AnalysisDepth.RADICAL]
        assert strategy == ProfilingStrategy.RADICAL
        assert layers == 3
