"""Unit tests for evolve tools."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tachyon.errors.handler import ToolResult
from tachyon.evolve.config import EvolveConfig
from tachyon.evolve.context import EvolveContext
from tachyon.evolve.models import ExperimentRecord, MetricSnapshot
from tachyon.evolve.session import EvolveSession
from tachyon.evolve.tools import register_evolve_tools, _parse_benchmark_output
from tachyon.tools.context import SessionContext
from tachyon.tools.registry import ToolRegistry


@pytest.fixture
def base_session(device_h100, launch_256x4096, report_compute_bound) -> SessionContext:
    return SessionContext(
        kernels=[report_compute_bound],
        allowed_source_paths=set(),
    )


@pytest.fixture
def evolve_config() -> EvolveConfig:
    return EvolveConfig(
        build_cmd="echo 'building'",
        run_cmd="echo 'running'",
        max_iterations=5,
        git_auto_commit=False,
    )


@pytest.fixture
def evolve_session(evolve_config: EvolveConfig) -> EvolveSession:
    return EvolveSession(config=evolve_config)


@pytest.fixture
def evolve_ctx(
    base_session: SessionContext,
    evolve_session: EvolveSession,
    evolve_config: EvolveConfig,
    tmp_path: Path,
) -> EvolveContext:
    git = MagicMock()
    git.repo_root = tmp_path

    return EvolveContext(
        base=base_session,
        evolve=evolve_session,
        git=git,
        config=evolve_config,
    )


@pytest.fixture
def evolve_registry(
    evolve_ctx: EvolveContext,
) -> ToolRegistry:
    registry = ToolRegistry()
    register_evolve_tools(registry, evolve_ctx)
    return registry


class TestParseBenchmarkOutput:

    def test_parse_time_ms(self) -> None:
        output = "Kernel time: 5.234 ms\n"
        metrics = _parse_benchmark_output(output)
        assert "time_ms" in metrics
        assert metrics["time_ms"] == pytest.approx(5.234)

    def test_parse_throughput(self) -> None:
        output = "Throughput: 1234.5 GB/s\n"
        metrics = _parse_benchmark_output(output)
        assert "throughput" in metrics

    def test_parse_multiple(self) -> None:
        output = "elapsed: 1.234 ms\nThroughput: 500 MB/s\nMean: 2.5 ms\n"
        metrics = _parse_benchmark_output(output)
        assert len(metrics) >= 2

    def test_parse_with_filter(self) -> None:
        output = "elapsed: 1.234 ms\nThroughput: 500 MB/s\n"
        metrics = _parse_benchmark_output(output, metric_filter=["time"])
        assert "time_ms" in metrics or "time" in metrics

    def test_parse_empty(self) -> None:
        metrics = _parse_benchmark_output("")
        assert metrics == {}


class TestEvolveTools:

    @pytest.mark.asyncio
    async def test_get_evolve_status(
        self,
        evolve_registry: ToolRegistry,
        evolve_session: EvolveSession,
    ) -> None:
        result = await evolve_registry.execute(
            "get_evolve_status", "{}",
        )
        assert result.success
        data = result.data
        assert data["current_iteration"] == 0
        assert data["max_iterations"] == 5

    @pytest.mark.asyncio
    async def test_compile_kernel(
        self,
        evolve_registry: ToolRegistry,
    ) -> None:
        result = await evolve_registry.execute(
            "compile_kernel", "{}",
        )
        assert result.success
        data = result.data
        assert data["success"] is True
        assert data["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_compile_kernel_custom_cmd(
        self,
        evolve_registry: ToolRegistry,
    ) -> None:
        result = await evolve_registry.execute(
            "compile_kernel", '{"build_cmd": "echo custom"}',
        )
        assert result.success
        assert result.data["success"] is True

    @pytest.mark.asyncio
    async def test_compile_kernel_no_cmd(
        self,
        base_session: SessionContext,
        tmp_path: Path,
    ) -> None:
        cfg = EvolveConfig(build_cmd=None, git_auto_commit=False)
        session = EvolveSession(config=cfg)

        ctx = EvolveContext(
            base=base_session,
            evolve=session,
            git=MagicMock(repo_root=tmp_path),
            config=cfg,
        )
        registry = ToolRegistry()
        register_evolve_tools(registry, ctx)

        result = await registry.execute("compile_kernel", "{}")
        assert not result.success

    @pytest.mark.asyncio
    async def test_run_benchmark(
        self,
        evolve_registry: ToolRegistry,
    ) -> None:
        result = await evolve_registry.execute(
            "run_benchmark", "{}",
        )
        assert result.success
        assert result.data["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_run_benchmark_custom_cmd(
        self,
        evolve_registry: ToolRegistry,
    ) -> None:
        result = await evolve_registry.execute(
            "run_benchmark", '{"run_cmd": "echo elapsed: 2.5 ms"}',
        )
        assert result.success
        assert result.data["exit_code"] == 0
        assert "time_ms" in result.data["parsed_metrics"]

    @pytest.mark.asyncio
    async def test_compare_metrics_no_experiments(
        self,
        evolve_registry: ToolRegistry,
    ) -> None:
        result = await evolve_registry.execute(
            "compare_metrics", "{}",
        )
        assert not result.success

    @pytest.mark.asyncio
    async def test_compare_metrics_with_data(
        self,
        evolve_registry: ToolRegistry,
        evolve_session: EvolveSession,
    ) -> None:
        baseline = MetricSnapshot.from_kernel_metrics({
            "gpu__time_duration.sum": 10000000.0,
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 50.0,
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": 30.0,
        })
        optimized = MetricSnapshot.from_kernel_metrics({
            "gpu__time_duration.sum": 8000000.0,
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 60.0,
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": 35.0,
        })

        record = ExperimentRecord(
            iteration=0,
            baseline_metrics=baseline,
            optimized_metrics=optimized,
        )
        evolve_session.experiments.append(record)

        result = await evolve_registry.execute(
            "compare_metrics", "{}",
        )
        assert result.success
        data = result.data
        assert "comparison" in data
        assert data["avg_improvement_pct"] > 0

    @pytest.mark.asyncio
    async def test_edit_source_file_not_in_allowed(
        self,
        evolve_registry: ToolRegistry,
    ) -> None:
        result = await evolve_registry.execute(
            "edit_source_file",
            '{"file": "/etc/passwd", "old_content": "root", "new_content": "hacked"}',
        )
        assert not result.success


class TestCompileFailCount:
    """Test compile_fail_count tracking and FATAL response."""

    @pytest.mark.asyncio
    async def test_compile_fail_count_increments(
        self,
        evolve_ctx: EvolveContext,
    ) -> None:
        """compile_fail_count increments on each failure."""
        cfg = EvolveConfig(
            build_cmd="false",  # always exits 1
            git_auto_commit=False,
        )
        session = EvolveSession(config=cfg)
        ctx = EvolveContext(
            base=evolve_ctx.base,
            evolve=session,
            git=evolve_ctx.git,
            config=cfg,
        )
        registry = ToolRegistry()
        register_evolve_tools(registry, ctx)

        assert ctx.compile_fail_count == 0
        await registry.execute("compile_kernel", "{}")
        assert ctx.compile_fail_count == 1

    @pytest.mark.asyncio
    async def test_compile_fatal_after_two_failures(
        self,
        evolve_ctx: EvolveContext,
    ) -> None:
        """After 2 consecutive failures, result includes FATAL key."""
        cfg = EvolveConfig(
            build_cmd="false",
            git_auto_commit=False,
        )
        session = EvolveSession(config=cfg)
        ctx = EvolveContext(
            base=evolve_ctx.base,
            evolve=session,
            git=evolve_ctx.git,
            config=cfg,
        )
        registry = ToolRegistry()
        register_evolve_tools(registry, ctx)

        # First failure
        r1 = await registry.execute("compile_kernel", "{}")
        assert r1.success  # ToolResult.ok, but data shows failure
        assert r1.data["success"] is False
        assert "FATAL" not in r1.data

        # Second failure — triggers FATAL
        r2 = await registry.execute("compile_kernel", "{}")
        assert r2.success
        assert r2.data["success"] is False
        assert "FATAL" in r2.data
        assert "Do NOT call compile_kernel()" in r2.data["FATAL"]

    @pytest.mark.asyncio
    async def test_compile_fail_count_resets_on_success(
        self,
        evolve_registry: ToolRegistry,
        evolve_ctx: EvolveContext,
    ) -> None:
        """compile_fail_count resets to 0 after a successful build."""
        evolve_ctx.compile_fail_count = 3  # Simulate prior failures
        result = await evolve_registry.execute("compile_kernel", "{}")
        assert result.success
        assert result.data["success"] is True
        assert evolve_ctx.compile_fail_count == 0

    @pytest.mark.asyncio
    async def test_compile_result_includes_cwd(
        self,
        evolve_registry: ToolRegistry,
        evolve_ctx: EvolveContext,
    ) -> None:
        """compile_kernel result always includes cwd field."""
        result = await evolve_registry.execute("compile_kernel", "{}")
        assert "cwd" in result.data
        assert result.data["cwd"] == str(evolve_ctx.git.repo_root)


class TestRunFailCount:
    """Test run_fail_count tracking for validation failures."""

    @pytest.mark.asyncio
    async def test_run_fail_count_increments_on_validation_error(
        self,
        evolve_registry: ToolRegistry,
        evolve_ctx: EvolveContext,
    ) -> None:
        """run_fail_count increments when LLM-provided command is blocked."""
        assert evolve_ctx.run_fail_count == 0
        # "gemm" is not in _ALLOWED_COMMANDS
        await evolve_registry.execute(
            "run_benchmark", '{"run_cmd": "./gemm --size 1024"}',
        )
        assert evolve_ctx.run_fail_count == 1

    @pytest.mark.asyncio
    async def test_run_fatal_after_two_validation_failures(
        self,
        evolve_registry: ToolRegistry,
        evolve_ctx: EvolveContext,
    ) -> None:
        """After 2 validation failures, error message includes FATAL."""
        # First failure
        r1 = await evolve_registry.execute(
            "run_benchmark", '{"run_cmd": "./gemm --size 1024"}',
        )
        assert not r1.success
        assert "FATAL" not in (r1.error.message if r1.error else "")

        # Second failure — triggers FATAL
        r2 = await evolve_registry.execute(
            "run_benchmark", '{"run_cmd": "./gemm --size 2048"}',
        )
        assert not r2.success
        assert "FATAL" in (r2.error.message if r2.error else "")

    @pytest.mark.asyncio
    async def test_run_fail_count_resets_on_success(
        self,
        evolve_registry: ToolRegistry,
        evolve_ctx: EvolveContext,
    ) -> None:
        """run_fail_count resets to 0 after a successful benchmark."""
        evolve_ctx.run_fail_count = 3
        result = await evolve_registry.execute("run_benchmark", "{}")
        assert result.success
        assert evolve_ctx.run_fail_count == 0


class TestToolRegistration:

    def test_all_tools_registered(self, evolve_registry: ToolRegistry) -> None:
        names = evolve_registry.tool_names()
        expected = {
            "edit_source_file",
            "compile_kernel",
            "run_benchmark",
            "reprofile",
            "compare_metrics",
            "get_evolve_status",
        }
        assert expected.issubset(set(names))

    def test_tool_definitions_valid(self, evolve_registry: ToolRegistry) -> None:
        for name in evolve_registry.tool_names():
            tool = evolve_registry.get(name)
            assert tool is not None
            assert tool.handler is not None
            assert "type" in tool.parameters
