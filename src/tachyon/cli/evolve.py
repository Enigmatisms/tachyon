"""Evolve CLI — automated CUDA kernel optimization command.

Usage:
    tachyon evolve ./my_app --size 1024 --build "make -j8"
    tachyon evolve --report report.ncu-rep ./my_app --size 1024
    tachyon evolve -i ./my_app --size 1024    # interactive mode
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from tachyon.cli.main import app
from tachyon.config.settings import TachyonConfig

if TYPE_CHECKING:
    from tachyon.llm.backend import LLMBackend

console = Console()


@app.command()
@click.argument("executable", type=click.Path(exists=True))
@click.argument("exe_args", nargs=-1, type=str)
@click.option("--report", "report_file", default=None, type=click.Path(exists=True), help="Load existing .ncu-rep as baseline instead of auto-profiling.")
@click.option("--build", "build_cmd", default=None, help="Build command (e.g. 'make -j8')")
@click.option("--run", "run_cmd", default=None, help="Run/benchmark command")
@click.option("--config", "config_path", default=None, type=click.Path(), help="Path to evolve.toml config")
@click.option("--max-iterations", default=10, type=int, help="Maximum optimization iterations")
@click.option("--kernel", "-k", default=None, help="Target kernel name filter")
@click.option("--model", "-m", default=None, help="LLM model name")
@click.option("--provider", "-p", default=None, help="LLM provider")
@click.option("--verbose", "-v", is_flag=True, help="Show tool calls and debug info")
@click.option("--interactive", "-i", is_flag=True, help="Pause between iterations for user guidance")
@click.option("--export", "export_path", default=None, type=click.Path(), help="Export results to markdown file")
@click.option("--lang", default=None, help="Output language (en/zh)")
@click.option("--ncu-set", default="full", type=click.Choice(["basic", "detailed", "full"]), help="NCU metric set for profiling")
@click.option("--ncu-metrics", default=None, type=str, help="Comma-separated NCU metrics")
@click.option("--ncu-args", default=None, type=str, help="Extra arguments to pass to ncu")
@click.option("--timeout", default=3600, type=int, help="Total wall-clock timeout in seconds")
@click.option("--quiet", "-q", is_flag=True, help="Only show final summary, skip per-iteration reports")
def evolve(
    executable: str,
    exe_args: tuple,
    report_file: str | None,
    build_cmd: str | None,
    run_cmd: str | None,
    config_path: str | None,
    max_iterations: int,
    kernel: str | None,
    model: str | None,
    provider: str | None,
    verbose: bool,
    interactive: bool,
    export_path: str | None,
    lang: str | None,
    ncu_set: str | None,
    ncu_metrics: str | None,
    ncu_args: str | None,
    timeout: int,
    quiet: bool,
) -> None:
    """Automated CUDA kernel optimization via iterative profiling and editing.

    Automatically profiles the executable, analyzes bottlenecks, modifies
    source code, compiles, re-profiles, and iterates until convergence.

    EXECUTABLE: Path to the compiled binary to optimize.
    EXE_ARGS: Arguments to pass to the executable.

    \b
    Examples:
        tachyon evolve ./my_app --size 1024
        tachyon evolve --build "make -j8" ./my_app --size 1024
        tachyon evolve --report baseline.ncu-rep ./my_app --size 1024
        tachyon evolve -i --max-iterations 5 ./my_app --size 1024
    """
    import asyncio

    # Initialize debug recording (no-op unless TACHYON_DEBUG_RECORD is set)
    from tachyon.utils.debug_record import init as init_debug_record
    init_debug_record()

    # Load config
    config = TachyonConfig.load()
    if model:
        config.llm.model = model
    if provider:
        config.llm.provider = provider
    if lang:
        config.output.lang = lang

    # Initialize i18n
    import tachyon.i18n as i18n
    i18n.init(config.output.lang)

    from tachyon.evolve.config import EvolveConfig
    evolve_config = EvolveConfig.load(
        config_path=Path(config_path) if config_path else None,
        build_cmd=build_cmd or None,
        run_cmd=run_cmd or None,
        max_iterations=max_iterations,
        target_kernel=kernel,
    )

    # Default run_cmd if not set
    if not evolve_config.run_cmd:
        args_str = " ".join(exe_args) if exe_args else ""
        evolve_config.run_cmd = f"{executable} {args_str}".strip()

    # Verify git repo
    from tachyon.evolve.git import GitRollback
    git = GitRollback()

    if not git.verify_git_repo():
        console.print("[red]Error: Not a git repository.[/red]")
        console.print("tachyon evolve requires a git repo for safe rollback.")
        sys.exit(1)

    if not git.check_working_tree_clean():
        auto_commit = git.auto_commit_uncommitted()
        if auto_commit is None:
            console.print("[red]Error: Working tree has uncommitted changes and auto-commit failed.[/red]")
            sys.exit(1)
        _, short = auto_commit
        console.print(
            f"[yellow]Working tree had uncommitted changes, "
            f"auto-committed as {short}.[/yellow]"
        )

    git.ensure_gitignore_entries()
    git.cleanup_stale_artifacts()

    # --- Load or generate profiling data ---
    exe_arg_list = list(exe_args)

    # Track the report path for mapper
    _ncu_rep_path: str | None = None

    if report_file:
        # Mode A: Load existing .ncu-rep report
        console.print(
            f"Loading report: [bold]{report_file}[/bold]..."
        )
        from tachyon.reader.ncu_reader import NcuReportReader
        reader = NcuReportReader(config)
        result = reader.load(report_file)
        if not result.success or result.data is None:
            console.print(f"[red]Failed to load report: {result.error}[/red]")
            sys.exit(1)
        kernels = result.data
        _ncu_rep_path = report_file
        console.print(
            f"Loaded [bold green]{len(kernels)}[/bold green] kernel(s) from report."
        )
    else:
        # Mode B: Auto-profile the executable (single pass, no two-stage pipeline)
        from tachyon.profiler.ncu_profiler import NcuProfiler
        from tachyon.profiler.tool_path import ToolPathResolver
        from tachyon.utils.progress import print_error_panel

        extra_ncu = ncu_args.split() if ncu_args else None
        if kernel:
            extra_ncu = extra_ncu or []
            from tachyon.utils.kernel_filter import to_ncu_regex
            extra_ncu.extend(["--kernel-name-base", "mangled"])
            extra_ncu.extend(["--kernel-regex", to_ncu_regex(kernel)])

        console.print("[dim]Running NCU profiling...[/dim]")
        resolver = ToolPathResolver(config)
        profiler = NcuProfiler(config, resolver)
        profile_result = profiler.profile_basic(
            executable,
            exe_arg_list,
            extra_ncu_args=extra_ncu,
            metric_set_override=ncu_set,
            metrics_override=ncu_metrics,
            verbose=verbose,
        )

        if not profile_result.success:
            assert profile_result.error is not None
            print_error_panel(
                "Profiling Error",
                profile_result.error.message,
                suggestion=profile_result.error.suggestion,
            )
            sys.exit(1)

        assert profile_result.data is not None
        _ncu_rep_path = profile_result.data.ncu_rep_path

        # Load the generated report
        from tachyon.reader.ncu_reader import NcuReportReader
        reader = NcuReportReader(config)
        load_result = reader.load(str(_ncu_rep_path))
        if not load_result.success or load_result.data is None:
            console.print("[red]Failed to load generated profiling data.[/red]")
            sys.exit(1)
        kernels = load_result.data
        console.print(
            f"Profiled [bold green]{len(kernels)}[/bold green] kernel(s)."
        )

    if kernel:
        from tachyon.utils.kernel_filter import filter_kernels
        kernels = filter_kernels(kernels, kernel)

    # --- Build analysis context ---
    from tachyon.analyzers.base import AnalyzerRegistry
    from tachyon.correlator.source_correlator import SourceCorrelator
    from tachyon.tools.analysis import register_analysis_tools
    from tachyon.tools.context import SessionContext
    from tachyon.tools.data_query import register_data_query_tools
    from tachyon.tools.registry import ToolRegistry
    from tachyon.tools.source import register_source_tools
    from tachyon.tools.source_view import register_source_view_tools

    analyzer_registry = AnalyzerRegistry()
    analyzer_registry.auto_register()

    correlator = None
    if any(k.instanced_metrics for k in kernels):
        correlator = SourceCorrelator()

    mapper = None
    if _ncu_rep_path:
        try:
            from tachyon.correlator.source_mapper import NCUMappingSystem
            mapper = NCUMappingSystem(_ncu_rep_path)
        except Exception:
            pass

    base_session = SessionContext(
        kernels=kernels,
        action=reader,
        correlator=correlator,
        registry=analyzer_registry,
        mapper=mapper,
        allowed_source_paths=SessionContext.build_allowed_source_paths(kernels, mapper),
    )

    # Create tool registry with all tools
    tool_registry = ToolRegistry()
    register_data_query_tools(tool_registry, base_session)
    register_source_tools(tool_registry, base_session)
    register_source_view_tools(tool_registry, base_session)
    register_analysis_tools(tool_registry, base_session)

    # LLM backend is required
    backend = _try_create_backend(config)
    if backend is None:
        console.print("[red]Error: LLM backend required for evolve mode.[/red]")
        console.print("Set OPENAI_API_KEY or use --model/--provider.")
        sys.exit(1)

    # --- Set up evolve session (in-memory, no persistence) ---
    from tachyon.evolve.context import EvolveContext
    from tachyon.evolve.session import EvolveSession

    evolve_session = EvolveSession(config=evolve_config)

    if kernels:
        from tachyon.evolve.models import MetricSnapshot
        _init_metrics = {
            name: mv.value
            for name, mv in kernels[0].metrics.items()
            if mv is not None
        }
        if _init_metrics:
            evolve_session.baseline_metrics = MetricSnapshot.from_kernel_metrics(_init_metrics)

    evolve_ctx = EvolveContext(
        base=base_session,
        evolve=evolve_session,
        git=git,
        config=evolve_config,
        reader=reader,
        executable=executable,
        exe_args=exe_arg_list,
    )

    # Register evolve tools
    from tachyon.evolve.tools import register_evolve_tools
    register_evolve_tools(tool_registry, evolve_ctx)

    # --- Run evolve loop ---
    mode = "interactive" if interactive else "autonomous"
    profile_mode = "report" if report_file else "auto-profiled"
    body = (
        f"[bold cyan]Tachyon Evolve[/bold cyan] — Automated CUDA Kernel Optimization\n"
        f"  Mode: {mode}  |  Profile: {profile_mode}  |  "
        f"Max iterations: {max_iterations}  |  Target: {kernel or 'all kernels'}"
    )
    console.print(Panel(body, expand=False))
    console.print(
        f"Connected to [bold]{config.llm.provider}/{config.llm.model}[/bold]"
    )
    asyncio.run(_run_evolve(
        backend, tool_registry, evolve_ctx, evolve_config,
        verbose, interactive, quiet, export_path, timeout,
    ))


async def _run_evolve(
    backend: LLMBackend,
    tool_registry: ToolRegistry,
    evolve_ctx: EvolveContext,
    evolve_config,
    verbose: bool,
    interactive: bool,
    quiet: bool,
    export_path: str | None,
    total_timeout: int,
) -> None:
    """Run the full multi-iteration evolve loop."""
    from tachyon.evolve.display import EvolveProgressDisplay
    from tachyon.evolve.orchestrator import EvolveOrchestrator
    from tachyon.evolve.models import ExperimentStatus

    display = EvolveProgressDisplay(
        max_iterations=evolve_config.max_iterations,
        console=console,
    )

    def on_iteration(record) -> None:
        display.add_result(record)

    orchestrator = EvolveOrchestrator(
        backend=backend,
        registry=tool_registry,
        ctx=evolve_ctx,
        max_iterations=evolve_config.max_iterations,
        total_timeout=total_timeout,
        interactive=interactive,
        quiet=quiet,
        on_iteration=on_iteration,
        display=display,
    )

    display.start()
    try:
        experiments = await orchestrator.run()

        display.stop()
        display.render_summary(experiments)

        # Smart final summary
        _print_final_summary(evolve_ctx.evolve, experiments, quiet)

        # Check for optimization branches saved during convergence
        git = evolve_ctx.git
        tag_result = git._run(["branch", "-l", "tachyon-optimized-*"], check=False)
        if tag_result.returncode == 0 and tag_result.stdout.strip():
            branches = [b.strip() for b in tag_result.stdout.strip().splitlines() if b.strip()]
            if branches:
                console.print(
                    f"\n[dim]Optimized kernel states saved on branches: "
                    f"{', '.join(branches)}. "
                    f"Review with: git diff <branch>[/dim]"
                )

        # Check for staged (accepted) changes not yet committed
        staged_result = git._run(["diff", "--cached", "--stat"], check=False)
        if staged_result.returncode == 0 and staged_result.stdout.strip():
            console.print(
                "\n[dim]Accepted optimizations are staged. "
                "Commit when ready: git commit -m 'your message'[/dim]"
            )

        if export_path:
            _export_results(experiments, export_path)

    finally:
        display.stop()


def _print_final_summary(
    session,
    experiments: list,
    quiet: bool,
) -> None:
    """Print a smart final summary based on success/failure."""
    from tachyon.evolve.models import ExperimentStatus

    ok = [e for e in experiments if e.status == ExperimentStatus.SUCCESS]
    fail = [e for e in experiments if e.status != ExperimentStatus.SUCCESS]

    if session.best_iteration >= 0 and session.best_metrics is not session.baseline_metrics:
        improvement = session.get_best_improvement()
        best = session.best_iteration
        best_exp = next((e for e in experiments if e.iteration == best), None)

        console.print()
        console.rule(f"[bold green]Optimization Succeeded ({improvement:+.1f}%)[/bold green]")

        gpu_before = f"{session.baseline_metrics.duration_ms:.2f}ms" if session.baseline_metrics and session.baseline_metrics.duration_ms else "?"
        gpu_after = f"{session.best_metrics.duration_ms:.2f}ms" if session.best_metrics and session.best_metrics.duration_ms else "?"
        console.print(
            f"  Best: iter {best}  |  GPU: {gpu_before} -> {gpu_after}  |  "
            f"Success: {len(ok)}, Failed: {len(fail)}"
        )

        if not quiet and best_exp and best_exp.hypothesis:
            console.print(Text(best_exp.hypothesis[-500:], style="dim"))
    else:
        console.print()
        console.rule("[yellow]No Improvements Achieved[/yellow]")
        console.print(
            f"  Total: {len(experiments)} iterations  |  "
            f"Success: {len(ok)}, Failed: {len(fail)}"
        )


def _export_results(experiments, export_path: str) -> None:
    """Export evolve results to markdown."""
    from tachyon.chat.export import write_export

    lines = [
        "# Tachyon Evolve Results\n",
        f"Total iterations: {len(experiments)}\n",
        "## Iteration Summary\n",
        "| Iteration | Status | GPU Time | Decision |",
        "|-----------|--------|----------|----------|",
    ]

    for e in experiments:
        gpu = "-"
        if e.optimized_metrics is not None and e.optimized_metrics.duration_ms:
            gpu = f"{e.optimized_metrics.duration_ms:.2f}ms"
        lines.append(
            f"| {e.iteration} | {e.status.value} | {gpu} | {e.decision or ''} |"
        )

    content = "\n".join(lines)
    try:
        resolved = write_export(content, export_path)
        console.print(f"[green]Exported to {resolved}[/green]")
    except OSError as e:
        console.print(f"[red]Export failed: {e}[/red]")


def _try_create_backend(config: TachyonConfig) -> LLMBackend | None:
    """Try to create an LLM backend."""
    import os
    from tachyon.llm.backend import create_backend

    api_key = config.llm.api_key or os.environ.get(config.llm.api_key_env)
    try:
        return create_backend(
            provider=config.llm.provider,
            model=config.llm.model,
            api_key=api_key,
            base_url=config.llm.base_url,
        )
    except (ImportError, ValueError) as e:
        console.print(f"[yellow]LLM backend not available: {e}[/yellow]")
        return None
