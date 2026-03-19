"""Chat CLI — interactive AI-enhanced CUDA analysis.

Usage:
    tachyon chat report.ncu-rep [--model MODEL] [--no-ai]

Features:
  - Rich terminal output with streaming
  - Auto-fallback to Rule-Only when LLM unavailable
  - Special commands: /help, /kernels, /tree, /export, /quit
  - Token usage tracking
"""
from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from tachyon.cli.main import app
from tachyon.config.settings import TachyonConfig

if TYPE_CHECKING:
    from typing import Any

    from tachyon.analyzers.base import AnalyzerRegistry
    from tachyon.llm.backend import LLMBackend
    from tachyon.tools.context import SessionContext
    from tachyon.tools.registry import ToolRegistry

console = Console()


@app.command()
@click.argument("report_file", type=click.Path(exists=True))
@click.option(
    "--model", "-m",
    default=None,
    help="LLM model (e.g. gpt-4o, claude-sonnet-4-20250514)",
)
@click.option("--provider", "-p", default=None, help="LLM provider (openai/anthropic/litellm)")
@click.option("--no-ai", is_flag=True, help="Force Rule-Only mode (no LLM)")
@click.option("--kernel", "-k", default=None, help="Filter kernels by name (glob pattern, e.g. 'matmul*').")
@click.option("--lang", default=None, help="Language (en/zh)")
@click.option("--verbose", "-v", is_flag=True, help="Show tool calls and debug info")
def chat(
    report_file: str,
    model: str | None,
    provider: str | None,
    no_ai: bool,
    kernel: str | None,
    lang: str | None,
    verbose: bool,
) -> None:
    """Interactive AI-enhanced CUDA performance analysis.

    Load an NCU report and chat with the Tachyon AI agent about kernel
    performance. The agent uses 9 specialized tools to analyze metrics,
    source code, and SASS instructions.

    Falls back to Rule-Only mode when LLM is unavailable.
    """
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

    console.print(Panel.fit(
        "[bold cyan]Tachyon[/bold cyan] — AI-Powered CUDA Performance Analyzer",
        subtitle="Type /help for commands, /quit to exit",
    ))

    # Load report
    console.print(f"Loading report: [bold]{report_file}[/bold]...")
    kernels, reader = _load_report(report_file, config)
    if not kernels:
        console.print("[red]Error: No kernels found in report.[/red]")
        sys.exit(1)

    # Filter kernels if --kernel specified
    if kernel:
        from tachyon.utils.kernel_filter import filter_kernels
        kernels = filter_kernels(kernels, kernel)
        if not kernels:
            console.print(f"[red]No kernels matching '{kernel}'.[/red]")
            sys.exit(1)

    console.print(f"Loaded [bold green]{len(kernels)}[/bold green] kernel(s).")
    for i, k in enumerate(kernels):
        name = k.demangled_name or k.kernel_name
        console.print(f"  [{i}] {name}")

    # Set up tools
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

    import logging
    _logger = logging.getLogger(__name__)

    # Create correlator if we have instanced metrics and reader
    correlator = None
    if reader is not None and any(k.instanced_metrics for k in kernels):
        correlator = SourceCorrelator()
        _logger.info(
            "SourceCorrelator initialized: %d kernel(s) with instanced metrics",
            sum(1 for k in kernels if k.instanced_metrics),
        )
    else:
        # Diagnostic: explain why correlator was NOT created
        if reader is None:
            _logger.warning("SourceCorrelator NOT created: reader is None")
        elif not kernels:
            _logger.warning("SourceCorrelator NOT created: no kernels loaded")
        else:
            n_with_inst = sum(1 for k in kernels if k.instanced_metrics)
            _logger.warning(
                "SourceCorrelator NOT created: 0/%d kernels have instanced metrics. "
                "Source correlation requires '--set detailed' or higher (PC-sampling).",
                len(kernels),
            )

    # Initialize NCUMappingSystem independently — it loads data directly via
    # ncu_report.load_report() and does NOT depend on NcuReportReader.instanced_metrics.
    mapper = None
    if reader is not None:
        try:
            from tachyon.correlator.source_mapper import NCUMappingSystem
            mapper = NCUMappingSystem(report_file)
            _logger.info(
                "NCUMappingSystem initialized: %d mapped instructions",
                len(mapper._s2as_flat),
            )
        except Exception as e:
            _logger.warning("NCUMappingSystem NOT created: %s", e)

    # User-facing status: only warn if BOTH correlator AND mapper are unavailable
    if correlator is None and mapper is None:
        console.print(
            "[yellow]Source correlation unavailable[/yellow]: "
            "no PC-sampled metrics in report. "
            "Re-profile with [bold]--set detailed[/bold] or [bold]--set full[/bold]."
        )

    session = SessionContext(
        kernels=kernels,
        action=reader,  # NcuReportReader implements ActionHandle protocol
        correlator=correlator,
        registry=analyzer_registry,
        mapper=mapper,
        allowed_source_paths=SessionContext.build_allowed_source_paths(kernels, mapper),
    )

    tool_registry = ToolRegistry()
    register_data_query_tools(tool_registry, session)
    register_source_tools(tool_registry, session)
    register_source_view_tools(tool_registry, session)
    register_analysis_tools(tool_registry, session)

    # Try to create LLM backend
    backend = None
    if not no_ai:
        backend = _try_create_backend(config)

    if backend is None:
        console.print("[yellow]Running in Rule-Only mode (no LLM available).[/yellow]")
        _rule_only_mode(kernels, analyzer_registry, session)
        return

    console.print(
        f"Connected to [bold]{config.llm.provider}/{config.llm.model}[/bold]"
    )

    # Run interactive chat loop
    asyncio.run(_chat_loop(backend, tool_registry, kernels, verbose, config.llm.timeout, config.llm.move_timeout))


async def _chat_loop(
    backend: LLMBackend,
    tool_registry: ToolRegistry,
    kernels: list,
    verbose: bool,
    timeout: int = 600,
    move_timeout: int = 120,
) -> None:
    """Main interactive chat loop."""
    from tachyon.agent.loop import run_agent_loop
    from tachyon.agent.persona import (
        build_kernel_context,
        build_system_prompt,
    )
    from tachyon.utils.progress import AgentSpinner

    kernel_context = build_kernel_context(kernels)
    system_prompt = build_system_prompt(tool_registry, kernel_context)

    # Transparency: show user what context the AI agent has
    _show_agent_context(kernels, tool_registry)

    history = []
    total_tokens = 0
    prompt_session: PromptSession[str] = PromptSession()

    while True:
        try:
            user_input = (await prompt_session.prompt_async(
                HTML("\n<cyan><b>You&gt;</b></cyan> ")
            )).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        # Handle special commands
        if user_input.startswith("/"):
            should_continue = _handle_command(
                user_input, kernels, tool_registry, total_tokens,
            )
            if should_continue is None:
                break  # /quit
            continue

        # Run agent loop — show tool calls in real time for transparency
        text_buffer = []
        tool_calls_made: list[str] = []
        spinner = AgentSpinner(timeout=timeout, console=console)
        spinner.start()
        try:
            async for event in run_agent_loop(
                backend=backend,
                registry=tool_registry,
                user_message=user_input,
                system_prompt=system_prompt,
                history=history[-10:],
                stream=False,
                timeout=timeout,
                move_timeout=move_timeout,
            ):
                if event.type == "text":
                    spinner.set_synthesizing()
                    text_buffer.append(event.content or "")
                elif event.type == "thinking":
                    if event.content:
                        console.print(f"  [dim italic]{event.content[:120]}[/dim italic]")
                elif event.type == "tool_call":
                    name = event.data["name"] if event.data else "?"
                    spinner.set_tool(name)
                    args = event.data.get("arguments", {}) if event.data else {}
                    tool_calls_made.append(name)
                    args_short = ", ".join(
                        f"{k}={v}" for k, v in list(args.items())[:3]
                    )
                    console.print(f"  [cyan]▶ {name}[/cyan]({args_short})")
                elif event.type == "tool_result":
                    spinner.set_status("waiting for LLM")
                    if event.data:
                        summary = event.data.get("summary", "")
                        ok = "✓" if event.data.get("success") else "✗"
                        t = event.data.get("elapsed", 0)
                        console.print(f"    [dim]{ok} {summary[:140]}  ({t:.2f}s)[/dim]")
                elif event.type == "system":
                    if event.data and "turn" in event.data:
                        # Per-turn timing
                        console.print(f"  [dim]{event.content}[/dim]")
                    elif event.content:
                        console.print(f"  [yellow]{event.content}[/yellow]")
                elif event.type == "done":
                    spinner.stop()
                    if event.data:
                        turns = event.data.get("turns", 0)
                        n_tools = event.data.get("tool_calls", 0)
                        tokens = event.data.get("total_tokens", 0)
                        total_elapsed = event.data.get("total_elapsed", 0)
                        total_tokens += tokens
                        console.print(
                            f"  [dim]Done: {turns} turns, {n_tools} tool calls, "
                            f"{tokens:,} tokens, {total_elapsed:.1f}s[/dim]"
                        )
        finally:
            spinner.stop()

        # Render collected text as markdown
        full_text = "".join(text_buffer)
        if full_text:
            console.print()
            console.print(Panel(
                Markdown(full_text),
                title="[bold green]Tachyon[/bold green]",
                border_style="green",
            ))

        # Update history
        from tachyon.llm.backend import Message, Role
        history.append(Message(role=Role.USER, content=user_input))
        if full_text:
            history.append(Message(role=Role.ASSISTANT, content=full_text))


def _handle_command(
    cmd: str,
    kernels: list,
    registry: ToolRegistry,
    total_tokens: int,
) -> bool | None:
    """Handle special /commands. Returns None for /quit, True otherwise."""
    parts = cmd.split()
    command = parts[0].lower()

    if command in ("/quit", "/q", "/exit"):
        console.print("[dim]Goodbye![/dim]")
        return None

    if command == "/help":
        console.print(Panel(
            "[bold]/help[/bold] — Show this help\n"
            "[bold]/kernels[/bold] — List loaded kernels\n"
            "[bold]/tree <id>[/bold] — Show OptTree for kernel\n"
            "[bold]/tokens[/bold] — Show token usage\n"
            "[bold]/quit[/bold] — Exit chat",
            title="Commands",
        ))
        return True

    if command == "/kernels":
        for i, k in enumerate(kernels):
            name = k.demangled_name or k.kernel_name
            console.print(f"  [{i}] {name}")
        return True

    if command == "/tree":
        kernel_id = int(parts[1]) if len(parts) > 1 else 0
        _show_opt_tree(kernels, kernel_id)
        return True

    if command == "/tokens":
        console.print(f"Total tokens used: [bold]{total_tokens:,}[/bold]")
        return True

    console.print(f"[red]Unknown command: {command}[/red]. Type /help.")
    return True


def _show_opt_tree(kernels: list, kernel_id: int) -> None:
    """Render OptTree for a kernel."""
    from tachyon.analyzers.base import AnalyzerRegistry
    from tachyon.tree.opt_tree import OptimizationTree

    if kernel_id >= len(kernels):
        console.print(f"[red]Invalid kernel_id: {kernel_id}[/red]")
        return

    reg = AnalyzerRegistry()
    reg.auto_register()
    findings = reg.run_all(kernels[kernel_id])
    tree = OptimizationTree(findings)
    md = tree.to_markdown()
    console.print(Markdown(md))


def _show_agent_context(kernels: list, tool_registry: ToolRegistry) -> None:
    """Show the user what context and tools the AI agent has access to."""
    from rich.table import Table

    # Tools overview
    tool_names = [t.name for t in tool_registry.all_definitions()]
    console.print(
        f"  [dim]Agent tools ({len(tool_names)}): "
        f"{', '.join(tool_names)}[/dim]"
    )

    # Key metrics excerpt per kernel
    table = Table(
        title="Agent Context (data available to LLM)",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        title_style="bold",
        expand=False,
    )
    table.add_column("Kernel", style="green", max_width=50)
    table.add_column("SM%", justify="right")
    table.add_column("DRAM%", justify="right")
    table.add_column("Occup%", justify="right")
    table.add_column("Regs", justify="right")
    table.add_column("Shared", justify="right")
    table.add_column("Duration", justify="right")

    for k in kernels:
        name = (k.demangled_name or k.kernel_name)
        if len(name) > 50:
            name = name[:47] + "..."
        sm = k.metric_value("sm__throughput.avg.pct_of_peak_sustained_elapsed")
        dram = k.metric_value("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed")
        occ = k.metric_value("sm__warps_active.avg.pct_of_peak_sustained_active")
        dur = k.metric_value("gpu__time_duration.sum")
        table.add_row(
            name,
            f"{sm:.1f}" if sm is not None else "-",
            f"{dram:.1f}" if dram is not None else "-",
            f"{occ:.1f}" if occ is not None else "-",
            str(k.launch_params.registers_per_thread),
            f"{k.launch_params.shared_mem_bytes}B",
            f"{dur / 1e6:.2f}ms" if dur is not None else "-",
        )

    console.print(table)
    console.print(
        "  [dim]The agent can call tools to fetch detailed metrics, "
        "source hotspots, SASS instructions, and stall analysis.[/dim]"
    )
    console.print()


def _load_report(report_file: str, config: TachyonConfig) -> tuple[list, Any]:
    """Load .ncu-rep file and return list of KernelReport objects and reader.

    Returns:
        Tuple of (kernels, reader) where reader implements ActionHandle protocol.
    """
    try:
        from tachyon.reader.ncu_reader import NcuReportReader
        reader = NcuReportReader(config)
        result = reader.load(report_file)
        if result.success and result.data is not None:
            return result.data, reader
        console.print(f"[red]Failed to load report: {result.error}[/red]")
        return [], None
    except Exception as e:
        console.print(f"[red]Failed to load report: {e}[/red]")
        return [], None


def _try_create_backend(config: TachyonConfig) -> LLMBackend | None:
    """Try to create an LLM backend. Returns None on failure."""
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


def _rule_only_mode(
    kernels: list,
    analyzer_registry: AnalyzerRegistry,
    session: SessionContext,
) -> None:
    """Fallback: run rule-based analysis and print report."""
    from tachyon.report.markdown import MarkdownReporter, ReportContext
    from tachyon.tree.opt_tree import OptimizationTree

    for kernel in kernels:
        findings = analyzer_registry.run_all(kernel)
        opt_tree = OptimizationTree(findings)
        ctx = ReportContext(
            kernel_name=kernel.kernel_name,
            demangled_name=kernel.demangled_name,
            device_name=kernel.device_info.name,
            compute_capability=(
                f"{kernel.device_info.compute_capability[0]}"
                f".{kernel.device_info.compute_capability[1]}"
            ),
            findings=findings,
            hotspots=[],
            opt_tree=opt_tree,
            degradation_warning=None,
            verbose=True,
        )
        reporter = MarkdownReporter()
        md = reporter.render(ctx)
        console.print(Markdown(md))
