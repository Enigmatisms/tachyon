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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel

from tachyon.cli.main import app
from tachyon.config.settings import TachyonConfig
from tachyon.llm.backend import Message, Role

if TYPE_CHECKING:
    from tachyon.analyzers.base import AnalyzerRegistry
    from tachyon.chat.export import ChatHistory
    from tachyon.llm.backend import LLMBackend
    from tachyon.tools.context import SessionContext
    from tachyon.tools.registry import ToolRegistry

# Reuse CJK detection from agent loop (single source of truth).
from tachyon.agent.loop import _CJK_RE


console = Console()


@dataclass
class _DeepSession:
    """Encapsulates deep analysis mode state and stage logic."""

    specs: list = field(default_factory=list)  # StageSpec list
    stage: int = 0           # 0=not started, 1=metrics, 2=source+recommend
    user_request: str | None = None  # original first user input
    prefer_lang: str | None = None   # cached from first input

    @property
    def active(self) -> bool:
        return bool(self.specs)

    def begin_stage1(self, user_input: str) -> str:
        """Build the stage 1 user message. Returns the user message to send."""
        self.stage = 1
        self.user_request = user_input
        self.prefer_lang = (
            "zh" if bool(_CJK_RE.search(user_input)) else None
        )
        return self.specs[0].prompt + "\n\n---\nUser request:\n" + user_input

    def advance_stage(self) -> str | None:
        """Advance to next stage. Returns stage prompt or None if done."""
        if self.stage >= len(self.specs):
            return None
        self.stage += 1
        spec = self.specs[self.stage - 1]
        msg = spec.prompt
        if self.user_request:
            msg += "\n\n---\nUser request:\n" + self.user_request
        return msg

    @property
    def is_stage1(self) -> bool:
        return self.stage == 1

    def augment_input(self, user_input: str) -> str:
        """For stage 2+ follow-ups, prepend the original request."""
        if self.stage >= 2 and self.user_request:
            return self.user_request + "\n\n---\n" + user_input
        return user_input

    def loop_overrides(self) -> dict[str, Any]:
        """Return extra kwargs for run_agent_loop based on current stage."""
        overrides: dict[str, Any] = {}
        if self.is_stage1:
            stage_prompt = self.specs[0].prompt
            overrides["trim_user_after_turn0"] = stage_prompt
            overrides["synthesis_prompt"] = stage_prompt
        if self.prefer_lang:
            overrides["prefer_lang"] = self.prefer_lang
        return overrides


@dataclass
class _EvolveState:
    """Encapsulates evolve mode state for chat integration."""

    active: bool = False
    ctx: Any = None              # EvolveContext
    tool_registry: ToolRegistry | None = None
    original_registry: ToolRegistry | None = None
    original_system_prompt: str | None = None
    original_lean_prompt: str | None = None
    evolve_system_prompt: str | None = None  # C3: evolve-specific prompt

    def activate(
        self,
        tool_registry: ToolRegistry,
        evolve_tool_registry: ToolRegistry,
        evolve_ctx: Any,
        system_prompt: str,
        lean_prompt: str,
        evolve_system_prompt: str | None = None,
    ) -> None:
        """Enter evolve mode: swap tool registry and system prompt."""
        self.active = True
        self.original_registry = tool_registry
        self.original_system_prompt = system_prompt
        self.original_lean_prompt = lean_prompt
        self.tool_registry = evolve_tool_registry
        self.ctx = evolve_ctx
        self.evolve_system_prompt = evolve_system_prompt

    def deactivate(self) -> tuple[ToolRegistry, str, str | None]:
        """Exit evolve mode: restore original tools and prompt."""
        self.active = False
        return (
            self.original_registry,
            self.original_system_prompt,
            self.original_lean_prompt,
        )

    @property
    def status_summary(self) -> dict:
        if self.ctx is None:
            return {"active": False}
        return self.ctx.evolve.get_status_summary()


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
@click.option("--deep", is_flag=True, help="Deep analysis mode with staged output (2 stages).")
def chat(
    report_file: str,
    model: str | None,
    provider: str | None,
    no_ai: bool,
    kernel: str | None,
    lang: str | None,
    verbose: bool,
    deep: bool,
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

    # Initialize debug recording (no-op unless TACHYON_DEBUG_RECORD is set)
    from tachyon.utils.debug_record import init as init_debug_record
    init_debug_record()

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
    from tachyon.tools import register_all_tools
    from tachyon.tools.context import SessionContext
    from tachyon.tools.registry import ToolRegistry

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
    register_all_tools(tool_registry, session)

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
    asyncio.run(_chat_loop(
        backend, tool_registry, kernels, verbose,
        config.llm.timeout, config.llm.move_timeout,
        deep=deep,
        model_info=f"{config.llm.provider}/{config.llm.model}",
    ))


async def _chat_loop(
    backend: LLMBackend,
    tool_registry: ToolRegistry,
    kernels: list,
    verbose: bool,
    timeout: int = 600,
    move_timeout: int = 120,
    deep: bool = False,
    model_info: str = "",
) -> None:
    """Main interactive chat loop."""
    from tachyon.agent.loop import run_agent_loop
    from tachyon.agent.persona import (
        build_kernel_context,
        build_lean_system_prompt,
        build_system_prompt,
    )
    from tachyon.chat.export import (
        ChatHistory,
        ToolCallRecord,
        TurnRecord,
        UsageRecord,
    )
    from tachyon.utils.progress import ChatStatusDisplay

    kernel_context = build_kernel_context(kernels)
    system_prompt = build_system_prompt(tool_registry, kernel_context)

    # Deep mode: append stage overview to system prompt
    deep_note = ""
    if deep:
        import tachyon.i18n as _i18n
        deep_note = _i18n.t("stage.deep_mode.note", fallback="")
        if deep_note:
            system_prompt += "\n\n" + deep_note

    # Lean system prompt: identity + tool catalog + key rules only.
    # Applied after turn 0 to save ~2000 tokens per subsequent turn.
    lean_prompt = build_lean_system_prompt(tool_registry, extra=deep_note)

    # Transparency: show user what context the AI agent has
    _show_agent_context(kernels, tool_registry)

    history = []
    total_tokens = 0
    history_records = ChatHistory(model_info=model_info)
    prompt_session: PromptSession[str] = PromptSession()

    # Deep mode state
    ds = _DeepSession()
    if deep:
        from tachyon.analysis.stages import build_stage_prompts
        ds.specs = build_stage_prompts(mode="chat")
        console.print("[cyan]Deep analysis mode[/cyan]: Stage 1/2 — Metric Analysis")
        console.print("[dim]Type /next to jump to next stage, /help for commands.[/dim]")

    # Evolve mode state
    es = _EvolveState()

    _collapsed = True  # Ctrl+O toggle state, persists across turns (default collapsed)

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
            cmd_result = _handle_command(
                user_input, kernels, tool_registry, total_tokens,
                history_records, evolve_state=es,
            )
            if cmd_result is None:
                break  # /quit
            if cmd_result == "NEXT_STAGE" and ds.active:
                msg = ds.advance_stage()
                if msg is None:
                    console.print("[dim]All stages complete.[/dim]")
                    continue
                spec = ds.specs[ds.stage - 1]
                console.rule(f"[bold cyan]{spec.name}[/bold cyan]")
                user_input = msg  # fall through to agent loop
            elif cmd_result == "EVOLVE_ENTER":
                if es.active:
                    console.print("[yellow]Already in evolve mode.[/yellow]")
                    continue
                _enter_evolve_mode(
                    es, tool_registry, kernels, system_prompt, lean_prompt,
                )
                if es.active:
                    console.print(
                        "\n[bold cyan]Evolve mode[/bold cyan] active. "
                        "Use evolve tools to edit, compile, and optimize."
                    )
                    console.print("[dim]Type /evolve-status or /evolve-exit.[/dim]")
                continue
            elif cmd_result == "EVOLVE_STATUS":
                _show_evolve_status(es)
                continue
            elif cmd_result == "EVOLVE_ROLLBACK":
                _evolve_rollback(es)
                continue
            elif cmd_result == "EVOLVE_EXIT":
                if not es.active:
                    console.print("[yellow]Not in evolve mode.[/yellow]")
                    continue
                _exit_evolve_mode(es)
                if not es.active:
                    console.print("[cyan]Exited evolve mode.[/cyan]")
                continue
            else:
                continue

        # Deep mode: first input triggers stage 1
        if ds.active and ds.stage == 0:
            user_input = ds.begin_stage1(user_input)

        # Deep mode stage 2+: prepend original request to follow-ups
        if ds.active:
            user_input = ds.augment_input(user_input)

        # Run agent loop — show tool calls in real time for transparency
        text_buffer = []
        tool_calls_made: list[str] = []
        tool_call_records: list[ToolCallRecord] = []
        pending_tool_args: dict[str, dict] = {}
        done_data: dict = {}
        display = ChatStatusDisplay(timeout=timeout, console=console, collapsed=_collapsed)
        display.start()
        try:
            # Use evolve tools and prompt when in evolve mode
            active_registry = es.tool_registry if es.active else tool_registry
            # C3: Use evolve-specific system prompt when in evolve mode
            active_prompt = es.evolve_system_prompt if es.active and es.evolve_system_prompt else system_prompt
            active_lean = es.original_lean_prompt if es.active else lean_prompt

            loop_kwargs = dict(
                backend=backend,
                registry=active_registry,
                user_message=user_input,
                system_prompt=active_prompt,
                history=history[-10:],
                stream=False,
                timeout=timeout,
                move_timeout=move_timeout,
                lean_system_prompt=active_lean,
            )
            if ds.active:
                loop_kwargs.update(ds.loop_overrides())

            async for event in run_agent_loop(**loop_kwargs):
                if event.type == "text":
                    display.set_synthesizing()
                    text_buffer.append(event.content or "")
                elif event.type == "thinking":
                    if event.content:
                        console.print(f"  [dim italic]{event.content[:120]}[/dim italic]")
                elif event.type == "tool_call":
                    name = event.data["name"] if event.data else "?"
                    display.set_tool(name)
                    args = event.data.get("arguments", {}) if event.data else {}
                    tool_calls_made.append(name)
                    pending_tool_args[name] = args
                    args_short = ", ".join(
                        f"{k}={v}" for k, v in list(args.items())[:3]
                    )
                    display.add_tool_entry(
                        f"  [cyan]\u25b6 {name}[/cyan]({escape(args_short)})"
                    )
                elif event.type == "tool_result":
                    display.set_status("waiting for LLM")
                    if event.data:
                        tc_name = event.data.get("name", "")
                        tool_call_records.append(ToolCallRecord(
                            name=tc_name,
                            arguments=pending_tool_args.pop(tc_name, {}),
                            success=event.data.get("success", False),
                            summary=event.data.get("summary", ""),
                            elapsed=event.data.get("elapsed", 0),
                        ))
                        summary = event.data.get("summary", "")
                        ok = "\u2713" if event.data.get("success") else "\u2717"
                        t = event.data.get("elapsed", 0)
                        display.add_tool_entry(
                            f"    [dim]{ok} {escape(summary[:140])}  ({t:.2f}s)[/dim]"
                        )
                elif event.type == "system":
                    if event.data and "turn" in event.data:
                        # Per-turn timing
                        console.print(f"  [dim]{event.content}[/dim]")
                    elif event.content:
                        console.print(f"  [yellow]{event.content}[/yellow]")
                elif event.type == "done":
                    _collapsed = display.collapsed  # persist toggle state
                    display.stop()
                    display.flush()  # print tool entries to scrollback
                    done_data = event.data or {}
                    if done_data:
                        turns = done_data.get("turns", 0)
                        n_tools = done_data.get("tool_calls", 0)
                        tokens = done_data.get("total_tokens", 0)
                        total_elapsed = done_data.get("total_elapsed", 0)
                        total_tokens += tokens

                        budget = done_data.get("budget", 0)
                        budget_pct = done_data.get("budget_pct", 0)
                        budget_remaining = done_data.get("budget_remaining", 0)
                        budget_str = ""
                        if budget > 0:
                            remaining_pct = round(
                                (budget - budget_remaining) / budget * 100, 1
                            )
                            budget_str = (
                                f", ctx {budget_remaining:,}/{budget:,} "
                                f"({remaining_pct}% used, {100 - remaining_pct:.0f}% remaining)"
                            )

                        console.print(
                            f"  [dim]Done: {turns} turns, {n_tools} tools, "
                            f"{tokens:,} tokens{budget_str}, {total_elapsed:.1f}s[/dim]"
                        )
        finally:
            if not display.stopped:
                _collapsed = display.collapsed
            display.stop()
            display.flush()

        # Render collected text as markdown
        full_text = "".join(text_buffer)

        # Record turn for /export
        usage_rec = None
        if done_data:
            usage_rec = UsageRecord(
                turns=done_data.get("turns", 0),
                tool_calls=done_data.get("tool_calls", 0),
                total_tokens=done_data.get("total_tokens", 0),
                total_elapsed=done_data.get("total_elapsed", 0),
                budget=done_data.get("budget", 0),
                budget_peak=done_data.get("budget_peak", 0),
                budget_remaining=done_data.get("budget_remaining", 0),
                budget_pct=done_data.get("budget_pct", 0),
                compaction_count=done_data.get("compaction_count", 0),
            )
        history_records.record_turn(TurnRecord(
            user_input=user_input,
            ai_response=full_text,
            tool_calls=tool_call_records,
            usage=usage_rec,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ))
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

        # Deep mode: prompt for stage transition after agent completes
        if ds.is_stage1 and full_text:
            console.print(
                "\n[dim]Stage 1 complete. Type [bold]/next[/bold] to proceed to "
                "Stage 2 (Source Attribution + Recommendations), "
                "or ask follow-up questions.[/dim]"
            )


def _handle_command(
    cmd: str,
    kernels: list,
    registry: ToolRegistry,
    total_tokens: int,
    history_records: ChatHistory | None = None,
    evolve_state: _EvolveState | None = None,
) -> bool | None | str:
    """Handle special /commands. Returns None for /quit, True otherwise, 'NEXT_STAGE' for /next."""
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
            "[bold]/export [path][/bold] — Export last turn to markdown\n"
            "[bold]/export-all [path][/bold] — Export all turns to markdown\n"
            "[bold]/next[/bold] — Jump to next analysis stage (deep mode)\n"
            "[bold]/evolve[/bold] — Enter evolve optimization mode\n"
            "[bold]/evolve-status[/bold] — Show evolve session status\n"
            "[bold]/evolve-rollback[/bold] — Rollback last experiment\n"
            "[bold]/evolve-exit[/bold] — Exit evolve mode\n"
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

    if command == "/next":
        return "NEXT_STAGE"

    # --- Evolve mode commands ---
    if command == "/evolve":
        return "EVOLVE_ENTER"

    if command == "/evolve-status":
        return "EVOLVE_STATUS"

    if command == "/evolve-rollback":
        return "EVOLVE_ROLLBACK"

    if command == "/evolve-exit":
        return "EVOLVE_EXIT"

    if command == "/export":
        if history_records is None or not history_records.turns:
            console.print("[red]No turns to export yet.[/red]")
            return True
        from tachyon.chat.export import (
            render_turn_markdown,
            write_export,
        )
        path = parts[1] if len(parts) > 1 else "chat-export.md"
        last = history_records.last_turn()
        if last is None:
            console.print("[red]No turns to export.[/red]")
            return True
        content = render_turn_markdown(last, history_records.model_info)
        try:
            resolved = write_export(content, path)
            console.print(f"[green]Exported to {resolved}[/green]")
        except OSError as e:
            console.print(f"[red]Export failed: {e}[/red]")
        return True

    if command == "/export-all":
        if history_records is None or not history_records.turns:
            console.print("[red]No turns to export yet.[/red]")
            return True
        from tachyon.chat.export import (
            render_all_turns_markdown,
            write_export,
        )
        path = parts[1] if len(parts) > 1 else "chat-export-all.md"
        content = render_all_turns_markdown(
            history_records.turns, history_records.model_info,
        )
        try:
            resolved = write_export(content, path)
            console.print(
                f"[green]Exported {len(history_records.turns)} turns "
                f"to {resolved}[/green]"
            )
        except OSError as e:
            console.print(f"[red]Export failed: {e}[/red]")
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


# ---------------------------------------------------------------------------
# Evolve mode helpers
# ---------------------------------------------------------------------------

def _enter_evolve_mode(
    es: _EvolveState,
    base_registry: ToolRegistry,
    kernels: list,
    system_prompt: str,
    lean_prompt: str,
) -> None:
    """Set up evolve mode: register evolve tools, build evolve system prompt."""
    from tachyon.evolve.config import EvolveConfig
    from tachyon.evolve.context import EvolveContext
    from tachyon.evolve.git import GitRollback
    from tachyon.evolve.persona import build_evolve_system_prompt
    from tachyon.evolve.session import EvolveSession
    from tachyon.evolve.tools import register_evolve_tools

    # Create evolve registry with all existing + evolve tools
    evolve_registry = ToolRegistry()

    # Re-register all existing tools (they capture base ctx via closure)
    for tool_def in base_registry.all_definitions():
        evolve_registry.register(tool_def)

    # Set up evolve infrastructure
    evolve_config = EvolveConfig()
    git = GitRollback()

    if not git.verify_git_repo():
        console.print("[red]Error: Not a git repository. Evolve requires git for rollback.[/red]")
        return

    if not git.check_working_tree_clean():
        auto_commit = git.auto_commit_uncommitted()
        if auto_commit is not None:
            _, short = auto_commit
            console.print(
                f"[yellow]Working tree had uncommitted changes, "
                f"auto-committed as {short}.[/yellow]"
            )

    git.ensure_gitignore_entries()
    git.cleanup_stale_artifacts()

    from tachyon.analyzers.base import AnalyzerRegistry
    from tachyon.tools.context import SessionContext

    analyzer_registry = AnalyzerRegistry()
    analyzer_registry.auto_register()

    base_session = SessionContext(
        kernels=kernels,
        allowed_source_paths=SessionContext.build_allowed_source_paths(kernels),
        registry=analyzer_registry,
    )

    evolve_session = EvolveSession(config=evolve_config)

    evolve_ctx = EvolveContext(
        base=base_session,
        evolve=evolve_session,
        git=git,
        config=evolve_config,
    )

    # Register evolve tools (they capture evolve_ctx via closure)
    register_evolve_tools(evolve_registry, evolve_ctx)

    # Build evolve system prompt
    evolve_prompt = build_evolve_system_prompt(
        evolve_registry,
    )

    es.activate(
        base_registry, evolve_registry, evolve_ctx,
        system_prompt, lean_prompt,
        evolve_system_prompt=evolve_prompt,
    )


def _show_evolve_status(es: _EvolveState) -> None:
    """Display current evolve session status."""
    if not es.active:
        console.print("[yellow]Not in evolve mode. Use /evolve to enter.[/yellow]")
        return

    summary = es.status_summary
    if "active" in summary and not summary["active"]:
        console.print("[yellow]No active evolve session.[/yellow]")
        return

    from rich.table import Table

    table = Table(title="Evolve Status", show_header=False, border_style="cyan")
    table.add_column("Key", style="cyan")
    table.add_column("Value")

    table.add_row("Iteration", f"{summary['current_iteration']} / {summary['max_iterations']}")
    table.add_row("Converged", str(summary["has_converged"]))
    table.add_row("Finished", str(summary["is_finished"]))
    table.add_row("Best Iteration", str(summary["best_iteration"]))
    table.add_row("Best Improvement", f"{summary['best_improvement_pct']:.1f}%")

    if summary.get("experiments_summary"):
        exps = summary["experiments_summary"]
        table.add_row("Experiments", str(len(exps)))
        for e in exps[-5:]:
            table.add_row(f"  Iter {e['iteration']}", f"{e['status']} — {e.get('decision', '')[:40]}")

    console.print(table)


def _evolve_rollback(es: _EvolveState) -> None:
    """Rollback the most recent experiment."""
    if not es.active or es.ctx is None:
        console.print("[yellow]Not in evolve mode.[/yellow]")
        return

    experiments = es.ctx.evolve.experiments
    if not experiments:
        console.print("[yellow]No experiments to rollback.[/yellow]")
        return

    last = experiments[-1]
    if last.git_commit_hash:
        success = es.ctx.git.rollback(last.git_commit_hash)
        if success:
            console.print(f"[green]Rolled back iteration {last.iteration}.[/green]")
        else:
            console.print(f"[red]Rollback failed for iteration {last.iteration}.[/red]")
    else:
        console.print("[yellow]No commit hash for rollback.[/yellow]")


def _exit_evolve_mode(es: _EvolveState) -> None:
    """Exit evolve mode, restore normal tools and prompt."""
    if not es.active:
        return
    es.deactivate()

