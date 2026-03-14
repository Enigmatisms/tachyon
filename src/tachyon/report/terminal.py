"""TerminalReporter — Rich-based terminal output with Conclusion-First ordering.

Layout for each kernel:
1. Kernel header (name, grid, block, registers, shared mem)
2. Conclusion box: top 1-5 CRITICAL/WARNING findings with severity badge
3. Detailed findings table (Sev, Finding, Action, Source columns)

Uses Rich library: Panel, Table, Tree, Text, Console, with color-coded severity.
String-buffered output enables both stdout display and --output file writing.
"""
from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

if TYPE_CHECKING:
    from tachyon.models.finding import Finding
    from tachyon.models.kernel import KernelReport

# Severity → Rich style mapping
_SEVERITY_STYLE: dict[str, str] = {
    "critical": "bold red",
    "warning": "bold yellow",
    "info": "dim",
}

# Severity → Rich markup badge for tree display
_SEVERITY_BADGE: dict[str, str] = {
    "critical": "[red]CRITICAL[/red]",
    "warning": "[yellow]WARNING[/yellow]",
    "info": "[dim]INFO[/dim]",
}


class TerminalReporter:
    """Renders analysis results to terminal using Rich."""

    def render(
        self,
        reports: list[KernelReport],
        findings_map: dict[str, list[Finding]],
    ) -> str:
        """Render all kernel reports to a string (for both stdout and file output).

        Args:
            reports: List of KernelReport objects to render.
            findings_map: Mapping from demangled kernel name to its findings.

        Returns:
            Complete rendered string with ANSI escape codes for terminal display.
        """
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120)

        console.print()
        console.rule("[bold]Tachyon Performance Analysis[/bold]")
        console.print(f"  Kernels analyzed: {len(reports)}\n")

        for report in reports:
            findings = findings_map.get(report.demangled_name, [])
            self._render_kernel(console, report, findings)

        return buf.getvalue()

    def render_single_kernel(
        self, report: KernelReport, findings: list[Finding]
    ) -> str:
        """Convenience: render just one kernel to a string.

        Useful for tests and single-kernel display scenarios.
        """
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120)
        self._render_kernel(console, report, findings)
        return buf.getvalue()

    def _render_kernel(
        self, console: Console, report: KernelReport, findings: list[Finding]
    ) -> None:
        """Render a single kernel's analysis to the given console.

        Layout:
        1. Header Panel — kernel name + launch configuration
        2. Key Findings Tree — top CRITICAL/WARNING (max 5), conclusion-first
        3. Detailed Findings Table — all findings with severity, action, source
        """
        # ── 1. Kernel Header Panel ──
        header = (
            f"[bold]{report.demangled_name}[/bold]\n"
            f"Grid: {report.launch_params.grid} | "
            f"Block: {report.launch_params.block} | "
            f"Registers: {report.launch_params.registers_per_thread}/thread | "
            f"Shared: {report.launch_params.shared_mem_bytes}B"
        )
        console.print(Panel(header, title="Kernel", border_style="cyan"))

        if not findings:
            console.print("  [dim]No significant findings.[/dim]\n")
            return

        # ── 2. Conclusion-First: Key Findings Tree ──
        critical_and_warning = [
            f for f in findings if f.severity.value in ("critical", "warning")
        ]
        if critical_and_warning:
            tree = Tree("[bold]Key Findings[/bold]")
            for f in critical_and_warning[:5]:
                badge = _SEVERITY_BADGE.get(f.severity.value, "")
                tree.add(f"{badge} {f.title}")
            console.print(tree)
            console.print()

        # ── 3. Detailed Findings Table ──
        table = Table(show_header=True, header_style="bold", expand=True)
        table.add_column("Sev", width=10, justify="center")
        table.add_column("Finding", ratio=3)
        table.add_column("Action", ratio=2)
        table.add_column("Source", width=20)

        for f in findings:
            style = _SEVERITY_STYLE.get(f.severity.value, "")
            # Truncate long action text to keep table readable
            action_text = f.action[:100] + "..." if len(f.action) > 100 else f.action
            table.add_row(
                Text(f.severity.value.upper(), style=style),
                f.title,
                action_text,
                f.source,
            )

        console.print(table)
        console.print()
