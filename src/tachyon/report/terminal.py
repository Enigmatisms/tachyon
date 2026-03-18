"""TerminalReporter — Rich-based terminal output with Conclusion-First ordering.

Layout for each kernel:
1. Kernel header (name, grid, block, registers, shared mem)
2. Metrics Overview — key performance indicators at a glance
3. Key Findings Tree — top CRITICAL/WARNING findings with severity badge
4. Detailed Findings — each finding with quantitative detail + metrics + action

Uses Rich library: Panel, Table, Tree, Text, Console, with color-coded severity.
String-buffered output enables both stdout display and --output file writing.
"""
from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING

from rich.columns import Columns
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

# Severity → Rich markup badge
_SEVERITY_BADGE: dict[str, str] = {
    "critical": "[red]CRITICAL[/red]",
    "warning": "[yellow]WARNING[/yellow]",
    "info": "[dim]INFO[/dim]",
}

# Key metrics to show in the overview panel (metric_name → display_label)
_OVERVIEW_METRICS: list[tuple[str, str, str]] = [
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "SM Throughput", "%"),
    ("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", "DRAM Throughput", "%"),
    ("gpu__time_duration.sum", "Duration", "ns"),
    ("launch__occupancy_limit_registers", "Occupancy (reg limit)", "%"),
    ("sm__warps_active.avg.pct_of_peak_sustained_active", "Active Warps", "%"),
]


class TerminalReporter:
    """Renders analysis results to terminal using Rich."""

    def render(
        self,
        reports: list[KernelReport],
        findings_map: dict[str, list[Finding]],
    ) -> str:
        """Render all kernel reports to a string.

        Args:
            reports: List of KernelReport objects to render.
            findings_map: Mapping from demangled kernel name to its findings.

        Returns:
            Complete rendered string with ANSI escape codes.
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
        """Convenience: render just one kernel to a string."""
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120)
        self._render_kernel(console, report, findings)
        return buf.getvalue()

    def _render_kernel(
        self, console: Console, report: KernelReport, findings: list[Finding]
    ) -> None:
        """Render a single kernel's analysis.

        Layout:
        1. Header Panel — kernel identity + launch config
        2. Metrics Overview — key performance numbers
        3. Key Findings Tree — top CRITICAL/WARNING (max 5)
        4. Detailed Findings — each with quantitative detail
        """
        # ── 1. Kernel Header ──
        lp = report.launch_params
        header = (
            f"[bold]{report.demangled_name}[/bold]\n"
            f"Grid: {lp.grid}  Block: {lp.block}  "
            f"Regs: {lp.registers_per_thread}/thread  "
            f"Shared: {lp.shared_mem_bytes}B"
        )
        if hasattr(report, 'device_info') and report.device_info:
            header += f"\nDevice: {report.device_info.name}"
        console.print(Panel(header, title="[bold cyan]Kernel[/bold cyan]", border_style="cyan"))

        # ── 2. Metrics Overview ──
        overview_items: list[tuple[str, str]] = []
        for metric_name, label, unit in _OVERVIEW_METRICS:
            val = report.metric_value(metric_name)
            if val is not None:
                if unit == "%" :
                    overview_items.append((label, f"{val:.1f}%"))
                elif unit == "ns" and val > 1e6:
                    overview_items.append((label, f"{val / 1e6:.2f} ms"))
                elif unit == "ns" and val > 1e3:
                    overview_items.append((label, f"{val / 1e3:.1f} us"))
                else:
                    overview_items.append((label, f"{val:.1f} {unit}"))

        if overview_items:
            mtable = Table.grid(padding=(0, 3))
            mtable.add_column(style="bold cyan")
            mtable.add_column()
            for label, val_str in overview_items:
                mtable.add_row(f"{label}:", val_str)
            console.print(Panel(mtable, title="[bold]Metrics Overview[/bold]", border_style="dim", expand=False))

        if not findings:
            console.print("  [dim]No significant findings.[/dim]\n")
            return

        # ── 3. Key Findings Tree (conclusion-first) ──
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

        # ── 4. Detailed Findings ──
        for i, f in enumerate(findings):
            self._render_finding(console, f, i + 1)

        console.print()

    def _render_finding(
        self, console: Console, f: Finding, index: int
    ) -> None:
        """Render a single finding with full quantitative detail."""
        style = _SEVERITY_STYLE.get(f.severity.value, "")
        badge = _SEVERITY_BADGE.get(f.severity.value, f.severity.value)

        # Title line
        console.print(
            f"  {badge}  [bold]{f.title}[/bold]",
            style="" if style == "dim" else "",
        )

        # Detail (the quantitative narrative — this was previously hidden!)
        if f.detail:
            for line in f.detail.split("\n"):
                console.print(f"    {line}", style="dim" if f.severity.value == "info" else "")

        # Metrics key-values (raw numbers for precision)
        if f.metrics:
            parts: list[str] = []
            for k, v in f.metrics.items():
                short_name = k.split(".")[-1] if "." in k else k
                if "pct" in k or k.endswith("_pct"):
                    parts.append(f"{short_name}={v:.1f}%")
                elif v > 1e6:
                    parts.append(f"{short_name}={v:.0f}")
                elif v == int(v):
                    parts.append(f"{short_name}={int(v)}")
                else:
                    parts.append(f"{short_name}={v:.2f}")
            console.print(f"    [cyan]Metrics:[/cyan] {', '.join(parts)}")

        # Source location (M2)
        if f.source_location:
            loc = f.source_location
            loc_str = f"{loc.file}:{loc.line}"
            if loc.function:
                loc_str += f" ({loc.function})"
            console.print(f"    [dim]Source:[/dim] {loc_str}")

        # SASS evidence (M2)
        if f.sass_evidence:
            console.print(f"    [dim]SASS:[/dim] {f.sass_evidence}")

        # Action
        if f.action:
            console.print(f"    [green]Action:[/green] {f.action}")

        console.print(f"    [dim]Analyzer: {f.source}[/dim]")
        console.print()
