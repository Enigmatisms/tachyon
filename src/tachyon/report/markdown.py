"""Markdown report generator -- GFM format, Conclusion-First structure.

Output order (per architecture doc section 11.3):
1. Metadata (generation time, device, kernel)
2. Executive Summary (always first, Rule-Only capable)
3. Top Findings (by severity, max 5)
4. Evidence Chains (metric -> source -> SASS)
5. Optimization Tree (active paths)
6. Detailed Metrics (--verbose only)

Sections are joined with horizontal rules for clear visual separation.
All user-facing strings go through ``tachyon.i18n.t()`` for localization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from tachyon.i18n import t
from tachyon.models.finding import Finding, Severity

if TYPE_CHECKING:
    from tachyon.models.hotspot import SourceHotspot

# Severity -> text marker (no emoji; plain text markers for GFM compatibility)
_SEVERITY_ICON: dict[str, str] = {
    "critical": "!!!",
    "warning": "!!",
    "info": "i",
}


@dataclass
class ReportContext:
    """All data needed to render a complete markdown report."""

    kernel_name: str
    demangled_name: str
    device_name: str
    compute_capability: str
    findings: list[Finding]
    hotspots: list[SourceHotspot] = field(default_factory=list)
    opt_tree: Any = None  # OptimizationTree (from tachyon.tree.opt_tree)
    duration_us: float | None = None
    degradation_warning: str | None = None
    verbose: bool = False


class MarkdownReporter:
    """Renders analysis results as GitHub Flavored Markdown."""

    def render(self, ctx: ReportContext) -> str:
        """Render full markdown report.

        Args:
            ctx: ReportContext containing all data for the report.

        Returns:
            Complete GFM markdown string.
        """
        sections = [
            self._metadata(ctx),
            self._executive_summary(ctx),
            self._top_findings(ctx),
            self._evidence_chains(ctx),
            self._optimization_tree(ctx),
        ]
        if ctx.verbose:
            sections.append(self._detailed_metrics(ctx))
        # Filter out empty sections before joining
        sections = [s for s in sections if s.strip()]
        return "\n\n---\n\n".join(sections)

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _metadata(self, ctx: ReportContext) -> str:
        """Render report header: title, generation time, device, kernel."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        lines = [
            "# Tachyon Performance Report",
            "",
            f"> Generated: {now}",
            f"> Device: {ctx.device_name} (SM {ctx.compute_capability})",
            f"> Kernel: `{ctx.demangled_name}`",
        ]
        if ctx.duration_us is not None:
            lines.append(f"> Duration: {ctx.duration_us:.1f} us")
        if ctx.degradation_warning:
            lines.append("")
            lines.append(f"> **Warning**: {ctx.degradation_warning}")
        return "\n".join(lines)

    def _executive_summary(self, ctx: ReportContext) -> str:
        """Generate 1-2 sentence summary from Rule Engine findings alone.

        Identifies the primary bottleneck from the RooflineAnalyzer finding
        and counts critical issues.
        """
        lines = [f"## {t('report.executive_summary')}"]
        lines.append("")

        if not ctx.findings:
            lines.append(t("report.no_significant_findings"))
            return "\n".join(lines)

        # Find primary bottleneck from RooflineAnalyzer
        bottleneck = "undetermined"
        for f in ctx.findings:
            if f.source == "roofline":
                bottleneck = f.category or f.title
                break

        # Find top source location from hotspots
        top_loc = ""
        if ctx.hotspots:
            h = ctx.hotspots[0]
            if h.source_file and h.source_file != "<no debug info>":
                top_loc = f", primary hotspot at `{h.source_file}:{h.source_line}`"

        critical_count = sum(
            1 for f in ctx.findings if f.severity == Severity.CRITICAL
        )

        lines.append(
            f"Kernel `{ctx.demangled_name}` is **{bottleneck}**{top_loc}. "
            f"{critical_count} critical issue(s) identified."
        )
        return "\n".join(lines)

    def _top_findings(self, ctx: ReportContext) -> str:
        """Render top-5 findings sorted by severity (CRITICAL first)."""
        lines = [f"## {t('report.top_findings')}"]
        lines.append("")

        if not ctx.findings:
            lines.append(t("report.no_significant_findings"))
            return "\n".join(lines)

        # Sort: CRITICAL > WARNING > INFO
        sorted_findings = sorted(ctx.findings, key=lambda f: f.severity, reverse=True)

        for i, f in enumerate(sorted_findings[:5], 1):
            sev_icon = _SEVERITY_ICON.get(f.severity.value, "")
            lines.append(
                f"### {i}. [{sev_icon}] {t(f.title, fallback=f.title)}"
            )
            lines.append("")
            lines.append(t(f.detail, fallback=f.detail))

            if f.source_location:
                loc = f.source_location
                loc_str = f"`{loc.file}:{loc.line}`"
                if loc.function:
                    loc_str += f" in `{loc.function}`"
                lines.append(f"  - **Location**: {loc_str}")

            if f.sass_evidence:
                lines.append(f"  - **SASS**: `{f.sass_evidence}`")

            lines.append(
                f"  - **Action**: {t(f.action, fallback=f.action)}"
            )
            lines.append("")

        if len(ctx.findings) > 5:
            lines.append(
                f"*({len(ctx.findings) - 5} additional findings omitted)*"
            )

        return "\n".join(lines)

    def _evidence_chains(self, ctx: ReportContext) -> str:
        """Render evidence chains: metric -> source -> SASS for non-INFO findings."""
        lines = [f"## {t('report.evidence_chains')}"]
        lines.append("")

        rendered = 0
        for f in ctx.findings:
            if f.severity == Severity.INFO:
                continue
            if rendered >= 5:
                break

            lines.append(f"### {t(f.title, fallback=f.title)}")
            lines.append("")

            # Metric evidence
            if f.metrics:
                lines.append("**Metrics:**")
                for mname, mval in f.metrics.items():
                    lines.append(f"  - `{mname}` = {mval}")
                lines.append("")

            # Source evidence
            if f.source_location:
                loc = f.source_location
                lines.append(f"**Source:** `{loc.file}:{loc.line}`")
                if loc.function:
                    lines.append(f"  - Function: `{loc.function}`")
                lines.append("")

            # SASS evidence
            if f.sass_evidence:
                lines.append("**SASS:**")
                lines.append("```asm")
                lines.append(f"{f.sass_evidence}")
                lines.append("```")
                lines.append("")

            rendered += 1

        if rendered == 0:
            lines.append(t("report.no_significant_findings"))

        return "\n".join(lines)

    def _optimization_tree(self, ctx: ReportContext) -> str:
        """Render OptTree via its built-in to_markdown() method.

        Falls back to a simple header if no OptTree is available.
        """
        header = f"## {t('report.optimization_tree')}"
        if ctx.opt_tree is not None and hasattr(ctx.opt_tree, "to_markdown"):
            tree_md = ctx.opt_tree.to_markdown()
            if tree_md:
                return f"{header}\n\n{tree_md}"
        return f"{header}\n\n*No optimization tree available.*"

    def _detailed_metrics(self, ctx: ReportContext) -> str:
        """Render full metrics table (--verbose only).

        Aggregates all metrics from all findings into a single GFM table.
        """
        lines = [f"## {t('report.detailed_metrics')}"]
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")

        seen: set[str] = set()
        for f in ctx.findings:
            for mname, mval in f.metrics.items():
                if mname not in seen:
                    lines.append(f"| `{mname}` | {mval} |")
                    seen.add(mname)

        if not seen:
            lines.append("| *(no metrics)* | - |")

        return "\n".join(lines)
