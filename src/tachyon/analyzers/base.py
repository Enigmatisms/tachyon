"""Analyzer base class and registry -- plug-in architecture for rule engine.

M2 enhancement: optional SourceCorrelator injection for source-level
attribution.
"""
from __future__ import annotations

import logging
import traceback
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from tachyon.models.finding import Finding, Severity, SourceLocation
from tachyon.models.kernel import KernelReport

if TYPE_CHECKING:
    from tachyon.correlator.source_correlator import (
        ActionHandle,
        SourceCorrelator,
    )
    from tachyon.models.hotspot import SourceHotspot

logger = logging.getLogger(__name__)


class Analyzer(ABC):
    """Abstract base class for all performance analyzers.

    Subclasses must implement four methods:
      - name()             — unique identifier (e.g. "roofline")
      - category()         — grouping tag (e.g. "compute", "memory")
      - required_metrics() — list of metric names needed
      - analyze(report)    — produce a list of Findings

    M2 additions: optional SourceCorrelator injection for source attribution.
    If inject_correlator() is never called, analyzers operate in M1 mode
    (no source attribution).
    """

    def __init__(self) -> None:
        self._correlator: SourceCorrelator | None = None
        self._hotspots: list[SourceHotspot] | None = None

    @abstractmethod
    def name(self) -> str:
        """Unique short name for this analyzer."""

    @abstractmethod
    def category(self) -> str:
        """Category tag (memory, compute, latency, ...)."""

    @abstractmethod
    def required_metrics(self) -> list[str]:
        """Metric names that *must* be present for analyze() to run."""

    @abstractmethod
    def analyze(self, report: KernelReport) -> list[Finding]:
        """Run analysis and return findings."""

    def can_run(self, report: KernelReport) -> bool:
        """Check whether *report* contains all required metrics."""
        return report.has_metrics(self.required_metrics())

    # ── M2: Source-level attribution helpers ──

    def inject_correlator(self, correlator: SourceCorrelator) -> None:
        """Inject SourceCorrelator instance for source-level attribution.

        Called by AnalyzerRegistry before analyze(). If not called,
        Analyzer operates in M1 mode (no source attribution).
        """
        self._correlator = correlator

    def set_hotspots(self, hotspots: list[SourceHotspot]) -> None:
        """Pre-computed hotspots from SourceCorrelator.correlate().

        AnalyzerRegistry calls correlate() once, then distributes hotspots
        to all analyzers to avoid redundant computation.
        """
        self._hotspots = hotspots

    def _find_hotspot_for_stall(
        self, stall_keyword: str
    ) -> SourceHotspot | None:
        """Find the hottest source line for a given stall type.

        Args:
            stall_keyword: Substring to match in stall metric names,
                          e.g. "long_scoreboard", "barrier".

        Returns:
            The SourceHotspot with the highest ratio for that stall type,
            or None if no hotspot matches.
        """
        if not self._hotspots:
            return None
        best: SourceHotspot | None = None
        best_ratio = 0.0
        for h in self._hotspots:
            for k, v in h.stall_reasons.items():
                if stall_keyword in k and v > best_ratio:
                    best = h
                    best_ratio = v
        return best

    def _attach_source_evidence(
        self, finding: Finding, hotspot: SourceHotspot | None
    ) -> Finding:
        """Enrich a Finding with source location and SASS evidence from a hotspot.

        If hotspot is None or degraded, leaves Finding fields unchanged
        (M1 backward compatibility).
        """
        if hotspot is None:
            return finding
        if not getattr(hotspot, "degraded", False):
            finding.source_location = SourceLocation(
                file=hotspot.source_file,
                line=hotspot.source_line,
            )
        if hotspot.sass_instruction:
            finding.sass_evidence = hotspot.sass_instruction
        return finding


class AnalyzerRegistry:
    """Registry and pipeline runner for Analyzers.

    Usage:
        registry = AnalyzerRegistry()
        registry.auto_register()
        findings = registry.run_all(report)

    M2 enhancement: if correlator and action are provided, computes
    SourceHotspots once and distributes to all analyzers.
    """

    def __init__(self) -> None:
        self._analyzers: list[Analyzer] = []

    def register(self, analyzer: Analyzer) -> None:
        """Register a single analyzer instance."""
        self._analyzers.append(analyzer)
        logger.debug("Registered analyzer: %s", analyzer.name())

    def auto_register(self, include_nvrules: bool = False) -> None:
        """Import and register all built-in analyzers.

        Import here (not at module level) to avoid circular imports and
        to make registration explicit and testable.

        Args:
            include_nvrules: If True, also register NvRulesAdapter which
                converts NCU built-in rule results into Findings. Disabled
                by default because the built-in analyzers cover the same
                domains with richer quantitative detail. Enable when you
                want raw NCU rule output (e.g. for debugging or comparison).
        """
        from tachyon.analyzers.instruction import InstructionAnalyzer
        from tachyon.analyzers.launch import LaunchConfigAnalyzer
        from tachyon.analyzers.memory import MemoryAnalyzer
        from tachyon.analyzers.occupancy import OccupancyAnalyzer
        from tachyon.analyzers.roofline import RooflineAnalyzer
        from tachyon.analyzers.warp_stall import WarpStallAnalyzer

        for cls in [
            RooflineAnalyzer,
            MemoryAnalyzer,
            WarpStallAnalyzer,
            OccupancyAnalyzer,
            InstructionAnalyzer,
            LaunchConfigAnalyzer,
        ]:
            self.register(cls())

        if include_nvrules:
            from tachyon.analyzers.nvrules import NvRulesAdapter
            self.register(NvRulesAdapter())

    def all_analyzers(self) -> list[Analyzer]:
        """Return a copy of the registered analyzer list."""
        return list(self._analyzers)

    def run_all(
        self,
        report: KernelReport,
        action: ActionHandle | None = None,
        correlator: SourceCorrelator | None = None,
    ) -> list[Finding]:
        """Execute all registered analyzers against a KernelReport.

        Fault-tolerant: a failing analyzer emits a diagnostic Finding
        and the pipeline continues. Findings are returned sorted by severity
        (CRITICAL first).

        M2 enhancement: if correlator and action are provided, computes
        SourceHotspots once and distributes to all analyzers for
        source-level attribution.
        """
        # M2: compute hotspots once, share with all analyzers
        hotspots: list[SourceHotspot] = []
        if correlator is not None and action is not None:
            try:
                instanced = report.instanced_metrics_as_tuples()
                hotspots = correlator.correlate(
                    action, instanced, kernel_name=report.kernel_name,
                )
            except Exception:
                logger.warning(
                    "SourceCorrelator failed, continuing without source attribution: %s",
                    traceback.format_exc(),
                )

        findings: list[Finding] = []
        skipped: list[str] = []

        for analyzer in self._analyzers:
            if not analyzer.can_run(report):
                missing = (
                    set(analyzer.required_metrics())
                    - set(report.metrics.keys())
                )
                logger.debug(
                    "Skipping '%s': missing metrics %s", analyzer.name(), missing
                )
                skipped.append(analyzer.name())
                continue

            # M2: inject correlator + hotspots before analyze()
            if correlator is not None:
                analyzer.inject_correlator(correlator)
                analyzer.set_hotspots(hotspots)

            try:
                results = analyzer.analyze(report)
                findings.extend(results)
                logger.debug(
                    "Analyzer '%s' produced %d findings", analyzer.name(), len(results)
                )
            except Exception:
                tb = traceback.format_exc()
                logger.warning("Analyzer '%s' failed: %s", analyzer.name(), tb)
                findings.append(Finding(
                    severity=Severity.INFO,
                    title=f"Analyzer execution error: {analyzer.name()}",
                    detail=tb,
                    action="Check if the report contains the required NCU sections.",
                    source=analyzer.name(),
                    category=analyzer.category(),
                ))

        # Add user-visible diagnostic when analyzers were skipped
        if skipped:
            total = len(self._analyzers)
            ran = total - len(skipped)
            skipped_list = ", ".join(skipped)
            findings.append(Finding(
                severity=Severity.INFO,
                title=f"Metric coverage: {ran}/{total} analyzers ran ({len(skipped)} skipped)",
                detail=(
                    f"Skipped analyzers: {skipped_list}. "
                    "These analyzers require NCU metrics not present in the report. "
                    "The current profiling set may not include the required sections."
                ),
                action=(
                    "Re-profile with `--ncu-set detailed` (or `--ncu-set full`) to "
                    "collect all metrics. For example: "
                    "`tachyon profile --ncu-set full ./app`"
                ),
                source="registry",
                category="diagnostic",
            ))

        # Sort: CRITICAL > WARNING > INFO
        findings.sort(key=lambda f: f.severity, reverse=True)
        return findings
