"""NvRules Adapter — convert NCU built-in rule results into Findings.

NCU's built-in rules produce ``RuleResult`` objects (stored in
``KernelReport.rule_results``) with severity strings "OK", "LOW", "MED",
"HIGH".  This adapter maps them into the unified Finding model so that
downstream consumers (OptTree, Report) see a single finding stream.
"""
from __future__ import annotations

from tachyon.analyzers.base import Analyzer
from tachyon.models.finding import Finding, Severity
from tachyon.models.kernel import KernelReport

# Severity mapping from NCU rule severity strings to Tachyon Severity.
_SEVERITY_MAP: dict[str, Severity] = {
    "LOW": Severity.INFO,
    "MED": Severity.WARNING,
    "HIGH": Severity.CRITICAL,
}


class NvRulesAdapter(Analyzer):
    """Wrap NCU rule_results as first-class Findings."""

    def name(self) -> str:
        return "nvrules"

    def category(self) -> str:
        return "nvrules"

    def required_metrics(self) -> list[str]:
        # NvRules does not depend on individual metrics; it reads
        # pre-computed rule_results.
        return []

    def can_run(self, report: KernelReport) -> bool:
        """Override: runnable when there are rule results (not metric-based)."""
        return len(report.rule_results) > 0

    def analyze(self, report: KernelReport) -> list[Finding]:
        findings: list[Finding] = []
        for rr in report.rule_results:
            # Skip "OK" results — they carry no actionable information.
            if rr.severity == "OK":
                continue

            severity = _SEVERITY_MAP.get(rr.severity)
            if severity is None:
                # Unknown severity string — default to INFO to avoid losing data.
                severity = Severity.INFO

            findings.append(
                Finding(
                    severity=severity,
                    title=f"[NCU] {rr.rule_name}",
                    detail=rr.message,
                    action="See NCU rule documentation for detailed guidance.",
                    source=self.name(),
                    category=self.category(),
                )
            )

        return findings
