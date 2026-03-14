"""Finding model — unified output format for all Analyzers and NvRules."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(Enum):
    """Finding severity levels, ordered for sorting."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

    def __lt__(self, other: Severity) -> bool:
        order = {Severity.INFO: 0, Severity.WARNING: 1, Severity.CRITICAL: 2}
        return order[self] < order[other]

    def __le__(self, other: Severity) -> bool:
        return self == other or self < other

    def __gt__(self, other: Severity) -> bool:
        return not self <= other

    def __ge__(self, other: Severity) -> bool:
        return not self < other


@dataclass(frozen=True)
class SourceLocation:
    """Source code location reference."""
    file: str
    line: int
    function: str | None = None


@dataclass
class Finding:
    """Analysis finding — the universal output unit.

    Seven core fields (severity through sass_evidence) plus two auxiliary
    fields (category, metrics) for downstream consumption by OptTree and Agent.
    """
    severity: Severity
    title: str
    detail: str
    action: str
    source: str                                     # producing analyzer name
    source_location: SourceLocation | None = None
    sass_evidence: str | None = None
    category: str = ""                              # "memory", "compute", "latency", ...
    metrics: dict[str, float] = field(default_factory=dict)
