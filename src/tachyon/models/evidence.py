"""Evidence chain dataclasses for Conclusion-First reporting.

Three layers of evidence ensure every Finding is traceable
from metric -> source -> SASS instruction.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MetricEvidence:
    """Metric-layer evidence: what the metric says."""

    name: str           # Full NCU metric name
    value: float        # Observed value
    threshold: float    # Threshold that was exceeded
    status: str         # "HIGH" | "LOW" | "OK"
    unit: str = ""      # Unit string (e.g. "%", "byte/cycle")


@dataclass
class SourceEvidence:
    """Source-layer evidence: where in the code."""

    file: str           # Source file path
    line: int           # Line number
    snippet: str        # Source code snippet (from .ncu-rep embedded source)
    function: str | None = None  # Enclosing function name


@dataclass
class SassEvidence:
    """Instruction-layer evidence: what the hardware is doing."""

    pc: int             # Program Counter address
    instruction: str    # SASS disassembly text
    note: str           # Human-readable annotation
    ptx: str | None = None  # Corresponding PTX (if available)


@dataclass
class EvidenceChain:
    """Complete three-layer evidence chain for a Finding.

    Constructed by MarkdownReporter from Finding + SourceHotspot data.
    Used in both Rule-Only and AI-enhanced report modes.
    """

    metric_evidence: list[MetricEvidence] = field(default_factory=list)
    source_evidence: SourceEvidence | None = None
    sass_evidence: SassEvidence | None = None
