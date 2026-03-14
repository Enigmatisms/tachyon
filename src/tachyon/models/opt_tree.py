"""OptimizationNode model -- core data structure for the Optimization Tree."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OptimizationNode:
    """A node in the Optimization Tree.

    Each node represents either a strategy category (internal node) or a
    concrete optimisation action (leaf).  The tree is built by the
    ``OptimizationTree`` builder (``tachyon.tree.opt_tree``) and pruned so
    that only branches relevant to the detected bottleneck remain active.

    Fields:
        strategy_name: Human-readable name of this strategy / category.
        category: One of "compute", "memory", "latency", "load-balancing",
                  "balanced", or "root".
        description: Optional longer explanation of the strategy.
        applicable: Whether this node is relevant to the current kernel.
        pruned: Whether this node has been pruned (not on the active path).
        pruned_reason: Human-readable reason if pruned.
        estimated_speedup: Multiplicative speedup estimate (e.g. 1.3 = 30%).
        children: Child nodes (empty for leaves).
        evidence: Finding objects attached as evidence for this node.
    """

    strategy_name: str
    category: str  # "compute", "memory", "latency", "load-balancing", "balanced", "root"
    description: str = ""
    applicable: bool = False
    pruned: bool = False
    pruned_reason: str | None = None
    estimated_speedup: float | None = None  # e.g. 1.3 = 30% estimated speedup
    children: list[OptimizationNode] = field(default_factory=list)
    evidence: list[Any] = field(default_factory=list)  # Finding objects as evidence
