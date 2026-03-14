"""OptimizationTree builder and pruning engine.

Consumes Analyzer Findings (not raw metrics) to build a 5-branch
optimization exploration tree, then prunes branches that do not match
the detected bottleneck.

Key design invariant: the OptTree reads ``Finding.source`` and
``Finding.category`` -- thresholds stay in Analyzers.  The tree never
inspects raw NCU metric values.
"""
from __future__ import annotations

from tachyon.models.finding import Finding, Severity
from tachyon.models.opt_tree import OptimizationNode

# ---------------------------------------------------------------------------
# Bottleneck classification constants
# ---------------------------------------------------------------------------

BOTTLENECK_COMPUTE = "compute-bound"
BOTTLENECK_MEMORY = "memory-bound"
BOTTLENECK_LATENCY = "latency-bound"
BOTTLENECK_LOADBAL = "load-balancing-bound"
BOTTLENECK_BALANCED = "well-balanced"

# Internal mapping: Finding title keyword -> bottleneck constant
_TITLE_KEYWORDS: list[tuple[str, str]] = [
    ("compute-bound", BOTTLENECK_COMPUTE),
    ("memory-bound", BOTTLENECK_MEMORY),
    ("latency-bound", BOTTLENECK_LATENCY),
    ("balanced", BOTTLENECK_BALANCED),
]

# Bottleneck -> tree branch category
_BOTTLENECK_CATEGORY: dict[str, str] = {
    BOTTLENECK_COMPUTE: "compute",
    BOTTLENECK_MEMORY: "memory",
    BOTTLENECK_LATENCY: "latency",
    BOTTLENECK_LOADBAL: "load-balancing",
    BOTTLENECK_BALANCED: "balanced",
}

# Sources whose findings are relevant to each branch category
_CATEGORY_SOURCES: dict[str, list[str]] = {
    "compute": ["roofline"],
    "memory": ["memory", "roofline"],
    "latency": ["warp_stall", "roofline"],
    "load-balancing": ["warp_stall"],
    "balanced": ["roofline"],
}


class OptimizationTree:
    """5-branch optimization exploration tree.

    Consumes Analyzer Findings to build and prune the tree.
    Key design: OptTree reads Finding.source and Finding.category,
    NOT raw metrics. Thresholds stay in Analyzers.
    """

    def __init__(self, findings: list[Finding]) -> None:
        self.findings = findings
        self.root = self._build_tree()

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def _build_tree(self) -> OptimizationNode:
        """Build the full 5-branch tree, then prune based on bottleneck."""
        bottleneck = self._extract_bottleneck_from_findings()

        # Build root with all 5 branches
        root = OptimizationNode(
            strategy_name="Optimization Root",
            category="root",
            description=f"Primary bottleneck: {bottleneck}",
            applicable=True,
        )

        compute_branch = self._build_compute_subtree()
        memory_branch = self._build_memory_subtree()
        latency_branch = self._build_latency_subtree()
        loadbal_branch = self._build_loadbalance_subtree()
        balanced_node = self._build_balanced_node()

        branches = [
            compute_branch,
            memory_branch,
            latency_branch,
            loadbal_branch,
            balanced_node,
        ]

        # Determine which branch category is the primary one
        primary_category = _BOTTLENECK_CATEGORY.get(bottleneck, "latency")

        for branch in branches:
            # Populate evidence from findings
            self._populate_evidence(branch, branch.category)

            if branch.category == primary_category:
                branch.applicable = True
            else:
                # Prune non-primary branches
                self._prune_branch(
                    branch,
                    reason=f"Not primary bottleneck (kernel is {bottleneck})",
                )

            root.children.append(branch)

        return root

    def _extract_bottleneck_from_findings(self) -> str:
        """Determine primary bottleneck from roofline findings.

        Looks for a Finding with ``source == "roofline"`` and inspects its
        title for keywords: "compute-bound", "memory-bound",
        "latency-bound", or "balanced".

        Falls back to ``BOTTLENECK_LATENCY`` when no roofline finding exists.
        """
        for f in self.findings:
            if f.source != "roofline":
                continue

            title_lower = f.title.lower()
            for keyword, bottleneck in _TITLE_KEYWORDS:
                if keyword in title_lower:
                    return bottleneck

        # Default fallback: assume latency-bound (most conservative — it
        # opens the widest set of optimisation suggestions).
        return BOTTLENECK_LATENCY

    # ------------------------------------------------------------------
    # Subtree builders
    # ------------------------------------------------------------------

    def _build_compute_subtree(self) -> OptimizationNode:
        """Build the compute-bound branch."""
        instruction_mix = OptimizationNode(
            strategy_name="Instruction Mix",
            category="compute",
            description="Optimise the instruction mix for higher throughput.",
            children=[
                self._leaf("FP32 to FP16 conversion", "compute"),
                self._leaf("Tensor Core utilisation", "compute"),
                self._leaf("FP4/FP6 quantised math", "compute"),
            ],
        )

        pipeline_util = OptimizationNode(
            strategy_name="Pipeline Utilisation",
            category="compute",
            description="Maximise functional-unit throughput.",
            children=[
                self._leaf("FMA pipeline saturation", "compute"),
                self._leaf("ALU/LSU balance", "compute"),
            ],
        )

        divergence = OptimizationNode(
            strategy_name="Thread Divergence",
            category="compute",
            description="Reduce warp-level control-flow divergence.",
            children=[
                self._leaf("Branch divergence reduction", "compute"),
            ],
        )

        return OptimizationNode(
            strategy_name="Compute-Bound Optimisations",
            category="compute",
            description="Strategies for compute-bound kernels.",
            children=[instruction_mix, pipeline_util, divergence],
        )

    def _build_memory_subtree(self) -> OptimizationNode:
        """Build the memory-bound branch."""
        dram = OptimizationNode(
            strategy_name="DRAM Optimisations",
            category="memory",
            description="Reduce off-chip memory traffic.",
            children=[
                self._leaf("Memory compression", "memory"),
                self._leaf("L2 persistence hints", "memory"),
                self._leaf("Reduce data movement", "memory"),
            ],
        )

        l2_cache = OptimizationNode(
            strategy_name="L2 Cache Optimisations",
            category="memory",
            description="Improve L2 cache hit rate.",
            children=[
                self._leaf("Data tiling for L2", "memory"),
                self._leaf("Cache policy tuning", "memory"),
            ],
        )

        l1_shared = OptimizationNode(
            strategy_name="L1/Shared Memory Optimisations",
            category="memory",
            description="Improve near-core memory efficiency.",
            children=[
                self._leaf("Bank conflict elimination", "memory"),
                self._leaf("Global load coalescing", "memory"),
                self._leaf("Shared memory tiling", "memory"),
            ],
        )

        register = OptimizationNode(
            strategy_name="Register Pressure",
            category="memory",
            description="Reduce register usage to improve occupancy.",
            children=[
                self._leaf("Register spill reduction", "memory"),
            ],
        )

        return OptimizationNode(
            strategy_name="Memory-Bound Optimisations",
            category="memory",
            description="Strategies for memory-bound kernels.",
            children=[dram, l2_cache, l1_shared, register],
        )

    def _build_latency_subtree(self) -> OptimizationNode:
        """Build the latency-bound branch."""
        occupancy = OptimizationNode(
            strategy_name="Occupancy",
            category="latency",
            description="Increase warp-level parallelism to hide latency.",
            children=[
                self._leaf("Block size tuning", "latency"),
                self._leaf("Launch bounds annotation", "latency"),
                self._leaf("Shared memory reduction", "latency"),
            ],
        )

        warp_stall = OptimizationNode(
            strategy_name="Warp Stall Reduction",
            category="latency",
            description="Address dominant warp stall reasons.",
            children=[
                self._leaf("Long Scoreboard (prefetch)", "latency"),
                self._leaf("Barrier (reduce sync)", "latency"),
                self._leaf("LG Throttle", "latency"),
                self._leaf("MIO Throttle", "latency"),
            ],
        )

        launch = OptimizationNode(
            strategy_name="Launch Overhead",
            category="latency",
            description="Reduce kernel launch overhead.",
            children=[
                self._leaf("Kernel fusion", "latency"),
            ],
        )

        return OptimizationNode(
            strategy_name="Latency-Bound Optimisations",
            category="latency",
            description="Strategies for latency-bound kernels.",
            children=[occupancy, warp_stall, launch],
        )

    def _build_loadbalance_subtree(self) -> OptimizationNode:
        """Build the load-balancing branch."""
        return OptimizationNode(
            strategy_name="Load-Balancing Optimisations",
            category="load-balancing",
            description="Strategies for load-imbalanced kernels.",
            children=[
                self._leaf("Warp imbalance", "load-balancing"),
                self._leaf("SM imbalance", "load-balancing"),
                self._leaf("Scheduler stall", "load-balancing"),
                self._leaf("Cross-kernel balancing", "load-balancing"),
            ],
        )

    def _build_balanced_node(self) -> OptimizationNode:
        """Build the well-balanced branch."""
        return OptimizationNode(
            strategy_name="Well-Balanced Optimisations",
            category="balanced",
            description="Strategies when no single bottleneck dominates.",
            children=[
                self._leaf("Micro-optimisation", "balanced"),
                self._leaf("Algorithmic restructuring", "balanced"),
            ],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _leaf(name: str, category: str) -> OptimizationNode:
        """Create a leaf OptimizationNode."""
        return OptimizationNode(
            strategy_name=name,
            category=category,
        )

    def _populate_evidence(
        self, branch: OptimizationNode, category_filter: str
    ) -> None:
        """Attach matching Findings as evidence and set speedup estimates.

        A finding matches if:
          - ``f.category == category_filter``, OR
          - ``f.source`` is in the relevant source list for *category_filter*.

        Branches (or leaves) with no evidence after population are pruned
        (unless they have children with evidence).

        Speedup heuristic:
          - CRITICAL finding present -> estimated_speedup = 1.3
          - WARNING finding present  -> estimated_speedup = 1.1
          - INFO only                -> estimated_speedup = 1.05
        """
        relevant_sources = _CATEGORY_SOURCES.get(category_filter, [])

        matching: list[Finding] = []
        for f in self.findings:
            if f.category == category_filter:
                matching.append(f)
            elif f.source in relevant_sources:
                matching.append(f)

        # Attach evidence to the branch node itself
        branch.evidence = matching

        # Set speedup estimate based on highest severity in evidence
        if matching:
            max_severity = max(f.severity for f in matching)
            if max_severity == Severity.CRITICAL:
                branch.estimated_speedup = 1.3
            elif max_severity == Severity.WARNING:
                branch.estimated_speedup = 1.1
            else:
                branch.estimated_speedup = 1.05
            branch.applicable = True

        # Recursively populate children
        for child in branch.children:
            self._populate_evidence(child, category_filter)

        # If this node has no evidence AND none of its children have evidence,
        # prune it (but only if it is not already pruned for other reasons).
        if not branch.pruned and not branch.evidence:
            has_child_evidence = any(
                not c.pruned for c in branch.children
            )
            if not has_child_evidence and branch.children:
                self._prune_branch(branch, reason="No supporting evidence")
            elif not branch.children:
                self._prune_branch(branch, reason="No supporting evidence")

    def _prune_branch(self, node: OptimizationNode, reason: str) -> None:
        """Mark a node and all its descendants as pruned."""
        node.pruned = True
        node.pruned_reason = reason
        node.applicable = False
        for child in node.children:
            self._prune_branch(child, reason=reason)

    # ------------------------------------------------------------------
    # Active path extraction
    # ------------------------------------------------------------------

    def active_paths(self) -> list[list[OptimizationNode]]:
        """Return all non-pruned root-to-leaf paths, sorted by speedup desc.

        Each path is a list of OptimizationNode from root to leaf.
        Paths are sorted by the maximum ``estimated_speedup`` along the path
        (descending), so the most impactful paths come first.
        """
        result: list[list[OptimizationNode]] = []
        self._collect_paths(self.root, [], result)

        # Sort by max speedup along each path (descending)
        def _path_speedup(path: list[OptimizationNode]) -> float:
            speedups = [
                n.estimated_speedup
                for n in path
                if n.estimated_speedup is not None
            ]
            return max(speedups) if speedups else 1.0

        result.sort(key=_path_speedup, reverse=True)
        return result

    def _collect_paths(
        self,
        node: OptimizationNode,
        current: list[OptimizationNode],
        result: list[list[OptimizationNode]],
    ) -> None:
        """Recursive helper: collect non-pruned root-to-leaf paths."""
        if node.pruned:
            return

        path = current + [node]

        if not node.children:
            # Leaf node — emit the full path
            result.append(path)
            return

        # Check if any child is non-pruned
        has_active_child = False
        for child in node.children:
            if not child.pruned:
                has_active_child = True
                self._collect_paths(child, path, result)

        # If all children are pruned, treat this as a leaf
        if not has_active_child:
            result.append(path)

    # ------------------------------------------------------------------
    # Markdown rendering
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        """Render the tree as GitHub-Flavored Markdown.

        Markers:
          - Pruned nodes: ~~strikethrough~~
          - High estimated speedup (>= 1.2): ``>>>`` prefix
          - Nodes with evidence: ``[N findings]`` suffix
        """
        lines: list[str] = []
        lines.append("# Optimization Tree")
        lines.append("")
        lines.append(f"**{self.root.description}**")
        lines.append("")

        for child in self.root.children:
            self._render_node(child, indent=0, lines=lines)

        # Active paths summary
        paths = self.active_paths()
        if paths:
            lines.append("")
            lines.append("## Active Paths (by estimated speedup)")
            lines.append("")
            for i, path in enumerate(paths, 1):
                names = " -> ".join(n.strategy_name for n in path[1:])  # skip root
                speedups = [
                    n.estimated_speedup
                    for n in path
                    if n.estimated_speedup is not None
                ]
                speedup_str = (
                    f"{max(speedups):.2f}x" if speedups else "unknown"
                )
                lines.append(f"{i}. **{speedup_str}**: {names}")

        lines.append("")
        return "\n".join(lines)

    def _render_node(
        self,
        node: OptimizationNode,
        indent: int,
        lines: list[str],
    ) -> None:
        """Recursively render a single node and its children."""
        prefix = "  " * indent

        # Build the node label
        label = node.strategy_name

        # Evidence count suffix
        evidence_suffix = ""
        if node.evidence:
            evidence_suffix = f" [{len(node.evidence)} findings]"

        # Speedup prefix
        speedup_prefix = ""
        if (
            node.estimated_speedup is not None
            and node.estimated_speedup >= 1.2
            and not node.pruned
        ):
            speedup_prefix = ">>> "

        if node.pruned:
            lines.append(
                f"{prefix}- ~~{label}~~ *(pruned: {node.pruned_reason})*"
            )
        else:
            lines.append(
                f"{prefix}- {speedup_prefix}**{label}**{evidence_suffix}"
            )

        for child in node.children:
            self._render_node(child, indent + 1, lines)
