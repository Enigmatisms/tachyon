"""Unit tests for OptimizationTree."""
from __future__ import annotations

from tachyon.models.finding import Finding, Severity
from tachyon.tree.opt_tree import (
    BOTTLENECK_BALANCED,
    BOTTLENECK_COMPUTE,
    BOTTLENECK_LATENCY,
    BOTTLENECK_MEMORY,
    OptimizationNode,
    OptimizationTree,
)

# ---------------------------------------------------------------------------
# Helpers — lightweight Finding factories
# ---------------------------------------------------------------------------

def _roofline_finding(title: str, category: str = "compute") -> Finding:
    return Finding(
        severity=Severity.INFO,
        title=title,
        detail="d",
        action="a",
        source="roofline",
        category=category,
    )


def _memory_finding() -> Finding:
    return Finding(
        severity=Severity.WARNING,
        title="Poor coalescing",
        detail="d",
        action="a",
        source="memory",
        category="memory",
    )


def _stall_finding() -> Finding:
    return Finding(
        severity=Severity.WARNING,
        title="High stall",
        detail="d",
        action="a",
        source="warp_stall",
        category="latency",
    )


# ===================================================================
# TestOptimizationNode
# ===================================================================

class TestOptimizationNode:
    """Verify default field values of OptimizationNode."""

    def test_default_fields(self):
        node = OptimizationNode(strategy_name="test", category="compute")
        assert node.pruned is False
        assert node.applicable is False
        assert node.children == []
        assert node.evidence == []
        assert node.estimated_speedup is None
        assert node.pruned_reason is None
        assert node.description == ""


# ===================================================================
# TestExtractBottleneck
# ===================================================================

class TestExtractBottleneck:
    """Verify _extract_bottleneck_from_findings via the public tree."""

    def test_compute_bound(self):
        findings = [_roofline_finding("Kernel is compute-bound", "compute")]
        tree = OptimizationTree(findings)
        # The root description encodes the detected bottleneck
        assert BOTTLENECK_COMPUTE in tree.root.description

    def test_memory_bound(self):
        findings = [_roofline_finding("Kernel is memory-bound", "memory")]
        tree = OptimizationTree(findings)
        assert BOTTLENECK_MEMORY in tree.root.description

    def test_latency_bound(self):
        findings = [_roofline_finding("Kernel is latency-bound", "latency")]
        tree = OptimizationTree(findings)
        assert BOTTLENECK_LATENCY in tree.root.description

    def test_balanced(self):
        findings = [
            _roofline_finding(
                "Kernel is balanced (compute \u2248 memory)", "compute"
            )
        ]
        tree = OptimizationTree(findings)
        assert BOTTLENECK_BALANCED in tree.root.description

    def test_no_roofline_defaults_to_latency(self):
        # Only non-roofline findings present
        findings = [_memory_finding(), _stall_finding()]
        tree = OptimizationTree(findings)
        assert BOTTLENECK_LATENCY in tree.root.description


# ===================================================================
# TestTreeConstruction
# ===================================================================

class TestTreeConstruction:
    """Verify tree structure: 5 branches, pruning logic."""

    def test_five_branches(self):
        findings = [_roofline_finding("Kernel is compute-bound")]
        tree = OptimizationTree(findings)
        assert len(tree.root.children) == 5

    def test_compute_bound_prunes_others(self):
        findings = [_roofline_finding("Kernel is compute-bound", "compute")]
        tree = OptimizationTree(findings)
        branches = {c.category: c for c in tree.root.children}

        # Compute branch is the primary — not pruned at top level
        assert branches["compute"].pruned is False
        # Other branches should be pruned
        assert branches["memory"].pruned is True
        assert branches["latency"].pruned is True
        assert branches["load-balancing"].pruned is True
        assert branches["balanced"].pruned is True

    def test_memory_bound_prunes_others(self):
        findings = [
            _roofline_finding("Kernel is memory-bound", "memory"),
            _memory_finding(),
        ]
        tree = OptimizationTree(findings)
        branches = {c.category: c for c in tree.root.children}

        assert branches["memory"].pruned is False
        assert branches["compute"].pruned is True
        assert branches["latency"].pruned is True

    def test_latency_bound_tree(self):
        stall = _stall_finding()
        roofline = _roofline_finding("Kernel is latency-bound", "latency")
        tree = OptimizationTree([roofline, stall])
        branches = {c.category: c for c in tree.root.children}

        latency_branch = branches["latency"]
        assert latency_branch.pruned is False
        # Evidence should contain our warp_stall finding
        assert stall in latency_branch.evidence


# ===================================================================
# TestActivePaths
# ===================================================================

class TestActivePaths:
    """Verify active_paths() extraction and ordering."""

    def test_active_paths_compute(self):
        findings = [_roofline_finding("Kernel is compute-bound", "compute")]
        tree = OptimizationTree(findings)
        paths = tree.active_paths()

        # All active paths should go through the compute branch (index [1])
        for path in paths:
            # path[0] is root, path[1] should be the compute branch
            assert len(path) >= 2
            branch_categories = {n.category for n in path[1:]}
            assert "compute" in branch_categories

    def test_active_paths_sorted_by_speedup(self):
        # Create findings that produce different speedup estimates
        critical_finding = Finding(
            severity=Severity.CRITICAL,
            title="Kernel is memory-bound",
            detail="d",
            action="a",
            source="roofline",
            category="memory",
        )
        warning_finding = _memory_finding()
        tree = OptimizationTree([critical_finding, warning_finding])
        paths = tree.active_paths()

        if len(paths) >= 2:
            # Extract speedup for each path
            def path_speedup(path):
                speedups = [
                    n.estimated_speedup
                    for n in path
                    if n.estimated_speedup is not None
                ]
                return max(speedups) if speedups else 1.0

            speedups = [path_speedup(p) for p in paths]
            # Should be sorted descending
            assert speedups == sorted(speedups, reverse=True)

    def test_active_paths_empty_findings(self):
        tree = OptimizationTree([])
        paths = tree.active_paths()
        # With no findings, all branches get pruned for lack of evidence
        # (after the primary latency branch is also pruned for no evidence).
        # Result: active_paths may be empty or very short.
        assert isinstance(paths, list)


# ===================================================================
# TestToMarkdown
# ===================================================================

class TestToMarkdown:
    """Verify markdown output structure."""

    def test_markdown_has_pruned(self):
        findings = [_roofline_finding("Kernel is compute-bound", "compute")]
        tree = OptimizationTree(findings)
        md = tree.to_markdown()
        # Non-primary branches are pruned — markdown uses ~~strikethrough~~
        assert "pruned" in md.lower()

    def test_markdown_has_findings_count(self):
        findings = [
            _roofline_finding("Kernel is memory-bound", "memory"),
            _memory_finding(),
        ]
        tree = OptimizationTree(findings)
        md = tree.to_markdown()
        # Memory branch should have evidence → "[N findings]" marker
        assert "findings]" in md

    def test_markdown_has_active_paths_section(self):
        findings = [
            _roofline_finding("Kernel is compute-bound", "compute"),
        ]
        tree = OptimizationTree(findings)
        md = tree.to_markdown()
        assert "Active Paths" in md
