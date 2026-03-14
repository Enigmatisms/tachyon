"""ProfileDiffer — compares two lists of KernelReport.

Matches kernels by ``kernel_name`` and computes per-metric deltas,
percentage changes, and a simple regression heuristic based on whether
each metric is higher-is-better or lower-is-better.

Usage::

    differ = ProfileDiffer()
    diffs = differ.diff(before_kernels, after_kernels)
    print(differ.summary(diffs))
"""
from __future__ import annotations

from dataclasses import dataclass

from tachyon.models.kernel import KernelReport


@dataclass(frozen=True)
class MetricDelta:
    """Change in a single metric between two reports."""

    name: str
    before: float
    after: float
    delta: float          # after - before
    delta_pct: float      # (after - before) / before * 100 if before != 0, else 0.0


@dataclass
class KernelDiff:
    """Diff result for a single kernel."""

    kernel_name: str
    demangled_name: str
    metric_deltas: list[MetricDelta]
    only_in_before: list[str]   # metric names only in before
    only_in_after: list[str]    # metric names only in after

    @property
    def significant_changes(self) -> list[MetricDelta]:
        """Metrics with abs(delta_pct) > 5%."""
        return [m for m in self.metric_deltas if abs(m.delta_pct) > 5.0]

    @property
    def regressions(self) -> list[MetricDelta]:
        """Throughput metrics that decreased, or latency metrics that increased.

        Simple heuristic: metrics with 'throughput' or 'pct' in name are
        higher-is-better. Others (duration, stall) are lower-is-better.
        """
        result: list[MetricDelta] = []
        for m in self.metric_deltas:
            if abs(m.delta_pct) <= 5.0:
                continue
            higher_is_better = any(
                kw in m.name
                for kw in ["throughput", "pct", "utilization", "hit_rate"]
            )
            if higher_is_better and m.delta < 0:
                result.append(m)
            elif not higher_is_better and m.delta > 0:
                result.append(m)
        return result


class ProfileDiffer:
    """Compare two lists of KernelReport.

    Usage::

        differ = ProfileDiffer()
        diffs = differ.diff(before_kernels, after_kernels)
    """

    def diff(
        self,
        before: list[KernelReport],
        after: list[KernelReport],
    ) -> list[KernelDiff]:
        """Compare kernel reports by matching kernel_name.

        Kernels present in both *before* and *after* are compared.
        Unmatched kernels are noted but not diffed.
        """
        before_map = {k.kernel_name: k for k in before}
        after_map = {k.kernel_name: k for k in after}

        common_names = set(before_map.keys()) & set(after_map.keys())

        diffs: list[KernelDiff] = []
        for name in sorted(common_names):
            b = before_map[name]
            a = after_map[name]
            diffs.append(self._diff_kernel(b, a))

        return diffs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _diff_kernel(
        self,
        before: KernelReport,
        after: KernelReport,
    ) -> KernelDiff:
        """Compare metrics between two reports of the same kernel."""
        b_metrics = set(before.metrics.keys())
        a_metrics = set(after.metrics.keys())
        common = sorted(b_metrics & a_metrics)

        deltas: list[MetricDelta] = []
        for metric_name in common:
            bv = before.metrics[metric_name].value
            av = after.metrics[metric_name].value
            delta = av - bv
            delta_pct = (delta / bv * 100) if bv != 0 else 0.0
            deltas.append(
                MetricDelta(
                    name=metric_name,
                    before=bv,
                    after=av,
                    delta=delta,
                    delta_pct=delta_pct,
                )
            )

        return KernelDiff(
            kernel_name=before.kernel_name,
            demangled_name=before.demangled_name,
            metric_deltas=deltas,
            only_in_before=sorted(b_metrics - a_metrics),
            only_in_after=sorted(a_metrics - b_metrics),
        )

    def summary(self, diffs: list[KernelDiff]) -> str:
        """Generate a compact text summary of differences."""
        lines: list[str] = []
        for d in diffs:
            sig = d.significant_changes
            reg = d.regressions
            if not sig:
                lines.append(f"  {d.demangled_name}: no significant changes")
                continue
            lines.append(
                f"  {d.demangled_name}: "
                f"{len(sig)} significant changes, {len(reg)} regressions"
            )
            for m in sig[:10]:  # top 10
                arrow = "\u2191" if m.delta > 0 else "\u2193"
                lines.append(
                    f"    {m.name}: {m.before:.2f} \u2192 {m.after:.2f} "
                    f"({arrow}{abs(m.delta_pct):.1f}%)"
                )

        return "\n".join(lines) if lines else "No differences found."
