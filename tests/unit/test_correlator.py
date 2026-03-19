"""Unit tests for SourceCorrelator."""
from unittest.mock import MagicMock

from tachyon.correlator.source_correlator import (
    CorrelatorConfig,
    LineAccumulator,
    PcAccumulator,
    SourceCorrelator,
    SourceInfo,
)
from tachyon.models.hotspot import SourceHotspot

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_mock_action(pc_source_map: dict[int, tuple[str, int]]) -> MagicMock:
    """Create a mock ActionHandle with configurable PC -> source mapping.

    Args:
        pc_source_map: Maps ``{pc: (file_name, line_number)}``.
            PCs not in the map produce ``source_info(...) -> None``.

    Returns:
        A ``MagicMock`` that satisfies the ``ActionHandle`` protocol.
    """
    action = MagicMock()

    def _source_info(pc: int, kernel_name: str | None = None):
        if pc in pc_source_map:
            f, l = pc_source_map[pc]
            return SourceInfo(file_name=f, line=l)
        return None

    def _sass_by_pc(pc: int, kernel_name: str | None = None):
        return f"SASS@0x{pc:x}"

    def _ptx_by_pc(pc: int, kernel_name: str | None = None):
        return f"PTX@0x{pc:x}"

    action.source_info.side_effect = _source_info
    action.sass_by_pc.side_effect = _sass_by_pc
    action.ptx_by_pc.side_effect = _ptx_by_pc
    return action


# ━━━━━━━━━━━━━━━━━━━━━━━ PcAccumulator ━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPcAccumulator:
    """Tests for the per-PC metric accumulator."""

    def test_add_stall_sums(self):
        """Stall metrics should be SUMMED when add_sample is called twice."""
        acc = PcAccumulator(pc=0x100)
        stall = "smsp__pcsamp_warps_issue_stalled_barrier"
        acc.add_sample(stall, 10.0)
        acc.add_sample(stall, 25.0)
        assert acc.samples[stall] == 35.0

    def test_add_throughput_maxes(self):
        """Non-stall / non-exec metrics should keep the MAX."""
        acc = PcAccumulator(pc=0x200)
        metric = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
        acc.add_sample(metric, 40.0)
        acc.add_sample(metric, 70.0)
        acc.add_sample(metric, 55.0)
        assert acc.samples[metric] == 70.0

    def test_total_stall_samples(self):
        """total_stall_samples should sum only stall metrics."""
        acc = PcAccumulator(pc=0x300)
        acc.add_sample("smsp__pcsamp_warps_issue_stalled_barrier", 10.0)
        acc.add_sample("smsp__pcsamp_warps_issue_stalled_wait", 30.0)
        # Non-stall metrics should be excluded
        acc.add_sample("inst_executed", 500.0)
        acc.add_sample("sm__throughput.avg", 80.0)
        assert acc.total_stall_samples == 40.0

    def test_total_samples_inst_executed(self):
        """total_samples should prefer inst_executed when available."""
        acc = PcAccumulator(pc=0x400)
        acc.add_sample("inst_executed", 200.0)
        acc.add_sample("smsp__pcsamp_warps_issue_stalled_barrier", 50.0)
        acc.add_sample("smsp__pcsamp_warps_issue_stalled_wait", 30.0)
        assert acc.total_samples == 200.0

    def test_total_samples_fallback(self):
        """total_samples should fall back to stall sum when inst_executed is absent."""
        acc = PcAccumulator(pc=0x500)
        acc.add_sample("smsp__pcsamp_warps_issue_stalled_barrier", 50.0)
        acc.add_sample("smsp__pcsamp_warps_issue_stalled_long_scoreboard", 120.0)
        assert acc.total_samples == 170.0


# ━━━━━━━━━━━━━━━━━━━━━━━ LineAccumulator ━━━━━━━━━━━━━━━━━━━━━━━━


class TestLineAccumulator:
    """Tests for the per-source-line metric accumulator."""

    def test_merge_pc(self):
        """Merging two PCs into one line should aggregate metrics correctly."""
        pc1 = PcAccumulator(pc=0x100)
        pc1.add_sample("smsp__pcsamp_warps_issue_stalled_barrier", 10.0)
        pc1.add_sample("inst_executed", 100.0)
        pc1.sass_text = "SASS_A"

        pc2 = PcAccumulator(pc=0x104)
        pc2.add_sample("smsp__pcsamp_warps_issue_stalled_barrier", 20.0)
        pc2.add_sample("inst_executed", 50.0)
        pc2.sass_text = "SASS_B"

        la = LineAccumulator()
        la.merge_pc(pc1)
        la.merge_pc(pc2)

        assert len(la.pcs) == 2
        # Stall metric: SUM
        assert la.aggregated["smsp__pcsamp_warps_issue_stalled_barrier"] == 30.0
        # Exec metric: SUM
        assert la.aggregated["inst_executed"] == 150.0
        assert la.total_samples == 150.0
        assert la.total_stall_samples == 30.0
        assert la.sass_list == ["SASS_A", "SASS_B"]

    def test_sass_dedup(self):
        """Duplicate SASS text from different PCs should not be repeated."""
        pc1 = PcAccumulator(pc=0x100, sass_text="LDG.E R2, [R4]")
        pc1.add_sample("smsp__pcsamp_warps_issue_stalled_barrier", 5.0)

        pc2 = PcAccumulator(pc=0x104, sass_text="LDG.E R2, [R4]")  # same SASS
        pc2.add_sample("smsp__pcsamp_warps_issue_stalled_barrier", 3.0)

        la = LineAccumulator()
        la.merge_pc(pc1)
        la.merge_pc(pc2)

        assert len(la.sass_list) == 1
        assert la.sass_list[0] == "LDG.E R2, [R4]"

    def test_dominant_stall_reason(self):
        """dominant_stall_reason should return the highest stall metric."""
        la = LineAccumulator()
        pc = PcAccumulator(pc=0x100)
        pc.add_sample("smsp__pcsamp_warps_issue_stalled_barrier", 10.0)
        pc.add_sample("smsp__pcsamp_warps_issue_stalled_long_scoreboard", 90.0)
        pc.add_sample("smsp__pcsamp_warps_issue_stalled_wait", 30.0)
        la.merge_pc(pc)

        name, val = la.dominant_stall_reason
        assert name == "smsp__pcsamp_warps_issue_stalled_long_scoreboard"
        assert val == 90.0

    def test_no_stalls(self):
        """When no stall metrics exist, dominant_stall_reason returns ("none", 0.0)."""
        la = LineAccumulator()
        pc = PcAccumulator(pc=0x100)
        pc.add_sample("inst_executed", 500.0)
        la.merge_pc(pc)

        name, val = la.dominant_stall_reason
        assert name == "none"
        assert val == 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━ SourceCorrelator ━━━━━━━━━━━━━━━━━━━━━━━


class TestSourceCorrelator:
    """Tests for the main SourceCorrelator.correlate() pipeline."""

    def test_basic_correlation(self):
        """Three PCs, two lines: verify aggregation, sorting, and hotspot fields."""
        pc_map = {
            0x1000: ("test.cu", 42),
            0x1004: ("test.cu", 42),
            0x1008: ("test.cu", 58),
        }
        action = _make_mock_action(pc_map)

        instanced = {
            "smsp__pcsamp_warps_issue_stalled_long_scoreboard": [
                (0x1000, 150.0),
                (0x1004, 80.0),
                (0x1008, 20.0),
            ],
            "inst_executed": [
                (0x1000, 200.0),
                (0x1004, 100.0),
                (0x1008, 300.0),
            ],
        }

        cfg = CorrelatorConfig(
            local_ratio_threshold=0.30,
            global_ratio_threshold=0.10,
        )
        correlator = SourceCorrelator(cfg)
        hotspots = correlator.correlate(action, instanced)

        # --- Line 42 aggregation ---
        # stall = 150 + 80 = 230, inst_executed = 200 + 100 = 300
        # local_ratio = 230/300 = 0.766..., global_ratio = 300/600 = 0.5
        # Both exceed thresholds -> is_hot = True

        # --- Line 58 ---
        # stall = 20, inst_executed = 300
        # local_ratio = 20/300 = 0.066..., global_ratio = 300/600 = 0.5
        # local_ratio < 0.30 -> NOT hot

        # Only line 42 should be a hotspot
        assert len(hotspots) == 1
        h = hotspots[0]
        assert h.source_file == "test.cu"
        assert h.source_line == 42
        assert h.is_hot is True
        assert h.degraded is False

        # Verify aggregated metric values
        assert h.metric_values["inst_executed"] == 300.0
        assert h.metric_values[
            "smsp__pcsamp_warps_issue_stalled_long_scoreboard"
        ] == 230.0

        # Verify ratios (kernel_total = 300 + 300 = 600)
        assert abs(h.local_ratio - 230.0 / 300.0) < 1e-9
        assert abs(h.global_ratio - 300.0 / 600.0) < 1e-9

        # SASS should contain entries from both PCs on line 42
        assert "SASS@0x1000" in h.sass_instruction
        assert "SASS@0x1004" in h.sass_instruction

    def test_no_debug_info(self):
        """All PCs with no source_info should produce degraded hotspots."""
        # Empty map -> all PCs fall back to <no debug info>
        action = _make_mock_action({})

        instanced = {
            # Use stall metrics only so local_ratio = 1.0 (well above 0.30)
            "smsp__pcsamp_warps_issue_stalled_barrier": [
                (0xA00, 100.0),
                (0xB00, 200.0),
            ],
        }

        cfg = CorrelatorConfig(
            local_ratio_threshold=0.30,
            global_ratio_threshold=0.05,  # low so both PCs pass global
        )
        correlator = SourceCorrelator(cfg)
        hotspots = correlator.correlate(action, instanced)

        # Each PC gets its own pseudo-line because key is (None, pc)
        assert len(hotspots) == 2
        for h in hotspots:
            assert h.degraded is True
            assert h.source_file == "<no debug info>"

    def test_thresholds_exact(self):
        """Values exactly at thresholds should NOT be hot (strict >).

        Slightly above should qualify.
        """
        # Design: two source lines.
        # Line 10: local_ratio = exactly 0.30, global_ratio = exactly 0.10 -> NOT hot
        # Line 20: local_ratio = 0.31, global_ratio = 0.11 -> hot
        #
        # We use stall-only metrics (no inst_executed) so total = stall sum.
        #
        # For line 10: want local_ratio = dominant/total = 0.30
        #   dominant = 30, total = 100, local_ratio = 0.30
        # For line 20: want local_ratio ~0.31
        #   dominant = 31, total = 100, local_ratio = 0.31
        #
        # kernel_total: need global_ratio = total_line / kernel_total
        # For line 10: global_ratio = 100/1000 = 0.10
        # For line 20: global_ratio = 100/1000 ... that's also 0.10, need >0.10
        #   So use a different split. Let's use kernel_total = 909.09..
        #   Actually, let's set it up more carefully:
        #
        # Line 10: stall_A = 30, stall_B = 70 (total = 100)
        # Line 20: stall_A = 31, stall_B = 69 (total = 100)
        # Line 99 (filler): stall_A = 800 (total = 800) -- won't be hot if local < 0.30
        #   Actually filler would be hot too. Let's use non-stall to pad kernel_total.
        #
        # Simpler approach: use inst_executed to control totals independently.
        # Line 10: stall_A = 30, inst_executed = 100
        #   local_ratio = 30/100 = 0.30 (exactly at threshold)
        # Line 20: stall_A = 31, inst_executed = 100
        #   local_ratio = 31/100 = 0.31 (above threshold)
        #
        # kernel_total (inst_executed) = 100 + 100 + 800 = 1000
        # global_ratio line 10 = 100/1000 = 0.10 (exactly at threshold)
        # global_ratio line 20 = 100/1000 = 0.10 ... still exactly at 0.10.
        #
        # Adjust: line 20 inst_executed = 110
        # kernel_total = 100 + 110 + 800 = 1010
        # global_ratio line 10 = 100/1010 = 0.099.. < 0.10 -> not hot anyway
        # global_ratio line 20 = 110/1010 = 0.1089.. > 0.10 -> hot (if local also passes)
        # local_ratio line 20 = 31/110 = 0.2818.. < 0.30 -> not hot.
        #
        # This is getting complex. Let's simplify by targeting each threshold separately.

        # We'll use two PCs on separate lines.
        # PC 0x100 -> line 10: exactly at both thresholds -> NOT hot
        # PC 0x200 -> line 20: slightly above both thresholds -> hot

        pc_map = {
            0x100: ("a.cu", 10),
            0x200: ("a.cu", 20),
            0x300: ("a.cu", 99),  # filler to control kernel_total
        }
        action = _make_mock_action(pc_map)

        # Line 10: stall = 300, inst_executed = 1000
        #   local_ratio = 300/1000 = 0.30 (exactly at threshold, NOT hot)
        # Line 20: stall = 310, inst_executed = 1000
        #   local_ratio = 310/1000 = 0.31 (above threshold)
        # Line 99 (filler): inst_executed = 8000 (no stall -> local=0 -> not hot)
        #
        # kernel_total = 1000 + 1000 + 8000 = 10000
        # global_ratio line 10 = 1000/10000 = 0.10 (exactly, NOT hot)
        # global_ratio line 20 = 1000/10000 = 0.10 (exactly, NOT hot!)
        #
        # Need line 20 global > 0.10 => inst_executed = 1001
        # kernel_total = 1000 + 1001 + 8000 = 10001
        # global_ratio line 20 = 1001/10001 = 0.10009.. > 0.10  -> hot

        instanced = {
            "smsp__pcsamp_warps_issue_stalled_long_scoreboard": [
                (0x100, 300.0),
                (0x200, 310.0),
                # no stall on filler
            ],
            "inst_executed": [
                (0x100, 1000.0),
                (0x200, 1001.0),
                (0x300, 8000.0),
            ],
        }

        cfg = CorrelatorConfig(
            local_ratio_threshold=0.30,
            global_ratio_threshold=0.10,
        )
        correlator = SourceCorrelator(cfg)
        hotspots = correlator.correlate(action, instanced)

        # Line 10: local=0.30 (not > 0.30) -> NOT hot
        # Line 20: local=310/1001=0.3097, global=1001/10001=0.1001 -> hot
        # Line 99: no stall -> local=0 -> NOT hot
        assert len(hotspots) == 1
        h = hotspots[0]
        assert h.source_line == 20
        assert h.is_hot is True
        assert h.local_ratio > 0.30
        assert h.global_ratio > 0.10

    def test_empty_metrics(self):
        """Empty instanced_metrics should return empty list without crashing."""
        action = _make_mock_action({})
        correlator = SourceCorrelator()
        hotspots = correlator.correlate(action, {})
        assert hotspots == []

    def test_stall_ratios(self):
        """Stall_reasons dict values should sum to approximately 1.0."""
        pc_map = {0x100: ("k.cu", 10)}
        action = _make_mock_action(pc_map)

        instanced = {
            "smsp__pcsamp_warps_issue_stalled_barrier": [(0x100, 40.0)],
            "smsp__pcsamp_warps_issue_stalled_long_scoreboard": [(0x100, 50.0)],
            "smsp__pcsamp_warps_issue_stalled_wait": [(0x100, 10.0)],
        }
        # local_ratio = 50/100 = 0.5 > 0.30, global_ratio = 1.0 > 0.10
        cfg = CorrelatorConfig(
            local_ratio_threshold=0.30,
            global_ratio_threshold=0.05,
        )
        correlator = SourceCorrelator(cfg)
        hotspots = correlator.correlate(action, instanced)

        assert len(hotspots) == 1
        h = hotspots[0]
        assert len(h.stall_reasons) == 3
        ratio_sum = sum(h.stall_reasons.values())
        assert abs(ratio_sum - 1.0) < 1e-9
        # Individual ratios
        assert abs(h.stall_reasons[
            "smsp__pcsamp_warps_issue_stalled_barrier"
        ] - 0.40) < 1e-9
        assert abs(h.stall_reasons[
            "smsp__pcsamp_warps_issue_stalled_long_scoreboard"
        ] - 0.50) < 1e-9
        assert abs(h.stall_reasons[
            "smsp__pcsamp_warps_issue_stalled_wait"
        ] - 0.10) < 1e-9

    def test_max_hotspots(self):
        """Config max_hotspots=3 should limit output even if more qualify."""
        # Create 12 PCs each on a separate line, all qualifying as hot.
        pc_map = {}
        stall_entries = []
        for i in range(12):
            pc = 0x1000 + i * 4
            pc_map[pc] = ("big.cu", 100 + i)
            stall_entries.append((pc, 100.0))

        action = _make_mock_action(pc_map)

        instanced = {
            # Each line: stall = 100, total (stall-only) = 100
            # local_ratio = 100/100 = 1.0 > 0.30
            # kernel_total = 12 * 100 = 1200
            # global_ratio = 100/1200 = 0.083.. < 0.10? Need to lower threshold.
            "smsp__pcsamp_warps_issue_stalled_barrier": stall_entries,
        }

        cfg = CorrelatorConfig(
            local_ratio_threshold=0.30,
            global_ratio_threshold=0.05,  # low enough for 1/12 = 0.083 to pass
            max_hotspots=3,
        )
        correlator = SourceCorrelator(cfg)
        hotspots = correlator.correlate(action, instanced)

        assert len(hotspots) == 3

    def test_check_degradation_none(self):
        """All hotspots with source info -> check_degradation returns None."""
        correlator = SourceCorrelator()
        hotspots = [
            SourceHotspot(
                source_file="a.cu", source_line=10, degraded=False,
            ),
            SourceHotspot(
                source_file="b.cu", source_line=20, degraded=False,
            ),
        ]
        assert correlator.check_degradation(hotspots) is None

    def test_check_degradation_all(self):
        """All degraded hotspots -> returns full warning with -lineinfo."""
        correlator = SourceCorrelator()
        hotspots = [
            SourceHotspot(
                source_file="<no debug info>", degraded=True,
            ),
            SourceHotspot(
                source_file="<no debug info>", degraded=True,
            ),
        ]
        warning = correlator.check_degradation(hotspots)
        assert warning is not None
        assert "-lineinfo" in warning
        assert "SASS PC-level" in warning

    def test_check_degradation_partial(self):
        """Some degraded hotspots -> returns partial warning with count."""
        correlator = SourceCorrelator()
        hotspots = [
            SourceHotspot(source_file="a.cu", degraded=False),
            SourceHotspot(source_file="<no debug info>", degraded=True),
            SourceHotspot(source_file="b.cu", degraded=False),
            SourceHotspot(source_file="<no debug info>", degraded=True),
            SourceHotspot(source_file="c.cu", degraded=False),
        ]
        warning = correlator.check_degradation(hotspots)
        assert warning is not None
        assert "2/5" in warning
        assert "-lineinfo" in warning

    def test_check_degradation_empty(self):
        """Empty hotspot list -> check_degradation returns None."""
        correlator = SourceCorrelator()
        assert correlator.check_degradation([]) is None

    def test_sorted_by_global_ratio(self):
        """Hotspots should be returned sorted by global_ratio descending."""
        pc_map = {
            0x100: ("s.cu", 10),
            0x200: ("s.cu", 20),
            0x300: ("s.cu", 30),
        }
        action = _make_mock_action(pc_map)

        # Each line has local_ratio = 1.0 (stall-only, single stall metric).
        # Global ratios differ by inst_executed values.
        instanced = {
            "smsp__pcsamp_warps_issue_stalled_barrier": [
                (0x100, 100.0),
                (0x200, 300.0),
                (0x300, 200.0),
            ],
        }

        cfg = CorrelatorConfig(
            local_ratio_threshold=0.30,
            global_ratio_threshold=0.05,
        )
        correlator = SourceCorrelator(cfg)
        hotspots = correlator.correlate(action, instanced)

        assert len(hotspots) == 3
        # Sorted by global_ratio descending: 300/600 > 200/600 > 100/600
        assert hotspots[0].source_line == 20
        assert hotspots[1].source_line == 30
        assert hotspots[2].source_line == 10
