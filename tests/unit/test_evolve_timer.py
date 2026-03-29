"""Unit tests for DebugTimer."""
import time

from tachyon.evolve.timer import DebugTimer


class TestDebugTimer:

    def test_disabled_noop(self) -> None:
        timer = DebugTimer(enabled=False)
        with timer.phase("test"):
            pass
        timer.record("manual", 1.0)
        assert timer.format_report() == ""

    def test_phase_records_time(self) -> None:
        timer = DebugTimer(enabled=True)
        with timer.phase("sleep"):
            time.sleep(0.05)
        assert "sleep" in timer._totals
        assert timer._totals["sleep"] >= 0.04
        assert timer._counts["sleep"] == 1

    def test_record_accumulates(self) -> None:
        timer = DebugTimer(enabled=True)
        timer.record("llm_api", 1.5)
        timer.record("llm_api", 2.5)
        assert timer._totals["llm_api"] == 4.0
        assert timer._counts["llm_api"] == 2

    def test_format_report_sorted(self) -> None:
        timer = DebugTimer(enabled=True)
        timer.record("small", 1.0)
        timer.record("big", 10.0)
        timer.record("medium", 5.0)
        report = timer.format_report(wall_time=20.0)
        lines = report.strip().splitlines()
        # First line should be "big" (highest)
        assert "big" in lines[0]
        assert "small" in lines[2]
        # Should have TOTAL line
        assert "TOTAL" in report

    def test_format_report_with_untracked(self) -> None:
        timer = DebugTimer(enabled=True)
        timer.record("phase_a", 5.0)
        report = timer.format_report(wall_time=10.0)
        assert "(untracked)" in report

    def test_format_report_empty(self) -> None:
        timer = DebugTimer(enabled=True)
        assert timer.format_report() == ""

    def test_multiple_phases(self) -> None:
        timer = DebugTimer(enabled=True)
        with timer.phase("a"):
            time.sleep(0.01)
        with timer.phase("a"):
            time.sleep(0.01)
        assert timer._counts["a"] == 2
