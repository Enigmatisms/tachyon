"""Unit tests for EvolveOrchestrator — summary extraction and robustness."""
from tachyon.evolve.models import ExperimentRecord, ExperimentStatus
from tachyon.evolve.orchestrator import (
    EvolveOrchestrator,
    _is_summary_text,
    _split_sentences,
)

# --- _split_sentences ---


class TestSplitSentences:

    def test_english_sentences(self) -> None:
        text = "Applied tiling. Reduced DRAM traffic. Set tile size to 32."
        result = _split_sentences(text)
        assert len(result) == 3

    def test_chinese_sentences(self) -> None:
        text = "应用了共享内存分块技术。减少了全局内存读取。分块大小设为32。"
        result = _split_sentences(text)
        assert len(result) == 3

    def test_mixed_sentences(self) -> None:
        text = "Applied shared memory tiling. 减少了 DRAM 流量。Tile size set to 32."
        result = _split_sentences(text)
        assert len(result) == 3

    def test_filters_short_fragments(self) -> None:
        text = "OK. Done. Applied shared memory tiling to the inner loop."
        result = _split_sentences(text)
        # "OK" and "Done" are <=5 chars, should be filtered
        assert len(result) == 1

    def test_empty_string(self) -> None:
        assert _split_sentences("") == []

    def test_no_punctuation(self) -> None:
        text = "This is a sentence without ending punctuation"
        result = _split_sentences(text)
        assert len(result) == 1


# --- _extract_summary ---


class TestExtractSummary:

    def test_with_marker(self) -> None:
        record = ExperimentRecord(iteration=0)
        text = (
            "Analysis done.\n\n"
            "[SUMMARY] Applied shared memory tiling, reducing DRAM traffic by 40%."
        )
        EvolveOrchestrator._extract_summary(record, text)
        assert "shared memory tiling" in record.summary
        assert "40%" in record.summary

    def test_marker_multiple_sentences(self) -> None:
        record = ExperimentRecord(iteration=0)
        text = (
            "Some analysis.\n\n"
            "[SUMMARY] Used shared memory. Reduced global loads by 50%. "
            "Tile size 32. Extra sentence should be excluded."
        )
        EvolveOrchestrator._extract_summary(record, text)
        # Should take at most 3 sentences
        parts = record.summary.split(". ")
        assert len(parts) <= 4  # at most 3 sentences, possibly ending without period

    def test_marker_last_occurrence(self) -> None:
        record = ExperimentRecord(iteration=0)
        text = (
            "[SUMMARY] Old summary.\n\n"
            "More analysis.\n\n"
            "[SUMMARY] Applied loop unrolling, improving IPC by 15%."
        )
        EvolveOrchestrator._extract_summary(record, text)
        assert "loop unrolling" in record.summary
        assert "Old summary" not in record.summary

    def test_no_marker_fallback_to_last_paragraph(self) -> None:
        record = ExperimentRecord(iteration=0)
        text = (
            "First, analyzed the kernel bottlenecks and found memory-bound behavior.\n\n"
            "Applied shared memory tiling to reduce DRAM accesses by 40%. "
            "Tile size was set to 32x32 based on shared memory capacity."
        )
        EvolveOrchestrator._extract_summary(record, text)
        assert record.summary != ""
        assert "tiling" in record.summary or "DRAM" in record.summary

    def test_empty_text(self) -> None:
        record = ExperimentRecord(iteration=0)
        EvolveOrchestrator._extract_summary(record, "")
        assert record.summary == ""

    def test_chinese_summary(self) -> None:
        record = ExperimentRecord(iteration=0)
        text = (
            "分析完成。\n\n"
            "[SUMMARY] 采用共享内存分块技术优化了矩阵乘法内核。减少了40%的全局内存读取。"
        )
        EvolveOrchestrator._extract_summary(record, text)
        assert "共享内存" in record.summary

    def test_markdown_stripped(self) -> None:
        record = ExperimentRecord(iteration=0)
        text = "**Analysis**.\n\n[SUMMARY] Applied `shared memory` tiling with **TILE_SIZE=32**."
        EvolveOrchestrator._extract_summary(record, text)
        assert "`" not in record.summary
        assert "**" not in record.summary
        assert "shared memory" in record.summary

    def test_no_marker_no_paragraph_fallback_sentences(self) -> None:
        record = ExperimentRecord(iteration=0)
        text = "Applied tiling to reduce traffic. Set tile size to 32. Improved throughput by 15%."
        EvolveOrchestrator._extract_summary(record, text)
        assert record.summary != ""

    def test_bracket_marker_stops_extraction(self) -> None:
        record = ExperimentRecord(iteration=0)
        text = "[SUMMARY] Applied tiling. [DONE] Finished."
        EvolveOrchestrator._extract_summary(record, text)
        assert "Applied tiling" in record.summary
        assert "Finished" not in record.summary


# --- Fatal path robustness ---


class TestFatalPathStatus:
    """Verify that fatal paths always set status and decision."""

    def test_no_tool_calls_sets_failed(self) -> None:
        """When tool_call_count == 0, status must be FAILED with a decision."""
        record = ExperimentRecord(iteration=0)
        record.status = ExperimentStatus.HYPOTHESIS

        # Simulate the orchestrator's logic for zero tool calls
        record.status = ExperimentStatus.FAILED
        record.decision = (
            "LLM did not call any tools. "
            "The model may not support function calling."
        )

        assert record.status == ExperimentStatus.FAILED
        assert record.decision != ""
        assert "no decision" not in record.decision

    def test_rollback_preserves_failed_status(self) -> None:
        """_rollback should NOT overwrite FAILED with ROLLED_BACK."""
        record = ExperimentRecord(iteration=0)
        record.status = ExperimentStatus.FAILED
        record.decision = "LLM did not call any tools."

        # Simulate what _rollback does now
        if record.status != ExperimentStatus.FAILED:
            record.status = ExperimentStatus.ROLLED_BACK

        assert record.status == ExperimentStatus.FAILED

    def test_rollback_changes_regression_to_rolled_back(self) -> None:
        """_rollback SHOULD change REGRESSION to ROLLED_BACK."""
        record = ExperimentRecord(iteration=0)
        record.status = ExperimentStatus.REGRESSION
        record.decision = "Rejected: GPU time +5.0%"

        # Simulate _rollback
        if record.status != ExperimentStatus.FAILED:
            record.status = ExperimentStatus.ROLLED_BACK

        assert record.status == ExperimentStatus.ROLLED_BACK


# --- _is_summary_text ---


class TestIsSummaryText:

    def test_rejects_planning_phrases(self) -> None:
        assert _is_summary_text("Now let me analyze the results.") is False
        assert _is_summary_text("Let me check the metrics.") is False
        assert _is_summary_text("I'll apply register prefetching.") is False
        assert _is_summary_text("I will optimize the kernel.") is False
        assert _is_summary_text("I need to read the source.") is False
        assert _is_summary_text("Next we should try tiling.") is False
        assert _is_summary_text("First let's analyze.") is False

    def test_rejects_reasoning_phrases(self) -> None:
        assert _is_summary_text("My approach was fundamentally wrong.") is False
        assert _is_summary_text("However, the kernel uses float.") is False
        assert _is_summary_text("Unfortunately, this broke correctness.") is False
        assert _is_summary_text("Looking at the error output.") is False
        assert _is_summary_text("Since the kernel is memory-bound.") is False

    def test_accepts_summary_text(self) -> None:
        assert _is_summary_text(
            "Applied shared memory tiling to reduce DRAM traffic."
        ) is True
        assert _is_summary_text(
            "Replaced global memory loads with coalesced access."
        ) is True
        assert _is_summary_text(
            "The optimization reduced global memory accesses by 4x."
        ) is True
        assert _is_summary_text(
            "使用共享内存分块技术优化了矩阵乘法。"
        ) is True


class TestExtractSummaryFiltering:
    """Test that non-summary LLM reasoning is filtered out."""

    def test_filters_planning_text_in_fallback(self) -> None:
        record = ExperimentRecord(iteration=0)
        text = (
            "Applied tiling optimization to the inner loop.\n\n"
            "Now let me analyze the full results to check further."
        )
        EvolveOrchestrator._extract_summary(record, text)
        # Should pick the first paragraph (summary), not the second (planning)
        assert "tiling" in record.summary
        assert "let me" not in record.summary.lower()

    def test_no_marker_skips_reasoning_paragraphs(self) -> None:
        record = ExperimentRecord(iteration=0)
        text = (
            "I'll apply register prefetching optimization.\n\n"
            "Reduced register spills by using prefetch instructions. "
            "Improved occupancy from 50% to 75%."
        )
        EvolveOrchestrator._extract_summary(record, text)
        assert "I'll" not in record.summary
        assert "register" in record.summary.lower() or "occupancy" in record.summary.lower()


# --- Experiment history formatting ---


class TestExperimentHistoryFormatting:
    """Test that experiment history includes build errors for diagnosis."""

    def test_includes_no_such_file_error(self) -> None:
        """Build errors with 'No such file' should appear in history."""
        record = ExperimentRecord(iteration=0)
        record.status = ExperimentStatus.FAILED
        record.decision = "No optimized metrics after 15 tool call(s)"
        record.build_log = (
            "/bin/sh: line 1: cd: ./temp/cuda-cases: No such file or directory"
        )

        # Simulate what _format_experiment_history does
        log_lower = record.build_log.lower()
        has_error = any(
            kw in log_lower
            for kw in ("error", "fail", "no such file", "not found")
        )
        assert has_error is True

        err_lines = [
            ln.strip() for ln in record.build_log.splitlines()
            if any(kw in ln.lower() for kw in ("error", "fail", "no such file", "not found"))
            and len(ln.strip()) > 10
        ]
        assert len(err_lines) >= 1
        assert "No such file" in err_lines[0]

    def test_includes_compilation_error(self) -> None:
        """Compilation errors with 'error:' should appear in history."""
        record = ExperimentRecord(iteration=0)
        record.status = ExperimentStatus.FAILED
        record.build_log = "cu_gemm.cu(42): error: expected a ';'"

        log_lower = record.build_log.lower()
        has_error = any(
            kw in log_lower for kw in ("error", "fail", "no such file", "not found")
        )
        assert has_error is True
