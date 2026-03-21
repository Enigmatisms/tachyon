"""Unit tests for tachyon.chat.export module."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from tachyon.chat.export import (
    ChatHistory,
    ToolCallRecord,
    TurnRecord,
    UsageRecord,
    render_all_turns_markdown,
    render_turn_markdown,
    write_export,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_tool_calls():
    return [
        ToolCallRecord(
            name="get_kernel_summary",
            arguments={"kernel_id": 0},
            success=True,
            summary="Found 5 metrics for kernel",
            elapsed=0.45,
        ),
        ToolCallRecord(
            name="get_stall_analysis",
            arguments={"kernel_id": 0, "detail": "full"},
            success=False,
            summary="Error: no stall metrics",
            elapsed=1.23,
        ),
    ]


@pytest.fixture
def sample_usage():
    return UsageRecord(
        turns=3, tool_calls=2, total_tokens=4567,
        total_elapsed=12.3, budget=120000,
        budget_peak=45000, budget_remaining=75000,
        budget_pct=37.5, compaction_count=0,
    )


@pytest.fixture
def sample_turn(sample_tool_calls, sample_usage):
    return TurnRecord(
        user_input="What is the SM throughput?",
        ai_response="This kernel has SM throughput of 85%.",
        tool_calls=sample_tool_calls,
        usage=sample_usage,
        timestamp="2026-03-21T12:34:56Z",
    )


# ---------------------------------------------------------------------------
# TurnRecord
# ---------------------------------------------------------------------------

class TestTurnRecord:

    def test_has_tool_calls_true(self, sample_turn):
        assert sample_turn.has_tool_calls

    def test_has_tool_calls_false(self):
        assert not TurnRecord(user_input="hello").has_tool_calls

    def test_empty_turn(self):
        t = TurnRecord()
        assert t.user_input == ""
        assert t.ai_response == ""
        assert t.tool_calls == []
        assert t.usage is None


# ---------------------------------------------------------------------------
# ChatHistory
# ---------------------------------------------------------------------------

class TestChatHistory:

    def test_record_and_retrieve(self, sample_turn):
        h = ChatHistory()
        h.record_turn(sample_turn)
        assert len(h.turns) == 1
        assert h.last_turn() is sample_turn

    def test_last_turn_empty(self):
        h = ChatHistory()
        assert h.last_turn() is None

    def test_model_info(self):
        h = ChatHistory(model_info="openai/gpt-4o")
        assert h.model_info == "openai/gpt-4o"

    def test_clear(self, sample_turn):
        h = ChatHistory()
        h.record_turn(sample_turn)
        h.clear()
        assert len(h.turns) == 0

    def test_multiple_turns(self, sample_turn):
        h = ChatHistory()
        h.record_turn(sample_turn)
        h.record_turn(TurnRecord(ai_response="second"))
        assert len(h.turns) == 2
        assert h.last_turn().ai_response == "second"


# ---------------------------------------------------------------------------
# render_turn_markdown
# ---------------------------------------------------------------------------

class TestRenderTurnMarkdown:

    def test_full_render(self, sample_turn):
        md = render_turn_markdown(sample_turn, model_info="openai/gpt-4o")
        assert "# Tachyon Chat Export" in md
        assert "openai/gpt-4o" in md
        assert "What is the SM throughput?" in md
        assert "85%" in md
        assert "get_kernel_summary" in md
        assert "get_stall_analysis" in md
        assert "OK" in md
        assert "FAIL" in md
        assert "Agent turns" in md
        assert "4,567" in md

    def test_no_model_info(self, sample_turn):
        md = render_turn_markdown(sample_turn)
        assert "Model:" not in md

    def test_empty_response(self):
        turn = TurnRecord(user_input="hello")
        md = render_turn_markdown(turn)
        assert "*(No response)*" in md

    def test_no_tool_calls(self, sample_usage):
        turn = TurnRecord(user_input="hello", ai_response="hi", usage=sample_usage)
        md = render_turn_markdown(turn)
        assert "Tool Calls" not in md

    def test_no_usage(self, sample_tool_calls):
        turn = TurnRecord(
            user_input="hello", ai_response="hi", tool_calls=sample_tool_calls,
        )
        md = render_turn_markdown(turn)
        assert "Usage" not in md

    def test_no_user_input(self, sample_usage):
        turn = TurnRecord(ai_response="hi", usage=sample_usage)
        md = render_turn_markdown(turn)
        assert "## You" not in md
        assert "hi" in md

    def test_pure_markdown_no_rich_markup(self, sample_turn):
        md = render_turn_markdown(sample_turn)
        assert "[bold]" not in md
        assert "[/bold]" not in md
        assert "[dim]" not in md

    def test_long_arguments_truncated(self):
        tc = ToolCallRecord(
            name="some_tool",
            arguments={"very_long_key": "x" * 100},
        )
        turn = TurnRecord(user_input="q", ai_response="a", tool_calls=[tc])
        md = render_turn_markdown(turn)
        assert "..." in md

    def test_many_arguments_shows_more_count(self):
        tc = ToolCallRecord(
            name="tool",
            arguments={f"arg{i}": i for i in range(5)},
        )
        turn = TurnRecord(user_input="q", ai_response="a", tool_calls=[tc])
        md = render_turn_markdown(turn)
        assert "+2 more" in md

    def test_budget_shown(self, sample_turn):
        md = render_turn_markdown(sample_turn)
        assert "Context window" in md
        assert "45,000/120,000" in md

    def test_no_budget(self, sample_usage):
        usage = UsageRecord(turns=1, tool_calls=0, total_tokens=100)
        turn = TurnRecord(ai_response="hi", usage=usage)
        md = render_turn_markdown(turn)
        assert "Context window" not in md

    def test_timestamp_in_output(self, sample_turn):
        md = render_turn_markdown(sample_turn)
        assert "2026-03-21T12:34:56Z" in md

    def test_empty_arguments_shows_dash(self):
        tc = ToolCallRecord(name="tool")
        turn = TurnRecord(user_input="q", ai_response="a", tool_calls=[tc])
        md = render_turn_markdown(turn)
        assert "| 1 | `tool` | - | OK | - | 0.00s |" in md

    def test_empty_summary_shows_dash(self):
        tc = ToolCallRecord(name="tool", summary="")
        turn = TurnRecord(user_input="q", ai_response="a", tool_calls=[tc])
        md = render_turn_markdown(turn)
        assert "| - |" in md  # summary column shows dash


# ---------------------------------------------------------------------------
# render_all_turns_markdown
# ---------------------------------------------------------------------------

class TestRenderAllTurnsMarkdown:

    def test_single_turn(self, sample_turn):
        md = render_all_turns_markdown([sample_turn])
        assert "# Tachyon Chat Export (All Turns)" in md
        assert "## Turn 1" in md
        assert "85%" in md

    def test_multiple_turns(self):
        turns = [
            TurnRecord(ai_response="Response 1"),
            TurnRecord(ai_response="Response 2"),
            TurnRecord(ai_response="Response 3"),
        ]
        md = render_all_turns_markdown(turns)
        assert "## Turn 1" in md
        assert "## Turn 2" in md
        assert "## Turn 3" in md

    def test_no_user_input_in_export_all(self, sample_turn):
        md = render_all_turns_markdown([sample_turn])
        assert "What is the SM throughput?" not in md

    def test_no_tool_details_in_export_all(self, sample_turn):
        md = render_all_turns_markdown([sample_turn])
        assert "Tool Calls" not in md
        assert "get_kernel_summary" not in md

    def test_no_usage_in_export_all(self, sample_turn):
        md = render_all_turns_markdown([sample_turn])
        assert "### Usage" not in md

    def test_empty_turns(self):
        md = render_all_turns_markdown([])
        assert "# Tachyon Chat Export (All Turns)" in md
        assert "## Turn" not in md

    def test_empty_response_shows_placeholder(self):
        md = render_all_turns_markdown([TurnRecord()])
        assert "*(No response)*" in md

    def test_model_info(self):
        turns = [TurnRecord(ai_response="hi")]
        md = render_all_turns_markdown(turns, model_info="test/model")
        assert "test/model" in md


# ---------------------------------------------------------------------------
# write_export
# ---------------------------------------------------------------------------

class TestWriteExport:

    def test_write_to_file(self, sample_turn):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "export.md")
            content = render_turn_markdown(sample_turn)
            result = write_export(content, path)
            assert result.exists()
            text = result.read_text()
            assert "Tachyon Chat Export" in text

    def test_creates_parent_dirs(self, sample_turn):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "a", "b", "c", "export.md")
            write_export("hello", path)
            assert Path(path).exists()

    def test_returns_resolved_path(self, sample_turn):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "export.md")
            result = write_export("hello", path)
            assert result == Path(path).resolve()

    def test_write_failure(self):
        # /proc/... is not writable on Linux
        with pytest.raises(OSError):
            write_export("content", "/proc/1/cannot_write_here.md")
