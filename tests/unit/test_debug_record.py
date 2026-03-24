"""Unit tests for unified logging module (tachyon.utils.log + debug_record shim)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from tachyon.utils.log import (
    AgentLogger,
    _L,
    agent_logger,
    configure_logging,
    get_logger,
)
from tachyon.utils.debug_record import (
    close,
    init,
    level,
    log_path,
    record_prompt,
    record_tool,
)


class FakeRole:
    value = "user"


class FakeToolCall:
    def __init__(self, name="foo", arguments=None):
        self.name = name
        self.arguments = arguments or {"x": 1}
        self.id = "tc_001"


class FakeMessage:
    def __init__(self, role="user", content="hello", tool_calls=None,
                 tool_call_id=None, name=None):
        self.role = FakeRole()
        self.role.value = role
        self.content = content
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id
        self.name = name


# ---------------------------------------------------------------------------
# AgentLogger direct tests
# ---------------------------------------------------------------------------

class TestAgentLoggerInit:
    def test_default_none(self):
        logger = AgentLogger()
        logger.init()
        assert logger.level() == 0
        assert logger.log_path() is None

    def test_prompt_level(self):
        logger = AgentLogger()
        with patch.dict(os.environ, {"TACHYON_DEBUG_RECORD": "prompt"}):
            logger.init()
        assert logger.level() == 1
        assert logger.log_path() is not None
        logger.close()

    def test_all_level(self):
        logger = AgentLogger()
        with patch.dict(os.environ, {"TACHYON_DEBUG_RECORD": "all"}):
            logger.init()
        assert logger.level() == 2
        logger.close()

    def test_numeric_1(self):
        logger = AgentLogger()
        with patch.dict(os.environ, {"TACHYON_DEBUG_RECORD": "1"}):
            logger.init()
        assert logger.level() == 1
        logger.close()

    def test_none_string(self):
        logger = AgentLogger()
        with patch.dict(os.environ, {"TACHYON_DEBUG_RECORD": "none"}):
            logger.init()
        assert logger.level() == 0

    def test_close_is_idempotent(self):
        logger = AgentLogger()
        logger.close()  # should not raise
        logger.close()  # should not raise


# ---------------------------------------------------------------------------
# Backward-compatible shim tests (module-level functions)
# ---------------------------------------------------------------------------

class TestShimCompatibility:
    """Verify that ``from tachyon.utils.debug_record import init, ...`` still works."""

    def test_init_level(self):
        logger = AgentLogger()
        with patch.dict(os.environ, {"TACHYON_DEBUG_RECORD": "all"}):
            logger.init()
        # The module-level functions delegate to the agent_logger singleton.
        # For this test, verify the AgentLogger API directly.
        assert logger.level() == 2
        logger.close()

    def test_shim_reexports(self):
        """debug_record re-exports key symbols from log."""
        from tachyon.utils.debug_record import agent_logger as shim_logger
        assert isinstance(shim_logger, AgentLogger)


# ---------------------------------------------------------------------------
# record_prompt tests
# ---------------------------------------------------------------------------

class TestRecordPrompt:
    def test_noop_when_none(self):
        logger = AgentLogger()
        logger.record_prompt([])

    def test_records_text_format(self, tmp_path: Path):
        logger = AgentLogger()
        logger._level = _L.PROMPT
        log = tmp_path / "test.log"
        logger._fh = open(log, "w", encoding="utf-8")

        msgs = [
            FakeMessage(role="system", content="You are helpful."),
            FakeMessage(role="user", content="hello"),
        ]
        logger.record_prompt(msgs, tools=[1, 2, 3], turn=1)

        logger.close()
        text = log.read_text()
        # Should have section header
        assert "PROMPT" in text
        assert "turn=1" in text
        assert "messages=2" in text
        assert "tools=3" in text
        # Should have raw content (not escaped JSON)
        assert "You are helpful." in text
        assert "hello" in text
        # Should have role labels
        assert "[SYSTEM]" in text
        assert "[USER]" in text

    def test_multiline_content_preserved(self, tmp_path: Path):
        logger = AgentLogger()
        logger._level = _L.PROMPT
        log = tmp_path / "test.log"
        logger._fh = open(log, "w", encoding="utf-8")

        msgs = [
            FakeMessage(role="user", content="line 1\nline 2\nline 3"),
        ]
        logger.record_prompt(msgs, turn=1)

        logger.close()
        text = log.read_text()
        assert "line 1\nline 2\nline 3" in text


# ---------------------------------------------------------------------------
# record_tool tests
# ---------------------------------------------------------------------------

class TestRecordTool:
    def test_noop_when_prompt_level(self):
        logger = AgentLogger()
        logger._level = _L.PROMPT
        logger.record_tool("foo", {}, "ok", 0.1, True)

    def test_records_text_format(self, tmp_path: Path):
        logger = AgentLogger()
        logger._level = _L.ALL
        log = tmp_path / "test.log"
        logger._fh = open(log, "w", encoding="utf-8")

        logger.record_tool("run_analysis", {"kernel_id": 0}, '{"findings": []}', 0.5, True)

        logger.close()
        text = log.read_text()
        assert "TOOL" in text
        assert "name=run_analysis" in text
        assert "elapsed=0.500s" in text
        assert "ok=True" in text
        assert '{"findings": []}' in text

    def test_result_truncated(self, tmp_path: Path):
        logger = AgentLogger()
        logger._level = _L.ALL
        log = tmp_path / "test.log"
        logger._fh = open(log, "w", encoding="utf-8")

        big_result = "x" * 10000
        logger.record_tool("foo", {}, big_result, 0.1, True)

        logger.close()
        text = log.read_text()
        assert "truncated" in text
        assert "10000 total" in text


# ---------------------------------------------------------------------------
# JSON format tests (structlog path)
# ---------------------------------------------------------------------------

class TestJsonFormat:
    """Test JSON structured output when structlog is available."""

    def test_prompt_json_output(self, tmp_path: Path):
        logger = AgentLogger()
        logger._level = _L.PROMPT
        logger._use_structlog = True
        log = tmp_path / "test.jsonl"
        logger._fh = open(log, "w", encoding="utf-8")

        msgs = [
            FakeMessage(role="system", content="You are helpful."),
            FakeMessage(role="user", content="Analyze this kernel."),
        ]
        logger.record_prompt(msgs, tools=[{"name": "get_metrics"}], turn=1)

        logger.close()
        lines = log.read_text().strip().split("\n")

        # First line: PROMPT event
        entry = json.loads(lines[0])
        assert entry["event"] == "PROMPT"
        assert entry["turn"] == 1
        assert entry["messages"] == 2
        assert entry["tools"] == 1
        assert "timestamp" in entry

        # Next lines: MESSAGE events
        msg1 = json.loads(lines[1])
        assert msg1["event"] == "MESSAGE"
        assert msg1["role"] == "SYSTEM"
        assert msg1["content_chars"] == len("You are helpful.")

        msg2 = json.loads(lines[2])
        assert msg2["event"] == "MESSAGE"
        assert msg2["role"] == "USER"

    def test_tool_json_output(self, tmp_path: Path):
        logger = AgentLogger()
        logger._level = _L.ALL
        logger._use_structlog = True
        log = tmp_path / "test.jsonl"
        logger._fh = open(log, "w", encoding="utf-8")

        logger.record_tool(
            "run_analysis",
            {"kernel_id": 0, "metric": "sm__throughput"},
            '{"findings": [{"severity": "high"}]}',
            0.532,
            True,
        )

        logger.close()
        lines = log.read_text().strip().split("\n")
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert entry["event"] == "TOOL_CALL"
        assert entry["name"] == "run_analysis"
        assert entry["elapsed"] == 0.532
        assert entry["success"] is True
        assert set(entry["args_keys"]) == {"kernel_id", "metric"}
        assert entry["result_chars"] == len('{"findings": [{"severity": "high"}]}')

    def test_tool_with_tool_calls_in_message(self, tmp_path: Path):
        logger = AgentLogger()
        logger._level = _L.PROMPT
        logger._use_structlog = True
        log = tmp_path / "test.jsonl"
        logger._fh = open(log, "w", encoding="utf-8")

        msgs = [
            FakeMessage(
                role="assistant",
                content="Let me check",
                tool_calls=[FakeToolCall(name="get_metrics", arguments={"id": 0})],
            ),
        ]
        logger.record_prompt(msgs, turn=2)

        logger.close()
        lines = log.read_text().strip().split("\n")

        # Second line: MESSAGE with tool_calls
        msg_entry = json.loads(lines[1])
        assert msg_entry["role"] == "ASSISTANT"
        assert "tool_calls" in msg_entry
        assert len(msg_entry["tool_calls"]) == 1
        assert msg_entry["tool_calls"][0]["name"] == "get_metrics"
        assert msg_entry["tool_calls"][0]["args_keys"] == ["id"]

    def test_all_lines_are_valid_json(self, tmp_path: Path):
        logger = AgentLogger()
        logger._level = _L.ALL
        logger._use_structlog = True
        log = tmp_path / "test.jsonl"
        logger._fh = open(log, "w", encoding="utf-8")

        msgs = [FakeMessage(role="user", content="test")]
        logger.record_prompt(msgs, turn=1)
        logger.record_tool("foo", {"a": 1}, "result", 0.1, True)

        logger.close()
        for line in log.read_text().strip().split("\n"):
            json.loads(line)  # should not raise


# ---------------------------------------------------------------------------
# configure_logging tests
# ---------------------------------------------------------------------------

class TestConfigureLogging:
    def test_default_level(self):
        import logging
        configure_logging(force=True)
        assert logging.getLogger().level == logging.WARNING

    def test_custom_level(self):
        import logging
        configure_logging(level=logging.DEBUG, force=True)
        assert logging.getLogger().level == logging.DEBUG

    def test_env_var_level(self):
        import logging
        with patch.dict(os.environ, {"TACHYON_LOG_LEVEL": "DEBUG"}):
            configure_logging(force=True)
        assert logging.getLogger().level == logging.DEBUG

    def test_httpx_suppressed(self):
        import logging
        configure_logging(level=logging.DEBUG, force=True)
        # httpx should be at least WARNING even when root is DEBUG
        assert logging.getLogger("httpx").level >= logging.WARNING

    def test_get_logger(self):
        logger = get_logger("tachyon.test")
        assert logger.name == "tachyon.test"
