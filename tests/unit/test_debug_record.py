"""Unit tests for debug recording module."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

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


class TestInit:
    def test_default_none(self):
        with patch.dict(os.environ, {}, clear=True):
            import tachyon.utils.debug_record as dr
            dr._level = dr._L.NONE
            dr._fh = None
            dr._path = None
            dr.init()
            assert level() == 0

    def test_prompt_level(self):
        with patch.dict(os.environ, {"TACHYON_DEBUG_RECORD": "prompt"}):
            import tachyon.utils.debug_record as dr
            dr._level = dr._L.NONE
            dr._fh = None
            dr._path = None
            dr.init()
            assert level() == 1
            assert log_path() is not None
            dr.close()

    def test_all_level(self):
        with patch.dict(os.environ, {"TACHYON_DEBUG_RECORD": "all"}):
            import tachyon.utils.debug_record as dr
            dr._level = dr._L.NONE
            dr._fh = None
            dr._path = None
            dr.init()
            assert level() == 2
            dr.close()

    def test_numeric_1(self):
        with patch.dict(os.environ, {"TACHYON_DEBUG_RECORD": "1"}):
            import tachyon.utils.debug_record as dr
            dr._level = dr._L.NONE
            dr._fh = None
            dr._path = None
            dr.init()
            assert level() == 1
            dr.close()

    def test_none_string(self):
        with patch.dict(os.environ, {"TACHYON_DEBUG_RECORD": "none"}):
            import tachyon.utils.debug_record as dr
            dr._level = dr._L.NONE
            dr._fh = None
            dr._path = None
            dr.init()
            assert level() == 0


class TestRecordPrompt:
    def test_noop_when_none(self):
        import tachyon.utils.debug_record as dr
        dr._level = dr._L.NONE
        record_prompt([])

    def test_records_human_readable(self, tmp_path: Path):
        import tachyon.utils.debug_record as dr
        dr._level = dr._L.PROMPT
        log = tmp_path / "test.log"
        dr._fh = open(log, "w", encoding="utf-8")

        msgs = [
            FakeMessage(role="system", content="You are helpful."),
            FakeMessage(role="user", content="hello"),
        ]
        record_prompt(msgs, tools=[1, 2, 3], turn=1)

        dr.close()
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
        import tachyon.utils.debug_record as dr
        dr._level = dr._L.PROMPT
        log = tmp_path / "test.log"
        dr._fh = open(log, "w", encoding="utf-8")

        msgs = [
            FakeMessage(role="user", content="line 1\nline 2\nline 3"),
        ]
        record_prompt(msgs, turn=1)

        dr.close()
        text = log.read_text()
        assert "line 1\nline 2\nline 3" in text


class TestRecordTool:
    def test_noop_when_prompt_level(self):
        import tachyon.utils.debug_record as dr
        dr._level = dr._L.PROMPT
        record_tool("foo", {}, "ok", 0.1, True)

    def test_records_human_readable(self, tmp_path: Path):
        import tachyon.utils.debug_record as dr
        dr._level = dr._L.ALL
        log = tmp_path / "test.log"
        dr._fh = open(log, "w", encoding="utf-8")

        record_tool("run_analysis", {"kernel_id": 0}, '{"findings": []}', 0.5, True)

        dr.close()
        text = log.read_text()
        assert "TOOL" in text
        assert "name=run_analysis" in text
        assert "elapsed=0.500s" in text
        assert "ok=True" in text
        assert '{"findings": []}' in text

    def test_result_truncated(self, tmp_path: Path):
        import tachyon.utils.debug_record as dr
        dr._level = dr._L.ALL
        log = tmp_path / "test.log"
        dr._fh = open(log, "w", encoding="utf-8")

        big_result = "x" * 10000
        record_tool("foo", {}, big_result, 0.1, True)

        dr.close()
        text = log.read_text()
        assert "truncated" in text
        assert "10000 total" in text
