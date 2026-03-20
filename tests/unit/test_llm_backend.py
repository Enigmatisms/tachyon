"""Comprehensive unit tests for the Tachyon LLM backend layer.

Covers:
  - Data models (Role, Message, ToolCall, Usage, CompletionResponse, StreamChunk)
  - create_backend factory
  - ToolDefinition format converters (to_openai, to_anthropic, to_mcp)
  - OpenAIBackend (message formatting, response parsing, tool call parsing)
  - AnthropicBackend (_split_system helper, tool choice mapping, finish reason mapping)
  - ContextManager (token estimation, compaction, distillation, budget checks)

All external SDK dependencies (openai, anthropic, litellm) are mocked so that
tests run without any provider packages installed.
"""
from __future__ import annotations

import json
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tachyon.llm.backend import (
    CompletionResponse,
    Message,
    Role,
    StreamChunk,
    ToolCall,
    Usage,
    create_backend,
)
from tachyon.tools.registry import ToolDefinition

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _make_tool_def(name: str = "read_file") -> ToolDefinition:
    """Create a sample ToolDefinition for testing format converters."""
    return ToolDefinition(
        name=name,
        description="Read the contents of a file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute file path"},
            },
            "required": ["path"],
        },
    )


def _fake_openai_module() -> ModuleType:
    """Build a minimal fake ``openai`` module with an ``AsyncOpenAI`` stub."""
    mod = ModuleType("openai")
    mod.AsyncOpenAI = MagicMock  # type: ignore[attr-defined]
    return mod


def _fake_anthropic_module() -> ModuleType:
    """Build a minimal fake ``anthropic`` module with an ``AsyncAnthropic`` stub."""
    mod = ModuleType("anthropic")
    mod.AsyncAnthropic = MagicMock  # type: ignore[attr-defined]
    return mod


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. Data Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDataModels:
    """Verify all LLM data model structures."""

    def test_role_enum_values(self):
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"
        assert Role.TOOL.value == "tool"

    def test_role_is_str_enum(self):
        """Role inherits from str so it can be used directly as a string."""
        assert isinstance(Role.USER, str)
        assert Role.USER == "user"

    def test_message_minimal(self):
        msg = Message(role=Role.USER, content="hello")
        assert msg.role is Role.USER
        assert msg.content == "hello"
        assert msg.tool_calls is None
        assert msg.tool_call_id is None
        assert msg.name is None

    def test_message_all_fields(self):
        tc = ToolCall(id="tc_1", name="read_file", arguments={"path": "/a"})
        msg = Message(
            role=Role.ASSISTANT,
            content="Let me read that.",
            tool_calls=[tc],
            tool_call_id="tc_1",
            name="read_file",
        )
        assert msg.tool_calls == [tc]
        assert msg.tool_call_id == "tc_1"
        assert msg.name == "read_file"

    def test_message_content_none_for_tool_call_only(self):
        """Assistant messages may have content=None when only tool_calls present."""
        msg = Message(
            role=Role.ASSISTANT,
            tool_calls=[ToolCall(id="tc_2", name="grep", arguments={"q": "x"})],
        )
        assert msg.content is None
        assert len(msg.tool_calls) == 1

    def test_tool_call_creation(self):
        tc = ToolCall(id="call_abc", name="run_bash", arguments={"cmd": "ls"})
        assert tc.id == "call_abc"
        assert tc.name == "run_bash"
        assert tc.arguments == {"cmd": "ls"}

    def test_usage_defaults(self):
        u = Usage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0

    def test_usage_with_values(self):
        u = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert u.prompt_tokens == 100
        assert u.completion_tokens == 50
        assert u.total_tokens == 150

    def test_completion_response_defaults(self):
        cr = CompletionResponse()
        assert cr.content is None
        assert cr.tool_calls == []
        assert isinstance(cr.usage, Usage)
        assert cr.finish_reason == "stop"

    def test_completion_response_with_tool_calls(self):
        tc = ToolCall(id="tc_5", name="edit", arguments={"file": "a.py"})
        cr = CompletionResponse(
            content=None,
            tool_calls=[tc],
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            finish_reason="tool_calls",
        )
        assert cr.content is None
        assert len(cr.tool_calls) == 1
        assert cr.finish_reason == "tool_calls"

    def test_stream_chunk_defaults(self):
        sc = StreamChunk()
        assert sc.content is None
        assert sc.tool_call_delta is None
        assert sc.finish_reason is None
        assert sc.usage is None

    def test_stream_chunk_with_content(self):
        sc = StreamChunk(content="Hello", finish_reason="stop")
        assert sc.content == "Hello"
        assert sc.finish_reason == "stop"

    def test_stream_chunk_with_tool_delta(self):
        sc = StreamChunk(
            tool_call_delta={"id": "tc_1", "name": "bash", "arguments": ""},
        )
        assert sc.tool_call_delta["id"] == "tc_1"

    def test_stream_chunk_with_usage(self):
        sc = StreamChunk(usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8))
        assert sc.usage.total_tokens == 8


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. create_backend factory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCreateBackend:
    """Test the create_backend factory function."""

    def test_unknown_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_backend(provider="invalid_provider", model="some-model")

    def test_unknown_provider_message_includes_supported_list(self):
        with pytest.raises(ValueError, match="openai, anthropic, litellm"):
            create_backend(provider="foo", model="m")

    def test_openai_provider_with_mock(self):
        """Providing 'openai' imports the OpenAI backend (mocked SDK)."""
        fake_mod = _fake_openai_module()
        with patch.dict(sys.modules, {"openai": fake_mod}):
            backend = create_backend(provider="openai", model="gpt-4o", api_key="test-key")
        assert backend.model == "gpt-4o"
        assert backend.api_key == "test-key"

    def test_anthropic_provider_with_mock(self):
        """Providing 'anthropic' imports the Anthropic backend (mocked SDK)."""
        fake_mod = _fake_anthropic_module()
        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            backend = create_backend(provider="anthropic", model="claude-sonnet-4-20250514", api_key="ak")
        assert backend.model == "claude-sonnet-4-20250514"

    def test_litellm_provider_creates_backend(self):
        """LiteLLMBackend does lazy import so it always constructs successfully."""
        backend = create_backend(provider="litellm", model="gpt-4o")
        assert backend.model == "gpt-4o"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. ToolDefinition format converters
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestToolDefinitionFormats:
    """Validate to_openai / to_anthropic / to_mcp output structures."""

    def test_to_openai_structure(self):
        tool = _make_tool_def()
        result = tool.to_openai()
        assert result["type"] == "function"
        assert result["function"]["name"] == "read_file"
        assert result["function"]["description"] == "Read the contents of a file."
        assert "properties" in result["function"]["parameters"]
        assert "path" in result["function"]["parameters"]["properties"]

    def test_to_anthropic_structure(self):
        tool = _make_tool_def()
        result = tool.to_anthropic()
        assert result["name"] == "read_file"
        assert result["description"] == "Read the contents of a file."
        assert result["input_schema"]["type"] == "object"
        assert "path" in result["input_schema"]["properties"]

    def test_to_mcp_structure(self):
        tool = _make_tool_def()
        result = tool.to_mcp()
        assert result["name"] == "read_file"
        assert result["description"] == "Read the contents of a file."
        assert result["inputSchema"]["type"] == "object"
        assert result["inputSchema"]["required"] == ["path"]

    def test_to_openai_no_extra_keys(self):
        """OpenAI format should only have 'type' and 'function' at top level."""
        tool = _make_tool_def()
        result = tool.to_openai()
        assert set(result.keys()) == {"type", "function"}
        assert set(result["function"].keys()) == {"name", "description", "parameters"}

    def test_to_anthropic_no_extra_keys(self):
        tool = _make_tool_def()
        result = tool.to_anthropic()
        assert set(result.keys()) == {"name", "description", "input_schema"}

    def test_to_mcp_no_extra_keys(self):
        tool = _make_tool_def()
        result = tool.to_mcp()
        assert set(result.keys()) == {"name", "description", "inputSchema"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4. OpenAIBackend
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestOpenAIBackend:
    """Test OpenAIBackend message formatting and response parsing."""

    @pytest.fixture(autouse=True)
    def _mock_openai_sdk(self):
        """Inject a fake openai module so OpenAIBackend can be imported."""
        fake = _fake_openai_module()
        with patch.dict(sys.modules, {"openai": fake}):
            from tachyon.llm.openai_backend import OpenAIBackend
            self.BackendCls = OpenAIBackend
            self.backend = OpenAIBackend(model="gpt-4o", api_key="test-key")
            yield

    # ---- _format_message ----

    def test_format_system_message(self):
        msg = Message(role=Role.SYSTEM, content="You are an assistant.")
        result = self.BackendCls._format_message(msg)
        assert result == {"role": "system", "content": "You are an assistant."}

    def test_format_user_message(self):
        msg = Message(role=Role.USER, content="What is 2+2?")
        result = self.BackendCls._format_message(msg)
        assert result == {"role": "user", "content": "What is 2+2?"}

    def test_format_assistant_message_plain(self):
        msg = Message(role=Role.ASSISTANT, content="4")
        result = self.BackendCls._format_message(msg)
        assert result == {"role": "assistant", "content": "4"}

    def test_format_tool_message(self):
        msg = Message(
            role=Role.TOOL,
            content="file contents here",
            tool_call_id="call_123",
            name="read_file",
        )
        result = self.BackendCls._format_message(msg)
        assert result["role"] == "tool"
        assert result["content"] == "file contents here"
        assert result["tool_call_id"] == "call_123"
        assert result["name"] == "read_file"

    def test_format_assistant_with_tool_calls(self):
        tc = ToolCall(id="call_abc", name="bash", arguments={"cmd": "ls"})
        msg = Message(role=Role.ASSISTANT, content=None, tool_calls=[tc])
        result = self.BackendCls._format_message(msg)
        assert result["role"] == "assistant"
        assert "content" not in result  # content is None, so omitted
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "call_abc"
        assert result["tool_calls"][0]["type"] == "function"
        assert result["tool_calls"][0]["function"]["name"] == "bash"
        args = json.loads(result["tool_calls"][0]["function"]["arguments"])
        assert args == {"cmd": "ls"}

    def test_format_assistant_with_content_and_tool_calls(self):
        tc = ToolCall(id="call_xyz", name="grep", arguments={"q": "foo"})
        msg = Message(role=Role.ASSISTANT, content="I will search for that.", tool_calls=[tc])
        result = self.BackendCls._format_message(msg)
        assert result["content"] == "I will search for that."
        assert len(result["tool_calls"]) == 1

    # ---- _parse_response ----

    def test_parse_response_text_only(self):
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "The answer is 42."
        mock_message.tool_calls = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )

        result = self.backend._parse_response(mock_response)

        assert isinstance(result, CompletionResponse)
        assert result.content == "The answer is 42."
        assert result.tool_calls == []
        assert result.finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15

    def test_parse_response_with_tool_calls(self):
        mock_tc = MagicMock()
        mock_tc.id = "call_001"
        mock_tc.function.name = "read_file"
        mock_tc.function.arguments = json.dumps({"path": "/tmp/x"})

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tc]

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(
            prompt_tokens=20, completion_tokens=10, total_tokens=30
        )

        result = self.backend._parse_response(mock_response)

        assert result.content is None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_001"
        assert result.tool_calls[0].name == "read_file"
        assert result.tool_calls[0].arguments == {"path": "/tmp/x"}
        assert result.finish_reason == "tool_calls"

    def test_parse_response_no_usage(self):
        mock_message = MagicMock()
        mock_message.content = "ok"
        mock_message.tool_calls = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        result = self.backend._parse_response(mock_response)
        assert result.usage.prompt_tokens == 0
        assert result.usage.total_tokens == 0

    # ---- parse_tool_calls ----

    def test_parse_tool_calls_empty(self):
        mock_message = MagicMock()
        mock_message.tool_calls = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        result = self.backend.parse_tool_calls(mock_response)
        assert result == []

    def test_parse_tool_calls_valid(self):
        mock_tc = MagicMock()
        mock_tc.id = "call_999"
        mock_tc.function.name = "edit_file"
        mock_tc.function.arguments = json.dumps({"path": "a.py", "content": "new"})

        mock_message = MagicMock()
        mock_message.tool_calls = [mock_tc]
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        result = self.backend.parse_tool_calls(mock_response)
        assert len(result) == 1
        assert result[0].name == "edit_file"
        assert result[0].arguments["path"] == "a.py"

    def test_parse_tool_calls_malformed_json(self):
        """Malformed arguments JSON should produce empty dict, not crash."""
        mock_tc = MagicMock()
        mock_tc.id = "call_bad"
        mock_tc.function.name = "broken"
        mock_tc.function.arguments = "{not valid json"

        mock_message = MagicMock()
        mock_message.tool_calls = [mock_tc]
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        result = self.backend.parse_tool_calls(mock_response)
        assert len(result) == 1
        assert result[0].arguments == {}

    def test_parse_tool_calls_multiple(self):
        """Multiple tool calls in one response."""
        tcs = []
        for i in range(3):
            mock_tc = MagicMock()
            mock_tc.id = f"call_{i}"
            mock_tc.function.name = f"tool_{i}"
            mock_tc.function.arguments = json.dumps({"idx": i})
            tcs.append(mock_tc)

        mock_message = MagicMock()
        mock_message.tool_calls = tcs
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        result = self.backend.parse_tool_calls(mock_response)
        assert len(result) == 3
        assert [tc.name for tc in result] == ["tool_0", "tool_1", "tool_2"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  5. AnthropicBackend
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAnthropicBackend:
    """Test Anthropic-specific message splitting and helpers."""

    @pytest.fixture(autouse=True)
    def _mock_anthropic_sdk(self):
        """Inject a fake anthropic module."""
        fake = _fake_anthropic_module()
        with patch.dict(sys.modules, {"anthropic": fake}):
            from tachyon.llm.anthropic_backend import (
                AnthropicBackend,
                _split_system,
            )
            self.BackendCls = AnthropicBackend
            self.backend = AnthropicBackend(model="claude-sonnet-4-20250514", api_key="test-key")
            self._split_system = _split_system
            yield

    # ---- _split_system ----

    def test_split_system_extracts_system(self):
        messages = [
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="Hi"),
        ]
        system_text, converted = self._split_system(messages)
        assert system_text == "You are helpful."
        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert converted[0]["content"] == "Hi"

    def test_split_system_no_system_message(self):
        messages = [Message(role=Role.USER, content="Hello")]
        system_text, converted = self._split_system(messages)
        assert system_text is None
        assert len(converted) == 1

    def test_split_system_tool_result_conversion(self):
        """Role.TOOL messages become user-role with tool_result content blocks."""
        messages = [
            Message(
                role=Role.TOOL,
                content="file contents here",
                tool_call_id="tu_123",
                name="read_file",
            ),
        ]
        _, converted = self._split_system(messages)
        assert len(converted) == 1
        msg = converted[0]
        assert msg["role"] == "user"
        assert len(msg["content"]) == 1
        block = msg["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "tu_123"
        assert block["content"] == "file contents here"

    def test_split_system_tool_result_empty_content(self):
        """Tool with None content gets empty string."""
        messages = [
            Message(role=Role.TOOL, content=None, tool_call_id="tu_456"),
        ]
        _, converted = self._split_system(messages)
        assert converted[0]["content"][0]["content"] == ""

    def test_split_system_assistant_with_tool_calls(self):
        """Assistant messages with tool_calls become tool_use content blocks."""
        tc = ToolCall(id="tu_789", name="bash", arguments={"cmd": "pwd"})
        messages = [
            Message(role=Role.ASSISTANT, content="Let me check.", tool_calls=[tc]),
        ]
        _, converted = self._split_system(messages)
        assert len(converted) == 1
        msg = converted[0]
        assert msg["role"] == "assistant"
        # Should have text block + tool_use block
        assert len(msg["content"]) == 2
        assert msg["content"][0] == {"type": "text", "text": "Let me check."}
        assert msg["content"][1]["type"] == "tool_use"
        assert msg["content"][1]["id"] == "tu_789"
        assert msg["content"][1]["name"] == "bash"
        assert msg["content"][1]["input"] == {"cmd": "pwd"}

    def test_split_system_assistant_tool_calls_no_text(self):
        """Assistant with tool_calls but no content -> only tool_use blocks."""
        tc = ToolCall(id="tu_001", name="grep", arguments={"q": "pattern"})
        messages = [
            Message(role=Role.ASSISTANT, content=None, tool_calls=[tc]),
        ]
        _, converted = self._split_system(messages)
        msg = converted[0]
        assert len(msg["content"]) == 1
        assert msg["content"][0]["type"] == "tool_use"

    def test_split_system_plain_user_and_assistant(self):
        """Plain user/assistant messages pass through."""
        messages = [
            Message(role=Role.USER, content="question"),
            Message(role=Role.ASSISTANT, content="answer"),
        ]
        _, converted = self._split_system(messages)
        assert converted[0] == {"role": "user", "content": "question"}
        assert converted[1] == {"role": "assistant", "content": "answer"}

    def test_split_system_full_conversation(self):
        """Full multi-turn conversation with system, user, tool calls, tool results."""
        tc = ToolCall(id="tu_full", name="read_file", arguments={"path": "/a"})
        messages = [
            Message(role=Role.SYSTEM, content="System prompt."),
            Message(role=Role.USER, content="Read my file."),
            Message(role=Role.ASSISTANT, content="Sure.", tool_calls=[tc]),
            Message(role=Role.TOOL, content="file data", tool_call_id="tu_full", name="read_file"),
            Message(role=Role.ASSISTANT, content="Here is the file data."),
        ]
        system_text, converted = self._split_system(messages)
        assert system_text == "System prompt."
        assert len(converted) == 4  # user, assistant+tool_use, user+tool_result, assistant
        assert converted[0]["role"] == "user"
        assert converted[1]["role"] == "assistant"
        assert converted[2]["role"] == "user"  # tool result mapped to user
        assert converted[3]["role"] == "assistant"

    # ---- _map_tool_choice ----

    def test_map_tool_choice_auto(self):
        result = self.BackendCls._map_tool_choice("auto")
        assert result == {"type": "auto"}

    def test_map_tool_choice_none(self):
        result = self.BackendCls._map_tool_choice("none")
        assert result == {"type": "none"}

    def test_map_tool_choice_required(self):
        result = self.BackendCls._map_tool_choice("required")
        assert result == {"type": "any"}

    def test_map_tool_choice_specific_tool(self):
        result = self.BackendCls._map_tool_choice("read_file")
        assert result == {"type": "tool", "name": "read_file"}

    # ---- _map_finish_reason ----

    def test_map_finish_reason_end_turn(self):
        assert self.BackendCls._map_finish_reason("end_turn") == "stop"

    def test_map_finish_reason_stop_sequence(self):
        assert self.BackendCls._map_finish_reason("stop_sequence") == "stop"

    def test_map_finish_reason_tool_use(self):
        assert self.BackendCls._map_finish_reason("tool_use") == "tool_calls"

    def test_map_finish_reason_max_tokens(self):
        assert self.BackendCls._map_finish_reason("max_tokens") == "length"

    def test_map_finish_reason_none(self):
        assert self.BackendCls._map_finish_reason(None) == "stop"

    # ---- parse_tool_calls ----

    def test_parse_tool_calls_with_tool_use_blocks(self):
        """parse_tool_calls extracts ToolCalls from content blocks."""
        block1 = MagicMock()
        block1.type = "text"
        block1.text = "I will read the file."

        block2 = MagicMock()
        block2.type = "tool_use"
        block2.id = "tu_42"
        block2.name = "read_file"
        block2.input = {"path": "/etc/hosts"}

        mock_response = MagicMock()
        mock_response.content = [block1, block2]

        result = self.backend.parse_tool_calls(mock_response)
        assert len(result) == 1
        assert result[0].id == "tu_42"
        assert result[0].name == "read_file"
        assert result[0].arguments == {"path": "/etc/hosts"}

    def test_parse_tool_calls_empty_response(self):
        mock_response = MagicMock(spec=[])  # no content attr
        result = self.backend.parse_tool_calls(mock_response)
        assert result == []

    def test_parse_tool_calls_non_dict_input(self):
        """If block.input is not a dict, arguments should be empty dict."""
        block = MagicMock()
        block.type = "tool_use"
        block.id = "tu_99"
        block.name = "bad_tool"
        block.input = "not-a-dict"

        mock_response = MagicMock()
        mock_response.content = [block]

        result = self.backend.parse_tool_calls(mock_response)
        assert len(result) == 1
        assert result[0].arguments == {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  6. LiteLLMBackend helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLiteLLMBackend:
    """Test LiteLLMBackend helper methods (no SDK needed for these)."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from tachyon.llm.litellm_backend import (
            LiteLLMBackend,
            _message_to_dict,
            _role_to_str,
        )
        self.BackendCls = LiteLLMBackend
        self.backend = LiteLLMBackend(model="gpt-4o")
        self._message_to_dict = _message_to_dict
        self._role_to_str = _role_to_str

    def test_role_to_str(self):
        assert self._role_to_str(Role.SYSTEM) == "system"
        assert self._role_to_str(Role.TOOL) == "tool"

    def test_message_to_dict_user(self):
        msg = Message(role=Role.USER, content="hello")
        d = self._message_to_dict(msg)
        assert d == {"role": "user", "content": "hello"}

    def test_message_to_dict_assistant_no_content_with_tool_calls(self):
        """Assistant with tool_calls and no content: content key omitted."""
        tc = ToolCall(id="tc_1", name="bash", arguments={"cmd": "ls"})
        msg = Message(role=Role.ASSISTANT, content=None, tool_calls=[tc])
        d = self._message_to_dict(msg)
        assert "content" not in d
        assert len(d["tool_calls"]) == 1

    def test_message_to_dict_non_assistant_no_content_gets_empty_string(self):
        """Non-assistant messages with None content get empty string for OpenAI compat."""
        msg = Message(role=Role.USER, content=None)
        d = self._message_to_dict(msg)
        assert d["content"] == ""

    def test_normalize_finish_reason(self):
        assert self.BackendCls._normalize_finish_reason("stop") == "stop"
        assert self.BackendCls._normalize_finish_reason("tool_calls") == "tool_calls"
        assert self.BackendCls._normalize_finish_reason("function_call") == "tool_calls"
        assert self.BackendCls._normalize_finish_reason("length") == "length"
        assert self.BackendCls._normalize_finish_reason("content_filter") == "content_filter"

    def test_parse_usage_none(self):
        result = self.BackendCls._parse_usage(None)
        assert result.prompt_tokens == 0

    def test_parse_usage_dict(self):
        result = self.BackendCls._parse_usage(
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        )
        assert result.prompt_tokens == 100
        assert result.total_tokens == 150

    def test_parse_usage_object(self):
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 200
        mock_usage.completion_tokens = 80
        mock_usage.total_tokens = 280
        result = self.BackendCls._parse_usage(mock_usage)
        assert result.prompt_tokens == 200
        assert result.total_tokens == 280

    def test_parse_tool_calls_str_arguments(self):
        """Arguments as JSON string should be parsed."""
        mock_tc = MagicMock()
        mock_tc.id = "tc_lit_1"
        mock_tc.function.name = "grep"
        mock_tc.function.arguments = json.dumps({"pattern": "foo"})

        mock_message = MagicMock()
        mock_message.tool_calls = [mock_tc]
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        result = self.backend.parse_tool_calls(mock_response)
        assert result[0].arguments == {"pattern": "foo"}

    def test_parse_tool_calls_dict_arguments(self):
        """Arguments already a dict should be used directly."""
        mock_tc = MagicMock()
        mock_tc.id = "tc_lit_2"
        mock_tc.function.name = "edit"
        mock_tc.function.arguments = {"file": "a.py"}

        mock_message = MagicMock()
        mock_message.tool_calls = [mock_tc]
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        result = self.backend.parse_tool_calls(mock_response)
        assert result[0].arguments == {"file": "a.py"}

    def test_parse_tool_calls_no_choices(self):
        mock_response = MagicMock()
        mock_response.choices = []
        result = self.backend.parse_tool_calls(mock_response)
        assert result == []

    def test_ensure_litellm_raises_if_missing(self):
        """_ensure_litellm raises ImportError when litellm is not installed."""
        with patch.dict(sys.modules, {"litellm": None}):
            backend = self.BackendCls(model="test")
            with pytest.raises(ImportError, match="litellm is required"):
                backend._ensure_litellm()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  7. ContextManager
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestContextManager:
    """Test the ContextManager from tachyon.agent.context."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from tachyon.agent.context import ContextManager
        self.cm = ContextManager(budget=1000)

    def test_estimate_tokens_empty(self):
        assert self.cm.estimate_tokens([]) == 0

    def test_estimate_tokens_system_message(self):
        """System messages use weight=3 (len/3)."""
        msg = Message(role=Role.SYSTEM, content="x" * 300)
        tokens = self.cm.estimate_tokens([msg])
        assert tokens == 100  # 300 // 3

    def test_estimate_tokens_user_message(self):
        """Non-system messages use weight=4 (len/4)."""
        msg = Message(role=Role.USER, content="x" * 400)
        tokens = self.cm.estimate_tokens([msg])
        assert tokens == 100  # 400 // 4

    def test_estimate_tokens_with_tool_calls(self):
        """Tool call arguments are also counted."""
        tc = ToolCall(id="tc_1", name="bash", arguments={"cmd": "echo hello world"})
        msg = Message(role=Role.ASSISTANT, content="ok", tool_calls=[tc])
        tokens = self.cm.estimate_tokens([msg])
        # "ok" is 2 chars // 4 = 0, plus json.dumps(arguments) length // 4
        args_json = json.dumps(tc.arguments)
        expected = len("ok") // 4 + len(args_json) // 4
        assert tokens == expected

    def test_estimate_tokens_multiple_messages(self):
        messages = [
            Message(role=Role.SYSTEM, content="a" * 30),    # 30 // 3 = 10
            Message(role=Role.USER, content="b" * 40),      # 40 // 4 = 10
            Message(role=Role.ASSISTANT, content="c" * 80),  # 80 // 4 = 20
        ]
        assert self.cm.estimate_tokens(messages) == 40

    def test_compact_tool_results_shortens_old_results(self):
        """Old tool results (beyond keep_last_n) with long content are compacted."""
        messages = [
            Message(role=Role.USER, content="read files"),
            Message(
                role=Role.TOOL,
                content="A" * 300,  # old, long -> should be compacted
                tool_call_id="tc_old",
                name="read_file",
            ),
            Message(
                role=Role.TOOL,
                content="B" * 300,  # recent -> should be kept
                tool_call_id="tc_new",
                name="grep",
            ),
        ]
        self.cm.compact_tool_results(messages, keep_last_n=1)

        # Old result should be compacted
        assert messages[1].content.startswith("[read_file:")
        assert "..." in messages[1].content
        assert len(messages[1].content) < 300

        # Recent result should remain intact
        assert messages[2].content == "B" * 300

    def test_compact_tool_results_keeps_recent_intact(self):
        """When keep_last_n covers all tool results, nothing changes."""
        original_content = "Short result"
        messages = [
            Message(role=Role.TOOL, content=original_content, tool_call_id="tc_1", name="t"),
        ]
        self.cm.compact_tool_results(messages, keep_last_n=1)
        assert messages[0].content == original_content

    def test_compact_tool_results_skips_short_content(self):
        """Old results with content shorter than threshold are left alone."""
        messages = [
            Message(role=Role.TOOL, content="short", tool_call_id="tc_1", name="t1"),
            Message(role=Role.TOOL, content="recent", tool_call_id="tc_2", name="t2"),
        ]
        self.cm.compact_tool_results(messages, keep_last_n=1)
        assert messages[0].content == "short"  # under 200 chars, not compacted

    def test_compact_tool_results_preserves_metadata(self):
        """Compacted messages retain tool_call_id and name."""
        messages = [
            Message(role=Role.TOOL, content="X" * 300, tool_call_id="tc_old", name="read_file"),
            Message(role=Role.TOOL, content="recent", tool_call_id="tc_new", name="grep"),
        ]
        self.cm.compact_tool_results(messages, keep_last_n=1)
        assert messages[0].tool_call_id == "tc_old"
        assert messages[0].name == "read_file"

    def test_distill_collapses_tool_pairs(self):
        """Tool-call/result pairs are collapsed into a single summary."""
        tc = ToolCall(id="tc_1", name="read_file", arguments={"path": "/a"})
        messages = [
            Message(role=Role.SYSTEM, content="sys"),
            Message(role=Role.USER, content="read my file"),
            Message(role=Role.ASSISTANT, content=None, tool_calls=[tc]),
            Message(role=Role.TOOL, content="file data", tool_call_id="tc_1", name="read_file"),
            Message(role=Role.ASSISTANT, content="Done."),
        ]
        result = self.cm.distill(messages)

        # System, user, collapsed summary, final assistant = 4 messages
        assert len(result) == 4
        assert result[0].role is Role.SYSTEM
        assert result[1].role is Role.USER
        assert result[2].role is Role.ASSISTANT
        assert "read_file" in result[2].content
        assert result[2].content.startswith("[Agent called:")
        assert result[3].content == "Done."

    def test_distill_preserves_plain_messages(self):
        """Non-tool messages pass through distillation unchanged."""
        messages = [
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi there!"),
        ]
        result = self.cm.distill(messages)
        assert len(result) == 2
        assert result[0].content == "Hello"
        assert result[1].content == "Hi there!"

    def test_distill_multiple_tool_calls(self):
        """Multiple tool_calls in one assistant message are all listed in summary."""
        tcs = [
            ToolCall(id="tc_1", name="read_file", arguments={}),
            ToolCall(id="tc_2", name="grep", arguments={}),
        ]
        messages = [
            Message(role=Role.ASSISTANT, tool_calls=tcs),
            Message(role=Role.TOOL, content="r1", tool_call_id="tc_1", name="read_file"),
            Message(role=Role.TOOL, content="r2", tool_call_id="tc_2", name="grep"),
        ]
        result = self.cm.distill(messages)
        assert len(result) == 1
        assert "read_file" in result[0].content
        assert "grep" in result[0].content

    def test_should_compact_before_threshold(self):
        assert not self.cm.should_compact(0)
        assert not self.cm.should_compact(4)

    def test_should_compact_at_threshold(self):
        assert self.cm.should_compact(5)

    def test_should_compact_after_threshold(self):
        assert self.cm.should_compact(10)

    def test_is_near_budget_below(self):
        """Messages well below budget should return False."""
        messages = [Message(role=Role.USER, content="x" * 40)]  # ~10 tokens
        assert not self.cm.is_near_budget(messages)

    def test_is_near_budget_above(self):
        """Messages exceeding 85% of budget should return True."""
        # budget=1000, 85% = 850 tokens needed
        # weight=4 for user messages, so need 850*4=3400 chars
        messages = [Message(role=Role.USER, content="x" * 3500)]
        assert self.cm.is_near_budget(messages)

    def test_is_near_budget_at_boundary(self):
        """Test near the exact 85% boundary."""
        # budget=1000, threshold=850 tokens
        # 3400 chars / 4 = 850 tokens exactly -> should return False (not >)
        messages = [Message(role=Role.USER, content="x" * 3400)]
        assert not self.cm.is_near_budget(messages)

        # 3404 chars / 4 = 851 tokens -> should return True
        messages_over = [Message(role=Role.USER, content="x" * 3404)]
        assert self.cm.is_near_budget(messages_over)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  8. OpenAIBackend — chat_completion integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestOpenAIBackendChatCompletion:
    """Test OpenAIBackend.chat_completion with a fully mocked openai SDK."""

    @pytest.fixture(autouse=True)
    def _mock_openai_sdk(self):
        """Inject a fake openai module and create a backend with a mock client."""
        fake = _fake_openai_module()
        with patch.dict(sys.modules, {"openai": fake}):
            from tachyon.llm.openai_backend import OpenAIBackend
            self.BackendCls = OpenAIBackend
            self.backend = OpenAIBackend(model="gpt-4o", api_key="test-key")
            # Replace the client with a fully controlled mock
            self.mock_client = MagicMock()
            self.backend._client = self.mock_client
            yield

    def _make_mock_response(
        self,
        content="test response",
        tool_calls_data=None,
        finish_reason="stop",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    ):
        """Build a mock OpenAI ChatCompletion response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = content
        mock_response.choices[0].finish_reason = finish_reason

        if tool_calls_data:
            mock_tcs = []
            for tc_data in tool_calls_data:
                mock_tc = MagicMock()
                mock_tc.id = tc_data["id"]
                mock_tc.function.name = tc_data["name"]
                mock_tc.function.arguments = json.dumps(tc_data["arguments"])
                mock_tcs.append(mock_tc)
            mock_response.choices[0].message.tool_calls = mock_tcs
        else:
            mock_response.choices[0].message.tool_calls = None

        mock_response.usage = MagicMock(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        return mock_response

    @pytest.mark.asyncio
    async def test_chat_completion_non_streaming(self):
        """Non-streaming chat_completion returns a CompletionResponse."""
        mock_resp = self._make_mock_response(
            content="Hello world",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        self.mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        messages = [Message(role=Role.USER, content="Say hello")]
        result = await self.backend.chat_completion(messages, stream=False)

        assert isinstance(result, CompletionResponse)
        assert result.content == "Hello world"
        assert result.tool_calls == []
        assert result.finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_chat_completion_non_streaming_with_tools(self):
        """Non-streaming chat_completion passes tool definitions and extracts tool calls."""
        mock_resp = self._make_mock_response(
            content=None,
            tool_calls_data=[
                {"id": "call_abc", "name": "read_file", "arguments": {"path": "/tmp/x"}},
            ],
            finish_reason="tool_calls",
        )
        self.mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        tool_def = _make_tool_def("read_file")
        messages = [Message(role=Role.USER, content="Read my file")]
        result = await self.backend.chat_completion(
            messages, tools=[tool_def], tool_choice="auto"
        )

        assert isinstance(result, CompletionResponse)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_abc"
        assert result.tool_calls[0].name == "read_file"
        assert result.tool_calls[0].arguments == {"path": "/tmp/x"}
        assert result.finish_reason == "tool_calls"

        # Verify tools were passed to the API call
        call_kwargs = self.mock_client.chat.completions.create.call_args
        assert "tools" in call_kwargs.kwargs
        assert "tool_choice" in call_kwargs.kwargs

    @pytest.mark.asyncio
    async def test_chat_completion_non_streaming_multiple_tool_calls(self):
        """Multiple tool calls in one response are all extracted."""
        mock_resp = self._make_mock_response(
            content="Let me check both.",
            tool_calls_data=[
                {"id": "call_1", "name": "read_file", "arguments": {"path": "/a"}},
                {"id": "call_2", "name": "grep", "arguments": {"pattern": "foo"}},
            ],
            finish_reason="tool_calls",
        )
        self.mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        messages = [Message(role=Role.USER, content="Check both")]
        result = await self.backend.chat_completion(messages)

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "read_file"
        assert result.tool_calls[1].name == "grep"

    @pytest.mark.asyncio
    async def test_chat_completion_streaming_returns_async_iterator(self):
        """Streaming chat_completion returns an async iterator."""

        async def _fake_stream():
            # Chunk with text content
            chunk1 = MagicMock()
            chunk1.choices = [MagicMock()]
            chunk1.choices[0].delta.content = "Hello"
            chunk1.choices[0].delta.tool_calls = None
            chunk1.choices[0].finish_reason = None
            chunk1.usage = None
            yield chunk1

            # Final chunk with finish_reason and usage
            chunk2 = MagicMock()
            chunk2.choices = [MagicMock()]
            chunk2.choices[0].delta.content = None
            chunk2.choices[0].delta.tool_calls = None
            chunk2.choices[0].finish_reason = "stop"
            chunk2.usage = MagicMock(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            )
            yield chunk2

        self.mock_client.chat.completions.create = AsyncMock(
            return_value=_fake_stream()
        )

        messages = [Message(role=Role.USER, content="Stream test")]
        result = await self.backend.chat_completion(messages, stream=True)

        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].content == "Hello"
        assert chunks[1].finish_reason == "stop"
        assert chunks[1].usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_chat_completion_streaming_usage_only_chunk(self):
        """Final chunk with no choices but usage is handled correctly."""

        async def _fake_stream():
            # Normal content chunk
            chunk1 = MagicMock()
            chunk1.choices = [MagicMock()]
            chunk1.choices[0].delta.content = "Hi"
            chunk1.choices[0].delta.tool_calls = None
            chunk1.choices[0].finish_reason = None
            chunk1.usage = None
            yield chunk1

            # Usage-only chunk (no choices)
            chunk2 = MagicMock()
            chunk2.choices = []
            chunk2.usage = MagicMock(
                prompt_tokens=20, completion_tokens=10, total_tokens=30
            )
            yield chunk2

        self.mock_client.chat.completions.create = AsyncMock(
            return_value=_fake_stream()
        )

        messages = [Message(role=Role.USER, content="Test")]
        result = await self.backend.chat_completion(messages, stream=True)

        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].content == "Hi"
        assert chunks[1].usage is not None
        assert chunks[1].usage.prompt_tokens == 20

    @pytest.mark.asyncio
    async def test_chat_completion_streaming_tool_call_delta(self):
        """Streaming with tool call deltas."""

        async def _fake_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = None

            tc_delta = MagicMock()
            tc_delta.index = 0
            tc_delta.id = "call_s1"
            tc_delta.function.name = "read_file"
            tc_delta.function.arguments = '{"path":'
            chunk.choices[0].delta.tool_calls = [tc_delta]
            chunk.choices[0].finish_reason = None
            chunk.usage = None
            yield chunk

        self.mock_client.chat.completions.create = AsyncMock(
            return_value=_fake_stream()
        )

        messages = [Message(role=Role.USER, content="Test")]
        result = await self.backend.chat_completion(messages, stream=True)

        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].tool_call_delta is not None
        assert chunks[0].tool_call_delta["id"] == "call_s1"
        assert chunks[0].tool_call_delta["function"]["name"] == "read_file"

    def test_format_message_tool_role(self):
        """Tool role messages include tool_call_id and name."""
        msg = Message(
            role=Role.TOOL,
            content="result data",
            tool_call_id="call_xyz",
            name="my_tool",
        )
        result = self.BackendCls._format_message(msg)
        assert result["role"] == "tool"
        assert result["content"] == "result data"
        assert result["tool_call_id"] == "call_xyz"
        assert result["name"] == "my_tool"

    def test_format_message_content_none_excluded(self):
        """When content is None, it should not appear in the dict."""
        msg = Message(role=Role.ASSISTANT, content=None)
        result = self.BackendCls._format_message(msg)
        assert "content" not in result

    def test_format_tool_definitions(self):
        """format_tool_definitions calls to_openai on each tool."""
        tool = _make_tool_def("bash")
        result = self.backend.format_tool_definitions([tool])
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "bash"

    def test_parse_response_finish_reason_none_defaults_to_stop(self):
        """If finish_reason is None, it defaults to 'stop'."""
        mock_resp = self._make_mock_response(content="ok")
        mock_resp.choices[0].finish_reason = None
        result = self.backend._parse_response(mock_resp)
        assert result.finish_reason == "stop"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  9. AnthropicBackend — chat_completion + method tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAnthropicBackendMethods:
    """Test AnthropicBackend methods with a fully mocked SDK."""

    @pytest.fixture(autouse=True)
    def _mock_anthropic_sdk(self):
        """Inject a fake anthropic module and create a backend."""
        fake = _fake_anthropic_module()
        with patch.dict(sys.modules, {"anthropic": fake}):
            from tachyon.llm.anthropic_backend import (
                AnthropicBackend,
                _split_system,
            )
            self.BackendCls = AnthropicBackend
            self.backend = AnthropicBackend(
                model="claude-sonnet-4-20250514", api_key="test-key"
            )
            self._split_system = _split_system
            # Replace client with controlled mock
            self.mock_client = MagicMock()
            self.backend._client = self.mock_client
            yield

    # ---- _split_system edge cases ----

    def test_split_system_multiple_system_messages(self):
        """Multiple system messages: last one wins."""
        messages = [
            Message(role=Role.SYSTEM, content="First system"),
            Message(role=Role.SYSTEM, content="Second system"),
            Message(role=Role.USER, content="Hi"),
        ]
        system_text, converted = self._split_system(messages)
        assert system_text == "Second system"
        assert len(converted) == 1

    def test_split_system_assistant_tool_calls_multiple(self):
        """Assistant with multiple tool_calls produces multiple tool_use blocks."""
        tc1 = ToolCall(id="tu_a", name="read_file", arguments={"path": "/a"})
        tc2 = ToolCall(id="tu_b", name="grep", arguments={"pattern": "x"})
        messages = [
            Message(role=Role.ASSISTANT, content=None, tool_calls=[tc1, tc2]),
        ]
        _, converted = self._split_system(messages)
        assert len(converted) == 1
        blocks = converted[0]["content"]
        assert len(blocks) == 2  # no text block since content is None
        assert blocks[0]["type"] == "tool_use"
        assert blocks[0]["name"] == "read_file"
        assert blocks[1]["type"] == "tool_use"
        assert blocks[1]["name"] == "grep"
        assert blocks[1]["input"] == {"pattern": "x"}

    def test_split_system_user_none_content(self):
        """User message with None content gets empty string."""
        messages = [Message(role=Role.USER, content=None)]
        _, converted = self._split_system(messages)
        assert converted[0]["content"] == ""

    def test_split_system_empty_messages(self):
        """Empty messages list returns None system and empty converted."""
        system_text, converted = self._split_system([])
        assert system_text is None
        assert converted == []

    # ---- _complete (non-streaming) ----

    @pytest.mark.asyncio
    async def test_complete_text_only(self):
        """Non-streaming completion with only text content."""
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Here is my response."

        mock_resp = MagicMock()
        mock_resp.content = [text_block]
        mock_resp.stop_reason = "end_turn"
        mock_resp.usage = MagicMock(input_tokens=50, output_tokens=25)

        self.mock_client.messages.create = AsyncMock(return_value=mock_resp)

        messages = [Message(role=Role.USER, content="Hello")]
        result = await self.backend.chat_completion(messages)

        assert isinstance(result, CompletionResponse)
        assert result.content == "Here is my response."
        assert result.tool_calls == []
        assert result.finish_reason == "stop"
        assert result.usage.prompt_tokens == 50
        assert result.usage.completion_tokens == 25
        assert result.usage.total_tokens == 75

    @pytest.mark.asyncio
    async def test_complete_with_tool_use(self):
        """Non-streaming completion with tool_use content blocks."""
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "I will read the file."

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "tu_42"
        tool_block.name = "read_file"
        tool_block.input = {"path": "/etc/hosts"}

        mock_resp = MagicMock()
        mock_resp.content = [text_block, tool_block]
        mock_resp.stop_reason = "tool_use"
        mock_resp.usage = MagicMock(input_tokens=30, output_tokens=20)

        self.mock_client.messages.create = AsyncMock(return_value=mock_resp)

        messages = [Message(role=Role.USER, content="Read my file")]
        result = await self.backend.chat_completion(messages)

        assert result.content == "I will read the file."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "tu_42"
        assert result.tool_calls[0].name == "read_file"
        assert result.tool_calls[0].arguments == {"path": "/etc/hosts"}
        assert result.finish_reason == "tool_calls"
        assert result.usage.total_tokens == 50

    @pytest.mark.asyncio
    async def test_complete_with_system_prompt(self):
        """System message is extracted and passed as top-level 'system' kwarg."""
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "OK"

        mock_resp = MagicMock()
        mock_resp.content = [text_block]
        mock_resp.stop_reason = "end_turn"
        mock_resp.usage = MagicMock(input_tokens=10, output_tokens=5)

        self.mock_client.messages.create = AsyncMock(return_value=mock_resp)

        messages = [
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="Hi"),
        ]
        await self.backend.chat_completion(messages)

        call_kwargs = self.mock_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "You are helpful."
        # Verify system message is NOT in the messages list
        api_messages = call_kwargs["messages"]
        assert all(m["role"] != "system" for m in api_messages)

    @pytest.mark.asyncio
    async def test_complete_with_tools_passes_tool_choice(self):
        """When tools are provided, tool definitions and tool_choice are sent."""
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Done"

        mock_resp = MagicMock()
        mock_resp.content = [text_block]
        mock_resp.stop_reason = "end_turn"
        mock_resp.usage = MagicMock(input_tokens=10, output_tokens=5)

        self.mock_client.messages.create = AsyncMock(return_value=mock_resp)

        tool_def = _make_tool_def("read_file")
        messages = [Message(role=Role.USER, content="Read")]
        await self.backend.chat_completion(
            messages, tools=[tool_def], tool_choice="required"
        )

        call_kwargs = self.mock_client.messages.create.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tool_choice"] == {"type": "any"}  # "required" -> "any"

    @pytest.mark.asyncio
    async def test_complete_no_text_only_tool_use(self):
        """Response with only tool_use blocks and no text."""
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "tu_99"
        tool_block.name = "bash"
        tool_block.input = {"cmd": "ls"}

        mock_resp = MagicMock()
        mock_resp.content = [tool_block]
        mock_resp.stop_reason = "tool_use"
        mock_resp.usage = MagicMock(input_tokens=10, output_tokens=15)

        self.mock_client.messages.create = AsyncMock(return_value=mock_resp)

        messages = [Message(role=Role.USER, content="List files")]
        result = await self.backend.chat_completion(messages)

        assert result.content is None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "bash"

    @pytest.mark.asyncio
    async def test_complete_tool_use_non_dict_input(self):
        """tool_use block with non-dict input results in empty arguments."""
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "tu_bad"
        tool_block.name = "broken"
        tool_block.input = "not-a-dict"

        mock_resp = MagicMock()
        mock_resp.content = [tool_block]
        mock_resp.stop_reason = "tool_use"
        mock_resp.usage = MagicMock(input_tokens=5, output_tokens=5)

        self.mock_client.messages.create = AsyncMock(return_value=mock_resp)

        messages = [Message(role=Role.USER, content="Trigger")]
        result = await self.backend.chat_completion(messages)

        assert result.tool_calls[0].arguments == {}

    # ---- parse_tool_calls from response object ----

    def test_parse_tool_calls_text_and_tool_use_blocks(self):
        """parse_tool_calls extracts only tool_use blocks, ignoring text blocks."""
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Some text"

        tool_block1 = MagicMock()
        tool_block1.type = "tool_use"
        tool_block1.id = "tu_1"
        tool_block1.name = "read_file"
        tool_block1.input = {"path": "/a"}

        tool_block2 = MagicMock()
        tool_block2.type = "tool_use"
        tool_block2.id = "tu_2"
        tool_block2.name = "grep"
        tool_block2.input = {"pattern": "x"}

        mock_response = MagicMock()
        mock_response.content = [text_block, tool_block1, tool_block2]

        result = self.backend.parse_tool_calls(mock_response)
        assert len(result) == 2
        assert result[0].id == "tu_1"
        assert result[1].id == "tu_2"

    # ---- format_tool_definitions ----

    def test_format_tool_definitions_calls_to_anthropic(self):
        """format_tool_definitions should call to_anthropic on each tool."""
        tool = _make_tool_def("bash")
        result = self.backend.format_tool_definitions([tool])
        assert len(result) == 1
        assert result[0]["name"] == "bash"
        assert "input_schema" in result[0]

    # ---- _map_finish_reason edge cases ----

    def test_map_finish_reason_unknown(self):
        """Unknown stop_reason defaults to 'stop'."""
        assert self.BackendCls._map_finish_reason("unknown_reason") == "stop"

    def test_map_finish_reason_empty_string(self):
        """Empty string maps to 'stop'."""
        assert self.BackendCls._map_finish_reason("") == "stop"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  10. LiteLLMBackend — chat_completion + method tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLiteLLMBackendMethods:
    """Test LiteLLMBackend with mocked litellm module."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from tachyon.llm.litellm_backend import LiteLLMBackend, _message_to_dict
        self.BackendCls = LiteLLMBackend
        self._message_to_dict = _message_to_dict

        # Create backend and inject a mock litellm module
        self.backend = LiteLLMBackend(model="gpt-4o", api_key="test-key", base_url="http://localhost:8080")
        self.mock_litellm = MagicMock()
        self.backend._litellm = self.mock_litellm
        yield

    def _make_mock_response(
        self,
        content="test",
        tool_calls_data=None,
        finish_reason="stop",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    ):
        """Build a mock OpenAI-style response (as litellm returns)."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = content
        mock_response.choices[0].finish_reason = finish_reason

        if tool_calls_data:
            mock_tcs = []
            for tc_data in tool_calls_data:
                mock_tc = MagicMock()
                mock_tc.id = tc_data["id"]
                mock_tc.function.name = tc_data["name"]
                mock_tc.function.arguments = json.dumps(tc_data["arguments"])
                mock_tcs.append(mock_tc)
            mock_response.choices[0].message.tool_calls = mock_tcs
        else:
            mock_response.choices[0].message.tool_calls = None

        mock_response.usage = MagicMock(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        return mock_response

    # ---- _parse_response ----

    def test_parse_response_text_only(self):
        """_parse_response extracts content, usage, and finish_reason."""
        mock_resp = self._make_mock_response(
            content="The answer is 42.",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        result = self.backend._parse_response(mock_resp)

        assert isinstance(result, CompletionResponse)
        assert result.content == "The answer is 42."
        assert result.tool_calls == []
        assert result.finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15

    def test_parse_response_with_tool_calls(self):
        """_parse_response extracts tool calls correctly."""
        mock_resp = self._make_mock_response(
            content=None,
            tool_calls_data=[
                {"id": "call_lit_1", "name": "bash", "arguments": {"cmd": "pwd"}},
            ],
            finish_reason="tool_calls",
        )
        result = self.backend._parse_response(mock_resp)

        assert result.content is None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_lit_1"
        assert result.tool_calls[0].name == "bash"
        assert result.tool_calls[0].arguments == {"cmd": "pwd"}
        assert result.finish_reason == "tool_calls"

    def test_parse_response_function_call_finish_reason(self):
        """Legacy 'function_call' finish_reason is normalized to 'tool_calls'."""
        mock_resp = self._make_mock_response(finish_reason="function_call")
        result = self.backend._parse_response(mock_resp)
        assert result.finish_reason == "tool_calls"

    def test_parse_response_none_finish_reason(self):
        """None finish_reason defaults to 'stop'."""
        mock_resp = self._make_mock_response()
        mock_resp.choices[0].finish_reason = None
        result = self.backend._parse_response(mock_resp)
        assert result.finish_reason == "stop"

    def test_parse_response_none_usage(self):
        """None usage produces zero-value Usage."""
        mock_resp = self._make_mock_response()
        mock_resp.usage = None
        result = self.backend._parse_response(mock_resp)
        assert result.usage.prompt_tokens == 0
        assert result.usage.completion_tokens == 0
        assert result.usage.total_tokens == 0

    # ---- chat_completion ----

    @pytest.mark.asyncio
    async def test_chat_completion_non_streaming(self):
        """Full non-streaming chat_completion through litellm.acompletion."""
        mock_resp = self._make_mock_response(content="Hello from litellm")
        self.mock_litellm.acompletion = AsyncMock(return_value=mock_resp)

        messages = [Message(role=Role.USER, content="Hi")]
        result = await self.backend.chat_completion(messages, stream=False)

        assert isinstance(result, CompletionResponse)
        assert result.content == "Hello from litellm"

        # Verify acompletion was called with correct kwargs
        call_kwargs = self.mock_litellm.acompletion.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["stream"] is False
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["api_base"] == "http://localhost:8080"

    @pytest.mark.asyncio
    async def test_chat_completion_with_tools(self):
        """chat_completion passes tool definitions when provided."""
        mock_resp = self._make_mock_response(content="Done")
        self.mock_litellm.acompletion = AsyncMock(return_value=mock_resp)

        tool_def = _make_tool_def("read_file")
        messages = [Message(role=Role.USER, content="Read")]
        await self.backend.chat_completion(
            messages, tools=[tool_def], tool_choice="auto"
        )

        call_kwargs = self.mock_litellm.acompletion.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tool_choice"] == "auto"
        # Verify tool is in OpenAI format
        assert call_kwargs["tools"][0]["type"] == "function"
        assert call_kwargs["tools"][0]["function"]["name"] == "read_file"

    @pytest.mark.asyncio
    async def test_chat_completion_streaming(self):
        """Streaming chat_completion returns an async iterator of StreamChunks."""

        async def _fake_stream():
            chunk1 = MagicMock()
            chunk1.choices = [MagicMock()]
            chunk1.choices[0].delta.content = "Hello"
            chunk1.choices[0].delta.tool_calls = None
            chunk1.choices[0].finish_reason = None
            chunk1.usage = None
            yield chunk1

            chunk2 = MagicMock()
            chunk2.choices = [MagicMock()]
            chunk2.choices[0].delta.content = " world"
            chunk2.choices[0].delta.tool_calls = None
            chunk2.choices[0].finish_reason = "stop"
            chunk2.usage = MagicMock(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            )
            yield chunk2

        self.mock_litellm.acompletion = AsyncMock(return_value=_fake_stream())

        messages = [Message(role=Role.USER, content="Stream")]
        result = await self.backend.chat_completion(messages, stream=True)

        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " world"
        assert chunks[1].finish_reason == "stop"
        assert chunks[1].usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_chat_completion_streaming_usage_only_final_chunk(self):
        """Final chunk with no choices but usage is yielded."""

        async def _fake_stream():
            chunk1 = MagicMock()
            chunk1.choices = [MagicMock()]
            chunk1.choices[0].delta.content = "Hi"
            chunk1.choices[0].delta.tool_calls = None
            chunk1.choices[0].finish_reason = None
            chunk1.usage = None
            yield chunk1

            # Usage-only chunk
            chunk2 = MagicMock()
            chunk2.choices = []
            chunk2.usage = MagicMock(
                prompt_tokens=20, completion_tokens=10, total_tokens=30
            )
            yield chunk2

        self.mock_litellm.acompletion = AsyncMock(return_value=_fake_stream())

        messages = [Message(role=Role.USER, content="Test")]
        result = await self.backend.chat_completion(messages, stream=True)

        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[1].usage.prompt_tokens == 20

    @pytest.mark.asyncio
    async def test_chat_completion_streaming_tool_deltas(self):
        """Streaming with tool call deltas accumulates correctly."""

        async def _fake_stream():
            # First chunk: start of tool call
            tc_delta1 = MagicMock()
            tc_delta1.index = 0
            tc_delta1.id = "call_s1"
            tc_delta1.function.name = "read_file"
            tc_delta1.function.arguments = '{"pat'

            chunk1 = MagicMock()
            chunk1.choices = [MagicMock()]
            chunk1.choices[0].delta.content = None
            chunk1.choices[0].delta.tool_calls = [tc_delta1]
            chunk1.choices[0].finish_reason = None
            chunk1.usage = None
            yield chunk1

            # Second chunk: continuation of tool call args
            tc_delta2 = MagicMock()
            tc_delta2.index = 0
            tc_delta2.id = None
            tc_delta2.function.name = None
            tc_delta2.function.arguments = 'h": "/a"}'

            chunk2 = MagicMock()
            chunk2.choices = [MagicMock()]
            chunk2.choices[0].delta.content = None
            chunk2.choices[0].delta.tool_calls = [tc_delta2]
            chunk2.choices[0].finish_reason = "tool_calls"
            chunk2.usage = None
            yield chunk2

        self.mock_litellm.acompletion = AsyncMock(return_value=_fake_stream())

        messages = [Message(role=Role.USER, content="Test")]
        result = await self.backend.chat_completion(messages, stream=True)

        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].tool_call_delta is not None
        assert chunks[0].tool_call_delta["id"] == "call_s1"
        assert chunks[0].tool_call_delta["name"] == "read_file"
        assert chunks[1].finish_reason == "tool_calls"

    # ---- format_tool_definitions ----

    def test_format_tool_definitions_uses_to_openai(self):
        """format_tool_definitions delegates to tool.to_openai()."""
        tool = _make_tool_def("bash")
        result = self.backend.format_tool_definitions([tool])
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "bash"

    def test_format_tool_definitions_multiple_tools(self):
        """Multiple tools are all converted."""
        tools = [_make_tool_def("bash"), _make_tool_def("grep")]
        result = self.backend.format_tool_definitions(tools)
        assert len(result) == 2
        names = [r["function"]["name"] for r in result]
        assert "bash" in names
        assert "grep" in names

    # ---- parse_tool_calls edge cases ----

    def test_parse_tool_calls_malformed_json_string(self):
        """Malformed JSON string arguments produce empty dict."""
        mock_tc = MagicMock()
        mock_tc.id = "tc_bad"
        mock_tc.function.name = "broken"
        mock_tc.function.arguments = "{not valid"

        mock_message = MagicMock()
        mock_message.tool_calls = [mock_tc]
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        result = self.backend.parse_tool_calls(mock_response)
        assert len(result) == 1
        assert result[0].arguments == {}

    def test_parse_tool_calls_non_string_non_dict_arguments(self):
        """Arguments that are neither string nor dict produce empty dict."""
        mock_tc = MagicMock()
        mock_tc.id = "tc_weird"
        mock_tc.function.name = "weird"
        mock_tc.function.arguments = 12345  # int, not str or dict

        mock_message = MagicMock()
        mock_message.tool_calls = [mock_tc]
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        result = self.backend.parse_tool_calls(mock_response)
        assert len(result) == 1
        assert result[0].arguments == {}

    def test_parse_tool_calls_no_tool_calls_attr(self):
        """Response with no tool_calls attribute on message returns empty list."""
        mock_message = MagicMock()
        mock_message.tool_calls = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        result = self.backend.parse_tool_calls(mock_response)
        assert result == []

    # ---- _message_to_dict edge cases ----

    def test_message_to_dict_tool_message(self):
        """Tool messages include tool_call_id and name."""
        msg = Message(
            role=Role.TOOL,
            content="result here",
            tool_call_id="tc_xyz",
            name="my_tool",
        )
        d = self._message_to_dict(msg)
        assert d["role"] == "tool"
        assert d["content"] == "result here"
        assert d["tool_call_id"] == "tc_xyz"
        assert d["name"] == "my_tool"

    def test_message_to_dict_with_tool_calls_serializes_json(self):
        """Tool calls arguments are serialized to JSON string."""
        tc = ToolCall(id="tc_1", name="bash", arguments={"cmd": "echo hello"})
        msg = Message(role=Role.ASSISTANT, content="Running", tool_calls=[tc])
        d = self._message_to_dict(msg)
        assert d["tool_calls"][0]["function"]["arguments"] == json.dumps(
            {"cmd": "echo hello"}, ensure_ascii=False
        )

    # ---- _ensure_litellm ----

    def test_ensure_litellm_returns_cached_module(self):
        """After first call, _ensure_litellm returns the cached module."""
        result = self.backend._ensure_litellm()
        assert result is self.mock_litellm

    # ---- _parse_usage edge cases ----

    def test_parse_usage_dict_missing_keys(self):
        """Dict with missing keys defaults to 0."""
        result = self.BackendCls._parse_usage({"prompt_tokens": 5})
        assert result.prompt_tokens == 5
        assert result.completion_tokens == 0
        assert result.total_tokens == 0

    def test_parse_usage_object_with_none_values(self):
        """Object with None values defaults to 0."""
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = None
        mock_usage.completion_tokens = None
        mock_usage.total_tokens = None
        result = self.BackendCls._parse_usage(mock_usage)
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.total_tokens == 0
