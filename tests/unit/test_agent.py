"""Unit tests for the Tachyon Agent layer: AgentEvent, ContextManager, run_agent_loop,
_collect_stream, and Persona.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tachyon.agent.context import (
    COMPACT_AFTER_TURN,
    ContextManager,
)
from tachyon.agent.loop import (
    MAX_TURNS,
    AgentEvent,
    AgentUsage,
    _collect_stream,
    _update_usage,
    _usage_dict,
    run_agent_loop,
)
from tachyon.agent.persona import (
    AGENT_IDENTITY,
    build_kernel_context,
    build_system_prompt,
)
from tachyon.errors.handler import ErrorCode, ToolResult
from tachyon.llm.backend import (
    CompletionResponse,
    Message,
    Role,
    StreamChunk,
    ToolCall,
    Usage,
)
from tachyon.tools.registry import ToolDefinition, ToolRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(role: Role, content: str, **kwargs: Any) -> Message:
    """Shortcut for creating a Message."""
    return Message(role=role, content=content, **kwargs)


def _tool_msg(content: str, tool_call_id: str = "tc_1", name: str = "my_tool") -> Message:
    """Shortcut for creating a TOOL Message."""
    return Message(role=Role.TOOL, content=content, tool_call_id=tool_call_id, name=name)


def _assistant_with_tools(tool_calls: list[ToolCall], content: str | None = None) -> Message:
    """Shortcut for an ASSISTANT message that contains tool_calls."""
    return Message(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)


def _make_registry(*tools: ToolDefinition) -> ToolRegistry:
    """Create a ToolRegistry pre-loaded with the given ToolDefinitions."""
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


def _dummy_tool_def(name: str = "list_kernels", desc: str = "List all kernels.") -> ToolDefinition:
    """Create a minimal ToolDefinition for testing."""
    return ToolDefinition(
        name=name,
        description=desc,
        parameters={
            "type": "object",
            "properties": {"kernel_id": {"type": "integer"}},
        },
    )


async def _async_gen(*items):
    """Create an async generator from items."""
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# TestAgentEvent
# ---------------------------------------------------------------------------

class TestAgentEvent:
    def test_text_event(self):
        ev = AgentEvent(type="text", content="Hello")
        assert ev.type == "text"
        assert ev.content == "Hello"
        assert ev.data is None

    def test_tool_call_event(self):
        ev = AgentEvent(type="tool_call", content="Calling list_kernels...", data={"name": "list_kernels"})
        assert ev.type == "tool_call"
        assert ev.data["name"] == "list_kernels"

    def test_tool_result_event(self):
        ev = AgentEvent(type="tool_result", content="list_kernels: OK", data={"success": True})
        assert ev.type == "tool_result"
        assert ev.data["success"] is True

    def test_system_event(self):
        ev = AgentEvent(type="system", content="Context compressed")
        assert ev.type == "system"
        assert ev.content == "Context compressed"

    def test_done_event(self):
        ev = AgentEvent(type="done", data={"turns": 3, "tool_calls": 2})
        assert ev.type == "done"
        assert ev.data["turns"] == 3

    def test_default_none_fields(self):
        ev = AgentEvent(type="done")
        assert ev.content is None
        assert ev.data is None

    def test_all_fields_populated(self):
        ev = AgentEvent(type="text", content="hi", data={"key": "value"})
        assert ev.content == "hi"
        assert ev.data == {"key": "value"}


# ---------------------------------------------------------------------------
# TestContextManager
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_estimate_tokens_basic(self):
        ctx = ContextManager()
        msgs = [
            _msg(Role.SYSTEM, "a" * 300),   # 300/3 = 100
            _msg(Role.USER, "b" * 400),      # 400/4 = 100
        ]
        tokens = ctx.estimate_tokens(msgs)
        assert tokens == 200

    def test_estimate_tokens_with_tool_calls(self):
        ctx = ContextManager()
        tc = ToolCall(id="tc1", name="foo", arguments={"key": "val"})
        msgs = [
            Message(role=Role.ASSISTANT, content="thinking", tool_calls=[tc]),
        ]
        tokens = ctx.estimate_tokens(msgs)
        # "thinking" = 8 chars / 4 = 2  +  json.dumps({"key": "val"}) / 4
        args_tokens = len(json.dumps({"key": "val"})) // 4
        assert tokens == 2 + args_tokens

    def test_estimate_tokens_empty_list(self):
        ctx = ContextManager()
        assert ctx.estimate_tokens([]) == 0

    def test_compact_tool_results_old_shortened(self):
        """Old tool results beyond keep_last_n with long content get compacted."""
        ctx = ContextManager()
        long_content = "First line of output\n" + "x" * 500
        messages = [
            _msg(Role.SYSTEM, "sys"),
            _assistant_with_tools([ToolCall(id="tc1", name="t1", arguments={})]),
            _tool_msg(long_content, tool_call_id="tc1", name="tool_alpha"),
            _assistant_with_tools([ToolCall(id="tc2", name="t2", arguments={})]),
            _tool_msg("recent result", tool_call_id="tc2", name="tool_beta"),
        ]
        ctx.compact_tool_results(messages, keep_last_n=1)
        # First tool msg (index 2) should be compacted
        assert messages[2].content.startswith("[tool_alpha:")
        assert messages[2].content.endswith("...]")
        # Last tool msg (index 4) should remain unchanged
        assert messages[4].content == "recent result"

    def test_compact_tool_results_fewer_than_keep_last_n(self):
        """When there are fewer tool results than keep_last_n, nothing changes."""
        ctx = ContextManager()
        messages = [
            _msg(Role.SYSTEM, "sys"),
            _tool_msg("only one tool result", tool_call_id="tc1", name="t1"),
        ]
        original_content = messages[1].content
        ctx.compact_tool_results(messages, keep_last_n=2)
        assert messages[1].content == original_content

    def test_compact_tool_results_short_content_preserved(self):
        """Tool results shorter than threshold are not compacted."""
        ctx = ContextManager()
        short = "ok"
        messages = [
            _msg(Role.SYSTEM, "sys"),
            _tool_msg(short, tool_call_id="tc1", name="t1"),
            _tool_msg("second", tool_call_id="tc2", name="t2"),
        ]
        ctx.compact_tool_results(messages, keep_last_n=1)
        # First tool result is short — should not be compacted
        assert messages[1].content == short

    def test_distill_collapses_tool_pairs(self):
        """distill() should collapse assistant(tool_calls) + tool results into summary."""
        ctx = ContextManager()
        tc = ToolCall(id="tc1", name="get_metrics", arguments={"id": 1})
        messages = [
            _msg(Role.SYSTEM, "sys"),
            _msg(Role.USER, "analyze kernel 0"),
            _assistant_with_tools([tc], content="Let me check"),
            _tool_msg('{"result": "ok"}', tool_call_id="tc1", name="get_metrics"),
            _msg(Role.ASSISTANT, content="Here is the result"),
        ]
        result = ctx.distill(messages)
        assert len(result) == 4  # system, user, summary, final assistant
        assert result[0].role == Role.SYSTEM
        assert result[1].role == Role.USER
        assert "[Agent called: get_metrics]" in result[2].content
        assert result[3].content == "Here is the result"

    def test_distill_preserves_system_and_user(self):
        """distill() must not drop system or user messages."""
        ctx = ContextManager()
        messages = [
            _msg(Role.SYSTEM, "system prompt"),
            _msg(Role.USER, "user question"),
            _msg(Role.ASSISTANT, "plain answer"),
        ]
        result = ctx.distill(messages)
        assert len(result) == 3
        assert result[0].content == "system prompt"
        assert result[1].content == "user question"
        assert result[2].content == "plain answer"

    def test_distill_multiple_tool_calls(self):
        """distill() collapses multiple tools in a single assistant message."""
        ctx = ContextManager()
        tcs = [
            ToolCall(id="tc1", name="tool_a", arguments={}),
            ToolCall(id="tc2", name="tool_b", arguments={}),
        ]
        messages = [
            _msg(Role.SYSTEM, "sys"),
            _assistant_with_tools(tcs),
            _tool_msg("result a", tool_call_id="tc1", name="tool_a"),
            _tool_msg("result b", tool_call_id="tc2", name="tool_b"),
        ]
        result = ctx.distill(messages)
        assert len(result) == 2  # system + collapsed summary
        assert "tool_a" in result[1].content
        assert "tool_b" in result[1].content

    def test_should_compact_below_threshold(self):
        ctx = ContextManager()
        for turn in range(COMPACT_AFTER_TURN):
            assert ctx.should_compact(turn) is False

    def test_should_compact_at_threshold(self):
        ctx = ContextManager()
        assert ctx.should_compact(COMPACT_AFTER_TURN) is True
        assert ctx.should_compact(COMPACT_AFTER_TURN + 5) is True

    def test_is_near_budget_below(self):
        ctx = ContextManager(budget=1000)
        msgs = [_msg(Role.USER, "x" * 100)]  # ~25 tokens, well below 850
        assert ctx.is_near_budget(msgs) is False

    def test_is_near_budget_above(self):
        ctx = ContextManager(budget=100)
        # 85% of 100 = 85 tokens. 400 chars / 4 = 100 tokens > 85
        msgs = [_msg(Role.USER, "x" * 400)]
        assert ctx.is_near_budget(msgs) is True

    def test_is_near_budget_exactly_at_boundary(self):
        # budget=100, threshold=85. Need estimated > 85.
        # Role.USER: len/4.  340/4=85, not > 85. 344/4=86 > 85.
        ctx = ContextManager(budget=100)
        msgs = [_msg(Role.USER, "x" * 344)]
        assert ctx.is_near_budget(msgs) is True


# ---------------------------------------------------------------------------
# TestCollectStream
# ---------------------------------------------------------------------------

class TestCollectStream:
    @pytest.mark.asyncio
    async def test_text_only_chunks(self):
        chunks = _async_gen(
            StreamChunk(content="Hello "),
            StreamChunk(content="world"),
            StreamChunk(finish_reason="stop"),
        )
        result = await _collect_stream(chunks)
        assert result["content"] == "Hello world"
        assert result["tool_calls"] == []
        assert result["usage"] == {}

    @pytest.mark.asyncio
    async def test_tool_call_delta_chunks(self):
        chunks = _async_gen(
            StreamChunk(tool_call_delta={"id": "tc_1", "name": "list_kernels", "arguments": '{"ker'}),
            StreamChunk(tool_call_delta={"arguments": 'nel_id": 0}'}),
            StreamChunk(finish_reason="tool_calls"),
        )
        result = await _collect_stream(chunks)
        assert result["content"] is None
        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc.id == "tc_1"
        assert tc.name == "list_kernels"
        assert tc.arguments == {"kernel_id": 0}

    @pytest.mark.asyncio
    async def test_usage_in_final_chunk(self):
        chunks = _async_gen(
            StreamChunk(content="Answer"),
            StreamChunk(finish_reason="stop", usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)),
        )
        result = await _collect_stream(chunks)
        assert result["usage"]["prompt_tokens"] == 100
        assert result["usage"]["completion_tokens"] == 50

    @pytest.mark.asyncio
    async def test_mixed_text_and_tool_calls(self):
        chunks = _async_gen(
            StreamChunk(content="Let me check. "),
            StreamChunk(tool_call_delta={"id": "tc_1", "name": "get_summary", "arguments": "{}"}),
            StreamChunk(finish_reason="tool_calls"),
        )
        result = await _collect_stream(chunks)
        assert result["content"] == "Let me check. "
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0].name == "get_summary"

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        chunks = _async_gen()
        result = await _collect_stream(chunks)
        assert result["content"] is None
        assert result["tool_calls"] == []

    @pytest.mark.asyncio
    async def test_malformed_arguments_json(self):
        """Invalid JSON in tool arguments should produce empty dict."""
        chunks = _async_gen(
            StreamChunk(tool_call_delta={"id": "tc_1", "name": "foo", "arguments": "{bad json"}),
            StreamChunk(finish_reason="tool_calls"),
        )
        result = await _collect_stream(chunks)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0].arguments == {}


# ---------------------------------------------------------------------------
# TestAgentLoop
# ---------------------------------------------------------------------------

async def _collect_events(ait) -> list[AgentEvent]:
    """Collect all events from an async iterator."""
    events = []
    async for ev in ait:
        events.append(ev)
    return events


def _mock_backend() -> MagicMock:
    """Create a MagicMock LLMBackend with AsyncMock chat_completion."""
    backend = MagicMock()
    backend.chat_completion = AsyncMock()
    return backend


def _mock_registry_with_tool() -> MagicMock:
    """Create a MagicMock ToolRegistry with execute and all_definitions."""
    registry = MagicMock()
    registry.all_definitions.return_value = [_dummy_tool_def()]
    registry.execute = AsyncMock(
        return_value=ToolResult.ok({"status": "done"}),
    )
    return registry


class TestAgentLoop:
    @pytest.mark.asyncio
    async def test_text_only_response(self):
        """LLM returns text without tool calls -> yields text + done."""
        backend = _mock_backend()
        backend.chat_completion.return_value = CompletionResponse(
            content="The kernel is memory-bound.",
            tool_calls=[],
            usage=Usage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
        )
        registry = _mock_registry_with_tool()

        events = await _collect_events(
            run_agent_loop(backend, registry, "analyze kernel 0", "You are Tachyon")
        )

        types = [e.type for e in events]
        assert "text" in types
        assert types[-1] == "done"
        text_ev = next(e for e in events if e.type == "text")
        assert text_ev.content == "The kernel is memory-bound."

    @pytest.mark.asyncio
    async def test_tool_call_then_text(self):
        """LLM returns tool_calls first, then text on second call."""
        backend = _mock_backend()
        tc = ToolCall(id="tc_1", name="list_kernels", arguments={"kernel_id": 0})
        # First call: tool call
        resp1 = CompletionResponse(
            content="Let me check.",
            tool_calls=[tc],
            usage=Usage(prompt_tokens=40, completion_tokens=10, total_tokens=50),
            finish_reason="tool_calls",
        )
        # Second call: text only
        resp2 = CompletionResponse(
            content="Kernel 0 is compute-bound.",
            tool_calls=[],
            usage=Usage(prompt_tokens=80, completion_tokens=30, total_tokens=110),
        )
        backend.chat_completion.side_effect = [resp1, resp2]

        registry = _mock_registry_with_tool()

        events = await _collect_events(
            run_agent_loop(backend, registry, "analyze", "sys")
        )

        types = [e.type for e in events]
        assert "tool_call" in types
        assert "tool_result" in types
        assert "text" in types
        assert types[-1] == "done"

        # Verify tool was executed
        registry.execute.assert_called_once_with("list_kernels", {"kernel_id": 0})

    @pytest.mark.asyncio
    async def test_llm_error_yields_system_error_and_done(self):
        """When LLM raises an exception, yields system error + done."""
        backend = _mock_backend()
        backend.chat_completion.side_effect = RuntimeError("API timeout")
        registry = _mock_registry_with_tool()

        events = await _collect_events(
            run_agent_loop(backend, registry, "go", "sys")
        )

        types = [e.type for e in events]
        assert "system" in types
        assert types[-1] == "done"
        sys_ev = next(e for e in events if e.type == "system")
        assert "LLM error" in sys_ev.content
        assert "API timeout" in sys_ev.content

    @pytest.mark.asyncio
    async def test_max_turns_enforcement(self):
        """Agent should stop after MAX_TURNS, even if LLM keeps returning tool calls."""
        backend = _mock_backend()
        tc = ToolCall(id="tc_1", name="list_kernels", arguments={})

        # Create responses: first MAX_TURNS-1 return tool_calls, last is forced synthesis
        responses = []
        for i in range(MAX_TURNS - 1):
            responses.append(CompletionResponse(
                content=None,
                tool_calls=[ToolCall(id=f"tc_{i}", name="list_kernels", arguments={})],
                usage=Usage(prompt_tokens=10, completion_tokens=5),
                finish_reason="tool_calls",
            ))
        # Final turn: forced tool_choice="none", so LLM returns text
        responses.append(CompletionResponse(
            content="Final synthesis.",
            tool_calls=[],
            usage=Usage(prompt_tokens=10, completion_tokens=5),
        ))
        backend.chat_completion.side_effect = responses

        registry = _mock_registry_with_tool()

        events = await _collect_events(
            run_agent_loop(backend, registry, "analyze", "sys")
        )

        done_ev = next(e for e in events if e.type == "done")
        assert done_ev.data["turns"] == MAX_TURNS

        # Verify final call used tool_choice="none" and tools=None
        final_call = backend.chat_completion.call_args_list[-1]
        assert final_call.kwargs.get("tool_choice") == "none" or final_call[1].get("tool_choice") == "none"

    @pytest.mark.asyncio
    async def test_context_compression_triggered(self):
        """When context is near budget, compression events should be yielded."""
        backend = _mock_backend()
        backend.chat_completion.return_value = CompletionResponse(
            content="Answer",
            tool_calls=[],
            usage=Usage(prompt_tokens=10, completion_tokens=5),
        )
        registry = _mock_registry_with_tool()

        # Use a very small budget so is_near_budget triggers immediately
        events = await _collect_events(
            run_agent_loop(
                backend, registry, "analyze " + "x" * 2000, "sys " * 500,
                context_budget=100,
            )
        )

        types = [e.type for e in events]
        assert "system" in types
        sys_events = [e for e in events if e.type == "system"]
        compression_evts = [e for e in sys_events if "Compressing" in (e.content or "")]
        assert len(compression_evts) >= 1

    @pytest.mark.asyncio
    async def test_history_prepended(self):
        """History messages should be included between system and user."""
        backend = _mock_backend()
        backend.chat_completion.return_value = CompletionResponse(
            content="Got it.",
            tool_calls=[],
            usage=Usage(prompt_tokens=10, completion_tokens=5),
        )
        registry = _mock_registry_with_tool()

        history = [
            _msg(Role.USER, "previous question"),
            _msg(Role.ASSISTANT, "previous answer"),
        ]

        events = await _collect_events(
            run_agent_loop(backend, registry, "new question", "sys", history=history)
        )

        # Verify messages passed to LLM include history
        call_args = backend.chat_completion.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages") or call_args[0][0]
        # messages: [system, prev_user, prev_assistant, new_user]
        assert len(messages) >= 4
        assert messages[0].role == Role.SYSTEM
        assert messages[1].content == "previous question"
        assert messages[2].content == "previous answer"
        assert messages[3].content == "new question"

    @pytest.mark.asyncio
    async def test_tool_execution_error_reported(self):
        """When tool execution fails, the error is reported properly."""
        backend = _mock_backend()
        tc = ToolCall(id="tc_1", name="list_kernels", arguments={})
        resp1 = CompletionResponse(
            content=None,
            tool_calls=[tc],
            usage=Usage(prompt_tokens=10, completion_tokens=5),
            finish_reason="tool_calls",
        )
        resp2 = CompletionResponse(
            content="I see an error occurred.",
            tool_calls=[],
            usage=Usage(prompt_tokens=20, completion_tokens=10),
        )
        backend.chat_completion.side_effect = [resp1, resp2]

        registry = _mock_registry_with_tool()
        registry.execute.return_value = ToolResult.fail(
            ErrorCode.UNKNOWN, "Tool crashed", "Try again"
        )

        events = await _collect_events(
            run_agent_loop(backend, registry, "go", "sys")
        )

        result_events = [e for e in events if e.type == "tool_result"]
        assert len(result_events) == 1
        assert result_events[0].data["success"] is False
        assert "ERROR" in result_events[0].content

    @pytest.mark.asyncio
    async def test_usage_accumulation(self):
        """Token usage should accumulate across multiple turns."""
        backend = _mock_backend()
        tc = ToolCall(id="tc_1", name="list_kernels", arguments={})
        resp1 = CompletionResponse(
            content=None,
            tool_calls=[tc],
            usage=Usage(prompt_tokens=100, completion_tokens=50),
            finish_reason="tool_calls",
        )
        resp2 = CompletionResponse(
            content="Done.",
            tool_calls=[],
            usage=Usage(prompt_tokens=200, completion_tokens=80),
        )
        backend.chat_completion.side_effect = [resp1, resp2]
        registry = _mock_registry_with_tool()

        events = await _collect_events(
            run_agent_loop(backend, registry, "go", "sys")
        )

        done_ev = next(e for e in events if e.type == "done")
        assert done_ev.data["prompt_tokens"] == 300
        assert done_ev.data["completion_tokens"] == 130
        assert done_ev.data["total_tokens"] == 430
        assert done_ev.data["tool_calls"] == 1
        assert done_ev.data["turns"] == 2

    @pytest.mark.asyncio
    async def test_no_content_no_text_event(self):
        """If LLM returns no content and no tool calls, no text event is yielded."""
        backend = _mock_backend()
        backend.chat_completion.return_value = CompletionResponse(
            content=None,
            tool_calls=[],
            usage=Usage(prompt_tokens=10, completion_tokens=0),
        )
        registry = _mock_registry_with_tool()

        events = await _collect_events(
            run_agent_loop(backend, registry, "go", "sys")
        )

        text_events = [e for e in events if e.type == "text"]
        assert len(text_events) == 0
        assert events[-1].type == "done"


# ---------------------------------------------------------------------------
# TestUpdateUsage and TestUsageDict (helpers)
# ---------------------------------------------------------------------------

class TestUpdateUsage:
    def test_accumulates(self):
        usage = AgentUsage()
        _update_usage(usage, {"prompt_tokens": 10, "completion_tokens": 5})
        _update_usage(usage, {"prompt_tokens": 20, "completion_tokens": 15})
        assert usage.prompt_tokens == 30
        assert usage.completion_tokens == 20

    def test_missing_keys(self):
        usage = AgentUsage()
        _update_usage(usage, {})
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0


class TestUsageDict:
    def test_conversion(self):
        usage = AgentUsage(prompt_tokens=100, completion_tokens=50, turns=3, tool_calls=2)
        d = _usage_dict(usage)
        assert d["prompt_tokens"] == 100
        assert d["completion_tokens"] == 50
        assert d["total_tokens"] == 150
        assert d["turns"] == 3
        assert d["tool_calls"] == 2


# ---------------------------------------------------------------------------
# TestPersona
# ---------------------------------------------------------------------------

class TestPersona:
    def test_agent_identity_has_expected_fields(self):
        assert "name" in AGENT_IDENTITY
        assert AGENT_IDENTITY["name"] == "Tachyon"
        assert "role" in AGENT_IDENTITY
        assert "expertise" in AGENT_IDENTITY
        assert isinstance(AGENT_IDENTITY["expertise"], list)
        assert len(AGENT_IDENTITY["expertise"]) > 0

    def test_agent_identity_expertise_contains_cuda(self):
        expertise_str = " ".join(AGENT_IDENTITY["expertise"]).lower()
        assert "cuda" in expertise_str

    def test_build_system_prompt_loads_template(self):
        """build_system_prompt should produce a string containing persona.md content."""
        registry = _make_registry(_dummy_tool_def("list_kernels", "List all kernels in the report."))
        prompt = build_system_prompt(registry)
        assert "Tachyon" in prompt
        assert "CUDA" in prompt

    def test_build_system_prompt_substitutes_tool_catalog(self):
        """Tool catalog should appear in the generated prompt."""
        tool = _dummy_tool_def("get_metrics", "Get kernel metrics. Very detailed.")
        registry = _make_registry(tool)
        prompt = build_system_prompt(registry)
        assert "get_metrics" in prompt
        assert "kernel_id: integer" in prompt

    def test_build_system_prompt_substitutes_kernel_list(self):
        """kernel_context should appear as kernel_list in template."""
        registry = _make_registry(_dummy_tool_def())
        kernel_ctx = "  [0] my_kernel — grid=(128,1,1), block=(256,1,1)"
        prompt = build_system_prompt(registry, kernel_context=kernel_ctx)
        assert "my_kernel" in prompt
        assert "grid=(128,1,1)" in prompt

    def test_build_system_prompt_no_kernel_context(self):
        """When no kernel context provided, default placeholder is used."""
        registry = _make_registry(_dummy_tool_def())
        prompt = build_system_prompt(registry, kernel_context=None)
        assert "(no report loaded yet)" in prompt

    def test_build_system_prompt_multiple_tools(self):
        """Multiple tools should all appear in the catalog."""
        t1 = _dummy_tool_def("tool_a", "Do alpha.")
        t2 = _dummy_tool_def("tool_b", "Do beta.")
        registry = _make_registry(t1, t2)
        prompt = build_system_prompt(registry)
        assert "tool_a" in prompt
        assert "tool_b" in prompt

    def test_build_kernel_context_with_kernels(self):
        """Formats a list of kernel-like objects."""
        @dataclass
        class FakeLaunch:
            grid: tuple = (128, 1, 1)
            block: tuple = (256, 1, 1)

        @dataclass
        class FakeKernel:
            demangled_name: str = "matmul_kernel"
            launch_params: FakeLaunch = None

            def __post_init__(self):
                if self.launch_params is None:
                    self.launch_params = FakeLaunch()

        kernels = [
            FakeKernel(demangled_name="matmul_kernel"),
            FakeKernel(demangled_name="reduce_kernel"),
        ]
        result = build_kernel_context(kernels)
        assert "[0] matmul_kernel" in result
        assert "[1] reduce_kernel" in result
        assert "grid=(128, 1, 1)" in result
        assert "block=(256, 1, 1)" in result

    def test_build_kernel_context_empty_list(self):
        result = build_kernel_context([])
        assert "empty report" in result.lower() or "no kernels" in result.lower()

    def test_build_kernel_context_fallback_name(self):
        """Falls back to kernel_name if demangled_name is None."""
        @dataclass
        class FakeLaunch:
            grid: tuple = (1, 1, 1)
            block: tuple = (1, 1, 1)

        @dataclass
        class FakeKernel:
            demangled_name: str | None = None
            kernel_name: str = "fallback_name"
            launch_params: FakeLaunch = None

            def __post_init__(self):
                if self.launch_params is None:
                    self.launch_params = FakeLaunch()

        kernels = [FakeKernel()]
        result = build_kernel_context(kernels)
        assert "fallback_name" in result
