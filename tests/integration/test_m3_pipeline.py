"""Integration tests — M3 Agent pipeline (mock LLM).

Tests the full Agent flow: SessionContext → ToolRegistry → AgentLoop → output.
No real LLM calls; uses mock backend that simulates tool-calling patterns.
"""

import pytest

import tachyon.i18n as i18n
from tachyon.agent.loop import run_agent_loop
from tachyon.agent.persona import build_kernel_context, build_system_prompt
from tachyon.llm.backend import (
    CompletionResponse,
    LLMBackend,
    ToolCall,
    Usage,
)
from tachyon.models.kernel import KernelReport
from tachyon.tools.analysis import register_analysis_tools
from tachyon.tools.context import SessionContext
from tachyon.tools.data_query import register_data_query_tools
from tachyon.tools.registry import ToolRegistry
from tachyon.tools.source import register_source_tools


@pytest.fixture(autouse=True)
def init_i18n():
    i18n._packs.clear()
    i18n._current_lang = "en"
    i18n.init("en")
    yield
    i18n._packs.clear()


def _build_session(kernels: list[KernelReport]) -> SessionContext:
    """Create a SessionContext with analyzers registered."""
    from tachyon.analyzers.base import AnalyzerRegistry
    reg = AnalyzerRegistry()
    reg.auto_register()
    return SessionContext(
        kernels=kernels,
        action=None,
        correlator=None,
        registry=reg,
    )


def _build_tool_registry(session: SessionContext) -> ToolRegistry:
    """Create a ToolRegistry with all 9 tools registered."""
    registry = ToolRegistry()
    register_data_query_tools(registry, session)
    register_source_tools(registry, session)
    register_analysis_tools(registry, session)
    return registry


class _MockBackend(LLMBackend):
    """Mock LLM backend that replays a scripted sequence of responses."""

    def __init__(self, responses: list[CompletionResponse]):
        super().__init__(model="mock-model")
        self._responses = list(responses)
        self._call_count = 0

    async def chat_completion(self, messages, tools=None, tool_choice="auto",
                               stream=False, max_tokens=4096, temperature=0.1):
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]

    def format_tool_definitions(self, tools):
        return [t.to_openai() for t in tools]

    def parse_tool_calls(self, response):
        return []


class TestM3Pipeline:

    async def test_text_only_response(self, report_compute_bound: KernelReport):
        """Agent returns text without calling any tools."""
        session = _build_session([report_compute_bound])
        registry = _build_tool_registry(session)

        backend = _MockBackend([
            CompletionResponse(
                content="This kernel is compute-bound with 85% SM throughput.",
                tool_calls=[],
                usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            ),
        ])

        prompt = build_system_prompt(registry, build_kernel_context([report_compute_bound]))

        events = []
        async for event in run_agent_loop(
            backend=backend, registry=registry,
            user_message="Analyze this kernel",
            system_prompt=prompt, stream=False,
        ):
            events.append(event)

        # Should have text + done events
        types = [e.type for e in events]
        assert "text" in types
        assert "done" in types
        text_content = "".join(e.content for e in events if e.type == "text")
        assert "compute-bound" in text_content

    async def test_tool_call_then_text(self, report_compute_bound: KernelReport):
        """Agent calls a tool, then produces text answer."""
        session = _build_session([report_compute_bound])
        registry = _build_tool_registry(session)

        backend = _MockBackend([
            # Turn 1: call list_kernels
            CompletionResponse(
                content=None,
                tool_calls=[ToolCall(id="tc1", name="list_kernels", arguments={})],
                usage=Usage(prompt_tokens=200, completion_tokens=30, total_tokens=230),
            ),
            # Turn 2: text response
            CompletionResponse(
                content="Found 1 kernel. It is compute-bound.",
                tool_calls=[],
                usage=Usage(prompt_tokens=400, completion_tokens=60, total_tokens=460),
            ),
        ])

        prompt = build_system_prompt(registry, build_kernel_context([report_compute_bound]))

        events = []
        async for event in run_agent_loop(
            backend=backend, registry=registry,
            user_message="What kernels are in this report?",
            system_prompt=prompt, stream=False,
        ):
            events.append(event)

        types = [e.type for e in events]
        assert "tool_call" in types
        assert "tool_result" in types
        assert "text" in types
        assert "done" in types

        # Tool call should be list_kernels
        tc_event = next(e for e in events if e.type == "tool_call")
        assert tc_event.data["name"] == "list_kernels"

        # Tool result should be OK
        tr_event = next(e for e in events if e.type == "tool_result")
        assert tr_event.data["success"]

    async def test_run_analysis_tool(self, report_compute_bound: KernelReport):
        """Agent calls run_analysis tool and gets findings."""
        session = _build_session([report_compute_bound])
        registry = _build_tool_registry(session)

        backend = _MockBackend([
            CompletionResponse(
                content=None,
                tool_calls=[ToolCall(id="tc1", name="run_analysis",
                                      arguments={"kernel_id": 0})],
                usage=Usage(prompt_tokens=200, completion_tokens=30, total_tokens=230),
            ),
            CompletionResponse(
                content="Analysis complete. Kernel is compute-bound.",
                tool_calls=[],
                usage=Usage(prompt_tokens=600, completion_tokens=80, total_tokens=680),
            ),
        ])

        prompt = build_system_prompt(registry)
        events = []
        async for event in run_agent_loop(
            backend=backend, registry=registry,
            user_message="Run analysis on kernel 0",
            system_prompt=prompt, stream=False,
        ):
            events.append(event)

        tr_events = [e for e in events if e.type == "tool_result"]
        assert len(tr_events) == 1
        assert tr_events[0].data["success"]

    async def test_opt_tree_tool(self, report_compute_bound: KernelReport):
        """Agent calls get_optimization_tree."""
        session = _build_session([report_compute_bound])
        registry = _build_tool_registry(session)

        backend = _MockBackend([
            CompletionResponse(
                content=None,
                tool_calls=[ToolCall(id="tc1", name="get_optimization_tree",
                                      arguments={"kernel_id": 0})],
                usage=Usage(prompt_tokens=200, completion_tokens=30, total_tokens=230),
            ),
            CompletionResponse(
                content="OptTree shows compute as the primary bottleneck.",
                tool_calls=[],
                usage=Usage(prompt_tokens=500, completion_tokens=60, total_tokens=560),
            ),
        ])

        prompt = build_system_prompt(registry)
        events = []
        async for event in run_agent_loop(
            backend=backend, registry=registry,
            user_message="Show optimization tree",
            system_prompt=prompt, stream=False,
        ):
            events.append(event)

        tr_events = [e for e in events if e.type == "tool_result"]
        assert len(tr_events) == 1
        assert tr_events[0].data["success"]

    async def test_multi_kernel_pipeline(
        self,
        report_compute_bound: KernelReport,
        report_memory_bound: KernelReport,
    ):
        """Pipeline with multiple kernels."""
        session = _build_session([report_compute_bound, report_memory_bound])
        registry = _build_tool_registry(session)

        backend = _MockBackend([
            # list_kernels
            CompletionResponse(
                content=None,
                tool_calls=[ToolCall(id="tc1", name="list_kernels", arguments={})],
                usage=Usage(prompt_tokens=200, completion_tokens=20, total_tokens=220),
            ),
            # get_kernel_summary for kernel 1
            CompletionResponse(
                content=None,
                tool_calls=[ToolCall(id="tc2", name="get_kernel_summary",
                                      arguments={"kernel_id": 1})],
                usage=Usage(prompt_tokens=400, completion_tokens=30, total_tokens=430),
            ),
            # Final text
            CompletionResponse(
                content="Kernel 0 is compute-bound, kernel 1 is memory-bound.",
                tool_calls=[],
                usage=Usage(prompt_tokens=600, completion_tokens=70, total_tokens=670),
            ),
        ])

        prompt = build_system_prompt(registry, build_kernel_context(
            [report_compute_bound, report_memory_bound]
        ))
        events = []
        async for event in run_agent_loop(
            backend=backend, registry=registry,
            user_message="Compare the two kernels",
            system_prompt=prompt, stream=False,
        ):
            events.append(event)

        tc_events = [e for e in events if e.type == "tool_call"]
        assert len(tc_events) == 2
        assert tc_events[0].data["name"] == "list_kernels"
        assert tc_events[1].data["name"] == "get_kernel_summary"

    async def test_llm_error_fallback(self, report_compute_bound: KernelReport):
        """LLM error yields system error event + done."""
        session = _build_session([report_compute_bound])
        registry = _build_tool_registry(session)

        class ErrorBackend(LLMBackend):
            async def chat_completion(self, *args, **kwargs):
                raise ConnectionError("API timeout")
            def format_tool_definitions(self, tools): return []
            def parse_tool_calls(self, response): return []

        backend = ErrorBackend(model="fail")
        prompt = build_system_prompt(registry)

        events = []
        async for event in run_agent_loop(
            backend=backend, registry=registry,
            user_message="Hello",
            system_prompt=prompt, stream=False,
        ):
            events.append(event)

        types = [e.type for e in events]
        assert "system" in types
        assert "done" in types
        sys_event = next(e for e in events if e.type == "system")
        assert "error" in sys_event.content.lower() or "API timeout" in sys_event.content

    async def test_token_usage_tracking(self, report_compute_bound: KernelReport):
        """Done event contains accumulated token usage."""
        session = _build_session([report_compute_bound])
        registry = _build_tool_registry(session)

        backend = _MockBackend([
            CompletionResponse(
                content=None,
                tool_calls=[ToolCall(id="tc1", name="list_kernels", arguments={})],
                usage=Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
            ),
            CompletionResponse(
                content="Done.",
                tool_calls=[],
                usage=Usage(prompt_tokens=200, completion_tokens=30, total_tokens=230),
            ),
        ])

        prompt = build_system_prompt(registry)
        events = []
        async for event in run_agent_loop(
            backend=backend, registry=registry,
            user_message="Hello",
            system_prompt=prompt, stream=False,
        ):
            events.append(event)

        done = next(e for e in events if e.type == "done")
        assert done.data["prompt_tokens"] == 300  # 100 + 200
        assert done.data["completion_tokens"] == 50  # 20 + 30
        assert done.data["turns"] == 2
        assert done.data["tool_calls"] == 1


class TestPersonaIntegration:

    def test_system_prompt_contains_tools(self, report_compute_bound: KernelReport):
        """System prompt lists registered tools."""
        session = _build_session([report_compute_bound])
        registry = _build_tool_registry(session)

        prompt = build_system_prompt(registry)
        # Should contain all 9 tool names
        for name in [
            "list_kernels", "get_kernel_metrics", "get_kernel_summary",
            "get_ncu_rule_results", "get_source_hotspots",
            "get_sass_for_source_line", "get_stall_analysis_for_line",
            "run_analysis", "get_optimization_tree",
        ]:
            assert name in prompt

    def test_system_prompt_contains_kernel_context(
        self, report_compute_bound: KernelReport,
    ):
        """System prompt includes kernel list when provided."""
        session = _build_session([report_compute_bound])
        registry = _build_tool_registry(session)

        ctx = build_kernel_context([report_compute_bound])
        prompt = build_system_prompt(registry, ctx)
        assert "[0]" in prompt

    def test_system_prompt_methodology(self):
        """System prompt contains 7-step methodology."""
        registry = ToolRegistry()
        prompt = build_system_prompt(registry)
        assert "ORIENT" in prompt
        assert "DIAGNOSE" in prompt
        assert "RECOMMEND" in prompt
