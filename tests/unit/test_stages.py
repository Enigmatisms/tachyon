"""Unit tests for multi-stage deep analysis (stages.py)."""
from unittest.mock import patch

import pytest

from tachyon.analysis.stages import (
    StageSpec,
    build_stage_prompts,
    run_staged_analysis,
)


# ── StageSpec ──


class TestStageSpec:
    def test_defaults(self):
        spec = StageSpec(name="test", prompt="do stuff")
        assert spec.name == "test"
        assert spec.prompt == "do stuff"
        assert spec.max_turns == 10

    def test_custom_max_turns(self):
        spec = StageSpec(name="test", prompt="do stuff", max_turns=5)
        assert spec.max_turns == 5


# ── build_stage_prompts ──


class TestBuildStagePrompts:
    def test_profile_mode_three_stages(self):
        specs = build_stage_prompts(mode="profile")
        assert len(specs) == 3
        assert all(isinstance(s, StageSpec) for s in specs)

    def test_chat_mode_two_stages(self):
        specs = build_stage_prompts(mode="chat")
        assert len(specs) == 2
        assert all(isinstance(s, StageSpec) for s in specs)

    def test_default_mode_is_profile(self):
        specs = build_stage_prompts()
        assert len(specs) == 3

    def test_prompts_contain_key_tool_names(self):
        """Stage prompts should reference required tools regardless of language."""
        specs = build_stage_prompts(mode="profile")
        # Stage 1: metric tools
        assert "list_kernels" in specs[0].prompt
        assert "run_analysis" in specs[0].prompt
        # Stage 2: source tools
        assert "get_performance_hotspots" in specs[1].prompt
        assert "read_source_file" in specs[1].prompt
        # Stage 3: recommendation format
        assert "HIGH" in specs[2].prompt
        assert "MEDIUM" in specs[2].prompt

    def test_prompts_without_i18n_use_fallback(self):
        """When i18n is not initialized, prompts should use English fallback."""
        import tachyon.i18n as i18n
        i18n._packs.clear()
        i18n._current_lang = "en"
        specs = build_stage_prompts(mode="profile")
        # Should still work with fallback values
        assert len(specs) == 3
        assert "Stage 1" in specs[0].name
        assert "Stage 2" in specs[1].name
        assert "Stage 3" in specs[2].name

    def test_chat_combined_prompt_has_both(self):
        """Chat mode stage 2 should contain content from both source and recommend."""
        specs = build_stage_prompts(mode="chat")
        assert "Source" in specs[1].prompt or "Attribution" in specs[1].prompt
        assert "Optimization" in specs[1].prompt or "Plan" in specs[1].prompt

    def test_i18n_zh_stage_names(self):
        """When i18n is set to zh, stage names should be in Chinese."""
        import tachyon.i18n as i18n
        i18n.init("zh")
        specs = build_stage_prompts(mode="profile")
        assert "第一阶段" in specs[0].name or "指标" in specs[0].name
        assert "第二阶段" in specs[1].name or "源码" in specs[1].name
        assert "第三阶段" in specs[2].name or "改进" in specs[2].name


# ── run_staged_analysis ──


def _make_text_event(content: str):
    """Create a fake AgentEvent for text output."""
    from tachyon.agent.loop import AgentEvent
    return AgentEvent(type="text", content=content)


def _make_done_event():
    """Create a fake AgentEvent for completion."""
    from tachyon.agent.loop import AgentEvent
    return AgentEvent(
        type="done",
        data={"turns": 2, "tool_calls": 1, "total_tokens": 500, "total_elapsed": 3.0},
    )


class TestRunStagedAnalysis:
    @pytest.mark.asyncio
    async def test_single_stage(self):
        """Single stage should return one result."""
        specs = [StageSpec(name="S1", prompt="analyze metrics")]

        async def fake_loop(*args, **kwargs):
            yield _make_text_event("result text")
            yield _make_done_event()

        with patch("tachyon.agent.loop.run_agent_loop", side_effect=fake_loop):
            results = await run_staged_analysis(
                None, None,
                user_prompt="data context",
                system_prompt="system",
                stage_specs=specs,
            )

        assert len(results) == 1
        assert "result text" in results[0]

    @pytest.mark.asyncio
    async def test_two_stages_context_propagation(self):
        """Second stage should receive ALL previous stage outputs."""
        specs = [
            StageSpec(name="S1", prompt="stage 1 instructions"),
            StageSpec(name="S2", prompt="stage 2 instructions"),
        ]

        call_args_history = []

        async def fake_loop(*args, **kwargs):
            call_args_history.append(kwargs.get("user_message", ""))
            yield _make_text_event("output for stage")
            yield _make_done_event()

        with patch("tachyon.agent.loop.run_agent_loop", side_effect=fake_loop):
            results = await run_staged_analysis(
                None, None,
                user_prompt="pre-computed data",
                system_prompt="system prompt",
                stage_specs=specs,
            )

        assert len(results) == 2
        # Stage 1: pre-computed data + stage instructions
        assert "pre-computed data" in call_args_history[0]
        assert "stage 1 instructions" in call_args_history[0]
        # Stage 2: should contain ALL previous stage outputs
        assert "previous_stage_output" in call_args_history[1]
        assert "stage 2 instructions" in call_args_history[1]
        # Stage 2 must contain stage 1's output text
        assert "output for stage" in call_args_history[1]
        # Stage 2 must show the stage name from stage 1
        assert "S1" in call_args_history[1]

    @pytest.mark.asyncio
    async def test_three_stages_accumulate_context(self):
        """Stage 3 should have outputs from both stage 1 and stage 2."""
        specs = [
            StageSpec(name="S1", prompt="p1"),
            StageSpec(name="S2", prompt="p2"),
            StageSpec(name="S3", prompt="p3"),
        ]

        call_count = 0

        async def fake_loop(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            yield _make_text_event(f"stage-{call_count} output")
            yield _make_done_event()

        call_args_history = []

        with patch("tachyon.agent.loop.run_agent_loop", side_effect=fake_loop):
            original_run = fake_loop

            async def capturing_loop(*args, **kwargs):
                call_args_history.append(kwargs.get("user_message", ""))
                async for e in original_run(*args, **kwargs):
                    yield e

            with patch("tachyon.agent.loop.run_agent_loop", side_effect=capturing_loop):
                results = await run_staged_analysis(
                    None, None,
                    user_prompt="data", system_prompt="sys",
                    stage_specs=specs,
                )

        assert len(results) == 3
        # Stage 3 user message must contain outputs from stage 1 AND stage 2
        assert "stage-1 output" in call_args_history[2]
        assert "stage-2 output" in call_args_history[2]

    @pytest.mark.asyncio
    async def test_stage_failure_graceful(self):
        """Failed stage should produce error message, not crash."""
        specs = [StageSpec(name="S1", prompt="analyze")]

        with patch("tachyon.agent.loop.run_agent_loop", side_effect=RuntimeError("LLM down")):
            results = await run_staged_analysis(
                None, None,
                user_prompt="data", system_prompt="sys",
                stage_specs=specs,
            )

        assert len(results) == 1
        assert "Stage 1 failed" in results[0]

    @pytest.mark.asyncio
    async def test_render_fn_called_for_tool_event(self):
        """render_fn callback should be invoked for tool_call events."""
        specs = [StageSpec(name="S1", prompt="analyze")]
        rendered = []

        def render(idx, name, text):
            rendered.append((idx, name, text))

        async def fake_loop(*args, **kwargs):
            from tachyon.agent.loop import AgentEvent
            yield AgentEvent(
                type="tool_call",
                content="Calling get_kernel_summary...",
                data={"name": "get_kernel_summary", "arguments": {"kernel_id": 0}},
            )
            yield _make_text_event("analysis text")
            yield _make_done_event()

        with patch("tachyon.agent.loop.run_agent_loop", side_effect=fake_loop):
            await run_staged_analysis(
                None, None,
                user_prompt="data", system_prompt="sys",
                stage_specs=specs,
                render_fn=render,
            )

        assert len(rendered) == 1
        assert rendered[0][2] == "[tool] get_kernel_summary"

    @pytest.mark.asyncio
    async def test_empty_output_fallback(self):
        """Stage with no text output should get fallback message."""
        specs = [StageSpec(name="S1", prompt="analyze")]

        async def empty_gen(*args, **kwargs):
            yield _make_done_event()

        with patch("tachyon.agent.loop.run_agent_loop", side_effect=empty_gen):
            results = await run_staged_analysis(
                None, None,
                user_prompt="data", system_prompt="sys",
                stage_specs=specs,
            )

        assert len(results) == 1
        assert "No output" in results[0]

    @pytest.mark.asyncio
    async def test_stage_1_includes_data_prompt(self):
        """Stage 1 user message must contain the data prompt."""
        specs = [StageSpec(name="S1", prompt="instructions")]

        async def fake_loop(*args, **kwargs):
            # Verify user_message was passed correctly
            assert "pre-computed kernel data" in kwargs.get("user_message", "")
            assert "instructions" in kwargs.get("user_message", "")
            yield _make_text_event("ok")
            yield _make_done_event()

        with patch("tachyon.agent.loop.run_agent_loop", side_effect=fake_loop):
            await run_staged_analysis(
                None, None,
                user_prompt="pre-computed kernel data",
                system_prompt="sys",
                stage_specs=specs,
            )

    @pytest.mark.asyncio
    async def test_context_uses_i18n(self):
        """Context header should use i18n when available."""
        import tachyon.i18n as i18n
        i18n.init("zh")
        specs = [
            StageSpec(name="S1", prompt="p1"),
            StageSpec(name="S2", prompt="p2"),
        ]

        call_args_history = []

        async def fake_loop(*args, **kwargs):
            call_args_history.append(kwargs.get("user_message", ""))
            yield _make_text_event("out")
            yield _make_done_event()

        with patch("tachyon.agent.loop.run_agent_loop", side_effect=fake_loop):
            await run_staged_analysis(
                None, None,
                user_prompt="data", system_prompt="sys",
                stage_specs=specs,
            )

        # Stage 2 context header should be in Chinese
        assert "前面" in call_args_history[1] or "以下" in call_args_history[1]
