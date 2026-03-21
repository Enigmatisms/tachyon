"""Unit tests for multi-stage deep analysis (stages.py)."""
from unittest.mock import patch

import pytest

from tachyon.analysis.stages import (
    StageResult,
    StageSpec,
    _extract_scratch_board,
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
        assert "result text" in results[0].text

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
        assert "Stage 1 failed" in results[0].text

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
        assert "No output" in results[0].text

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


# ── Deep mode and radical mode ──


class TestDeepRadicalModes:
    def test_deep_mode_two_stages(self):
        """mode='deep' returns 2 stages."""
        specs = build_stage_prompts(mode="deep")
        assert len(specs) == 2
        assert all(isinstance(s, StageSpec) for s in specs)
        # Stage 1 should have scratch board hint
        assert "scratch_board" in specs[0].prompt.lower()

    def test_radical_mode_three_stages(self):
        """mode='radical' returns 3 stages (same as mode='profile')."""
        specs = build_stage_prompts(mode="radical")
        assert len(specs) == 3
        assert all(isinstance(s, StageSpec) for s in specs)

    def test_profile_mode_has_scratch_board_hints(self):
        """profile/radical modes should have scratch board hints in stage 1 and 2."""
        for mode in ("profile", "radical"):
            specs = build_stage_prompts(mode=mode)
            assert "scratch_board" in specs[0].prompt.lower()
            assert "scratch_board" in specs[1].prompt.lower()

    def test_chat_mode_no_scratch_board_hints(self):
        """chat mode should NOT have scratch board hints."""
        specs = build_stage_prompts(mode="chat")
        assert "scratch_board" not in specs[0].prompt.lower()


# ── Scratch board extraction ──


class TestExtractScratchBoard:
    def test_extract_scratch_board_present(self):
        """Extracts and removes scratch board from text."""
        text = "Analysis here.\n<scratch_board>Hotspot: k0, compute-bound, SM=85%</scratch_board>\nMore text."
        cleaned, board = _extract_scratch_board(text)
        assert "scratch_board" not in cleaned
        assert "Hotspot: k0" in board
        assert "SM=85%" in board
        assert "Analysis here" in cleaned
        assert "More text" in cleaned

    def test_extract_scratch_board_multiline(self):
        """Handles multiline scratch board content."""
        text = "Output\n<scratch_board>\n- hotspot1\n- hotspot2\n- hotspot3\n</scratch_board>\nEnd"
        cleaned, board = _extract_scratch_board(text)
        assert "hotspot1" in board
        assert "hotspot2" in board
        assert "hotspot3" in board
        assert "Output" in cleaned
        assert "End" in cleaned

    def test_extract_scratch_board_absent(self):
        """Returns full text and empty board when no scratch board found."""
        text = "Just analysis text with no scratch board."
        cleaned, board = _extract_scratch_board(text)
        assert cleaned == text
        assert board == ""

    def test_extract_scratch_board_empty_tag(self):
        """Handles empty scratch board tags."""
        text = "Text<scratch_board></scratch_board>More"
        cleaned, board = _extract_scratch_board(text)
        assert board == ""
        assert "Text" in cleaned
        assert "More" in cleaned


# ── Scratch board context injection ──


class TestScratchBoardContextInjection:
    @pytest.mark.asyncio
    async def test_scratch_board_used_for_context(self):
        """When stage 1 produces scratch board, stage 2 gets scratch_boards instead of full output."""
        specs = [
            StageSpec(name="S1", prompt="stage 1"),
            StageSpec(name="S2", prompt="stage 2"),
        ]

        call_args_history = []

        async def fake_loop(*args, **kwargs):
            call_args_history.append(kwargs.get("user_message", ""))
            if kwargs.get("user_message", "").startswith("data"):
                yield _make_text_event(
                    "Analysis.\n<scratch_board>Top hotspot: k0, SM=90%</scratch_board>"
                )
            else:
                yield _make_text_event("Deep analysis")
            yield _make_done_event()

        with patch("tachyon.agent.loop.run_agent_loop", side_effect=fake_loop):
            results = await run_staged_analysis(
                None, None,
                user_prompt="data",
                system_prompt="sys",
                stage_specs=specs,
            )

        assert len(results) == 2
        # Stage 1 output should have scratch board stripped
        assert "scratch_board" not in results[0].text
        assert "Analysis" in results[0].text
        # Stage 2 should receive scratch_boards tag, not previous_stage_output
        stage2_msg = call_args_history[1]
        assert "<scratch_boards>" in stage2_msg
        assert "Top hotspot: k0" in stage2_msg
        assert "SM=90%" in stage2_msg

    @pytest.mark.asyncio
    async def test_fallback_to_full_output_when_no_scratch_board(self):
        """When scratch board is empty, falls back to full previous output."""
        specs = [
            StageSpec(name="S1", prompt="stage 1"),
            StageSpec(name="S2", prompt="stage 2"),
        ]

        call_args_history = []

        async def fake_loop(*args, **kwargs):
            call_args_history.append(kwargs.get("user_message", ""))
            yield _make_text_event("Full analysis without scratch board")
            yield _make_done_event()

        with patch("tachyon.agent.loop.run_agent_loop", side_effect=fake_loop):
            await run_staged_analysis(
                None, None,
                user_prompt="data",
                system_prompt="sys",
                stage_specs=specs,
            )

        stage2_msg = call_args_history[1]
        # Should fall back to previous_stage_output tag
        assert "<previous_stage_output>" in stage2_msg
        assert "Full analysis" in stage2_msg


# ── StageResult and status_display ──


class TestStageResult:
    def test_stage_result_creation(self):
        sr = StageResult(text="analysis", done_data={"turns": 3})
        assert sr.text == "analysis"
        assert sr.done_data["turns"] == 3

    def test_stage_result_defaults(self):
        sr = StageResult(text="text", done_data={})
        assert sr.text == "text"
        assert sr.done_data == {}


class TestRunStagedAnalysisStatusDisplay:
    @pytest.mark.asyncio
    async def test_status_display_set_stage_called(self):
        """status_display.set_stage should be called for each stage."""
        from unittest.mock import MagicMock

        specs = [StageSpec(name="S1", prompt="p1"), StageSpec(name="S2", prompt="p2")]
        mock_display = MagicMock()

        async def fake_loop(*args, **kwargs):
            yield _make_text_event("output")
            yield _make_done_event()

        with patch("tachyon.agent.loop.run_agent_loop", side_effect=fake_loop):
            await run_staged_analysis(
                None, None,
                user_prompt="data", system_prompt="sys",
                stage_specs=specs,
                status_display=mock_display,
            )

        # set_stage should be called for stage 0 and stage 1
        assert mock_display.set_stage.call_count == 2
        mock_display.set_stage.assert_any_call(0, 2)
        mock_display.set_stage.assert_any_call(1, 2)

    @pytest.mark.asyncio
    async def test_status_display_set_tool_called(self):
        """status_display.set_tool should be called for tool_call events."""
        from unittest.mock import MagicMock

        specs = [StageSpec(name="S1", prompt="p1")]
        mock_display = MagicMock()

        async def fake_loop(*args, **kwargs):
            from tachyon.agent.loop import AgentEvent
            yield AgentEvent(
                type="tool_call",
                content="Calling get_kernel_summary...",
                data={"name": "get_kernel_summary"},
            )
            yield _make_text_event("text")
            yield _make_done_event()

        with patch("tachyon.agent.loop.run_agent_loop", side_effect=fake_loop):
            results = await run_staged_analysis(
                None, None,
                user_prompt="data", system_prompt="sys",
                stage_specs=specs,
                status_display=mock_display,
            )

        mock_display.set_tool.assert_called_once_with("get_kernel_summary")

    @pytest.mark.asyncio
    async def test_done_data_collected(self):
        """done event data should be captured in StageResult.done_data."""
        specs = [StageSpec(name="S1", prompt="p1")]

        async def fake_loop(*args, **kwargs):
            yield _make_text_event("output")
            yield _make_done_event()

        with patch("tachyon.agent.loop.run_agent_loop", side_effect=fake_loop):
            results = await run_staged_analysis(
                None, None,
                user_prompt="data", system_prompt="sys",
                stage_specs=specs,
            )

        assert results[0].done_data["turns"] == 2
        assert results[0].done_data["tool_calls"] == 1
        assert results[0].done_data["total_tokens"] == 500

    @pytest.mark.asyncio
    async def test_render_fn_fallback_without_status_display(self):
        """render_fn should still work when status_display is None (backward compat)."""
        specs = [StageSpec(name="S1", prompt="p1")]
        rendered = []

        def render(idx, name, text):
            rendered.append((idx, name, text))

        async def fake_loop(*args, **kwargs):
            from tachyon.agent.loop import AgentEvent
            yield AgentEvent(
                type="tool_call",
                content="Calling tool...",
                data={"name": "some_tool"},
            )
            yield _make_text_event("text")
            yield _make_done_event()

        with patch("tachyon.agent.loop.run_agent_loop", side_effect=fake_loop):
            await run_staged_analysis(
                None, None,
                user_prompt="data", system_prompt="sys",
                stage_specs=specs,
                render_fn=render,
            )

        assert len(rendered) == 1
        assert rendered[0][2] == "[tool] some_tool"

