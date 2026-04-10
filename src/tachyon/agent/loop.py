"""Multi-turn agent loop with tool execution and streaming output.

Flow:
  1. Build messages (system + history + user)
  2. Loop up to MAX_TURNS:
     a. Call LLM with messages + tool definitions
     b. If tool_calls → execute → append results → continue
     c. If text only → yield → done
     d. Last 2 turns → force synthesis (no tools)
  3. Yield "done" event with token usage
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..llm.backend import (
    CompletionResponse,
    LLMBackend,
    Message,
    Role,
    StreamChunk,
    ToolCall,
)
from ..tools.registry import ToolConcurrency, ToolRegistry
from .context import ContextManager
from ..utils.debug_record import record_prompt, record_tool, level as _debug_level

_log = logging.getLogger(__name__)

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_CJK_LANG_HINT = "（用中文回答）\n\n"


def _prepend_lang_hint(text: str, *, explicit_lang: bool = False) -> str:
    """Prepend a Chinese-response hint when needed.

    When *explicit_lang* is True the hint is added regardless of text content
    (caller has determined the user's language preference). Otherwise it is
    added only when *text* contains CJK characters.
    """
    if (explicit_lang or _CJK_RE.search(text)) and not text.startswith("（"):
        return _CJK_LANG_HINT + text
    return text

MAX_TURNS = 15
TYPICAL_TURNS = 5


def _has_called_tool(messages: list[Message], tool_name: str) -> bool:
    """Check whether a specific tool has been called in the message history.

    Scans TOOL-role messages for tool_call_id / name references and
    ASSISTANT-role messages for tool_calls with the given name.
    """
    for msg in messages:
        if msg.role == Role.TOOL and msg.name == tool_name:
            return True
        if msg.role == Role.ASSISTANT and msg.tool_calls:
            if any(tc.name == tool_name for tc in msg.tool_calls):
                return True
    return False


def _extract_tool_names_from_messages(messages: list[Message]) -> set[str]:
    """Extract tool names from a list of messages (for initial scan only)."""
    used = set()
    for msg in messages:
        if msg.role == Role.TOOL and msg.name:
            used.add(msg.name)
        if msg.role == Role.ASSISTANT and msg.tool_calls:
            for tc in msg.tool_calls:
                used.add(tc.name)
    return used


def _get_deferred_tool_defs(
    registry: ToolRegistry,
    used_tools: set[str],
    turn: int,
    critical_tools: set[str] | None = None,
) -> list:
    """Get tool definitions with deferred loading for token efficiency.

    Turn 0: Return all tool definitions (LLM needs to know all available tools).
    Turn > 0: Only return tools that have been used + critical tools.

    Critical tools are those that should always be available (e.g., edit, compile).
    """
    if turn == 0:
        return registry.all_definitions()

    critical = critical_tools or set()

    # Combine used tools with critical tools
    needed = used_tools | critical

    # Filter registry to only include needed tools
    if needed:
        return [td for td in registry.all_definitions() if td.name in needed]

    # Fallback: if no tools have been used, return all (shouldn't happen in turn > 0)
    return registry.all_definitions()


@dataclass
class AgentEvent:
    """Event yielded by the agent loop.

    Types: "text", "thinking", "tool_call", "tool_result", "system", "done".
    """
    type: str
    content: str | None = None
    data: dict[str, Any] | None = None


@dataclass
class AgentUsage:
    """Accumulated token usage across all turns."""
    prompt_tokens: int = 0          # latest API-reported context size
    completion_tokens: int = 0      # accumulated completion tokens (incremental)
    total_cost: int = 0             # accumulated API cost: sum(prompt + completion) per turn
    turns: int = 0
    tool_calls: int = 0
    last_prompt_tokens: int = 0     # most recent API prompt token count
    peak_prompt_tokens: int = 0     # highest prompt_tokens seen
    compaction_count: int = 0       # number of context compressions applied


def _serialize_tool_result(result: Any) -> str:
    """Serialize a ToolResult to JSON string for LLM consumption.

    Success → json.dumps(result.data)
    Failure → {"error": message, "suggestion": ..., "code": ...}
    """
    from ..errors.handler import ToolResult
    if result.success:
        return json.dumps(result.data, ensure_ascii=False, default=str)
    # Defensive: handle missing error info
    if result.error is None:
        return json.dumps({"error": "Tool execution failed"}, ensure_ascii=False)
    error_obj: dict[str, Any] = {"error": result.error.message}
    if result.error.suggestion:
        error_obj["suggestion"] = result.error.suggestion
    if result.error.code:
        error_obj["code"] = result.error.code.value
    return json.dumps(error_obj, ensure_ascii=False, default=str)


# Large result persistence: save to disk if result exceeds this size
_LARGE_RESULT_THRESHOLD = 2048  # 2KB
_PERSIST_DIR = Path(".tachyon") / "persisted_results"


def _persist_large_result(
    result_str: str,
    tool_name: str,
    tool_call_id: str,
) -> str:
    """Persist large tool result to disk and return a preview string.

    When result_str exceeds _LARGE_RESULT_THRESHOLD, saves the full content
    to disk and returns a summary with preview for the LLM message.

    Args:
        result_str: The serialized tool result string
        tool_name: Name of the tool that produced the result
        tool_call_id: Unique ID for this tool call

    Returns:
        Either the original result_str (if small) or a preview summary
    """
    if len(result_str) <= _LARGE_RESULT_THRESHOLD:
        return result_str

    # Create persistence directory
    persist_dir = _PERSIST_DIR
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename
    import hashlib
    content_hash = hashlib.md5(result_str.encode()).hexdigest()[:8]
    filename = f"{tool_name}_{tool_call_id}_{content_hash}.json"
    filepath = persist_dir / filename

    # Save full result
    filepath.write_text(result_str, encoding="utf-8")

    # Create preview (first 1KB)
    preview = result_str[:1024]
    if len(result_str) > 1024:
        preview += "..."

    # Return summary for LLM
    size_kb = len(result_str) / 1024
    return (
        f'<persisted-output>\n'
        f'Output too large ({size_kb:.1f}KB). Full output saved to: {filepath}\n'
        f'Preview (first 1KB):\n{preview}\n'
        f'</persisted-output>'
    )


# Large result persistence: save to disk if exceeds threshold
_PERSIST_THRESHOLD = 2048  # 2KB
_PREVIEW_LENGTH = 512  # Show first 512 chars in message
_PERSIST_DIR = ".tachyon/persisted"


def _persist_large_result(result_str: str, tool_name: str, turn: int) -> str:
    """Persist large tool result to disk and return preview for message.

    If result is small, return as-is. If large, save to disk and return
    a truncated preview with path to full file.
    """
    if len(result_str) <= _PERSIST_THRESHOLD:
        return result_str

    from datetime import datetime
    from pathlib import Path

    # Create persisted directory
    persist_dir = Path(_PERSIST_DIR)
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = tool_name.replace("/", "_").replace("\\", "_")
    filename = f"{safe_name}_{turn}_{timestamp}.json"
    filepath = persist_dir / filename

    # Write full result to disk
    filepath.write_text(result_str, encoding="utf-8")

    # Return preview with path
    preview = result_str[:_PREVIEW_LENGTH]
    remaining = len(result_str) - _PREVIEW_LENGTH

    return (
        f"<persisted-output size=\"{len(result_str)} bytes\">\n"
        f"Output too large ({len(result_str):,} bytes). "
        f"Full output saved to: {filepath}\n"
        f"Preview (first {_PREVIEW_LENGTH} chars):\n"
        f"{preview}\n"
        f"... ({remaining:,} more chars)\n"
        f"</persisted-output>"
    )


async def run_agent_loop(
    backend: LLMBackend,
    registry: ToolRegistry,
    user_message: str,
    system_prompt: str,
    history: list[Message] | None = None,
    stream: bool = False,
    context_budget: int = 120_000,
    timeout: int = 600,
    move_timeout: int = 120,
    max_retries: int = 1,
    lean_system_prompt: str | None = None,
    trim_user_after_turn0: str | None = None,
    synthesis_prompt: str | None = None,
    prefer_lang: str | None = None,
    max_turns: int = MAX_TURNS,
    skip_synthesis: bool = False,
    urgent_compile_tool: str | None = None,
    force_stop_check: Callable[[list[Message], int], str | None] | None = None,
    temperature: float = 0.1,
    critical_tools: set[str] | None = None,
) -> AsyncIterator[AgentEvent]:
    """Execute the multi-turn agent loop.

    Args:
        backend: LLMBackend instance.
        registry: ToolRegistry with all tools.
        user_message: User's question or request.
        system_prompt: Assembled system prompt (full, for turn 0).
        history: Optional conversation history.
        stream: Whether to stream LLM output.
        context_budget: Token budget for context management.
        timeout: Total timeout in seconds (default 600 = 10 min).
        move_timeout: Per-move timeout for LLM call (default 120s).
        max_retries: Retry count for 429/timeout errors (default 1).
        lean_system_prompt: Minimal system prompt for turns after 0.
            Saves ~2000 tokens per subsequent turn.
        trim_user_after_turn0: If set, replace the first USER message with
            this shorter text after turn 0 (saves tokens on turns 1+).
        synthesis_prompt: Override the synthesis USER message injected at
            the penultimate turn.
        prefer_lang: Force output language. ``"zh"`` for Chinese, ``None``
            for auto-detect from *user_message* content.
        urgent_compile_tool: If set (e.g., ``"compile_kernel"``), inject an
            urgent USER message when ≤5 turns remain and this tool has not
            been called yet, forcing the LLM into the compile phase.
        force_stop_check: Optional callback invoked after each turn's tool
            calls complete.  If it returns a non-None string, the agent loop
            terminates immediately and yields a system event with that reason.
        critical_tools: Set of tool names that should always be available
            (in addition to already-used tools). Used for deferred tool
            schema loading to save tokens. If None, defaults to empty set.

    Yields:
        AgentEvent instances for the caller to render.
    """
    ctx = ContextManager(budget=context_budget)
    usage = AgentUsage()
    t_start = time.monotonic()

    messages: list[Message] = [
        Message(role=Role.SYSTEM, content=system_prompt),
    ]
    if history:
        messages.extend(history)
    _chinese = prefer_lang == "zh" or (
        prefer_lang is None and bool(_CJK_RE.search(user_message))
    )
    messages.append(Message(role=Role.USER, content=_prepend_lang_hint(user_message, explicit_lang=_chinese)))

    # Prepare critical tools set for deferred loading
    _critical = critical_tools or set()
    if urgent_compile_tool:
        _critical.add(urgent_compile_tool)

    # Cache used tools incrementally (avoid O(n) scan each turn)
    used_tools: set[str] = _extract_tool_names_from_messages(messages)

    for turn in range(max_turns):
        usage.turns = turn + 1

        # Deferred tool schema loading: Turn 0 sends all, Turn > 0 sends only used + critical
        tool_defs = _get_deferred_tool_defs(registry, used_tools, turn, _critical)

        # Total timeout check
        elapsed = time.monotonic() - t_start
        if elapsed > timeout:
            yield AgentEvent(
                type="system",
                content=f"Timeout ({timeout}s) reached after {elapsed:.1f}s. "
                        f"Returning partial results.",
            )
            break

        # Token-driven context compression
        if usage.last_prompt_tokens > 0:
            ctx.update_token_count(usage.last_prompt_tokens)

        tier = ctx.needs_compaction()
        if tier > 0:
            budget_pct = ctx.get_budget_pct(messages)
            actions = await ctx.compact(messages, backend=backend)
            for t, desc in actions:
                yield AgentEvent(
                    type="system",
                    content=f"Context compression (Tier {t}): {desc}",
                    data={
                        "compaction": {
                            "tier": t,
                            "budget_pct": round(budget_pct, 1),
                            "tokens": ctx.current_tokens,
                            "budget": ctx.budget,
                        },
                    },
                )
            usage.compaction_count = ctx.compaction_count

        # Reserve last 2 turns for synthesis (force no tools)
        remaining = max_turns - turn
        is_synthesis = (not skip_synthesis) and remaining <= 2 and turn > 0
        is_final = (turn == max_turns - 1)

        if is_synthesis and remaining == 2:
            synth = synthesis_prompt or (
                "You have gathered enough data. Now synthesize your "
                "findings into a comprehensive analysis. Do NOT call "
                "any more tools — provide your final diagnosis, root "
                "causes, and prioritized recommendations based on "
                "all the evidence collected above."
            )
            if _chinese:
                synth = "（请用中文综合你的发现）\n\n" + synth
            messages.append(Message(role=Role.USER, content=synth))

        # Urgent compile injection: if running low on turns and the
        # specified critical tool (e.g., compile_kernel) hasn't been
        # called yet, inject a USER message to force the LLM into
        # the compilation phase.
        if (
            not is_synthesis
            and urgent_compile_tool
            and remaining <= max(max_turns // 3, 3)
            and turn > 0
            and urgent_compile_tool not in used_tools
        ):
            yield AgentEvent(
                type="system",
                content=(
                    f"URGENT: {remaining} turns remaining and "
                    f"{urgent_compile_tool} has NOT been called. "
                    "Injecting compile-first directive."
                ),
            )
            messages.append(Message(
                role=Role.USER,
                content=(
                    f"URGENT: Only {remaining} turns remaining and "
                    f"{urgent_compile_tool} has NOT been called. "
                    f"Call {urgent_compile_tool} NOW with whatever "
                    "edits you have. Incomplete optimization is better "
                    "than a FAILED iteration."
                ),
            ))

        # Signal that we're waiting for LLM
        yield AgentEvent(
            type="system",
            content="llm_start",
            data={"turn": turn + 1, "llm_start": True},
        )

        # Call LLM with per-move timeout and retry
        t_turn = time.monotonic()
        response = None
        llm_error = None
        effective_timeout = move_timeout  # resets each turn; grows on timeout retries

        # Debug: record the exact prompt sent to the LLM
        record_prompt(
            messages,
            tools=tool_defs if not is_synthesis else None,
            turn=turn + 1,
        )

        for attempt in range(1 + max_retries):
            try:
                response = await asyncio.wait_for(
                    backend.chat_completion(
                        messages=messages,
                        tools=tool_defs if not is_synthesis else None,
                        tool_choice="none" if is_synthesis else "auto",
                        stream=stream,
                        max_tokens=4096,
                        temperature=temperature,
                    ),
                    timeout=effective_timeout,
                )
                llm_error = None
                break  # success
            except asyncio.TimeoutError:
                llm_error = f"API timeout ({effective_timeout}s)"
                if attempt < max_retries:
                    backoff = min(5 * 2 ** attempt, 30)
                    next_timeout = min(int(effective_timeout * 1.5), 300)
                    yield AgentEvent(
                        type="system",
                        content=f"LLM call timed out after {effective_timeout}s, "
                                f"retrying in {backoff}s with "
                                f"{next_timeout}s timeout "
                                f"({attempt + 1}/{max_retries})...",
                    )
                    effective_timeout = next_timeout
                    await asyncio.sleep(backoff)
            except Exception as e:
                err_str = str(e)
                is_retryable = (
                    "429" in err_str
                    or "Error code: 5" in err_str  # 5xx server errors
                )
                if is_retryable and attempt < max_retries:
                    wait = 5 * (attempt + 1)  # 5s, 10s backoff
                    yield AgentEvent(
                        type="system",
                        content=f"LLM error ({err_str[:60]}), "
                                f"waiting {wait}s before retry "
                                f"({attempt + 1}/{max_retries})...",
                    )
                    await asyncio.sleep(wait)
                else:
                    llm_error = err_str
                    break

        if response is None:
            yield AgentEvent(
                type="system",
                content=f"LLM error ({llm_error}): giving up after "
                        f"{max_retries + 1} attempts.",
            )
            yield AgentEvent(type="done", data=_usage_dict(usage, t_start, ctx.budget))
            return

        llm_elapsed = time.monotonic() - t_turn

        # Process response
        if stream and isinstance(response, AsyncIterator):
            collected = await _collect_stream(response)
            content = collected["content"]
            tool_calls = collected["tool_calls"]
            _update_usage(usage, collected.get("usage", {}))
        else:
            assert isinstance(response, CompletionResponse)
            content = response.content
            tool_calls = response.tool_calls
            u = response.usage
            if isinstance(u, dict):
                _update_usage(usage, u)
            else:
                _update_usage(usage, {
                    "prompt_tokens": getattr(u, "prompt_tokens", 0),
                    "completion_tokens": getattr(u, "completion_tokens", 0),
                })

        # No tool calls → final answer
        if not tool_calls:
            if content:
                yield AgentEvent(
                    type="text",
                    content=content,
                    data={"elapsed": round(llm_elapsed, 1)},
                )
            break

        # Tool calls with accompanying text → yield as "thinking"
        messages.append(Message(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
        ))

        # Update used_tools cache incrementally (avoid O(n) scan next turn)
        for tc in tool_calls:
            used_tools.add(tc.name)

        if content:
            yield AgentEvent(type="thinking", content=content)

        # --- Parallel Tool Execution ---
        # Inspired by Claude Code's pnH design:
        # 1. SAFE tools can run in parallel (read-only operations)
        # 2. SEQUENTIAL tools must run one at a time (file modifications)
        # 3. EXCLUSIVE tools must run alone (resource-intensive)

        safe_calls = []
        seq_calls = []

        for tc in tool_calls:
            tool_def = registry.get(tc.name)
            if tool_def and tool_def.is_concurrency_safe():
                safe_calls.append(tc)
            else:
                seq_calls.append(tc)

        # Execute SAFE tools in parallel
        if safe_calls:
            for tc in safe_calls:
                usage.tool_calls += 1
                yield AgentEvent(
                    type="tool_call",
                    content=f"Calling {tc.name}...",
                    data={"name": tc.name, "arguments": tc.arguments},
                )

            t_parallel = time.monotonic()
            safe_results = await asyncio.gather(
                *[registry.execute(tc.name, tc.arguments) for tc in safe_calls],
                return_exceptions=True
            )

            for tc, result in zip(safe_calls, safe_results):
                tool_elapsed = time.monotonic() - t_parallel
                if isinstance(result, Exception):
                    result = ToolResult.fail(ErrorCode.UNKNOWN, str(result))

                result_str = _serialize_tool_result(result)

                record_tool(tc.name, tc.arguments, result_str, tool_elapsed, result.success)

                yield AgentEvent(
                    type="tool_result",
                    content=f"{tc.name}: {'OK' if result.success else 'ERROR'}",
                    data={"name": tc.name, "success": result.success,
                          "summary": result_str[:200],
                          "elapsed": round(tool_elapsed, 2)},
                )

                messages.append(Message(
                    role=Role.TOOL,
                    content=result_str,
                    tool_call_id=tc.id,
                    name=tc.name,
                ))

        # Execute SEQUENTIAL/EXCLUSIVE tools one at a time
        for tc in seq_calls:
            usage.tool_calls += 1
            yield AgentEvent(
                type="tool_call",
                content=f"Calling {tc.name}...",
                data={"name": tc.name, "arguments": tc.arguments},
            )

            t_tool = time.monotonic()
            result = await registry.execute(tc.name, tc.arguments)
            tool_elapsed = time.monotonic() - t_tool

            result_str = _serialize_tool_result(result)

            record_tool(tc.name, tc.arguments, result_str, tool_elapsed, result.success)

            yield AgentEvent(
                type="tool_result",
                content=f"{tc.name}: {'OK' if result.success else 'ERROR'}",
                data={"name": tc.name, "success": result.success,
                      "summary": result_str[:200],
                      "elapsed": round(tool_elapsed, 2)},
            )

            messages.append(Message(
                role=Role.TOOL,
                content=result_str,
                tool_call_id=tc.id,
                name=tc.name,
            ))

        # Emit turn timing
        turn_elapsed = time.monotonic() - t_turn

        # Force-stop check: after all tool calls in this turn,
        # ask the caller whether the loop should be terminated early.
        if force_stop_check is not None and tool_calls:
            stop_reason = force_stop_check(messages, turn)
            if stop_reason:
                yield AgentEvent(
                    type="system",
                    content=f"Force stop: {stop_reason}",
                    data={"force_stop": True},
                )
                break

        # After turn 0: trim system prompt to save tokens on subsequent turns
        if turn == 0 and lean_system_prompt and messages[0].role == Role.SYSTEM:
            old_len = len(messages[0].content)
            messages[0] = Message(role=Role.SYSTEM, content=lean_system_prompt)
            if _debug_level() > 0:
                saved = old_len - len(lean_system_prompt)
                yield AgentEvent(
                    type="system",
                    content=f"System prompt trimmed ({old_len:,} -> "
                            f"{len(lean_system_prompt):,} chars, "
                            f"saved ~{saved:,} chars per turn)",
                )

        # After turn 0: trim first user message if requested (saves tokens)
        if turn == 0 and trim_user_after_turn0 is not None:
            trimmed = _prepend_lang_hint(trim_user_after_turn0, explicit_lang=_chinese)
            for i, m in enumerate(messages):
                if m.role == Role.USER:
                    messages[i] = Message(role=Role.USER, content=trimmed)
                    if _debug_level() > 0:
                        yield AgentEvent(
                            type="system",
                            content=f"User message trimmed to: {trim_user_after_turn0[:80]}",
                        )
                    break

        yield AgentEvent(
            type="system",
            content=f"Turn {turn + 1}: {turn_elapsed:.1f}s "
                    f"(LLM {llm_elapsed:.1f}s + {len(tool_calls)} tools)",
            data={"turn": turn + 1, "elapsed": round(turn_elapsed, 1)},
        )

        _log.debug("Agent at turn %d/%d (%.1fs)", turn + 1, max_turns, turn_elapsed)

    yield AgentEvent(type="done", data=_usage_dict(usage, t_start, ctx.budget))


def _update_usage(usage: AgentUsage, data: dict) -> None:
    """Update token usage from a turn.

    ``prompt_tokens`` from the API is the *current context size* (not
    incremental), so we store it directly.  ``completion_tokens`` is
    per-turn and should be accumulated.
    """
    prompt = data.get("prompt_tokens", 0)
    completion = data.get("completion_tokens", 0)
    if prompt == 0:
        _log.warning("API returned prompt_tokens=0 (total_cost will undercount)")
    usage.last_prompt_tokens = prompt
    usage.peak_prompt_tokens = max(usage.peak_prompt_tokens, prompt)
    usage.prompt_tokens = prompt
    usage.completion_tokens += completion
    usage.total_cost += prompt + completion  # real per-turn API cost


def _usage_dict(
    usage: AgentUsage, t_start: float | None = None, budget: int = 0,
) -> dict:
    """Convert AgentUsage to dict for AgentEvent.data."""
    d = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_cost,
        "turns": usage.turns,
        "tool_calls": usage.tool_calls,
        "peak_prompt_tokens": usage.peak_prompt_tokens,
        "compaction_count": usage.compaction_count,
    }
    if budget > 0:
        d["budget"] = budget
        d["budget_peak"] = usage.peak_prompt_tokens  # peak context size (pre-compact)
        d["budget_remaining"] = max(0, budget - usage.peak_prompt_tokens)
        d["budget_pct"] = round((usage.peak_prompt_tokens / budget) * 100, 1)
    if t_start is not None:
        d["total_elapsed"] = round(time.monotonic() - t_start, 1)
    return d


async def _collect_stream(
    stream: AsyncIterator[StreamChunk],
) -> dict[str, Any]:
    """Collect streaming chunks into content + tool_calls + usage."""
    content_parts: list[str] = []
    tool_call_buffers: dict[str, dict] = {}
    usage: dict = {}

    async for chunk in stream:
        if chunk.content:
            content_parts.append(chunk.content)

        if chunk.tool_call_delta:
            delta = chunk.tool_call_delta
            tc_id = delta.get("id")
            if tc_id and tc_id not in tool_call_buffers:
                tool_call_buffers[tc_id] = {
                    "name": delta.get("name", ""),
                    "arguments_str": "",
                }
            if "arguments" in delta:
                target_id = tc_id or (
                    list(tool_call_buffers.keys())[-1]
                    if tool_call_buffers else None
                )
                if target_id and target_id in tool_call_buffers:
                    tool_call_buffers[target_id]["arguments_str"] += delta["arguments"]

        if chunk.usage:
            usage = {
                "prompt_tokens": chunk.usage.prompt_tokens,
                "completion_tokens": chunk.usage.completion_tokens,
            }

    tool_calls = []
    for tc_id, buf in tool_call_buffers.items():
        try:
            args = json.loads(buf["arguments_str"])
        except json.JSONDecodeError:
            args = {}
        tool_calls.append(ToolCall(id=tc_id, name=buf["name"], arguments=args))

    return {
        "content": "".join(content_parts) or None,
        "tool_calls": tool_calls,
        "usage": usage,
    }
