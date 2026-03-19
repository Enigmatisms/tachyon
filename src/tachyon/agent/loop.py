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
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from ..llm.backend import (
    CompletionResponse,
    LLMBackend,
    Message,
    Role,
    StreamChunk,
    ToolCall,
)
from ..tools.registry import ToolRegistry
from .context import ContextManager

_log = logging.getLogger(__name__)

MAX_TURNS = 10
TYPICAL_TURNS = 5


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
    prompt_tokens: int = 0
    completion_tokens: int = 0
    turns: int = 0
    tool_calls: int = 0


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
) -> AsyncIterator[AgentEvent]:
    """Execute the multi-turn agent loop.

    Args:
        backend: LLMBackend instance.
        registry: ToolRegistry with all tools.
        user_message: User's question or request.
        system_prompt: Assembled system prompt.
        history: Optional conversation history.
        stream: Whether to stream LLM output.
        context_budget: Token budget for context management.
        timeout: Total timeout in seconds (default 600 = 10 min).
        move_timeout: Per-move timeout for LLM call (default 120s).
        max_retries: Retry count for 429/timeout errors (default 1).

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
    messages.append(Message(role=Role.USER, content=user_message))

    tool_defs = registry.all_definitions()

    for turn in range(MAX_TURNS):
        usage.turns = turn + 1

        # Total timeout check
        elapsed = time.monotonic() - t_start
        if elapsed > timeout:
            yield AgentEvent(
                type="system",
                content=f"Timeout ({timeout}s) reached after {elapsed:.1f}s. "
                        f"Returning partial results.",
            )
            break

        # Context compression after turn 3
        if ctx.should_compact(turn):
            ctx.compact_tool_results(messages)

        # Token budget check
        if ctx.is_near_budget(messages):
            yield AgentEvent(type="system", content="Compressing context...")
            messages = ctx.distill(messages)

        # Reserve last 2 turns for synthesis (force no tools)
        remaining = MAX_TURNS - turn
        is_synthesis = remaining <= 2 and turn > 0
        is_final = (turn == MAX_TURNS - 1)

        if is_synthesis and remaining == 2:
            messages.append(Message(
                role=Role.USER,
                content=(
                    "You have gathered enough data. Now synthesize your "
                    "findings into a comprehensive analysis. Do NOT call "
                    "any more tools — provide your final diagnosis, root "
                    "causes, and prioritized recommendations based on "
                    "all the evidence collected above."
                ),
            ))

        # Call LLM with per-move timeout and retry
        t_turn = time.monotonic()
        response = None
        llm_error = None
        for attempt in range(1 + max_retries):
            try:
                response = await asyncio.wait_for(
                    backend.chat_completion(
                        messages=messages,
                        tools=tool_defs if not is_synthesis else None,
                        tool_choice="none" if is_synthesis else "auto",
                        stream=stream,
                        max_tokens=4096,
                        temperature=0.1,
                    ),
                    timeout=move_timeout,
                )
                llm_error = None
                break  # success
            except asyncio.TimeoutError:
                llm_error = "timeout"
                if attempt < max_retries:
                    yield AgentEvent(
                        type="system",
                        content=f"LLM call timed out ({move_timeout}s), "
                                f"retrying ({attempt + 1}/{max_retries})...",
                    )
                    await asyncio.sleep(2)  # brief pause before retry
            except Exception as e:
                err_str = str(e)
                if "429" in err_str and attempt < max_retries:
                    llm_error = "rate_limit"
                    wait = 5 * (attempt + 1)  # 5s, 10s backoff
                    yield AgentEvent(
                        type="system",
                        content=f"LLM rate limited (429), waiting {wait}s "
                                f"before retry ({attempt + 1}/{max_retries})...",
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
            yield AgentEvent(type="done", data=_usage_dict(usage, t_start))
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

        if content:
            yield AgentEvent(type="thinking", content=content)

        for tc in tool_calls:
            usage.tool_calls += 1
            yield AgentEvent(
                type="tool_call",
                content=f"Calling {tc.name}...",
                data={"name": tc.name, "arguments": tc.arguments},
            )

            t_tool = time.monotonic()
            result = await registry.execute(tc.name, tc.arguments)
            tool_elapsed = time.monotonic() - t_tool

            result_str = json.dumps(
                result.data if result.success else {"error": result.error.message},
                ensure_ascii=False,
                default=str,
            )

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
        yield AgentEvent(
            type="system",
            content=f"Turn {turn + 1}: {turn_elapsed:.1f}s "
                    f"(LLM {llm_elapsed:.1f}s + {len(tool_calls)} tools)",
            data={"turn": turn + 1, "elapsed": round(turn_elapsed, 1)},
        )

        _log.debug("Agent at turn %d/%d (%.1fs)", turn + 1, MAX_TURNS, turn_elapsed)

    yield AgentEvent(type="done", data=_usage_dict(usage, t_start))


def _update_usage(usage: AgentUsage, data: dict) -> None:
    """Accumulate token usage from a turn."""
    usage.prompt_tokens += data.get("prompt_tokens", 0)
    usage.completion_tokens += data.get("completion_tokens", 0)


def _usage_dict(usage: AgentUsage, t_start: float | None = None) -> dict:
    """Convert AgentUsage to dict for AgentEvent.data."""
    d = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.prompt_tokens + usage.completion_tokens,
        "turns": usage.turns,
        "tool_calls": usage.tool_calls,
    }
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
