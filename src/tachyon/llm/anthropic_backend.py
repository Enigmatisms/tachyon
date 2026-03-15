"""Anthropic Claude backend for Tachyon LLM integration.

Implements the LLMBackend interface using Anthropic's Messages API.

Key Anthropic-specific behaviors:
  - System prompt is a top-level parameter, not a message role
  - Tool use results are user-role messages with tool_result content blocks
  - Assistant tool calls use tool_use content blocks
  - Usage reports input_tokens / output_tokens (not prompt_tokens / completion_tokens)
  - stop_reason "tool_use" maps to our "tool_calls" finish_reason
  - Streaming uses client.messages.stream() context manager
"""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from .backend import (
    CompletionResponse,
    LLMBackend,
    Message,
    Role,
    StreamChunk,
    ToolCall,
    Usage,
)

if TYPE_CHECKING:
    from ..tools.registry import ToolDefinition

_log = logging.getLogger(__name__)


def _split_system(
    messages: list[Message],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Split system message and convert remaining messages to Anthropic format.

    Anthropic's Messages API takes system as a top-level string parameter,
    not as a message in the messages list.  This helper:

    1. Extracts the system message content (if any) as a separate string.
    2. Converts Role.TOOL messages to user-role messages with ``tool_result``
       content blocks (Anthropic requires tool results in this format).
    3. Converts assistant messages that carry tool_calls into content blocks
       with ``tool_use`` entries (so the conversation round-trips correctly).
    4. Passes through plain user / assistant messages unchanged.

    Returns:
        (system_text, anthropic_messages)
    """
    system_text: str | None = None
    converted: list[dict[str, Any]] = []

    for msg in messages:
        # ---- system ----
        if msg.role is Role.SYSTEM:
            system_text = msg.content
            continue

        # ---- tool result (Role.TOOL → user with tool_result block) ----
        if msg.role is Role.TOOL:
            converted.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content or "",
                    },
                ],
            })
            continue

        # ---- assistant with tool_calls → tool_use content blocks ----
        if msg.role is Role.ASSISTANT and msg.tool_calls:
            content_blocks: list[dict[str, Any]] = []
            # If the assistant also produced text, include it first.
            if msg.content:
                content_blocks.append({"type": "text", "text": msg.content})
            for tc in msg.tool_calls:
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })
            converted.append({
                "role": "assistant",
                "content": content_blocks,
            })
            continue

        # ---- plain user / assistant ----
        converted.append({
            "role": msg.role.value,
            "content": msg.content or "",
        })

    return system_text, converted


class AnthropicBackend(LLMBackend):
    """LLM backend powered by Anthropic's Messages API (Claude models).

    The ``anthropic`` SDK is lazily imported so the rest of Tachyon does not
    hard-depend on it.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)

        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for the Anthropic backend. "
                "Install it with: pip install anthropic"
            ) from None

        client_kwargs: dict[str, Any] = {}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if base_url is not None:
            client_kwargs["base_url"] = base_url

        # Support ducc environment: inject custom headers from env
        import os
        custom_headers_env = os.environ.get("ANTHROPIC_CUSTOM_HEADERS")
        if custom_headers_env:
            # Format: "key:value" — parse and inject as default_headers
            parts = custom_headers_env.split(":", 1)
            if len(parts) == 2:
                client_kwargs.setdefault("default_headers", {})
                client_kwargs["default_headers"][parts[0]] = parts[1]

        self._client = anthropic.AsyncAnthropic(**client_kwargs)
        self._anthropic = anthropic  # keep module reference for type checks

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str = "auto",
        stream: bool = False,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> CompletionResponse | AsyncIterator[StreamChunk]:
        """Send a chat completion request to the Anthropic Messages API."""
        system_text, api_messages = _split_system(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system_text is not None:
            kwargs["system"] = system_text

        if tools:
            kwargs["tools"] = self.format_tool_definitions(tools)
            kwargs["tool_choice"] = self._map_tool_choice(tool_choice)

        if stream:
            return self._stream(kwargs)
        return await self._complete(kwargs)

    def format_tool_definitions(self, tools: list[ToolDefinition]) -> list[dict]:
        """Convert Tachyon ToolDefinitions to Anthropic tool_use format."""
        return [tool.to_anthropic() for tool in tools]

    def parse_tool_calls(self, response: Any) -> list[ToolCall]:
        """Extract ToolCalls from an Anthropic Message response object.

        Anthropic returns tool use as content blocks with type ``tool_use``.
        """
        tool_calls: list[ToolCall] = []
        if not hasattr(response, "content"):
            return tool_calls
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )
        return tool_calls

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _map_tool_choice(choice: str) -> dict[str, str]:
        """Map Tachyon's tool_choice string to Anthropic's dict format.

        Anthropic uses ``{"type": "auto"}``, ``{"type": "any"}``, or
        ``{"type": "tool", "name": "<name>"}``.
        """
        if choice in ("auto", "none"):
            return {"type": choice}
        if choice == "required":
            # "required" = must call at least one tool → Anthropic "any"
            return {"type": "any"}
        # Assume it is a specific tool name.
        return {"type": "tool", "name": choice}

    @staticmethod
    def _map_finish_reason(stop_reason: str | None) -> str:
        """Map Anthropic stop_reason to Tachyon's finish_reason vocabulary."""
        mapping = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
        }
        return mapping.get(stop_reason or "", "stop")

    async def _complete(self, kwargs: dict[str, Any]) -> CompletionResponse:
        """Non-streaming completion."""
        response = await self._client.messages.create(**kwargs)

        # Extract text content and tool calls from content blocks.
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
            elif getattr(block, "type", None) == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        content = "\n".join(text_parts) if text_parts else None

        usage = Usage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        return CompletionResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=self._map_finish_reason(response.stop_reason),
        )

    async def _stream(self, kwargs: dict[str, Any]) -> AsyncIterator[StreamChunk]:
        """Streaming completion using Anthropic's async stream context manager.

        Yields StreamChunk objects as events arrive.  The final chunk carries
        aggregated Usage when available.
        """
        return _AnthropicStreamIterator(self._client, kwargs, self._map_finish_reason)


class _AnthropicStreamIterator:
    """Async iterator adapter for Anthropic's streaming Messages API.

    Wraps ``client.messages.stream()`` into an ``AsyncIterator[StreamChunk]``
    that the agent loop can consume with ``async for chunk in stream:``.

    Anthropic stream events (simplified):
      - message_start          → contains usage.input_tokens
      - content_block_start    → new text or tool_use block
      - content_block_delta    → text delta or tool input JSON delta
      - content_block_stop     → block finished
      - message_delta          → stop_reason, output_tokens
      - message_stop           → end of stream
    """

    def __init__(
        self,
        client: Any,
        kwargs: dict[str, Any],
        finish_reason_mapper: Any,
    ) -> None:
        self._client = client
        self._kwargs = kwargs
        self._finish_reason_mapper = finish_reason_mapper

        # Stream state -- initialized lazily in __aiter__.
        self._stream_cm: Any = None  # the context-manager object
        self._stream: Any = None     # the entered stream

        # Accumulate tool_use input JSON fragments per block index.
        self._tool_blocks: dict[int, dict[str, Any]] = {}
        self._current_block_idx: int = -1
        self._input_tokens: int = 0
        self._output_tokens: int = 0

    def __aiter__(self) -> _AnthropicStreamIterator:
        return self

    async def __anext__(self) -> StreamChunk:
        # Lazily enter the stream context manager on first iteration.
        if self._stream is None:
            self._stream_cm = self._client.messages.stream(**self._kwargs)
            self._stream = await self._stream_cm.__aenter__()
            self._event_iter = self._stream.__aiter__()

        while True:
            try:
                event = await self._event_iter.__anext__()
            except StopAsyncIteration:
                # Clean up the context manager.
                if self._stream_cm is not None:
                    try:
                        await self._stream_cm.__aexit__(None, None, None)
                    except Exception:
                        pass
                raise

            chunk = self._process_event(event)
            if chunk is not None:
                return chunk

    def _process_event(self, event: Any) -> StreamChunk | None:
        """Convert a single Anthropic stream event to a StreamChunk (or None to skip)."""
        event_type = getattr(event, "type", "")

        # ---- message_start: carries input token count ----
        if event_type == "message_start":
            message = getattr(event, "message", None)
            if message and hasattr(message, "usage"):
                self._input_tokens = getattr(message.usage, "input_tokens", 0)
            return None

        # ---- content_block_start: new text or tool_use block ----
        if event_type == "content_block_start":
            self._current_block_idx = getattr(event, "index", self._current_block_idx + 1)
            block = getattr(event, "content_block", None)
            if block and getattr(block, "type", None) == "tool_use":
                self._tool_blocks[self._current_block_idx] = {
                    "id": block.id,
                    "name": block.name,
                    "input_json": "",
                }
                return StreamChunk(
                    tool_call_delta={
                        "id": block.id,
                        "name": block.name,
                        "arguments": "",
                    }
                )
            return None

        # ---- content_block_delta: incremental text or tool input ----
        if event_type == "content_block_delta":
            delta = getattr(event, "delta", None)
            if delta is None:
                return None

            delta_type = getattr(delta, "type", "")

            if delta_type == "text_delta":
                return StreamChunk(content=getattr(delta, "text", ""))

            if delta_type == "input_json_delta":
                partial_json = getattr(delta, "partial_json", "")
                idx = getattr(event, "index", self._current_block_idx)
                if idx in self._tool_blocks:
                    self._tool_blocks[idx]["input_json"] += partial_json
                return StreamChunk(
                    tool_call_delta={
                        "arguments_delta": partial_json,
                    }
                )

            return None

        # ---- content_block_stop: finalize tool call if applicable ----
        if event_type == "content_block_stop":
            idx = getattr(event, "index", self._current_block_idx)
            if idx in self._tool_blocks:
                block_info = self._tool_blocks[idx]
                raw_json = block_info["input_json"]
                try:
                    parsed = json.loads(raw_json) if raw_json else {}
                except json.JSONDecodeError:
                    _log.warning(
                        "Failed to parse tool input JSON for %s: %s",
                        block_info["name"],
                        raw_json[:200],
                    )
                    parsed = {}
                return StreamChunk(
                    tool_call_delta={
                        "id": block_info["id"],
                        "name": block_info["name"],
                        "arguments": parsed,
                        "complete": True,
                    }
                )
            return None

        # ---- message_delta: stop_reason + output tokens ----
        if event_type == "message_delta":
            delta = getattr(event, "delta", None)
            stop_reason = getattr(delta, "stop_reason", None) if delta else None
            usage_obj = getattr(event, "usage", None)
            if usage_obj:
                self._output_tokens = getattr(usage_obj, "output_tokens", 0)

            finish = self._finish_reason_mapper(stop_reason) if stop_reason else None
            total = self._input_tokens + self._output_tokens

            return StreamChunk(
                finish_reason=finish,
                usage=Usage(
                    prompt_tokens=self._input_tokens,
                    completion_tokens=self._output_tokens,
                    total_tokens=total,
                ) if (self._input_tokens or self._output_tokens) else None,
            )

        # ---- message_stop: end of stream ----
        if event_type == "message_stop":
            # Don't yield anything; iteration ends on StopAsyncIteration.
            return None

        return None
