"""LiteLLM backend — unified interface to 100+ LLM providers.

LiteLLM auto-detects the provider from the model prefix:
  - gpt-4o, gpt-4-turbo  -> OpenAI
  - claude-xxx            -> Anthropic
  - gemini/xxx            -> Google
  - command-r-plus        -> Cohere
  - together_ai/xxx       -> Together AI
  - etc.

Uses OpenAI message format and OpenAI tool format for all providers.
The litellm library is lazily imported on first use so that Tachyon
can start without litellm installed (falling back to native backends).
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


def _role_to_str(role: Role) -> str:
    """Convert a Role enum to the OpenAI string representation."""
    return role.value


def _message_to_dict(msg: Message) -> dict[str, Any]:
    """Serialize a Tachyon Message to the OpenAI dict format used by LiteLLM."""
    d: dict[str, Any] = {"role": _role_to_str(msg.role)}

    if msg.content is not None:
        d["content"] = msg.content
    elif msg.role != Role.ASSISTANT:
        # OpenAI format requires content for non-assistant messages;
        # assistant messages may omit content when tool_calls are present.
        d["content"] = ""

    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                },
            }
            for tc in msg.tool_calls
        ]

    if msg.tool_call_id is not None:
        d["tool_call_id"] = msg.tool_call_id

    if msg.name is not None:
        d["name"] = msg.name

    return d


class LiteLLMBackend(LLMBackend):
    """LLM backend powered by LiteLLM's unified API.

    LiteLLM translates OpenAI-format requests into provider-native calls,
    enabling a single backend to reach OpenAI, Anthropic, Google, Cohere,
    Together AI, Replicate, and 100+ other providers.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)
        # Extra kwargs are forwarded to litellm.acompletion as default overrides.
        self._extra_kwargs: dict[str, Any] = kwargs
        # litellm module reference, lazily loaded.
        self._litellm: Any | None = None

    # ------------------------------------------------------------------
    # Lazy import
    # ------------------------------------------------------------------

    def _ensure_litellm(self) -> Any:
        """Import and cache the litellm module. Raises ImportError if missing."""
        if self._litellm is not None:
            return self._litellm
        try:
            import litellm  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "litellm is required for the LiteLLM backend. "
                "Install it with: pip install litellm"
            ) from None
        self._litellm = litellm
        return litellm

    # ------------------------------------------------------------------
    # Public API
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
        """Send a chat completion via litellm.acompletion.

        When *stream=False*, returns a :class:`CompletionResponse`.
        When *stream=True*, returns an async iterator of :class:`StreamChunk`.
        """
        litellm = self._ensure_litellm()

        oai_messages = [_message_to_dict(m) for m in messages]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            **self._extra_kwargs,
        }

        # LiteLLM uses api_key / api_base (NOT base_url).
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        if self.base_url is not None:
            kwargs["api_base"] = self.base_url

        # Tools
        if tools:
            kwargs["tools"] = self.format_tool_definitions(tools)
            kwargs["tool_choice"] = tool_choice

        _log.debug(
            "litellm.acompletion model=%s stream=%s tools=%d",
            self.model,
            stream,
            len(tools) if tools else 0,
        )

        response = await litellm.acompletion(**kwargs)

        if stream:
            return self._stream_chunks(response)
        return self._parse_response(response)

    def format_tool_definitions(self, tools: list[ToolDefinition]) -> list[dict]:
        """Convert Tachyon ToolDefinitions to OpenAI function-calling format.

        LiteLLM expects the same JSON schema that OpenAI uses::

            [
                {
                    "type": "function",
                    "function": {
                        "name": "...",
                        "description": "...",
                        "parameters": { ... }
                    }
                },
                ...
            ]
        """
        return [tool.to_openai() for tool in tools]

    def parse_tool_calls(self, response: Any) -> list[ToolCall]:
        """Extract ToolCalls from an OpenAI-style completion response.

        *response* is the raw object returned by ``litellm.acompletion``
        (which mirrors the OpenAI ``ChatCompletion`` schema).  The method
        walks ``response.choices[0].message.tool_calls`` and returns
        a list of :class:`ToolCall` instances.
        """
        tool_calls: list[ToolCall] = []

        choices = getattr(response, "choices", None) or []
        if not choices:
            return tool_calls

        message = choices[0].message
        raw_calls = getattr(message, "tool_calls", None)
        if not raw_calls:
            return tool_calls

        for tc in raw_calls:
            func = tc.function
            name = func.name
            raw_args = func.arguments

            # arguments may be a JSON string or an already-parsed dict.
            if isinstance(raw_args, str):
                try:
                    arguments = json.loads(raw_args)
                except (json.JSONDecodeError, TypeError):
                    _log.warning(
                        "Failed to parse tool call arguments for %s: %r",
                        name,
                        raw_args,
                    )
                    arguments = {}
            elif isinstance(raw_args, dict):
                arguments = raw_args
            else:
                arguments = {}

            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=name,
                    arguments=arguments,
                )
            )

        return tool_calls

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_response(self, response: Any) -> CompletionResponse:
        """Convert a non-streaming LiteLLM response to CompletionResponse."""
        choice = response.choices[0]
        message = choice.message

        # Content
        content = getattr(message, "content", None)

        # Tool calls
        tool_calls = self.parse_tool_calls(response)

        # Finish reason — normalize to Tachyon convention.
        raw_finish = getattr(choice, "finish_reason", "stop") or "stop"
        finish_reason = self._normalize_finish_reason(raw_finish)

        # Usage
        raw_usage = getattr(response, "usage", None)
        usage = self._parse_usage(raw_usage)

        return CompletionResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
        )

    async def _stream_chunks(self, response: Any) -> AsyncIterator[StreamChunk]:
        """Yield StreamChunks from an async streaming response.

        LiteLLM streaming mirrors the OpenAI SSE format:
        each chunk has ``choices[0].delta`` with optional content/tool_calls,
        plus an optional ``usage`` in the final chunk.
        """
        # Accumulate partial tool call fragments keyed by index.
        tool_call_buffers: dict[int, dict[str, Any]] = {}

        async for chunk in response:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                # Final chunk may carry only usage.
                raw_usage = getattr(chunk, "usage", None)
                if raw_usage:
                    yield StreamChunk(usage=self._parse_usage(raw_usage))
                continue

            delta = choices[0].delta
            finish_reason_raw = getattr(choices[0], "finish_reason", None)
            finish_reason = (
                self._normalize_finish_reason(finish_reason_raw)
                if finish_reason_raw
                else None
            )

            # --- Text content delta ---
            content = getattr(delta, "content", None)

            # --- Tool call deltas ---
            tool_call_delta: dict | None = None
            raw_tc_list = getattr(delta, "tool_calls", None)
            if raw_tc_list:
                for tc_delta in raw_tc_list:
                    idx = getattr(tc_delta, "index", 0)
                    func_delta = getattr(tc_delta, "function", None)

                    if idx not in tool_call_buffers:
                        tool_call_buffers[idx] = {
                            "id": getattr(tc_delta, "id", None),
                            "name": "",
                            "arguments": "",
                        }

                    buf = tool_call_buffers[idx]

                    # ID is usually only in the first delta for a given index.
                    tc_id = getattr(tc_delta, "id", None)
                    if tc_id:
                        buf["id"] = tc_id

                    if func_delta:
                        name_part = getattr(func_delta, "name", None)
                        if name_part:
                            buf["name"] += name_part

                        args_part = getattr(func_delta, "arguments", None)
                        if args_part:
                            buf["arguments"] += args_part

                    # Emit a delta for the caller to display progress.
                    tool_call_delta = {
                        "index": idx,
                        "id": buf["id"],
                        "name": buf["name"],
                        "arguments_delta": (
                            getattr(func_delta, "arguments", "")
                            if func_delta
                            else ""
                        ),
                    }

            # --- Usage (some providers include it in the final chunk) ---
            raw_usage = getattr(chunk, "usage", None)
            usage = self._parse_usage(raw_usage) if raw_usage else None

            yield StreamChunk(
                content=content,
                tool_call_delta=tool_call_delta,
                finish_reason=finish_reason,
                usage=usage,
            )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_usage(raw: Any) -> Usage:
        """Extract a Usage from a raw usage object (or dict)."""
        if raw is None:
            return Usage()
        if isinstance(raw, dict):
            return Usage(
                prompt_tokens=raw.get("prompt_tokens", 0),
                completion_tokens=raw.get("completion_tokens", 0),
                total_tokens=raw.get("total_tokens", 0),
            )
        return Usage(
            prompt_tokens=getattr(raw, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(raw, "completion_tokens", 0) or 0,
            total_tokens=getattr(raw, "total_tokens", 0) or 0,
        )

    @staticmethod
    def _normalize_finish_reason(raw: str) -> str:
        """Map provider finish reasons to Tachyon convention.

        Tachyon recognizes: "stop", "tool_calls", "length".
        OpenAI/LiteLLM may return "stop", "tool_calls", "length",
        "content_filter", or "function_call" (legacy).
        """
        mapping: dict[str, str] = {
            "stop": "stop",
            "tool_calls": "tool_calls",
            "function_call": "tool_calls",  # legacy OpenAI
            "length": "length",
        }
        return mapping.get(raw, raw)
