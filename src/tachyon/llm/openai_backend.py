"""OpenAI-compatible LLM backend.

Supports OpenAI, Azure OpenAI, vLLM, Ollama, and any OpenAI-compatible
endpoint via configurable base_url.

Uses the official ``openai`` SDK (async client). The SDK is lazily
imported at construction time so that the rest of the codebase does not
pay for it when a different backend is selected.
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
    StreamChunk,
    ToolCall,
    Usage,
)

if TYPE_CHECKING:
    from ..tools.registry import ToolDefinition

_log = logging.getLogger(__name__)


class OpenAIBackend(LLMBackend):
    """Backend for OpenAI-compatible chat completion APIs.

    Args:
        model: Model identifier (e.g. ``"gpt-4o"``).
        api_key: API key. Falls back to ``"dummy"`` for endpoints
            that don't require auth (e.g. local Ollama).
        base_url: Optional base URL for alternative endpoints
            (vLLM, Ollama, Azure, etc.).
        **kwargs: Forwarded to the ``AsyncOpenAI`` constructor.
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
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAIBackend. "
                "Install it with:  pip install openai"
            ) from exc

        self._client = AsyncOpenAI(
            api_key=api_key or "dummy",
            base_url=base_url,
            **kwargs,
        )

    # ── public interface ────────────────────────────────────────────

    async def chat_completion(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str = "auto",
        stream: bool = False,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> CompletionResponse | AsyncIterator[StreamChunk]:
        """Send a chat-completion request (streaming or non-streaming)."""
        api_messages = [self._format_message(m) for m in messages]

        api_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            api_kwargs["tools"] = self.format_tool_definitions(tools)
            api_kwargs["tool_choice"] = tool_choice

        if stream:
            api_kwargs["stream"] = True
            api_kwargs["stream_options"] = {"include_usage": True}
            response = await self._client.chat.completions.create(**api_kwargs)
            return self._stream_chunks(response)

        response = await self._client.chat.completions.create(**api_kwargs)
        return self._parse_response(response)

    def format_tool_definitions(self, tools: list[ToolDefinition]) -> list[dict]:
        """Convert ToolDefinitions to OpenAI function-calling format."""
        return [tool.to_openai() for tool in tools]

    def parse_tool_calls(self, response: Any) -> list[ToolCall]:
        """Extract ToolCalls from an OpenAI ChatCompletion response."""
        message = response.choices[0].message
        if not message.tool_calls:
            return []
        result: list[ToolCall] = []
        for tc in message.tool_calls:
            try:
                arguments = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                _log.warning(
                    "Failed to parse tool call arguments for %s: %s",
                    tc.function.name,
                    tc.function.arguments,
                )
                arguments = {}
            result.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments,
                )
            )
        return result

    # ── private helpers ─────────────────────────────────────────────

    @staticmethod
    def _format_message(message: Message) -> dict[str, Any]:
        """Convert a :class:`Message` to the dict format expected by the OpenAI SDK."""
        msg: dict[str, Any] = {"role": message.role.value}

        if message.content is not None:
            msg["content"] = message.content

        if message.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in message.tool_calls
            ]

        if message.tool_call_id is not None:
            msg["tool_call_id"] = message.tool_call_id

        if message.name is not None:
            msg["name"] = message.name

        return msg

    def _parse_response(self, response: Any) -> CompletionResponse:
        """Convert an OpenAI ``ChatCompletion`` to :class:`CompletionResponse`."""
        choice = response.choices[0]
        message = choice.message

        tool_calls = self.parse_tool_calls(response)

        usage = Usage()
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return CompletionResponse(
            content=message.content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=choice.finish_reason or "stop",
        )

    @staticmethod
    async def _stream_chunks(response: Any) -> AsyncIterator[StreamChunk]:
        """Yield :class:`StreamChunk` objects from an OpenAI streaming response.

        The ``stream_options={"include_usage": True}`` request parameter causes
        the API to include token usage in the final chunk (where
        ``chunk.usage`` is non-None).
        """
        async for chunk in response:
            # Final chunk may have usage but no choices
            if not chunk.choices:
                usage = None
                if chunk.usage:
                    usage = Usage(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                    )
                if usage:
                    yield StreamChunk(usage=usage)
                continue

            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason

            # Extract text content delta
            content = delta.content if delta.content else None

            # Extract tool-call delta (partial JSON)
            tool_call_delta: dict | None = None
            if delta.tool_calls:
                tc = delta.tool_calls[0]
                tool_call_delta = {
                    "index": tc.index,
                }
                if tc.id:
                    tool_call_delta["id"] = tc.id
                if tc.function:
                    fn: dict[str, str] = {}
                    if tc.function.name:
                        fn["name"] = tc.function.name
                    if tc.function.arguments:
                        fn["arguments"] = tc.function.arguments
                    tool_call_delta["function"] = fn

            # Extract usage from final chunk (when include_usage is set)
            usage = None
            if chunk.usage:
                usage = Usage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                )

            yield StreamChunk(
                content=content,
                tool_call_delta=tool_call_delta,
                finish_reason=finish_reason,
                usage=usage,
            )
