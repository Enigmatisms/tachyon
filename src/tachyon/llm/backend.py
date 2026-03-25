"""LLM backend abstraction — provider-agnostic interface for chat completions.

Provides:
  - Data models: Role, Message, ToolCall, Usage, CompletionResponse, StreamChunk
  - Abstract base: LLMBackend (chat_completion, format_tool_definitions, parse_tool_calls)
  - Factory: create_backend(config) → LLMBackend

Token-efficiency notes:
  - Message.content is Optional (None when assistant has only tool_calls)
  - Usage tracks prompt/completion/total for budget monitoring
  - StreamChunk is minimal: only non-None fields are populated per chunk
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..tools.registry import ToolDefinition


class Role(str, Enum):
    """Message role in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """A tool invocation requested by the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """A single message in a conversation."""
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # for Role.TOOL responses
    name: str | None = None          # tool name for Role.TOOL responses


@dataclass
class Usage:
    """Token usage statistics for a completion."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class CompletionResponse:
    """Full (non-streaming) response from an LLM."""
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = "stop"  # "stop" | "tool_calls" | "length"


@dataclass
class StreamChunk:
    """A single chunk from a streaming response."""
    content: str | None = None           # text delta
    tool_call_delta: dict | None = None  # partial tool call JSON
    finish_reason: str | None = None
    usage: Usage | None = None           # present in final chunk


class LLMBackend(ABC):
    """Abstract LLM backend. All provider-specific logic lives in subclasses."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    async def chat_completion(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str = "auto",
        stream: bool = False,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> CompletionResponse | AsyncIterator[StreamChunk]:
        """Send a chat completion request.

        When stream=False, returns CompletionResponse.
        When stream=True, returns an async iterator of StreamChunk.
        """
        ...

    @abstractmethod
    def format_tool_definitions(self, tools: list[ToolDefinition]) -> Any:
        """Convert Tachyon ToolDefinitions to provider-specific format."""
        ...

    @abstractmethod
    def parse_tool_calls(self, response: Any) -> list[ToolCall]:
        """Extract ToolCalls from a provider-specific response object."""
        ...


def create_backend(
    provider: str,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> LLMBackend:
    """Create an LLMBackend from configuration.

    Args:
        provider: Backend provider name ("openai", "anthropic").
        model: Model identifier (e.g. "gpt-4o", "claude-sonnet-4-20250514").
        api_key: API key (if None, backends read from env).
        base_url: Optional base URL for OpenAI-compatible endpoints.

    Raises:
        ValueError: Unknown provider name.
        ImportError: Provider SDK not installed.
    """
    match provider:
        case "openai":
            from .openai_backend import OpenAIBackend
            return OpenAIBackend(model=model, api_key=api_key, base_url=base_url, **kwargs)
        case "anthropic":
            from .anthropic_backend import AnthropicBackend
            return AnthropicBackend(model=model, api_key=api_key, base_url=base_url, **kwargs)
        case _:
            raise ValueError(
                f"Unknown LLM provider: {provider!r}. "
                f"Supported: openai, anthropic"
            )
