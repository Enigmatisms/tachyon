"""LLM backend abstraction layer (M3).

Public API:
  - create_backend(provider, model, ...) → LLMBackend
  - Role, Message, ToolCall, Usage, CompletionResponse, StreamChunk
"""
from .backend import (
    CompletionResponse,
    LLMBackend,
    Message,
    Role,
    StreamChunk,
    ToolCall,
    Usage,
    create_backend,
)

__all__ = [
    "CompletionResponse",
    "LLMBackend",
    "Message",
    "Role",
    "StreamChunk",
    "ToolCall",
    "Usage",
    "create_backend",
]
