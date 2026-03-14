"""Structured error handling — never crash, always degrade gracefully.

Provides ErrorCode enum, ErrorInfo dataclass, and ToolResult generic
for consistent error handling throughout the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class ErrorCode(str, Enum):
    """Enumerated error codes for structured error handling."""
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    METRIC_NOT_FOUND = "METRIC_NOT_FOUND"
    NO_DEBUG_INFO = "NO_DEBUG_INFO"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    INVALID_BINARY = "INVALID_BINARY"
    OOM = "OOM"
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    ANALYZER_FAILED = "ANALYZER_FAILED"
    UNKNOWN = "UNKNOWN"


@dataclass
class ErrorInfo:
    """Structured error information with actionable suggestion."""
    code: ErrorCode
    message: str
    suggestion: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult(Generic[T]):
    """Generic result type wrapping success/failure with data or error info.

    Usage:
        result = ToolResult.ok(data)
        result = ToolResult.fail(ErrorCode.INVALID_BINARY, "message", "suggestion")
    """
    success: bool
    data: T | None = None
    error: ErrorInfo | None = None

    @classmethod
    def ok(cls, data: T) -> ToolResult[T]:
        """Create a successful result."""
        return cls(success=True, data=data)

    @classmethod
    def fail(
        cls,
        code: ErrorCode,
        message: str,
        suggestion: str = "",
        context: dict[str, Any] | None = None,
    ) -> ToolResult[Any]:
        """Create a failed result with error info."""
        return cls(
            success=False,
            error=ErrorInfo(
                code=code,
                message=message,
                suggestion=suggestion,
                context=context or {},
            ),
        )


# Pre-defined error → suggestion mapping for Rule-Only mode
ERROR_SUGGESTIONS: dict[ErrorCode, str] = {
    ErrorCode.UNSUPPORTED_FORMAT: (
        "Tachyon supports NCU 2023.x-2025.x .ncu-rep files. "
        "Please re-profile with a supported NCU version."
    ),
    ErrorCode.METRIC_NOT_FOUND: (
        "The required metric is not in the report. "
        "Re-profile with --strategy radical to collect full metrics."
    ),
    ErrorCode.NO_DEBUG_INFO: (
        "No DWARF debug info found. "
        "Recompile with -lineinfo for source-level attribution."
    ),
    ErrorCode.LLM_UNAVAILABLE: (
        "LLM is not available. "
        "Running in Rule-Only mode. Set OPENAI_API_KEY or use --no-ai."
    ),
    ErrorCode.TOOL_NOT_FOUND: (
        "NCU tools not found in PATH. "
        "Install CUDA Toolkit or set [tools] in ~/.tachyon/config.toml."
    ),
}
