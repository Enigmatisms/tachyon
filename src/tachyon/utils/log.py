"""Unified logging module for Tachyon.

Wraps Python stdlib logging via structlog for structured output. All existing
``logging.getLogger(__name__)`` calls work unchanged (structlog wraps stdlib).

Environment variables:

- ``TACHYON_LOG_LEVEL``: stderr console level (default: ``WARNING``).
  Values: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``.
- ``TACHYON_DEBUG_RECORD``: debug log file level (default: none).
  Values: ``none``, ``prompt`` (LLM prompts only), ``all`` (prompts + tool calls).
- ``TACHYON_LOG_FORMAT``: console output format (default: ``text``).
  Values: ``text`` (human-readable), ``json`` (structured).

Log files: ``~/.tachyon/debug_logs/agent_<timestamp>.log`` (JSONL when using
structlog, human-readable text for backward compatibility).
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import IO, Any, TextIO

__all__ = [
    "configure_logging",
    "AgentLogger",
    "agent_logger",
    "get_logger",
]

# Re-export for backward compatibility with debug_record.py shim
__all__ += ["init", "close", "level", "log_path", "record_prompt", "record_tool"]


class _L(IntEnum):
    NONE = 0
    PROMPT = 1
    ALL = 2


# ---------------------------------------------------------------------------
# Console logging (via structlog wrapping stdlib)
# ---------------------------------------------------------------------------

_structlog_configured = False


def configure_logging(
    level: int | None = None,
    *,
    force: bool = False,
) -> None:
    """Configure unified logging for the application.

    Sets up stdlib logging handlers and optionally wraps with structlog for
    structured output. Safe to call multiple times (no-op after first call
    unless *force* is True).

    Args:
        level: Override log level. If None, reads ``TACHYON_LOG_LEVEL`` env var
            (default: ``WARNING``).
        force: Re-configure even if already configured.
    """
    global _structlog_configured
    if _structlog_configured and not force:
        return

    if level is None:
        level_name = os.environ.get("TACHYON_LOG_LEVEL", "WARNING").upper()
        level = getattr(logging, level_name, logging.WARNING)

    log_format = os.environ.get("TACHYON_LOG_FORMAT", "text").lower()

    # Configure stdlib root logger
    root = logging.getLogger()
    handlers: list[logging.Handler] = []

    if log_format == "json" or force:
        # Remove existing handlers on force
        root.handlers.clear()

    # Console handler (stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    if log_format == "json":
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    console_handler.setLevel(level)
    handlers.append(console_handler)

    # Configure root logger
    root.setLevel(level)
    if force:
        root.handlers = handlers
    else:
        for h in handlers:
            if not root.handlers:
                root.addHandler(h)

    # Suppress chatty third-party loggers
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(max(level, logging.WARNING))

    # Configure structlog to wrap stdlib
    try:
        import structlog

        shared_processors: list[Any] = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.UnicodeDecoder(),
        ]

        if log_format == "json":
            renderer = structlog.processors.JSONRenderer()
        else:
            renderer = _ColorConsoleRenderer()

        structlog.configure(
            processors=[
                *shared_processors,
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
        )
        console_handler.setFormatter(formatter)

        _structlog_configured = True
    except ImportError:
        # structlog not installed — stdlib logging still works fine
        _structlog_configured = False


def get_logger(name: str) -> logging.Logger:
    """Get a stdlib logger (same as ``logging.getLogger(name)``)."""
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Minimal color console renderer (fallback when structlog available)
# ---------------------------------------------------------------------------

class _ColorConsoleRenderer:
    """Lightweight structlog renderer for colored console output."""

    _LEVEL_COLORS = {
        "debug": "\033[36m",     # cyan
        "info": "\033[32m",      # green
        "warning": "\033[33m",   # yellow
        "error": "\033[31m",     # red
        "critical": "\033[1;31m", # bold red
    }
    _RESET = "\033[0m"

    def __call__(self, logger: Any, name: str, event_dict: dict) -> str:
        level = event_dict.get("level", "info").lower()
        color = self._LEVEL_COLORS.get(level, "")
        ts = event_dict.get("timestamp", "")
        msg = event_dict.get("event", "")
        if ts:
            return f"{color}{level:8s}{self._RESET} {ts}  {msg}"
        return f"{color}{level:8s}{self._RESET}  {msg}"


# ---------------------------------------------------------------------------
# Agent logger (structured JSON for LLM interaction recording)
# ---------------------------------------------------------------------------

class AgentLogger:
    """Structured logger for agent-LLM interactions.

    Replaces the standalone ``debug_record.py`` file-based system with
    structlog-backed JSON output. The public API is identical to the old
    ``debug_record`` module for backward compatibility.

    Usage::

        from tachyon.utils.log import agent_logger

        agent_logger.init()              # call once at startup
        agent_logger.record_prompt(msgs, tools=[...], turn=1)
        agent_logger.record_tool("name", args, result, 0.5, True)
        agent_logger.close()             # or atexit
    """

    def __init__(self) -> None:
        self._level: _L = _L.NONE
        self._fh: IO[str] | None = None
        self._path: Path | None = None
        self._use_structlog: bool = False
        self._json_logger: Any = None  # structlog.BoundLogger

    def init(self) -> None:
        """Read ``TACHYON_DEBUG_RECORD`` and open log file if needed."""
        raw = os.environ.get("TACHYON_DEBUG_RECORD", "").strip().lower()
        if raw in ("prompt", "1"):
            self._level = _L.PROMPT
        elif raw in ("all", "2"):
            self._level = _L.ALL
        else:
            return

        log_dir = Path.home() / ".tachyon" / "debug_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = log_dir / f"agent_{ts}.log"
        self._fh = open(self._path, "a", encoding="utf-8")

        # Try structlog for JSON output; fall back to text
        try:
            import structlog

            self._json_logger = structlog.getLogger("tachyon.agent")
            self._use_structlog = True
        except ImportError:
            self._use_structlog = False

        import atexit
        atexit.register(self.close)
        print(f"[debug] TACHYON_DEBUG_RECORD={raw} -> {self._path}")

    def close(self) -> None:
        """Close the log file handle."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def level(self) -> int:
        """Return current recording level (0=none, 1=prompt, 2=all)."""
        return self._level

    def log_path(self) -> Path | None:
        """Return the path to the current debug log file."""
        return self._path

    def record_prompt(
        self,
        messages: list[Any],
        tools: list[Any] | None = None,
        turn: int = 0,
    ) -> None:
        """Record the exact message list sent to the LLM."""
        if self._level < _L.PROMPT or self._fh is None:
            return
        try:
            if self._use_structlog:
                self._write_prompt_json(messages, tools, turn)
            else:
                _write_prompt_text(self._fh, messages, tools, turn)
        except Exception:
            pass

    def record_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        result: str,
        elapsed: float,
        success: bool,
    ) -> None:
        """Record a tool execution and its result."""
        if self._level < _L.ALL or self._fh is None:
            return
        try:
            if self._use_structlog:
                self._write_tool_json(name, arguments, result, elapsed, success)
            else:
                _write_tool_text(self._fh, name, arguments, result, elapsed, success)
        except Exception:
            pass

    # --- JSON output (structured) ---

    def _write_prompt_json(
        self,
        messages: list[Any],
        tools: list[Any] | None,
        turn: int,
    ) -> None:
        w = self._fh
        assert w is not None
        entry: dict[str, Any] = {
            "event": "PROMPT",
            "turn": turn,
            "messages": len(messages),
        }
        if tools:
            entry["tools"] = len(tools)
        entry["timestamp"] = datetime.now().isoformat()
        w.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Message summary (one line per message)
        for m in messages:
            role = m.role.value if hasattr(m.role, "value") else str(m.role)
            content = m.get("content") if hasattr(m, "get") else getattr(m, "content", None)
            cl = len(content) if content else 0
            msg_entry: dict[str, Any] = {
                "event": "MESSAGE",
                "role": role.upper(),
                "content_chars": cl,
            }
            if hasattr(m, "tool_calls") and m.tool_calls:
                msg_entry["tool_calls"] = [
                    {"name": tc.name, "args_keys": list(tc.arguments.keys()) if isinstance(tc.arguments, dict) else []}
                    for tc in m.tool_calls
                ]
            w.write(json.dumps(msg_entry, ensure_ascii=False) + "\n")
        w.flush()

    def _write_tool_json(
        self,
        name: str,
        arguments: dict[str, Any],
        result: str,
        elapsed: float,
        success: bool,
    ) -> None:
        w = self._fh
        assert w is not None
        entry = {
            "event": "TOOL_CALL",
            "name": name,
            "elapsed": round(elapsed, 3),
            "success": success,
            "args_keys": list(arguments.keys()) if isinstance(arguments, dict) else [],
            "result_chars": len(result),
        }
        w.write(json.dumps(entry, ensure_ascii=False) + "\n")
        w.flush()


# ---------------------------------------------------------------------------
# Text-format fallback writers (backward compatible with old debug_record)
# ---------------------------------------------------------------------------

def _write_prompt_text(
    w: IO[str],
    messages: list[Any],
    tools: list[Any] | None,
    turn: int,
) -> None:
    """Write a PROMPT entry in human-readable text format."""
    tool_info = f"  tools={len(tools)}" if tools else ""
    w.write(f"\n{'='*60}\n")
    w.write(f"PROMPT  turn={turn}  messages={len(messages)}{tool_info}\n")
    w.write(f"{'='*60}\n")

    for m in messages:
        role = m.role.value if hasattr(m.role, "value") else str(m.role)
        content = m.get("content") if hasattr(m, "get") else getattr(m, "content", None)
        cl = len(content) if content else 0
        w.write(f"\n[{role.upper()}] ({cl} chars)")
        if hasattr(m, "tool_calls") and m.tool_calls:
            w.write(f"  tool_calls={len(m.tool_calls)}")
        if hasattr(m, "tool_call_id") and m.tool_call_id:
            w.write(f"  tool_call_id={m.tool_call_id}")
        if hasattr(m, "name") and m.name:
            w.write(f"  name={m.name}")
        w.write("\n")
        if content:
            w.write(content)
            w.write("\n")
        if hasattr(m, "tool_calls") and m.tool_calls:
            import json as _json
            for tc in m.tool_calls:
                args_str = _json.dumps(tc.arguments, ensure_ascii=False)[:200]
                w.write(f"  -> {tc.name}({args_str})\n")
    w.write("\n")
    w.flush()


def _write_tool_text(
    w: IO[str],
    name: str,
    arguments: dict[str, Any],
    result: str,
    elapsed: float,
    success: bool,
) -> None:
    """Write a TOOL entry in human-readable text format."""
    import json as _json
    w.write(f"\n{'-'*60}\n")
    w.write(f"TOOL   name={name}  elapsed={elapsed:.3f}s  ok={success}\n")
    w.write(f"{'-'*60}\n")
    w.write(f"Args: {_json.dumps(arguments, ensure_ascii=False)[:500]}\n")
    w.write(f"Result ({len(result)} chars):\n")
    w.write(result[:5000])
    if len(result) > 5000:
        w.write(f"\n... [truncated, {len(result)} total]")
    w.write("\n")
    w.flush()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

agent_logger = AgentLogger()


# ---------------------------------------------------------------------------
# Backward-compatible module-level functions (shim for debug_record imports)
# ---------------------------------------------------------------------------

def init() -> None:
    """Backward-compatible init: delegates to ``agent_logger.init()``."""
    agent_logger.init()


def close() -> None:
    """Backward-compatible close: delegates to ``agent_logger.close()``."""
    agent_logger.close()


def level() -> int:
    """Backward-compatible level: delegates to ``agent_logger.level()``."""
    return agent_logger.level()


def log_path() -> Path | None:
    """Backward-compatible log_path: delegates to ``agent_logger.log_path()``."""
    return agent_logger.log_path()


def record_prompt(
    messages: list[Any],
    tools: list[Any] | None = None,
    turn: int = 0,
) -> None:
    """Backward-compatible record_prompt."""
    agent_logger.record_prompt(messages, tools=tools, turn=turn)


def record_tool(
    name: str,
    arguments: dict[str, Any],
    result: str,
    elapsed: float,
    success: bool,
) -> None:
    """Backward-compatible record_tool."""
    agent_logger.record_tool(name, arguments, result, elapsed, success)
