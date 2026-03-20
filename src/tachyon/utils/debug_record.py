"""Debug recording for LLM agent interactions.

Controlled by ``TACHYON_DEBUG_RECORD`` env var:

- unset / empty / ``"none"``: no recording (zero overhead — one int check per call site)
- ``"prompt"``: record prompts sent to the LLM (system + user messages, as-is)
- ``"all"``: record prompts + tool calls + tool results

Log file: ``~/.tachyon/debug_logs/agent_<timestamp>.log`` (human-readable).

Usage::

    from tachyon.utils.debug_record import init, record_prompt, record_tool, close

    init()                       # call once at startup
    record_prompt(messages, turn=1)
    record_tool(name, args, result, elapsed)
    close()                      # call at shutdown
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import IO, Any

__all__ = ["init", "close", "level", "log_path", "record_prompt", "record_tool"]


class _L(IntEnum):
    NONE = 0
    PROMPT = 1
    ALL = 2


_level: _L = _L.NONE
_fh: IO[str] | None = None
_path: Path | None = None


def init() -> None:
    """Read ``TACHYON_DEBUG_RECORD`` and open log file if needed."""
    global _level, _fh, _path
    raw = os.environ.get("TACHYON_DEBUG_RECORD", "").strip().lower()
    if raw in ("prompt", "1"):
        _level = _L.PROMPT
    elif raw in ("all", "2"):
        _level = _L.ALL
    else:
        return

    log_dir = Path.home() / ".tachyon" / "debug_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _path = log_dir / f"agent_{ts}.log"
    _fh = open(_path, "a", encoding="utf-8")
    import atexit
    atexit.register(close)
    print(f"[debug] TACHYON_DEBUG_RECORD={raw} -> {_path}")


def close() -> None:
    global _fh
    if _fh is not None:
        _fh.close()
        _fh = None


def level() -> int:
    return _level


def log_path() -> Path | None:
    return _path


# ------------------------------------------------------------------
# Internal: human-readable writer
# ------------------------------------------------------------------

def _safe_flush() -> None:
    if _fh is not None:
        try:
            _fh.flush()
        except Exception:
            pass


def _write_prompt(
    messages: list[Any],
    tools: list[Any] | None,
    turn: int,
) -> None:
    """Write a PROMPT entry with raw message content."""
    w = _fh
    assert w is not None

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
            for tc in m.tool_calls:
                args_str = json.dumps(tc.arguments, ensure_ascii=False)[:200]
                w.write(f"  -> {tc.name}({args_str})\n")

    w.write("\n")
    _safe_flush()


def _write_tool(
    name: str,
    arguments: dict[str, Any],
    result: str,
    elapsed: float,
    success: bool,
) -> None:
    """Write a TOOL entry."""
    w = _fh
    assert w is not None

    w.write(f"\n{'-'*60}\n")
    w.write(f"TOOL   name={name}  elapsed={elapsed:.3f}s  ok={success}\n")
    w.write(f"{'-'*60}\n")
    w.write(f"Args: {json.dumps(arguments, ensure_ascii=False)[:500]}\n")
    w.write(f"Result ({len(result)} chars):\n")
    w.write(result[:5000])
    if len(result) > 5000:
        w.write(f"\n... [truncated, {len(result)} total]")
    w.write("\n")
    _safe_flush()


# ------------------------------------------------------------------
# Public recording API
# ------------------------------------------------------------------

def record_prompt(
    messages: list[Any],
    tools: list[Any] | None = None,
    turn: int = 0,
) -> None:
    """Record the exact message list sent to the LLM.

    Call right before ``backend.chat_completion()``.
    """
    if _level < _L.PROMPT or _fh is None:
        return
    try:
        _write_prompt(messages, tools, turn)
    except Exception:
        pass


def record_tool(
    name: str,
    arguments: dict[str, Any],
    result: str,
    elapsed: float,
    success: bool,
) -> None:
    """Record a tool execution and its result.

    Call right after ``registry.execute()``.
    """
    if _level < _L.ALL or _fh is None:
        return
    try:
        _write_tool(name, arguments, result, elapsed, success)
    except Exception:
        pass
