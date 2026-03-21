"""Chat conversation export — data structures, Markdown rendering, and file I/O.

Provides:
    - TurnRecord / ToolCallRecord / UsageRecord — immutable data snapshots
    - ChatHistory — accumulates turn records during a chat session
    - render_turn_markdown / render_all_turns_markdown — pure Markdown output
    - write_export — atomic file write with directory creation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolCallRecord:
    """Single tool invocation within a turn."""
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    summary: str = ""
    elapsed: float = 0.0


@dataclass(frozen=True)
class UsageRecord:
    """Token usage and timing for a single turn."""
    turns: int = 0
    tool_calls: int = 0
    total_tokens: int = 0
    total_elapsed: float = 0.0
    budget: int = 0
    budget_peak: int = 0
    budget_remaining: int = 0
    budget_pct: float = 0.0
    compaction_count: int = 0


@dataclass
class TurnRecord:
    """Complete record of one user-AI exchange."""
    user_input: str = ""
    ai_response: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    usage: UsageRecord | None = None
    timestamp: str = ""

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class ChatHistory:
    """Accumulates turn records for the duration of a chat session."""
    turns: list[TurnRecord] = field(default_factory=list)
    model_info: str = ""

    def record_turn(self, turn: TurnRecord) -> None:
        self.turns.append(turn)

    def last_turn(self) -> TurnRecord | None:
        return self.turns[-1] if self.turns else None

    def clear(self) -> None:
        self.turns.clear()


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def render_turn_markdown(turn: TurnRecord, model_info: str = "") -> str:
    """Render a single turn with full detail (user input, response, tool table, usage)."""
    lines: list[str] = ["# Tachyon Chat Export", ""]

    # Metadata
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines.append(f"> Generated: {now}")
    if model_info:
        lines.append(f"> Model: {model_info}")
    if turn.timestamp:
        lines.append(f"> Turn timestamp: {turn.timestamp}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # User input
    if turn.user_input:
        lines.append("## You")
        lines.append("")
        lines.append(turn.user_input)
        lines.append("")

    # AI response
    lines.append("## Tachyon")
    lines.append("")
    lines.append(turn.ai_response or "*(No response)*")
    lines.append("")

    # Tool calls table
    if turn.has_tool_calls:
        lines.append("### Tool Calls")
        lines.append("")
        lines.append("| # | Tool | Arguments | Status | Summary | Time |")
        lines.append("|---|------|-----------|--------|---------|------|")
        for i, tc in enumerate(turn.tool_calls, 1):
            args_str = _format_args(tc.arguments)
            status = "OK" if tc.success else "FAIL"
            summary = tc.summary[:100] if tc.summary else "-"
            lines.append(
                f"| {i} | `{tc.name}` | {args_str} | {status} "
                f"| {summary} | {tc.elapsed:.2f}s |"
            )
        lines.append("")

    # Usage
    if turn.usage is not None:
        lines.append("### Usage")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        u = turn.usage
        lines.append(f"| Agent turns | {u.turns} |")
        lines.append(f"| Tool calls | {u.tool_calls} |")
        lines.append(f"| Tokens | {u.total_tokens:,} |")
        lines.append(f"| Elapsed | {u.total_elapsed:.1f}s |")
        if u.budget > 0:
            lines.append(
                f"| Context window | {u.budget_peak:,}/{u.budget:,} "
                f"({u.budget_pct:.0f}%) |"
            )
        lines.append("")

    return "\n".join(lines)


def render_all_turns_markdown(
    turns: list[TurnRecord],
    model_info: str = "",
) -> str:
    """Render all turns as AI responses only (no tool details, no user input)."""
    lines: list[str] = [
        "# Tachyon Chat Export (All Turns)",
        "",
        f"> Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
    ]
    if model_info:
        lines.append(f"> Model: {model_info}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for i, turn in enumerate(turns, 1):
        lines.append(f"## Turn {i}")
        lines.append("")
        lines.append(turn.ai_response or "*(No response)*")
        lines.append("")
        if i < len(turns):
            lines.append("---")
            lines.append("")

    return "\n".join(lines)


def _format_args(args: dict[str, Any]) -> str:
    """Format tool arguments for a markdown table cell."""
    if not args:
        return "-"
    parts: list[str] = []
    for k, v in list(args.items())[:3]:
        v_str = str(v)
        if len(v_str) > 30:
            v_str = v_str[:27] + "..."
        parts.append(f"`{k}`={v_str}")
    if len(args) > 3:
        parts.append(f"+{len(args) - 3} more")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def write_export(content: str, path: str | Path) -> Path:
    """Write content to file, creating parent dirs as needed.

    Returns the resolved absolute Path on success.
    Raises OSError on write failure.
    """
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p
