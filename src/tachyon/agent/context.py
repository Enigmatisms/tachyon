"""Context window management — token-aware tiered compression with LLM-assisted summarization.

Compression tiers (triggered by token usage as % of budget):
  Tier 0 (<70%): No compression.
  Tier 1 (>=70%): Compress OTHER-value tool results (list_source_files, etc.).
  Tier 2 (>=80%): Compress META-value tool results (get_kernel_metrics, etc.).
  Tier 3 (>=95%): LLM-assisted summarization of findings so far.
  Tier 4 (100%): Emergency — compress SOURCE+ANALYSIS results (truncate only).

SOURCE-value results are never touched until Tier 4.  Each tier is applied
at most once (irreversible) to prevent progressive information loss.

Token estimation: ~4 chars/token for English/code, ~3 chars/token for system prompts.
When the LLM reports actual token counts via update_token_count(), those take
precedence over heuristics.
"""
from __future__ import annotations

import asyncio
import json
import logging
from enum import IntEnum
from typing import TYPE_CHECKING, Any

from ..llm.backend import Message, Role

if TYPE_CHECKING:
    from ..llm.backend import LLMBackend

_log = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_TOKEN_BUDGET = 120_000
DEEP_TOKEN_BUDGET = 200_000  # used by staged analysis for larger context
COMPACT_THRESHOLD = 200       # chars — below this, tool results are kept as-is
COMPACT_SUMMARY_LEN = 200     # chars for non-preserved tool summaries
PRESERVE_SUMMARY_LEN = 2000   # chars for preserved tool summaries
_LLM_SUMMARY_TIMEOUT = 30     # seconds — timeout for the summarization LLM call


class ToolValue(IntEnum):
    """Value level of a tool result — higher = more important to preserve."""
    SOURCE = 4    # read_source_file, get_sass_for_source_line
    ANALYSIS = 3  # get_stall_analysis_for_line, get_performance_hotspots,
                  #   get_source_hotspots, run_analysis
    META = 2      # get_optimization_tree, get_kernel_metrics, get_kernel_summary,
                  #   list_kernels
    OTHER = 1     # list_source_files, get_ncu_rule_results, everything else


# Map tool names to their value level.
_TOOL_VALUE_MAP: dict[str, ToolValue] = {
    "read_source_file": ToolValue.SOURCE,
    "get_sass_for_source_line": ToolValue.SOURCE,
    "get_stall_analysis_for_line": ToolValue.ANALYSIS,
    "get_performance_hotspots": ToolValue.ANALYSIS,
    "get_source_hotspots": ToolValue.ANALYSIS,
    "run_analysis": ToolValue.ANALYSIS,
    "get_optimization_tree": ToolValue.META,
    "get_kernel_metrics": ToolValue.META,
    "get_kernel_summary": ToolValue.META,
    "list_kernels": ToolValue.META,
}


def _tool_value(name: str | None) -> ToolValue:
    """Return the value level for a tool name, defaulting to OTHER."""
    if name is None:
        return ToolValue.OTHER
    return _TOOL_VALUE_MAP.get(name, ToolValue.OTHER)


# --- Tier thresholds (percentage of budget) ---
_TIER_THRESHOLDS = [70, 80, 95, 100]  # index 0 → tier 1, etc.


class ContextManager:
    """Manages conversation context to fit within LLM token limits.

    Tracks actual token counts from API responses when available, falling
    back to heuristic estimation.  Compression is tiered and token-driven
    rather than turn-count-driven.
    """

    def __init__(self, budget: int = DEFAULT_TOKEN_BUDGET) -> None:
        self.budget = budget
        self.current_tokens: int = 0
        self._max_tier_applied: int = 0  # highest tier already applied (0 = none)
        self._compaction_count: int = 0

    # ------------------------------------------------------------------
    # Token tracking
    # ------------------------------------------------------------------

    def update_token_count(self, prompt_tokens: int) -> None:
        """Update with the real prompt token count from an API response."""
        self.current_tokens = prompt_tokens

    def get_budget_pct(self, messages: list[Message]) -> float:
        """Return current token usage as a percentage of the budget."""
        tokens = (
            self.current_tokens
            if self.current_tokens > 0
            else self.estimate_tokens(messages)
        )
        return (tokens / self.budget) * 100 if self.budget > 0 else 0.0

    def estimate_tokens(self, messages: list[Message]) -> int:
        """Estimate total tokens in message list.

        Heuristic: system=len/3, tool/code=len/4, natural=len/4.
        """
        total = 0
        for msg in messages:
            content = msg.content or ""
            weight = 3 if msg.role == Role.SYSTEM else 4
            total += len(content) // weight
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    total += len(json.dumps(tc.arguments)) // 4
        return total

    # ------------------------------------------------------------------
    # Tier assessment
    # ------------------------------------------------------------------

    def needs_compaction(self) -> int:
        """Return the compression tier (0-4) based on current token usage.

        Returns 0 if no real token count is available yet.
        """
        if self.current_tokens <= 0 or self.budget <= 0:
            return 0
        pct = (self.current_tokens / self.budget) * 100
        tier = 0
        for i, threshold in enumerate(_TIER_THRESHOLDS, start=1):
            if pct >= threshold:
                tier = i
        return tier

    # ------------------------------------------------------------------
    # Unified compaction
    # ------------------------------------------------------------------

    async def compact(
        self, messages: list[Message], backend: LLMBackend | None = None,
    ) -> list[tuple[int, str]]:
        """Apply tiered compression based on current token usage.

        Returns a list of ``(tier, description)`` pairs for event emission.
        Mutates *messages* in-place.
        """
        tier = self.needs_compaction()
        if tier == 0:
            return []

        actions: list[tuple[int, str]] = []

        if tier >= 1 and self._max_tier_applied < 1:
            _compress_by_value(messages, max_value=ToolValue.OTHER, keep_last_n=3)
            self._mark_tier(1)
            actions.append((1, "Compressed low-value tool results (list_source_files etc.)"))

        if tier >= 2 and self._max_tier_applied < 2:
            _compress_by_value(messages, max_value=ToolValue.META, keep_last_n=3)
            self._mark_tier(2)
            actions.append((2, "Compressed meta tool results (get_kernel_metrics etc.)"))

        if tier >= 3 and self._max_tier_applied < 3:
            before = len(messages)
            if backend:
                try:
                    await _llm_summarize_and_replace(messages, backend)
                    self._mark_tier(3)
                    actions.append(
                        (3, f"LLM-assisted summarization ({before} → {len(messages)} messages)")
                    )
                except Exception:
                    _log.warning("LLM summarization failed, falling back to rule compression")
                    _compress_by_value(messages, max_value=ToolValue.ANALYSIS, keep_last_n=2)
                    self._mark_tier(3)
                    actions.append((3, "LLM summarization failed, compressed analysis results"))
            else:
                _compress_by_value(messages, max_value=ToolValue.ANALYSIS, keep_last_n=2)
                self._mark_tier(3)
                actions.append((3, "Compressed analysis results (no LLM backend)"))

        if tier >= 4 and self._max_tier_applied < 4:
            _compress_by_value(messages, max_value=ToolValue.SOURCE, keep_last_n=1)
            self._mark_tier(4)
            actions.append((4, "Emergency: compressed source/analysis results"))

        return actions

    def _mark_tier(self, tier: int) -> None:
        """Mark a tier as applied."""
        self._max_tier_applied = max(self._max_tier_applied, tier)
        self._compaction_count += 1

    @property
    def compaction_count(self) -> int:
        """Number of compression tiers applied so far."""
        return self._compaction_count


# ======================================================================
# Module-level helpers
# ======================================================================

def _compress_by_value(
    messages: list[Message],
    max_value: ToolValue,
    keep_last_n: int = 3,
) -> None:
    """Compress tool results whose value <= *max_value*, except the last *keep_last_n*."""
    tool_indices = [
        i for i, m in enumerate(messages)
        if m.role == Role.TOOL and _tool_value(m.name) <= max_value
    ]
    if len(tool_indices) <= keep_last_n:
        return
    cutoff_indices = set(tool_indices[:-keep_last_n])
    for i in cutoff_indices:
        msg = messages[i]
        if not msg.content or len(msg.content) <= COMPACT_THRESHOLD:
            continue
        tool_name = msg.name or "unknown"
        tv = _tool_value(tool_name)
        limit = PRESERVE_SUMMARY_LEN if tv >= ToolValue.ANALYSIS else COMPACT_SUMMARY_LEN
        messages[i] = Message(
            role=Role.TOOL,
            content=_smart_truncate(msg.content, tool_name, limit),
            tool_call_id=msg.tool_call_id,
            name=msg.name,
        )


def _smart_truncate(content: str, tool_name: str, limit: int) -> str:
    """Intelligently truncate content, preserving structure.

    JSON content: parse and extract key-value pairs.
    Plain text: keep first paragraph.
    """
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            kept: dict[str, Any] = {}
            for k, v in list(data.items()):
                kept[k] = json.dumps(v, ensure_ascii=False)[:100]
            return f"[{tool_name} (compressed): {json.dumps(kept, ensure_ascii=False)[:limit]}]"
    except (json.JSONDecodeError, TypeError):
        pass
    first_para = content.split("\n\n")[0]
    return f"[{tool_name} (compressed): {first_para[:limit]}]"


def _find_turn_boundary(messages: list[Message], keep_turns: int) -> int:
    """Find the split index so that the tail keeps the last *keep_turns* complete turns.

    A "turn" is a sequence: ASSISTANT (optionally with tool_calls) → [TOOL*].
    We scan backwards to find the start of the earliest complete turn to keep,
    ensuring no orphan TOOL messages (without a preceding ASSISTANT+tool_calls).
    """
    if keep_turns <= 0 or len(messages) == 0:
        return len(messages)

    turn_starts: list[int] = []
    i = len(messages) - 1
    # Scan backwards, finding the start of each complete turn
    while i >= 0 and len(turn_starts) < keep_turns:
        msg = messages[i]
        if msg.role == Role.TOOL:
            # Skip all consecutive TOOL messages
            while i >= 0 and messages[i].role == Role.TOOL:
                i -= 1
            # Now i points to the ASSISTANT message (or before)
            if i >= 0 and messages[i].role == Role.ASSISTANT:
                turn_starts.append(i)
                i -= 1
            else:
                # Malformed — orphan TOOL, stop here
                break
        elif msg.role == Role.ASSISTANT:
            # Text-only assistant (no tools) — a complete turn by itself
            turn_starts.append(i)
            i -= 1
        else:
            # USER or SYSTEM — not part of a backward turn scan
            break

    if not turn_starts:
        return len(messages)
    return turn_starts[-1]


async def _llm_summarize_and_replace(
    messages: list[Message],
    backend: LLMBackend,
) -> None:
    """Use the LLM to generate a concise summary of findings, replacing old messages.

    Extracts key data from SOURCE and ANALYSIS tool results, asks the LLM to
    produce a findings summary, then replaces old messages with the summary
    while keeping the last 2 complete turns intact (preserving message structure).
    """
    if len(messages) <= 4:
        return

    # Extract key findings from SOURCE/ANALYSIS tool results
    findings_parts: list[str] = []
    for msg in messages:
        if msg.role != Role.TOOL or _tool_value(msg.name) < ToolValue.ANALYSIS:
            continue
        content = msg.content or ""
        if not content:
            continue
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                summary_dict = {k: json.dumps(v, ensure_ascii=False)[:300]
                                for k, v in list(data.items())[:10]}
                findings_parts.append(
                    f"Tool: {msg.name}\n{json.dumps(summary_dict, ensure_ascii=False)[:500]}"
                )
                continue
        except (json.JSONDecodeError, TypeError):
            pass
        findings_parts.append(f"Tool: {msg.name}\n{content[:500]}")

    if not findings_parts:
        return

    summary_prompt = (
        "Based on the tool results below, write a concise summary of all key "
        "findings. Focus on root causes, performance bottlenecks, and evidence. "
        "Keep it under 800 words. Use structured bullet points.\n\n"
        "Tool Results:\n" + "\n---\n".join(findings_parts)
    )

    response = await asyncio.wait_for(
        backend.chat_completion(
            messages=[
                Message(role=Role.SYSTEM, content="You are a concise summarizer. Output only the summary."),
                Message(role=Role.USER, content=summary_prompt),
            ],
            tools=None,
            tool_choice="none",
            stream=False,
            max_tokens=2048,
            temperature=0.0,
        ),
        timeout=_LLM_SUMMARY_TIMEOUT,
    )

    summary_text = getattr(response, "content", "") or ""
    if not summary_text:
        return

    # Collect system messages and find first user message
    system_msgs: list[Message] = []
    first_user_idx = -1
    for i, msg in enumerate(messages):
        if msg.role == Role.SYSTEM:
            system_msgs.append(msg)
        elif msg.role == Role.USER and first_user_idx < 0:
            first_user_idx = i
            break

    # Find boundary that preserves complete turn structure
    tail_start = _find_turn_boundary(messages, keep_turns=2)
    tail = messages[tail_start:]

    compressed = f"[Context compressed — previous analysis summarized]\n\n{summary_text}"
    new_messages = system_msgs + [
        Message(role=Role.USER, content=messages[first_user_idx].content),
        Message(role=Role.ASSISTANT, content=compressed),
    ] + tail

    messages.clear()
    messages.extend(new_messages)
