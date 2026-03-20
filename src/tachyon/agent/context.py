"""Context window management — compression, distillation, and token tracking.

Strategies (applied in order of aggressiveness):
  1. compact_tool_results: Summarize old tool results, preserving high-value ones
  2. distill: Collapse tool-call/result pairs into summaries (keeps key evidence)
  3. truncate: Drop oldest turns (keep system + last N)

Token estimation: ~4 chars/token for English/code, ~3 chars/token for system prompts.
"""
from __future__ import annotations

import json
import logging

from ..llm.backend import Message, Role

_log = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_TOKEN_BUDGET = 120_000
DEEP_TOKEN_BUDGET = 200_000
COMPACT_THRESHOLD = 200  # chars — below this, tool results are kept as-is
COMPACT_SUMMARY_LEN = 80  # chars for non-preserved tool summaries
PRESERVE_SUMMARY_LEN = 2000  # chars for preserved tool summaries (more context)
COMPACT_AFTER_TURN = 5  # start compacting after this many turns (was 3)

# Tool names whose results are "high-value" and should retain more context
# when compacted (instead of being reduced to a single line).
PRESERVE_TOOLS = frozenset({
    "read_source_file",
    "get_stall_analysis_for_line",
    "get_sass_for_source_line",
    "get_performance_hotspots",
    "run_analysis",
})


class ContextManager:
    """Manages conversation context to fit within LLM token limits."""

    def __init__(self, budget: int = DEFAULT_TOKEN_BUDGET) -> None:
        self.budget = budget

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

    def compact_tool_results(
        self, messages: list[Message], keep_last_n: int = 2,
    ) -> None:
        """Summarize old tool results, preserving high-value ones. Mutates in-place.

        Preserves the last keep_last_n tool result turns intact.
        Tools in PRESERVE_TOOLS get a longer summary (PRESERVE_SUMMARY_LEN)
        to retain code context and evidence.
        """
        tool_indices = [
            i for i, m in enumerate(messages) if m.role == Role.TOOL
        ]
        if len(tool_indices) <= keep_last_n:
            return
        cutoff_indices = set(tool_indices[:-keep_last_n])
        for i in cutoff_indices:
            msg = messages[i]
            if not msg.content or len(msg.content) <= COMPACT_THRESHOLD:
                continue
            tool_name = msg.name or "unknown"
            is_preserved = tool_name in PRESERVE_TOOLS
            limit = PRESERVE_SUMMARY_LEN if is_preserved else COMPACT_SUMMARY_LEN
            first_part = msg.content.split("\n")[0][:limit]
            messages[i] = Message(
                role=Role.TOOL,
                content=f"[{tool_name}: {first_part}...]",
                tool_call_id=msg.tool_call_id,
                name=msg.name,
            )

    def distill(self, messages: list[Message]) -> list[Message]:
        """Compress: collapse tool-call/result pairs into summaries.

        For preserved tools, keeps the first portion of the result as summary
        so the LLM retains key evidence. For others, just keeps tool names.
        """
        result: list[Message] = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg.role in (Role.SYSTEM, Role.USER):
                result.append(msg)
                i += 1
            elif msg.role == Role.ASSISTANT and msg.tool_calls:
                tool_names = [tc.name for tc in msg.tool_calls]
                has_preserved = any(n in PRESERVE_TOOLS for n in tool_names)
                i += 1
                # Collect tool results and build a summary
                summaries: list[str] = []
                while i < len(messages) and messages[i].role == Role.TOOL:
                    tool_msg = messages[i]
                    tn = tool_msg.name or "unknown"
                    content = tool_msg.content or ""
                    if tn in PRESERVE_TOOLS and content:
                        # Keep first portion for evidence retention
                        summaries.append(
                            f"[{tn}: {content[:PRESERVE_SUMMARY_LEN]}...]"
                        )
                    i += 1
                if summaries:
                    combined = "\n".join(summaries)
                    result.append(Message(
                        role=Role.ASSISTANT,
                        content=f"[Agent called: {', '.join(tool_names)}]\n{combined}",
                    ))
                else:
                    result.append(Message(
                        role=Role.ASSISTANT,
                        content=f"[Agent called: {', '.join(tool_names)}]",
                    ))
            else:
                result.append(msg)
                i += 1
        _log.info("Distilled %d → %d messages", len(messages), len(result))
        return result

    def should_compact(self, turn: int) -> bool:
        """Whether to compact old tool results at this turn."""
        return turn >= COMPACT_AFTER_TURN

    def is_near_budget(self, messages: list[Message]) -> bool:
        """Whether estimated tokens are near 85% of budget."""
        return self.estimate_tokens(messages) > self.budget * 0.85
