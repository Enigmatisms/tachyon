"""Context window management — compression, distillation, and token tracking.

Strategies (applied in order of aggressiveness):
  1. compact_tool_results: Replace old tool results (>threshold chars) with summaries
  2. distill: Collapse tool-call/result pairs into single summary messages
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
TOOL_RESULT_COMPACT_THRESHOLD = 200  # chars
COMPACT_AFTER_TURN = 3  # start compacting after this many turns


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
        self, messages: list[Message], keep_last_n: int = 1,
    ) -> None:
        """Replace old tool result content with short summaries. Mutates in-place.

        Preserves the last keep_last_n tool result turns intact.
        """
        tool_indices = [
            i for i, m in enumerate(messages) if m.role == Role.TOOL
        ]
        if len(tool_indices) <= keep_last_n:
            return
        cutoff_indices = set(tool_indices[:-keep_last_n])
        for i in cutoff_indices:
            msg = messages[i]
            if msg.content and len(msg.content) > TOOL_RESULT_COMPACT_THRESHOLD:
                tool_name = msg.name or "unknown"
                first_line = msg.content.split("\n")[0][:80]
                messages[i] = Message(
                    role=Role.TOOL,
                    content=f"[{tool_name}: {first_line}...]",
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                )

    def distill(self, messages: list[Message]) -> list[Message]:
        """Aggressively compress: collapse tool-call/result pairs into summaries."""
        result: list[Message] = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg.role in (Role.SYSTEM, Role.USER):
                result.append(msg)
                i += 1
            elif msg.role == Role.ASSISTANT and msg.tool_calls:
                tool_names = [tc.name for tc in msg.tool_calls]
                i += 1
                # Skip following tool results
                while i < len(messages) and messages[i].role == Role.TOOL:
                    i += 1
                summary = f"[Agent called: {', '.join(tool_names)}]"
                result.append(Message(role=Role.ASSISTANT, content=summary))
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
