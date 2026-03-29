"""Persona definition — the agent's identity and system prompt.

The system prompt lives in persona.md (loaded at runtime) so it can be
edited without touching Python code. Template variables:
  {tool_catalog}  — auto-generated from ToolRegistry
  {kernel_list}   — populated per-session from loaded report
"""
from __future__ import annotations

from pathlib import Path

from ..tools.registry import ToolRegistry

_PERSONA_MD = Path(__file__).with_name("persona.md")

_LEAN_SYSTEM = """\
You are **Tachyon**, a CUDA/HPC performance analyst.

## Tools
{tool_catalog}

**CRITICAL**: Call tools to gather evidence before making claims. \
NEVER fabricate metrics, source code, or SASS instructions.

## Output
- Evidence-first: cite ``[metric=value]`` or ``[file:line -> SASS opcode]``.
- Numbered recommendations with priority (HIGH/MEDIUM/LOW).
- Be token-efficient."""


def _build_tool_catalog(registry: ToolRegistry) -> str:
    """Build compact one-line-per-tool catalog from registry."""
    lines = []
    for tool in registry.all_definitions():
        params = ", ".join(
            f"{k}: {v.get('type', 'any')}"
            for k, v in tool.parameters.get("properties", {}).items()
        )
        desc = tool.description.split(". ")[0]  # first sentence only
        lines.append(f"  - `{tool.name}({params})` — {desc}")
    return "\n".join(lines)


def build_system_prompt(
    registry: ToolRegistry,
    kernel_context: str | None = None,
    skill_knowledge: str = "",
) -> str:
    """Build the complete system prompt for an agent session (turn 0).

    Args:
        registry: ToolRegistry with all tools registered.
        kernel_context: Optional pre-formatted kernel list.
        skill_knowledge: Optional CUDA optimization knowledge from SkillRegistry.

    Returns:
        Complete system prompt with template variables filled.
    """
    template = _PERSONA_MD.read_text(encoding="utf-8")
    tool_catalog = _build_tool_catalog(registry)

    body = template.format(
        tool_catalog=tool_catalog,
        kernel_list=kernel_context or "(no report loaded yet)",
    )

    if skill_knowledge:
        body += "\n\n## CUDA Optimization Knowledge\n\n" + skill_knowledge

    return _lang_prefix() + body


def build_lean_system_prompt(
    registry: ToolRegistry,
    extra: str = "",
    skill_brief: str = "",
) -> str:
    """Build a minimal system prompt for turns after 0.

    Keeps only identity + tool catalog + key rules.
    Saves ~9000 chars (~2000 tokens) per subsequent turn.
    """
    tool_catalog = _build_tool_catalog(registry)
    body = _LEAN_SYSTEM.format(tool_catalog=tool_catalog)
    result = _lang_prefix() + body
    if extra:
        result += "\n\n" + extra
    if skill_brief:
        result += "\n\n## Optimization Skills\n" + skill_brief
    return result


def _lang_prefix() -> str:
    """Return a one-line language instruction to PREPEND to the system prompt.

    Placed at the very top so the LLM sees it first (highest attention weight).
    """
    import tachyon.i18n as _i18n
    instruction = _i18n.t("prompt.output_lang.text", fallback="")
    if not instruction:
        return ""
    return f"[Language] {instruction}\n\n"


def build_kernel_context(kernels: list) -> str:
    """Build kernel list string for persona template.

    Args:
        kernels: List of KernelReport objects.

    Returns:
        Formatted string listing all kernels with key info.
    """
    if not kernels:
        return "(empty report — no kernels found)"
    lines = []
    for i, k in enumerate(kernels):
        name = getattr(k, "demangled_name", None) or getattr(k, "kernel_name", "?")
        grid = getattr(k.launch_params, "grid", (0, 0, 0))
        block = getattr(k.launch_params, "block", (0, 0, 0))
        lines.append(f"  [{i}] {name} — grid={grid}, block={block}")
    return "\n".join(lines)
