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

AGENT_IDENTITY = {
    "name": "Tachyon",
    "role": "CUDA/HPC Performance Analysis Expert",
    "expertise": [
        "NVIDIA GPU microarchitecture",
        "CUDA kernel optimization",
        "SASS/PTX instruction analysis",
        "Source-to-assembly correlation",
    ],
}


def build_system_prompt(
    registry: ToolRegistry,
    kernel_context: str | None = None,
) -> str:
    """Build the complete system prompt for an agent session.

    Args:
        registry: ToolRegistry with all tools registered.
        kernel_context: Optional pre-formatted kernel list.

    Returns:
        Complete system prompt with template variables filled.
    """
    template = _PERSONA_MD.read_text(encoding="utf-8")

    # Build compact tool catalog
    tool_lines = []
    for tool in registry.all_definitions():
        params = ", ".join(
            f"{k}: {v.get('type', 'any')}"
            for k, v in tool.parameters.get("properties", {}).items()
        )
        desc = tool.description.split(". ")[0]  # first sentence only
        tool_lines.append(f"  - `{tool.name}({params})` — {desc}")
    tool_catalog = "\n".join(tool_lines)

    return template.format(
        tool_catalog=tool_catalog,
        kernel_list=kernel_context or "(no report loaded yet)",
    )


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
