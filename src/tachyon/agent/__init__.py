"""Agent layer — LLM-driven analysis (M3).

Public API:
  - run_agent_loop() — multi-turn agent loop
  - AgentEvent — event yielded by the loop
  - build_system_prompt() — persona assembly
"""
from .loop import AgentEvent, run_agent_loop
from .persona import build_system_prompt

__all__ = [
    "AgentEvent",
    "run_agent_loop",
    "build_system_prompt",
]
