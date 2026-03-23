"""Tachyon Evolve — CUDA kernel self-evolution framework.

Provides automated source-code optimization via iterative:
  NCU analysis → hypothesis → edit → compile → benchmark → reprofile → decide

Three-layer architecture:
  Layer 1: Framework & Base (config, models, git, session, context)
  Layer 2: Tools (edit_source_file, compile_kernel, run_benchmark, reprofile, ...)
  Layer 3: Agent Logic (persona, orchestrator, CLI)
"""
