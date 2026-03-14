"""Example: Use Tachyon as a library (SDK mode).

Demonstrates programmatic access to the core components:
  - TachyonConfig: layered configuration
  - ToolPathResolver: NVIDIA tool discovery
  - NcuProfiler: two-stage profiling engine
  - ToolRegistry & ToolDefinition: tool schema export

Usage:
    python examples/sdk_usage.py

No GPU or CUDA installation required -- all external calls are skipped.
"""
from __future__ import annotations

import asyncio
import json

from tachyon import NcuProfiler, ProfilingStrategy, TachyonConfig, ToolPathResolver
from tachyon.tools.registry import ToolDefinition, ToolRegistry


def demo_config() -> TachyonConfig:
    """Demonstrate TachyonConfig with defaults and CLI overrides."""
    print("=" * 60)
    print("1. TachyonConfig -- layered configuration")
    print("=" * 60)

    # Load defaults (no config file needed)
    config = TachyonConfig()
    print(f"  Default LLM provider: {config.llm.provider}")
    print(f"  Default model:        {config.llm.model}")
    print(f"  Default strategy:     {config.profiling.strategy}")
    print(f"  Default language:     {config.output.lang}")

    # Apply CLI overrides (highest priority)
    config.apply_cli_overrides(model="claude-3-opus", lang="zh")
    print(f"\n  After CLI override:")
    print(f"    model -> {config.llm.model}")
    print(f"    lang  -> {config.output.lang}")
    print()

    return config


def demo_tool_path_resolver(config: TachyonConfig) -> None:
    """Demonstrate ToolPathResolver -- multi-source tool discovery."""
    print("=" * 60)
    print("2. ToolPathResolver -- NVIDIA tool discovery")
    print("=" * 60)

    resolver = ToolPathResolver(config)

    for tool_name in ["ncu", "nvdisasm", "cuobjdump"]:
        result = resolver.resolve_safe(tool_name)
        if result.success:
            print(f"  {tool_name}: {result.data}")
        else:
            print(f"  {tool_name}: not found ({result.error.message[:60]}...)")

    print()


def demo_profiler_config(config: TachyonConfig) -> None:
    """Demonstrate NcuProfiler configuration (without actually running ncu)."""
    print("=" * 60)
    print("3. NcuProfiler -- two-stage profiling engine (config only)")
    print("=" * 60)

    from tachyon.profiler.ncu_profiler import STRATEGY_CONFIGS

    for strategy in ProfilingStrategy:
        stage1, stage2 = STRATEGY_CONFIGS[strategy]
        print(f"\n  Strategy: {strategy.value}")
        print(f"    Stage 1 (Quick Scan): metric_set={stage1.metric_set}, "
              f"timeout={stage1.timeout_sec}s")
        print(f"    Stage 2 (Deep Dive): metric_set={stage2.metric_set}, "
              f"timeout={stage2.timeout_sec}s, "
              f"source_counters={stage2.source_counters}")

    print()
    print("  To actually profile, you would call:")
    print("    resolver = ToolPathResolver(config)")
    print("    profiler = NcuProfiler(config, resolver)")
    print("    result = profiler.profile_basic('./my_cuda_app', ['--size', '1024'])")
    print()


def demo_tool_registry() -> None:
    """Demonstrate ToolRegistry -- tool definition and schema export."""
    print("=" * 60)
    print("4. ToolRegistry -- tool schema management")
    print("=" * 60)

    registry = ToolRegistry()

    # Register a custom tool
    async def my_handler(kernel_id: int) -> dict:
        return {"status": "ok", "kernel_id": kernel_id}

    registry.register(ToolDefinition(
        name="my_custom_tool",
        description="A custom analysis tool for demonstration.",
        parameters={
            "type": "object",
            "properties": {
                "kernel_id": {
                    "type": "integer",
                    "description": "Kernel index to analyze",
                }
            },
            "required": ["kernel_id"],
        },
        handler=my_handler,
    ))

    print(f"\n  Registered tools: {registry.tool_names()}")

    # Export to different LLM formats
    tool = registry.get("my_custom_tool")
    assert tool is not None

    print("\n  OpenAI format:")
    print(f"    {json.dumps(tool.to_openai(), indent=4)[:200]}...")

    print("\n  Anthropic format:")
    print(f"    {json.dumps(tool.to_anthropic(), indent=4)[:200]}...")

    print("\n  MCP format:")
    print(f"    {json.dumps(tool.to_mcp(), indent=4)[:200]}...")

    # Execute the tool
    async def run_tool():
        from tachyon.errors.handler import ToolResult
        result = await registry.execute("my_custom_tool", {"kernel_id": 0})
        print(f"\n  Execution result: success={result.success}, data={result.data}")

    asyncio.run(run_tool())
    print()


def demo_full_tool_stack() -> None:
    """Show all 9 built-in tools that Tachyon registers for Agent use."""
    print("=" * 60)
    print("5. Built-in Tool Stack -- all 9 Tachyon tools")
    print("=" * 60)

    from tachyon.analyzers.base import AnalyzerRegistry
    from tachyon.tools.analysis import register_analysis_tools
    from tachyon.tools.context import SessionContext
    from tachyon.tools.data_query import register_data_query_tools
    from tachyon.tools.source import register_source_tools

    # Create an empty session (no report loaded)
    analyzer_registry = AnalyzerRegistry()
    analyzer_registry.auto_register()

    session = SessionContext(kernels=[], registry=analyzer_registry)

    registry = ToolRegistry()
    register_data_query_tools(registry, session)
    register_source_tools(registry, session)
    register_analysis_tools(registry, session)

    print(f"\n  Total tools: {len(registry.tool_names())}")
    for name in registry.tool_names():
        tool = registry.get(name)
        desc = tool.description[:70] if tool else "?"
        print(f"    - {name}: {desc}...")

    print()


def main() -> None:
    print("Tachyon SDK Usage Example")
    print("=" * 60)
    print()

    config = demo_config()
    demo_tool_path_resolver(config)
    demo_profiler_config(config)
    demo_tool_registry()
    demo_full_tool_stack()

    print("Done. All SDK components demonstrated successfully.")


if __name__ == "__main__":
    main()
