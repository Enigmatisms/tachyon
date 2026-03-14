"""CLI serve command — start MCP server for external agents.

Usage:
    tachyon serve --mcp --report report.ncu-rep
"""
from __future__ import annotations

import asyncio
import sys

import click

from tachyon.cli.main import app
from tachyon.config.settings import TachyonConfig


@app.command()
@click.option("--mcp", is_flag=True, help="Start MCP server (stdio transport)")
@click.option(
    "--report",
    type=click.Path(exists=True),
    default=None,
    help="Pre-load this .ncu-rep file for analysis",
)
def serve(mcp: bool, report: str | None) -> None:
    """Start Tachyon as a tool server for external agents.

    MCP mode (--mcp): stdio transport, integrates with Claude Code / Cursor.
    """
    config = TachyonConfig.load()

    if mcp:
        from tachyon.server.mcp import TachyonMCPServer

        server = TachyonMCPServer(config, report_path=report)
        try:
            asyncio.run(server.run())
        except ImportError as e:
            click.secho(f"Error: {e}", fg="red", err=True)
            sys.exit(1)
    else:
        click.echo("Specify --mcp to start MCP server. See: tachyon serve --help")
        sys.exit(1)
