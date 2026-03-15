"""Tachyon CLI entrypoint — Click-based command group.

Subcommands are registered via explicit imports in their respective modules.
M1 ships only `analyze`. M3 adds `chat`, M4 adds `profile` and `serve`.
"""
from __future__ import annotations

import logging
import os

import click

from tachyon import __version__


def _configure_logging() -> None:
    """Configure logging based on TACHYON_LOG_LEVEL env var.

    Default is WARNING — suppresses httpx INFO, analyzer skip messages, etc.
    Set TACHYON_LOG_LEVEL=DEBUG for full diagnostics.
    """
    level_name = os.environ.get("TACHYON_LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, level_name, logging.WARNING)
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    # Always suppress chatty httpx/httpcore regardless
    logging.getLogger("httpx").setLevel(max(level, logging.WARNING))
    logging.getLogger("httpcore").setLevel(max(level, logging.WARNING))


_configure_logging()


@click.group()
@click.version_option(version=__version__, prog_name="tachyon")
def app() -> None:
    """Tachyon — AI-Powered CUDA Performance Analyzer."""


# Import subcommand modules so they register themselves with @app.command().
# This import must come after `app` is defined to avoid circular imports.
import tachyon.cli.analyze  # noqa: E402, F401
import tachyon.cli.chat  # noqa: E402, F401  # M3: AI Agent chat
import tachyon.cli.diff  # noqa: E402, F401  # M5: Profile diff
import tachyon.cli.profile  # noqa: E402, F401  # M4: E2E profiling
import tachyon.cli.serve  # noqa: E402, F401  # M5: MCP serve
