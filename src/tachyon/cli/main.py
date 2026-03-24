"""Tachyon CLI entrypoint — Click-based command group.

Subcommands are registered via explicit imports in their respective modules.
M1 ships only `analyze`. M3 adds `chat`, M4 adds `profile` and `serve`.
"""
from __future__ import annotations

import click

from tachyon import __version__
from tachyon.utils.log import configure_logging


configure_logging()


@click.group()
@click.version_option(version=__version__, prog_name="tachyon")
def app() -> None:
    """Tachyon — AI-Powered CUDA Performance Analyzer."""


# Import subcommand modules so they register themselves with @app.command().
# This import must come after `app` is defined to avoid circular imports.
import tachyon.cli.analyze  # noqa: E402, F401
import tachyon.cli.chat  # noqa: E402, F401  # M3: AI Agent chat
import tachyon.cli.diff  # noqa: E402, F401  # M5: Profile diff
import tachyon.cli.evolve  # noqa: E402, F401  # M6: Evolve optimization
import tachyon.cli.profile  # noqa: E402, F401  # M4: E2E profiling
import tachyon.cli.serve  # noqa: E402, F401  # M5: MCP serve
