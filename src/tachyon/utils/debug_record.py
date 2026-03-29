"""Backward-compatible shim — delegates to :mod:`tachyon.utils.log`.

All public symbols are re-exported so existing ``from tachyon.utils.debug_record
import init, record_prompt, ...`` continues to work without changes.

The actual implementation lives in :mod:`tachyon.utils.log` (AgentLogger).
"""
from tachyon.utils.log import (  # noqa: F401
    AgentLogger,
    agent_logger,
    close,
    configure_logging,
    get_logger,
    init,
    level,
    log_path,
    record_prompt,
    record_tool,
)

__all__ = [
    "init",
    "close",
    "level",
    "log_path",
    "record_prompt",
    "record_tool",
]
