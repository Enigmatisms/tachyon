"""CLI command smoke tests — verify all subcommands are registered and show help.

Uses Click's CliRunner for isolated command invocation.
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from tachyon.cli.main import app


@pytest.fixture
def runner():
    return CliRunner()


class TestCLICommands:

    def test_main_help(self, runner):
        """tachyon --help lists all subcommands."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "chat" in result.output
        assert "profile" in result.output

    def test_chat_help(self, runner):
        """tachyon chat --help shows options."""
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "report" in result.output.lower() or "chat" in result.output.lower()

    def test_profile_help(self, runner):
        """tachyon profile --help shows options."""
        result = runner.invoke(app, ["profile", "--help"])
        assert result.exit_code == 0
        assert "--deep" in result.output or "--radical" in result.output

    def test_serve_help(self, runner):
        """tachyon serve --help shows options."""
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--mcp" in result.output or "serve" in result.output.lower()

    def test_diff_help(self, runner):
        """tachyon diff --help shows options."""
        result = runner.invoke(app, ["diff", "--help"])
        assert result.exit_code == 0
        assert "BEFORE" in result.output or "before" in result.output.lower()

    def test_main_no_args(self, runner):
        """tachyon with no args shows usage (Click returns exit code 0 or 2)."""
        result = runner.invoke(app, [])
        assert result.exit_code in (0, 2)
        assert "Usage" in result.output
