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
        assert "analyze" in result.output
        assert "chat" in result.output
        assert "profile" in result.output

    def test_analyze_help(self, runner):
        """tachyon analyze --help shows options."""
        result = runner.invoke(app, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "--format" in result.output or "REPORT" in result.output

    def test_chat_help(self, runner):
        """tachyon chat --help shows options."""
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "report" in result.output.lower() or "chat" in result.output.lower()

    def test_profile_help(self, runner):
        """tachyon profile --help shows options."""
        result = runner.invoke(app, ["profile", "--help"])
        assert result.exit_code == 0
        assert "--strategy" in result.output

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

    def test_analyze_missing_file(self, runner):
        """tachyon analyze nonexistent.ncu-rep shows error."""
        result = runner.invoke(app, ["analyze", "nonexistent.ncu-rep"])
        # Should fail gracefully (nonzero exit or error message)
        assert result.exit_code != 0 or "error" in result.output.lower() or "not" in result.output.lower()

    def test_main_no_args(self, runner):
        """tachyon with no args shows usage (Click returns exit code 0 or 2)."""
        result = runner.invoke(app, [])
        assert result.exit_code in (0, 2)
        assert "Usage" in result.output or "analyze" in result.output
