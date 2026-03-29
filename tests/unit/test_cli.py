"""Unit tests for CLI commands using Click's test runner."""
from click.testing import CliRunner

from tachyon.cli.main import app


class TestCliVersion:
    def test_version_flag(self):
        runner = CliRunner()
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestCliHelp:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "AI-Powered CUDA Performance Analyzer" in result.output
