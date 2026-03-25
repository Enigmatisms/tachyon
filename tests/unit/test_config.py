"""Unit tests for configuration management."""
import os
from pathlib import Path
from unittest.mock import patch

from tachyon.config.settings import (
    TachyonConfig,
)


class TestDefaultConfig:
    def test_defaults(self):
        cfg = TachyonConfig()
        assert cfg.llm.provider == "openai"
        assert cfg.llm.model == "gpt-4o"
        assert cfg.llm.api_key is None
        assert cfg.llm.api_key_env == "OPENAI_API_KEY"
        assert cfg.profiling.depth == "basic"
        assert cfg.output.lang == ""
        assert cfg.output.format == "terminal"
        assert cfg.tools.ncu_path is None

    @patch.dict(os.environ, {}, clear=True)
    def test_load_no_file(self, tmp_path: Path):
        """Load with nonexistent config file returns defaults (clean env)."""
        cfg = TachyonConfig.load(tmp_path / "nonexistent.toml")
        assert cfg.llm.provider == "openai"
        assert cfg.output.format == "terminal"


class TestTomlOverride:
    def test_load_toml(self, tmp_path: Path):
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[llm]
provider = "anthropic"
model = "claude-3-opus"
temperature = 0.5

[profiling]
depth = "radical"

[output]
lang = "zh"
format = "markdown"

[tools]
ncu_path = "/custom/ncu"
""")
        cfg = TachyonConfig.load(config_file)
        assert cfg.llm.provider == "anthropic"
        assert cfg.llm.model == "claude-3-opus"
        assert cfg.llm.temperature == 0.5
        assert cfg.profiling.depth == "radical"
        assert cfg.output.lang == "zh"
        assert cfg.output.format == "markdown"
        assert cfg.tools.ncu_path == "/custom/ncu"


class TestEnvOverride:
    def test_env_overrides(self):
        env = {
            "TACHYON_LLM_PROVIDER": "anthropic",
            "TACHYON_MODEL": "gpt-4-turbo",
            "TACHYON_LANG": "zh",
            "TACHYON_DEPTH": "radical",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = TachyonConfig()
            cfg._apply_env()
            assert cfg.llm.provider == "anthropic"
            assert cfg.llm.model == "gpt-4-turbo"
            assert cfg.output.lang == "zh"
            assert cfg.profiling.depth == "radical"

    def test_env_strategy_backward_compat(self):
        """TACHYON_STRATEGY still works for backward compatibility."""
        env = {
            "TACHYON_STRATEGY": "radical",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = TachyonConfig()
            cfg._apply_env()
            assert cfg.profiling.depth == "radical"

    def test_env_ncu_report_path(self):
        with patch.dict(os.environ, {"TACHYON_NCU_REPORT_PATH": "/my/ncu"}):
            cfg = TachyonConfig()
            cfg._apply_env()
            assert cfg.tools.ncu_report_path == "/my/ncu"

    def test_env_api_key_direct(self):
        """TACHYON_API_KEY sets api_key directly."""
        with patch.dict(os.environ, {"TACHYON_API_KEY": "sk-my-key"}, clear=False):
            cfg = TachyonConfig()
            cfg._apply_env()
            assert cfg.llm.api_key == "sk-my-key"

    def test_env_api_key_env_override(self):
        """TACHYON_API_KEY_ENV changes which env var holds the key."""
        with patch.dict(os.environ, {"TACHYON_API_KEY_ENV": "AGENT_API_KEY"}, clear=False):
            cfg = TachyonConfig()
            cfg._apply_env()
            assert cfg.llm.api_key_env == "AGENT_API_KEY"

    def test_env_base_url(self):
        """TACHYON_BASE_URL sets llm.base_url."""
        with patch.dict(os.environ, {"TACHYON_BASE_URL": "https://api.minimax.chat/v1"}, clear=False):
            cfg = TachyonConfig()
            cfg._apply_env()
            assert cfg.llm.base_url == "https://api.minimax.chat/v1"


class TestCliOverride:
    def test_cli_overrides(self):
        cfg = TachyonConfig()
        cfg.apply_cli_overrides(
            model="custom-model",
            lang="zh",
            format="json",
            depth="radical",
        )
        assert cfg.llm.model == "custom-model"
        assert cfg.output.lang == "zh"
        assert cfg.output.format == "json"
        assert cfg.profiling.depth == "radical"

    def test_cli_partial_override(self):
        cfg = TachyonConfig()
        cfg.apply_cli_overrides(format="markdown")
        assert cfg.output.format == "markdown"
        assert cfg.llm.provider == "openai"  # unchanged


class TestPriorityOrder:
    def test_env_overrides_toml(self, tmp_path: Path):
        """Environment variables should override TOML config."""
        config_file = tmp_path / "config.toml"
        config_file.write_text('[llm]\nprovider = "openai"\n')

        with patch.dict(os.environ, {"TACHYON_LLM_PROVIDER": "anthropic"}):
            cfg = TachyonConfig.load(config_file)
            assert cfg.llm.provider == "anthropic"

    def test_cli_overrides_env(self, tmp_path: Path):
        """CLI should override environment variables."""
        with patch.dict(os.environ, {"TACHYON_LANG": "zh"}):
            cfg = TachyonConfig.load(tmp_path / "none.toml")
            cfg.apply_cli_overrides(lang="en")
            assert cfg.output.lang == "en"
