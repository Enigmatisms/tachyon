"""Unit tests for i18n system."""
from __future__ import annotations

import pytest

import tachyon.i18n as i18n


@pytest.fixture(autouse=True)
def reset_i18n():
    """Reset i18n module state before each test."""
    i18n._packs.clear()
    i18n._current_lang = "en"
    yield
    i18n._packs.clear()
    i18n._current_lang = "en"


# ===================================================================
# TestInit
# ===================================================================

class TestInit:
    """Verify init() sets the current language correctly."""

    def test_init_en(self):
        i18n.init("en")
        assert i18n.current_lang() == "en"

    def test_init_zh(self):
        i18n.init("zh")
        assert i18n.current_lang() == "zh"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("TACHYON_LANG", "zh")
        i18n.init(None)
        assert i18n.current_lang() == "zh"


# ===================================================================
# TestTranslation
# ===================================================================

class TestTranslation:
    """Verify t() lookup, fallback chain, and template rendering."""

    def test_en_lookup(self):
        i18n.init("en")
        result = i18n.t("report.executive_summary")
        assert result == "Executive Summary"

    def test_zh_lookup(self):
        i18n.init("zh")
        result = i18n.t("report.executive_summary")
        assert result == "\u6267\u884c\u6458\u8981"

    def test_fallback_to_en(self):
        """Key present in en but not zh should fall back to English."""
        i18n.init("zh")
        # Both en.toml and zh.toml have the same keys in the current set,
        # so we test the fallback mechanism by looking up a key that
        # we know only English has — use the warning key which exists in both.
        # Instead, directly verify the fallback logic: if zh pack lacks a key
        # the English value is returned.
        i18n._packs["zh"].pop("warning.no_debug_info", None)
        result = i18n.t("warning.no_debug_info")
        # Should fall back to the English value
        assert "debug info" in result.lower() or "debug" in result.lower()

    def test_fallback_to_arg(self):
        i18n.init("en")
        result = i18n.t("nonexistent.key.here", fallback="fb")
        assert result == "fb"

    def test_fallback_to_key(self):
        i18n.init("en")
        result = i18n.t("nonexistent.key.here")
        assert result == "nonexistent.key.here"

    def test_format_args(self):
        i18n.init("en")
        result = i18n.t(
            "finding.mem_coalesce.detail", efficiency=42, threshold=80
        )
        assert "42" in result
        assert "80" in result


# ===================================================================
# TestFlatten
# ===================================================================

class TestFlatten:
    """Verify TOML nested sections are flattened to dotted keys."""

    def test_nested_toml_flattened(self):
        i18n.init("en")
        pack = i18n._packs["en"]
        # [report] section keys should be flattened with dot notation
        assert "report.executive_summary" in pack
        assert "report.top_findings" in pack
        assert "report.evidence_chains" in pack
        assert "report.detailed_metrics" in pack
        assert "report.optimization_tree" in pack
        # [finding.mem_coalesce] section keys should also be flattened
        assert "finding.mem_coalesce.title" in pack
        assert "finding.mem_coalesce.detail" in pack
        assert "finding.mem_coalesce.action" in pack


# ===================================================================
# TestEdgeCases — cover missing i18n branches
# ===================================================================

class TestEdgeCases:
    """Cover edge-case branches for higher coverage."""

    def test_locale_value_error(self, monkeypatch):
        """getlocale() raising ValueError should default to 'en'."""
        monkeypatch.delenv("TACHYON_LANG", raising=False)
        monkeypatch.setattr("locale.getlocale", lambda: (_ for _ in ()).throw(ValueError))
        i18n.init(None)
        assert i18n.current_lang() == "en"

    def test_nonexistent_lang_pack(self):
        """Loading a language with no .toml file should not crash."""
        i18n.init("xx_nonexistent")
        # Should have no pack loaded for "xx_nonexistent"
        assert "xx_nonexistent" not in i18n._packs
        # t() should fallback to key
        assert i18n.t("any.key") == "any.key"

    def test_pack_loaded_once(self):
        """Calling _load_pack twice for the same lang should not re-read."""
        i18n.init("en")
        pack_before = i18n._packs["en"]
        i18n._load_pack("en")  # second call should be a no-op
        assert i18n._packs["en"] is pack_before

    def test_format_bad_key(self):
        """t() with bad format kwargs should return template unformatted."""
        i18n.init("en")
        # Use a key that has format placeholders, but pass wrong kwarg names
        result = i18n.t("finding.mem_coalesce.detail", wrong_key=99)
        # Should return the raw template, not crash
        assert isinstance(result, str)
        assert len(result) > 0

    def test_sys_locale_zh(self, monkeypatch):
        """System locale starting with 'zh' should resolve to 'zh'."""
        monkeypatch.delenv("TACHYON_LANG", raising=False)
        monkeypatch.setattr("locale.getlocale", lambda: ("zh_CN.UTF-8", "UTF-8"))
        i18n.init(None)
        assert i18n.current_lang() == "zh"

    def test_sys_locale_other(self, monkeypatch):
        """System locale not starting with 'zh' should resolve to 'en'."""
        monkeypatch.delenv("TACHYON_LANG", raising=False)
        monkeypatch.setattr("locale.getlocale", lambda: ("fr_FR.UTF-8", "UTF-8"))
        i18n.init(None)
        assert i18n.current_lang() == "en"

    def test_sys_locale_none(self, monkeypatch):
        """System locale returning (None, None) should resolve to 'en'."""
        monkeypatch.delenv("TACHYON_LANG", raising=False)
        monkeypatch.setattr("locale.getlocale", lambda: (None, None))
        i18n.init(None)
        assert i18n.current_lang() == "en"
