"""Internationalization (i18n) for Tachyon.

Language pack priority: CLI --lang > TACHYON_LANG env > system locale > fallback "en".
Template rendering: Python str.format() with named placeholders.
Language packs: TOML files co-located in this package directory.
"""
from __future__ import annotations

import locale
import os
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]  # Python < 3.11

_PACK_DIR = Path(__file__).parent
_packs: dict[str, dict[str, str]] = {}
_current_lang: str = "en"


def init(lang: str | None = None) -> None:
    """Initialize i18n with the specified language.

    Args:
        lang: Language code ("en", "zh"). If None, auto-detect
              from TACHYON_LANG env -> system locale -> fallback "en".
    """
    global _current_lang
    _current_lang = _resolve_lang(lang)
    _load_pack(_current_lang)
    if _current_lang != "en":
        _load_pack("en")  # Always load English as fallback


def _resolve_lang(explicit: str | None) -> str:
    """Resolve language with priority: explicit > env > locale > 'en'."""
    if explicit:
        return explicit
    if env_lang := os.environ.get("TACHYON_LANG"):
        return env_lang
    # locale.getlocale() returns (language_code, encoding) or (None, None).
    # NOTE: We intentionally avoid locale.getdefaultlocale() which is deprecated
    # since Python 3.11.
    try:
        sys_locale = locale.getlocale()[0] or ""
    except ValueError:
        # getlocale() can raise ValueError on some platforms with unusual locale
        sys_locale = ""
    if sys_locale.startswith("zh"):
        return "zh"
    return "en"


def _load_pack(lang: str) -> None:
    """Load a TOML language pack into memory, flattening nested keys with dots."""
    if lang in _packs:
        return
    path = _PACK_DIR / f"{lang}.toml"
    if not path.exists():
        return
    with open(path, "rb") as f:
        data = tomllib.load(f)
    # Flatten nested TOML into dotted keys:
    # {"finding": {"mem_coalesce": {"title": "..."}}}
    # -> {"finding.mem_coalesce.title": "..."}
    flat: dict[str, str] = {}
    _flatten(data, "", flat)
    _packs[lang] = flat


def _flatten(
    data: dict[str, Any], prefix: str, result: dict[str, str]
) -> None:
    """Recursively flatten a nested dict into dotted-key -> string entries."""
    for k, v in data.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _flatten(v, key, result)
        else:
            result[key] = str(v)


def t(key: str, fallback: str | None = None, **kwargs: Any) -> str:
    """Look up an i18n key and render with str.format().

    Priority: current language -> English fallback -> fallback arg -> key itself.

    Args:
        key: Dotted key, e.g. "finding.mem_coalesce.title".
        fallback: Value to return if key not found in any pack.
                  If None, returns the key itself.
        **kwargs: Named format arguments for template rendering.

    Returns:
        Rendered string in the current language.

    Example:
        t("finding.mem_coalesce.detail", efficiency=42, threshold=80)
        -> "Global memory coalescing efficiency is 42%, below 80% threshold."
    """
    # Try current language first, then English fallback
    template = None
    if _current_lang in _packs:
        template = _packs[_current_lang].get(key)
    if template is None and "en" in _packs:
        template = _packs["en"].get(key)
    if template is None:
        return fallback if fallback is not None else key

    if kwargs:
        try:
            return template.format(**kwargs)
        except (KeyError, IndexError):
            return template
    return template


def current_lang() -> str:
    """Return the current language code."""
    return _current_lang
