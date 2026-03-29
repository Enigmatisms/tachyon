"""CUDA optimization skill knowledge base.

Scans ``.md`` files in ``skills/cuda/`` (and optional extra directories),
parses lightweight frontmatter, and provides tag+mode-based queries for
injecting domain knowledge into LLM prompts.

Skill file format::

    ---
    tags: memory-bound, shared-memory
    mode: evolve, chat
    brief: Shared memory tiling routes and prerequisites
    ---

    # Content here ...

Files without valid ``---`` frontmatter are silently skipped.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

_log = logging.getLogger(__name__)

_SKILL_DIR = Path(__file__).parent / "cuda"

# Match the opening frontmatter block: starts with '---\n', ends with '---\n'.
_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\n(.*?\n)---[ \t]*\n", re.DOTALL)

# Key: value lines inside the frontmatter block.
_KV_RE = re.compile(r"^([\w][\w-]*)\s*:\s*(.+)$", re.MULTILINE)


def _parse_csv(value: str) -> frozenset[str]:
    """Parse a comma-separated value string into a lowercase frozenset."""
    return frozenset(
        item
        for raw in value.split(",")
        if (item := raw.strip().lower())
    )


@dataclass(frozen=True)
class SkillEntry:
    """A single parsed skill file."""

    name: str
    path: Path
    tags: frozenset[str]
    modes: frozenset[str]
    brief: str
    content: str


class SkillRegistry:
    """Scans, indexes, and queries ``.md`` skill files.

    When no skill directories are provided, the built-in ``skills/cuda/``
    directory is scanned.  If the directory does not exist or is empty,
    all query methods return ``""`` — zero impact on prompts.
    """

    __slots__ = ("_skills",)

    def __init__(self, *skill_dirs: Path) -> None:
        self._skills: dict[str, SkillEntry] = {}
        dirs = skill_dirs or (_SKILL_DIR,)
        for d in dirs:
            self._scan_dir(d)

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def _scan_dir(self, directory: Path) -> None:
        if not directory.is_dir():
            return
        for path in sorted(directory.glob("*.md")):
            entry = self._parse_file(path)
            if entry is not None:
                if entry.name in self._skills:
                    _log.warning(
                        "Duplicate skill name '%s' — %s overrides %s",
                        entry.name, path, self._skills[entry.name].path,
                    )
                self._skills[entry.name] = entry

    @staticmethod
    def _parse_file(path: Path) -> SkillEntry | None:
        """Parse a single ``.md`` file.  Returns *None* if invalid."""
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            _log.warning("Cannot read skill file: %s", path)
            return None

        if not text.strip():
            return None

        match = _FRONTMATTER_RE.search(text)
        if match is None:
            # No frontmatter — skip (README.md, etc.)
            return None

        fm_block = match.group(1)
        content = text[match.end():].strip()
        if not content:
            return None

        # Parse key-value pairs from frontmatter
        kv: dict[str, str] = {}
        for kv_match in _KV_RE.finditer(fm_block):
            kv[kv_match.group(1).lower()] = kv_match.group(2).strip()

        tags = _parse_csv(kv.get("tags", ""))
        modes = _parse_csv(kv.get("mode", ""))
        brief = kv.get("brief", "")

        return SkillEntry(
            name=path.stem,
            path=path,
            tags=tags,
            modes=modes,
            brief=brief,
            content=content,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        return len(self._skills)

    def _match(
        self,
        entry: SkillEntry,
        tags: set[str] | None,
        mode: str | None,
    ) -> bool:
        """Check whether *entry* matches the given filters."""
        if tags and not (tags & entry.tags):
            return False
        if mode and entry.modes and mode.lower() not in entry.modes:
            return False
        return True

    def query(
        self,
        *,
        tags: set[str] | None = None,
        mode: str | None = None,
        max_chars: int = 4000,
    ) -> str:
        """Return full content of matching skills, capped at *max_chars*.

        Skills are sorted by name for determinism.  When appending a skill
        would exceed the budget, its *brief* is used instead.  If even the
        brief would exceed, the skill is skipped entirely.
        """
        matches = [
            e for e in self._skills.values()
            if self._match(e, tags, mode)
        ]
        if not matches:
            return ""

        parts: list[str] = []
        used = 0
        for entry in matches:
            section = f"### {entry.name}\n\n{entry.content}"
            if used + len(section) + 2 <= max_chars:
                parts.append(section)
                used += len(section) + 2  # +2 for "\n\n" separator
            elif entry.brief:
                fallback = f"- **{entry.name}**: {entry.brief}"
                if used + len(fallback) + 1 <= max_chars:
                    parts.append(fallback)
                    used += len(fallback) + 1

        return "\n\n".join(parts)

    def query_brief(
        self,
        *,
        tags: set[str] | None = None,
        mode: str | None = None,
    ) -> str:
        """Return a compact index of matching skills (name + brief)."""
        matches = [
            e for e in self._skills.values()
            if self._match(e, tags, mode)
        ]
        if not matches:
            return ""

        lines = []
        for entry in matches:
            brief = entry.brief or "(no description)"
            lines.append(f"- **{entry.name}**: {brief}")
        return "\n".join(lines)
