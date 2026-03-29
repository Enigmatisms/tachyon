"""Unit tests for SkillRegistry."""
from pathlib import Path

import pytest

from tachyon.skills import SkillEntry, SkillRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_skill(tmp_path: Path, name: str, content: str) -> Path:
    """Write a .md skill file and return its path."""
    p = tmp_path / f"{name}.md"
    p.write_text(content, encoding="utf-8")
    return p


_FULL_SKILL = """\
---
tags: memory-bound, shared-memory
mode: evolve, chat
brief: Shared memory tiling routes
---

# Shared Memory Tiling

Use shared memory to reduce DRAM traffic.
"""

_COMPUTE_SKILL = """\
---
tags: compute-bound
mode: evolve
brief: Loop unrolling techniques
---

# Loop Unrolling

Unroll inner loops for ILP.
"""

_NO_MODE_SKILL = """\
---
tags: general
brief: General CUDA tips
---

# General Tips

Always profile before optimizing.
"""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestSkillParsing:

    def test_parse_valid_frontmatter(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "test", _FULL_SKILL)
        reg = SkillRegistry(tmp_path)
        assert reg.count == 1
        entry = list(reg._skills.values())[0]
        assert entry.name == "test"
        assert "memory-bound" in entry.tags
        assert "shared-memory" in entry.tags
        assert "evolve" in entry.modes
        assert "chat" in entry.modes
        assert entry.brief == "Shared memory tiling routes"
        assert "shared memory" in entry.content.lower()

    def test_parse_missing_tags(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "notags", """\
---
brief: No tags skill
---

Some content here.
""")
        reg = SkillRegistry(tmp_path)
        assert reg.count == 1
        entry = list(reg._skills.values())[0]
        assert entry.tags == frozenset()
        assert entry.brief == "No tags skill"

    def test_parse_no_frontmatter_skipped(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "readme", "# README\n\nThis is a readme.")
        reg = SkillRegistry(tmp_path)
        assert reg.count == 0

    def test_parse_empty_file_skipped(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "empty", "")
        reg = SkillRegistry(tmp_path)
        assert reg.count == 0

    def test_parse_frontmatter_only_no_content_skipped(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "fmonly", "---\ntags: x\n---\n")
        reg = SkillRegistry(tmp_path)
        assert reg.count == 0

    def test_tags_normalized_lowercase(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "norm", """\
---
tags: Memory-Bound, COMPUTE-BOUND
mode: Evolve
brief: test
---

Content.
""")
        reg = SkillRegistry(tmp_path)
        entry = list(reg._skills.values())[0]
        assert "memory-bound" in entry.tags
        assert "compute-bound" in entry.tags
        assert "evolve" in entry.modes


# ---------------------------------------------------------------------------
# Registry scanning
# ---------------------------------------------------------------------------


class TestSkillRegistry:

    def test_empty_directory(self, tmp_path: Path) -> None:
        reg = SkillRegistry(tmp_path)
        assert reg.count == 0

    def test_nonexistent_directory(self) -> None:
        reg = SkillRegistry(Path("/nonexistent/path/skills"))
        assert reg.count == 0

    def test_scan_multiple_files(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "a", _FULL_SKILL)
        _write_skill(tmp_path, "b", _COMPUTE_SKILL)
        reg = SkillRegistry(tmp_path)
        assert reg.count == 2

    def test_skip_non_md_files(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("not a skill")
        _write_skill(tmp_path, "real", _FULL_SKILL)
        reg = SkillRegistry(tmp_path)
        assert reg.count == 1

    def test_deterministic_order(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "z_skill", _FULL_SKILL)
        _write_skill(tmp_path, "a_skill", _COMPUTE_SKILL)
        reg = SkillRegistry(tmp_path)
        names = list(reg._skills.keys())
        assert names == sorted(names)

    def test_multiple_directories(self, tmp_path: Path) -> None:
        d1 = tmp_path / "dir1"
        d2 = tmp_path / "dir2"
        d1.mkdir()
        d2.mkdir()
        _write_skill(d1, "skill_a", _FULL_SKILL)
        _write_skill(d2, "skill_b", _COMPUTE_SKILL)
        reg = SkillRegistry(d1, d2)
        assert reg.count == 2


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class TestQuery:

    def test_query_by_tag(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "mem", _FULL_SKILL)
        _write_skill(tmp_path, "comp", _COMPUTE_SKILL)
        reg = SkillRegistry(tmp_path)
        result = reg.query(tags={"memory-bound"})
        assert "Shared Memory Tiling" in result
        assert "Loop Unrolling" not in result

    def test_query_by_mode(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "mem", _FULL_SKILL)        # mode: evolve, chat
        _write_skill(tmp_path, "comp", _COMPUTE_SKILL)    # mode: evolve
        reg = SkillRegistry(tmp_path)
        result = reg.query(mode="chat")
        assert "Shared Memory Tiling" in result
        assert "Loop Unrolling" not in result

    def test_query_combined_tag_and_mode(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "mem", _FULL_SKILL)
        _write_skill(tmp_path, "comp", _COMPUTE_SKILL)
        reg = SkillRegistry(tmp_path)
        result = reg.query(tags={"compute-bound"}, mode="evolve")
        assert "Loop Unrolling" in result
        assert "Shared Memory" not in result

    def test_query_no_match(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "mem", _FULL_SKILL)
        reg = SkillRegistry(tmp_path)
        assert reg.query(tags={"latency-bound"}) == ""

    def test_query_empty_registry(self, tmp_path: Path) -> None:
        reg = SkillRegistry(tmp_path)
        assert reg.query(tags={"memory-bound"}) == ""

    def test_query_empty_modes_matches_all(self, tmp_path: Path) -> None:
        """Skills without mode field should match any mode query."""
        _write_skill(tmp_path, "general", _NO_MODE_SKILL)
        reg = SkillRegistry(tmp_path)
        assert reg.query(mode="evolve") != ""
        assert reg.query(mode="chat") != ""

    def test_query_max_chars_truncation(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "mem", _FULL_SKILL)
        _write_skill(tmp_path, "comp", _COMPUTE_SKILL)
        reg = SkillRegistry(tmp_path)
        # Very tight budget — should fall back to briefs
        result = reg.query(tags={"memory-bound", "compute-bound"}, max_chars=50)
        # At least something is returned (brief or partial)
        assert len(result) <= 50 or result == ""

    def test_query_no_filters_returns_all(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "mem", _FULL_SKILL)
        _write_skill(tmp_path, "comp", _COMPUTE_SKILL)
        reg = SkillRegistry(tmp_path)
        result = reg.query()
        assert "Shared Memory Tiling" in result
        assert "Loop Unrolling" in result


# ---------------------------------------------------------------------------
# Query brief
# ---------------------------------------------------------------------------


class TestQueryBrief:

    def test_brief_format(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "mem", _FULL_SKILL)
        reg = SkillRegistry(tmp_path)
        result = reg.query_brief(tags={"memory-bound"})
        assert "- **mem**:" in result
        assert "Shared memory tiling routes" in result

    def test_brief_no_match(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "mem", _FULL_SKILL)
        reg = SkillRegistry(tmp_path)
        assert reg.query_brief(tags={"latency-bound"}) == ""

    def test_brief_missing_brief_field(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "nobrief", """\
---
tags: memory-bound
---

Content without brief.
""")
        reg = SkillRegistry(tmp_path)
        result = reg.query_brief(tags={"memory-bound"})
        assert "(no description)" in result


# ---------------------------------------------------------------------------
# Integration with persona builders
# ---------------------------------------------------------------------------


class TestPersonaIntegration:

    def test_evolve_system_prompt_with_skills(self) -> None:
        from tachyon.evolve.persona import build_evolve_system_prompt
        from tachyon.tools.registry import ToolRegistry

        reg = ToolRegistry()
        prompt = build_evolve_system_prompt(
            reg, skill_knowledge="### mem\n\nUse shared memory.",
        )
        assert "## Optimization Knowledge" in prompt
        assert "Use shared memory" in prompt

    def test_evolve_system_prompt_without_skills(self) -> None:
        from tachyon.evolve.persona import build_evolve_system_prompt
        from tachyon.tools.registry import ToolRegistry

        reg = ToolRegistry()
        prompt = build_evolve_system_prompt(reg)
        assert "## Optimization Knowledge" not in prompt

    def test_evolve_lean_with_brief(self) -> None:
        from tachyon.evolve.persona import build_evolve_lean_prompt
        from tachyon.tools.registry import ToolRegistry

        reg = ToolRegistry()
        prompt = build_evolve_lean_prompt(
            reg, skill_brief="- **mem**: tiling tips",
        )
        assert "## Optimization Skills" in prompt
        assert "tiling tips" in prompt

    def test_evolve_lean_without_brief(self) -> None:
        from tachyon.evolve.persona import build_evolve_lean_prompt
        from tachyon.tools.registry import ToolRegistry

        reg = ToolRegistry()
        prompt = build_evolve_lean_prompt(reg)
        assert "## Optimization Skills" not in prompt

    def test_chat_system_prompt_with_skills(self) -> None:
        from tachyon.agent.persona import build_system_prompt
        from tachyon.tools.registry import ToolRegistry

        reg = ToolRegistry()
        prompt = build_system_prompt(
            reg, skill_knowledge="### tips\n\nProfile first.",
        )
        assert "## CUDA Optimization Knowledge" in prompt
        assert "Profile first" in prompt

    def test_chat_system_prompt_without_skills(self) -> None:
        from tachyon.agent.persona import build_system_prompt
        from tachyon.tools.registry import ToolRegistry

        reg = ToolRegistry()
        prompt = build_system_prompt(reg)
        assert "## CUDA Optimization Knowledge" not in prompt

    def test_chat_lean_with_brief(self) -> None:
        from tachyon.agent.persona import build_lean_system_prompt
        from tachyon.tools.registry import ToolRegistry

        reg = ToolRegistry()
        prompt = build_lean_system_prompt(
            reg, skill_brief="- **tips**: general advice",
        )
        assert "## Optimization Skills" in prompt

    def test_chat_lean_without_brief(self) -> None:
        from tachyon.agent.persona import build_lean_system_prompt
        from tachyon.tools.registry import ToolRegistry

        reg = ToolRegistry()
        prompt = build_lean_system_prompt(reg)
        assert "## Optimization Skills" not in prompt
