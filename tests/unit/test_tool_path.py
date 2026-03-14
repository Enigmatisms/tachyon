"""Unit tests for ToolPathResolver — 5-level priority detection, caching, error handling."""
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tachyon.config.settings import TachyonConfig
from tachyon.errors.handler import ErrorCode
from tachyon.profiler.tool_path import GLOB_SEARCH_PATTERNS, TOOL_NAMES, ToolPathResolver

# ━━━━━━━━━━━━━━━━━━━━━━━ Helpers ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _make_resolver(**tool_paths: str | None) -> ToolPathResolver:
    """Create a ToolPathResolver with optional explicit tool paths in config."""
    cfg = TachyonConfig()
    for attr, val in tool_paths.items():
        setattr(cfg.tools, attr, val)
    return ToolPathResolver(config=cfg)


def _patch_valid_file(target_path: str):
    """Return side_effect functions that make only *target_path* pass is_file + X_OK."""

    original_is_file = Path.is_file
    original_access = os.access

    def mock_is_file(self):
        if str(self) == target_path:
            return True
        return False

    def mock_access(p, mode, **kw):
        if str(p) == target_path and mode == os.X_OK:
            return True
        return False

    return mock_is_file, mock_access


def _patch_resolve_identity():
    """Make Path.resolve() return the path itself (no symlink resolution)."""
    return lambda self: self


# ━━━━━━━━━━━━━━━━━━━━━━━ Priority 1: Config explicit path ━━━━━━━━━━━


class TestConfigExplicitPath:
    """Priority 1 — user-configured path wins over all auto-detection."""

    def test_config_explicit_path_highest_priority(self):
        """Config path wins even when shutil.which() would also find the tool."""
        config_path = "/opt/custom/ncu"
        resolver = _make_resolver(ncu_path=config_path)
        mock_is_file, mock_access = _patch_valid_file(config_path)

        with (
            patch.object(Path, "is_file", mock_is_file),
            patch("os.access", mock_access),
            patch.object(Path, "resolve", _patch_resolve_identity()),
            patch("shutil.which", return_value="/usr/bin/ncu"),
        ):
            result = resolver.resolve("ncu")

        assert result == config_path

    def test_config_invalid_falls_through(self):
        """Config path exists but is not executable -> fall through to which."""
        config_path = "/opt/broken/ncu"
        which_path = "/usr/bin/ncu"
        resolver = _make_resolver(ncu_path=config_path)

        def mock_is_file(self):
            # Config path file exists but won't pass os.access
            if str(self) == config_path:
                return True
            return False

        def mock_access(p, mode, **kw):
            # Config path is NOT executable
            if str(p) == config_path:
                return False
            return False

        with (
            patch.object(Path, "is_file", mock_is_file),
            patch("os.access", mock_access),
            patch.object(Path, "resolve", _patch_resolve_identity()),
            patch("shutil.which", return_value=which_path),
        ):
            result = resolver.resolve("ncu")

        assert result == which_path

    def test_config_path_nonexistent_falls_through(self):
        """Config path file does not exist -> fall through to which."""
        config_path = "/nonexistent/ncu"
        which_path = "/usr/local/bin/ncu"
        resolver = _make_resolver(ncu_path=config_path)

        with (
            patch.object(Path, "is_file", lambda self: False),
            patch("os.access", lambda p, m, **kw: False),
            patch.object(Path, "resolve", _patch_resolve_identity()),
            patch("shutil.which", return_value=which_path),
        ):
            result = resolver.resolve("ncu")

        assert result == which_path


# ━━━━━━━━━━━━━━━━━━━━━━━ Priority 2: shutil.which ━━━━━━━━━━━━━━━━


class TestWhichFallback:
    """Priority 2 — system PATH via shutil.which."""

    def test_which_fallback(self):
        """No config path -> which() finds the tool."""
        which_path = "/usr/local/bin/ncu"
        resolver = _make_resolver()  # No ncu_path configured

        with (
            patch("shutil.which", return_value=which_path),
            patch.object(Path, "resolve", _patch_resolve_identity()),
        ):
            result = resolver.resolve("ncu")

        assert result == which_path

    def test_which_returns_none_continues(self):
        """which() returns None -> continue to CUDA_HOME check."""
        cuda_home_path = "/opt/cuda-12.4/bin/ncu"
        resolver = _make_resolver()

        mock_is_file, mock_access = _patch_valid_file(cuda_home_path)

        with (
            patch("shutil.which", return_value=None),
            patch.dict(os.environ, {"CUDA_HOME": "/opt/cuda-12.4"}, clear=False),
            patch.object(Path, "is_file", mock_is_file),
            patch("os.access", mock_access),
            patch.object(Path, "resolve", _patch_resolve_identity()),
        ):
            result = resolver.resolve("ncu")

        assert result == cuda_home_path


# ━━━━━━━━━━━━━━━━━━━━━━━ Priority 3: CUDA_HOME ━━━━━━━━━━━━━━━━━━━━


class TestCudaHomeFallback:
    """Priority 3 — $CUDA_HOME/bin/<tool>."""

    def test_cuda_home_fallback(self):
        """CUDA_HOME env var set with valid binary."""
        cuda_home_bin = "/opt/cuda-12.6/bin/nvdisasm"
        resolver = _make_resolver()

        mock_is_file, mock_access = _patch_valid_file(cuda_home_bin)

        with (
            patch("shutil.which", return_value=None),
            patch.dict(os.environ, {"CUDA_HOME": "/opt/cuda-12.6"}, clear=False),
            patch.object(Path, "is_file", mock_is_file),
            patch("os.access", mock_access),
            patch.object(Path, "resolve", _patch_resolve_identity()),
        ):
            result = resolver.resolve("nvdisasm")

        assert result == cuda_home_bin

    def test_cuda_home_not_set(self):
        """CUDA_HOME not in env -> skip to default path."""
        default_path = "/usr/local/cuda/bin/ncu"
        resolver = _make_resolver()

        mock_is_file, mock_access = _patch_valid_file(default_path)

        with (
            patch("shutil.which", return_value=None),
            patch.dict(os.environ, {}, clear=False),
            patch.object(os.environ, "get", lambda key, default=None: None if key == "CUDA_HOME" else os.environ.get(key, default)),
            patch.object(Path, "is_file", mock_is_file),
            patch("os.access", mock_access),
            patch.object(Path, "resolve", _patch_resolve_identity()),
        ):
            # This may or may not hit default path depending on CUDA_HOME presence.
            # Use a simpler approach: remove CUDA_HOME from env
            pass

        # Re-do with clean env patch
        env_copy = os.environ.copy()
        env_copy.pop("CUDA_HOME", None)
        with (
            patch("shutil.which", return_value=None),
            patch.dict(os.environ, env_copy, clear=True),
            patch.object(Path, "is_file", mock_is_file),
            patch("os.access", mock_access),
            patch.object(Path, "resolve", _patch_resolve_identity()),
        ):
            result = resolver.resolve("ncu")

        assert result == default_path

    def test_cuda_home_binary_not_executable(self):
        """CUDA_HOME set but binary not executable -> falls to default."""
        default_path = "/usr/local/cuda/bin/cuobjdump"
        resolver = _make_resolver()

        def mock_is_file(self):
            s = str(self)
            if s == "/opt/cuda/bin/cuobjdump":
                return True  # Exists but not executable
            if s == default_path:
                return True
            return False

        def mock_access(p, mode, **kw):
            s = str(p)
            if s == "/opt/cuda/bin/cuobjdump":
                return False  # Not executable
            if s == default_path and mode == os.X_OK:
                return True
            return False

        with (
            patch("shutil.which", return_value=None),
            patch.dict(os.environ, {"CUDA_HOME": "/opt/cuda"}, clear=False),
            patch.object(Path, "is_file", mock_is_file),
            patch("os.access", mock_access),
            patch.object(Path, "resolve", _patch_resolve_identity()),
        ):
            result = resolver.resolve("cuobjdump")

        assert result == default_path


# ━━━━━━━━━━━━━━━━━━━━━━━ Priority 4: Default /usr/local/cuda/bin ━━━━


class TestDefaultCudaPath:
    """Priority 4 — /usr/local/cuda/bin/<tool>."""

    def test_default_cuda_path(self):
        """Default CUDA install path found."""
        default_path = "/usr/local/cuda/bin/ncu"
        resolver = _make_resolver()

        mock_is_file, mock_access = _patch_valid_file(default_path)
        env_copy = os.environ.copy()
        env_copy.pop("CUDA_HOME", None)

        with (
            patch("shutil.which", return_value=None),
            patch.dict(os.environ, env_copy, clear=True),
            patch.object(Path, "is_file", mock_is_file),
            patch("os.access", mock_access),
            patch.object(Path, "resolve", _patch_resolve_identity()),
        ):
            result = resolver.resolve("ncu")

        assert result == default_path

    def test_default_cuda_path_not_found_continues_to_glob(self):
        """Default path missing -> continues to glob search."""
        glob_match = "/opt/nvidia/nsight-compute/2024.3/ncu"
        resolver = _make_resolver()

        def mock_is_file(self):
            return str(self) == glob_match

        def mock_access(p, mode, **kw):
            return str(p) == glob_match and mode == os.X_OK

        env_copy = os.environ.copy()
        env_copy.pop("CUDA_HOME", None)

        with (
            patch("shutil.which", return_value=None),
            patch.dict(os.environ, env_copy, clear=True),
            patch.object(Path, "is_file", mock_is_file),
            patch("os.access", mock_access),
            patch.object(Path, "resolve", _patch_resolve_identity()),
            patch("tachyon.profiler.tool_path.globmod.glob", return_value=[glob_match]),
        ):
            result = resolver.resolve("ncu")

        assert result == glob_match


# ━━━━━━━━━━━━━━━━━━━━━━━ Priority 5: Glob patterns ━━━━━━━━━━━━━━━━━


class TestGlobFallback:
    """Priority 5 — glob multi-version patterns."""

    def test_glob_fallback(self):
        """Only glob pattern matches."""
        glob_match = "/opt/nvidia/nsight-compute/2025.1/ncu"
        resolver = _make_resolver()

        mock_is_file, mock_access = _patch_valid_file(glob_match)
        env_copy = os.environ.copy()
        env_copy.pop("CUDA_HOME", None)

        with (
            patch("shutil.which", return_value=None),
            patch.dict(os.environ, env_copy, clear=True),
            patch.object(Path, "is_file", mock_is_file),
            patch("os.access", mock_access),
            patch.object(Path, "resolve", _patch_resolve_identity()),
            patch("tachyon.profiler.tool_path.globmod.glob", return_value=[glob_match]),
        ):
            result = resolver.resolve("ncu")

        assert result == glob_match

    def test_glob_selects_newest_first(self):
        """Glob matches sorted reverse (newest version first)."""
        matches = [
            "/opt/nvidia/nsight-compute/2023.1/ncu",
            "/opt/nvidia/nsight-compute/2025.2/ncu",
            "/opt/nvidia/nsight-compute/2024.3/ncu",
        ]
        resolver = _make_resolver()

        # After reverse sort: 2025.2, 2024.3, 2023.1
        # First one that passes is_file + X_OK wins
        newest = "/opt/nvidia/nsight-compute/2025.2/ncu"

        def mock_is_file(self):
            return str(self) in matches

        def mock_access(p, mode, **kw):
            return str(p) in matches and mode == os.X_OK

        env_copy = os.environ.copy()
        env_copy.pop("CUDA_HOME", None)

        with (
            patch("shutil.which", return_value=None),
            patch.dict(os.environ, env_copy, clear=True),
            patch.object(Path, "is_file", mock_is_file),
            patch("os.access", mock_access),
            patch.object(Path, "resolve", _patch_resolve_identity()),
            patch("tachyon.profiler.tool_path.globmod.glob", return_value=matches),
        ):
            result = resolver.resolve("ncu")

        assert result == newest

    def test_glob_nvdisasm(self):
        """Glob pattern for nvdisasm uses /usr/local/cuda-*/bin/nvdisasm."""
        glob_match = "/usr/local/cuda-12.4/bin/nvdisasm"
        resolver = _make_resolver()

        mock_is_file, mock_access = _patch_valid_file(glob_match)
        env_copy = os.environ.copy()
        env_copy.pop("CUDA_HOME", None)

        with (
            patch("shutil.which", return_value=None),
            patch.dict(os.environ, env_copy, clear=True),
            patch.object(Path, "is_file", mock_is_file),
            patch("os.access", mock_access),
            patch.object(Path, "resolve", _patch_resolve_identity()),
            patch(
                "tachyon.profiler.tool_path.globmod.glob",
                return_value=[glob_match],
            ),
        ):
            result = resolver.resolve("nvdisasm")

        assert result == glob_match

    def test_glob_no_executable_match(self):
        """Glob finds files but none are executable -> not found."""
        matches = ["/opt/nvidia/nsight-compute/2025.1/ncu"]
        resolver = _make_resolver()

        env_copy = os.environ.copy()
        env_copy.pop("CUDA_HOME", None)

        with (
            patch("shutil.which", return_value=None),
            patch.dict(os.environ, env_copy, clear=True),
            patch.object(Path, "is_file", lambda self: str(self) in matches),
            patch("os.access", lambda p, m, **kw: False),  # Not executable
            patch.object(Path, "resolve", _patch_resolve_identity()),
            patch("tachyon.profiler.tool_path.globmod.glob", return_value=matches),
        ):
            with pytest.raises(FileNotFoundError):
                resolver.resolve("ncu")


# ━━━━━━━━━━━━━━━━━━━━━━━ Not found ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestNotFound:
    """Tool not found in any search path."""

    def test_not_found_raises(self):
        """All 5 levels fail -> FileNotFoundError with actionable message."""
        resolver = _make_resolver()

        env_copy = os.environ.copy()
        env_copy.pop("CUDA_HOME", None)

        with (
            patch("shutil.which", return_value=None),
            patch.dict(os.environ, env_copy, clear=True),
            patch.object(Path, "is_file", lambda self: False),
            patch("os.access", lambda p, m, **kw: False),
            patch("tachyon.profiler.tool_path.globmod.glob", return_value=[]),
        ):
            with pytest.raises(FileNotFoundError, match="ncu not found"):
                resolver.resolve("ncu")

    def test_not_found_message_includes_search_locations(self):
        """Error message mentions all search locations."""
        resolver = _make_resolver()

        env_copy = os.environ.copy()
        env_copy.pop("CUDA_HOME", None)

        with (
            patch("shutil.which", return_value=None),
            patch.dict(os.environ, env_copy, clear=True),
            patch.object(Path, "is_file", lambda self: False),
            patch("os.access", lambda p, m, **kw: False),
            patch("tachyon.profiler.tool_path.globmod.glob", return_value=[]),
        ):
            with pytest.raises(FileNotFoundError) as exc_info:
                resolver.resolve("ncu")

            msg = str(exc_info.value)
            assert "config.toml" in msg
            assert "PATH" in msg
            assert "CUDA_HOME" in msg
            assert "/usr/local/cuda/bin" in msg

    def test_not_found_nvdisasm(self):
        """nvdisasm not found raises with correct tool name."""
        resolver = _make_resolver()

        env_copy = os.environ.copy()
        env_copy.pop("CUDA_HOME", None)

        with (
            patch("shutil.which", return_value=None),
            patch.dict(os.environ, env_copy, clear=True),
            patch.object(Path, "is_file", lambda self: False),
            patch("os.access", lambda p, m, **kw: False),
            patch("tachyon.profiler.tool_path.globmod.glob", return_value=[]),
        ):
            with pytest.raises(FileNotFoundError, match="nvdisasm not found"):
                resolver.resolve("nvdisasm")


# ━━━━━━━━━━━━━━━━━━━━━━━ Caching ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCaching:
    """Session cache prevents repeated filesystem probing."""

    def test_cache_prevents_repeated_detection(self):
        """resolve() called twice -> _detect only called once."""
        resolver = _make_resolver()
        resolver._detect = MagicMock(return_value="/usr/bin/ncu")

        first = resolver.resolve("ncu")
        second = resolver.resolve("ncu")

        assert first == "/usr/bin/ncu"
        assert second == "/usr/bin/ncu"
        resolver._detect.assert_called_once_with("ncu")

    def test_cache_not_found_raises(self):
        """After caching None, second resolve() raises immediately without re-detect."""
        resolver = _make_resolver()
        resolver._detect = MagicMock(return_value=None)

        with pytest.raises(FileNotFoundError):
            resolver.resolve("ncu")

        # Second call should raise from cache without calling _detect again
        with pytest.raises(FileNotFoundError, match="not found"):
            resolver.resolve("ncu")

        resolver._detect.assert_called_once_with("ncu")

    def test_cache_isolated_per_tool(self):
        """Different tools have independent cache entries."""
        resolver = _make_resolver()
        resolver._detect = MagicMock(
            side_effect=lambda name: f"/usr/bin/{name}"
        )

        assert resolver.resolve("ncu") == "/usr/bin/ncu"
        assert resolver.resolve("nvdisasm") == "/usr/bin/nvdisasm"
        assert resolver._detect.call_count == 2

        # Now cached — no more _detect calls
        assert resolver.resolve("ncu") == "/usr/bin/ncu"
        assert resolver.resolve("nvdisasm") == "/usr/bin/nvdisasm"
        assert resolver._detect.call_count == 2

    def test_cache_starts_empty(self):
        """Fresh resolver has empty cache."""
        resolver = _make_resolver()
        assert resolver._cache == {}

    def test_cache_populated_after_resolve(self):
        """Cache contains entry after successful resolve."""
        resolver = _make_resolver()
        resolver._detect = MagicMock(return_value="/usr/bin/ncu")

        resolver.resolve("ncu")
        assert "ncu" in resolver._cache
        assert resolver._cache["ncu"] == "/usr/bin/ncu"

    def test_cache_stores_none_on_not_found(self):
        """Cache stores None when tool is not found."""
        resolver = _make_resolver()
        resolver._detect = MagicMock(return_value=None)

        with pytest.raises(FileNotFoundError):
            resolver.resolve("ncu")

        assert "ncu" in resolver._cache
        assert resolver._cache["ncu"] is None


# ━━━━━━━━━━━━━━━━━━━━━━━ resolve_safe ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestResolveSafe:
    """Non-throwing resolve_safe returns ToolResult."""

    def test_resolve_safe_success(self):
        """Successful resolution returns ToolResult.ok with path."""
        resolver = _make_resolver()
        resolver._detect = MagicMock(return_value="/usr/bin/ncu")

        result = resolver.resolve_safe("ncu")

        assert result.success is True
        assert result.data == "/usr/bin/ncu"
        assert result.error is None

    def test_resolve_safe_failure(self):
        """Failed resolution returns ToolResult.fail with TOOL_NOT_FOUND."""
        resolver = _make_resolver()
        resolver._detect = MagicMock(return_value=None)

        result = resolver.resolve_safe("ncu")

        assert result.success is False
        assert result.data is None
        assert result.error is not None
        assert result.error.code == ErrorCode.TOOL_NOT_FOUND
        assert "ncu" in result.error.message

    def test_resolve_safe_failure_has_suggestion(self):
        """Failed result includes actionable suggestion."""
        resolver = _make_resolver()
        resolver._detect = MagicMock(return_value=None)

        result = resolver.resolve_safe("ncu")

        assert result.error.suggestion
        assert "config.toml" in result.error.suggestion
        assert "ncu_path" in result.error.suggestion

    def test_resolve_safe_uses_cache(self):
        """resolve_safe also benefits from caching."""
        resolver = _make_resolver()
        resolver._detect = MagicMock(return_value="/usr/bin/ncu")

        r1 = resolver.resolve_safe("ncu")
        r2 = resolver.resolve_safe("ncu")

        assert r1.success is True
        assert r2.success is True
        resolver._detect.assert_called_once()


# ━━━━━━━━━━━━━━━━━━━━━━━ All three tools ━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAllThreeTools:
    """ncu, nvdisasm, cuobjdump all resolve independently."""

    def test_all_three_tools(self):
        """Each tool resolves to its own path independently."""
        resolver = _make_resolver()
        resolver._detect = MagicMock(
            side_effect=lambda name: f"/usr/local/cuda/bin/{name}"
        )

        ncu = resolver.resolve("ncu")
        nvdisasm = resolver.resolve("nvdisasm")
        cuobjdump = resolver.resolve("cuobjdump")

        assert ncu == "/usr/local/cuda/bin/ncu"
        assert nvdisasm == "/usr/local/cuda/bin/nvdisasm"
        assert cuobjdump == "/usr/local/cuda/bin/cuobjdump"

    def test_mixed_success_failure(self):
        """Some tools found, others not — independent behavior."""
        resolver = _make_resolver()

        def mock_detect(name):
            if name == "ncu":
                return "/usr/bin/ncu"
            return None

        resolver._detect = MagicMock(side_effect=mock_detect)

        assert resolver.resolve("ncu") == "/usr/bin/ncu"

        with pytest.raises(FileNotFoundError):
            resolver.resolve("nvdisasm")

        # ncu still works from cache
        assert resolver.resolve("ncu") == "/usr/bin/ncu"

    def test_each_tool_has_config_attribute(self):
        """Config has separate path attributes for each tool."""
        cfg = TachyonConfig()
        cfg.tools.ncu_path = "/a/ncu"
        cfg.tools.nvdisasm_path = "/b/nvdisasm"
        cfg.tools.cuobjdump_path = "/c/cuobjdump"

        resolver = ToolPathResolver(config=cfg)

        # Verify _get_config_path reads the correct attribute
        assert resolver._get_config_path("ncu") == "/a/ncu"
        assert resolver._get_config_path("nvdisasm") == "/b/nvdisasm"
        assert resolver._get_config_path("cuobjdump") == "/c/cuobjdump"


# ━━━━━━━━━━━━━━━━━━━━━━━ Unknown tool name ━━━━━━━━━━━━━━━━━━━━━━━━


class TestUnknownToolName:
    """Tool name not in TOOL_NAMES dict still works (uses name as binary)."""

    def test_unknown_tool_name_uses_name_as_binary(self):
        """Unknown tool uses the tool_name itself as the binary name for which()."""
        resolver = _make_resolver()

        with (
            patch("shutil.which", return_value="/usr/bin/custom_tool"),
            patch.object(Path, "resolve", _patch_resolve_identity()),
        ):
            result = resolver.resolve("custom_tool")

        assert result == "/usr/bin/custom_tool"

    def test_unknown_tool_name_no_config(self):
        """Unknown tool has no config attribute -> _get_config_path returns None."""
        resolver = _make_resolver()
        assert resolver._get_config_path("custom_tool") is None

    def test_unknown_tool_no_glob_patterns(self):
        """Unknown tool has no glob patterns -> only checks config/which/CUDA_HOME/default."""
        resolver = _make_resolver()

        env_copy = os.environ.copy()
        env_copy.pop("CUDA_HOME", None)

        with (
            patch("shutil.which", return_value=None),
            patch.dict(os.environ, env_copy, clear=True),
            patch.object(Path, "is_file", lambda self: False),
            patch("os.access", lambda p, m, **kw: False),
            patch("tachyon.profiler.tool_path.globmod.glob", return_value=[]) as mock_glob,
        ):
            with pytest.raises(FileNotFoundError):
                resolver.resolve("custom_tool")

        # glob should NOT have been called since no patterns exist for "custom_tool"
        mock_glob.assert_not_called()


# ━━━━━━━━━━━━━━━━━━━━━━━ Module-level constants ━━━━━━━━━━━━━━━━━━━


class TestModuleConstants:
    """Verify module-level TOOL_NAMES and GLOB_SEARCH_PATTERNS."""

    def test_tool_names_contains_three(self):
        assert "ncu" in TOOL_NAMES
        assert "nvdisasm" in TOOL_NAMES
        assert "cuobjdump" in TOOL_NAMES
        assert len(TOOL_NAMES) == 3

    def test_tool_names_values(self):
        assert TOOL_NAMES["ncu"] == "ncu"
        assert TOOL_NAMES["nvdisasm"] == "nvdisasm"
        assert TOOL_NAMES["cuobjdump"] == "cuobjdump"

    def test_glob_patterns_ncu(self):
        patterns = GLOB_SEARCH_PATTERNS["ncu"]
        assert any("nsight-compute" in p for p in patterns)

    def test_glob_patterns_nvdisasm(self):
        patterns = GLOB_SEARCH_PATTERNS["nvdisasm"]
        assert any("cuda-*" in p for p in patterns)

    def test_glob_patterns_cuobjdump(self):
        patterns = GLOB_SEARCH_PATTERNS["cuobjdump"]
        assert any("cuda-*" in p for p in patterns)


# ━━━━━━━━━━━━━━━━━━━━━━━ _get_config_path ━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGetConfigPath:
    """Unit tests for _get_config_path helper."""

    def test_returns_none_when_not_set(self):
        resolver = _make_resolver()
        assert resolver._get_config_path("ncu") is None

    def test_returns_path_when_set(self):
        resolver = _make_resolver(ncu_path="/custom/ncu")
        assert resolver._get_config_path("ncu") == "/custom/ncu"

    def test_returns_none_for_missing_attr(self):
        """Unknown tool -> getattr with default None."""
        resolver = _make_resolver()
        assert resolver._get_config_path("nonexistent_tool") is None


# ━━━━━━━━━━━━━━━━━━━━━━━ Dataclass behavior ━━━━━━━━━━━━━━━━━━━━━━━


class TestDataclassBehavior:
    """ToolPathResolver is a dataclass with proper init and repr."""

    def test_init_requires_config(self):
        """config is required, _cache is auto-initialized."""
        cfg = TachyonConfig()
        resolver = ToolPathResolver(config=cfg)
        assert resolver.config is cfg
        assert resolver._cache == {}

    def test_cache_not_in_init(self):
        """_cache is field(init=False), cannot be passed to __init__."""
        cfg = TachyonConfig()
        # _cache should not appear in repr
        r = repr(ToolPathResolver(config=cfg))
        assert "_cache" not in r

    def test_separate_instances_have_separate_caches(self):
        """Two resolvers do not share cache."""
        r1 = _make_resolver()
        r2 = _make_resolver()
        r1._cache["ncu"] = "/a"
        assert "ncu" not in r2._cache


# ━━━━━━━━━━━━━━━━━━━━━━━ Logging ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLogging:
    """Verify logging behavior."""

    def test_logs_resolved_path(self, caplog):
        """Successful resolution logs at INFO level."""
        import logging

        resolver = _make_resolver()
        resolver._detect = MagicMock(return_value="/usr/bin/ncu")

        with caplog.at_level(logging.INFO, logger="tachyon.profiler.tool_path"):
            resolver.resolve("ncu")

        assert any("Resolved ncu" in msg for msg in caplog.messages)
        assert any("/usr/bin/ncu" in msg for msg in caplog.messages)

    def test_logs_warning_on_config_fallthrough(self, caplog):
        """Invalid config path logs a warning."""
        import logging

        config_path = "/opt/broken/ncu"
        resolver = _make_resolver(ncu_path=config_path)

        with (
            caplog.at_level(logging.WARNING, logger="tachyon.profiler.tool_path"),
            patch.object(Path, "is_file", lambda self: str(self) == config_path),
            patch("os.access", lambda p, m, **kw: False),  # Not executable
            patch("shutil.which", return_value="/usr/bin/ncu"),
            patch.object(Path, "resolve", _patch_resolve_identity()),
        ):
            resolver.resolve("ncu")

        assert any("falling through" in msg for msg in caplog.messages)


# ━━━━━━━━━━━━━━━━━━━━━━━ Edge cases ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_config_path_empty_string(self):
        """Empty string config path is falsy -> skipped."""
        resolver = _make_resolver(ncu_path="")

        with (
            patch("shutil.which", return_value="/usr/bin/ncu"),
            patch.object(Path, "resolve", _patch_resolve_identity()),
        ):
            result = resolver.resolve("ncu")

        assert result == "/usr/bin/ncu"

    def test_which_returns_relative_path(self):
        """which() returning a relative path gets resolved to absolute."""
        resolver = _make_resolver()

        with (
            patch("shutil.which", return_value="./ncu"),
            patch.object(Path, "resolve", lambda self: Path("/resolved/ncu")),
        ):
            result = resolver.resolve("ncu")

        assert result == "/resolved/ncu"

    def test_glob_empty_patterns_list(self):
        """Tool with empty glob patterns list -> nothing matched."""
        with patch.dict(GLOB_SEARCH_PATTERNS, {"test_tool": []}):
            resolver = _make_resolver()

            env_copy = os.environ.copy()
            env_copy.pop("CUDA_HOME", None)

            with (
                patch("shutil.which", return_value=None),
                patch.dict(os.environ, env_copy, clear=True),
                patch.object(Path, "is_file", lambda self: False),
                patch("os.access", lambda p, m, **kw: False),
            ):
                with pytest.raises(FileNotFoundError):
                    resolver.resolve("test_tool")

    def test_resolve_called_from_resolve_safe_shares_cache(self):
        """resolve_safe internally calls resolve, shares same cache."""
        resolver = _make_resolver()
        resolver._detect = MagicMock(return_value="/usr/bin/ncu")

        # First via resolve_safe
        result = resolver.resolve_safe("ncu")
        assert result.success is True

        # Then via resolve — should use cache
        path = resolver.resolve("ncu")
        assert path == "/usr/bin/ncu"

        # _detect only called once
        resolver._detect.assert_called_once()
