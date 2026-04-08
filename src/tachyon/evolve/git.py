"""Git safety layer — snapshot, rollback, and experiment tagging.

All subprocess calls use ``list`` args (no ``shell=True``) following the
pattern in ``NcuProfiler._run_ncu()`` (profiler/ncu_profiler.py).

Provides both sync and async interfaces:
  - Sync methods: for non-async contexts
  - Async methods: for use within async tool handlers (non-blocking)
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_log = logging.getLogger(__name__)

_TACHYON_GITIGNORE_ENTRY = """\
# Tachyon evolve session data
.tachyon/evolve/
"""

_NCU_GITIGNORE_ENTRY = "*.ncu-rep\n"


class GitRollback:
    """Safe git operations for evolve experiments.

    Provides atomic snapshot/rollback so the working tree is always
    restorable to a known-good state.
    """

    def __init__(self, repo_root: Path | None = None) -> None:
        self.repo_root = repo_root or Path.cwd()
        self._git = "git"

    # --- Public API ---

    def verify_git_repo(self) -> bool:
        """Check that the current directory is inside a git repository."""
        return self._run(["rev-parse", "--is-inside-work-tree"]).returncode == 0

    def auto_commit_uncommitted(self) -> tuple[str, str] | None:
        """Stage all uncommitted changes and create a tachyon-managed commit.

        Returns the commit hash if a commit was created, or None if the
        working tree was already clean.
        """
        if self.check_working_tree_clean():
            return None

        # Stash any staged-but-not-committed work that could conflict
        self._run(["add", "-A"])
        msg = f"[tachyon] auto-commit uncommitted changes — {self._timestamp()}"
        result = self._run(["commit", "--allow-empty", "-m", msg])
        if result.returncode != 0:
            _log.warning("Auto-commit failed: %s", result.stderr.strip())
            return None

        commit_hash = self._run(["rev-parse", "HEAD"]).stdout.strip()
        short_hash = self._run(["rev-parse", "--short", "HEAD"]).stdout.strip()
        _log.info("Auto-committed uncommitted changes as %s", short_hash)
        return commit_hash, short_hash

    def check_working_tree_clean(self) -> bool:
        """Return True if the working tree has no uncommitted changes."""
        result = self._run(["status", "--porcelain"])
        # Empty output = clean
        return not result.stdout.strip()

    def get_current_branch(self) -> str:
        """Return the current branch name."""
        return self._run(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()

    def create_branch(self, name: str) -> bool:
        """Create a new branch from HEAD (force-replace if exists)."""
        self._run(["branch", "-f", name])
        return True

    def checkout(self, ref: str) -> bool:
        """Checkout a branch or commit."""
        return self._run(["checkout", ref]).returncode == 0

    def snapshot(self, message: str = "") -> tuple[str, str]:
        """Create a git commit snapshot of the current state.

        Stages all changes (including untracked files in allowed paths),
        commits, and returns ``(commit_hash, short_hash)``.

        Raises:
            RuntimeError: If the working tree has unstaged changes and
                the commit fails.
        """
        # Stage everything
        self._run(["add", "-A"])
        # Commit
        msg = message or f"[tachyon] evolve snapshot — {self._timestamp()}"
        result = self._run(["commit", "--allow-empty", "-m", msg])
        if result.returncode != 0:
            raise RuntimeError(f"Git snapshot failed: {result.stderr.strip()}")

        commit_hash = self._run(["rev-parse", "HEAD"]).stdout.strip()
        short_hash = self._run(["rev-parse", "--short", "HEAD"]).stdout.strip()
        _log.info("Snapshot created: %s (%s)", short_hash, commit_hash)
        return commit_hash, short_hash

    def rollback(self, commit_hash: str) -> bool:
        """Roll back the working tree to a previous commit.

        Uses ``git reset --hard`` which is safe here because we always
        snapshot before making changes.
        """
        _log.info("Rolling back to %s", commit_hash)
        result = self._run(["reset", "--hard", commit_hash])
        success = result.returncode == 0
        if success:
            _log.info("Rollback to %s successful.", commit_hash)
        else:
            _log.error("Rollback failed: %s", result.stderr.strip())
        return success

    def get_diff(
        self,
        commit_a: str,
        commit_b: str | None = None,
    ) -> str:
        """Get unified diff between two commits.

        If ``commit_b`` is None, compares against the working tree.
        """
        args = ["diff", commit_a]
        if commit_b is not None:
            args.append(commit_b)
        result = self._run(args)
        return result.stdout

    def tag_experiment(self, iteration: int, commit_hash: str) -> None:
        """Tag a commit with the experiment iteration number."""
        tag = f"tachyon-evolve-{iteration}"
        # Delete existing tag if any
        self._run(["tag", "-d", tag], check=False)
        self._run(["tag", tag, commit_hash])
        _log.info("Tagged %s as %s", commit_hash, tag)

    def ensure_gitignore_entries(self) -> None:
        """Add .tachyon/evolve/ and *.ncu-rep to .gitignore if not present."""
        gitignore = self.repo_root / ".gitignore"
        existing = ""
        if gitignore.is_file():
            existing = gitignore.read_text(encoding="utf-8")

        additions: list[str] = []
        if ".tachyon/evolve/" not in existing:
            additions.append(_TACHYON_GITIGNORE_ENTRY.strip())
        if "*.ncu-rep" not in existing:
            additions.append(_NCU_GITIGNORE_ENTRY.strip())

        if additions:
            with open(gitignore, "a", encoding="utf-8") as f:
                for entry in additions:
                    f.write(f"\n{entry}\n")
            _log.info("Updated .gitignore with %d entries.", len(additions))

    def get_current_hash(self) -> str:
        """Return the current HEAD commit hash."""
        return self._run(["rev-parse", "HEAD"]).stdout.strip()

    def stage_files(self, files: list[str]) -> bool:
        """Stage specific files (accepted changes) into the index.

        Args:
            files: Absolute or repo-relative file paths.
        """
        if not files:
            return True
        result = self._run(["add", "--"] + files)
        if result.returncode != 0:
            _log.warning("git add failed: %s", result.stderr.strip())
        return result.returncode == 0

    def save_working_state(self) -> str:
        """Save unstaged working tree changes as a patch for rollback.

        Uses ``git diff`` (not ``git diff HEAD``) so that staged (accepted)
        changes are excluded from the patch and preserved on restore.
        """
        result = self._run(["diff"])
        state_dir = self.repo_root / ".tachyon" / "evolve" / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / "working.patch"
        path.write_text(result.stdout, encoding="utf-8")
        return str(path)

    def restore_working_state(self, patch_path: str) -> bool:
        """Restore working tree from a previously saved patch.

        Discards all current changes and applies the saved state.
        If the patch is empty (working tree was clean when saved),
        just resets the working tree to HEAD.
        """
        patch_content = Path(patch_path).read_text(encoding="utf-8").strip()
        self._run(["checkout", "."])
        self._run(["clean", "-fd"])
        if not patch_content:
            return True
        result = self._run(["apply", patch_path], check=False)
        if result.returncode != 0:
            _log.warning("Failed to restore working state: %s", result.stderr.strip())
            return False
        _log.info("Restored working state from %s", patch_path)
        return True

    def reset_to_head(self) -> bool:
        """Reset working tree to HEAD, discarding all uncommitted changes."""
        return self._run(["reset", "--hard", "HEAD"]).returncode == 0

    def cleanup_stale_artifacts(self) -> None:
        """Remove stale evolve artifacts from previous runs.

        Cleans up:
          - .tachyon/evolve/ directory (session.json, snapshots)
          - tachyon-evolve-* git tags
        """
        import shutil

        evolve_dir = self.repo_root / ".tachyon" / "evolve"
        if evolve_dir.is_dir():
            shutil.rmtree(evolve_dir, ignore_errors=True)
            _log.info("Cleaned up stale evolve directory: %s", evolve_dir)

        # Remove stale tachyon-evolve-* tags
        tag_result = self._run(["tag", "-l", "tachyon-evolve-*"], check=False)
        if tag_result.returncode == 0 and tag_result.stdout.strip():
            for tag in tag_result.stdout.strip().splitlines():
                tag = tag.strip()
                if tag:
                    self._run(["tag", "-d", tag], check=False)
            _log.info("Cleaned up stale tachyon-evolve tags.")

    # --- Internal ---

    def _run(
        self,
        args: list[str],
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Run a git command safely (no shell=True)."""
        cmd = [self._git] + args
        _log.debug("Running: %s", cmd)
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.repo_root),
                check=False,
            )
        except FileNotFoundError:
            raise RuntimeError("git is not installed or not in PATH.")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"git command timed out: {cmd}")

    async def _run_async(
        self,
        args: list[str],
        timeout: float = 30.0,
    ) -> subprocess.CompletedProcess[str]:
        """Run a git command asynchronously (non-blocking for event loop).

        This is the async equivalent of `_run`, designed for use within
        async tool handlers to prevent blocking the event loop.
        """
        cmd = [self._git] + args
        _log.debug("Running async: %s", cmd)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.repo_root),
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout
            )
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=proc.returncode or 0,
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
            )
        except FileNotFoundError:
            raise RuntimeError("git is not installed or not in PATH.")
        except asyncio.TimeoutExpired:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"git command timed out: {cmd}")

    # --- Async Public API (non-blocking for event loop) ---

    async def check_working_tree_clean_async(self) -> bool:
        """Async version: Return True if the working tree has no uncommitted changes."""
        result = await self._run_async(["status", "--porcelain"])
        return not result.stdout.strip()

    async def auto_commit_uncommitted_async(self) -> tuple[str, str] | None:
        """Async version: Stage all uncommitted changes and create a tachyon-managed commit."""
        if await self.check_working_tree_clean_async():
            return None

        await self._run_async(["add", "-A"])
        msg = f"[tachyon] auto-commit uncommitted changes — {self._timestamp()}"
        result = await self._run_async(["commit", "--allow-empty", "-m", msg])
        if result.returncode != 0:
            _log.warning("Auto-commit failed: %s", result.stderr.strip())
            return None

        commit_result = await self._run_async(["rev-parse", "HEAD"])
        short_result = await self._run_async(["rev-parse", "--short", "HEAD"])
        commit_hash = commit_result.stdout.strip()
        short_hash = short_result.stdout.strip()
        _log.info("Auto-committed uncommitted changes as %s", short_hash)
        return commit_hash, short_hash

    async def get_current_branch_async(self) -> str:
        """Async version: Return the current branch name."""
        result = await self._run_async(["rev-parse", "--abbrev-ref", "HEAD"])
        return result.stdout.strip()

    async def get_current_hash_async(self) -> str:
        """Async version: Return the current HEAD commit hash."""
        result = await self._run_async(["rev-parse", "HEAD"])
        return result.stdout.strip()

    async def snapshot_async(self, message: str = "") -> tuple[str, str]:
        """Async version: Create a git commit snapshot of the current state."""
        await self._run_async(["add", "-A"])
        msg = message or f"[tachyon] evolve snapshot — {self._timestamp()}"
        result = await self._run_async(["commit", "--allow-empty", "-m", msg])
        if result.returncode != 0:
            raise RuntimeError(f"Git snapshot failed: {result.stderr.strip()}")

        commit_result = await self._run_async(["rev-parse", "HEAD"])
        short_result = await self._run_async(["rev-parse", "--short", "HEAD"])
        commit_hash = commit_result.stdout.strip()
        short_hash = short_result.stdout.strip()
        _log.info("Snapshot created: %s (%s)", short_hash, commit_hash)
        return commit_hash, short_hash

    async def rollback_async(self, commit_hash: str) -> bool:
        """Async version: Roll back the working tree to a previous commit."""
        _log.info("Rolling back to %s", commit_hash)
        result = await self._run_async(["reset", "--hard", commit_hash])
        success = result.returncode == 0
        if success:
            _log.info("Rollback to %s successful.", commit_hash)
        else:
            _log.error("Rollback failed: %s", result.stderr.strip())
        return success

    async def get_diff_async(
        self,
        commit_a: str,
        commit_b: str | None = None,
    ) -> str:
        """Async version: Get unified diff between two commits."""
        args = ["diff", commit_a]
        if commit_b is not None:
            args.append(commit_b)
        result = await self._run_async(args)
        return result.stdout

    async def reset_to_head_async(self) -> bool:
        """Async version: Reset working tree to HEAD."""
        result = await self._run_async(["reset", "--hard", "HEAD"])
        return result.returncode == 0

    async def save_working_state_async(self) -> str:
        """Async version: Save unstaged working tree changes as a patch."""
        result = await self._run_async(["diff"])
        state_dir = self.repo_root / ".tachyon" / "evolve" / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / "working.patch"
        path.write_text(result.stdout, encoding="utf-8")
        return str(path)

    async def restore_working_state_async(self, patch_path: str) -> bool:
        """Async version: Restore working tree from a previously saved patch."""
        patch_content = Path(patch_path).read_text(encoding="utf-8").strip()
        await self._run_async(["checkout", "."])
        await self._run_async(["clean", "-fd"])
        if not patch_content:
            return True
        result = await self._run_async(["apply", patch_path])
        if result.returncode != 0:
            _log.warning("Failed to restore working state: %s", result.stderr.strip())
            return False
        _log.info("Restored working state from %s", patch_path)
        return True

    async def stage_files_async(self, files: Sequence[str]) -> bool:
        """Async version: Stage specific files into the index."""
        if not files:
            return True
        result = await self._run_async(["add", "--"] + list(files))
        if result.returncode != 0:
            _log.warning("git add failed: %s", result.stderr.strip())
        return result.returncode == 0

    async def checkout_async(self, ref: str) -> bool:
        """Async version: Checkout a branch or commit."""
        result = await self._run_async(["checkout", ref])
        return result.returncode == 0

    async def create_branch_async(self, name: str) -> bool:
        """Async version: Create a new branch from HEAD (force-replace if exists)."""
        await self._run_async(["branch", "-f", name])
        return True

    async def checkout_files_async(self) -> bool:
        """Async version: Checkout all files to clean working tree."""
        result = await self._run_async(["checkout", "."])
        return result.returncode == 0

    async def clean_untracked_async(self) -> bool:
        """Async version: Remove untracked files and directories."""
        result = await self._run_async(["clean", "-fd"])
        return result.returncode == 0

    async def diff_cached_stat_async(self) -> str:
        """Async version: Get stat of staged changes."""
        result = await self._run_async(["diff", "--cached", "--stat"])
        return result.stdout

    async def checkout_from_branch_async(self, branch: str) -> bool:
        """Async version: Checkout all files from a branch to working tree."""
        result = await self._run_async(["checkout", branch, "--", "."])
        return result.returncode == 0

    async def reset_head_async(self) -> bool:
        """Async version: Reset index to HEAD (unstage all)."""
        result = await self._run_async(["reset", "HEAD"])
        return result.returncode == 0

    async def list_branches_async(self, pattern: str) -> list[str]:
        """Async version: List branches matching pattern."""
        result = await self._run_async(
            ["for-each-ref", "--format=%(refname:short)", f"refs/heads/{pattern}"]
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        return [b.strip() for b in result.stdout.strip().splitlines() if b.strip()]

    async def delete_branch_async(self, name: str) -> bool:
        """Async version: Delete a branch."""
        result = await self._run_async(["branch", "-D", name])
        return result.returncode == 0

    @staticmethod
    def _timestamp() -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
