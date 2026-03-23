"""Unit tests for GitRollback safety layer."""
import os
import subprocess
from pathlib import Path

import pytest

from tachyon.evolve.git import GitRollback


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repo for testing."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    os.chdir(repo)

    subprocess.run(["git", "init"], capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        capture_output=True, check=True,
    )

    # Create initial commit
    (repo / "README.md").write_text("Initial\n")
    subprocess.run(["git", "add", "."], capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        capture_output=True, check=True,
    )

    yield repo


class TestGitRollback:

    def test_verify_git_repo(self, git_repo: Path) -> None:
        rb = GitRollback(git_repo)
        assert rb.verify_git_repo() is True

    def test_verify_not_git_repo(self, tmp_path: Path) -> None:
        rb = GitRollback(tmp_path)
        assert rb.verify_git_repo() is False

    def test_check_working_tree_clean(self, git_repo: Path) -> None:
        rb = GitRollback(git_repo)
        assert rb.check_working_tree_clean() is True

    def test_check_working_tree_dirty(self, git_repo: Path) -> None:
        rb = GitRollback(git_repo)
        (git_repo / "new_file.txt").write_text("dirty\n")
        assert rb.check_working_tree_clean() is False

    def test_snapshot(self, git_repo: Path) -> None:
        rb = GitRollback(git_repo)
        commit_hash, short_hash = rb.snapshot("test snapshot")
        assert len(commit_hash) == 40
        assert len(short_hash) == 7

    def test_rollback(self, git_repo: Path) -> None:
        rb = GitRollback(git_repo)

        # Save current state
        pre_hash = rb.get_current_hash()

        # Make a change
        (git_repo / "README.md").write_text("Modified\n")
        rb.snapshot("modified")

        # Rollback
        success = rb.rollback(pre_hash)
        assert success is True
        assert (git_repo / "README.md").read_text() == "Initial\n"

    def test_get_diff(self, git_repo: Path) -> None:
        rb = GitRollback(git_repo)
        pre_hash = rb.get_current_hash()

        (git_repo / "README.md").write_text("Modified\n")
        rb.snapshot("modified")

        diff = rb.get_diff(pre_hash)
        assert "-Initial" in diff
        assert "+Modified" in diff

    def test_tag_experiment(self, git_repo: Path) -> None:
        rb = GitRollback(git_repo)
        commit_hash = rb.get_current_hash()
        rb.tag_experiment(0, commit_hash)

        # Verify tag exists
        result = subprocess.run(
            ["git", "tag", "-l", "tachyon-evolve-0"],
            capture_output=True, text=True, cwd=str(git_repo),
        )
        assert "tachyon-evolve-0" in result.stdout

    def test_ensure_gitignore(self, git_repo: Path) -> None:
        rb = GitRollback(git_repo)
        rb.ensure_gitignore_entries()

        gitignore = git_repo / ".gitignore"
        assert gitignore.is_file()
        content = gitignore.read_text()
        assert ".tachyon/evolve/" in content
        assert "*.ncu-rep" in content

    def test_ensure_gitignore_idempotent(self, git_repo: Path) -> None:
        rb = GitRollback(git_repo)
        rb.ensure_gitignore_entries()
        rb.ensure_gitignore_entries()  # second call should not add duplicates

        content = (git_repo / ".gitignore").read_text()
        assert content.count(".tachyon/evolve/") == 1

    def test_stage_files(self, git_repo: Path) -> None:
        rb = GitRollback(git_repo)

        # Create and stage a file
        (git_repo / "kernel.cu").write_text("__global__ void f() {}\n")
        result = rb.stage_files([str(git_repo / "kernel.cu")])
        assert result is True

        # Verify staged via git diff --cached
        diff = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True, text=True, cwd=str(git_repo),
        )
        assert "kernel.cu" in diff.stdout

    def test_stage_files_empty_list(self, git_repo: Path) -> None:
        rb = GitRollback(git_repo)
        assert rb.stage_files([]) is True

    def test_save_working_state_excludes_staged(self, git_repo: Path) -> None:
        """save_working_state should capture only unstaged changes."""
        rb = GitRollback(git_repo)

        # Stage a file (simulating an accepted optimization)
        (git_repo / "accepted.cu").write_text("// accepted\n")
        subprocess.run(
            ["git", "add", "accepted.cu"],
            capture_output=True, cwd=str(git_repo),
        )

        # Make an unstaged change (simulating a new iteration edit)
        (git_repo / "README.md").write_text("Modified\n")

        # Save working state
        patch_path = rb.save_working_state()
        patch_content = Path(patch_path).read_text()

        # Patch should contain the unstaged change to README.md
        assert "Modified" in patch_content
        # Patch should NOT contain the staged accepted.cu
        assert "accepted.cu" not in patch_content

    def test_restore_preserves_staged(self, git_repo: Path) -> None:
        """restore_working_state should preserve the staging area."""
        rb = GitRollback(git_repo)

        # Stage a file (accepted)
        (git_repo / "accepted.cu").write_text("// accepted\n")
        subprocess.run(
            ["git", "add", "accepted.cu"],
            capture_output=True, cwd=str(git_repo),
        )

        # Save (empty patch since README is unchanged and accepted.cu is staged)
        patch_path = rb.save_working_state()

        # Make an unstaged change (failed iteration)
        (git_repo / "README.md").write_text("Failed edit\n")

        # Restore should discard unstaged but keep staged
        success = rb.restore_working_state(patch_path)
        assert success is True

        # accepted.cu should still be staged
        diff = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True, text=True, cwd=str(git_repo),
        )
        assert "accepted.cu" in diff.stdout

        # README.md should be restored to original
        assert (git_repo / "README.md").read_text() == "Initial\n"
