"""Tests for the wholesale-revert guard.

The guard exists because of a real incident: a ``git checkout -- <file>`` on
this shared tree destroyed another session's uncommitted work. Git offers no
``pre-checkout`` hook, so the guard cannot prevent that. What it can do is
catch the *committable* form of the same mistake and document the recovery path
for the part that is recoverable.

These tests build real git repositories in a temp dir rather than mocking git.
The signal the guard relies on -- a blob that appears verbatim in history --
is a property of the object store, and a mock would not exercise it.
"""
from __future__ import annotations

import pathlib
import subprocess

import pytest

from tools.revert_guard import (
    DEFAULT_LOOKBACK,
    MIN_REVERTED_LINES,
    dangling_blobs,
    find_reverts,
    index_blob,
    main,
    render,
)

BASE_LINES = 60


def _sh(args: list[str], cwd: pathlib.Path) -> str:
    # Shelling out to git is the point: the fixture builds real repositories.
    # `tests/*` already exempts S603/S607, so no noqa is needed here.
    return subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, check=True,
    ).stdout


@pytest.fixture
def repo(tmp_path: pathlib.Path) -> pathlib.Path:
    """A repo with one long file and a single commit on it."""
    _sh(["git", "init", "-q", "."], tmp_path)
    _sh(["git", "config", "user.email", "a@b"], tmp_path)
    _sh(["git", "config", "user.name", "a"], tmp_path)
    f = tmp_path / "shared.py"
    f.write_text("\n".join(f"line_{i}" for i in range(BASE_LINES)) + "\n")
    _sh(["git", "add", "shared.py"], tmp_path)
    _sh(["git", "commit", "-qm", "base"], tmp_path)
    return tmp_path


def _content(tag: str, n: int = 40) -> str:
    body = [f"line_{i}" for i in range(n)] + [f"{tag}_{i}" for i in range(20)]
    return "\n".join(body) + "\n"


class TestOrdinaryCommitsAreSilent:
    def test_untouched_tree_finds_nothing(self, repo: pathlib.Path) -> None:
        assert find_reverts(repo=repo) == []

    def test_small_ordinary_edit_finds_nothing(self, repo: pathlib.Path) -> None:
        """A normal edit produces a blob that exists nowhere in history.

        That is the discriminator the whole guard rests on, so it is pinned
        first: a guard that fired here would be ignored within a day.
        """
        f = repo / "shared.py"
        f.write_text("CHANGED_TOP\n" + "\n".join(
            f"line_{i}" for i in range(1, BASE_LINES)) + "\n")
        _sh(["git", "add", "shared.py"], repo)
        assert find_reverts(repo=repo) == []

    def test_appending_new_content_finds_nothing(self, repo: pathlib.Path) -> None:
        f = repo / "shared.py"
        f.write_text(f.read_text() + "brand_new_line\n")
        _sh(["git", "add", "shared.py"], repo)
        assert find_reverts(repo=repo) == []

    def test_multi_line_addition_finds_nothing(self, repo: pathlib.Path) -> None:
        """Even a large addition is not a revert, however many lines it adds."""
        f = repo / "shared.py"
        f.write_text(f.read_text() + "".join(f"new_{i}\n" for i in range(50)))
        _sh(["git", "add", "shared.py"], repo)
        assert find_reverts(repo=repo) == []

    def test_small_wholesale_revert_below_threshold_is_quiet(
        self, repo: pathlib.Path,
    ) -> None:
        """A 1-line revert is normal churn, not the failure mode."""
        f = repo / "shared.py"
        f.write_text("\n".join(
            ["REWRITTEN_TOP"] + [f"line_{i}" for i in range(1, BASE_LINES)]) + "\n")
        _sh(["git", "add", "shared.py"], repo)
        _sh(["git", "commit", "-qm", "rewrite top"], repo)
        f.write_text("\n".join(f"line_{i}" for i in range(BASE_LINES)) + "\n")
        _sh(["git", "add", "shared.py"], repo)
        # Restores the base content, but only ~1 line differs.
        assert find_reverts(repo=repo, min_lines=MIN_REVERTED_LINES) == []


class TestWholesaleRevertIsCaught:
    def test_restoring_a_previous_revision_is_flagged(self, repo: pathlib.Path) -> None:
        """The detectable form of the incident.

        Session A commits a large change, then something restores the file to
        the earlier revision wholesale. The staged blob matches history
        verbatim, which is the signature.
        """
        f = repo / "shared.py"
        f.write_text(_content("THEIRS"))
        _sh(["git", "add", "shared.py"], repo)
        _sh(["git", "commit", "-qm", "session A work"], repo)

        base = _sh(["git", "rev-parse", "HEAD~1"], repo).strip()
        _sh(["git", "checkout", base, "--", "shared.py"], repo)

        findings = find_reverts(repo=repo)
        assert len(findings) == 1, f"expected one finding, got {findings}"
        fnd = findings[0]
        assert fnd.path == "shared.py"
        assert fnd.lines_discarded >= MIN_REVERTED_LINES
        assert base[:12] == fnd.reverted_to

    def test_finding_message_names_the_file_and_the_revision(
        self, repo: pathlib.Path,
    ) -> None:
        f = repo / "shared.py"
        f.write_text(_content("THEIRS"))
        _sh(["git", "add", "shared.py"], repo)
        _sh(["git", "commit", "-qm", "session A work"], repo)
        base = _sh(["git", "rev-parse", "HEAD~1"], repo).strip()
        _sh(["git", "checkout", base, "--", "shared.py"], repo)

        text = render(find_reverts(repo=repo))
        assert "shared.py" in text
        assert base[:12] in text
        assert "discarding" in text

    def test_staged_index_is_what_is_inspected_not_the_worktree(
        self, repo: pathlib.Path,
    ) -> None:
        """The guard reads the index, so a revert in the index is seen.

        This is the distinction that made an earlier version of the guard
        wrong: comparing the worktree to HEAD reports a file as reverted when
        the commit would not change it at all.
        """
        f = repo / "shared.py"
        f.write_text(_content("THEIRS"))
        _sh(["git", "add", "shared.py"], repo)
        _sh(["git", "commit", "-qm", "session A work"], repo)
        base = _sh(["git", "rev-parse", "HEAD~1"], repo).strip()
        _sh(["git", "checkout", base, "--", "shared.py"], repo)
        assert index_blob("shared.py", repo) == _rev(repo, f"{base}:shared.py")
        assert find_reverts(repo=repo)


def _rev(repo: pathlib.Path, spec: str) -> str:
    return _sh(["git", "rev-parse", spec], repo).strip()


class TestGuardDoesNotOverreach:
    def test_only_staged_files_are_considered(self, repo: pathlib.Path) -> None:
        """An unstaged wholesale revert is invisible, and the guard is silent.

        Documented as a limit rather than tested as a win: this is exactly the
        case the original incident took, and no commit-time tool can see it.
        """
        f = repo / "shared.py"
        f.write_text(_content("THEIRS"))
        _sh(["git", "add", "shared.py"], repo)
        _sh(["git", "commit", "-qm", "session A work"], repo)
        base = _sh(["git", "rev-parse", "HEAD~1"], repo).strip()
        f.write_text(_sh(["git", "show", f"{base}:shared.py"], repo))
        # Nothing staged, so there is no commit to inspect.
        assert find_reverts(repo=repo) == []

    def test_reverting_a_different_file_is_quiet(self, repo: pathlib.Path) -> None:
        """Only the file that is actually being restored gets flagged."""
        other = repo / "other.py"
        other.write_text("a = 1\nb = 2\nc = 3\n")
        _sh(["git", "add", "other.py"], repo)
        _sh(["git", "commit", "-qm", "other file"], repo)
        f = repo / "shared.py"
        f.write_text(_content("THEIRS"))
        _sh(["git", "add", "shared.py"], repo)
        _sh(["git", "commit", "-qm", "big change"], repo)
        base = _sh(["git", "rev-parse", "HEAD~1"], repo).strip()
        _sh(["git", "checkout", base, "--", "shared.py"], repo)
        findings = find_reverts(repo=repo)
        assert [f_.path for f_ in findings] == ["shared.py"]


class TestRecoveryPathIsReal:
    def test_staged_work_survives_a_checkout_as_a_dangling_blob(
        self, repo: pathlib.Path,
    ) -> None:
        """The promise the warning makes is actually kept.

        Staged content is written to the object store before the checkout, so
        it survives as a dangling blob. Unstaged content does not. Verifying
        the recoverable half is what makes the warning honest.
        """
        f = repo / "shared.py"
        f.write_text(_content("RESCUE_ME"))
        _sh(["git", "add", "shared.py"], repo)
        _sh(["git", "checkout", "HEAD", "--", "shared.py"], repo)
        assert "RESCUE_ME_0" not in f.read_text(), "checkout should have wiped it"

        blobs = dangling_blobs(repo)
        assert blobs, "staged-then-lost work must leave a dangling blob"
        found = False
        for b in blobs:
            content = _sh(["git", "cat-file", "-p", b], repo)
            if "RESCUE_ME_0" in content:
                found = True
                break
        assert found, "the blob exists but does not contain the lost content"

    def test_unstaged_work_is_not_recoverable(self, repo: pathlib.Path) -> None:
        """The other half, asserted so the limit is not just a claim in prose.

        This is the case that cost the most: nothing was written to the object
        store, so no dangling blob exists and the content is simply gone.
        """
        f = repo / "shared.py"
        f.write_text(_content("NEVER_STAGED"))
        _sh(["git", "checkout", "HEAD", "--", "shared.py"], repo)
        assert "NEVER_STAGED_0" not in f.read_text()
        for b in dangling_blobs(repo):
            assert "NEVER_STAGED_0" not in _sh(["git", "cat-file", "-p", b], repo)


class TestCliBehaviour:
    def test_exit_zero_when_nothing_wrong(self, repo: pathlib.Path) -> None:
        assert main(["--repo", str(repo)]) == 0

    def test_exit_zero_even_when_flagged(self, repo: pathlib.Path) -> None:
        """Warns without blocking, so a legitimate revert stays possible."""
        f = repo / "shared.py"
        f.write_text(_content("THEIRS"))
        _sh(["git", "add", "shared.py"], repo)
        _sh(["git", "commit", "-qm", "work"], repo)
        base = _sh(["git", "rev-parse", "HEAD~1"], repo).strip()
        _sh(["git", "checkout", base, "--", "shared.py"], repo)
        assert main(["--repo", str(repo)]) == 0

    def test_warns_on_stderr_with_the_recovery_hint(
        self, repo: pathlib.Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        f = repo / "shared.py"
        f.write_text(_content("THEIRS"))
        _sh(["git", "add", "shared.py"], repo)
        _sh(["git", "commit", "-qm", "work"], repo)
        base = _sh(["git", "rev-parse", "HEAD~1"], repo).strip()
        _sh(["git", "checkout", base, "--", "shared.py"], repo)
        main(["--repo", str(repo), "--check-blobs"])
        err = capsys.readouterr().err
        assert "wholesale-reverts" in err
        assert "git fsck" in err
        # The limit must be stated in the warning itself, not only in the
        # module docstring: the person reading it at commit time is the one
        # who needs to know that unstaged work is already gone.
        assert "Unstaged work does not" in err
        assert "no-verify" in err

    def test_lookback_is_bounded_by_default(self) -> None:
        """A guard that scans all history is a guard nobody waits for."""
        assert 0 < DEFAULT_LOOKBACK <= 100
