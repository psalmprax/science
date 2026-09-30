"""Tests for the run-provenance guard.

Why mutation testing matters here
---------------------------------
This tool exists to stop me citing a stale log as evidence. A guard that
silently under-reports is worse than no guard, because it is trusted precisely
when it is not looking hard enough. The first version of ``commits_since``
reported 330 changed files for a 5-file commit -- it parsed every commit's file
list into the first record and exited 0 while doing it. Review did not catch
that; running it against a real commit with a known file count did.

So the assertions below pin *counts and identities* against ground truth from
git, not just "it returned something truthy".
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import run_provenance as rp


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A throwaway git repo with a real history, so tests exercise real git."""
    r = tmp_path / "repo"
    (r / "apedp").mkdir(parents=True)
    (r / "tests").mkdir()
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
           "PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path)}

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=r, check=True, capture_output=True,
                       env=env)

    git("init", "-q")
    (r / "apedp" / "core.py").write_text("X = 1\n")
    (r / "tests" / "test_core.py").write_text("def test_x(): pass\n")
    git("add", "-A")
    git("commit", "-q", "-m", "first")
    return r


class TestTreeScanning:
    def test_it_hashes_every_python_file_under_the_scan_dirs(self, repo: Path):
        files = rp.scan_tree(repo, ("apedp", "tests"))
        assert set(files) == {"apedp/core.py", "tests/test_core.py"}

    def test_it_hashes_untracked_files_too(self, repo: Path):
        # git ls-files would miss this. On a shared tree the untracked modules
        # are the ones most likely to be edited mid-run, so missing them would
        # make the manifest confidently wrong.
        (repo / "apedp" / "brand_new.py").write_text("Y = 2\n")
        assert "apedp/brand_new.py" in rp.scan_tree(repo, ("apedp", "tests"))

    def test_it_ignores_files_outside_the_scan_dirs(self, repo: Path):
        (repo / "docs").mkdir()
        (repo / "docs" / "guide.py").write_text("# not code\n")
        assert "docs/guide.py" not in rp.scan_tree(repo, ("apedp", "tests"))

    def test_it_honours_a_different_set_of_scan_dirs(self, repo: Path):
        # Scanning a fixed list regardless of the argument would make the
        # parameter decorative, and a manifest that quietly covers the wrong
        # tree is worse than one that covers none.
        assert set(rp.scan_tree(repo, ("apedp",))) == {"apedp/core.py"}
        assert set(rp.scan_tree(repo, ("tests",))) == {"tests/test_core.py"}

    def test_it_ignores_non_python_files(self, repo: Path):
        (repo / "apedp" / "notes.txt").write_text("hi\n")
        assert "apedp/notes.txt" not in rp.scan_tree(repo, ("apedp", "tests"))

    def test_a_missing_scan_dir_is_not_an_error(self, repo: Path):
        # A repo without tests/ must still produce a manifest, not a crash.
        assert rp.scan_tree(repo, ("apedp", "does_not_exist"))


class TestSnapshot:
    def test_it_records_head_and_a_timestamp(self, repo: Path):
        m = rp.snapshot(repo, ("apedp", "tests"))
        assert m["head"] == rp.head_commit(repo)
        assert m["launched_utc"] > 0
        assert m["head_subject"] == "first"

    def test_the_round_trip_through_json_preserves_the_files(self, repo: Path,
                                                             tmp_path: Path):
        m = rp.snapshot(repo, ("apedp", "tests"))
        path = rp.write_manifest(m, tmp_path)
        assert rp.read_manifest(path)["files"] == m["files"]

    def test_it_writes_a_file_named_for_purpose(self, repo: Path,
                                                tmp_path: Path):
        path = rp.write_manifest(rp.snapshot(repo), tmp_path)
        assert path.name == "run_provenance.json"


class TestDriftDetection:
    def test_no_drift_when_nothing_changed(self, repo: Path):
        m = rp.snapshot(repo, ("apedp", "tests"))
        drift, _ = rp.verify(m, repo)
        assert not drift.any
        assert drift.edited == [] and drift.added == [] and drift.removed == []

    def test_it_detects_an_edited_file(self, repo: Path):
        m = rp.snapshot(repo, ("apedp", "tests"))
        (repo / "apedp" / "core.py").write_text("X = 2\n")
        drift, _ = rp.verify(m, repo)
        assert drift.edited == ["apedp/core.py"]

    def test_it_detects_an_added_file(self, repo: Path):
        m = rp.snapshot(repo, ("apedp", "tests"))
        (repo / "apedp" / "new.py").write_text("Z = 3\n")
        drift, _ = rp.verify(m, repo)
        assert drift.added == ["apedp/new.py"]

    def test_it_detects_a_removed_file(self, repo: Path):
        m = rp.snapshot(repo, ("apedp", "tests"))
        (repo / "apedp" / "core.py").unlink()
        drift, _ = rp.verify(m, repo)
        assert drift.removed == ["apedp/core.py"]

    def test_a_file_changed_back_is_not_reported(self, repo: Path):
        # Content-addressed, so a revert to the original content is not drift.
        m = rp.snapshot(repo, ("apedp", "tests"))
        (repo / "apedp" / "core.py").write_text("X = 99\n")
        (repo / "apedp" / "core.py").write_text("X = 1\n")
        drift, _ = rp.verify(m, repo)
        assert not drift.any

    def test_diff_trees_classifies_each_kind(self):
        d = rp.diff_trees(
            {"a.py": "1", "b.py": "2", "c.py": "3"},
            {"a.py": "9", "b.py": "2", "d.py": "4"},
        )
        assert d.edited == ["a.py"]
        assert d.added == ["d.py"]
        assert d.removed == ["c.py"]
        assert d.any


class TestStaleCommitDetection:
    """The tests that would have caught the 330-files-for-5-files bug."""

    def _commit(self, repo: Path, msg: str, files: dict[str, str]) -> None:
        env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
               "PATH": "/usr/bin:/bin:/usr/local/bin",
               "HOME": str(repo.parent)}
        for rel, text in files.items():
            p = repo / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True,
                       capture_output=True, env=env)
        subprocess.run(["git", "commit", "-q", "-m", msg], cwd=repo, check=True,
                       capture_output=True, env=env)

    def test_it_finds_a_commit_after_the_launch_time(self, repo: Path):
        import time
        past = time.time() - 3600
        self._commit(repo, "later change", {"apedp/core.py": "X = 5\n"})
        found = rp.commits_since(repo, past, ("apedp", "tests"))
        # Two commits are in range: the fixture's own "first" and the new one.
        # Assert on identity rather than count, so the test states what it means.
        assert [c.subject for c in found] == ["later change", "first"]
        assert found[0].paths == ["apedp/core.py"]

    def test_the_path_count_matches_the_real_commit(self, repo: Path):
        # The exact regression: a 3-file commit must report 3 paths, not every
        # path in every subsequent commit as well.
        import time
        past = time.time() - 3600
        self._commit(repo, "c1", {
            "apedp/a.py": "A\n", "apedp/b.py": "B\n", "apedp/c.py": "C\n",
        })
        self._commit(repo, "c2", {"apedp/d.py": "D\n", "apedp/e.py": "E\n"})
        self._commit(repo, "c3", {"apedp/f.py": "F\n"})
        found = rp.commits_since(repo, past, ("apedp", "tests"))
        by_rev = {c.subject: c for c in found}
        assert {"c1", "c2", "c3"} <= set(by_rev)
        assert len(by_rev["c1"].paths) == 3
        assert len(by_rev["c2"].paths) == 2
        assert len(by_rev["c3"].paths) == 1
        assert "apedp/d.py" not in by_rev["c1"].paths

    def test_it_ignores_commits_that_only_touch_unscanned_paths(self, repo: Path):
        import time
        past = time.time() - 3600
        (repo / "docs").mkdir()
        self._commit(repo, "docs only", {"docs/readme.md": "hi\n"})
        found = rp.commits_since(repo, past, ("apedp", "tests"))
        assert "docs only" not in {c.subject for c in found}

    def test_it_ignores_commits_before_the_launch_time(self, repo: Path):
        import time
        self._commit(repo, "already done", {"apedp/core.py": "X = 7\n"})
        future = time.time() + 3600
        assert rp.commits_since(repo, future, ("apedp", "tests")) == []

    def test_it_returns_nothing_for_a_future_launch(self, repo: Path):
        import time
        assert rp.commits_since(repo, time.time() + 60, ("apedp",)) == []

    def test_each_commit_carries_its_own_subject_and_timestamp(self, repo: Path):
        import time
        past = time.time() - 3600
        self._commit(repo, "first change", {"apedp/x.py": "X\n"})
        self._commit(repo, "second change", {"apedp/y.py": "Y\n"})
        found = rp.commits_since(repo, past, ("apedp", "tests"))
        assert {"first change", "second change"} <= {c.subject for c in found}
        assert all(c.committed for c in found)

    def test_it_does_not_mix_one_commits_files_into_another(self, repo: Path):
        # The record-separator regression, stated directly: three commits with
        # disjoint, differently-sized file sets must each report their own.
        import time
        past = time.time() - 3600
        self._commit(repo, "one", {"apedp/only1.py": "1\n"})
        self._commit(repo, "two", {
            "apedp/only2.py": "2\n", "apedp/only3.py": "3\n",
            "apedp/only4.py": "4\n",
        })
        found = {c.subject: c.paths for c in rp.commits_since(
            repo, past, ("apedp", "tests"))}
        assert found["one"] == ["apedp/only1.py"]
        assert found["two"] == ["apedp/only2.py", "apedp/only3.py",
                                "apedp/only4.py"]

    def test_a_commit_at_the_launch_second_is_not_reported_as_after_it(self,
                                                                      repo: Path):
        # Boundary: a commit made in the same whole second as the snapshot is
        # not "after" the snapshot. Truncating the float timestamp toward zero
        # made every run launched and verified inside one second look stale,
        # which would train people to ignore the warning.
        #
        # The launch instant is read back out of git, not from time.time(),
        # because time.time() called after the commit is always strictly
        # greater and so cannot distinguish `<=` from `<` at the boundary.
        self._commit(repo, "just now", {"apedp/core.py": "X = 8\n"})
        ctime = float(rp._git(
            ["log", "-1", "--format=%ct"], repo).strip())
        assert rp.commits_since(repo, ctime, ("apedp", "tests")) == []
        # Half a second before that same commit and it is correctly in range.
        assert "just now" in {c.subject for c in rp.commits_since(
            repo, ctime - 0.5, ("apedp", "tests"))}

    def test_the_epoch_is_a_usable_lower_bound(self, repo: Path):
        # `--since=@0` reads to a human as "since the epoch" but git treats a
        # zero timestamp as no bound and returns nothing at all, so scanning
        # from 0 silently reported an empty history.
        found = rp.commits_since(repo, 0, ("apedp", "tests"))
        assert "first" in {c.subject for c in found}

    def test_a_far_future_launch_returns_nothing(self, repo: Path):
        import time
        assert rp.commits_since(repo, time.time() + 10_000, ("apedp",)) == []


class TestReportRendering:
    def _manifest(self, repo: Path, **over) -> dict:
        m = rp.snapshot(repo, ("apedp", "tests"))
        m.update(over)
        return m

    def test_a_clean_run_says_clean(self, repo: Path):
        m = self._manifest(repo)
        drift, stale = rp.verify(m, repo)
        assert "CLEAN" in rp.render_report(m, drift, stale)

    def test_a_moved_head_is_reported_as_stale(self, repo: Path):
        m = self._manifest(repo, head="0" * 40)
        drift, stale = rp.verify(m, repo)
        report = rp.render_report(m, drift, stale)
        assert "STALE" in report
        assert "HEAD is now" in report

    def test_drift_is_reported_with_the_file_named(self, repo: Path):
        m = self._manifest(repo)
        (repo / "apedp" / "core.py").write_text("X = 42\n")
        drift, stale = rp.verify(m, repo)
        report = rp.render_report(m, drift, stale)
        assert "DRIFT" in report
        assert "apedp/core.py" in report

    def test_it_states_the_caveat_in_words_not_just_a_flag(self, repo: Path):
        # A future reader must not have to know what "STALE" means.
        m = self._manifest(repo, head="0" * 40)
        drift, stale = rp.verify(m, repo)
        assert "not as it is now" in rp.render_report(m, drift, stale)

    def test_it_never_says_stale_and_clean_at_the_same_time(self, repo: Path):
        # The contradiction bug: a moved HEAD with no scanned-file commit
        # printed both STALE and CLEAN. A report that disagrees with itself
        # cannot be acted on, so pin the invariant rather than the wording.
        m = self._manifest(repo, head="0" * 40)
        drift, stale = rp.verify(m, repo)
        report = rp.render_report(m, drift, stale)
        assert not ("STALE" in report and "CLEAN" in report), report

    def test_a_moved_head_alone_counts_as_stale(self, repo: Path):
        # HEAD moving is itself a reason not to trust the result, even if no
        # commit touched the scanned directories.
        m = self._manifest(repo, head="0" * 40)
        drift, stale = rp.verify(m, repo)
        assert stale == [] and not drift.any
        assert rp.is_stale(m, repo)
        assert "STALE" in rp.render_report(m, drift, stale)

    def test_an_untouched_manifest_is_not_stale(self, repo: Path):
        m = rp.snapshot(repo, ("apedp", "tests"))
        assert not rp.is_stale(m, repo)

    def test_a_long_file_list_is_summarised_not_dumped(self, repo: Path):
        m = self._manifest(repo)
        for i in range(30):
            (repo / "apedp" / f"m{i}.py").write_text(f"X = {i}\n")
        drift, stale = rp.verify(m, repo)
        report = rp.render_report(m, drift, stale)
        assert "and 10 more" in report


class TestResolveRepo:
    def test_it_finds_the_repo_from_a_subdirectory(self, repo: Path,
                                                   monkeypatch):
        monkeypatch.chdir(repo / "apedp")
        # __file__ lives in the real project's tools/, so scope the search by
        # passing the tmp repo explicitly for determinism.
        assert rp.resolve_repo(repo) == repo.resolve()

    def test_it_refuses_a_directory_that_is_not_a_repo(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            rp.resolve_repo(tmp_path)

    def test_it_walks_up_from_the_current_directory(self, repo: Path,
                                                    monkeypatch):
        monkeypatch.chdir(repo / "apedp")
        assert (rp.resolve_repo() / ".git").exists()


class TestCli:
    def test_launch_then_verify_reports_clean(self, repo: Path, tmp_path: Path,
                                             capsys):
        out = tmp_path / "out"
        assert rp.main(["launch", "--repo", str(repo), "--out-dir", str(out),
                        "--scan-dirs", "apedp", "tests"]) == 0
        assert rp.main(["verify", "--repo", str(repo), "--manifest",
                        str(out / "run_provenance.json")]) == 0
        assert "CLEAN" in capsys.readouterr().out

    def test_verify_without_a_manifest_is_an_error(self, repo: Path,
                                                   tmp_path: Path):
        rc = rp.main(["verify", "--repo", str(repo), "--manifest",
                      str(tmp_path / "nope.json")])
        assert rc == 2

    def test_verify_on_a_bad_repo_exits_two(self, tmp_path: Path, capsys):
        assert rp.main(["verify", "--repo", str(tmp_path)]) == 2
        assert "ERROR" in capsys.readouterr().err

    def test_launch_prints_the_head_it_recorded(self, repo: Path,
                                                tmp_path: Path, capsys):
        rp.main(["launch", "--repo", str(repo), "--out-dir", str(tmp_path)])
        out = capsys.readouterr().out
        assert "head:" in out
        assert "files:" in out

    def test_staleness_warns_on_stderr_and_still_exits_zero(self, repo: Path,
                                                            tmp_path: Path,
                                                            capsys):
        # Warn, never block: a stale log is usually still worth reading.
        out = tmp_path / "out"
        rp.main(["launch", "--repo", str(repo), "--out-dir", str(out),
                 "--scan-dirs", "apedp", "tests"])
        (repo / "apedp" / "core.py").write_text("X = 1234\n")
        rc = rp.main(["verify", "--repo", str(repo), "--manifest",
                      str(out / "run_provenance.json")])
        captured = capsys.readouterr()
        assert rc == 0
        assert "DRIFT" in captured.err


class TestAgainstTheRealRepository:
    """Ground truth from the actual repo, not a synthetic fixture.

    This is the assertion that would have caught the record-separator bug: the
    real fc1d68d commit is known to touch exactly 5 files.
    """

    def test_the_real_last_commit_reports_its_true_file_count(self):
        repo = rp.resolve_repo(Path(__file__).resolve().parents[1])
        head = rp.head_commit(repo)
        found = rp.commits_since(repo, 0, ("apedp", "tests"))
        match = [c for c in found if c.rev == head]
        assert match, f"HEAD {head[:12]} not found in history scan"
        truth = subprocess.run(
            ["git", "show", "--name-only", "--format=", head],
            cwd=repo, capture_output=True, text=True, check=True,
        ).stdout.split()
        assert sorted(match[0].paths) == sorted(
            p for p in truth if p.startswith(("apedp/", "tests/"))
        )

    def test_the_real_repo_scans_to_a_plausible_file_count(self):
        repo = rp.resolve_repo(Path(__file__).resolve().parents[1])
        files = rp.scan_tree(repo)
        assert len(files) > 100, "scan found suspiciously few files"
