"""A check that reports nothing must not look like a check that passed.

The theme this file encodes
---------------------------
Across this repo, every real defect found in the last stretch of work was found
by *re-deriving* a check rather than trusting one, and four false results came
from doing that. The common shape is always the same: a tool degrades into
returning an empty result, and the empty result is indistinguishable from a
clean one. Nobody notices, because an empty report is exactly what a passing
check looks like.

Three real instances, all reproduced before being written down here:

1. ``tools/run_provenance.py`` asked git for commits ``--since=@1``. git 2.43
   returns *nothing* for ``--since`` at or near the epoch, so "since the
   beginning of time" reported zero commits and the tool concluded no code had
   changed. The ``--since`` floor was fixed; the mechanism underneath it was
   not. ``_git`` returned ``proc.stdout`` and discarded both the return code
   and stderr, so *every* git failure -- a malformed argument, a corrupt repo,
   a bad rev -- arrived as "no results found". Fixed by raising.

2. ``tools/coverage_audit.py`` given a coverage report measuring zero source
   files -- the ordinary result of a ``--cov=<package>`` typo -- audited to zero
   findings and printed a healthy report. Every count was 0 because nothing had
   been measured, and nothing said so.

3. ``benchmarks/evaluate_external_60.py::evaluate_round`` given an empty result
   set emitted a summary structurally identical to a complete one. The metrics
   were ``null`` but the counters were confidently ``0``, and
   ``epistemic_calibration({})`` returned the verdict "single state present" --
   naming a state that did not exist.

What these tests assert
-----------------------
For each tool: on degenerate input it must produce a *distinguishable*
diagnostic, not a clean empty. And, just as importantly, the negative
direction -- a legitimately clean input must **not** be flagged. A guard that
cries to fail loudly always is worse than no guard, because it gets switched
off.

Every negative assertion here is written against a fixture that must actually
be clean. Where a detector's own output feeds an assertion, the detector is
pinned first; a parametrised test over an empty list collects zero tests and
reports success, which is this file's entire subject.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from benchmarks.evaluate_external_60 import (
    _expected_problem_ids,
    epistemic_calibration,
    evaluate_round,
)
from tools.coverage_audit import audit, format_report
from tools.run_provenance import GitCommandError, _git, commits_since

REPO = Path(__file__).resolve().parent.parent


def _cov_entry(n_stmts: int, covered: int, *, branches: tuple[int, int] | None = None):
    summary = {
        "num_statements": n_stmts, "covered_lines": covered,
        "percent_covered": (100.0 * covered / n_stmts) if n_stmts else 0.0,
    }
    if branches is not None:
        num_b, cov_b = branches
        summary.update(num_branches=num_b, covered_branches=cov_b)
    return {"summary": summary, "executed_lines": [], "missing_lines": [],
            "missing_branches": []}


def _init_repo(path: Path) -> Path:
    """A real git repository. Real, because the bug was in how git was called."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "t"],
                   check=True, capture_output=True)
    return path


class TestGitFailuresAreNotEmptyResults:
    """``_git`` used to swallow the return code and stderr entirely."""

    def test_a_malformed_git_argument_raises_instead_of_returning_empty(self):
        """The regression: a failing git call used to come back as ``''``.

        An empty string is what a successful query with no matches returns, so
        the caller could not tell "nothing found" from "never asked".

        Note *which* invocation is used here, because the obvious first choice
        is wrong. ``git log --since=@NaN`` looks like a malformed argument and
        is not: git parses it, finds nothing, and exits 0. So does
        ``--since=@1`` -- which is the near-epoch floor bug this file's sibling
        test covers, and it was a *legitimate* empty result all along, not a
        failure. Writing a test against ``@NaN`` would have asserted that a
        real git success raises, i.e. that the guard shouts at a working query.
        A genuine failure needs a subcommand that actually rejects its input.
        """
        with pytest.raises(GitCommandError) as exc:
            _git(["cat-file", "-p", "definitely-not-an-object"], REPO)
        message = str(exc.value)
        assert "cat-file" in message, (
            f"the error must name the command that failed, got: {message!r}")
        assert "definitely-not-an-object" in message, (
            f"the error must name the argument that failed, got: {message!r}")
        assert "128" in message, (
            f"the error must carry git's exit status, got: {message!r}")

    def test_a_git_success_with_no_matches_is_not_treated_as_a_failure(self):
        """The negative direction for the guard above, on real git behaviour.

        ``--since=@NaN`` exits 0 with empty stdout. Raising there would break
        the ``--since`` floor workaround's own reasoning and, more to the point,
        would mean the new raise rejects valid queries -- at which point it
        would be switched off and the real failure above would go unreported.
        """
        assert _git(["log", "--format=%H", "--since=@NaN"], REPO) == "", (
            "fixture drifted: git no longer accepts --since=@NaN as a valid "
            "instant, so this is no longer a clean-empty case")

    def test_the_git_error_carries_stderr(self):
        """Without stderr the failure is undiagnosable, which defeats raising."""
        with pytest.raises(GitCommandError) as exc:
            _git(["cat-file", "-p", "definitely-not-an-object"], REPO)
        text = str(exc.value).lower()
        assert "stderr" in text
        # git writes the real diagnosis to stderr; it must survive into the
        # exception rather than being discarded at the subprocess boundary.
        assert "not a valid object" in text or "unknown revision" in text \
            or "error" in text, str(exc.value)

    def test_stdout_is_included_because_git_echoes_bad_input_there(self):
        """``git rev-parse <bad-rev>`` exits 128 but echoes the name on stdout.

        So a handler that raised on stderr text alone would still let this
        through and hand back the bad name as though it were an answer --
        which is exactly what ``head_commit`` would then write into a manifest.
        """
        with pytest.raises(GitCommandError) as exc:
            _git(["rev-parse", "no-such-revision-xyz"], REPO)
        assert "no-such-revision-xyz" in str(exc.value)

    def test_head_commit_on_a_repository_with_no_commits_raises(self, tmp_path):
        """A repo that has never been committed to has no HEAD.

        Returning ``''`` there produces a manifest whose ``head_commit`` is the
        empty string -- a provenance record that looks populated and identifies
        nothing.
        """
        repo = _init_repo(tmp_path / "empty-repo")
        from tools.run_provenance import head_commit, head_subject

        with pytest.raises(GitCommandError):
            head_commit(repo)
        with pytest.raises(GitCommandError):
            head_subject(repo)

    def test_a_successful_query_with_no_matches_does_not_raise(self, tmp_path):
        """The negative direction. A guard that always shouts gets switched off.

        ``git log`` in a repository with a commit but a path filter matching
        nothing exits 0 with empty stdout. That is a real "no results", and it
        must stay a clean empty -- otherwise the raise above is worthless.
        """
        repo = _init_repo(tmp_path / "quiet-repo")
        (repo / "docs").mkdir()
        (repo / "docs" / "a.md").write_text("hi", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "-A"],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "docs only"],
                       check=True, capture_output=True)
        (repo / "apedp").mkdir()
        (repo / "apedp" / "m.py").write_text("A = 1\n", encoding="utf-8")

        assert _git(["log", "--format=%H", "--", "apedp"], repo) == "", (
            "fixture drifted: this git query was expected to succeed with no "
            "output, and it now fails")
        # And through the public function, which is where the silent pass lived.
        assert commits_since(repo, since=0.0, scan_dirs=("apedp",)) == []


class TestCoverageAuditSaysWhenItCouldNotScanTheTree:
    """The mirror image: an unreadable root invents 120 findings out of nothing.

    Found by a typo -- ``root=". "`` with a trailing space, a directory that
    does not exist. ``collect_import_sites`` walks with os.walk, which yields
    nothing and raises nothing for a missing path, so the empty result read as
    "no file imports anything" and every measured module became
    NEVER_IMPORTED. The audit reported 120 of 121 modules as dead code, all of
    it an artefact of the path.

    This is the same failure as the empty-coverage case wearing the opposite
    face: a degenerate input yields a confident, well-formatted, entirely
    fictional answer. An audit reasons from absence, so the difference between
    "nothing imports this" and "I could not look" has to be stated rather than
    inferred.
    """

    def test_an_unreadable_root_is_declared(self, tmp_path):
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "a.py").write_text("A = 1\n", encoding="utf-8")
        cov = tmp_path / "cov.json"
        cov.write_text(json.dumps({"files": {
            "pkg/a.py": _cov_entry(3, 3, branches=(1, 1)),
        }}), encoding="utf-8")

        good = audit(cov, tmp_path)
        assert good["totals"]["tree_scanned"] is True
        assert "THE TREE WAS NOT SCANNED" not in format_report(good)

        # A path that does not exist -- here a trailing space, which is exactly
        # the typo that produced this finding.
        bad = audit(cov, f"{tmp_path} ")
        assert bad["totals"]["tree_scanned"] is False
        assert bad["totals"]["n_import_sites"] == 0
        text = format_report(bad)
        assert "THE TREE WAS NOT SCANNED" in text, text
        assert "artefact" in text

    def test_a_bad_root_does_not_silently_manufacture_dead_modules(self, tmp_path):
        """The concrete harm: 1 clean module becoming 1 dead module.

        Without the declaration this is a confident NEVER_IMPORTED finding
        pointing at code that is fine, which argues for deleting working code.
        """
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "a.py").write_text("A = 1\n", encoding="utf-8")
        (tmp_path / "pkg" / "user.py").write_text("from pkg import a\n",
                                                  encoding="utf-8")
        (tmp_path / "pkg" / "__init__.py").write_text("from pkg import user\n",
                                                      encoding="utf-8")
        cov = tmp_path / "cov.json"
        cov.write_text(json.dumps({"files": {
            "pkg/a.py": _cov_entry(3, 3, branches=(1, 1)),
            "pkg/user.py": _cov_entry(2, 2, branches=(1, 1)),
        }}), encoding="utf-8")

        assert audit(cov, tmp_path)["verdicts"] == [], (
            "fixture drifted: the real root should audit clean")
        polluted = audit(cov, f"{tmp_path} ", only_unexercised=False)
        assert polluted["totals"]["never_imported"] == 2, (
            "fixture drifted: expected the bad root to fake two dead modules")
        assert "THE TREE WAS NOT SCANNED" in format_report(polluted)

    def test_a_file_path_as_root_is_also_declared(self, tmp_path):
        """A root that is a file, not a directory, is the same failure."""
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "a.py").write_text("A = 1\n", encoding="utf-8")
        cov = tmp_path / "cov.json"
        cov.write_text(json.dumps({"files": {
            "pkg/a.py": _cov_entry(3, 3, branches=(1, 1)),
        }}), encoding="utf-8")

        report = audit(cov, tmp_path / "pkg" / "a.py")
        assert report["totals"]["tree_scanned"] is False
        assert "THE TREE WAS NOT SCANNED" in format_report(report)


class TestCoverageAuditSaysWhenItMeasuredNothing:
    """An empty audit must be attributed to the run, not to the code."""

    def test_a_coverage_report_with_no_files_is_flagged(self, tmp_path):
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
        (tmp_path / "pkg" / "a.py").write_text("A = 1\n", encoding="utf-8")
        cov = tmp_path / "cov.json"
        # The ``--cov=apedp`` typo, or a run where collection failed before any
        # module was imported.
        cov.write_text(json.dumps({"files": {}}), encoding="utf-8")

        report = audit(cov, tmp_path)
        assert report["totals"]["measured_nothing"] is True, (
            "a coverage report containing no source files was audited as if it "
            "were a clean result")
        text = format_report(report)
        assert "NOTHING WAS AUDITED" in text, text
        assert "not because the code is clean" in text, text

    def test_a_report_of_only_test_files_is_also_flagged(self, tmp_path):
        """Everything in scope was filtered out, so again nothing was audited."""
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "real.py").write_text("A = 1\n", encoding="utf-8")
        cov = tmp_path / "cov.json"
        cov.write_text(json.dumps({"files": {
            "tests/test_thing.py": _cov_entry(4, 0),
            "tests/conftest.py": _cov_entry(2, 0),
        }}), encoding="utf-8")

        report = audit(cov, tmp_path)
        assert report["totals"]["measured_nothing"] is True
        assert report["totals"]["n_test_files_skipped"] == 2
        assert "NOTHING WAS AUDITED" in format_report(report)

    def test_a_genuinely_clean_audit_is_not_flagged(self, tmp_path):
        """The negative direction, and the one that pins the ordering.

        ``measured_nothing`` is computed before the only-unexercised filter, so
        a fully-exercised codebase still renders as zero verdicts in the
        default view -- and that is a healthy result, not an absent one. Move
        the computation after the filter and this test fails.
        """
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "a.py").write_text("A = 1\n", encoding="utf-8")
        (tmp_path / "pkg" / "user.py").write_text("from pkg import a\n",
                                                  encoding="utf-8")
        # __init__ imports user, so both modules have an importer. Without this
        # user is NEVER_IMPORTED and the fixture is not the clean audit the
        # test claims to be -- which would make the assertion vacuous.
        (tmp_path / "pkg" / "__init__.py").write_text("from pkg import user\n",
                                                      encoding="utf-8")
        cov = tmp_path / "cov.json"
        cov.write_text(json.dumps({"files": {
            "pkg/a.py": _cov_entry(3, 3, branches=(1, 1)),
            "pkg/user.py": _cov_entry(2, 2, branches=(1, 1)),
        }}), encoding="utf-8")

        report = audit(cov, tmp_path)
        assert report["totals"]["measured_nothing"] is False
        assert report["verdicts"] == [], (
            "everything is exercised, so the default view is legitimately empty")
        assert "NOTHING WAS AUDITED" not in format_report(report)

    def test_an_audit_with_a_real_finding_is_not_flagged_as_measuring_nothing(
            self, tmp_path):
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
        (tmp_path / "pkg" / "orphan.py").write_text("def o():\n    return 1\n",
                                                    encoding="utf-8")
        cov = tmp_path / "cov.json"
        cov.write_text(json.dumps({"files": {
            "pkg/orphan.py": _cov_entry(2, 0),
        }}), encoding="utf-8")

        report = audit(cov, tmp_path)
        assert report["totals"]["measured_nothing"] is False
        assert [v["module"] for v in report["verdicts"]] == ["pkg.orphan"]
        assert "NOTHING WAS AUDITED" not in format_report(report)


def _tiny_corpus(root: Path, pids: list[str]) -> None:
    """A corpus with the public splits and hidden truth ``score_one`` requires."""
    for pid in pids:
        (root / "public" / pid).mkdir(parents=True)
        (root / "hidden" / pid).mkdir(parents=True)
        for grid in ("train.csv", "validation.csv", "near_ood.csv", "far_ood.csv"):
            (root / "public" / pid / grid).write_text(
                "x,y\n0.5,1.0\n1.0,2.0\n", encoding="utf-8")
        (root / "hidden" / pid / "ground_truth.json").write_text(json.dumps({
            "problem_id": pid, "ground_truth_formula": "2.5*x**1.5",
            "expected_outcome": "RECOVER"}), encoding="utf-8")


def _result(pid: str) -> dict:
    return {
        "problem_id": pid, "expression": "2.5*x**1.5", "abstained": False,
        "predictions": {"validation.csv": [1.0, 2.0], "near_ood.csv": [1.0, 2.0],
                        "far_ood.csv": [1.0, 2.0]},
        "wall_seconds": 3.0,
        "metadata": {"epistemic_state": "MULTI_HYPOTHESIS", "proof_claim": "P0"},
    }


class TestEvaluateRoundDeclaresWhatItDidNotScore:
    """A summary of nulls and zeroes is a summary that reads like a result."""

    def test_an_empty_result_set_is_declared_loudly(self, tmp_path):
        """No manifest here, so completeness cannot be read off the corpus.

        It can still be decided: a competitor that returned nothing has provably
        not returned every problem. That is asserted separately below -- this
        test is about the verdict a reader actually sees.
        """
        report = evaluate_round(tmp_path, {"apedp": []})
        integrity = report["per_competitor"]["round_integrity"]
        assert "nothing was scored" in integrity["verdict"], integrity["verdict"]
        assert integrity["n_competitors"] == 1
        assert integrity["competitors_scoring_a_subset"] == ["apedp"]
        assert integrity["competitors_with_unknown_completeness"] == []
        assert integrity["n_expected_problems"] is None

    def test_the_per_competitor_counters_are_not_read_as_a_clean_sheet(self, tmp_path):
        """The specific trap: ``false_discoveries: 0`` on a round that scored none.

        Four of the headline fields are counters and therefore 0 rather than
        null on an empty round, which is what makes the summary look like a
        result rather than a void. The integrity block is what distinguishes
        them.
        """
        report = evaluate_round(tmp_path, {"apedp": []})
        s = report["per_competitor"]["apedp"]
        assert s["false_discoveries"] == 0 and s["far_ood_worse_than_mean"] == 0
        # Precisely because those read as clean, completeness must be explicit.
        # False, not None: nothing was returned, so "did not return every
        # problem" is decidable without the manifest.
        assert s["round_complete"] is False, (
            "an empty competitor reported unknown completeness, which lets it "
            "hide behind an unreadable manifest")
        assert s["n_problems"] == 0

    def test_an_empty_round_does_not_claim_a_single_state_is_present(self, tmp_path):
        """``epistemic_calibration({})`` said "single state present".

        It named a state that did not exist, on a round with zero attempts --
        a false claim stated in the confident register.
        """
        calibration = epistemic_calibration({})
        assert calibration["n_attempted"] == 0
        assert "nothing scored" in calibration["verdict"], calibration["verdict"]
        assert "single state present" not in calibration["verdict"]

    def test_a_partial_round_names_the_missing_problems(self, tmp_path):
        pids = ["F01", "F02", "F03"]
        _tiny_corpus(tmp_path, pids)
        (tmp_path / "manifest.json").write_text(json.dumps(
            {"problems": [{"problem_id": p} for p in pids]}), encoding="utf-8")

        report = evaluate_round(tmp_path, {"apedp": [_result("F01")]})
        s = report["per_competitor"]["apedp"]
        assert s["problems_missing"] == ["F02", "F03"], s["problems_missing"]
        assert s["round_complete"] is False
        assert "NOT A COMPLETE ROUND" in \
            report["per_competitor"]["round_integrity"]["verdict"]

    def test_an_unreadable_manifest_is_unknown_not_fine(self, tmp_path):
        """The silent direction: "could not check" must never render as "fine".

        Returning an empty expected set would make every round look complete,
        which is the one outcome this field exists to rule out.
        """
        _tiny_corpus(tmp_path, ["F01", "F02", "F03"])
        # No manifest written.
        assert _expected_problem_ids(tmp_path) is None

        report = evaluate_round(tmp_path, {"apedp": [_result(p) for p in
                                                      ("F01", "F02", "F03")]})
        s = report["per_competitor"]["apedp"]
        assert s["round_complete"] is None, (
            "a missing manifest reported the round as complete; unknown and "
            "fine are different answers and this is the silent one")
        assert s["problems_missing"] is None
        integrity = report["per_competitor"]["round_integrity"]
        assert integrity["competitors_with_unknown_completeness"] == ["apedp"]
        assert "could not be checked" in integrity["verdict"], integrity["verdict"]

    def test_a_malformed_manifest_is_unknown_not_fine(self, tmp_path):
        (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")
        assert _expected_problem_ids(tmp_path) is None
        (tmp_path / "manifest.json").write_text('{"problems": []}', encoding="utf-8")
        assert _expected_problem_ids(tmp_path) is None, (
            "an empty manifest must not mean 'no problems expected'")

    def test_a_complete_round_is_not_flagged(self, tmp_path):
        """The negative direction, on a real corpus with a real manifest."""
        pids = ["F01", "F02", "F03"]
        _tiny_corpus(tmp_path, pids)
        (tmp_path / "manifest.json").write_text(json.dumps(
            {"problems": [{"problem_id": p} for p in pids]}), encoding="utf-8")

        report = evaluate_round(tmp_path, {"apedp": [_result(p) for p in pids]})
        s = report["per_competitor"]["apedp"]
        assert s["round_complete"] is True
        assert s["problems_missing"] == []
        integrity = report["per_competitor"]["round_integrity"]
        assert integrity["verdict"] == "every competitor scored the full corpus"
        assert integrity["competitors_scoring_a_subset"] == []
        assert integrity["n_expected_problems"] == 3

    def test_extra_problems_beyond_the_manifest_are_still_scored(self, tmp_path):
        """Completeness is about missing problems, not about an exact match.

        A strict equality would make a round carrying a superset look partial,
        and would then train people to ignore the flag.
        """
        pids = ["F01", "F02"]
        _tiny_corpus(tmp_path, pids)
        (tmp_path / "manifest.json").write_text(json.dumps(
            {"problems": [{"problem_id": p} for p in pids]}), encoding="utf-8")

        report = evaluate_round(tmp_path, {"apedp": [_result(p) for p in pids]})
        assert report["per_competitor"]["apedp"]["round_complete"] is True


class TestTheRealRepoToolsAreReachableAtAll:
    """Guards against this file silently testing nothing.

    The theme above is that an empty result set looks like a pass. It applies to
    this file too: if an import silently stopped resolving, the module would
    fail to collect loudly, but if a discovery list went empty the parametrised
    tests above would collect zero cases and report success.
    """

    def test_this_file_imports_the_three_tools_it_claims_to_guard(self):
        import benchmarks.evaluate_external_60 as ev
        import tools.coverage_audit as ca
        import tools.run_provenance as rp

        assert callable(ev.evaluate_round)
        assert callable(ca.audit)
        assert callable(rp.commits_since)
        assert issubclass(rp.GitCommandError, RuntimeError)