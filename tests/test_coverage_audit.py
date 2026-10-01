"""Tests for tools/coverage_audit.py.

The tool exists because two earlier versions of this audit reported the wrong
thing, so the tests pin the specific ways it went wrong rather than testing the
happy path only:

  1. A line-coverage percentage cannot separate "nothing imports this" from
     "imported but never called". Both read as 0%.
  2. A runtime import tracer cannot see imports inside test function bodies,
     which is how two genuinely covered modules were reported as dead.

Test 2 is the one that matters most: it is a false negative about code that
works, and it would have argued for deleting working code.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from tools.coverage_audit import (
    audit,
    collect_import_sites,
    format_report,
    is_test_file,
    module_name_for,
)


def _write(root: Path, rel: str, body: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body), encoding="utf-8")


def _cov_entry(n_stmts: int, covered: int, *, branches: tuple[int, int] | None = None):
    """A coverage.py-shaped per-file record."""
    summary = {
        "num_statements": n_stmts, "covered_lines": covered,
        "percent_covered": (100.0 * covered / n_stmts) if n_stmts else 0.0,
    }
    if branches is not None:
        num_b, cov_b = branches
        summary.update(num_branches=num_b, covered_branches=cov_b,
                       percent_covered_display=str(covered))
    return {"summary": summary, "executed_lines": list(range(1, covered + 1)),
            "missing_lines": list(range(covered + 1, n_stmts + 1)),
            "missing_branches": []}


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A miniature repo with one module in each interesting state."""
    _write(tmp_path, "pkg/__init__.py", "")
    # (a) imported at module level, executed
    _write(tmp_path, "pkg/imported.py", "VALUE = 1\n")
    # (b) imported only INSIDE a function body -- the import-tracer blind spot
    _write(tmp_path, "pkg/lazy.py", "def load():\n    return 7\n")
    # (c) imported at module level, never executed
    _write(tmp_path, "pkg/never_run.py", "def go():\n    return 1\n")
    # (d) no importer anywhere
    _write(tmp_path, "pkg/orphan.py", "def orphan():\n    return 2\n")
    # The consumers
    _write(tmp_path, "pkg/user.py", """
        from pkg import imported
        from pkg import never_run

        def f():
            from pkg.lazy import load   # lazy: invisible to a collection-time tracer
            return imported.VALUE + load()
    """)
    _write(tmp_path, "tests/test_user.py", """
        from pkg.user import f

        def test_f():
            assert f() == 8
    """)
    return tmp_path


def test_module_name_for_handles_modules_and_packages() -> None:
    assert module_name_for("apedp/decision/policy.py") == "apedp.decision.policy"
    assert module_name_for("apedp/decision/__init__.py") == "apedp.decision"
    assert module_name_for("top.py") == "top"


def test_import_scan_finds_module_level_and_lazy_imports(tree: Path) -> None:
    """The regression that matters: lazy imports must count as reachable."""
    sites = collect_import_sites(tree)
    assert "pkg.imported" in sites, "module-level import missed"
    assert "pkg.lazy" in sites, (
        "lazy import inside a function body missed -- this is exactly the blind "
        "spot that made apedp.priors look untested when it is 56% covered")
    assert "pkg.never_run" in sites
    assert "pkg.orphan" not in sites, "orphan has no importer and must not gain one"


def test_relative_imports_resolve_against_the_importing_package(tree: Path) -> None:
    _write(tree, "pkg/sub/__init__.py", "")
    _write(tree, "pkg/sub/leaf.py", "X = 1\n")
    _write(tree, "pkg/sub/user2.py", "from .leaf import X\n")
    _write(tree, "pkg/sub/user3.py", "from .. import lazy\n")
    sites = collect_import_sites(tree)
    assert "pkg.sub.leaf" in sites, "relative .leaf import not resolved"
    assert "pkg.lazy" in sites, "parent-relative ..lazy import not resolved"


def test_nested_import_implies_parent_packages(tree: Path) -> None:
    _write(tree, "pkg/deep/__init__.py", "")
    _write(tree, "pkg/deep/thing.py", "Y = 1\n")
    _write(tree, "pkg/main2.py", "from pkg.deep.thing import Y\n")
    sites = collect_import_sites(tree)
    for name in ("pkg", "pkg.deep", "pkg.deep.thing"):
        assert name in sites, f"{name} not registered as reachable"


def test_skip_dirs_exclude_other_sessions_checkouts(tmp_path: Path) -> None:
    """A worktree copy must not make a module look reachable from this tree."""
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/real.py", "A = 1\n")
    _write(tmp_path, "pkg/elsewhere.py", "B = 2\n")
    _write(tmp_path, ".kilo/worktrees/session-b/pkg/other.py",
           "from pkg.elsewhere import B\n")
    sites = collect_import_sites(tmp_path)
    assert "pkg.elsewhere" not in sites, (
        "scanned another session's worktree; a module looked reachable only "
        "because a copy of the tree nobody is testing imports it")


def test_audit_separates_the_three_states(tmp_path: Path) -> None:
    """0% lines is ambiguous; the classification must not be."""
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/imported.py", "VALUE = 1\n")
    _write(tmp_path, "pkg/lazy.py", "def load():\n    return 7\n")
    _write(tmp_path, "pkg/never_run.py", "def go():\n    return 1\n")
    _write(tmp_path, "pkg/orphan.py", "def orphan():\n    return 2\n")
    _write(tmp_path, "pkg/user.py", """
        from pkg import imported, never_run
        def f():
            from pkg.lazy import load
            return imported.VALUE + load()
    """)

    cov = tmp_path / "cov.json"
    cov.write_text(json.dumps({"files": {
        "pkg/__init__.py": _cov_entry(0, 0),
        "pkg/imported.py": _cov_entry(4, 4, branches=(6, 6)),
        "pkg/lazy.py": _cov_entry(5, 3, branches=(2, 1)),   # ran, outcome untaken
        "pkg/never_run.py": _cov_entry(9, 0),
        "pkg/orphan.py": _cov_entry(4, 0),
    }}), encoding="utf-8")

    report = audit(cov, tmp_path, only_unexercised=False)
    by = {v.module: v for v in report["_verdicts"]}

    assert by["pkg.orphan"].status == "NEVER_IMPORTED"
    assert by["pkg.never_run"].status == "IMPORTED_NOT_EXECUTED"
    # Lazy-imported AND actually executed: EXERCISED, and demonstrably not
    # reported dead. The two states a line percentage would have merged.
    assert by["pkg.lazy"].status == "EXERCISED"
    assert by["pkg.lazy"].covered_lines == 3

    assert report["totals"]["never_imported"] == 1
    assert report["totals"]["imported_not_executed"] == 1
    assert report["branch_coverage_present"] is True
    # The default view keeps only the two actionable buckets.
    assert {v.module for v in audit(cov, tmp_path)["_verdicts"]} == {
        "pkg.orphan", "pkg.never_run"}


def test_exercised_module_with_untaken_branches_is_flagged(tmp_path: Path) -> None:
    """100% of statements executed is not 100% of the code understood."""
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/guarded.py", "def f():\n    return 1\n")
    _write(tmp_path, "pkg/user.py", "from pkg import guarded\n")
    cov = tmp_path / "cov.json"
    cov.write_text(json.dumps({"files": {
        "pkg/__init__.py": _cov_entry(0, 0),
        "pkg/guarded.py": _cov_entry(4, 4, branches=(10, 4)),
    }}), encoding="utf-8")

    report = audit(cov, tmp_path, only_unexercised=False)
    guarded = next(v for v in report["_verdicts"] if v.module == "pkg.guarded")
    assert guarded.status == "EXERCISED"
    assert guarded.has_unexercised_branches is True
    assert report["totals"]["exercised_with_unexercised_branches"] == 1


def test_missing_branch_data_is_declared_not_assumed_full(tmp_path: Path) -> None:
    """A line-only coverage file must not read as 'fully covered'."""
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/a.py", "A = 1\n")
    _write(tmp_path, "pkg/user.py", "from pkg import a\n")
    cov = tmp_path / "cov.json"
    cov.write_text(json.dumps({"files": {
        "pkg/__init__.py": _cov_entry(0, 0),
        "pkg/a.py": _cov_entry(3, 3),   # no branch keys at all
    }}), encoding="utf-8")

    report = audit(cov, tmp_path, only_unexercised=False)
    assert report["branch_coverage_present"] is False
    assert "NO BRANCH DATA" in format_report(report)


def test_package_init_is_reachable_from_its_own_submodules(tmp_path: Path) -> None:
    """An empty __init__ is not a coverage gap.

    This is the ``apedp.api.routes`` false alarm: a package __init__ with no
    statements reports "0 covered of 0", which read as 0% and got reported by
    hand as an untested module until the file was opened. The tool has to make
    that impossible to repeat.
    """
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/sub/__init__.py", "")
    _write(tmp_path, "pkg/sub/thing.py", "X = 1\n")
    _write(tmp_path, "pkg/user.py", "from pkg.sub.thing import X\n")
    cov = tmp_path / "cov.json"
    cov.write_text(json.dumps({"files": {
        "pkg/__init__.py": _cov_entry(0, 0),
        "pkg/sub/__init__.py": _cov_entry(0, 0),
        "pkg/sub/thing.py": _cov_entry(3, 3, branches=(1, 1)),
    }}), encoding="utf-8")

    report = audit(cov, tmp_path, only_unexercised=False)
    by = {v.module: v for v in report["_verdicts"]}
    assert by["pkg"].status == "EXERCISED", by["pkg"]
    assert by["pkg.sub"].status == "EXERCISED", by["pkg.sub"]
    # And an empty __init__ with NO importer is still not dead code.
    _write(tmp_path, "pkg/empty2/__init__.py", "")
    cov2 = tmp_path / "cov2.json"
    cov2.write_text(json.dumps({"files": {"pkg/empty2/__init__.py": _cov_entry(0, 0)}}),
                    encoding="utf-8")
    report2 = audit(cov2, tmp_path, only_unexercised=False)
    assert report2["_verdicts"][0].status == "EXERCISED", report2["_verdicts"][0]


def test_from_package_import_resolves_only_real_submodules(tmp_path: Path) -> None:
    """``from pkg import x`` is a submodule import only if pkg/x.py exists.

    Assuming submodule in both directions is wrong twice over: assuming it
    hides an orphan, assuming attribute invents an importer. So the name is
    resolved against the module files that actually exist.
    """
    _write(tmp_path, "pkg/__init__.py", "CONSTANT = 1\n")
    _write(tmp_path, "pkg/real.py", "R = 1\n")
    _write(tmp_path, "pkg/other.py", "from pkg import CONSTANT, real, ghost\n")
    sites = collect_import_sites(tmp_path)
    assert "pkg.real" in sites, "real submodule not resolved"
    assert "pkg.ghost" not in sites, (
        "invented an importer for pkg.ghost, which does not exist -- that is "
        "how a dead module gets argued into looking alive")
    assert "pkg" in sites, "the package itself must be recorded"


def test_in_package_test_files_are_not_reported_as_dead_source(tmp_path: Path) -> None:
    """--cov=apedp measures apedp/neural_discovery/tests/*.py as source.

    Nothing imports them because pytest runs them under a different module
    name, so they score 0% forever. Counting them would put nine permanent
    phantom findings in every audit of this repo, and a report with permanent
    findings is a report nobody reads.
    """
    assert is_test_file("apedp/neural_discovery/tests/test_training.py")
    assert is_test_file("apedp/neural_discovery/tests/conftest.py")
    assert is_test_file("tests/test_coverage_audit.py")
    assert not is_test_file("apedp/model_synthesis.py")
    assert not is_test_file("apedp/closed_loop.py")

    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/thing.py", "A = 1\n")
    _write(tmp_path, "pkg/tests/test_thing.py", "def test_a():\n    pass\n")
    _write(tmp_path, "pkg/user.py", "from pkg import thing\n")
    cov = tmp_path / "cov.json"
    cov.write_text(json.dumps({"files": {
        "pkg/thing.py": _cov_entry(2, 2, branches=(1, 1)),
        "pkg/tests/test_thing.py": _cov_entry(2, 0),
    }}), encoding="utf-8")

    report = audit(cov, tmp_path, only_unexercised=False)
    assert [v.module for v in report["_verdicts"]] == ["pkg.thing"], report["verdicts"]
    assert report["totals"]["n_test_files_skipped"] == 1, report["totals"]
    assert "1 test file(s) skipped" in format_report(report)


def test_report_text_names_the_remedy_for_each_bucket(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/orphan.py", "def o():\n    return 1\n")
    _write(tmp_path, "pkg/never_run.py", "def g():\n    return 1\n")
    _write(tmp_path, "pkg/user.py", "from pkg import never_run\n")
    cov = tmp_path / "cov.json"
    cov.write_text(json.dumps({"files": {
        "pkg/__init__.py": _cov_entry(0, 0),
        "pkg/orphan.py": _cov_entry(2, 0),
        "pkg/never_run.py": _cov_entry(2, 0),
    }}), encoding="utf-8")

    text = format_report(audit(cov, tmp_path))
    assert "NEVER IMPORTED" in text and "pkg.orphan" in text
    assert "IMPORTED BUT NEVER EXECUTED" in text and "pkg.never_run" in text
    # The importer must be shown, because "who should call this" is the fix.
    assert "pkg/user.py" in text


def test_audit_survives_unparseable_and_missing_files(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/broken.py", "def (:\n")   # syntax error
    _write(tmp_path, "pkg/fine.py", "A = 1\n")
    _write(tmp_path, "pkg/user.py", "from pkg import fine\n")
    cov = tmp_path / "cov.json"
    cov.write_text(json.dumps({"files": {
        "pkg/fine.py": _cov_entry(2, 2, branches=(1, 1)),
    }}), encoding="utf-8")

    report = audit(cov, tmp_path, only_unexercised=False)
    assert [v["module"] for v in report["verdicts"]] == ["pkg.fine"]
