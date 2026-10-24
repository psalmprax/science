"""Tests for tools/affected_tests.py -- the tool that replaces a hand-listed run.

The tool exists because a verification run is a hand-written pytest command, and
a hand-written command once omitted the changed test file and let a
permanently-red test through two green verification runs before the full suite
caught it an hour later. If this tool is wrong in the same direction, it
reproduces the failure it was built to prevent, and it does so *quietly* --
which is the specific hazard. So the tests below are mostly negative.

Three properties are pinned:

1. A changed test file is always in its own run set.
2. Dependents are found by walking imports *backwards*, transitively.
3. An empty run set alongside changed Python is an error, not a clean answer.

Property 2 has a concrete regression behind it. The first implementation walked
forwards from the changed file -- its own dependencies rather than its
dependents -- and reported "no tests needed" for ``apedp/orchestration.py``,
which has two test files. Only property 3 caught it, and only because property 3
had already been built. The leaf-module case is pinned directly below so the
algorithm cannot regress while property 3 is switched off.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tools.affected_tests import (
    affected_tests,
    build_graph,
    dependents_of,
    main,
    staged_paths,
)

REPO = Path(__file__).resolve().parent.parent


def _write(root: Path, rel: str, body: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body), encoding="utf-8")


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A miniature import graph with the shapes that break reachability."""
    _write(tmp_path, "pkg/__init__.py", "")
    # A leaf: nothing in pkg imports it, only tests do.
    _write(tmp_path, "pkg/leaf.py", "VALUE = 1\n")
    # A middle layer: leaf-ish module imported by a hub.
    _write(tmp_path, "pkg/hub.py", "from pkg.leaf import VALUE\n")
    _write(tmp_path, "pkg/top.py", "from pkg.hub import VALUE\n")

    _write(tmp_path, "tests/test_leaf.py", "from pkg.leaf import VALUE\n")
    _write(tmp_path, "tests/test_hub.py", "from pkg.hub import VALUE\n")
    _write(tmp_path, "tests/test_top.py", "from pkg.top import VALUE\n")
    _write(tmp_path, "tests/test_unrelated.py", "CONST = 1\n")
    return tmp_path


class TestProperty1ChangedTestsAlwaysRun:
    def test_a_changed_test_file_is_in_its_own_run_set(self, tree: Path) -> None:
        report = affected_tests(["tests/test_leaf.py"], tree)
        assert report["run_set"] == ["tests/test_leaf.py"]
        assert report["direct_tests"] == ["tests/test_leaf.py"]

    def test_it_runs_even_when_nothing_imports_it(self, tree: Path) -> None:
        """A test file importing nothing is still a test file that was edited.

        Deriving from the import graph alone would silently skip exactly the
        file most likely to be wrong, because a broken test often fails to
        import anything.
        """
        report = affected_tests(["tests/test_unrelated.py"], tree)
        assert report["run_set"] == ["tests/test_unrelated.py"]

    def test_mixed_change_runs_both_kinds(self, tree: Path) -> None:
        report = affected_tests(["pkg/leaf.py", "tests/test_hub.py"], tree)
        assert "tests/test_leaf.py" in report["run_set"]
        assert "tests/test_hub.py" in report["run_set"]


class TestProperty2BackwardsTransitiveReach:
    def test_a_leaf_module_finds_its_direct_importer(self, tree: Path) -> None:
        """The regression. The forward walk returns nothing at all here.

        Asserted as membership rather than equality on purpose: the correct
        answer for this fixture is three tests, because ``pkg.leaf`` also
        reaches two more through the chain. Pinning equality to the single
        direct importer would have encoded the truncated one-ring walk as the
        expected answer.
        """
        report = affected_tests(["pkg/leaf.py"], tree)
        assert "tests/test_leaf.py" in report["run_set"], (
            "a leaf module with direct importers reported no affected tests -- "
            "that is the forward/backward walk bug, and it is silent")

    def test_dependents_are_found_transitively(self, tree: Path) -> None:
        """``pkg/leaf`` -> ``pkg.hub`` -> ``pkg.top`` -> a test that imports only top.

        The test never mentions leaf, and still must run: the change can reach
        it through two hops.
        """
        report = affected_tests(["pkg/leaf.py"], tree)
        assert "tests/test_top.py" in report["run_set"], (
            "transitive dependents are not reached; a test importing only the "
            "top of a chain can still be broken by a change at the bottom")
        assert "tests/test_hub.py" in report["run_set"]

    def test_an_unrelated_test_is_not_pulled_in(self, tree: Path) -> None:
        report = affected_tests(["pkg/leaf.py"], tree)
        assert "tests/test_unrelated.py" not in report["run_set"]

    def test_dependents_of_excludes_the_file_itself(self, tree: Path) -> None:
        _module_to_file, file_to_module, importers_of = build_graph(tree)
        found = dependents_of("pkg.leaf", file_to_module, importers_of)
        assert "pkg/leaf.py" not in found


class TestProperty3LoudOnAnEmptyAnswer:
    def test_python_changed_with_no_tests_is_a_failure(self, tree: Path,
                                                        capsys) -> None:
        _write(tree, "pkg/orphan.py", "X = 1\n")   # nothing imports it
        assert main(["pkg/orphan.py", "--root", str(tree)]) == 1
        err = capsys.readouterr().err
        assert "NO TESTS DERIVED" in err
        assert "not that the change is untested" in err

    def test_a_docs_only_change_is_a_clean_empty(self, tree: Path, capsys) -> None:
        """Legitimately nothing to run, and it must say so rather than error.

        The distinction from the case above is the whole point of property 3: an
        empty run set is only alarming when something test-shaped changed.
        """
        _write(tree, "docs/NOTES.md", "# notes\n")
        assert main(["docs/NOTES.md", "--root", str(tree)]) == 0

    def test_no_arguments_is_an_error_not_an_empty_success(self, capsys) -> None:
        with pytest.raises(SystemExit):
            main([])

    def test_staged_and_explicit_paths_are_mutually_exclusive(self) -> None:
        with pytest.raises(SystemExit):
            main(["--staged", "some/path.py"])

    def test_a_deleted_file_is_reported_unmapped(self, tree: Path) -> None:
        report = affected_tests(["pkg/gone.py"], tree)
        assert report["unmapped"] == ["pkg/gone.py"]


class TestStagedPathsAreReadFromTheIndex:
    def test_it_reads_git_not_the_working_tree(self) -> None:
        """The working tree on this repo carries other sessions' in-flight work.

        Asking git for unstaged changes would attribute another session's files
        to this commit and run their tests -- slow, noisy, and not this
        commit's responsibility.
        """
        out = staged_paths(REPO)
        assert isinstance(out, list)
        assert all(not p.startswith("/") for p in out), (
            "paths must be repo-relative for the run set to be usable")


class TestAgainstTheRealRepository:
    """The fixtures prove the algorithm; these prove it works on this tree."""

    def test_the_real_orchestration_module_finds_its_tests(self) -> None:
        """The exact case that motivated this tool, on the real tree.

        Forward-walk reported zero for ``apedp/orchestration.py``, which has
        three test files including one added specifically for its guards.
        """
        report = affected_tests(["apedp/orchestration.py"], REPO)
        assert "tests/test_orchestration_guards.py" in report["run_set"]

    def test_the_real_model_synthesis_change_finds_its_own_test_file(self) -> None:
        """A change to model_synthesis must run test_model_synthesis.py.

        The omission that motivated this tool was exactly this pairing. Pinning
        it on the real tree means the property survives a refactor of the graph
        code, and it is cheap: no synthesis runs.
        """
        report = affected_tests(["apedp/model_synthesis.py"], REPO)
        assert "tests/test_model_synthesis.py" in report["run_set"]

    def test_the_audit_tool_finds_the_tests_that_pin_it(self) -> None:
        report = affected_tests(["tools/coverage_audit.py"], REPO)
        assert "tests/test_coverage_audit.py" in report["run_set"]
        # Transitive: the facade contract test imports the audit tool but not
        # for the function under change.
        assert "tests/test_facade_contracts.py" in report["run_set"]

    def test_the_hook_is_installable_and_executable(self) -> None:
        hook = REPO / "tools" / "hooks" / "pre-commit"
        assert hook.is_file(), (
            "the hook is not in the repo, so the gate cannot be installed by a "
            "clone and every session reinvents the hand-listed run")
        assert hook.stat().st_mode & 0o111, (
            "tools/hooks/pre-commit is not executable; git will skip it silently")
        text = hook.read_text(encoding="utf-8")
        assert "affected_tests.py" in text, (
            "the hook must derive the run set rather than name one")