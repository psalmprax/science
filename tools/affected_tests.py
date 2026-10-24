"""Which tests does this change actually require? Derive it; do not list it.

The failure this prevents
-------------------------
A verification run is a hand-written pytest command: ``pytest tests/test_a.py
tests/test_b.py ...``. Writing that list is a manual step, and a manual step is
one that can be done wrong. It was. A test committed permanently red for a day
because the two hand-listed runs that day happened to name every changed file
*except* the one carrying the failing assertion -- both runs were green, the
tree was not, and only the full suite an hour later found it.

So the set is computed here, from the staged diff and a static import graph.
The premise is that the set of tests a change can affect is a fact about the
code, not a judgement call about which files feel relevant.

What it does
------------
``--staged`` reads ``git diff --cached --name-only``. Each changed file is
mapped to its module; the import graph is walked in reverse to find every file
that can reach it, transitively, because a test that imports ``apedp.a`` is
affected by a change to ``apedp.b`` when ``apedp.a`` imports ``apedp.b``. The
intersection with test files is the run set.

Three properties this is built to have:

1. **A changed test file is always in its own run set.** Non-negotiable and
   unconditional -- that is the whole point. A test that was modified must be
   run, whatever the graph says about it.
2. **It reuses the repo's existing import scanner** (``tools.coverage_audit``)
   rather than growing a second one. This repo has already been damaged by two
   divergent import analyses that each reported confidently and wrongly; the
   lesson was one scanner, interrogated carefully.
3. **It fails loudly when it cannot answer.** If a changed Python file cannot be
   mapped, or the scan finds no edges at all, the tool exits non-zero rather
   than reporting "no tests needed" -- which would be indistinguishable from a
   clean answer, the exact defect class this whole effort is about.

Usage::

    python3 tools/affected_tests.py --staged            # print the run set
    python3 tools/affected_tests.py --staged --run      # print and execute it
    python3 tools/affected_tests.py a.py tests/b.py     # explicit paths

Exit codes: 0 = a run set was produced (possibly empty, and it says so why);
1 = the set could not be determined, or ``--run`` was given and pytest failed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.coverage_audit import (
    _iter_python_files,
    collect_import_sites,
    is_test_file,
    module_name_for,
)

REPO = Path(__file__).resolve().parent.parent

#: Above this many test files the derived set stops being a shortcut and starts
#: being the suite. The tool says so rather than quietly running 90 files,
#: because a "targeted" run that is secretly the whole suite teaches people to
#: skip it.
FULL_SUITE_HINT_THRESHOLD = 40


def staged_paths(repo: Path = REPO) -> list[str]:
    """Files staged for commit.

    Read from the index, not the working tree: the question is what is about to
    be committed, and ``git status`` conflates that with unrelated work in
    progress from concurrent sessions on this shared tree.
    """
    # S607: asking git about the index is this tool's entire purpose, so
    # resolving `git` from PATH is the design. Arguments are fixed flags, no
    # user input reaches this call, and no shell is involved.
    # S603: same -- fixed argv, no shell.
    proc = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],  # noqa: S607
        cwd=repo, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"git diff --cached failed ({proc.returncode}): "
            f"{proc.stderr.strip() or '(no stderr)'}")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def build_graph(repo: Path = REPO) -> tuple[dict[str, str], dict[str, str],
                                           dict[str, set[str]]]:
    """``(module -> file, file -> module, module -> files importing it)``.

    The third map is exactly what ``collect_import_sites`` already returns, used
    in its natural orientation: a module name to the files that name it in an
    import statement. The second is its inverse and is not optional -- the walk
    in ``dependents_of`` hops file -> module -> importers at every step, and
    looking a file path up in the module-keyed map returns None and silently
    ends the traversal after one ring. That bug produced a two-file result for a
    three-hop chain and looked like a correct answer. Both maps come from the
    same file walk, so they cannot describe different trees.
    """
    module_to_file: dict[str, str] = {}
    file_to_module: dict[str, str] = {}
    for path in _iter_python_files(repo):
        rel = str(path.relative_to(repo))
        module = module_name_for(rel)
        module_to_file.setdefault(module, rel)
        file_to_module.setdefault(rel, module)
    return module_to_file, file_to_module, collect_import_sites(repo)


def dependents_of(module: str, file_to_module: dict[str, str],
                  importers_of: dict[str, set[str]]) -> set[str]:
    """Every file that can transitively import ``module``.

    Walks *backwards* along import edges, because the question is who depends on
    the changed file, not what it depends on. Walking forwards is the obvious
    mistake and produces an empty set for leaf modules: it reports the changed
    file's own dependencies rather than its dependents, which for
    ``apedp/orchestration.py`` -- imported by tests and by nothing it imports
    back -- is nothing at all. That version reported "no tests needed" for a
    module with three test files.

    Seeded with the direct importers of ``module``; from each of those, the next
    ring is whoever imports *that* file's module.
    """
    seen_files: set[str] = set()
    frontier: set[str] = set(importers_of.get(module, ()))
    while frontier:
        rel = frontier.pop()
        if rel in seen_files:
            continue
        seen_files.add(rel)
        rel_module = file_to_module.get(rel)
        if rel_module is None:
            continue
        for importer in importers_of.get(rel_module, ()):
            if importer not in seen_files:
                frontier.add(importer)
    return seen_files


def affected_tests(changed: Iterable[str], repo: Path = REPO) -> dict:
    changed = list(changed)
    _module_to_file, file_to_module, importers_of = build_graph(repo)

    run_set: set[str] = set()
    direct_tests: list[str] = []
    unmapped: list[str] = []
    non_python: list[str] = []
    all_test_files = {
        str(p.relative_to(repo)) for p in _iter_python_files(repo)
        if is_test_file(str(p.relative_to(repo)))
    }

    for rel in changed:
        if not rel.endswith(".py"):
            non_python.append(rel)
            continue
        if not (repo / rel).is_file():
            unmapped.append(rel)   # deleted or renamed away
            continue
        if is_test_file(rel):
            # Property 1. Unconditional.
            direct_tests.append(rel)
            run_set.add(rel)
            continue
        dependents = dependents_of(module_name_for(rel), file_to_module, importers_of)
        run_set |= (dependents & all_test_files)

    return {
        "run_set": sorted(run_set),
        "direct_tests": sorted(direct_tests),
        "unmapped": sorted(unmapped),
        "non_python": sorted(non_python),
        "n_test_files_total": len(all_test_files),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Not a mutually-exclusive group: argparse forbids a positional with
    # nargs="*" inside one, and the exclusion is checked by hand below so the
    # error message can say which of the two was given.
    parser.add_argument("--staged", action="store_true",
                        help="use files staged for commit")
    parser.add_argument("paths", nargs="*", help="explicit changed paths")
    parser.add_argument("--run", action="store_true", help="execute pytest on the set")
    parser.add_argument("--root", default=str(REPO))
    args = parser.parse_args(argv)

    if args.staged and args.paths:
        parser.error("give either --staged or explicit paths, not both")
    if not args.staged and not args.paths:
        parser.error("give --staged or one or more explicit paths")

    repo = Path(args.root).resolve()
    try:
        changed = staged_paths(repo) if args.staged else list(args.paths)
    except RuntimeError as exc:
        print(f"!! {exc}", file=sys.stderr)
        return 1

    if not changed:
        print("nothing staged; nothing to verify")
        return 0

    report = affected_tests(changed, repo)
    run_set = report["run_set"]

    print(f"changed: {len(changed)} file(s)")
    print(f"  {sum(1 for c in changed if c.endswith('.py'))} python, "
          f"{len(report['non_python'])} other")
    if report["direct_tests"]:
        print(f"  test files changed (always run): {', '.join(report['direct_tests'])}")
    print(f"tests to run: {len(run_set)} of {report['n_test_files_total']}")

    # Property 3. An empty run set is legitimate -- a docs-only change needs no
    # tests -- but only when there was genuinely nothing test-shaped to check.
    # If Python changed and the graph produced nothing, that is a broken scan,
    # not a clean answer.
    python_changed = [c for c in changed if c.endswith(".py")]
    if python_changed and not run_set:
        print("\n!! NO TESTS DERIVED, but python files changed.", file=sys.stderr)
        print("   The import graph produced no dependents. That means the scan "
              "failed, not that the change is untested -- refusing to report a "
              "clean result from an empty one.", file=sys.stderr)
        if report["unmapped"]:
            print(f"   unmappable: {', '.join(report['unmapped'])}", file=sys.stderr)
        return 1

    if len(run_set) >= FULL_SUITE_HINT_THRESHOLD:
        print(f"\nNOTE: {len(run_set)} test files is most of the suite. "
              "Consider `make test` instead of this shortcut.")

    for rel in run_set:
        print(f"  {rel}")

    if not args.run:
        return 0
    if not run_set:
        print("\nno tests to run.")
        return 0

    cmd = [sys.executable, "-m", "pytest", *run_set, "-p", "no:randomly", "-q"]
    print(f"\n$ {' '.join(cmd)}")
    # S603: argv is sys.executable plus a derived list of repo-relative paths,
    # never user input, and no shell is involved.
    return subprocess.run(cmd, cwd=repo, check=False).returncode  # noqa: S603


if __name__ == "__main__":
    sys.exit(main())