"""Classify apedp modules by *why* they are unexercised, not just that they are.

Motivation, because the first version of this analysis was wrong twice.

The naive approach is to run the suite under coverage and read the percentage.
That cannot distinguish two completely different failures:

  * a module nothing imports, which is dead by construction and cannot fail a
    test even in principle; and
  * a module that is imported on every run but whose body never executes --
    e.g. a lazy import inside a function nobody calls.

Reporting a single "0% covered" number conflates them, and the fix differs: the
first needs an importer or deletion, the second needs a caller.

The second wrong version was an import tracer. It hooked module-level imports
at pytest collection time and concluded that ``apedp.priors`` and
``apedp.jax_pde_solver`` were never loaded. Both are actually imported -- inside
test *function bodies*, after collection has finished -- and coverage shows them
at 56% and 43%. So this tool never traces imports at runtime at all. It parses
the source with ``ast`` and asks a static question: does any file in the tree
name this module in an import statement? That question has no collection-order
blind spot, because it does not depend on when anything ran.

Branch coverage is reported alongside line coverage for the same reason one
level down. A module at 100% lines with 40% branches has had every statement
executed and most *outcomes* never taken -- the untested ``except`` clause, the
never-true guard. Those are where silent failures live, and a line-only audit
reports them as fully covered.

Usage::

    python3 -m pytest --cov=apedp --cov-branch --cov-report=json:/tmp/cov.json
    python3 tools/coverage_audit.py /tmp/cov.json --root .

Exit status is 0 always: this reports, it does not gate. A repo-health tool
that turns CI red for a pre-existing condition trains people to ignore it.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

#: Directories never scanned for import sites. ``.kilo/worktrees`` holds other
#: sessions' checkouts: counting their imports would make a module look
#: reachable from a copy of the tree that is not the one being tested.
SKIP_DIRS = frozenset({
    ".git", "__pycache__", ".venv", "node_modules", "target", "outputs",
    ".refact", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".kilo", "worktrees", "htmlcov",
})

#: A module with fewer executed lines than this is reported but not called
#: "exercised"; the threshold is deliberately low, because the interesting
#: signal is a hard zero, not a partial number.
EXERCISED_FLOOR = 1


@dataclass(frozen=True)
class ModuleVerdict:
    """One module, and the reason it is or is not exercised."""

    module: str
    path: str
    statements: int
    covered_lines: int
    line_pct: float
    num_branches: int = 0
    covered_branches: int = 0
    branch_pct: float | None = None
    importers: tuple[str, ...] = ()
    missing_branch_lines: tuple[int, ...] = field(default=())

    @property
    def status(self) -> str:
        """One of NEVER_IMPORTED / IMPORTED_NOT_EXECUTED / EXERCISED.

        The three are deliberately distinct rather than collapsed into a
        percentage, because each names a different owner of the fix.

        A module with no statements is EXERCISED, not a gap: there is nothing
        to execute, so "0 of 0 covered" is vacuously complete. Several package
        __init__ files are empty, and counting them as untested would put a
        permanent phantom finding in every audit -- which is how a report ends
        up with findings nobody reads.
        """
        if self.statements == 0:
            return "EXERCISED"
        if not self.importers:
            return "NEVER_IMPORTED"
        if self.covered_lines < EXERCISED_FLOOR:
            return "IMPORTED_NOT_EXECUTED"
        return "EXERCISED"

    @property
    def has_unexercised_branches(self) -> bool:
        """Statements ran, but outcomes did not. Silent-failure territory."""
        return self.num_branches > self.covered_branches

    def to_dict(self) -> dict:    return {
            "module": self.module, "path": self.path, "status": self.status,
            "statements": self.statements, "covered_lines": self.covered_lines,
            "line_pct": round(self.line_pct, 1),
            "num_branches": self.num_branches,
            "covered_branches": self.covered_branches,
            "branch_pct": None if self.branch_pct is None else round(self.branch_pct, 1),
            "n_importers": len(self.importers),
            "importers": list(self.importers[:5]),
        }


def module_name_for(path: str) -> str:
    """``apedp/decision/policy.py`` -> ``apedp.decision.policy``."""
    p = Path(path)
    parts = list(p.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _iter_python_files(root: Path) -> Iterator[Path]:
    # os.walk yields str dirpaths, so root is joined back on rather than
    # dividing the yielded dirpath by a name.
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in sorted(filenames):
            if name.endswith(".py"):
                yield Path(dirpath) / name


def collect_import_sites(root: Path) -> dict[str, set[str]]:
    """Map ``module name -> set of files that import it``.

    Static, by AST. Deliberately not a runtime tracer: a tracer only sees
    imports that happen, and imports inside function bodies happen after
    collection is over, which is exactly how two genuinely-covered modules were
    once reported as dead.

    Handles ``import a.b``, ``from a.b import c`` and relative ``from .c
    import d`` (resolved against the importing file's package).

    ``from pkg import name`` is the awkward case: the AST records only the
    package, and ``name`` may be a submodule or merely an attribute of
    ``pkg/__init__.py``. Guessing either way is wrong -- assuming submodule
    hides a genuinely orphaned module, assuming attribute invents an importer
    for one. So the names of all real modules in the tree are collected first
    and the import is resolved against them.
    """
    files = list(_iter_python_files(root))
    known = {module_name_for(str(f.relative_to(root))) for f in files}

    sites: dict[str, set[str]] = {}
    for file in files:
        try:
            tree = ast.parse(file.read_text(encoding="utf-8", errors="ignore"))
        except (SyntaxError, ValueError):
            continue
        rel = str(file.relative_to(root))
        package = rel.rsplit("/", 1)[0].replace("/", ".") if "/" in rel else ""
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    # Relative import: anchor to the importing file's package.
                    base = package
                    for _ in range(node.level - 1):
                        base = base.rsplit(".", 1)[0] if "." in base else ""
                    target = f"{base}.{node.module}" if node.module else base
                else:
                    target = node.module or ""
                if not target:
                    continue
                targets.append(target)
                # Resolve ``from pkg import x`` against real module names only.
                for alias in node.names:
                    if alias.name != "*":
                        candidate = f"{target}.{alias.name}"
                        if candidate in known:
                            targets.append(candidate)
            for name in targets:
                if not name:
                    continue
                sites.setdefault(name, set()).add(rel)
                # An ``import apedp.x.y`` also makes ``apedp.x`` reachable.
                parts = name.split(".")
                for i in range(1, len(parts)):
                    sites.setdefault(".".join(parts[:i]), set()).add(rel)
    return sites


def _importers_for(module: str, sites: dict[str, set[str]]) -> tuple[str, ...]:
    return tuple(sorted(sites.get(module, ())))


def audit(coverage_json: str | Path, root: str | Path = ".",
          only_unexercised: bool = True) -> dict:
    """Classify every measured file. Returns a report dict."""
    root = Path(root).resolve()
    data = json.loads(Path(coverage_json).read_text(encoding="utf-8"))
    files = data.get("files", data)
    sites = collect_import_sites(root)

    verdicts: list[ModuleVerdict] = []
    for rel, record in files.items():
        summary = record.get("summary", {})
        module = module_name_for(rel)
        # A package __init__ is imported by its own submodules; attribute the
        # union of its parts' importers so "no importer" stays meaningful.
        importers = set(_importers_for(module, sites))
        if Path(rel).name == "__init__.py":
            prefix = module + "."
            for name, where in sites.items():
                if name.startswith(prefix):
                    importers |= where
        missing = tuple(record.get("missing_branches", ()) or ())
        num_b = summary.get("num_branches", 0) or 0
        cov_b = summary.get("covered_branches", 0) or 0
        verdicts.append(ModuleVerdict(
            module=module, path=rel,
            statements=summary.get("num_statements", 0),
            covered_lines=summary.get("covered_lines", 0),
            line_pct=summary.get("percent_covered", 0.0),
            num_branches=num_b, covered_branches=cov_b,
            branch_pct=(100.0 * cov_b / num_b) if num_b else None,
            importers=tuple(sorted(importers)),
            missing_branch_lines=missing,
        ))

    order = {"NEVER_IMPORTED": 0, "IMPORTED_NOT_EXECUTED": 1, "EXERCISED": 2}
    if only_unexercised:
        verdicts = [v for v in verdicts if v.status != "EXERCISED"]
    verdicts.sort(key=lambda v: (order[v.status], v.covered_lines, -v.statements))

    buckets: dict[str, list[ModuleVerdict]] = {
        "NEVER_IMPORTED": [], "IMPORTED_NOT_EXECUTED": [], "EXERCISED": [],
    }
    for v in verdicts:
        buckets[v.status].append(v)
    return {
        "totals": {
            "n_measured": len(verdicts),
            "never_imported": len(buckets["NEVER_IMPORTED"]),
            "imported_not_executed": len(buckets["IMPORTED_NOT_EXECUTED"]),
            "exercised": len(buckets["EXERCISED"]),
            "exercised_with_unexercised_branches": sum(
                1 for v in buckets["EXERCISED"] if v.has_unexercised_branches),
        },
        "branch_coverage_present": any(v.num_branches for v in verdicts),
        "verdicts": [v.to_dict() for v in verdicts],
        "_verdicts": verdicts,
    }


def format_report(report: dict, show_all: bool = False) -> str:
    lines: list[str] = []
    t = report["totals"]
    lines.append("module coverage audit (line + branch)")
    lines.append("=" * 74)
    lines.append(
        f"  measured {t['n_measured']}   never-imported {t['never_imported']}   "
        f"imported-not-executed {t['imported_not_executed']}   "
        f"exercised {t['exercised']}")
    if not report["branch_coverage_present"]:
        lines.append("")
        lines.append("  !! NO BRANCH DATA. Re-run with --cov-branch, or every")
        lines.append("     '100% covered' below is only 100% of the STATEMENTS.")
    elif t["exercised_with_unexercised_branches"]:
        lines.append(
            f"  {t['exercised_with_unexercised_branches']} exercised module(s) still have "
            "untaken branches")

    for status, title in (
        ("NEVER_IMPORTED", "NEVER IMPORTED -- no importer anywhere in the tree"),
        ("IMPORTED_NOT_EXECUTED", "IMPORTED BUT NEVER EXECUTED -- a caller is missing"),
    ):
        vs = [v for v in report["_verdicts"] if v.status == status]
        if not vs:
            continue
        lines.append("")
        lines.append(f"{title}  ({len(vs)})")
        lines.append("-" * 74)
        for v in vs:
            br = f"  branches {v.covered_branches}/{v.num_branches}"
            lines.append(f"  {v.module:<46} {v.covered_lines:>4}/{v.statements:<4}"
                         f" lines ({v.line_pct:5.1f}%){br}")
            if v.importers:
                lines.append(f"      importers: {', '.join(v.importers[:4])}")
    if show_all:
        lines.append("")
        lines.append("EXERCISED (summary)")
        for v in report["_verdicts"]:
            if v.status != "EXERCISED":
                continue
            br = (f"  branches {v.covered_branches}/{v.num_branches}"
                  if v.num_branches else "  (no branch data)")
            lines.append(f"  {v.module:<46} {v.line_pct:5.1f}%{br}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("coverage_json", help="path to coverage.json")
    parser.add_argument("--root", default=".", help="repo root to scan for imports")
    parser.add_argument("--all", action="store_true", help="include exercised modules")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args(argv)

    report = audit(args.coverage_json, args.root)
    if args.json:
        print(json.dumps({k: v for k, v in report.items() if k != "_verdicts"}, indent=2))
    else:
        print(format_report(report, show_all=args.all))
    return 0


if __name__ == "__main__":
    sys.exit(main())
